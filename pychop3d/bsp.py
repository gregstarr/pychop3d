import trimesh
import trimesh
import numpy as np
import copy
import itertools
import json
import os

from pychop3d import constants
from pychop3d import section
from pychop3d import utils
from pychop3d.configuration import Configuration


class BSPNode:

    def __init__(self, part, parent=None, num=None):
        config = Configuration.config
        self.part = part
        self.parent = parent
        self.children = []
        self.path = tuple()
        self.plane = None
        self.cross_section = None
        self.n_parts = np.prod(np.ceil(self.get_bounding_box_oriented().primitive.extents / config.printer_extents))
        self.terminated = np.all(self.get_bounding_box_oriented().primitive.extents <= config.printer_extents)
        # if this isn't the root node
        if self.parent is not None:
            self.path = (*self.parent.path, num)

    def split(self, plane):
        self.plane = plane
        origin, normal = plane
        parts, cross_section = section.bidirectional_split(self.part, origin, normal)

        if None in [parts, cross_section]:
            return False

        self.cross_section = cross_section
        try:
            for i, part in enumerate(parts):
                self.children.append(BSPNode(part, parent=self, num=i))
        except Exception as e:
            print(e)
            return False
        print('.', end='')
        return True

    def get_bounding_box_oriented(self):
        try:
            return self.part.bounding_box_oriented
        except Exception as e:
            print(e)
            print("REGULAR CONVEX HULL FAILED, USING APPROXIMATION")
            samples, *_ = self.part.nearest.on_surface(constants.CH_SAMPLE_POINTS)
            point_cloud = trimesh.PointCloud(samples)
            return point_cloud.bounding_box_oriented

    def auxiliary_normals(self):
        obb_xform = np.array(self.get_bounding_box_oriented().primitive.transform)
        return obb_xform[:3, :3]

    def get_planes(self, normal):
        config = Configuration.config
        projection = self.part.vertices @ normal
        limits = [projection.min(), projection.max()]
        planes = [(d * normal, normal) for d in np.arange(limits[0], limits[1], config.plane_spacing)][1:]
        if config.add_middle_plane:
            planes += [(normal * (projection.min() + projection.max()) / 2, normal)]  # add a plane through the middle
        return planes

    def different_from(self, other_node):
        config = Configuration.config
        o = other_node.plane[0]
        delta = o - self.plane[0]
        dist = abs(self.plane[1] @ delta)
        angle = trimesh.transformations.angle_between_vectors(self.plane[1], other_node.plane[1])
        angle = min(np.pi - angle, angle)

        return dist > config.different_origin_th or angle > config.different_angle_th

    def get_connection_objective(self):
        return max([cc.objective for cc in self.cross_section.connected_components])


class BSPTree:

    def __init__(self, part: trimesh.Trimesh):
        config = Configuration.config
        self.nodes = [BSPNode(part)]
        self._node_data = {}
        self.objectives = {
            'nparts': self.nparts_objective(),
            'utilization': self.utilization_objective(),
            'connector': 0,  # no connectors yet
            'fragility': 0,
            'seam': 0,
            'symmetry': 0
        }

    @classmethod
    def from_node_data(cls, part, node_data):
        tree = cls(part)
        for path, plane in node_data.items():
            node = tree.get_node(path)
            tree = tree.expand_node(plane, node)
        return tree

    def copy(self):
        new_tree = copy.deepcopy(self)
        return new_tree

    def expand_node(self, plane, node):
        new_tree = self.copy()
        new_node = new_tree.get_node(node.path)
        if not new_node.split(plane):
            return None
        new_tree._node_data[node.path] = plane
        new_tree.nodes += new_node.children
        return new_tree

    def get_node(self, path=None):
        node = self.nodes[0]
        if not path:
            return node
        else:
            for i in path:
                node = node.children[i]
        return node

    def get_leaves(self):
        nodes = [self.nodes[0]]
        leaves = []
        while nodes:
            node = nodes.pop()
            if len(node.children) == 0:
                leaves.append(node)
            else:
                nodes += node.children
        return leaves

    def terminated(self):
        leaves = self.get_leaves()
        for leaf in leaves:
            if not leaf.terminated:
                return False
        return True

    def largest_part(self):
        return sorted(self.get_leaves(), key=lambda x: x.n_parts)[-1]

    def sufficiently_different(self, node, tree_set):
        if not tree_set:
            return True
        for tree in tree_set:
            if not self.different_from(tree, node):
                self.different_from(tree, node)
                return False
        return True

    def different_from(self, tree, node):
        self_node = self.get_node(node.path)
        other_node = tree.get_node(node.path)
        return self_node.different_from(other_node)

    def nparts_objective(self):
        theta_0 = self.nodes[0].n_parts
        return sum([l.n_parts for l in self.get_leaves()]) / theta_0

    def utilization_objective(self):
        config = Configuration.config
        V = np.prod(config.printer_extents)
        if config.obb_utilization:
            return max([1 - leaf.get_bounding_box_oriented().volume / (leaf.n_parts * V) for leaf in self.get_leaves()])
        else:
            return max([1 - leaf.part.volume / (leaf.n_parts * V) for leaf in self.get_leaves()])

    def connector_objective(self):
        return max([n.get_connection_objective() for n in self.nodes if n.cross_section is not None])

    def fragility_objective(self):
        config = Configuration.config
        leaves = self.get_leaves()
        nodes = {}
        for leaf in leaves:
            path = tuple(leaf.parent.path)
            if path not in nodes:
                nodes[path] = leaf.parent

        for node in nodes.values():
            origin, normal = node.plane
            mesh = node.part
            possibly_fragile = np.abs(mesh.vertex_normals @ normal) > config.fragility_objective_th
            if not np.any(possibly_fragile):
                continue
            ray_origins = mesh.vertices[possibly_fragile] - .1 * mesh.vertex_normals[possibly_fragile]
            distance_to_plane = np.abs((ray_origins - origin) @ normal)
            side_mask = (ray_origins - origin) @ normal > 0
            ray_directions = np.ones((ray_origins.shape[0], 1)) * normal[None, :]
            ray_directions[side_mask] *= -1
            hits = mesh.ray.intersects_any(ray_origins, ray_directions)
            if np.any(distance_to_plane[~hits] < 1.5 * node.cross_section.get_average_connector_size()):
                return np.inf
            if not np.any(hits):
                continue
            locs, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, ray_directions)
            ray_mesh_dist = np.sqrt(np.sum((ray_origins[index_ray] - locs) ** 2, axis=1))
            if np.any((distance_to_plane[index_ray] < 1.5 * node.cross_section.get_average_connector_size()) *
                      (distance_to_plane[index_ray] < ray_mesh_dist)):
                return np.inf
        return 0

    def seam_objective(self):
        return 0

    def symmetry_objective(self):
        return 0

    def get_objective_old(self):
        part = self.a_part * self.nparts_objective()
        util = self.a_util * self.utilization_objective()
        connector = self.a_connector * self.connector_objective()
        fragility = self.a_fragility * self.fragility_objective()
        seam = self.a_seam * self.seam_objective()
        symmetry = self.a_symmetry * self.symmetry_objective()
        return part + util + connector + fragility + seam + symmetry

    def get_objective(self):
        config = Configuration.config
        part = config.objective_weights['part'] * self.objectives['nparts']
        util = config.objective_weights['utilization'] * self.objectives['utilization']
        connector = config.objective_weights['connector'] * self.objectives['connector']
        fragility = config.objective_weights['fragility'] * self.objectives['fragility']
        seam = config.objective_weights['seam'] * self.objectives['seam']
        symmetry = config.objective_weights['symmetry'] * self.objectives['symmetry']
        return part + util + connector + fragility + seam + symmetry

    def preview(self):
        scene = trimesh.scene.Scene()
        for leaf in self.get_leaves():
            leaf.part.visual.face_colors = np.random.rand(3)*255
            scene.add_geometry(leaf.part)
        scene.camera.z_far = 10_000
        scene.show()

    def save(self, filename, state=None):
        config = Configuration.config

        if state is None:
            state = []

        nodes = []
        for node in self.nodes:
            if node.plane is not None:
                this_node = {'path': node.path, 'origin': list(node.plane[0]), 'normal': list(node.plane[1])}
                nodes.append(this_node)

        with open(os.path.join(config.directory, filename), 'w') as f:
            json.dump({'nodes': nodes, 'state': [bool(s) for s in state]}, f)

    def export_stl(self):
        config = Configuration.config
        for i, leaf in enumerate(self.get_leaves()):
            leaf.part.export(os.path.join(config.directory, f"{i}.stl"))

