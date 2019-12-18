import trimesh
import numpy as np
import copy
import itertools
import json
import os

from pychop3d import constants
from pychop3d import utils
from pychop3d import bsp_mesh


class BSPNode:

    def __init__(self, part, parent=None, num=None):
        self.part = part
        self.parent = parent
        self.children = {}
        self.path = tuple()
        self.plane = None
        self.cross_section = None
        # if this isn't the root node
        if self.parent is not None:
            self.path = (*self.parent.path, num)

    def split(self, plane):
        self.plane = plane
        origin, normal = plane
        positive, negative, cross_section = self.part.bidirectional_split(origin, normal)

        if None in [positive, negative, cross_section]:
            return False

        if not cross_section.find_connector_sites(positive, negative):
            return False

        self.cross_section = cross_section
        self.children[0] = BSPNode(positive, parent=self, num=0)
        self.children[1] = BSPNode(negative, parent=self, num=1)
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

    def terminated(self):
        return np.all(self.get_bounding_box_oriented().primitive.extents < constants.PRINTER_EXTENTS)

    def number_of_parts_estimate(self):
        return np.prod(np.ceil(self.get_bounding_box_oriented().primitive.extents / constants.PRINTER_EXTENTS))

    def auxiliary_normals(self):
        obb_xform = np.array(self.get_bounding_box_oriented().primitive.transform)
        return obb_xform[:3, :3]

    def get_planes(self, normal, plane_spacing=constants.PLANE_SPACING):
        projection = self.part.vertices @ normal
        limits = [projection.min(), projection.max()]
        planes = [(d * normal, normal) for d in np.arange(limits[0], limits[1], plane_spacing)][1:]
        return planes

    def different_from(self, other_node):
        plane_transform = trimesh.points.plane_transform(*self.plane)
        o = other_node.plane[0]
        op = np.array([o[0], o[1], o[2], 1], dtype=float)
        op = (plane_transform @ op)[:3]
        angle = trimesh.transformations.angle_between_vectors(self.plane[1], other_node.plane[1])
        angle = min(np.pi - angle, angle)

        return (np.sqrt(np.sum(op ** 2)) > constants.DIFFERENT_ORIGIN_THRESHOLD or
                angle > constants.DIFFERENT_ANGLE_THRESHOLD)

    def get_connection_objective(self):
        return max([cc.objective for cc in self.cross_section.connected_components])


class BSPTree:

    def __init__(self, part: trimesh.Trimesh):
        self.nodes = [BSPNode(part)]
        self._node_data = {}
        self._objective = None

        self.a_part = constants.A_PART
        self.a_util = constants.A_UTIL
        self.a_connector = constants.A_CONNECTOR
        self.a_fragility = constants.A_FRAGILITY
        self.a_seam = constants.A_SEAM
        self.a_symmetry = constants.A_SYMMETRY

    @property
    def objective(self):
        if self._objective is None:
            self._objective = self.get_objective()
        return self._objective

    @classmethod
    def from_json(cls, config_fn):
        with open(config_fn) as f:
            config = json.load(f)

        mesh, config = utils.open_mesh(config)

        node_data = config['nodes']
        tree = cls(mesh)
        for n in node_data:
            plane = (np.array(n['origin']), np.array(n['normal']))
            node = tree.get_node(n['path'])
            tree = tree.expand_node(plane, node)

        return tree, config

    @classmethod
    def from_node_data(cls, part, node_data):
        tree = cls(part)
        for path, plane in node_data.items():
            node = tree.get_node(path)
            tree = tree.expand_node(plane, node)
        return tree

    def copy(self):
        # return BSPTree.from_node_data(self.nodes[0].part.copy(), self._node_data)
        return copy.deepcopy(self)

    def expand_node(self, plane, node):
        new_tree = self.copy()
        for n in new_tree.nodes:
            old_node = self.get_node(n.path)
            chull = old_node.part.convex_hull.copy()
            part = old_node.part.copy()
            n.part = bsp_mesh.BSPMesh.from_trimesh(part, chull)

        new_node = new_tree.get_node(node.path)
        if not new_node.split(plane):
            return None
        new_tree._node_data[node.path] = plane
        new_tree.nodes += list(new_node.children.values())
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
            if node.children == {}:
                leaves.append(node)
            else:
                nodes += list(node.children.values())
        return leaves

    def terminated(self):
        leaves = self.get_leaves()
        for leaf in leaves:
            if not leaf.terminated():
                return False
        return True

    def largest_part(self):
        return sorted(self.get_leaves(), key=lambda x: x.number_of_parts_estimate())[-1]

    def sufficiently_different(self, node, tree_set):
        if not tree_set:
            return True
        for tree in tree_set:
            if not self.different_from(tree, node):
                return False
        return True

    def different_from(self, tree, node):
        self_node = self.get_node(node.path)
        other_node = tree.get_node(node.path)
        return self_node.different_from(other_node)

    def nparts_objective(self):
        theta_0 = self.nodes[0].number_of_parts_estimate()
        return sum([l.number_of_parts_estimate() for l in self.get_leaves()]) / theta_0

    def utilization_objective(self):
        V = np.prod(constants.PRINTER_EXTENTS)
        return max([1 - leaf.part.volume / (leaf.number_of_parts_estimate() * V) for leaf in self.get_leaves()])

    def connector_objective(self):
        return max([n.get_connection_objective() for n in self.nodes if n.cross_section is not None])

    def fragility_objective(self):
        leaves = self.get_leaves()
        nodes = {}
        for leaf in leaves:
            path = tuple(leaf.parent.path)
            if path not in nodes:
                nodes[path] = leaf.parent

        for node in nodes.values():
            origin, normal = node.plane
            mesh = node.part
            possibly_fragile = np.abs(mesh.vertex_normals @ normal) > constants.FRAGILITY_THRESHOLD
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

    def get_objective(self):
        part = self.a_part * self.nparts_objective()
        util = self.a_util * self.utilization_objective()
        connector = self.a_connector * self.connector_objective()
        fragility = self.a_fragility * self.fragility_objective()
        seam = self.a_seam * self.seam_objective()
        symmetry = self.a_symmetry * self.symmetry_objective()
        return part + util + connector + fragility + seam + symmetry

    def preview(self):
        scene = trimesh.scene.Scene()
        for leaf in self.get_leaves():
            leaf.part.visual.face_colors = np.random.rand(3)*255
            scene.add_geometry(leaf.part)
        scene.show()

    def save(self, filename, config, state=None):
        nodes = []
        for node in self.nodes:
            if node.plane is not None:
                this_node = {'path': node.path, 'origin': list(node.plane[0]), 'normal': list(node.plane[1])}
                nodes.append(this_node)

        config['nodes'] = nodes

        if state is not None:
            config['state'] = [bool(s) for s in state]

        with open(os.path.join(config['directory'], filename), 'w') as f:
            json.dump(config, f)

    def export_stl(self, config):
        for i, leaf in enumerate(self.get_leaves()):
            leaf.part.export(os.path.join(config['directory'], f"{i}.stl"))

