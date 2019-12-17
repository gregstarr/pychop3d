import os
import trimesh
import copy
import numpy as np
import json

from pychop3d import constants
from pychop3d import node
from pychop3d import utils


class BSPTree:

    def __init__(self, part: trimesh.Trimesh):
        self.nodes = [node.Node(part)]
        self.a_part = constants.A_PART
        self.a_util = constants.A_UTIL
        self.a_connector = constants.A_CONNECTOR
        self.a_fragility = constants.A_FRAGILITY
        self.a_seam = constants.A_SEAM
        self.a_symmetry = constants.A_SYMMETRY
        self._objective = None

    @property
    def objective(self):
        if self._objective is None:
            part = self.a_part * self.nparts_objective()
            util = self.a_util * self.utilization_objective()
            connector = self.a_connector * self.connector_objective()
            fragility = self.a_fragility * self.fragility_objective()
            seam = self.a_seam * self.seam_objective()
            symmetry = self.a_symmetry * self.symmetry_objective()
            self._objective = part + util + connector + fragility + seam + symmetry
        return self._objective

    def get_node(self, path=None):
        node = self.nodes[0]
        if path is None or len(path) == 0:
            return node
        else:
            for i in path:
                node = node.children[i]
        return node

    def expand_node(self, plane, node):
        new_tree = copy.deepcopy(self)
        new_tree._objective = None
        node = new_tree.get_node(node.path)
        success = node.split(plane)
        new_tree.nodes += list(node.children.values())
        if success:
            return new_tree
        return None

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
            path = leaf.parent.path
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
            if np.any(distance_to_plane[~hits] < 1.5 * node.cross_section.get_average_connector_diameter()):
                return np.inf
            if not np.any(hits):
                continue
            locs, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, ray_directions)
            ray_mesh_dist = np.sqrt(np.sum((ray_origins[index_ray] - locs) ** 2, axis=1))
            if np.any((distance_to_plane[index_ray] < 1.5 * node.cross_section.get_average_connector_diameter()) *
                      (distance_to_plane[index_ray] < ray_mesh_dist)):
                return np.inf
        return 0

    def seam_objective(self):
        return 0

    def symmetry_objective(self):
        return 0

    def preview(self):
        scene = trimesh.scene.Scene()
        for leaf in self.get_leaves():
            leaf.part.visual.face_colors = np.random.rand(3)*255
            scene.add_geometry(leaf.part)
        scene.show()

    def save(self, filename, config):
        nodes = []
        for node in self.nodes:
            if node.plane is not None:
                this_node = {'path': node.path, 'origin': list(node.plane[0]), 'normal': list(node.plane[1])}
                nodes.append(this_node)
        new_config = config.copy()
        new_config['nodes'] = nodes
        with open(os.path.join(config['directory'], filename), 'w') as f:
            json.dump(new_config, f)

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            config = json.load(f)

        node_data = config['nodes']
        mesh, *_ = utils.open_and_convert_mesh(config)
        tree = cls(mesh)
        for n in node_data:
            plane = (np.array(n['origin']), np.array(n['normal']))
            node = tree.get_node(n['path'])
            tree = tree.expand_node(plane, node)
        return tree, config

    def export_stl(self, config):
        for i, leaf in enumerate(self.get_leaves()):
            leaf.part.export(os.path.join(config['directory'], f"{i}.stl"))
