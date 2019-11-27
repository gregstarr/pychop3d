import trimesh
import numpy as np
import copy

from pychop3d import constants
from pychop3d import utils


class BSPNode:

    def __init__(self, part: trimesh.Trimesh, parent=None, num=None):
        self.part = part
        self.parent = parent
        self.children = {}
        self.path = []
        self.plane = None
        if self.parent is not None:
            self.path = self.parent.path + [num]

    def split(self, plane):
        part = self.part
        self.plane = plane
        positive = utils.unidirectional_split(part, plane[0], plane[1])
        positive.remove_degenerate_faces()
        negative = utils.unidirectional_split(part, plane[0], -1 * plane[1])
        negative.remove_degenerate_faces()
        self.children[0] = BSPNode(positive, parent=self, num=0)
        self.children[1] = BSPNode(negative, parent=self, num=1)

    def terminated(self):
        return np.all(self.part.bounding_box_oriented.primitive.extents < constants.PRINTER_EXTENTS)

    def number_of_parts_estimate(self):
        try:
            return np.prod(np.ceil(self.part.bounding_box_oriented.primitive.extents / constants.PRINTER_EXTENTS))
        except Exception as e:
            print(e)

    def auxiliary_normals(self):
        obb_xform = np.array(self.part.bounding_box_oriented.primitive.transform)
        return obb_xform[:3, :3]

    def get_planes(self, normal, plane_spacing=constants.PLANE_SPACING):
        projection = self.part.vertices @ normal
        limits = [projection.min(), projection.max()]
        planes = [(d * normal, normal) for d in np.arange(limits[0], limits[1], plane_spacing)][1:]
        return planes

    def different_from(self, other_node, threshold):
        plane_transform = trimesh.points.plane_transform(*self.plane)
        o = other_node.plane[0]
        op = np.array([o[0], o[1], o[2], 1], dtype=float)
        op = (plane_transform @ op)[:3]
        return np.sqrt(np.sum(op ** 2)) > threshold


class BSPTree:

    def __init__(self, part: trimesh.Trimesh):
        self.root = BSPNode(part)
        self.a_part = constants.A_PART
        self.a_util = constants.A_UTIL
        self.a_connector = constants.A_CONNECTOR
        self.a_fragility = constants.A_FRAGILITY
        self.a_seam = constants.A_SEAM
        self.a_symmetry = constants.A_SYMMETRY

    def get_node(self, path=None):
        node = self.root
        if path is None:
            return node
        else:
            for i in path:
                node = node.children[i]
        return node

    def expand_node(self, plane, node):
        new_tree = copy.deepcopy(self)
        node = new_tree.get_node(node.path)
        node.split(plane)
        return new_tree

    def get_leaves(self):
        nodes = [self.root]
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

    def sufficiently_different(self, node, tree_set, threshold=constants.SUFFICIENTLY_DIFFERENT_THRESHOLD):
        if not tree_set:
            return True
        for tree in tree_set:
            if not self.different_from(tree, node, threshold):
                return False
        return True

    def different_from(self, tree, node, threshold):
        self_node = self.get_node(node.path)
        other_node = tree.get_node(node.path)
        return self_node.different_from(other_node, threshold)

    def nparts_objective(self):
        theta_0 = self.root.number_of_parts_estimate()
        try:
            return sum([l.number_of_parts_estimate() for l in self.get_leaves()]) / theta_0
        except Exception as e:
            print(e)

    def utilization_objective(self):
        V = np.prod(constants.PRINTER_EXTENTS)
        return max([1 - leaf.part.volume / (leaf.number_of_parts_estimate() * V) for leaf in self.get_leaves()])

    def connector_objective(self):
        return 0

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
            if np.any(distance_to_plane[~hits] < 1.5 * constants.CONNECTOR_DIAMETER):
                return np.inf
            if not np.any(hits):
                continue
            locs, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, ray_directions)
            ray_mesh_dist = np.sqrt(np.sum((ray_origins[index_ray] - locs) ** 2, axis=1))
            if np.any((distance_to_plane[index_ray] < 1.5 * constants.CONNECTOR_DIAMETER) *
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
