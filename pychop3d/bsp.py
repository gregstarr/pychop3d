import trimesh
import numpy as np
from pychop3d import constants
import copy


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
        origin = plane[0]
        norm = plane[1]
        part = self.part
        self.plane = plane
        d = np.sqrt(np.sum((origin - part.vertices) ** 2, axis=1)).max()
        xform = trimesh.transformations.translation_matrix(np.array([0., 0., -1 * d]))
        cutter = trimesh.primitives.Box(extents=np.ones(3) * d * 2, transform=xform)
        norm_aligner = trimesh.geometry.align_vectors(np.array([0, 0, 1]), norm)
        cutter.apply_transform(norm_aligner)
        translation_to_origin = trimesh.transformations.translation_matrix(origin)
        cutter.apply_transform(translation_to_origin)
        self.children[0] = BSPNode(part.difference(cutter, engine='scad'), parent=self, num=0)
        self.children[1] = BSPNode(part.intersection(cutter, engine='scad'), parent=self, num=1)

    def terminated(self):
        return np.all(self.part.bounding_box_oriented.extents < constants.PRINTER_EXTENTS)

    def number_of_parts_estimate(self):
        return np.prod(np.ceil(self.part.bounding_box_oriented.extents / constants.PRINTER_EXTENTS))

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
        self.a_part = 1
        self.a_util = .05
        self.a_connector = 1
        self.a_fragility = 1
        self.a_seam = .1
        self.a_symmetry = .25

    def get_node(self, path=None):
        node = self.root
        if path is None:
            return node
        else:
            for i in path:
                node = node.children[i]
        return node

    def expand_node(self, plane, node=None):
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
        return sum([l.number_of_parts_estimate() for l in self.get_leaves()]) / theta_0

    def utilization_objective(self):
        V = np.prod(constants.PRINTER_EXTENTS)
        return max([1 - leaf.part.volume / (leaf.number_of_parts_estimate() * V) for leaf in self.get_leaves()])

    def connector_objective(self):
        return 0

    def fragility_objective(self):
        return 0

    def seam_objective(self):
        return 0

    def symmetry_objective(self):
        return 0

    def get_objective(self):
        return (self.a_part * self.nparts_objective() +
                self.a_util * self.utilization_objective() +
                self.a_connector * self.connector_objective() +
                self.a_fragility * self.fragility_objective() +
                self.a_seam * self.seam_objective() +
                self.a_symmetry * self.symmetry_objective())

    def preview(self):
        scene = trimesh.scene.Scene()
        for leaf in self.get_leaves():
            leaf.part.visual.face_colors = np.random.rand(3)*255
            scene.add_geometry(leaf.part)
        scene.show()
