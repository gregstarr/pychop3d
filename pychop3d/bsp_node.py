import trimesh
import numpy as np

from pychop3d import section
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
        self.n_parts = np.prod(np.ceil(self.obb.primitive.extents / config.printer_extents))
        self.terminated = np.all(self.obb.primitive.extents <= config.printer_extents)
        # if this isn't the root node
        if self.parent is not None:
            self.path = (*self.parent.path, num)

    @property
    def obb(self):
        try:
            return self.part.bounding_box_oriented
        except Exception as e:
            print(e)
            print("REGULAR CONVEX HULL FAILED, USING APPROXIMATION")
            samples, *_ = self.part.nearest.on_surface(trimesh.primitives.Sphere(radius=1000).vertices)
            point_cloud = trimesh.PointCloud(samples)
            return point_cloud.bounding_box_oriented

    @property
    def auxiliary_normals(self):
        obb_xform = np.array(self.obb.primitive.transform)
        return obb_xform[:3, :3]

    @property
    def connection_objective(self):
        return max([cc.objective for cc in self.cross_section.connected_components])

    def different_from(self, other_node):
        config = Configuration.config
        o = other_node.plane[0]
        delta = o - self.plane[0]
        dist = abs(self.plane[1] @ delta)
        angle = trimesh.transformations.angle_between_vectors(self.plane[1], other_node.plane[1])
        angle = min(np.pi - angle, angle)
        return dist > config.different_origin_th or angle > config.different_angle_th


def split(node, plane):
    """Split a node along a plane

    :param node: BSPNode to split
    :type node: BSPNode
    :param plane: (origin, normal) pair defining the cutting plane
    :type plane: tuple
    :return: the input node split by the plane, will have at least 2 children
    :rtype: BSPNode
    """
    node.plane = plane
    origin, normal = plane
    # split the part
    parts, cross_section = section.bidirectional_split(node.part, origin, normal)

    # check for splitting errors
    if None in [parts, cross_section]:
        return None

    # The parts become this node's children
    node.cross_section = cross_section
    for i, part in enumerate(parts):
        # make sure each part has some volume
        if part.volume < .1:
            print('V', end='')
            return None
        node.children.append(BSPNode(part, parent=node, num=i))
    print('.', end='')
    return node
