import trimesh
import numpy as np
import traceback

from pychop3d import section
from pychop3d.configuration import Configuration


class ConvexHullError(Exception):
    pass


class BSPNode:

    def __init__(self, part, parent=None, num=None):
        """Initialize an instance of `BSPNode`, determine n_parts objective and termination status.

        :param part: mesh part associated with this node
        :param parent: BSPNode which was split to produce this node and this node's sibling nodes
        :param num: index of this node, e.g. 0 -> 0th child of this node's parent
        """
        config = Configuration.config  # collect configuration
        self.part = part
        self.parent = parent
        self.children = []
        self.path = tuple()
        self.plane = None  # this node will get a plane and a cross_section if and when it is split
        self.cross_section = None
        # determine n_parts and termination status
        self.n_parts = np.prod(np.ceil(self.obb.primitive.extents / config.printer_extents))
        self.terminated = np.all(self.obb.primitive.extents <= config.printer_extents)
        # if this isn't the root node
        if self.parent is not None:
            self.path = (*self.parent.path, num)

    @property
    def obb(self):
        """oriented bounding box

        :return: oriented bounding box
        :rtype: `trimesh.Trimesh`
        """
        try:
            return self.part.bounding_box_oriented
        except Exception:
            raise ConvexHullError("OBB failed")

    @property
    def auxiliary_normals(self):
        """(3 x 3) numpy array who's rows are the three unit vectors aligned with this node's part's oriented
        bounding box

        :return: oriented bounding box unit vectors
        :rtype: `numpy.ndarray`
        """
        obb_xform = np.array(self.obb.primitive.transform)
        return obb_xform[:3, :3]

    @property
    def connection_objective(self):
        """property containing the connection objective for this node

        :return: connection objective value for this node
        :rtype: float
        """
        return max([cc.objective for cc in self.cross_section.connected_components])

    def different_from(self, other_node):
        """determine if this node is different plane-wise from the same node on another tree (check if this node's
        plane is different from another node's plane)

        :param other_node: corresponding node on another tree
        :type other_node: `bsp_node.BSPNode`
        :return: boolean indicating if this node is different from another
        :rtype: bool
        """
        config = Configuration.config  # collect configuration
        o = other_node.plane[0]  # other plane origin
        delta = o - self.plane[0]  # vector from this plane's origin to the others'
        dist = abs(self.plane[1] @ delta)  # distance along this plane's normal to other plane
        # angle between this plane's normal vector and the other plane's normal vector
        angle = trimesh.transformations.angle_between_vectors(self.plane[1], other_node.plane[1])
        # also consider angle between the vectors in the opposite direction
        angle = min(np.pi - angle, angle)
        # check if either the distance or the angle are above their respective thresholds
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

    try:
        parts, cross_section, result = section.bidirectional_split(node.part, origin, normal)  # split the part
    except:
        traceback.print_exc()
        return None, 'unknown_mesh_split_error'
    if None in [parts, cross_section]:  # check for splitting errors
        return None, result
    node.cross_section = cross_section

    for i, part in enumerate(parts):
        if part.volume < .1:  # make sure each part has some volume
            return None, 'low_volume_error'
        try:
            child = BSPNode(part, parent=node, num=i)  # potential convex hull failure
        except ConvexHullError:
            return None, 'convex_hull_error'

        node.children.append(child)  # The parts become this node's children
    return node, 'success'
