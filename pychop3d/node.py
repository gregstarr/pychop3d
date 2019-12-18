import trimesh
import numpy as np

from pychop3d import constants


class Node:

    def __init__(self, part, parent=None, num=None):
        self.part = part
        self.parent = parent
        self.children = {}
        self.path = tuple()
        self.plane = None
        self.cross_section = None
        # if this isn't the root node
        if self.parent is not None:
            self.path = tuple([*self.parent.path, num])

    def split(self, plane):
        self.plane = plane
        origin, normal = plane
        positive, negative, cross_section = self.part.bidirectional_split(origin, normal)
        # check for splitting errors
        if None in [positive, negative]:
            return False
        self.cross_section = cross_section
        if not self.cross_section.find_connector_sites(positive, negative):
            return False
        self.children[0] = Node(positive, parent=self, num=0)
        self.children[1] = Node(negative, parent=self, num=1)
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
        plane_to_origin = np.array([o[0], o[1], o[2], 1], dtype=float)
        plane_to_origin = (plane_transform @ plane_to_origin)[:3]
        origin_diff = np.sum(plane_to_origin ** 2)
        normal_diff = trimesh.transformations.angle_between_vectors(self.plane[1], other_node.plane[1])
        return (np.sqrt(origin_diff) > constants.SUFFICIENTLY_DIFFERENT_ORIGIN_THRESHOLD or
                normal_diff > constants.SUFFICIENTLY_DIFFERENT_NORMAL_THRESHOLD)

    def get_connection_objective(self):
        return self.cross_section.get_connector_objective()