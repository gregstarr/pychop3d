"""Cross section and connected component"""
from __future__ import annotations
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import MultiPoint, Point, Polygon
from trimesh import Trimesh, creation, transform_points

from pychop3d import settings, utils, bsp_node
from pychop3d.logger import logger


class ConnectedComponent:
    """a connected component of a mesh-plane intersection"""

    def __init__(self, polygon: Polygon, xform: np.ndarray, plane: bsp_node.Plane):
        self.valid = False
        self.polygon = polygon
        self.plane = plane
        self.xform = xform
        self.area = self.polygon.area
        self.positive = None
        self.negative = None
        self.objective = None
        self.sites = None
        self.index = None

        if self.area < (settings.CONNECTOR_DIAMETER / 2) ** 2:
            return

        verts, faces = creation.triangulate_polygon(polygon, triangle_args="p")
        verts = np.column_stack((verts, np.zeros(len(verts))))
        verts = transform_points(verts, xform)
        faces = np.fliplr(faces)
        self.mesh = Trimesh(verts, faces)
        self.valid = True

    def evaluate_interface(self, positive: Trimesh, negative: Trimesh) -> bool:
        """compute objective and valid connector sites for the interface

        Args:
            positive (Trimesh): mesh on positive side of plane
            negative (Trimesh): mesh on negative side of plane

        Returns:
            bool: success
        """
        plane_samples = self.grid_sample_polygon()

        if len(plane_samples) == 0:
            return False

        # TODO: these checks should depend on the connector type
        normal = self.plane.normal
        CD = settings.CONNECTOR_DIAMETER
        mesh_samples = transform_points(
            np.column_stack((plane_samples, np.zeros(plane_samples.shape[0]))),
            self.xform,
        )
        pos_dists = positive.nearest.signed_distance(
            mesh_samples + (CD / 2) * normal
        )
        neg_dists = negative.nearest.signed_distance(
            mesh_samples + (CD / 2) * -1 * normal
        )
        pos_valid_mask = pos_dists >= 1
        neg_valid_mask = neg_dists >= 1
        valid_mask = np.logical_and(pos_valid_mask, neg_valid_mask)

        if valid_mask.sum() == 0:
            return False

        convex_hull_area = (
            MultiPoint(plane_samples[valid_mask]).buffer(CD / 2).convex_hull.area
        )
        self.objective = max(
            self.area / convex_hull_area - settings.CONNECTOR_OBJECTIVE_TH, 0
        )
        self.sites = mesh_samples[valid_mask]
        return True

    def grid_sample_polygon(self) -> np.ndarray:
        """Returns a grid of connector cantidate locations. The connected component
        (chop cross section) is rotated to align with the minimum rotated bounding box,
        then the resulting polygon is grid sampled with settings.CONNECTOR_SPACING

        Returns:
            np.ndarray: samples
        """
        eroded_polygon = self.polygon.buffer(-1 * settings.CONNECTOR_DIAMETER / 2)
        mrr_points = np.column_stack(eroded_polygon.minimum_rotated_rectangle.boundary.xy)
        mrr_edges = np.diff(mrr_points, axis=0)
        angle = -1 * np.arctan2(mrr_edges[0, 1], mrr_edges[0, 0])
        rotated_polygon = rotate(eroded_polygon, angle, use_radians=True, origin=(0, 0))
        min_x, min_y, max_x, max_y = rotated_polygon.bounds
        xp = np.arange(min_x, max_x, settings.CONNECTOR_SPACING)
        if len(xp) == 0:
            return np.array([])
        xp += (min_x + max_x) / 2 - (xp.min() + xp.max()) / 2
        yp = np.arange(min_y, max_y, settings.CONNECTOR_SPACING)
        if len(yp) == 0:
            return np.array([])
        yp += (min_y + max_y) / 2 - (yp.min() + yp.max()) / 2
        X, Y = np.meshgrid(xp, yp)
        xy = np.stack((X.ravel(), Y.ravel()), axis=1)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        xy = xy @ rotation
        mask = np.zeros(xy.shape[0], dtype=bool)
        for i in range(xy.shape[0]):
            point = Point(xy[i])
            if point.within(eroded_polygon):
                mask[i] = True
        return xy[mask]

    def get_sites(self, state: np.ndarray) -> np.ndarray:
        """return sites in this connected component enabled in state vector"""
        return self.sites[np.isin(self.index, np.arange(state.shape[0])[state])]

    def register_sites(self, n_connectors: int):
        """register site indices against global list"""
        self.index = np.arange(n_connectors, n_connectors + self.sites.shape[0])

    def get_indices(self, state: np.ndarray) -> np.ndarray:
        """return indices in global site list corresponding to this connected component
        enabled in state vector"""
        ind = np.isin(np.arange(state.shape[0]), self.index) * state
        return np.argwhere(ind)[:, 0]


class CrossSection:
    """cross section created by plane intersection with mesh, should contain at least
    one connected component"""

    def __init__(self, mesh: Trimesh, plane: bsp_node.Plane):
        self.valid = False
        self.cc_valid = True
        self.plane = plane
        self.connected_components = []
        path3d = mesh.section(plane_origin=plane.origin, plane_normal=plane.normal)
        if path3d is None:
            # 'Missed' the part basically
            return

        # triangulate the cross section
        path2d, self.xform = path3d.to_planar()
        path2d.merge_vertices()
        try:
            path2d.polygons_full
        except Exception:
            # Missed the part basically
            return
        for polygon in path2d.polygons_full:
            cc = ConnectedComponent(polygon, self.xform, self.plane)
            if not cc.valid:
                self.cc_valid = False
            self.connected_components.append(cc)
        self.valid = True

    def split(self, mesh: Trimesh) -> tuple[Trimesh, Trimesh]:
        """splits mesh

        Args:
            mesh (Trimesh)

        Returns:
            tuple[Trimesh, Trimesh]: two meshes resulting from split
        """
        cap = np.array([cc.mesh for cc in self.connected_components]).sum()
        utils.trimesh_repair(cap)

        positive = mesh.slice_plane(
            plane_origin=self.plane.origin, plane_normal=self.plane.normal
        )
        positive = positive + cap
        positive._validate = True
        positive.process()
        utils.trimesh_repair(positive)

        negative = mesh.slice_plane(
            plane_origin=self.plane.origin, plane_normal=-1 * self.plane.normal
        )
        negative = negative + cap
        negative._validate = True
        negative.process()
        utils.trimesh_repair(negative)

        return positive, negative


def bidirectional_split(mesh: Trimesh, plane: bsp_node.Plane):
    """https://github.com/mikedh/trimesh/issues/235"""
    positive_parts, negative_parts = [], []

    cross_section = CrossSection(mesh, plane)
    if not cross_section.valid:
        logger.warning("invalid_cross_section_error")
        return None, None, "invalid_cross_section_error"
    if not cross_section.cc_valid:
        logger.warning("invalid_connected_component_error")
        return None, None, "invalid_connected_component_error"
    positive, negative = cross_section.split(mesh)
    if settings.PART_SEPARATION:
        # split parts
        positive_parts = positive.split(only_watertight=False)
        negative_parts = negative.split(only_watertight=False)
    else:
        positive_parts = [positive]
        negative_parts = [negative]
    parts_list = list(np.concatenate((positive_parts, negative_parts)))

    if len(positive_parts) == 0 or len(negative_parts) == 0:
        logger.warning("bad_separation_error")
        return None, None, "bad_separation_error"

    for part in parts_list:
        utils.trimesh_repair(part)
    # assign 2 parts to each ConnectedComponent of the cross section
    for cc in cross_section.connected_components:
        cc_verts = np.round(cc.mesh.vertices, 3).view(
            dtype=[("", float), ("", float), ("", float)]
        )
        # assign the cc a positive part
        for i, part in enumerate(positive_parts):
            part_verts = np.round(part.vertices, 3).view(
                dtype=[("", float), ("", float), ("", float)]
            )
            if np.isin(cc_verts, part_verts).any():
                cc.positive = i
                break
        # assign the cc a negative part
        for i, part in enumerate(negative_parts):
            part_verts = np.round(part.vertices, 3).view(
                dtype=[("", float), ("", float), ("", float)]
            )
            if np.isin(cc_verts, part_verts).any():
                cc.negative = i + len(positive_parts)
                break

        if None in [cc.positive, cc.negative]:
            logger.warning("bad_separation_error")
            return None, None, "bad_separation_error"

        if not cc.evaluate_interface(parts_list[cc.positive], parts_list[cc.negative]):
            logger.warning("connector_location_error")
            return None, None, "connector_location_error"

    return parts_list, cross_section, "success"
