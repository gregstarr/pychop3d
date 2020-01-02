import trimesh
import numpy as np
import shapely.geometry as sg

from pychop3d.configuration import Configuration
from pychop3d import utils


class ConnectedComponent:

    def __init__(self, polygon, xform, normal, origin):
        config = Configuration.config
        self.polygon = polygon
        self.normal = normal
        self.origin = origin
        self.xform = xform
        self.area = self.polygon.area
        self.positive = None
        self.negative = None
        self.connector_diameter = None
        self.objective = None
        self.positive_sites = None
        self.pos_index = None
        self.negative_sites = None
        self.neg_index = None
        self.all_sites = None
        self.all_index = None

        if config.adaptive_connector_size:
            self.connector_diameter = np.clip(np.sqrt(self.polygon.area) / 6, config.connector_diameter_min,
                                              config.connector_diameter_max)
        else:
            self.connector_diameter = config.connector_diameter

        verts, faces = trimesh.creation.triangulate_polygon(polygon, triangle_args='p', allow_boundary_steiner=False)
        verts = np.column_stack((verts, np.zeros(len(verts))))
        verts = trimesh.transform_points(verts, xform)
        faces = np.fliplr(faces)
        self.mesh = trimesh.Trimesh(verts, faces)

    def evaluate_interface(self, positive, negative):
        config = Configuration.config

        plane_samples = self.grid_sample_polygon()

        if len(plane_samples) == 0:
            # no 'Connector' locations
            print('C', end='')
            return False

        mesh_samples = trimesh.transform_points(np.column_stack((plane_samples, np.zeros(plane_samples.shape[0]))),
                                                self.xform)
        pos_dists = positive.nearest.signed_distance(mesh_samples + (1 + self.connector_diameter) * self.normal)
        neg_dists = negative.nearest.signed_distance(mesh_samples + (1 + self.connector_diameter) * -1 * self.normal)
        # the 1.5 is a slight overestimate of sqrt(2) to make the radius larger than
        # half the diagonal of a square connector
        pos_valid_mask = pos_dists > 1.5 * self.connector_diameter / 2
        neg_valid_mask = neg_dists > 1.5 * self.connector_diameter / 2
        ch_area_mask = np.logical_or(pos_valid_mask, neg_valid_mask)

        if ch_area_mask.sum() == 0:
            # no 'Connector' locations
            print('C', end='')
            return False

        convex_hull_area = sg.MultiPoint(plane_samples[ch_area_mask]).buffer(self.connector_diameter / 2).convex_hull.area
        self.objective = max(self.area / convex_hull_area - config.connector_objective_th, 0)
        self.positive_sites = mesh_samples[pos_valid_mask]
        self.negative_sites = mesh_samples[neg_valid_mask]
        self.all_sites = np.concatenate((self.positive_sites, self.negative_sites), axis=0)
        return True

    def grid_sample_polygon(self):
        min_x, min_y, max_x, max_y = self.polygon.bounds
        xp = np.arange(min_x + self.connector_diameter / 2, max_x - self.connector_diameter / 2, self.connector_diameter)
        if len(xp) == 0:
            return []
        xp += (min_x + max_x) / 2 - (xp.min() + xp.max()) / 2
        yp = np.arange(min_y + self.connector_diameter / 2, max_y - self.connector_diameter / 2, self.connector_diameter)
        if len(yp) == 0:
            return []
        yp += (min_y + max_y) / 2 - (yp.min() + yp.max()) / 2
        X, Y = np.meshgrid(xp, yp)
        xy = np.stack((X.ravel(), Y.ravel()), axis=1)
        mask = np.zeros(xy.shape[0], dtype=bool)
        for i in range(xy.shape[0]):
            point = sg.Point(xy[i])
            if point.within(self.polygon):
                mask[i] = True
        return xy[mask]

    def get_sites(self, state):
        return self.all_sites[np.isin(self.all_index, np.arange(state.shape[0])[state])]

    def register_sites(self, n_connectors):
        self.pos_index = np.arange(n_connectors, n_connectors + self.positive_sites.shape[0])
        self.neg_index = np.arange(n_connectors + self.positive_sites.shape[0], n_connectors + self.all_sites.shape[0])
        self.all_index = np.arange(n_connectors, n_connectors + self.all_sites.shape[0])

    def get_indices(self, state):
        pos_ind = np.isin(np.arange(state.shape[0]), self.pos_index) * state
        neg_ind = np.isin(np.arange(state.shape[0]), self.neg_index) * state
        return np.argwhere(pos_ind)[:, 0], np.argwhere(neg_ind)[:, 0]


class CrossSection:

    def __init__(self, mesh, origin, normal):
        self.valid = False
        self.origin = origin
        self.normal = normal
        self.connected_components = []

        path3d = mesh.section(plane_origin=origin, plane_normal=normal)
        if path3d is None:
            # 'Missed' the part basically
            print('M', end='')
            return

        # triangulate the cross section
        path2d, self.xform = path3d.to_planar()
        path2d.merge_vertices()
        for polygon in path2d.polygons_full:
            self.connected_components.append(ConnectedComponent(polygon, self.xform, self.normal, self.origin))
        self.valid = True

    def split(self, mesh):
        cap = np.array([cc.mesh for cc in self.connected_components]).sum()

        positive = mesh.slice_plane(plane_origin=self.origin, plane_normal=self.normal)
        positive = positive + cap
        positive._validate = True
        positive.process()
        utils.trimesh_repair(positive)

        negative = mesh.slice_plane(plane_origin=self.origin, plane_normal=-1 * self.normal)
        negative = negative + cap
        negative._validate = True
        negative.process()
        utils.trimesh_repair(negative)

        return positive, negative

    def get_average_connector_size(self):
        return sum([cc.connector_diameter for cc in self.connected_components]) / len(self.connected_components)


def bidirectional_split(mesh, origin, normal):
    """https://github.com/mikedh/trimesh/issues/235"""
    tries = 0
    positive_parts, negative_parts = [], []
    while (len(positive_parts) == 0 or len(negative_parts) == 0) and tries < 5:
        tries += 1
        origin += (np.random.rand() - .5) * normal * .1
        # determine ConnectedComponents of the cross section
        cross_section = CrossSection(mesh, origin, normal)
        if not cross_section.valid:
            continue
        # split parts
        positive, negative = cross_section.split(mesh)
        positive_parts = positive.split()
        negative_parts = negative.split()
        parts_list = list(np.concatenate((positive_parts, negative_parts)))

    if len(positive_parts) == 0 or len(negative_parts) == 0:
        # bad 'Separation'
        print('S', end='')
        return None, None

    for part in parts_list:
        utils.trimesh_repair(part)
    # assign 2 parts to each ConnectedComponent of the cross section
    for cc in cross_section.connected_components:
        cc_verts = np.round(cc.mesh.vertices, 3).view(dtype=[('', float), ('', float), ('', float)])
        # assign the cc a positive part
        for i, part in enumerate(positive_parts):
            part_verts = np.round(part.vertices, 3).view(dtype=[('', float), ('', float), ('', float)])
            if np.isin(cc_verts, part_verts).any():
                cc.positive = i
                break
        # assign the cc a negative part
        for i, part in enumerate(negative_parts):
            part_verts = np.round(part.vertices, 3).view(dtype=[('', float), ('', float), ('', float)])
            if np.isin(cc_verts, part_verts).any():
                cc.negative = i + len(positive_parts)
                break

        if None in [cc.positive, cc.negative]:
            # bad 'Separation'
            print('S', end='')
            return None, None

        if not cc.evaluate_interface(parts_list[cc.positive], parts_list[cc.negative]):
            return None, None

    return parts_list, cross_section