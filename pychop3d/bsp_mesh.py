import trimesh
import numpy as np
import shapely.geometry as sg

from pychop3d.configuration import Configuration


class ConnectedComponent:

    def __init__(self, cross_section, polygon, positive, negative):
        config = Configuration.config
        self.valid = False
        self.connector_diameter = np.clip(np.sqrt(polygon.area) / 6, config.connector_diameter_min,
                                          config.connector_diameter_max)
        self.connector_diameter = config.connector_diameter
        self.area = polygon.area
        self.normal = cross_section.normal
        self.origin = cross_section.origin
        plane_samples = self.grid_sample_polygon(polygon)

        if plane_samples.size == 0:
            # no 'Connector' locations
            print('C', end='')
            return

        mesh_samples = trimesh.transform_points(np.column_stack((plane_samples, np.zeros(plane_samples.shape[0]))),
                                                cross_section.xform)
        pos_dists = positive.nearest.signed_distance(mesh_samples + (1 + self.connector_diameter) * cross_section.normal)
        neg_dists = negative.nearest.signed_distance(mesh_samples + (1 + self.connector_diameter) * -1 * cross_section.normal)
        pos_valid_mask = pos_dists > self.connector_diameter
        neg_valid_mask = neg_dists > self.connector_diameter
        ch_area_mask = np.logical_or(pos_valid_mask, neg_valid_mask)

        if ch_area_mask.sum() == 0:
            # no 'Connector' locations
            print('C', end='')
            return

        convex_hull_area = sg.MultiPoint(plane_samples[ch_area_mask]).buffer(self.connector_diameter / 2).convex_hull.area
        self.objective = max(self.area / convex_hull_area - config.connector_objective_th, 0)
        self.positive_sites = mesh_samples[pos_valid_mask]
        self.pos_index = None
        self.negative_sites = mesh_samples[neg_valid_mask]
        self.neg_index = None
        self.all_sites = np.concatenate((self.positive_sites, self.negative_sites), axis=0)
        self.all_index = None
        self.valid = True

    def grid_sample_polygon(self, polygon):
        min_x, min_y, max_x, max_y = polygon.bounds
        X, Y = np.meshgrid(np.arange(min_x, max_x, self.connector_diameter)[1:],
                           np.arange(min_y, max_y, self.connector_diameter)[1:])
        xy = np.stack((X.ravel(), Y.ravel()), axis=1)
        mask = np.zeros(xy.shape[0], dtype=bool)
        for i in range(xy.shape[0]):
            point = sg.Point(xy[i])
            if point.within(polygon):
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
            # plane 'Missed' part
            print('M', end='')
            return

        # triangulate the cross section
        self.path2d, self.xform = path3d.to_planar()
        v, f = [], []
        for polygon in self.path2d.polygons_full:
            tri = trimesh.creation.triangulate_polygon(polygon, triangle_args='p', allow_boundary_steiner=False)
            v.append(tri[0])
            f.append(tri[1])
        vf, ff = trimesh.util.append_faces(v, f)
        vf = np.column_stack((vf, np.zeros(len(vf))))
        vf = trimesh.transform_points(vf, self.xform)
        ff = np.fliplr(ff)
        # create cap which is the same for both sides of split
        self.mesh = trimesh.Trimesh(vf, ff)
        self.valid = True

    def split(self, mesh):
        positive_sliced = mesh.slice_plane(plane_origin=self.origin, plane_normal=self.normal)
        positive_capped = positive_sliced + self.mesh
        positive_capped._validate = True
        positive_capped.process()
        positive_capped.fix_normals()
        positive_capped.remove_degenerate_faces()

        negative_sliced = mesh.slice_plane(plane_origin=self.origin, plane_normal=-1 * self.normal)
        negative_capped = negative_sliced + self.mesh
        negative_capped._validate = True
        negative_capped.process()
        negative_capped.fix_normals()
        negative_capped.remove_degenerate_faces()
        return positive_capped, negative_capped

    def find_connector_sites(self, positive, negative):
        for polygon in self.path2d.polygons_full:
            cc = ConnectedComponent(self, polygon, positive, negative)
            if cc.valid:
                self.connected_components.append(cc)
            else:
                return False
        return True

    def get_average_connector_size(self):
        return sum([cc.connector_diameter for cc in self.connected_components]) / len(self.connected_components)


class BSPMesh(trimesh.Trimesh):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convex_hull = None

    @classmethod
    def from_trimesh(cls, mesh, chull):
        bspmesh = cls(mesh.vertices, mesh.faces)
        bspmesh.convex_hull = chull
        return bspmesh

    @property
    def convex_hull(self):
        return self._convex_hull

    @convex_hull.setter
    def convex_hull(self, chull):
        self._convex_hull = chull

    def bidirectional_split(self, origin, normal):
        """https://github.com/mikedh/trimesh/issues/235"""
        cross_section = CrossSection(self, origin, normal)

        if not cross_section.valid:
            return None, None, None

        positive, negative = cross_section.split(self)
        positive_chull, negative_chull = cross_section.split(self.convex_hull)
        positive = BSPMesh.from_trimesh(positive, positive_chull)
        negative = BSPMesh.from_trimesh(negative, negative_chull)

        return positive, negative, cross_section

    # def copy(self, include_cache=False):
    #     copied = super().copy(False)
    #     copied_chull = self.convex_hull.copy(False)
    #     return BSPMesh.from_trimesh(copied, copied_chull)
