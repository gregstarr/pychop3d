import numpy as np
import trimesh
from shapely import geometry as SG

from pychop3d import constants


def all_at_goal(trees):
    for tree in trees:
        if not tree.terminated():
            return False
    return True


def not_at_goal_set(trees):
    not_at_goal = []
    for tree in trees:
        if not tree.terminated():
            not_at_goal.append(tree)
    return not_at_goal


def uniform_normals(n=constants.N_RANDOM_NORMALS):
    """http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    theta = np.random.rand(n) * 2 * np.pi
    phi = np.arccos(1 - 2 * np.random.rand(n))
    return np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=1)


def bidirectional_split(mesh, origin, normal, get_connections=True):
    """https://github.com/mikedh/trimesh/issues/235"""
    s = mesh.section(plane_origin=origin, plane_normal=normal)
    # check for the possibility that the plane passes between separate components and doesn't intersect with anything
    if s is None:
        return None, None, None
    # triangulate the cross section
    on_plane, to_3D = s.to_planar()
    v, f = [], []
    for polygon in on_plane.polygons_full:
        tri = trimesh.creation.triangulate_polygon(polygon, triangle_args='p', allow_boundary_steiner=False)
        v.append(tri[0])
        f.append(tri[1])
    vf, ff = trimesh.util.append_faces(v, f)
    vf = np.column_stack((vf, np.zeros(len(vf))))
    vf = trimesh.transform_points(vf, to_3D)
    ff = np.fliplr(ff)
    # create cap which is the same for both sides of split
    cap = trimesh.Trimesh(vf, ff)

    # get positive section
    positive_sliced = mesh.slice_plane(plane_origin=origin, plane_normal=normal)
    positive_capped = positive_sliced + cap
    positive_capped._validate = True
    positive_capped.process()
    positive_capped.fix_normals()
    if np.any(positive_capped.extents < constants.EPSILON):
        return None, None, None
    positive_capped.remove_degenerate_faces()

    # get negative section
    negative_sliced = mesh.slice_plane(plane_origin=origin, plane_normal=-1 * normal)
    negative_capped = negative_sliced + cap
    negative_capped._validate = True
    negative_capped.process()
    negative_capped.fix_normals()
    if np.any(negative_capped.extents < constants.EPSILON):
        return None, None, None
    negative_capped.remove_degenerate_faces()

    if not get_connections:
        return positive_capped, negative_capped
    # find connector sites
    connected_components_site_info = []  # list of dictionaries, one for each connected component
    for polygon in on_plane.polygons_full:
        plane_samples = grid_sample_polygon(polygon)
        mesh_samples = trimesh.transform_points(np.column_stack((plane_samples, np.zeros(plane_samples.shape[0]))), to_3D)
        if mesh_samples.size == 0:
            return None, None, None
        pos_dists = positive_capped.nearest.signed_distance(mesh_samples + (1 + constants.CONNECTOR_DIAMETER) * normal)
        neg_dists = negative_capped.nearest.signed_distance(mesh_samples + (1 + constants.CONNECTOR_DIAMETER) * -1 * normal)
        pos_valid_mask = pos_dists > constants.CONNECTOR_DIAMETER
        neg_valid_mask = neg_dists > constants.CONNECTOR_DIAMETER
        if not np.any(pos_valid_mask) and not np.any(neg_valid_mask):
            return None, None, None
        ch_area_mask = np.logical_or(pos_valid_mask, neg_valid_mask)
        convex_hull_area = SG.MultiPoint(plane_samples[ch_area_mask]).buffer(constants.CONNECTOR_DIAMETER/2).convex_hull.area
        component_area = polygon.area
        objective = max(component_area / convex_hull_area - constants.CONNECTOR_OBJECTIVE_THRESHOLD, 0)
        pos_sites = mesh_samples[pos_valid_mask]
        pos_normals = np.ones((pos_valid_mask.sum(), 1)) * normal[None, :]
        neg_sites = mesh_samples[neg_valid_mask]
        neg_normals = np.ones((neg_valid_mask.sum(), 1)) * -1 * normal[None, :]
        sites = np.concatenate((pos_sites, neg_sites), axis=0)
        normals = np.concatenate((pos_normals, neg_normals), axis=0)
        side = np.concatenate((np.ones(pos_valid_mask.sum()), -1 * np.ones(neg_valid_mask.sum())))
        connected_component_dict = {'sites': sites, 'normals': normals, 'side': side,
                                    'area': component_area, 'objective': objective}
        connected_components_site_info.append(connected_component_dict)

    return positive_capped, negative_capped, connected_components_site_info


def grid_sample_polygon(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    X, Y = np.meshgrid(np.arange(min_x, max_x, constants.CONNECTOR_DIAMETER)[1:],
                       np.arange(min_y, max_y, constants.CONNECTOR_DIAMETER)[1:])
    xy = np.stack((X.ravel(), Y.ravel()), axis=1)
    mask = np.zeros(xy.shape[0], dtype=bool)
    for i in range(xy.shape[0]):
        point = SG.Point(xy[i])
        if point.within(polygon):
            mask[i] = True
    return xy[mask]


def plane(normal, origin, w=100):
    xform = np.linalg.inv(trimesh.points.plane_transform(origin, normal))
    box = trimesh.primitives.Box(extents=(w, w, .5), transform=xform)
    return box