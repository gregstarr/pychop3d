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


def unidirectional_split(mesh, origin, normal):
    """https://github.com/mikedh/trimesh/issues/235"""
    s = mesh.section(plane_origin=origin, plane_normal=normal)
    # check for the possibility that the plane passes between separate components and doesn't intersect with anything
    if s is None:
        return None, None
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
    cap = trimesh.Trimesh(vf, ff)
    sliced = mesh.slice_plane(plane_origin=origin, plane_normal=normal)
    capped = sliced + cap
    capped._validate = True
    capped.process()
    capped.fix_normals()
    if np.any(capped.extents < constants.EPSILON):
        return None, None, None
    capped.remove_degenerate_faces()
    # connector sites
    max_sites = None
    max_objective = 0
    for polygon in on_plane.polygons_full:
        plane_samples = grid_sample_polygon(polygon)
        mesh_samples = trimesh.transform_points(np.column_stack((plane_samples, np.zeros(plane_samples.shape[0]))), to_3D)
        if mesh_samples.size == 0:
            return capped, None, np.inf
        dists = capped.nearest.signed_distance(mesh_samples + (1 + constants.CONNECTOR_DIAMETER) * normal)
        valid_mask = dists > constants.CONNECTOR_DIAMETER
        if not np.any(valid_mask):
            return capped, None, np.inf
        convex_hull_area = SG.MultiPoint(plane_samples[valid_mask]).buffer(constants.CONNECTOR_DIAMETER/2).convex_hull.area
        component_area = polygon.area
        objective = max(component_area / convex_hull_area - constants.CONNECTOR_OBJECTIVE_THRESHOLD, 0)
        if objective > max_objective:
            max_objective = objective
            max_sites = mesh_samples[valid_mask]
    return capped, max_sites, max_objective


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