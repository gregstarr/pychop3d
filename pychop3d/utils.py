import numpy as np
import trimesh
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
    return capped
