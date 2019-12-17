import numpy as np
import trimesh

from pychop3d import constants
from pychop3d import bsp_mesh


def open_mesh(config):
    mesh = trimesh.load(config['mesh'])

    if 'scale' in config and config['scale']:
        if 'factor' in config:
            factor = config['factor']
        else:
            factor = np.ceil(1.1 / np.max(mesh.extents / constants.PRINTER_EXTENTS))
        if factor > 1:
            mesh.apply_scale(factor)
            config['factor'] = factor
    else:
        config['scale'] = False

    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh, chull)

    return mesh, config


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


def get_unique_normals(non_unique_normals):
    rounded = np.round(non_unique_normals, 3)
    view = rounded.view(dtype=[('', float), ('', float), ('', float)])
    unique = np.unique(view)
    return unique.view(dtype=float).reshape((unique.shape[0], -1))


def plane(normal, origin, w=100):
    xform = np.linalg.inv(trimesh.points.plane_transform(origin, normal))
    box = trimesh.primitives.Box(extents=(w, w, .5), transform=xform)
    return box

