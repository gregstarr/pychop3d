import numpy as np
import trimesh

from pychop3d import constants
from pychop3d import bsp_mesh
from pychop3d import bsp


def open_mesh(cfg):
    mesh = trimesh.load(cfg.mesh)

    if cfg.scale:
        if hasattr(cfg, 'factor'):
            factor = cfg.factor
        else:
            factor = np.ceil(1.1 / np.max(mesh.extents / cfg.printer_extents))
        if factor > 1:
            mesh.apply_scale(factor)
            cfg.factor = factor

    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh, chull)

    return mesh


def open_tree(cfg):
    mesh = cfg.open_mesh()
    tree = bsp.BSPTree(mesh)
    if cfg.nodes is not None:
        for n in cfg.nodes:
            plane = (np.array(n['origin']), np.array(n['normal']))
            node = tree.get_node(n['path'])
            tree = tree.expand_node(plane, node)
    return tree


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

