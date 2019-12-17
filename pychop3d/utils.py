import numpy as np
import trimesh

from pychop3d import constants
from pychop3d import bsp
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


def insert_connectors(tree, state):
    new_tree = bsp.BSPTree(tree.nodes[0].part)
    cc_last = None
    for i in range(state.sum()):
        cc = tree.connected_component[state][i]
        if cc != cc_last:
            node = tree.nodes[tree.connected_component_nodes[cc]]
            new_tree = new_tree.expand_node(node.plane, node)
            new_node = new_tree.get_node(node.path)
            cc_last = cc

        xform = tree.connectors[state][i].primitive.transform
        slot = trimesh.primitives.Box(
            extents=np.ones(3) * (constants.CONNECTOR_DIAMETER + constants.CONNECTOR_TOLERANCE),
            transform=xform)

        if tree.sides[state][i] == 1:
            new_node.children[0].part = new_node.children[0].part.difference(slot, engine='scad')
            new_node.children[1].part = new_node.children[1].part.union(tree.connectors[state][i], engine='scad')
        else:
            new_node.children[1].part = new_node.children[1].part.difference(slot, engine='scad')
            new_node.children[0].part = new_node.children[0].part.union(tree.connectors[state][i], engine='scad')

    return new_tree
