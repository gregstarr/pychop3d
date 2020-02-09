import numpy as np
import trimesh
import json
import os

from pychop3d import bsp_tree
from pychop3d.configuration import Configuration


def open_mesh():
    """open the mesh according to the configuration and apply any scaling or subdivision
    """
    config = Configuration.config
    # OPEN MESH
    mesh = trimesh.load(config.mesh)
    # REPAIR MESH
    trimesh_repair(mesh)
    # SCALE MESH
    if config.scale_factor > 0:
        mesh.apply_scale(config.scale_factor)
    # SUBDIVIDE MESH
    pass

    return mesh


def open_tree(tree_file):
    mesh = open_mesh()
    with open(tree_file) as f:
        data = json.load(f)

    node_data = data['nodes']
    tree = bsp_tree.BSPTree(mesh)
    for n in node_data:
        plane = (np.array(n['origin']), np.array(n['normal']))
        node = tree.get_node(n['path'])
        tree = bsp_tree.expand_node(tree, node.path, plane)
    return tree


def all_at_goal(trees):
    for tree in trees:
        if not tree.terminated:
            return False
    return True


def not_at_goal_set(trees):
    not_at_goal = []
    for tree in trees:
        if not tree.terminated:
            not_at_goal.append(tree)
    return not_at_goal


def get_unique_normals(non_unique_normals):
    rounded = np.round(non_unique_normals, 3)
    view = rounded.view(dtype=[('', float), ('', float), ('', float)])
    unique = np.unique(view)
    return unique.view(dtype=float).reshape((unique.shape[0], -1))


def make_plane(origin, normal, w=100):
    xform = np.linalg.inv(trimesh.points.plane_transform(origin, normal))
    box = trimesh.primitives.Box(extents=(w, w, .5), transform=xform)
    return box


def trimesh_repair(mesh):
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)


def preview_tree(tree, other_objects=None):
    if other_objects is None:
        other_objects = []
    scene = trimesh.scene.Scene()
    for leaf in tree.leaves + other_objects:
        leaf.part.visual.face_colors = np.random.rand(3)*255
        scene.add_geometry(leaf.part)
    scene.camera.z_far = 10_000
    scene.show()


def save_tree(tree, filename, state=None):
    config = Configuration.config

    if state is None:
        state = []

    nodes = []
    for node in tree.nodes:
        if node.plane is not None:
            this_node = {'path': node.path, 'origin': list(node.plane[0]), 'normal': list(node.plane[1])}
            nodes.append(this_node)

    with open(os.path.join(config.directory, filename), 'w') as f:
        json.dump({'nodes': nodes, 'state': [bool(s) for s in state]}, f)


def export_tree_stls(tree):
    config = Configuration.config
    for i, leaf in enumerate(tree.leaves):
        leaf.part.export(os.path.join(config.directory, f"{i}.stl"))
