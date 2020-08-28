import trimesh
import numpy as np
import os

from pychop3d import bsp_tree
from pychop3d import objective_functions


def test_number_of_parts():
    # test on a small sphere
    mesh = trimesh.primitives.Sphere(radius=10)

    tree = bsp_tree.BSPTree(mesh)
    assert tree.objectives['nparts'] == 1
    assert tree.nodes[0].n_parts == 1
    # test on a large box
    mesh = trimesh.primitives.Box(extents=(50, 50, 220))

    tree = bsp_tree.BSPTree(mesh)
    assert tree.objectives['nparts'] == 1
    assert tree.nodes[0].n_parts == 2
    # test splitting the box into 2 through the middle
    tree, result = bsp_tree.expand_node(tree, tree.nodes[0].path, (np.zeros(3), np.array([0, 0, 1])))
    assert tree.objectives['nparts'] == 1
    assert tree.get_node((0,)).n_parts == 1
    assert tree.get_node((1,)).n_parts == 1
    # rotate the box
    mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
    tree = bsp_tree.BSPTree(mesh)
    assert tree.objectives['nparts'] == 1
    assert tree.nodes[0].n_parts == 2


def test_utilization():
    # check that as the sphere gets larger, the utilization goes down
    mesh1 = trimesh.primitives.Sphere(radius=20)
    tree1 = bsp_tree.BSPTree(mesh1)
    mesh2 = trimesh.primitives.Sphere(radius=40)
    tree2 = bsp_tree.BSPTree(mesh2)
    mesh3 = trimesh.primitives.Sphere(radius=60)
    tree3 = bsp_tree.BSPTree(mesh3)
    print(f"\n{tree1.objectives['utilization']} > {tree2.objectives['utilization']} > {tree3.objectives['utilization']}")
    assert tree1.objectives['utilization'] > tree2.objectives['utilization'] > tree3.objectives['utilization']

    # check that a slice in the middle has a better utilization than a slice not down middle
    mesh = trimesh.primitives.Box(extents=(100., 100., 220.))
    tree1 = bsp_tree.BSPTree(mesh)
    tree1, result = bsp_tree.expand_node(tree1, tree1.nodes[0].path, (np.zeros(3), np.array([0., 0., 1.])))
    tree2 = bsp_tree.BSPTree(mesh)
    tree2, result = bsp_tree.expand_node(tree2, tree2.nodes[0].path, (np.array([0., 0., 100.]), np.array([0., 0., 1.])))
    objective_functions.evaluate_utilization_objective([tree1, tree2], ())
    print(f"\n{tree1.objectives['utilization']} < {tree2.objectives['utilization']}")
    assert tree1.objectives['utilization'] < tree2.objectives['utilization']


def test_fragility_function_already_fragile():
    mesh_fn = os.path.join(os.path.dirname(__file__), 'test_meshes', 'fragility_test_1.stl')
    mesh = trimesh.load(mesh_fn)
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()

    tree = bsp_tree.BSPTree(mesh)
    origin = np.zeros(3)
    normal = np.array([0., 0., 1.])
    plane = (origin, normal)
    trees = [bsp_tree.expand_node(tree, tree.nodes[0].path, plane)[0]]
    objective_functions.evaluate_fragility_objective(trees, tree.nodes[0].path)
    assert trees[0].objectives['fragility'] == 0


def test_fragility_function_multiple_trees(config):
    config.plane_spacing = 5
    config.connector_diameter = 5
    mesh_fn = os.path.join(os.path.dirname(__file__), 'test_meshes', 'fragility_test_1.stl')
    mesh = trimesh.load(mesh_fn)
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()

    tree = bsp_tree.BSPTree(mesh)
    normal = np.array([0., 0., 1.])
    planes = bsp_tree.get_planes(mesh, normal)
    trees = []
    for plane in planes:
        candidate, result = bsp_tree.expand_node(tree, tree.nodes[0].path, plane)
        trees.append(candidate)
    objective_functions.evaluate_fragility_objective(trees, tree.nodes[0].path)
    assert trees[0].objectives['fragility'] == np.inf
    assert trees[6].objectives['fragility'] == np.inf
    assert trees[7].objectives['fragility'] == np.inf
    assert trees[11].objectives['fragility'] == np.inf


def test_edge_fragility(config):
    config.connector_diameter = 3
    mesh_fn = os.path.join(os.path.dirname(__file__), 'test_meshes', 'fragility_test_2.stl')
    mesh = trimesh.load(mesh_fn)
    mesh = mesh.subdivide()

    tree = bsp_tree.BSPTree(mesh)
    origin = np.zeros(3)
    normal = np.array([0., 0., 1.])
    plane = (origin, normal)
    fragile_cut_tree, result = bsp_tree.expand_node(tree, tree.nodes[0].path, plane)
    objective_functions.evaluate_fragility_objective([fragile_cut_tree], tree.nodes[0].path)
    assert fragile_cut_tree.objectives['fragility'] == np.inf
