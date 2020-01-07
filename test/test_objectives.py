import trimesh
import numpy as np
import os

from pychop3d import bsp
from pychop3d import section
from pychop3d import objective_functions
from pychop3d import utils
from pychop3d.configuration import Configuration


def test_number_of_parts():
    # test on a small sphere
    mesh = trimesh.primitives.Sphere(radius=10)

    tree = bsp.BSPTree(mesh)
    assert tree.nparts_objective() == 1
    assert tree.nodes[0].n_parts == 1
    # test on a large box
    mesh = trimesh.primitives.Box(extents=(50, 50, 220))

    tree = bsp.BSPTree(mesh)
    assert tree.nparts_objective() == 1
    assert tree.nodes[0].n_parts == 2
    # test splitting the box into 2 through the middle
    tree = tree.expand_node((np.zeros(3), np.array([0, 0, 1])), tree.nodes[0])
    assert tree.nparts_objective() == 1
    assert tree.get_node((0,)).n_parts == 1
    assert tree.get_node((1,)).n_parts == 1
    # rotate the box
    mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
    tree = bsp.BSPTree(mesh)
    assert tree.nparts_objective() == 1
    assert tree.nodes[0].n_parts == 2


def test_utilization():
    # check that as the sphere gets larger, the utilization goes down
    mesh1 = trimesh.primitives.Sphere(radius=20)
    tree1 = bsp.BSPTree(mesh1)
    mesh2 = trimesh.primitives.Sphere(radius=40)
    tree2 = bsp.BSPTree(mesh2)
    mesh3 = trimesh.primitives.Sphere(radius=60)
    tree3 = bsp.BSPTree(mesh3)
    print(f"\n{tree1.utilization_objective()} > {tree2.utilization_objective()} > {tree3.utilization_objective()}")
    assert tree1.utilization_objective() > tree2.utilization_objective() > tree3.utilization_objective()

    # check that a slice in the middle has a better utilization than a slice not down middle
    mesh = trimesh.primitives.Box(extents=(100., 100., 220.))
    tree1 = bsp.BSPTree(mesh)
    tree1 = tree1.expand_node((np.zeros(3), np.array([0., 0., 1.])), tree1.nodes[0])
    tree2 = bsp.BSPTree(mesh)
    tree2 = tree2.expand_node((np.array([0., 0., 100.]), np.array([0., 0., 1.])), tree2.nodes[0])
    print(f"\n{tree1.utilization_objective()} < {tree2.utilization_objective()}")
    assert tree1.utilization_objective() < tree2.utilization_objective()


def test_fragility():
    config = Configuration.config
    mesh = trimesh.primitives.Box(extents=[50, 50, 200]).subdivide()

    tree = bsp.BSPTree(mesh)
    tree = tree.expand_node((np.array([0, 0, 100 - 1.5 * config.connector_diameter_max - 1]), np.array([0, 0, 1])), tree.nodes[0])
    fragility = tree.fragility_objective()
    assert fragility == 0
    tree = bsp.BSPTree(mesh)
    tree = tree.expand_node((np.array([0, 0, 100 - 1.5 * config.connector_diameter_min + 1]), np.array([0, 0, 1])), tree.nodes[0])
    fragility = tree.fragility_objective()
    assert fragility == np.inf


def test_fragility_function_already_fragile():
    mesh_fn = os.path.join(os.path.dirname(__file__), 'test_meshes', 'fragility_test_1.stl')
    mesh = trimesh.load(mesh_fn)
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()

    tree = bsp.BSPTree(mesh)
    origin = np.zeros(3)
    normal = np.array([0., 0., 1.])
    plane = (origin, normal)
    non_fragile_cut_tree = tree.expand_node(plane, tree.nodes[0])
    assert objective_functions.get_fragility_objective([non_fragile_cut_tree], tree.nodes[0].path)[0] == 0


def test_fragility_function_multiple_trees():
    config = Configuration.config
    config.plane_spacing = 5
    config.adaptive_connector_size = False
    config.connector_diameter = 5
    mesh_fn = os.path.join(os.path.dirname(__file__), 'test_meshes', 'fragility_test_1.stl')
    mesh = trimesh.load(mesh_fn)
    mesh = mesh.subdivide()
    mesh = mesh.subdivide()

    tree = bsp.BSPTree(mesh)
    normal = np.array([0., 0., 1.])
    planes = tree.nodes[0].get_planes(normal)
    trees = []
    for plane in planes:
        candidate = tree.expand_node(plane, tree.nodes[0])
        trees.append(candidate)
    fragility = objective_functions.get_fragility_objective(trees, tree.nodes[0].path)
    assert fragility[0] == np.inf
    assert fragility[6] == np.inf
    assert fragility[7] == np.inf
    assert fragility[11] == np.inf
    config.restore_defaults()


def test_edge_fragility():
    config = Configuration.config
    config.adaptive_connector_size = False
    config.connector_diameter = 1
    mesh_fn = os.path.join(os.path.dirname(__file__), 'test_meshes', 'fragility_test_2.stl')
    mesh = trimesh.load(mesh_fn)

    tree = bsp.BSPTree(mesh)
    origin = np.zeros(3)
    normal = np.array([0., 0., 1.])
    plane = (origin, normal)
    fragile_cut_tree = tree.expand_node(plane, tree.nodes[0])
    assert objective_functions.get_fragility_objective([fragile_cut_tree], tree.nodes[0].path)[0] == np.inf