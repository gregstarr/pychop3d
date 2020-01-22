"""
components to test:
    - evaluate_connector_objective
        - verify that connected components without a connector are heavily penalized
        - verify that the behavior is defined for zero connectors
        - verify that connector collisions are detected and penalized
        - verify that connector-cut intersections are detected and penalized
        - verify that large faces prefer multiple connectors
"""
import trimesh
import numpy as np

from pychop3d import bsp_tree
from pychop3d import connector
from pychop3d.configuration import Configuration


def test_sa_objective_1():
    """Verifies:
        - connected components without a connector are penalized
        - small connected components with a single connector have a reasonably low objective
        - connected components with a connector collision are penalized
    """
    config = Configuration.config
    mesh = trimesh.primitives.Box(extents=[10, 10, 40])
    tree = bsp_tree.BSPTree(mesh)
    normal = np.array([0, 0, 1])
    origin = np.zeros(3)
    tree = bsp_tree.expand_node(tree, tree.nodes[0].path, (origin, normal))
    connector_placer = connector.ConnectorPlacer(tree)
    assert connector_placer.evaluate_connector_objective(np.array([False, False])) >= 1 / config.empty_cc_penalty
    ob2 = connector_placer.evaluate_connector_objective(np.array([False, True]))
    ob3 = connector_placer.evaluate_connector_objective(np.array([True, False]))
    assert ob2 == ob3
    assert ob2 < 5
    assert connector_placer.evaluate_connector_objective(np.array([True, True])) >= config.connector_collision_penalty


def test_sa_objective_2():
    """Verifies:
        - large faces prefer multiple connectors
    """
    config = Configuration.config
    mesh = trimesh.primitives.Box(extents=[30, 30, 80])
    tree = bsp_tree.BSPTree(mesh)
    normal = np.array([0, 0, 1])
    origin = np.zeros(3)
    tree = bsp_tree.expand_node(tree, tree.nodes[0].path, (origin, normal))
    connector_placer = connector.ConnectorPlacer(tree)

    # single connector
    state = np.zeros(connector_placer.n_connectors, dtype=bool)
    state[12] = True
    ob1 = connector_placer.evaluate_connector_objective(state)

    # double connector in opposite corners
    state = np.zeros(connector_placer.n_connectors, dtype=bool)
    state[0] = True
    state[24] = True
    ob2 = connector_placer.evaluate_connector_objective(state)

    # connector in each corner
    state = np.zeros(connector_placer.n_connectors, dtype=bool)
    state[0] = True
    state[4] = True
    state[20] = True
    state[24] = True
    ob3 = connector_placer.evaluate_connector_objective(state)

    # connector in each corner and in middle (too many connectors)
    state = np.zeros(connector_placer.n_connectors, dtype=bool)
    state[0] = True
    state[4] = True
    state[12] = True
    state[20] = True
    state[24] = True
    ob4 = connector_placer.evaluate_connector_objective(state)

    assert ob1 > ob2 > ob3
    assert ob4 > ob3

    config.restore_defaults()
