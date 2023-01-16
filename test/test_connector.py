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
import pytest

from pychop3d import bsp_tree, connector, settings
from pychop3d.bsp_node import Plane


pytest.skip("haven't written a good test for this yet", allow_module_level=True)


def test_sa_objective_1():
    """Verifies:
        - connected components without a connector are penalized
        - small connected components with a single connector have a reasonably low objective
        - connected components with a connector collision are penalized
    """
    mesh = trimesh.primitives.Box(extents=[11, 11, 40])
    printer_extents = np.ones(3) * 200
    tree = bsp_tree.BSPTree(mesh, printer_extents)
    normal = np.array([0, 0, 1])
    origin = np.zeros(3)
    tree, result = bsp_tree.expand_node(tree, tree.nodes[0].path, Plane(origin, normal))
    print(result)
    connector_placer = connector.ConnectorPlacer(tree)
    assert connector_placer.evaluate_connector_objective(np.zeros(connector_placer.n_connectors, dtype=bool)) >= 1 / settings.EMPTY_CC_PENALTY
    ob2 = connector_placer.evaluate_connector_objective(np.array([False, True]))
    ob3 = connector_placer.evaluate_connector_objective(np.array([True, False]))
    assert ob2 == ob3
    assert ob2 < 5
    assert connector_placer.evaluate_connector_objective(np.array([True, True])) >= settings.CONNECTOR_COLLISION_PENALTY


@pytest.fixture
def set_part_connector_spacing():
    original = settings.CONNECTOR_SPACING
    settings.CONNECTOR_SPACING = 5
    yield
    settings.CONNECTOR_SPACING = original


def test_sa_objective_2(set_part_connector_spacing):
    """Verifies:
        - large faces prefer multiple connectors

        NOTE: every time grid_sample code changes, this will need to be changed which obviously isnt ideal
    """
    mesh = trimesh.primitives.Box(extents=[30, 30, 80]).as_mesh()
    printer_extents = np.ones(3) * 200
    tree = bsp_tree.BSPTree(mesh, printer_extents)
    normal = np.array([0, 0, 1])
    origin = np.zeros(3)
    tree, result = bsp_tree.expand_node(tree, tree.nodes[0].path, Plane(origin, normal))
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
