import trimesh
import os
import numpy as np
import pytest
import glob

from pychop3d import section
from pychop3d import search
from pychop3d import bsp
from pychop3d import utils
from pychop3d.configuration import Configuration


@pytest.mark.parametrize('file_number', range(1, 4))
def test_regression(file_number):
    print()
    # files
    tree_file = f"regression_tree_{file_number}.json"
    config_file = f"regression_config_{file_number}.yml"
    tree_file = os.path.join(os.path.dirname(__file__), 'regression_test_data', tree_file)
    config_file = os.path.join(os.path.dirname(__file__), 'regression_test_data', config_file)
    Configuration.config = Configuration(config_file)

    # open and prepare mesh
    mesh = utils.open_mesh()
    # open tree baseline
    baseline = utils.open_tree(mesh, tree_file)
    # run new tree
    tree = search.beam_search(mesh)
    # verify they are the same
    print()
    for baseline_node in baseline.nodes:
        print(f"path: {baseline_node.path}")
        # same path
        node = tree.get_node(baseline_node.path)
        if node.plane is None:
            assert baseline_node.plane is None               
        else:
            # same origin
            print(f"baseline origin {baseline_node.plane[0]}, test origin {node.plane[0]}")
            # same normal
            print(f"baseline normal {baseline_node.plane[1]}, test normal {node.plane[1]}")
            assert np.all(baseline_node.plane[0] == node.plane[0])
            assert np.all(baseline_node.plane[1] == node.plane[1])

    config = Configuration.config
    for i in range(config.beam_width):
        os.remove(os.path.join(os.path.dirname(__file__), 'regression_test_data', f'{i}.json'))
    for stl in glob.glob(os.path.join(os.path.dirname(__file__), 'regression_test_data', '*.stl')):
        os.remove(stl)
    config.restore_defaults()
