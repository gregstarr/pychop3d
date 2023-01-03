import os
import numpy as np
import pytest
import glob
import tempfile

from pychop3d import search
from pychop3d import utils


@pytest.mark.parametrize('file_number', range(1, 4))
def test_regression(config, file_number):
    print()
    # files
    tree_file = f"regression_tree_{file_number}.json"
    config_file = f"regression_config_{file_number}.yml"
    tree_file = os.path.join(os.path.dirname(__file__), 'test_data', tree_file)
    config_file = os.path.join(os.path.dirname(__file__), 'test_data', config_file)
    config.load(config_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        config.directory = tmpdir
        # open and prepare mesh
        mesh = utils.open_mesh()
        # open tree baseline
        baseline = utils.open_tree(tree_file)
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
                assert np.allclose(baseline_node.plane[0], node.plane[0])
                assert np.allclose(baseline_node.plane[1], node.plane[1])
