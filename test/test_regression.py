import trimesh
import os
import numpy as np
import pytest

from pychop3d import bsp_mesh
from pychop3d import search
from pychop3d import bsp
from pychop3d.configuration import Configuration


@pytest.mark.parametrize('file_number', range(1, 3))
def test_regression(file_number):
    print()
    # files
    tree_file = f"regression_tree_{file_number}.json"
    config_file = f"regression_config_{file_number}.yml"
    tree_file = os.path.join(os.path.dirname(__file__), tree_file)
    config_file = os.path.join(os.path.dirname(__file__), config_file)
    config = Configuration(config_file)
    Configuration.config = config

    # open and prepare mesh
    mesh = trimesh.load(config.mesh, validate=True)
    mesh.apply_scale(config.scale_factor)
    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh)
    mesh._convex_hull = chull
    # open tree baseline
    baseline, *_ = bsp.BSPTree.from_json(mesh, tree_file)
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
            assert np.all(baseline_node.plane[0] == node.plane[0])
            # same normal
            assert np.all(baseline_node.plane[1] == node.plane[1])
