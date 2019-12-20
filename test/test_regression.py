import trimesh
import os
import numpy as np

from pychop3d import bsp_mesh
from pychop3d import search
from pychop3d import bsp


def test_regression():
    # files
    mesh_file = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
    tree_file = os.path.join(os.path.dirname(__file__), "regression_test_tree.json")
    # open and prepare mesh
    mesh = trimesh.load(mesh_file, validate=True)
    mesh.apply_scale(2)
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
