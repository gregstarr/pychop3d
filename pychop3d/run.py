"""
TODO:
    - compute objective functions for multiple planes at a time
    - one ring edges
    - connection objective
    - seam objective
    - structural objective
    - symmetry objective
    - connector placement
    - improve robustness generally
"""
import trimesh
import os
from pychop3d.search import beam_search

fn = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
mesh = trimesh.load(fn, validate=True)
mesh.apply_scale(2)

best_tree = beam_search(mesh, 2)

for i, leaf in enumerate(best_tree.get_leaves()):
    leaf.part.export(os.path.join(os.path.dirname(__file__), f"../output/{i}.stl"))

