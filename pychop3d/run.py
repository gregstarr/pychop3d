"""
TODO:
    - compute objective functions for multiple planes at a time / general speedup
    - one ring edges
    - seam objective
    - structural objective
    - symmetry objective
    - clean up code big time
    - website
    - get random things from thingiverse and partition them
"""
import trimesh
import os
import time
import datetime

from pychop3d.search import beam_search
from pychop3d.bsp_mesh import BSPMesh
from pychop3d.bsp import BSPTree
from pychop3d.utils import insert_connectors

fn = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
mesh = trimesh.load(fn, validate=True)
mesh.apply_scale(2)
chull = mesh.convex_hull
mesh = BSPMesh.from_trimesh(mesh)
mesh._convex_hull = chull

t0 = time.time()
best_tree = beam_search(mesh)
print(f"Best BSP-tree found in {time.time() - t0} seconds")

t0 = time.time()
state = best_tree.simulated_annealing_connector_placement()
print(f"Best connector arrangement found in {time.time() - t0} seconds")

new_directory = os.path.join(os.path.dirname(__file__), "..\\output", datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(new_directory, exist_ok=True)

best_tree.save(os.path.join(new_directory, "tree.json"), state)

best_tree = insert_connectors(best_tree, state)
best_tree.export_stl(new_directory)
