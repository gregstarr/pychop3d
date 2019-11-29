"""
TODO:
    - compute objective functions for multiple planes at a time
    - one ring edges
    - seam objective
    - structural objective
    - symmetry objective
    - clean up code big time
"""
import trimesh
import os
import time

from pychop3d.search import beam_search
from pychop3d.bsp_mesh import BSPMesh

fn = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
mesh = trimesh.load(fn, validate=True)
mesh.apply_scale(2.5)
chull = mesh.convex_hull
mesh = BSPMesh.from_trimesh(mesh)
mesh._convex_hull = chull

t0 = time.time()
best_tree = beam_search(mesh, 2)
print(f"Best BSP-tree found in {time.time() - t0} seconds")

t0 = time.time()
state = best_tree.simulated_annealing_connector_placement()
print(f"Best connector arrangement found in {time.time() - t0} seconds")

best_tree.insert_connectors(state)

for i, leaf in enumerate(best_tree.get_leaves()):
    leaf.part.export(os.path.join(os.path.dirname(__file__), f"../output/{i}.stl"))
