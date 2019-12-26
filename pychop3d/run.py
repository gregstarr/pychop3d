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
from pychop3d.configuration import Configuration

date_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
config = Configuration.config
config.plane_spacing = 30
config.scale_factor = 2
config.save(f"{date_string}_config.yml")

mesh = trimesh.load(config.mesh, validate=True)
mesh.apply_scale(config.scale_factor)
chull = mesh.convex_hull
mesh = BSPMesh.from_trimesh(mesh)
mesh._convex_hull = chull

t0 = time.time()
best_tree = beam_search(mesh)
print(f"Best BSP-tree found in {time.time() - t0} seconds")

best_tree.save(os.path.join(config.directory, "final_tree.json"), [])

t0 = time.time()
state = best_tree.simulated_annealing_connector_placement()
print(f"Best connector arrangement found in {time.time() - t0} seconds")

best_tree.save(os.path.join(config.directory, "tree.json"), state)

best_tree = insert_connectors(best_tree, state)
best_tree.export_stl(config.directory)
