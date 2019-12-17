"""
TODO:
    1) rebuild
        - node subclasses
        - random thing downlaoder
        - config
        - chopper class
"""
import trimesh
import os
import time
import datetime

from pychop3d.search import beam_search
from pychop3d.bsp_mesh import BSPMesh
from pychop3d.bsp import BSPTree
from pychop3d import utils
from pychop3d import constants
from pychop3d import connector

config = constants.default_config.copy()
config['scale'] = True
mesh, config = utils.open_mesh(config)

t0 = time.time()
best_tree = beam_search(mesh, config)
print(f"Best BSP-tree found in {time.time() - t0} seconds")

t0 = time.time()
connector_placer = connector.ConnectorPlacer(best_tree)
connector_placer.simulated_annealing_connector_placement()
print(f"Best connector arrangement found in {time.time() - t0} seconds")

new_directory = os.path.join(os.path.dirname(__file__), "..\\output", datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(new_directory, exist_ok=True)

best_tree.save(os.path.join(new_directory, "tree.json"), state)

best_tree = insert_connectors(best_tree, state)
best_tree.export_stl(new_directory)
