"""
TODO:
    - node subclasses
        - plane node
        - root node
        - separation node
    - random thing downlaoder
    - config
    - chopper class
    - cross section area penalty
    - tests for all objectives
        - fragility has known issues
    - other connectors
        - tabs for bolting
        - shell / sheath type
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
mesh, config = utils.open_mesh(config)

t0 = time.time()
best_tree = beam_search(mesh, config)
print(f"Best BSP-tree found in {time.time() - t0} seconds")
best_tree.save("final_tree.json", config)

t0 = time.time()
connector_placer = connector.ConnectorPlacer(best_tree)
connector_placer.simulated_annealing_connector_placement()
print(f"Best connector arrangement found in {time.time() - t0} seconds")