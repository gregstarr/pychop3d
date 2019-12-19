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
    - calculate objectives for many planes at once
        - fragility speedup
        - trimesh.intersections.mesh_multiplane
"""
import time

from pychop3d.search import beam_search
from pychop3d import connector
from pychop3d.config import Configuration
from pychop3d import utils


def run():
    cfg = Configuration.cfg
    if cfg.nodes is None:
        starter = utils.open_mesh(cfg)
    else:
        starter = utils.open_tree(cfg)

    t0 = time.time()
    tree = beam_search(starter, cfg)
    print(f"Best BSP-tree found in {time.time() - t0} seconds")
    tree.save("final_tree.json")

    t0 = time.time()
    connector_placer = connector.ConnectorPlacer(tree)
    state = connector_placer.simulated_annealing_connector_placement()
    tree = connector_placer.insert_connectors(tree, state)
    print(f"Best connector arrangement found in {time.time() - t0} seconds")

    tree.export_stl(cfg)
    tree.save("final_tree_with_connectors.json", cfg, state)


if __name__ == "__main__":
    run()
