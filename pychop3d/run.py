"""
TODO:
    1) TESTS!
        - tests for all objectives
    2) features:
        - bugs:
            - memory issue
            - open scad error
        - oriented bounding box alternate
        - other connectors / connector class
            - tabs for bolting
            - shell / sheath type
        - calculate objectives for many planes at once
            - fragility speedup
            - trimesh.intersections.mesh_multiplane
        - proper logging
        - instead of fixing the connector diameter based on some arbitrary function of the cc area,
            why not consider several different sizes of connector and let the SA objective function decide?
        - metadata
            - time taken for the run
            - number of faces, verts, edges
    3) optional future ideas:
        - chopper class?
        - cross section area penalty?
        - website
"""
import time
import datetime
import os
import numpy as np

from pychop3d.search import beam_search
from pychop3d import connector
from pychop3d.configuration import Configuration
from pychop3d import utils


def run():
    # collect the already set config
    starter = utils.open_mesh()
    t0 = time.time()
    tree = beam_search(starter)
    print(f"Best BSP-tree found in {time.time() - t0} seconds")
    tree.save("final_tree.json")

    try:
        t0 = time.time()
        connector_placer = connector.ConnectorPlacer(tree)
        state = connector_placer.simulated_annealing_connector_placement()
        tree = connector_placer.insert_connectors(tree, state)
        print(f"Best connector arrangement found in {time.time() - t0} seconds")
    except:
        print("Connector placement failed")

    tree.export_stl()
    tree.save("final_tree_with_connectors.json", state)
    config = Configuration.config
    config.save()


if __name__ == "__main__":
    # name and save the config
    date_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', date_string))
    os.mkdir(new_directory)
    config = Configuration.config
    config.mesh = "C:\\Users\\Greg\\Documents\\shoe rack v12.stl"
    config.directory = new_directory
    config.beam_width = 3
    config.different_origin_th = 100
    run()
