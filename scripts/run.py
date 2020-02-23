"""
TODO:
    - bugs:
        - open scad error
    - other connectors / connector class
        - tabs for bolting
        - shell / sheath type
    - python logging
    - instead of fixing the connector diameter based on some arbitrary function of the cc area,
        why not consider several different sizes of connector and let the SA objective function decide?
    - metadata
        - time taken for the run
        - number of faces, verts, edges
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


def run(starter):
    t0 = time.time()
    tree = beam_search(starter)
    print(f"Best BSP-tree found in {time.time() - t0} seconds")
    utils.save_tree(tree, "final_tree.json")

    try:
        t0 = time.time()
        connector_placer = connector.ConnectorPlacer(tree)
        state = connector_placer.simulated_annealing_connector_placement()
        tree = connector_placer.insert_connectors(tree, state)
        print(f"Best connector arrangement found in {time.time() - t0} seconds")
        utils.save_tree(tree, "final_tree_with_connectors.json", state)
    except Exception as e:
        print("\nConnector placement failed")
        print(e)

    utils.export_tree_stls(tree)

    config = Configuration.config
    config.save()


if __name__ == "__main__":
    output_folder = "poster_frame_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', output_folder))
    os.mkdir(new_directory)
    config = Configuration.config
    config.mesh = "C:\\Users\\Greg\\Documents\\things\\poster frame\\frame.stl"
    config.directory = new_directory
    config.beam_width = 3
    config.connector_diameter = 6
    config.connector_spacing = 10
    config.part_separation = True
    starter = utils.open_mesh()
    run(starter)
