"""
TODO:
    1) TESTS!
        - tests for all objectives
        - connector placement tests
        - make a stable master branch
        - random thing downlaoder
    2) features:
        - node subclasses
            - plane node
            - root node
            - separation node
        - other connectors
            - tabs for bolting
            - shell / sheath type
        - calculate objectives for many planes at once
            - fragility speedup
            - trimesh.intersections.mesh_multiplane
        - proper logging
    3) optional future ideas:
        - chopper class?
        - cross section area penalty?
        - website
"""
import time
import datetime
import os

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

    t0 = time.time()
    connector_placer = connector.ConnectorPlacer(tree)
    state = connector_placer.simulated_annealing_connector_placement()
    tree = connector_placer.insert_connectors(tree, state)
    print(f"Best connector arrangement found in {time.time() - t0} seconds")

    tree.export_stl()
    tree.save("final_tree_with_connectors.json", state)


if __name__ == "__main__":
    # name and save the config
    date_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', date_string))
    os.mkdir(new_directory)
    config = Configuration.config
    config.directory = new_directory

    config.save(f"{date_string}_config.yml")
    run()
