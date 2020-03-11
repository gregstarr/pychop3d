import time
import datetime
import os
import sys
import logging

from pychop3d.search import beam_search
from pychop3d import connector
from pychop3d.configuration import Configuration
from pychop3d import utils


logger = logging.getLogger(__name__)


def run(starter):
    """This function goes through the complete Pychop3D process, including the beam search for the optimal
    cutting planes, determining the connector locations, adding the connectors to the part meshes, then saving the
    STLs, the tree json and the configuration file.

    :param starter: Either an unpartitioned mesh or an already partitioned tree to begin the process using
    :type starter: `trimesh.Trimesh`
    :type starter: `bsp_tree.BSPTree`
    """
    # mark starting time
    t0 = time.time()
    # complete the beam search using the starter, no search will take place if the starter tree is already
    # adequately partitioned
    tree = beam_search(starter)
    logger.info(f"Best BSP-tree found in {time.time() - t0} seconds")
    # save the tree now in case the connector placement fails
    utils.save_tree(tree, "final_tree.json")

    # mark starting time
    t0 = time.time()
    logger.info("finding best connector arrangement")
    # create connector placer object, this creates all potential connectors and determines their collisions
    connector_placer = connector.ConnectorPlacer(tree)
    if connector_placer.n_connectors > 0:
        # use simulated annealing to determine the best combination of connectors
        state = connector_placer.simulated_annealing_connector_placement()
        logger.info(f"Best connector arrangement found in {time.time() - t0} seconds")
        # save the final tree including the state
        utils.save_tree(tree, "final_tree_with_connectors.json", state)
        # add the connectors / subtract the slots from the parts of the partitioned input object
        logger.info(f"inserting {state.sum()} connectors...")
        tree = connector_placer.insert_connectors(tree, state)

    # export the parts of the partitioned object
    utils.export_tree_stls(tree)
    logger.info("Finished")


if __name__ == "__main__":
    name = "picture_frame_back"
    # name the folder based on the name of the object and the current date / time
    output_folder = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # create the new directory in the 'output' subdirectory of pychop3d
    new_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', output_folder))
    os.mkdir(new_directory)
    # set configuration options
    config = Configuration.config
    config.name = name
    config.mesh = "C:\\Users\\Greg\\Documents\\things\\poster frame 2\\back.stl"
    config.directory = new_directory
    config.beam_width = 3
    config.connector_diameter = 6
    config.connector_spacing = 10
    config.part_separation = False
    
    # save configuration
    config.save()
    # basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  [%(levelname)s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.directory, "info.log")),
            logging.StreamHandler()
        ]
    )
    # open the input mesh as the starter
    starter = utils.open_mesh()
    # separate pieces
    if config.part_separation and starter.body_count > 1:
        starter = utils.separate_starter(starter)
    # run through the process
    run(starter)
