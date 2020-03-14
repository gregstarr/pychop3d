"""
pychop3d - cli model chop utility

# INSTRUCTIONS:

1. Install dependencies from `requirements.txt`.
2. Run this script from the root of the pychop3d repository in the form.
```
   $ python main.py ./path/to/model.stl
```
3. Wait for chopping to finish. Chopped models will appear in the current working directory.
"""
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
    # basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  [%(levelname)s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.directory, "info.log")),
            logging.StreamHandler()
        ]
    )
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
    # Read mesh filepath from argument
    import argparse
    parser = argparse.ArgumentParser(description='Pychop3D command line runner')
    parser.add_argument('-c', '--config', type=str,
                        help='path to a config yaml file, for an example, see "bunny_config.yml"',
                        default="bunny_config.yml")
    parser.add_argument('-m', '--mesh', type=str, default=None,
                        help='Specify the mesh file path to chop. This will override the mesh file in the config yaml')
    args = parser.parse_args()

    # print help if no arguments specified
    if len(sys.argv) < 1:
        parser.print_help()
        sys.exit(0)

    # load specified or default config file   
    try:
        config = Configuration(args.config)
    except:
        parser.print_help()
        sys.exit(0)

    # override the mesh path in config if specified on command line
    if args.mesh:
        config.mesh = args.mesh
        config.name = os.path.splitext(os.path.basename(args.mesh)[0])

    # name the folder based on the name of the object and the current date / time
    output_folder = f"{config.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # create the new directory in the 'output' subdirectory of pychop3d
    new_directory = os.path.abspath(os.path.join(config.directory, output_folder))
    os.mkdir(new_directory)
    config.directory = new_directory
    # save configuration
    config.save()
    Configuration.config = config

    # open the input mesh as the starter
    starter = utils.open_mesh()
    # separate pieces
    if config.part_separation and starter.body_count > 1:
        starter = utils.separate_starter(starter)
    # run through the process
    run(starter)
