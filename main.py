'''
pychop3d - cli model chop utility

# INSTRUCTIONS:

1. Install dependencies from `requirements.txt`.
2. Run this script from the root of the pychop3d repository in the form.
```
   $ python main.py ./path/to/model.stl
```
3. Wait for chopping to finish. Chopped models will appear in the current working directory.
'''

import sys
import time

from pychop3d.search import beam_search
from pychop3d import connector
from pychop3d.configuration import Configuration
from pychop3d import utils

def main(mesh_filepath):

    # set configuration options
    config = Configuration.config
    config.name = 'output'
    config.mesh = mesh_filepath
    config.beam_width = 3
    config.connector_diameter = 6
    config.connector_spacing = 10
    config.part_separation = True
    config.scale_factor = 5
    config.save()

    # open the input mesh as the starter
    starter = utils.open_mesh()

    # separate pieces
    if config.part_separation and starter.body_count > 1:
        starter = utils.separate_starter(starter)

    # complete the beam search using the starter, no search will take place if the starter tree is adequately partitioned
    tree = beam_search(starter)
    # save the tree now in case the connector placement fails
    utils.save_tree(tree, "final_tree.json")

    try:
        # mark starting time
        t0 = time.time()
        # create connector placer object, this creates all potential connectors and determines their collisions
        connector_placer = connector.ConnectorPlacer(tree)
        if connector_placer.n_connectors > 0:
            # use simulated annealing to determine the best combination of connectors
            state = connector_placer.simulated_annealing_connector_placement()
            # save the final tree including the state
            utils.save_tree(tree, "final_tree_with_connectors.json", state)
            # add the connectors / subtract the slots from the parts of the partitioned input object
            tree = connector_placer.insert_connectors(tree, state)
    except Exception as e:
        # fail gently so that the STLs still get exported
        warnings.warn(e, Warning, stacklevel=2)

    # export the parts of the partitioned object
    utils.export_tree_stls(tree)

if __name__ == "__main__":
    # Read mesh filepath from argument
    mesh_filepath = sys.argv[1]
    # Mark start time
    t0 = time.time()
    # Chop model
    main(mesh_filepath)   
    
    print(f'Operation finished in {time.time() - t0}')
