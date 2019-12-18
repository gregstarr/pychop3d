from pychop3d import utils
from pychop3d import constants
from pychop3d import bsp
from pychop3d import search
from pychop3d import connector

config_fn = "C:\\Users\\Greg\\code\\pychop3d\\debug\\0.json"

tree, config = bsp.BSPTree.from_json(config_fn)
tree = search.beam_search(tree, config)
tree.save("final_tree.json", config)
tree.export_stl(config)

connector_placer = connector.ConnectorPlacer(tree)
connector_placer.simulated_annealing_connector_placement()

tree = insert_connectors(best_tree, state)
tree.export_stl(config)