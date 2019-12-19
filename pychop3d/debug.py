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
state = connector_placer.simulated_annealing_connector_placement()
connector_placer.evaluate_connector_objective(state)
tree = connector_placer.insert_connectors(tree, state)
tree.export_stl(config)
