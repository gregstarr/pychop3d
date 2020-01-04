import json

from pychop3d import bsp
from pychop3d import run
from pychop3d import utils
from pychop3d import connector
from pychop3d.configuration import Configuration

Configuration.config = Configuration("C:\\Users\\Greg\\code\\pychop3d\\output\\20200103_072659\\config.yml")
mesh = utils.open_mesh()
tree = utils.open_tree(mesh, "C:\\Users\\Greg\\code\\pychop3d\\output\\20200103_161335\\final_tree_with_connectors.json")

connector_placer = connector.ConnectorPlacer(tree)
state = connector_placer.simulated_annealing_connector_placement()
tree = connector_placer.insert_connectors(tree, state)

tree.export_stl()