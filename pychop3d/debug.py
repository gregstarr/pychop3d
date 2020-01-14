import json

from pychop3d import bsp
from pychop3d import run
from pychop3d import utils
from pychop3d import connector
from pychop3d.configuration import Configuration

Configuration.config = Configuration("C:\\Users\\Greg\\code\\pychop3d\\output\\20200112_003815\\config.yml")
Configuration.config.sa_initial_connector_ratio = .01
Configuration.config.connector_spacing = 12
mesh = utils.open_mesh()
tree = utils.open_tree("C:\\Users\\Greg\\code\\pychop3d\\output\\20200112_003815\\0.json")

connector_placer = connector.ConnectorPlacer(tree)
state = connector_placer.simulated_annealing_connector_placement()
tree = connector_placer.insert_connectors(tree, state)

tree.export_stl()
