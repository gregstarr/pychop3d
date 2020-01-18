import time

from pychop3d import utils
from pychop3d import connector
from pychop3d.configuration import Configuration

Configuration.config = Configuration("C:\\Users\\Greg\\code\\pychop3d\\output\\success\\config.yml")
Configuration.config.sa_initial_connector_ratio = .01
Configuration.config.connector_spacing = 20
Configuration.config.sa_iterations = 50_000
mesh = utils.open_mesh()
tree = utils.open_tree("C:\\Users\\Greg\\code\\pychop3d\\output\\success\\0.json")

connector_placer = connector.ConnectorPlacer(tree)
t0 = time.time()
state = connector_placer.simulated_annealing_connector_placement()
print(f"TIME: {time.time() - t0}")
tree = connector_placer.insert_connectors(tree, state)

tree.export_stl()
