import time
import logging

from pychop3d import utils
from pychop3d import connector
from pychop3d.configuration import Configuration

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)s  [%(levelname)s]  %(message)s",
                    handlers=[logging.StreamHandler()])

Configuration.config = Configuration("C:\\Users\\Greg\\Documents\\things\\table\\config.yml")
tree_file = "C:\\Users\\Greg\\Documents\\things\\table\\table_platform_20200503_040048\\final_tree_with_connectors.json"
tree = utils.open_tree(tree_file)
connector_placer = connector.ConnectorPlacer(tree)
state = utils.load_connector_configuration(tree_file)
tree = connector_placer.insert_connectors(tree, state)

utils.export_tree_stls(tree)
