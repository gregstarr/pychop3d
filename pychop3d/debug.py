from pychop3d import bsp
from pychop3d import run
from pychop3d.config import Configuration

cfg = Configuration("C:\\Users\\Greg\\code\\pychop3d\\debug\\final_tree.json")
Configuration.set_configuration(cfg)
run.run()
