from pychop3d import bsp_tree
from pychop3d.objective_functions import objectives
from pychop3d.configuration import Configuration
from pychop3d.logger import logger


def process_normal(normal, node, base_tree, config):
    Configuration.config = config
    trees_of_this_normal = []  # start a list of trees for splits along this normal
    for plane in bsp_tree.get_planes(node.part, normal):  # iterate over all valid cutting planes for the node
        tree, result = bsp_tree.expand_node(base_tree, node.path, plane)  # split the node using the plane
        if tree:  # only keep the tree if the split is successful
            trees_of_this_normal.append(tree)
    if len(trees_of_this_normal) == 0:  # avoid empty list errors during objective function evaluation
        return trees_of_this_normal
    # go through each objective function, evaluate the objective function for each tree in this normal's
    # list, fill in the data in each tree object in the list
    for evaluate_objective_func in objectives.values():
        evaluate_objective_func(trees_of_this_normal, node.path)
    return trees_of_this_normal
