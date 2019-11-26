import numpy as np

from pychop3d import utils
from pychop3d import bsp


def evaluate_cuts(base_tree, node):
    N = utils.uniform_normals()
    Np = node.auxiliary_normals()
    N = np.concatenate((N, Np), axis=0)
    trees = []
    for i in range(N.shape[0]):
        print(i)
        normal = N[i]
        for plane in node.get_planes(normal):
            tree2 = base_tree.expand_node(plane, node)
            trees.append(tree2)

    result_set = []
    for tree in sorted(trees, key=lambda x: x.get_objective()):
        if tree.sufficiently_different(node, result_set):
            result_set.append(tree)

    return result_set


def beam_search(obj, b=2):
    current_trees = [bsp.BSPTree(obj)]
    while not utils.all_at_goal(current_trees):
        new_bsps = []
        for tree in utils.not_at_goal_set(current_trees):
            current_trees.remove(tree)
            largest_node = tree.largest_part()
            new_bsps += evaluate_cuts(tree, largest_node)
        current_trees = sorted(new_bsps, key=lambda x: x.get_objective())[:b]
    return current_trees[0]
