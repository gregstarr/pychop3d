import numpy as np

from pychop3d import utils
from pychop3d import bsp
from pychop3d.configuration import Configuration


config = Configuration.config


def evaluate_cuts(base_tree, node):
    N = config.normals
    Np = node.auxiliary_normals()
    N = utils.get_unique_normals(np.concatenate((N, Np), axis=0))
    trees = []
    for i in range(N.shape[0]):
        normal = N[i]
        for plane in node.get_planes(normal):
            print('.', end='')
            tree2 = base_tree.expand_node(plane, node)
            if tree2:
                trees.append(tree2)

    print()
    result_set = []
    for tree in sorted(trees, key=lambda x: x.get_objective()):
        if tree.sufficiently_different(node, result_set):
            result_set.append(tree)
    print(f"considering {len(result_set)} trees")
    return result_set


def beam_search(obj):
    current_trees = [bsp.BSPTree(obj)]
    splits = 1
    while not utils.all_at_goal(current_trees):
        new_bsps = []
        for tree in utils.not_at_goal_set(current_trees):
            current_trees.remove(tree)
            largest_node = tree.largest_part()
            new_bsps += evaluate_cuts(tree, largest_node)
        current_trees = sorted(new_bsps, key=lambda x: x.get_objective())[:config.beam_width]
        print(f"Splits: {splits}, best objective: {current_trees[0].get_objective()}, estimated number of parts: "
              f"{current_trees[0].largest_part().n_parts}")
        splits += 1
    return current_trees[0]
