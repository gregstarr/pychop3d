import numpy as np
import trimesh

from pychop3d import utils
from pychop3d import bsp
from pychop3d import constants


def evaluate_cuts(base_tree, node):
    N = constants.UNIFORM_NORMALS
    Np = node.auxiliary_normals()
    N = np.concatenate((N, Np), axis=0)
    N = utils.get_unique_normals(N)
    trees = []
    for i in range(N.shape[0]):
        normal = N[i]
        print(i, normal, end='')
        for plane in node.get_planes(normal):
            tree2 = base_tree.expand_node(plane, node)
            if tree2:
                trees.append(tree2)
        print()

    result_set = []
    for tree in sorted(trees, key=lambda x: x.get_objective()):
        if tree.sufficiently_different(node, result_set):
            result_set.append(tree)
    print(f"{len(result_set)} valid trees")
    return result_set


def beam_search(starter, config):
    if isinstance(starter, trimesh.Trimesh):
        current_trees = [bsp.BSPTree(starter)]
    elif isinstance(starter, bsp.BSPTree):
        current_trees = [starter]
    else:
        raise NotImplementedError

    splits = 1
    while not utils.all_at_goal(current_trees):
        new_bsps = []
        for tree in utils.not_at_goal_set(current_trees):
            current_trees.remove(tree)
            largest_node = tree.largest_part()
            new_bsps += evaluate_cuts(tree, largest_node)

        current_trees += new_bsps
        current_trees = sorted(current_trees, key=lambda x: x.objective)[:config['beam_width']]

        print(f"Splits: {splits}, best objective: {current_trees[0].objective}, estimated number of parts: "
              f"{current_trees[0].largest_part().number_of_parts_estimate()}")

        for i, tree in enumerate(current_trees):
            tree.save(f"{i}.json", config)

        splits += 1
    return current_trees[0]
