import numpy as np
import trimesh
import os

from pychop3d import utils
from pychop3d import bsp
from pychop3d.objective_functions import objectives
from pychop3d.configuration import Configuration


def evaluate_cuts(base_tree, node):
    config = Configuration.config
    N = config.normals
    Np = node.auxiliary_normals()
    N = utils.get_unique_normals(np.concatenate((N, Np), axis=0))
    trees = []
    for i in range(N.shape[0]):
        trees_of_this_normal = []
        normal = N[i]
        print(i, normal, end='')
        for plane in node.get_planes(normal):
            tree = base_tree.expand_node(plane, node)
            if tree:
                trees_of_this_normal.append(tree)
        for evaluate_objective_func in objectives.values():
            evaluate_objective_func(trees_of_this_normal, node.path)
        trees += trees_of_this_normal
        print()

    result_set = []
    for tree in sorted(trees, key=lambda x: x.get_objective()):
        if tree.sufficiently_different(node, result_set):
            result_set.append(tree)
    print(f"{len(result_set)} valid trees")
    return result_set


def beam_search(starter):
    config = Configuration.config
    if isinstance(starter, trimesh.Trimesh):
        current_trees = [bsp.BSPTree(starter)]
    elif isinstance(starter, bsp.BSPTree):
        current_trees = [starter]
    else:
        raise NotImplementedError

    n_leaves = 1
    while not utils.all_at_goal(current_trees):
        new_bsps = []
        for tree in utils.not_at_goal_set(current_trees):
            if len(tree.get_leaves()) != n_leaves:
                continue
            current_trees.remove(tree)
            largest_node = tree.largest_part()
            new_bsps += evaluate_cuts(tree, largest_node)

        n_leaves += 1
        current_trees += new_bsps
        current_trees = sorted(current_trees, key=lambda x: x.get_objective())
        if config.part_separation:
            extra_leaves_trees = [t for t in current_trees if len(t.get_leaves()) > n_leaves]
        current_trees = current_trees[:config.beam_width]
        if config.part_separation:
            current_trees += [t for t in extra_leaves_trees if t not in current_trees]

        if len(current_trees) == 0:
            raise Exception("Pychop3D failed")

        print(f"Leaves: {n_leaves}, best objective: {current_trees[0].get_objective()}, estimated number of parts: "
              f"{sum([p.n_parts for p in current_trees[0].get_leaves()])}")

        for i, tree in enumerate(current_trees[:config.beam_width]):
            tree.save(f"{i}.json")

        current_trees[0].export_stl()

    return current_trees[0]
