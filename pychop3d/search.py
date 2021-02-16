import numpy as np
import trimesh
import logging
import multiprocessing

from pychop3d import utils
from pychop3d import bsp_tree
from pychop3d.process_normal import process_normal
from pychop3d.configuration import Configuration


logger = logging.getLogger(__name__)
PARALLEL = False


def evaluate_cuts(base_tree, node):
    """this function returns a list of unique trees by splitting a specified node of an input tree along all planes
    as defined in the configuration

    :param base_tree: tree to split at a particular node
    :type base_tree: `bsp_tree.BSPTree`
    :param node: node of the input tree to split
    :type node: `bsp_node.BSPNode`
    :return: list of 'unique' trees resulting from splitting the input tree at the specified node
    :rtype: list of `bsp_tree.BSPTree`
    """
    config = Configuration.config  # Collect configuration

    N = config.normals  # Collect predefined set of normal vectors
    N = np.append(N, node.auxiliary_normals, axis=0)  # Append partition's bounding-box-aligned vectors as normals
    N = np.unique(np.round(N, 3), axis=0)  # Return sorted unique elements of input array_like

    args = [(n, node, base_tree, config) for n in N]
    if PARALLEL:
        with multiprocessing.Pool(6) as p:
            pool_output = p.starmap(process_normal, args)
    else:
        pool_output = []
        for arg in args:
            pool_output.append(process_normal(*arg))
    print()
    trees = []
    for i in range(len(N)):
        trees += pool_output[i]
        logger.info(f"index {i}, normal {N[i]}, norm trees: {len(pool_output[i])}")
    logger.info(f"total trees: {len(trees)}")
    # go through the list of trees, best ones first, and throw away any that are too similar to another tree already
    # in the result list
    result_set = []
    for tree in sorted(trees, key=lambda x: x.objective):
        if tree.sufficiently_different(node, result_set):
            result_set.append(tree)
    logger.info(f"{len(result_set)} valid trees")
    return result_set


def beam_search(starter):
    """This function executes the beam search to find a good BSPTree partitioning of the input object

    :param starter: Either an unpartitioned mesh or an already partitioned tree to begin the process using
    :type starter: `trimesh.Trimesh`
    :type starter: `bsp_tree.BSPTree`
    :return: a BSPTree which adequately partitions the input object
    :rtype: `bsp_tree.BSPTree`
    """
    config = Configuration.config  # collect configuration
    # open up starter, this can either be a trimesh or an already partitioned object as a tree
    if isinstance(starter, trimesh.Trimesh):
        current_trees = [bsp_tree.BSPTree(starter)]
    elif isinstance(starter, bsp_tree.BSPTree):
        current_trees = [starter]
    else:
        raise NotImplementedError

    logger.info(f"Starting beam search with an instance of {type(starter)}")
    if isinstance(starter, trimesh.Trimesh):
        logger.info("Trimesh stats:")
        logger.info(f"verts: {starter.vertices.shape[0]} extents: {starter.extents}")
    else:
        logger.info(f"n_leaves: {len(starter.leaves)}")
        logger.info(f"Largest part trimesh stats:")
        logger.info(f"verts: {starter.largest_part.part.vertices.shape[0]}, extents: {starter.largest_part.part.extents}")

    if utils.all_at_goal(current_trees):
        raise Exception("Input mesh already small enough to fit in printer")

    # keep track of n_leaves, in each iteration we will only consider trees with the same number of leaves
    # I think the trees become less comparable when they don't have the same number of leaves
    n_leaves = 1
    while not utils.all_at_goal(current_trees):  # continue until we have at least {beam_width} trees
        new_bsps = []  # list of new bsps
        for tree in utils.not_at_goal_set(current_trees):  # look at all trees that haven't terminated
            if len(tree.leaves) != n_leaves:  # only consider trees with a certain number of leaves
                continue
            current_trees.remove(tree)  # remove the current tree (we will replace it with its best partition)
            largest_node = tree.largest_part  # split the largest node
            new_bsps += evaluate_cuts(tree, largest_node)  # consider many different cutting planes for the node

        n_leaves += 1  # on the next iteration, look at trees with more leaves
        current_trees += new_bsps
        current_trees = sorted(current_trees, key=lambda x: x.objective)  # sort all of the trees including the new ones
        # if we are considering part separation, some of the trees may have more leaves, put those away for later
        if config.part_separation:
            extra_leaves_trees = [t for t in current_trees if len(t.leaves) > n_leaves]
        current_trees = current_trees[:config.beam_width]  # only keep the best {beam_width} trees
        if config.part_separation:  # add back in the trees with extra leaves
            current_trees += [t for t in extra_leaves_trees if t not in current_trees]

        if len(current_trees) == 0:  # all of the trees failed
            raise Exception("No valid chops found")

        logger.info(f"Leaves: {n_leaves}, best objective: {current_trees[0].objective}, estimated number of parts: "
                    f"{sum([p.n_parts for p in current_trees[0].leaves])}")

        # save progress
        for i, tree in enumerate(current_trees[:config.beam_width]):
            utils.save_tree(tree, f"{config.name}_{i}.json")
        utils.export_tree_stls(current_trees[0])

    return current_trees[0]
