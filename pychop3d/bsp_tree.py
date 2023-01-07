import numpy as np
import copy
from trimesh import Trimesh
import typing

from pychop3d.configuration import Configuration
from pychop3d import bsp_node
from pychop3d.logger import logger


class BSPTree:
    nodes: typing.List[bsp_node.BSPNode]
    objectives: typing.Dict[str, float]

    def __init__(self, part: Trimesh):
        """start a new BSPTree from a single part / object

        :param part: original part / object to split
        :type part: `trimesh.Trimesh`
        """
        config = Configuration.config  # collect configuration
        self.nodes = [bsp_node.BSPNode(part)]  # create root node and the list of nodes
        # calculate initial nparts objective
        nparts = 1  # nparts = sum([l.n_parts for l in self.leaves]) / nparts_original --> measures part reduction
        # calculate initial utilization objective --> measures how much of the parts fill their oriented bounding boxes
        V = np.prod(config.printer_extents)
        if config.obb_utilization:
            utilization = 1 - self.nodes[0].obb.volume / (self.nodes[0].n_parts * V)
        else:
            utilization = 1 - self.nodes[0].part.volume / (self.nodes[0].n_parts * V)

        # create objectives dictionary
        self.objectives = {
            'nparts': nparts,
            'utilization': utilization,
            'connector': 0,  # no connectors yet
            'fragility': 0,
            'seam': 0,
            'symmetry': 0
        }

    def copy(self):
        """copy function in case I ever want to do something more complicated or specific than deepcopy

        :return: copy of this tree
        :rtype: `BSPTree`
        """
        new_tree = copy.deepcopy(self)
        return new_tree

    def get_node(self, path=None):
        """get node with path `path`, if no `path` is provided, get the root node. Paths are specified as follows:
            - empty tuple ( ) is the root node
            - every number in the tuple specified which of the nodes at each level of the tree to select, for example
                (0, 1) means the 1st child of the 0th child of the root node

        :param path: tuple specifying the path to the node.
        :type path: tuple
        :return: the node at path `path`
        :rtype: `bsp_node.BSPNode`
        """
        node = self.nodes[0]  # get root node
        if not path:  # if no path specified, return the root node
            return node
        else:
            for i in path:  # descend the tree according to the path
                node = node.children[i]  # take the ith child at each tree level where i is in `path`
        return node

    @property
    def leaves(self) -> typing.List[bsp_node.BSPNode]:
        """property containing the leaves of the tree. The leaves of the final BSP tree correspond to parts
        small enough to fit in the printer

        :return: list of the leaves of this tree
        :rtype: list of `bsp_node.BSPNode`
        """
        nodes = [self.nodes[0]]  # collect root node, start list of nodes to check for leaves
        leaves = []  # start list of leaves
        while nodes:
            node = nodes.pop()  # take node out of list to check
            if len(node.children) == 0:  # check if the node has no children (this means it is a leaf)
                leaves.append(node)  # if so, add to list of leaves
            else:
                nodes += node.children  # otherwise add children to list of nodes to check
        return leaves

    @property
    def terminated(self):
        """property indicating whether all of this tree's nodes are small enough to be printed

        :return: terminated
        :rtype: bool
        """
        # if any of the leaves are not terminated, the tree is not terminated
        for leaf in self.leaves:
            if not leaf.terminated:
                return False
        return True

    @property
    def largest_part(self):
        """property pointing to this trees largest part by number of parts
        """
        # sort leaves by n_parts, give the last (highest) one
        return sorted(self.leaves, key=lambda x: x.n_parts)[-1]

    def different_from(self, tree, node):
        """determine if a node in this tree is different enough from a node in a different tree. The two trees will
        be the same tree except one of the nodes will have a different cutting plane. This function checks if that
        cutting plane is different from the corresponding cutting plane on the other tree. Two cutting planes are
        considered different if their relative translation or rotation passes the corresponding thresholds.

        :param tree: other tree to consider
        :type tree: `bsp_tree.BSPTree`
        :param node: which node on this tree to compare with the corresponding node on the other tree
        :type node: `bsp_node.BSPNode`
        :return: boolean indicating if the specified node is different between the trees
        :rtype: bool
        """
        self_node = self.get_node(node.path)  # get the node on this tree
        other_node = tree.get_node(node.path)  # get the corresponding node on the other tree
        # check if the node on this tree is different from the node on the other tree
        return self_node.different_from(other_node)

    def sufficiently_different(self, node, tree_set):
        """same as `bsp_tree.BSPTree.different_from` except this tree is compared to a list of other trees instead of
        just one.

        :param node: which node on this tree to compare with the corresponding node on the other trees
        :type node: `bsp_node.BSPNode`
        :param tree_set: list of other trees to consider
        :type tree_set: list of `bsp_tree.BSPTree`
        :return: boolean indicating if the specified node is different between this tree and all the other trees
        :rtype: bool
        """
        if not tree_set:  # if the tree set is empty, then this tree is unique
            return True
        for tree in tree_set:
            if not self.different_from(tree, node):  # go through the tree set and call `different_from` on each one
                self.different_from(tree, node)
                return False
        return True

    @property
    def objective(self):
        """calculates the weighted sum of the objective function components

        :return: value of the objective function for this tree
        :rtype: float
        """
        config = Configuration.config
        part = config.objective_weights['part'] * self.objectives['nparts']
        util = config.objective_weights['utilization'] * self.objectives['utilization']
        connector = config.objective_weights['connector'] * self.objectives['connector']
        fragility = config.objective_weights['fragility'] * self.objectives['fragility']
        seam = config.objective_weights['seam'] * self.objectives['seam']
        symmetry = config.objective_weights['symmetry'] * self.objectives['symmetry']
        return part + util + connector + fragility + seam + symmetry


def expand_node(tree, path, plane):
    """Splits a `tree` at the node given by `path` using `plane`. Returns a copy of the original tree but with the
    split node

    :param tree: tree to split
    :type tree: `bsp_tree.BSPTree`
    :param path: path pointing to the node to split
    :type path: tuple
    :param plane: splitting plane (origin, normal)
    :type plane: tuple of (3, ) shape `numpy.ndarray`
    :return: (split tree, result)
    :rtype: (`bsp_tree.BSPTree`, str)
    """
    new_tree = tree.copy()  # copy input tree
    new_node = new_tree.get_node(path)  # collect the node of the copied tree according to `path`
    new_node, result = bsp_node.split(new_node, plane)  # attempt to split the node using `plane`
    if result != 'success':  # if not successful, `new_node` may be None
        return None, result
    new_tree.nodes += new_node.children  # add `new_node`'s children to the `new_tree`'s list of nodes
    return new_tree, result


def get_planes(part, normal):
    """get all planes in the form of (origin, normal) pairs corresponding to valid cuts of the input part. Planes are
    in the direction specified by `normal` and are spaced according to the `plane_spacing` configuration parameter.

    :param part: object to determine valid cutting planes for
    :type part: `trimesh.Trimesh`
    :param normal: unit vector defining the normal vector for the planes
    :type normal: (3, ) shape `numpy.ndarray`
    :return: list of all valid cutting planes for the input object
    :rtype: list
    """
    config = Configuration.config  # collect configuration
    projection = part.vertices @ normal  # project all vertices of the input object onto the input normal vector
    # determine the extent of the object in the direction defined by the normal vector
    limits = [projection.min(), projection.max()]
    # create planes spaced out according to the configuration
    planes = [(d * normal, normal) for d in np.arange(limits[0], limits[1], config.plane_spacing)][1:]
    if config.add_middle_plane:  # add the middle plane
        planes += [(normal * (projection.min() + projection.max()) / 2, normal)]  # add a plane through the middle
    return planes
