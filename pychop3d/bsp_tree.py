"""BSPTree class and related functions"""
import copy
from typing import Dict
import json
from pathlib import Path

import numpy as np
from trimesh import Trimesh, load

from pychop3d import objective_functions, settings, utils
from pychop3d.bsp_node import BSPNode, Plane, split
from pychop3d.logger import logger


class BSPTree:
    """Represents object and partitions"""

    nodes: list[BSPNode]
    objectives: Dict[str, float]
    printer_extents: np.ndarray

    def __init__(self, part: Trimesh, printer_extents: np.ndarray):
        """BSPTree Initialization

        Args:
            part (Trimesh): original object to split
            printer_extents (np.ndarray): printer dimensions (mm)
            obb_utilization (False): use oriented bounding box for part volume instead
                of mesh volume
        """
        # create root node and the list of nodes
        self.nodes = [BSPNode(part, None, printer_extents)]
        self.printer_extents = printer_extents
        # calculate initial nparts objective
        # nparts = sum([l.n_parts for l in self.leaves]) / nparts_original -->
        # measures part reduction
        nparts = 1
        # calculate initial utilization objective -->
        # measures how much of the parts fill their oriented bounding boxes
        vol = np.prod(printer_extents)
        if settings.OBB_UTILIZATION:
            utilization = 1 - self.nodes[0].obb.volume / (self.nodes[0].n_parts * vol)
        else:
            utilization = 1 - self.nodes[0].part.volume / (self.nodes[0].n_parts * vol)

        # create objectives dictionary
        self.objectives = {
            "nparts": nparts,
            "utilization": utilization,
            "connector": 0,  # no connectors yet
            "fragility": 0,
            "seam": 0,
            "symmetry": 0,
        }

    def copy(self) -> "BSPTree":
        """copy function in case I ever want to do something more complicated or
        specific than deepcopy

        Returns:
            BSPTree: copy
        """
        new_tree = copy.deepcopy(self)
        return new_tree

    def get_node(self, path: tuple = None) -> BSPNode:
        """get node with path `path`, if no `path` is provided, get the root node.

        Paths are specified as follows:
            - empty tuple ( ) is the root node
            - every number in the tuple specified which of the nodes at each level of
                the tree to  select, for example (0, 1) means the 1st child of the 0th
                child of the root node

        Args:
            path (tuple, optional): path to node. Defaults to None which returns root
                node.

        Returns:
            BSPNode
        """
        node = self.nodes[0]  # get root node
        if path is None:  # if no path specified, return the root node
            return node
        for i in path:  # descend the tree according to the path
            # take the ith child at each tree level where i is in `path`
            node = node.children[i]
        return node

    @property
    def leaves(self) -> list[BSPNode]:
        """leaves of the tree. The leaves of the final BSP tree correspond to parts
        small enough to fit in the printer

        Returns:
            list[BSPNode]: list of the leaves of this tree
        """
        # collect root node, start list of nodes to check for leaves
        nodes = [self.nodes[0]]
        leaves = []  # start list of leaves
        while nodes:
            node = nodes.pop()  # take node out of list to check
            # check if the node has no children (this means it is a leaf)
            if len(node.children) == 0:
                leaves.append(node)  # if so, add to list of leaves
            else:
                # otherwise add children to list of nodes to check
                nodes += node.children
        return leaves

    @property
    def terminated(self) -> bool:
        """all of this tree's nodes are small enough to be printed

        Returns:
            bool
        """
        # if any of the leaves are not terminated, the tree is not terminated
        for leaf in self.leaves:
            if not leaf.terminated:
                return False
        return True

    @property
    def largest_part(self) -> BSPNode:
        """tree's largest part by number of parts

        Returns:
            BSPNode
        """
        # sort leaves by n_parts, give the last (highest) one
        return sorted(self.leaves, key=lambda x: x.n_parts)[-1]

    def different_from(self, tree: "BSPTree", node: BSPNode) -> bool:
        """determines if a node in this tree is different enough from a node in a
        different tree.

        The two trees will be the same tree except one of the nodes will have a
        different cutting plane. This function checks if that cutting plane is
        different from the corresponding  cutting plane on the other tree. Two cutting
        planes are considered different if their  relative translation or rotation
        passes the corresponding thresholds.

        Args:
            tree (BSPTree): other tree to compare to
            node (BSPNode): which node on this tree to compare with the corresponding
                node on the other tree

        Returns:
            bool: indicates if the specified node is different between the trees
        """
        self_node = self.get_node(node.path)  # get the node on this tree
        # get the corresponding node on the other tree
        other_node = tree.get_node(node.path)
        # check if the node on this tree is different from the node on the other tree
        return self_node.different_from(other_node)

    def sufficiently_different(self, node: BSPNode, tree_set: list["BSPTree"]) -> bool:
        """same as `BSPTree.different_from` except this tree is compared to a list of
        other trees instead of just one.

        Args:
            node (BSPNode): which node on this tree to compare with the corresponding
                node on the other trees
            tree_set (list[BSPTree])

        Returns:
            bool: indicates if the specified node is different between this tree and
                all the other trees
        """
        if not tree_set:  # if the tree set is empty, then this tree is unique
            return True
        for tree in tree_set:
            # go through the tree set and call `different_from` on each one
            if not self.different_from(tree, node):
                self.different_from(tree, node)
                return False
        return True

    @property
    def objective(self) -> float:
        """calculates the weighted sum of the objective function components

        Returns:
            float: objective value
        """
        weights = settings.OBJECTIVE_WEIGHTS
        part = weights["part"] * self.objectives["nparts"]
        util = weights["utilization"] * self.objectives["utilization"]
        connector = weights["connector"] * self.objectives["connector"]
        fragility = weights["fragility"] * self.objectives["fragility"]
        seam = weights["seam"] * self.objectives["seam"]
        symmetry = weights["symmetry"] * self.objectives["symmetry"]
        return part + util + connector + fragility + seam + symmetry


def expand_node(tree: BSPTree, path: tuple, plane: Plane) -> tuple[BSPTree, str]:
    """Splits a tree at the node given by `path` using `plane`. Returns a copy of the
    original tree but with the split node

    Args:
        tree (BSPTree): tree to split
        path (tuple): path to the node to split
        plane (Plane): splitting plane

    Returns:
        BSPTree: tree with split node
        str: result string
    """
    new_tree = tree.copy()  # copy input tree
    # collect the node of the copied tree according to `path`
    new_node = new_tree.get_node(path)
    # attempt to split the node using `plane`
    new_node, result = split(new_node, plane)
    # if not successful, `new_node` may be None
    if result != "success":
        return None, result
    # add `new_node`'s children to the `new_tree`'s list of nodes
    new_tree.nodes += new_node.children
    return new_tree, result


def get_planes(
    part: Trimesh,
    normal: np.ndarray,
) -> list[Plane]:
    """get all planes corresponding to valid cuts of the input part. Planes are in the
    direction specified by `normal` and are spaced according to the `plane_spacing`
    configuration parameter.

    Args:
        part (Trimesh): object to determine valid cutting planes for
        normal (np.ndarray): unit vector defining the normal vector for the planes
        plane_spacing (float): spacing between planes (mm)
        add_middle_plane (bool, optional): Defaults to True.

    Returns:
        list[Plane]: list of all valid cutting planes for the input object
    """
    logger.info("getting planes")
    # project all vertices of the input object onto the input normal vector
    projection = part.vertices @ normal
    # determine the extent of the object in the direction defined by the normal vector
    limits = [projection.min(), projection.max()]
    # create planes spaced out according to the configuration
    delta = settings.PLANE_SPACING
    planes = [
        Plane(d * normal, normal)
        for d in np.arange(limits[0] + delta, limits[1], delta)
    ]
    if settings.ADD_MIDDLE_PLANE:  # add the middle plane
        planes.append(Plane(normal * (projection.min() + projection.max()) / 2, normal))
    return planes


def process_normal(
    normal: np.ndarray, node: BSPNode, base_tree: BSPTree
) -> list[BSPTree]:
    """evaluate objective functions on planes using given normal vector

    Args:
        normal (np.ndarray): normal vector
        node (BSPNode): node to split
        base_tree (BSPTree): tree to evaluate on

    Returns:
        list[BSPTree]: list of trees with objective functions evaluated
    """
    trees_of_this_normal = []  # start a list of trees for splits along this normal
    # iterate over all valid cutting planes for the node
    for plane in get_planes(node.part, normal):
        # split the node using the plane
        tree, result = expand_node(base_tree, node.path, plane)
        logger.info(result)
        if tree:  # only keep the tree if the split is successful
            trees_of_this_normal.append(tree)
    # avoid empty list errors during objective function evaluation
    if len(trees_of_this_normal) == 0:
        return trees_of_this_normal
    # go through each objective function, evaluate the objective function for each tree
    # in this normal's list, fill in the data in each tree object in the list
    objectives = objective_functions.objectives
    for evaluate_objective_func in objectives.values():
        evaluate_objective_func(trees_of_this_normal, node.path)
    return trees_of_this_normal


def separate_starter(mesh: Trimesh, printer_extents: np.ndarray) -> BSPTree:
    """turns a mesh with more than one body into a tree who's root node is the original
    mesh and the roots children are each of the bodies of the mesh

    Args:
        mesh (Trimesh): mesh with multiple bodies
        printer_extents (np.ndarray): printer dims (mm)

    Returns:
        BSPTree
    """
    logger.info("separating starter mesh")
    parts = mesh.split(only_watertight=False)  # split into separate components
    logger.info("starter mesh split into %s children", len(parts))
    tree = BSPTree(mesh, printer_extents)  # create starter tree
    for i, part in enumerate(parts):
        # make a new node for each separate part
        new_node = BSPNode(part, tree.nodes[0], num=i)
        # make the new node the root node's child
        tree.nodes[0].children.append(new_node)
        tree.nodes.append(new_node)
    # update nparts and utilization objectives
    objective_functions.evaluate_nparts_objective([tree], tuple())
    objective_functions.evaluate_utilization_objective([tree], tuple(), printer_extents)
    return tree


def open_tree(tree_file: Path, mesh_file: Path, printer_extents: np.ndarray) -> BSPTree:
    """loads tree

    Args:
        tree_file (Path): path to tree file json
        mesh_file (Path): path to mesh file stl
        printer_extents (np.ndarray): printer dims (mm)

    Returns:
        BSPTree
    """
    mesh = load(mesh_file)
    utils.trimesh_repair(mesh)
    with open(tree_file, encoding="utf8") as f:
        data = json.load(f)

    node_data = data["nodes"]
    tree = BSPTree(mesh, printer_extents)
    for n in node_data:
        plane = Plane(np.array(n["origin"]), np.array(n["normal"]))
        node = tree.get_node(n["path"])
        tree, result_code = expand_node(tree, node.path, plane)
        logger.info(result_code)
    return tree