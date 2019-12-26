import pytest
import trimesh
import numpy as np

from pychop3d import bsp
from pychop3d.configuration import Configuration
from pychop3d import bsp_mesh


config = Configuration.config


def test_expand_node():
    # open and prepare mesh
    mesh = trimesh.load(config.mesh, validate=True)
    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh)
    mesh._convex_hull = chull
    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = bsp.BSPTree(mesh)
    node = tree.largest_part()
    extents = node.part.bounding_box_oriented.primitive.extents
    normal = np.array([1, 0, 0])
    planes = node.get_planes(normal)
    plane = planes[len(planes) // 2]
    tree = tree.expand_node(plane, node)


def test_different_from():
    pass
