import pytest
import trimesh
import numpy as np

from pychop3d import bsp
from pychop3d import constants
from pychop3d import utils


@pytest.mark.parametrize('i', np.arange(100))
def test_expand_node(i):
    print()
    fn = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
    mesh = trimesh.load(fn, validate=True)
    mesh.apply_scale(3)
    tree = bsp.BSPTree(mesh)
    for i in range(10):
        if tree is None:
            print("SKIPPING")
            return
        node = tree.largest_part()
        extents = node.part.bounding_box_oriented.primitive.extents
        normal = trimesh.unitize(np.random.rand(3))
        planes = node.get_planes(normal, extents.min()/10)
        plane = planes[np.random.randint(0, len(planes))]
        tree = tree.expand_node(plane, node)


def test_different_from():
    pass
