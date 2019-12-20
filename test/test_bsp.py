import pytest
import trimesh
import numpy as np
import copy

from pychop3d import bsp
from pychop3d.configuration import Configuration
from pychop3d import bsp_mesh


config = Configuration.config


def test_expand_node():
    print()
    # open and prepare mesh
    mesh = trimesh.load(config.mesh, validate=True)
    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh, chull)

    tree = bsp.BSPTree(mesh)
    node = tree.nodes[0]
    extents = node.part.bounding_box_oriented.primitive.extents
    normal = trimesh.unitize(np.random.rand(3))
    planes = node.get_planes(normal, extents.min()/10)
    plane = planes[np.random.randint(0, len(planes))]
    tree = tree.expand_node(plane, node)


def test_different_from():
    print()
    # open and prepare mesh
    mesh = trimesh.load(config.mesh, validate=True)
    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh, chull)

    tree = bsp.BSPTree(mesh)
    root = tree.nodes[0]
    extents = root.part.bounding_box_oriented.primitive.extents
    normal = trimesh.unitize(np.random.rand(3))
    planes = root.get_planes(normal, extents.min() / 10)
    plane = planes[np.random.randint(0, len(planes))]
    base_node = root.copy()
    base_node.split(plane)

    # smaller origin offset, should not be different
    test_node = root.copy()
    origin = plane[0] + trimesh.unitize(np.random.rand(3)) * .095 * np.sqrt(np.sum(config.printer_extents ** 2))
    test_plane = (origin, plane[1])
    test_node.split(test_plane)
    assert not base_node.different_from(test_node)

    # larger origin offset, should be different
    test_node = root.copy()
    origin = plane[0] + trimesh.unitize(np.random.rand(3)) * .15 * np.sqrt(np.sum(config.printer_extents ** 2))
    test_plane = (origin, plane[1])
    test_node.split(test_plane)
    assert base_node.different_from(test_node)

    # smaller angle difference, should not be different
    test_node = root.copy()
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, plane[1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 11, axis)
    normal = trimesh.transform_points(plane[1][None, :], rotation)[0]
    test_plane = (plane[0], normal)
    test_node.split(test_plane)
    assert not base_node.different_from(test_node)

    # larger angle difference, should be different
    test_node = root.copy()
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, plane[1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 9, axis)
    normal = trimesh.transform_points(plane[1][None, :], rotation)[0]
    test_plane = (plane[0], normal)
    test_node.split(test_plane)
    assert base_node.different_from(test_node)


def test_copy():
    # open and prepare mesh
    mesh = trimesh.load(config.mesh, validate=True)
    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh, chull)

    mesh = copy.deepcopy(mesh)
    for i in range(10000):
        mesh = copy.deepcopy(mesh)

    tree = bsp.BSPTree(mesh)
    node = tree.nodes[0]
    for i in range(10000):
        node = copy.deepcopy(node)

    for i in range(10000):
        tree = copy.deepcopy(tree)


def test_expand_node():
    # open and prepare mesh
    mesh = trimesh.load(config.mesh, validate=True)
    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh, chull)
    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = bsp.BSPTree(mesh)
    node = tree.largest_part()
    extents = node.part.bounding_box_oriented.primitive.extents
    normal = np.array([1, 0, 0])
    planes = node.get_planes(normal, extents.min() / 10)
    plane = planes[len(planes) // 2]
    tree = tree.expand_node(plane, node)
