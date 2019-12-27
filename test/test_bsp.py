import pytest
import trimesh
import numpy as np
import copy

from pychop3d import bsp
from pychop3d.configuration import Configuration
from pychop3d import section


def test_different_from():
    config = Configuration.config
    print()
    mesh = trimesh.load(config.mesh, validate=True)

    tree = bsp.BSPTree(mesh)
    root = tree.nodes[0]
    extents = root.part.bounding_box_oriented.primitive.extents
    normal = trimesh.unitize(np.random.rand(3))
    planes = root.get_planes(normal)
    plane = planes[np.random.randint(0, len(planes))]
    base_node = copy.deepcopy(root)
    base_node.split(plane)

    # smaller origin offset, should not be different
    test_node = copy.deepcopy(root)
    origin = plane[0] + trimesh.unitize(np.random.rand(3)) * .095 * np.sqrt(np.sum(config.printer_extents ** 2))
    test_plane = (origin, plane[1])
    test_node.split(test_plane)
    assert not base_node.different_from(test_node)

    # larger origin offset, should be different
    test_node = copy.deepcopy(root)
    origin = plane[0] + trimesh.unitize(np.random.rand(3)) * .15 * np.sqrt(np.sum(config.printer_extents ** 2))
    test_plane = (origin, plane[1])
    test_node.split(test_plane)
    assert base_node.different_from(test_node)

    # smaller angle difference, should not be different
    test_node = copy.deepcopy(root)
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, plane[1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 11, axis)
    normal = trimesh.transform_points(plane[1][None, :], rotation)[0]
    test_plane = (plane[0], normal)
    test_node.split(test_plane)
    assert not base_node.different_from(test_node)

    # larger angle difference, should be different
    test_node = copy.deepcopy(root)
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, plane[1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 9, axis)
    normal = trimesh.transform_points(plane[1][None, :], rotation)[0]
    test_plane = (plane[0], normal)
    test_node.split(test_plane)
    assert base_node.different_from(test_node)


def test_copy_tree():
    config = Configuration.config
    mesh = trimesh.load(config.mesh, validate=True)

    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = bsp.BSPTree(mesh)
    node = tree.largest_part()
    normal = np.array([0, 0, 1])
    planes = node.get_planes(normal)
    plane = planes[len(planes) // 2]
    tree = tree.expand_node(plane, node)
    print("tree objective: ", tree.objective)
    assert tree._objective is not None
    new_tree = tree.copy()
    assert new_tree._objective is None


def test_expand_node():
    config = Configuration.config
    mesh = trimesh.load(config.mesh, validate=True)

    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = bsp.BSPTree(mesh)

    node = tree.largest_part()
    normal = np.array([0, 0, 1])
    planes = node.get_planes(normal)
    plane = planes[len(planes) // 2]
    tree1 = tree.expand_node(plane, node)
    print("tree objective: ", tree1.objective)

    node = tree1.largest_part()
    planes = node.get_planes(normal)
    plane = planes[len(planes) // 2]
    tree2 = tree1.expand_node(plane, node)
    assert tree2._objective is None


def test_grid_sample():
    config = Configuration.config
    origin, normal = (np.zeros(3), np.array([0, 0, 1]))

    cd = config.connector_diameter
    mesh = trimesh.primitives.Box(extents=[cd / 1.9, cd / 1.9, 40])
    positive, negative, cross_section = section.bidirectional_split(mesh, origin, normal)
    polygon = cross_section.path2d.polygons_full[0]
    cc = section.ConnectedComponent(cross_section, polygon, positive, negative)
    assert cc.valid

    mesh.apply_translation([3, 0, 0])
    positive, negative, cross_section = section.bidirectional_split(mesh, origin, normal)
    polygon = cross_section.path2d.polygons_full[0]
    cc = section.ConnectedComponent(cross_section, polygon, positive, negative)
    assert cc.valid

    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/4, np.array([0, 0, 1])))
    positive, negative, cross_section = section.bidirectional_split(mesh, origin, normal)
    polygon = cross_section.path2d.polygons_full[0]
    cc = section.ConnectedComponent(cross_section, polygon, positive, negative)
    assert cc.valid