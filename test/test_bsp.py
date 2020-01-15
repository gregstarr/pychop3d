import pytest
import trimesh
import numpy as np
import copy
import os

from pychop3d import bsp
from pychop3d.configuration import Configuration
from pychop3d import section


def test_get_planes():
    """verify that for the default bunny mesh, which is a single part, all planes returned by `BSPNode.get_planes`
        cut through the mesh (they have a good cross section)
    """
    config = Configuration.config
    mesh = trimesh.load(config.mesh, validate=True)

    for i in range(100):
        tree = bsp.BSPTree(mesh)
        root = tree.nodes[0]
        normal = trimesh.unitize(np.random.rand(3))
        planes = root.get_planes(normal)

        for origin, normal in planes:
            path3d = mesh.section(plane_origin=origin, plane_normal=normal)
            assert path3d is not None


def test_different_from():
    """verify that `BSPNode.different_from` has the expected behavior
    """
    config = Configuration.config
    print()
    mesh = trimesh.load(config.mesh, validate=True)

    tree = bsp.BSPTree(mesh)
    root = tree.nodes[0]
    normal = trimesh.unitize(np.random.rand(3))
    planes = root.get_planes(normal)
    base_node = copy.deepcopy(root)
    base_node.split(planes[0])

    for plane in planes[1:]:
        # smaller origin offset, should not be different
        test_node = copy.deepcopy(root)
        test_node.split(plane)
        if abs((plane[0] - planes[0][0]) @ planes[0][1]) > config.different_origin_th:
            assert base_node.different_from(test_node)
        else:
            assert not base_node.different_from(test_node)

    # smaller angle difference, should not be different
    test_node = copy.deepcopy(root)
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, planes[0][1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 11, axis)
    normal = trimesh.transform_points(planes[0][1][None, :], rotation)[0]
    test_plane = (planes[0][0], normal)
    test_node.split(test_plane)
    assert not base_node.different_from(test_node)

    # larger angle difference, should be different
    test_node = copy.deepcopy(root)
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, planes[0][1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 9, axis)
    normal = trimesh.transform_points(planes[0][1][None, :], rotation)[0]
    test_plane = (planes[0][0], normal)
    test_node.split(test_plane)
    assert base_node.different_from(test_node)


def test_copy_tree():
    """Now that objectives are calculated outside of the tree (using the objective function evaluators), verify
    that copying a tree doesn't modify its objectives dict
    """
    config = Configuration.config
    mesh = trimesh.load(config.mesh, validate=True)

    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = bsp.BSPTree(mesh)
    node = tree.largest_part()
    normal = np.array([0, 0, 1])
    planes = node.get_planes(normal)
    plane = planes[len(planes) // 2]
    tree = tree.expand_node(plane, node)
    new_tree = tree.copy()
    assert new_tree.objectives == tree.objectives


def test_expand_node():
    """no errors when using expand_node, need to think of better tests here"""
    config = Configuration.config
    mesh = trimesh.load(config.mesh, validate=True)

    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = bsp.BSPTree(mesh)

    node = tree.largest_part()
    normal = np.array([0, 0, 1])
    planes = node.get_planes(normal)
    plane = planes[len(planes) // 2]
    tree1 = tree.expand_node(plane, node)
    print("tree objective: ", tree1.get_objective())

    node = tree1.largest_part()
    planes = node.get_planes(normal)
    plane = planes[len(planes) // 2]
    tree2 = tree1.expand_node(plane, node)


def test_grid_sample():
    """verify that when the cross section is barely larger than the connector diameter, only 1 sample is
    returned by `ConnectedComponent.grid_sample_polygon`"""
    config = Configuration.config
    origin, normal = (np.zeros(3), np.array([0, 0, 1]))

    # test
    cd = config.connector_diameter
    mesh = trimesh.primitives.Box(extents=[1.1 * cd, 1.1 * cd, 40])
    cross_section = section.CrossSection(mesh, origin, normal)
    samples = cross_section.connected_components[0].grid_sample_polygon()
    assert samples.size > 0

    mesh.apply_translation([3, 0, 0])
    cross_section = section.CrossSection(mesh, origin, normal)
    samples = cross_section.connected_components[0].grid_sample_polygon()
    assert samples.size > 0

    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/4, np.array([0, 0, 1])))
    cross_section = section.CrossSection(mesh, origin, normal)
    samples = cross_section.connected_components[0].grid_sample_polygon()
    assert samples.size > 0


def test_basic_separation():
    config = Configuration.config
    config.part_separation = True
    mesh = trimesh.load(os.path.join(os.path.dirname(__file__), 'test_meshes', 'separate_test.stl'))
    tree = bsp.BSPTree(mesh)
    node = tree.largest_part()
    plane = (np.zeros(3), np.array([1, 0, 0]))
    tree = tree.expand_node(plane, node)
    # 1 root, three leaves come out of the split
    assert len(tree.nodes) == 4
    config.restore_defaults()
