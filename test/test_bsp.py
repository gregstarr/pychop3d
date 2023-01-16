import trimesh
import numpy as np
import copy
from pathlib import Path
import pytest

from pychop3d.bsp_tree import BSPTree, get_planes, expand_node
from pychop3d.bsp_node import split, Plane
from pychop3d.section import CrossSection
from pychop3d import utils, settings


def test_get_planes(bunny_mesh):
    """verify that for the default bunny mesh, which is a single part, all planes returned by `bsp_tree.get_planes`
        cut through the mesh (they have a good cross section)
    """
    mesh = trimesh.load(bunny_mesh, validate=True)

    for _ in range(100):
        normal = trimesh.unitize(np.random.rand(3))
        planes = get_planes(mesh, normal)

        for origin, normal in planes:
            path3d = mesh.section(plane_origin=origin, plane_normal=normal)
            assert path3d is not None


def test_different_from():
    """verify that `BSPNode.different_from` has the expected behavior

    Get a list of planes. Split the object using the first plane, then for each of the other planes, split the object,
    check if the plane is far enough away given the config, then assert that `BSPNode.different_from` returns the
    correct value. This skips any splits that fail.
    """

    mesh = trimesh.primitives.Sphere(radius=50)
    printer_extents = np.ones(3) * 200

    tree = BSPTree(mesh, printer_extents)
    root = tree.nodes[0]
    normal = trimesh.unitize(np.random.rand(3))
    planes = get_planes(mesh, normal)
    base_node = copy.deepcopy(root)
    base_node, result = split(base_node, planes[0])
    print(result)

    for plane in planes[1:]:
        # smaller origin offset, should not be different
        test_node = copy.deepcopy(root)
        test_node, result = split(test_node, plane)
        if abs((plane[0] - planes[0][0]) @ planes[0][1]) > settings.DIFFERENT_ORIGIN_TH:
            assert base_node.different_from(test_node)
        else:
            assert not base_node.different_from(test_node)

    # smaller angle difference, should not be different
    test_node = copy.deepcopy(root)
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, planes[0][1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 11, axis)
    normal = trimesh.transform_points(planes[0][1][None, :], rotation)[0]
    test_plane = Plane(planes[0][0], normal)
    test_node, result = split(test_node, test_plane)
    assert not base_node.different_from(test_node)

    # larger angle difference, should be different
    test_node = copy.deepcopy(root)
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, planes[0][1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 9, axis)
    normal = trimesh.transform_points(planes[0][1][None, :], rotation)[0]
    test_plane = Plane(planes[0][0], normal)
    test_node, result = split(test_node, test_plane)
    assert base_node.different_from(test_node)


def test_copy_tree(bunny_mesh):
    """Now that objectives are calculated outside of the tree (using the objective function evaluators), verify
    that copying a tree doesn't modify its objectives dict
    """
    mesh = trimesh.load(bunny_mesh, validate=True)
    printer_extents = np.ones(3) * 200

    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = BSPTree(mesh, printer_extents)
    node = tree.largest_part
    normal = np.array([0, 0, 1])
    planes = get_planes(node.part, normal)
    plane = planes[len(planes) // 2]
    tree, result = expand_node(tree, node.path, plane)
    print(result)
    new_tree = tree.copy()
    assert new_tree.objectives == tree.objectives


def test_expand_node(bunny_mesh):
    """no errors when using expand_node, need to think of better tests here"""
    mesh = trimesh.load(bunny_mesh, validate=True)
    printer_extents = np.ones(3) * 200

    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = BSPTree(mesh, printer_extents)

    node = tree.largest_part
    normal = np.array([0, 0, 1])
    planes = get_planes(node.part, normal)
    plane = planes[0]
    tree1, result = expand_node(tree, node.path, plane)
    assert result == 'success'
    print("tree objective: ", tree1.objective)

    node = tree1.largest_part
    planes = get_planes(node.part, normal)
    plane = planes[0]
    tree2, result = expand_node(tree1, node.path, plane)
    print(tree2)
    assert result == 'success'


def test_grid_sample():
    """verify that when the cross section is barely larger than the connector diameter, only 1 sample is
    returned by `ConnectedComponent.grid_sample_polygon`"""
    plane = Plane(np.zeros(3), np.array([0, 0, 1]))

    # test
    cd = settings.CONNECTOR_DIAMETER
    mesh = trimesh.primitives.Box(extents=[cd + .1, cd + .1, 40]).to_mesh()
    cross_section = CrossSection(mesh, plane)
    samples = cross_section.connected_components[0].grid_sample_polygon()
    assert samples.shape[0] == 1

    mesh.apply_translation([3, 0, 0])
    cross_section = CrossSection(mesh, plane)
    samples = cross_section.connected_components[0].grid_sample_polygon()
    assert samples.shape[0] == 1

    xform = trimesh.transformations.rotation_matrix(np.pi/4, np.array([0, 0, 1]))
    mesh.apply_transform(xform)
    cross_section = CrossSection(mesh, plane)
    samples = cross_section.connected_components[0].grid_sample_polygon()
    assert samples.shape[0] == 1


@pytest.fixture
def set_part_separation():
    original = settings.PART_SEPARATION
    settings.PART_SEPARATION = True
    yield
    settings.PART_SEPARATION = original


def test_basic_separation(set_part_separation):
    mesh = trimesh.load(Path(__file__).parent / 'test_meshes' / 'separate_test.stl')
    printer_extents = np.ones(3) * 200
    tree = BSPTree(mesh, printer_extents)
    node = tree.largest_part
    plane = Plane(np.zeros(3), np.array([1, 0, 0]))
    tree, result = expand_node(tree, node.path, plane)
    print(result)
    # 1 root, three leaves come out of the split
    assert len(tree.nodes) == 4
