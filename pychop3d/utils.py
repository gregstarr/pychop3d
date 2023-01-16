from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import trimesh
from trimesh import Trimesh

from pychop3d import bsp_tree


def load_connector_configuration(tree_file: Path) -> np.ndarray:
    """loads `state` from tree file

    Args:
        tree_file (Path): path to tree file json

    Returns:
        np.ndarray: connector state vector
    """
    with open(tree_file, encoding="utf8") as f:
        data = json.load(f)
    return np.array(data["state"])


def make_plane(plane, w=100):
    xform = np.linalg.inv(trimesh.points.plane_transform(plane.origin, plane.normal))
    box = trimesh.primitives.Box(extents=(w, w, 0.5), transform=xform)
    return box


def trimesh_repair(mesh: Trimesh):
    """runs a few trimesh repair utilities on a mesh in-place

    Args:
        mesh (Trimesh)
    """
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    mesh.process()


def preview_tree(tree: bsp_tree.BSPTree, other_objects: Trimesh = None):
    """runs the trimesh pyglet utility, shows the mesh and other optional objects,
    e.g. make_plane

    Args:
        tree (BSPTree)
        other_objects (Trimesh, optional): Defaults to None.
    """
    if other_objects is None:
        other_objects = []
    scene = trimesh.Scene()
    for leaf in tree.leaves:
        leaf.part.visual.face_colors = np.random.rand(3) * 255
        scene.add_geometry(leaf.part)
    for ob in other_objects:
        scene.add_geometry(ob)
    scene.camera.z_far = 10_000
    scene.show()
