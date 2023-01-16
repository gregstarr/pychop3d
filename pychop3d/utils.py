import json
from pathlib import Path

import numpy as np
import trimesh
from trimesh import Trimesh


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


def preview_tree(tree: "BSPTree", other_objects: Trimesh = None):
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


def save_tree(tree: "BSPTree", save_path: Path, state: np.ndarray = None):
    """saves tree file json

    Args:
        tree (BSPTree)
        save_path (Path)
        state (np.ndarray, optional): Defaults to None.
    """
    if state is None:
        state = []

    nodes = []
    for node in tree.nodes:
        if node.plane is None:
            continue
        this_node = {
            "path": node.path,
            "origin": list(node.plane[0]),
            "normal": list(node.plane[1]),
        }
        nodes.append(this_node)

    with open(save_path, "w", encoding="utf8") as f:
        json.dump({"nodes": nodes, "state": [bool(s) for s in state]}, f)


def export_tree_stls(tree: "BSPTree", output_dir: Path, name: str):
    """Saves all of a tree's parts

    Args:
        tree (BSPTree)
        output_dir (Path)
        name (str)
    """
    for i, leaf in enumerate(tree.leaves):
        leaf.part.export(output_dir / f"{name}_{i}.stl")
