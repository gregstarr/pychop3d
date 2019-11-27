import trimesh
import numpy as np

from pychop3d import bsp
from pychop3d import constants


def test_get_planes():
    mesh = trimesh.primitives.Sphere(radius=10)
    normal = np.array([0, 0, 1], dtype=float)
    origin = np.array([0, 0, 9])
    s = mesh.section(plane_origin=origin, plane_normal=normal)
    origin = np.array([0, 0, 10])
    s = mesh.section(plane_origin=origin, plane_normal=normal)
    origin = np.array([0, 0, 11])
    s = mesh.section(plane_origin=origin, plane_normal=normal)
    print()
