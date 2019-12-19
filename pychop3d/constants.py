import numpy as np
import trimesh


def uniform_normals(n_theta, n_phi):
    """http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    theta = np.arange(0, np.pi, np.pi / n_theta)
    phi = np.arccos(1 - np.arccos(1 - np.arange(0, 1, 1 / n_phi)))
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.ravel()
    phi = phi.ravel()
    return np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=1)


PRINTER_EXTENTS = np.array([200, 200, 200], dtype=float)
PLANE_SPACING = 20
N_THETA = 5
N_PHI = 5
UNIFORM_NORMALS = uniform_normals(N_THETA, N_PHI)

DIFFERENT_ORIGIN_THRESHOLD = .1 * np.sqrt(np.sum(PRINTER_EXTENTS ** 2))
DIFFERENT_ANGLE_THRESHOLD = np.pi/10

# objective weightings
A_PART = 1
A_UTIL = .25
A_CONNECTOR = 1
A_FRAGILITY = 1
A_SEAM = .1
A_SYMMETRY = .25

CONNECTOR_COLLISION_WEIGHT = 10**10

FRAGILITY_THRESHOLD = .95

CONNECTOR_DIAMETER_MIN = 5
CONNECTOR_DIAMETER_MAX = 30
CONNECTOR_TOLERANCE = 1
CONNECTOR_OBJECTIVE_THRESHOLD = 5

EPSILON = 1e-6

CH_SAMPLE_POINTS = trimesh.primitives.Sphere(radius=1000).vertices

INITIAL_CONNECTOR_RATIO = .1
INITIALIZATION_ITERATIONS = 15_000
ANNEALING_ITERATIONS = 300_000

default_config = {'mesh': "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl",
                  'directory': "C:\\Users\\Greg\\code\\pychop3d\\debug",
                  'scale': True,
                  'beam_width': 2}
