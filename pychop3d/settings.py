"""Module for parameters that the user should not pass in"""
import numpy as np


def uniform_normals(n_t, n_p):
    """http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    theta = np.arange(0, np.pi, np.pi / n_t)
    phi = np.arccos(np.arange(0, 1, 1 / n_p))
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.ravel()
    phi = phi.ravel()
    return np.column_stack(
        (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))
    )

PLANE_SPACING = 20
N_THETA = 5
N_PHI = 5
NORMALS = uniform_normals(N_THETA, N_PHI)
ADD_MIDDLE_PLANE = True
DIFFERENT_ORIGIN_TH = 20
DIFFERENT_ANGLE_TH = np.pi / 10
# objective parameters
OBJECTIVE_WEIGHTS = {
    "part": 1,
    "utilization": .25,
    "connector": 1,
    "fragility": 1,
    "seam": 0,  # set to zero until implemented
    "symmetry": 0  # set to zero until implemented
}
FRAGILITY_OBJECTIVE_TH = .95
CONNECTOR_OBJECTIVE_TH = 10
OBB_UTILIZATION = False
# connector placement
CONNECTOR_COLLISION_PENALTY = 10 ** 6
EMPTY_CC_PENALTY = 10**-5
SA_INITIAL_CONNECTOR_RATIO = .1
SA_INITIALIZATION_ITERATIONS = 5_000
SA_ITERATIONS = 100_000
# connector settings
CONNECTOR_DIAMETER = 3
CONNECTOR_TOLERANCE = .2
CONNECTOR_SPACING = 10

BEAM_WIDTH = 3
PART_SEPARATION = False
MAX_FACES = 3000
