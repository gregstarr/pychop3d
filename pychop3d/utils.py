import numpy as np
from pychop3d import constants


def all_at_goal(trees):
    for tree in trees:
        if not tree.terminated():
            return False
    return True


def not_at_goal_set(trees):
    not_at_goal = []
    for tree in trees:
        if not tree.terminated():
            not_at_goal.append(tree)
    return not_at_goal


def uniform_normals(n=constants.N_RANDOM_NORMALS):
    """http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    theta = np.random.rand(n) * 2 * np.pi
    phi = np.arccos(1 - 2 * np.random.rand(n))
    return np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=1)
