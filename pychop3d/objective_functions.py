"""Objective functions"""
from __future__ import annotations
import numpy as np
from trimesh import Trimesh

from pychop3d import settings, bsp_tree
from pychop3d.logger import logger


def evaluate_nparts_objective(trees: list[bsp_tree.BSPTree], path: tuple):
    """Collect the "number of parts" objective for a set of trees"""
    theta_0 = trees[0].nodes[0].n_parts
    for tree in trees:
        node = tree.get_node(path)
        nparts = (sum([c.n_parts for c in node.children]) - node.n_parts) / theta_0
        tree.objectives["nparts"] += nparts


def evaluate_utilization_objective(
    trees: list[bsp_tree.BSPTree], path: tuple, printer_extents: np.ndarray
):
    V = np.prod(printer_extents)
    for tree in trees:
        node = tree.get_node(path)
        if settings.OBB_UTILIZATION:
            util_obj = max(
                tree.objectives["utilization"],
                max(1 - c.obb.volume / (c.n_parts * V) for c in node.children),
            )
        else:
            util_obj = max(
                tree.objectives["utilization"],
                max(1 - c.part.volume / (c.n_parts * V) for c in node.children),
            )
        tree.objectives["utilization"] = util_obj


def evaluate_connector_objective(trees: list[bsp_tree.BSPTree], path: tuple):
    for tree in trees:
        node = tree.get_node(path)
        conn_obj = max(tree.objectives["connector"], node.connection_objective)
        tree.objectives["connector"] = conn_obj


def get_fragility_for_normal(part: Trimesh, normal: np.ndarray, origins: np.ndarray):

    fragility_objective = np.zeros(origins.shape[0])

    origin_diffs = part.vertices[None, :, :] - origins[:, None, :]
    vertex_projections = origin_diffs @ normal
    distances_to_plane = np.abs(vertex_projections)

    # find vertices who's normals are almost parallel to the normal
    possibly_fragile = (part.vertex_normals @ normal) > settings.FRAGILITY_OBJECTIVE_TH
    possibly_fragile = (vertex_projections > 0) & possibly_fragile[None, :]

    # sink the ray origins inside the part a little
    ray_origins = part.vertices - 0.001 * part.vertex_normals
    ray_directions = np.ones((ray_origins.shape[0], 1)) * -1 * normal[None, :]
    hits = part.ray.intersects_any(ray_origins, ray_directions)

    no_hit_p_fragile = ~hits[None, :] * possibly_fragile
    close_to_plane = distances_to_plane < 1.5 * settings.CONNECTOR_DIAMETER
    mask = np.any(no_hit_p_fragile * close_to_plane, axis=1)
    fragility_objective[mask] = np.inf

    locs, index_ray, _ = part.ray.intersects_location(
        ray_origins[hits], ray_directions[hits], multiple_hits=False
    )
    ray_mesh_dist = np.sqrt(np.sum((ray_origins[index_ray] - locs) ** 2, axis=1))
    not_existing = distances_to_plane[:, index_ray] < ray_mesh_dist[None, :]
    mask = np.any(
        possibly_fragile[:, index_ray] * close_to_plane[:, index_ray] * not_existing,
        axis=1,
    )
    fragility_objective[mask] = np.inf

    return fragility_objective


def evaluate_fragility_objective(trees: list[bsp_tree.BSPTree], path: tuple):
    """Get fragility objective for a set of trees who only differ by the origin of their last cut

    - figure out possibly fragile points
    - cast rays from those points in the direction of normal
    - if the rays don't intersect the mesh somewhere else, check if the rays are longer than the Thold
    - if they do, check the thold but also make sure the ray hits the plane first
    """
    part = trees[0].get_node(path).part
    normal = trees[0].get_node(path).plane.normal
    origins = np.array([t.get_node(path).plane[0] for t in trees])

    positive_fragility = get_fragility_for_normal(part, normal, origins)
    negative_fragility = get_fragility_for_normal(part, -1 * normal, origins)

    fragility = positive_fragility + negative_fragility

    for i, tree in enumerate(trees):
        tree.objectives["fragility"] += fragility[i]


objectives = {
    "nparts": evaluate_nparts_objective,
    "utilization": evaluate_utilization_objective,
    "connector": evaluate_connector_objective,
    "fragility": evaluate_fragility_objective,
}
