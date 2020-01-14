"""
TODO:
    - reorder objective function calculation to calculate objectives for many planes all with same normal
    - objectives should be stored for previous trees to minimize recalculation (most of the nodes are shared)
    - implement symmetry objective
    - add other objectives to make the shoe rack chop better
"""
import numpy as np
import trimesh

from pychop3d.configuration import Configuration


def evaluate_nparts_objective(trees, path):
    """Collect the "number of parts" objective for a set of trees
    """
    theta_0 = trees[0].nodes[0].n_parts
    for tree in trees:
        node = tree.get_node(path)
        tree.objectives['nparts'] += (sum([c.n_parts for c in node.children]) - node.n_parts)/ theta_0


def evaluate_utilization_objective(trees, path):
    config = Configuration.config
    V = np.prod(config.printer_extents)
    if config.obb_utilization:
        for tree in trees:
            node = tree.get_node(path)
            tree.objectives['utilization'] = max(tree.objectives['utilization'],
                                                 max([1 - c.get_bounding_box_oriented().volume / (c.n_parts * V) for c in node.children]))
    else:
        for tree in trees:
            node = tree.get_node(path)
            tree.objectives['utilization'] = max(tree.objectives['utilization'],
                                                 max([1 - c.part.volume / (c.n_parts * V) for c in node.children]))


def evaluate_connector_objective(trees, path):
    for tree in trees:
        node = tree.get_node(path)
        tree.objectives['connector'] = max(tree.objectives['connector'], node.get_connection_objective())


def get_fragility_for_normal(part, normal, origins, normal_parallel_th, connector_sizes):
    fragility_objective = np.zeros(origins.shape[0])

    origin_diffs = part.vertices[None, :, :] - origins[:, None, :]
    vertex_projections = origin_diffs @ normal
    distances_to_plane = np.abs(vertex_projections)

    # find vertices who's normals are almost parallel to the normal
    possibly_fragile = (part.vertex_normals @ normal) > normal_parallel_th
    possibly_fragile = (vertex_projections > 0) * possibly_fragile[None, :]

    # sink the ray origins inside the part a little
    ray_origins = part.vertices - .001 * part.vertex_normals
    ray_directions = np.ones((ray_origins.shape[0], 1)) * -1 * normal[None, :]
    hits = part.ray.intersects_any(ray_origins, ray_directions)

    no_hit_p_fragile = ~hits[None, :] * possibly_fragile
    close_to_plane = distances_to_plane < 1.5 * connector_sizes[:, None]
    mask = np.any(no_hit_p_fragile * close_to_plane, axis=1)
    fragility_objective[mask] = np.inf

    locs, index_ray, index_tri = part.ray.intersects_location(ray_origins[hits], ray_directions[hits], multiple_hits=False)
    ray_mesh_dist = np.sqrt(np.sum((ray_origins[index_ray] - locs) ** 2, axis=1))
    not_existing = distances_to_plane[:, index_ray] < ray_mesh_dist[None, :]
    mask = np.any(possibly_fragile[:, index_ray] * close_to_plane[:, index_ray] * not_existing, axis=1)
    fragility_objective[mask] = np.inf

    return fragility_objective


def evaluate_fragility_objective(trees, path):
    """Get fragility objective for a set of trees who only differ by the origin of their last cut

        - figure out possibly fragile points
        - cast rays from those points in the direction of normal
        - if the rays don't intersect the mesh somewhere else, check if the rays are longer than the Thold
        - if they do, check the thold but also make sure the ray hits the plane first
    """
    config = Configuration.config
    part = trees[0].get_node(path).part
    normal = trees[0].get_node(path).plane[1]
    origins = np.array([t.get_node(path).plane[0] for t in trees])
    connector_sizes = np.array([t.get_node(path).cross_section.get_average_connector_size() for t in trees])

    positive_fragility = get_fragility_for_normal(part, normal, origins, config.fragility_objective_th, connector_sizes)
    negative_fragility = get_fragility_for_normal(part, -1 * normal, origins, config.fragility_objective_th, connector_sizes)

    fragility = positive_fragility + negative_fragility

    for i, tree in enumerate(trees):
        tree.objectives['fragility'] += fragility[i]


def evaluate_seam_objective(trees, path):
    return 0


def evaluate_symmetry_objective(trees, path):
    return 0


objectives = {
    'nparts': evaluate_nparts_objective,
    'utilization': evaluate_utilization_objective,
    'connector': evaluate_connector_objective,
    'fragility': evaluate_fragility_objective
}
