"""
TODO:
    - reorder objective function calculation to calculate objectives for many planes all with same normal
    - objectives should be stored for previous trees to minimize recalculation (most of the nodes are shared)
    - implement symmetry objective
    - add other objectives to make the shoe rack chop better
    - improve fragility
"""
import numpy as np

from pychop3d.configuration import Configuration


def get_nparts_objective(trees):
    """Collect the "number of parts" objective for a set of trees
    """
    theta_0 = trees[0].nodes[0].n_parts
    return np.array([sum([l.n_parts for l in t.get_leaves()]) for t in trees]) / theta_0


def get_utilization_objective(trees):
    config = Configuration.config
    V = np.prod(config.printer_extents)
    return np.array([max([1 - leaf.get_bounding_box_oriented().volume / (leaf.n_parts * V)
                          for leaf in t.get_leaves()]) for t in trees])


def get_connector_objective(self):
    return max([n.get_connection_objective() for n in self.nodes if n.cross_section is not None])


def get_fragility_for_normal(part, normal, origins, normal_parallel_th, connector_sizes):
    fragility_objective = np.zeros(origins.shape[0])
    # find vertices who's normals are almost parallel to the normal
    possibly_fragile = (part.vertex_normals @ normal) > normal_parallel_th
    # sink the ray origins inside the part a little
    ray_origins = part.vertices[possibly_fragile] - .1 * part.vertex_normals[possibly_fragile]
    ray_directions = np.ones((ray_origins.shape[0], 1)) * normal[None, :]
    hits = part.ray.intersects_any(ray_origins, ray_directions)  # N possibly fragile verts

    origin_diffs = origins[:, None, :] - ray_origins[None, :, :]  # N trees X N possibly fragile verts X 3
    vertex_projections = origin_diffs @ normal  # N trees X N possibly fragile verts
    distances_to_plane = np.abs(vertex_projections)  # N trees X N possibly fragile verts

    mask = np.any(distances_to_plane[:, ~hits] < 1.5 * connector_sizes, axis=1)
    fragility_objective[mask] = np.inf

    locs, index_ray, index_tri = part.ray.intersects_location(ray_origins, ray_directions)
    ray_mesh_dist = np.sqrt(np.sum((ray_origins[index_ray] - locs) ** 2, axis=1))
    thin_mask = distances_to_plane[:, index_ray] < 1.5 * connector_sizes
    not_existing = distances_to_plane[:, index_ray] < ray_mesh_dist[None, :]
    mask = np.any(thin_mask * not_existing, axis=1)
    fragility_objective[mask] = np.inf

    return fragility_objective


def get_fragility_objective(trees, path):
    """Get fragility objective for a set of trees who only differ by the origin of their last cut

        - figure out possibly fragile points
        - cast rays from those points in the direction of normal
        - if the rays don't intersect the mesh somewhere else,
    """
    config = Configuration.config
    part = trees[0].get_node(path[:-1]).part
    normal = trees[0].get_node(path[:-1]).normal
    origins = np.array([t.get_node(path[:-1]).plane[0] for t in trees])
    connector_sizes = np.array([t.get_node(path[:-1]).cross_section.get_average_connector_size() for t in trees])

    positive_fragility = get_fragility_for_normal(part, normal, origins, config.fragility_objective_th, connector_sizes)
    negative_fragility = get_fragility_for_normal(part, normal, origins, config.fragility_objective_th, connector_sizes)

    return positive_fragility + negative_fragility


def get_seam_objective(trees):
    return 0


def get_symmetry_objective(trees):
    return 0