import numpy as np
import trimesh

from pychop3d.configuration import Configuration
from pychop3d import bsp_tree
from pychop3d import utils
from pychop3d.logger import logger


def match_connectors(original_tree: bsp_tree.BSPTree, tree: bsp_tree.BSPTree):
    for i, node in enumerate(tree.nodes):
        if node.cross_section is None:
            continue
        # create global vector of connectors
        for j, cc in enumerate(node.cross_section.connected_components):
            original_tree.nodes[i].cross_section.connected_components[j].index = cc.index
            original_tree.nodes[i].cross_section.connected_components[j].sites = cc.sites


class ConnectorPlacer:

    def __init__(self, tree: bsp_tree.BSPTree):
        config = Configuration.config

        self.connected_components = []
        self.connectors = []
        self.n_connectors = 0
        caps = []
        if len(tree.nodes) < 2:
            raise Exception("input tree needs to have a chop")

        logger.info("Creating connectors...")
        for n, node in enumerate(tree.nodes):
            if node.cross_section is None:
                continue
            # create global vector of connectors
            for cc in node.cross_section.connected_components:
                caps.append(cc.mesh)
                # register the ConnectedComponent's sites with the global array of connectors
                cc.register_sites(len(self.connectors))
                self.connected_components.append(cc)
                for site in cc.sites:
                    conn_m = trimesh.primitives.Sphere(radius=cc.connector_diameter / 2)
                    conn_m.apply_transform(np.linalg.inv(trimesh.points.plane_transform(site, cc.normal)))
                    self.connectors.append(conn_m)

        self.connectors = np.array(self.connectors)
        self.n_connectors = self.connectors.shape[0]
        if self.n_connectors == 0:
            return
        logger.info(f"Number of connectors: {self.n_connectors}")
        self.collisions = np.zeros((self.n_connectors, self.n_connectors), dtype=bool)

        logger.info("determining connector-cut intersections")
        mass_centers = np.array([c.center_mass for c in self.connectors])
        intersections = np.zeros(self.n_connectors, dtype=int)
        for cap in caps:
            intersections[cap.nearest.on_surface(mass_centers)[1] < config.connector_diameter] += 1
        self.collisions[intersections > 1, :] = True
        self.collisions[:, intersections > 1] = True

        logger.info("determining connector mesh protrusions")
        for n, node in enumerate(tree.nodes):
            if node.cross_section is None:
                continue
            origin, normal = node.plane
            origin -= normal * .1
            for cc in node.cross_section.connected_components:
                for idx in cc.index:
                    part = node.children[cc.negative].part
                    xform = self.connectors[idx].primitive.transform
                    conn_f = trimesh.primitives.Sphere(radius=(cc.connector_diameter + config.connector_tolerance + 1) / 2, transform=xform)
                    sliced_conn_f = conn_f.slice_plane(origin, -1 * normal)
                    v_check = part.contains(sliced_conn_f.vertices)
                    protrude = not v_check.all()
                    if protrude:
                        self.collisions[idx, :] = True
                        self.collisions[:, idx] = True

        logger.info("determining connector-connector intersections")
        distances = np.sqrt(np.sum((mass_centers[:, None, :] - mass_centers[None, :, :]) ** 2, axis=2))
        mask = (distances > 0) & (distances < config.connector_diameter)
        self.collisions = np.logical_or(self.collisions, mask)

    def evaluate_connector_objective(self, state):
        config = Configuration.config
        objective = 0
        n_collisions = self.collisions[state, :][:, state].sum()
        objective += config.connector_collision_penalty * n_collisions

        for cc in self.connected_components:
            rc = min(np.sqrt(cc.area) / 2, 10 * cc.connector_diameter)
            sites = cc.get_sites(state)
            ci = 0
            if len(sites) > 0:
                ci = len(sites) * np.pi * rc ** 2
                distances = np.sqrt(np.sum((sites[:, None, :] - sites[None, :, :]) ** 2, axis=2))
                mask = (0 < distances) * (distances < 2 * rc)
                ci -= np.sum(np.pi * (rc - distances[mask] / 2) ** 2)
            objective += cc.area / (config.empty_cc_penalty + max(0, ci))
            if ci < 0:
                objective -= ci / cc.area

        return objective

    def get_initial_state(self):
        state = np.zeros(self.n_connectors, dtype=bool)
        for cc in self.connected_components:
            for i in np.random.choice(cc.index, 2, replace=False):
                state[i] = True
        return state

    def simulated_annealing_connector_placement(self):
        config = Configuration.config
        state = self.get_initial_state()
        objective = self.evaluate_connector_objective(state)
        logger.info(f"initial objective: {objective}")
        # initialization
        for i in range(config.sa_initialization_iterations):
            state, objective = self.sa_iteration(state, objective, 0)

        logger.info(f"post initialization objective: {objective}")
        initial_temp = objective / 2
        for i, temp in enumerate(np.linspace(initial_temp, 0, config.sa_iterations)):
            state, objective = self.sa_iteration(state, objective, temp)

        logger.info(f"final objective: {objective}")
        return state

    def sa_iteration(self, state, objective, temp):
        new_state = state.copy()
        if np.random.randint(0, 2) or not state.any() or state.all():
            e = np.random.randint(0, self.n_connectors)
            new_state[e] = 0 if state[e] else 1
        else:
            add = np.random.choice(np.argwhere(~state)[:, 0])
            remove = np.random.choice(np.argwhere(state)[:, 0])
            new_state[add] = 1
            new_state[remove] = 0

        new_objective = self.evaluate_connector_objective(new_state)
        if new_objective < objective:
            return new_state, new_objective
        elif temp > 0:
            if np.random.rand() < np.exp(-(new_objective - objective) / temp):
                return new_state, new_objective
        return state, objective

    def insert_connectors(self, tree, state):
        logger.info(f"inserting {state.sum()} connectors")
        config = Configuration.config
        if tree.nodes[0].plane is None:
            new_tree = utils.separate_starter(tree.nodes[0].part)
        else:
            new_tree = bsp_tree.BSPTree(tree.nodes[0].part)
        for node in tree.nodes:
            if node.plane is None:
                continue
            new_tree2, result = bsp_tree.expand_node(new_tree, node.path, node.plane)
            if result != 'success':
                # for debugging
                bsp_tree.expand_node(new_tree, node.path, node.plane)
            else:
                new_tree = new_tree2
            new_node = new_tree.get_node(node.path)
            if node.cross_section is None:
                continue
            origin, normal = node.plane
            origin += normal * .1
            for cc in node.cross_section.connected_components:
                index = cc.get_indices(state)
                pi = cc.positive
                ni = cc.negative
                for idx in index:
                    xform = self.connectors[idx].primitive.transform
                    conn_m = self.connectors[idx].slice_plane(origin, -1 * normal)
                    conn_f = trimesh.primitives.Sphere(radius=(cc.connector_diameter + config.connector_tolerance) / 2, transform=xform)
                    new_node.children[ni].part = insert_connector_f(new_node.children[ni].part, conn_f)
                    new_node.children[pi].part = insert_connector_m(new_node.children[pi].part, conn_m)
        return new_tree


def insert_connector_f(part: trimesh.Trimesh, conn_f: trimesh.Trimesh, retries=10):
    """Inserts a slot into a part using boolean difference. Operating under the assumption
    that inserting a slot MUST INCREASE the number of vertices of the resulting part.

    :param part: part to insert slot into
    :param slot: slot (connector expanded with tolerance) to remove from part
    :param retries: number of times to retry before raising an error, checking to see if number of
                    vertices increases. Default = 10
    :type retries: int
    :return: The part with the slot inserted
    :rtype: trimesh.Trimesh
    """
    utils.trimesh_repair(part)
    for t in range(retries):
        new_part = part.difference(conn_f)
        if len(new_part.vertices) > len(part.vertices):
            return new_part
    raise Exception("Couldn't insert connector")


def insert_connector_m(part: trimesh.Trimesh, conn_m: trimesh.Trimesh, retries=10):
    """Adds a box / connector to a part using boolean union. Operating under the assumption
    that adding a connector MUST INCREASE the number of vertices of the resulting part.

    :param part: part to add connector to
    :param box: connector to add to part
    :param retries: number of times to retry before raising an error, checking to see if number of
                    vertices increases. Default = 10
    :type retries: int
    :return: The part with the connector added
    :rtype: trimesh.Trimesh
    """
    utils.trimesh_repair(part)
    for t in range(retries):
        new_part = part.union(conn_m)
        if len(new_part.vertices) > len(part.vertices):
            return new_part
    raise Exception("Couldn't insert connector")
