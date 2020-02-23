import numpy as np
import logging
import trimesh

from pychop3d.configuration import Configuration
from pychop3d import bsp_tree
from pychop3d import utils


logger = logging.getLogger(__name__)


class ConnectorPlacer:

    def __init__(self, tree):
        config = Configuration.config

        self.connected_components = []
        self.connectors = []
        self.n_connectors = 0
        caps = []
        if len(tree.nodes) < 2:
            raise Exception("input tree needs to have a chop")

        logger.info("\nCreating connectors...")
        for n, node in enumerate(tree.nodes):
            if node.cross_section is None:
                continue

            # create global vector of connectors
            for cc in node.cross_section.connected_components:
                caps.append(cc.mesh)
                # register the ConnectedComponent's sites with the global array of connectors
                cc.register_sites(len(self.connectors))
                self.connected_components.append(cc)
                for site in cc.positive_sites:
                    box = trimesh.primitives.Box(extents=np.ones(3) * cc.connector_diameter)
                    box.apply_transform(np.linalg.inv(trimesh.points.plane_transform(site, cc.normal)))
                    box.apply_transform(trimesh.transformations.translation_matrix(cc.normal * (cc.connector_diameter / 2 - .1)))
                    self.connectors.append(box)
                for site in cc.negative_sites:
                    box = trimesh.primitives.Box(extents=np.ones(3) * cc.connector_diameter)
                    box.apply_transform(np.linalg.inv(trimesh.points.plane_transform(site, -1 * cc.normal)))
                    box.apply_transform(trimesh.transformations.translation_matrix(-1 * cc.normal * (cc.connector_diameter / 2 - .1)))
                    self.connectors.append(box)

        self.connectors = np.array(self.connectors)
        self.n_connectors = self.connectors.shape[0]
        logger.info(f"Number of connectors: {self.n_connectors}")
        self.collisions = np.zeros((self.n_connectors, self.n_connectors), dtype=bool)

        logger.info("determining connector-cut intersections")
        mass_centers = np.array([c.center_mass for c in self.connectors])
        intersections = np.zeros(self.n_connectors, dtype=int)
        for cap in caps:
            intersections[cap.nearest.on_surface(mass_centers)[1] < config.connector_diameter] += 1
        self.collisions[intersections > 1, :] = True
        self.collisions[:, intersections > 1] = True

        logger.info("determining connector-connector intersections")
        distances = np.sqrt(np.sum((mass_centers[:, None, :] - mass_centers[None, :, :]) ** 2, axis=2))
        mask = (distances > 0) * (distances < config.connector_diameter * 1.5)
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
            for i in np.random.choice(cc.all_index, 2, replace=False):
                state[i] = True
        return state

    def simulated_annealing_connector_placement(self):
        config = Configuration.config
        state = self.get_initial_state()
        objective = self.evaluate_connector_objective(state)
        logger.info(f"\ninitial objective: {objective}")
        # initialization
        for i in range(config.sa_initialization_iterations):
            if not i % (config.sa_initialization_iterations // 10):
                print('.', end='')
            state, objective = self.sa_iteration(state, objective, 0)

        logger.info(f"\npost initialization objective: {objective}")
        initial_temp = objective / 2
        for i, temp in enumerate(np.linspace(initial_temp, 0, config.sa_iterations)):
            if not i % (config.sa_iterations // 10):
                print('.', end='')
            state, objective = self.sa_iteration(state, objective, temp)

        logger.info(f"\nfinal objective: {objective}")
        return state

    def sa_iteration(self, state, objective, temp):
        new_state = state.copy()
        if np.random.randint(0, 2) or not state.any():
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
        config = Configuration.config
        new_tree = bsp_tree.BSPTree(tree.nodes[0].part)
        for node in tree.nodes:
            if node.plane is None:
                continue
            new_tree2 = bsp_tree.expand_node(new_tree, node.path, node.plane)
            if new_tree2 is None:
                bsp_tree.expand_node(new_tree, node.path, node.plane)
            else:
                new_tree = new_tree2
            new_node = new_tree.get_node(node.path)
            if node.cross_section is None:
                continue
            for cc in node.cross_section.connected_components:
                pos_index, neg_index = cc.get_indices(state)
                pi = cc.positive
                ni = cc.negative
                for idx in pos_index:
                    xform = self.connectors[idx].primitive.transform
                    slot = trimesh.primitives.Box(
                        extents=np.ones(3) * (cc.connector_diameter + config.connector_tolerance),
                        transform=xform)
                    try:
                        utils.trimesh_repair(new_node.children[pi].part)
                        new_part = new_node.children[pi].part.difference(slot, engine='scad')
                        new_node.children[pi].part = new_part

                        utils.trimesh_repair(new_node.children[ni].part)
                        new_part = new_node.children[ni].part.union(self.connectors[idx], engine='scad')
                        new_node.children[ni].part = new_part
                    except Exception as e:
                        logger.info("ignoring connector")
                for idx in neg_index:
                    xform = self.connectors[idx].primitive.transform
                    slot = trimesh.primitives.Box(
                        extents=np.ones(3) * (cc.connector_diameter + config.connector_tolerance),
                        transform=xform)
                    try:
                        utils.trimesh_repair(new_node.children[ni].part)
                        new_part = new_node.children[ni].part.difference(slot, engine='scad')
                        new_node.children[ni].part = new_part

                        utils.trimesh_repair(new_node.children[pi].part)
                        new_part = new_node.children[pi].part.union(self.connectors[idx], engine='scad')
                        new_node.children[pi].part = new_part
                    except Exception as e:
                        logger.info("ignoring connector")

        return new_tree
