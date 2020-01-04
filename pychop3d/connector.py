import numpy as np
import itertools
import trimesh

from pychop3d.configuration import Configuration
from pychop3d import bsp
from pychop3d import utils


class ConnectorPlacer:

    def __init__(self, tree):
        self.connected_components = []
        self.connectors = []
        self.n_connectors = 0
        caps = []
        if len(tree.nodes) < 2:
            raise Exception("input tree needs to have a chop")
        for n, node in enumerate(tree.nodes):
            if node.cross_section is None:
                continue
            for cc in node.cross_section.connected_components:
                caps.append(cc.mesh)
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
        self.collisions = np.zeros((self.n_connectors, self.n_connectors), dtype=bool)

        print("determining connector-cut intersections")
        for i, connector in enumerate(self.connectors):
            print('.', end='')
            th = connector.primitive.extents[0] / 2
            intersections = 0
            for m in caps:
                if m.nearest.on_surface(connector.vertices)[1].min() > th:
                    continue
                try:
                    m.intersection(connector, engine='scad')
                    intersections += 1
                except Exception as e:
                    continue
            if intersections != 1:
                self.collisions[i, :] = True
                self.collisions[:, i] = True
        print("\ndetermining connector-connector intersections")
        for i, j in itertools.combinations(range(self.n_connectors), 2):
            a = self.connectors[i]
            b = self.connectors[j]
            if (np.any(a.contains(b.vertices + .1 * np.random.rand(3))) or
                    np.any(b.contains(a.vertices + np.random.rand(3) * .1))):
                self.collisions[i, j] = True

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
                ci = len(sites) * np.pi * rc**2
                for i, j in itertools.combinations(range(len(sites)), 2):
                    d = np.sqrt(((sites[i, :3] - sites[j, :3]) ** 2).sum())
                    if d < 2 * rc:
                        ci -= 2 * np.pi * (rc - d/2) ** 2
            objective += cc.area / (config.empty_cc_penalty + max(0, ci))
            if ci < 0:
                objective -= ci / cc.area

        return objective

    def simulated_annealing_connector_placement(self):
        config = Configuration.config
        state = np.random.rand(self.n_connectors) > (1 - config.sa_initial_connector_ratio)
        objective = self.evaluate_connector_objective(state)
        print(f"initial objective: {objective}")
        # initialization
        for i in range(config.sa_initialization_iterations):
            if not i % (config.sa_initialization_iterations // 10):
                print('.', end='')
            state, objective = self.sa_iteration(state, objective, 0)

        print(f"\npost initialization objective: {objective}")
        initial_temp = objective / 2
        for i, temp in enumerate(np.linspace(initial_temp, 0, config.sa_iterations)):
            if not i % (config.sa_iterations // 10):
                print('.', end='')
            state, objective = self.sa_iteration(state, objective, temp)

        print(f"\nfinal objective: {objective}")
        return state

    def sa_iteration(self, state, objective, temp):
        new_state = state.copy()
        if np.random.randint(0, 2):
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
        new_tree = bsp.BSPTree(tree.nodes[0].part)
        for node in tree.nodes:
            if node.plane is None:
                continue
            new_tree = new_tree.expand_node(node.plane, node)
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
                        new_node.children[pi].part = new_node.children[pi].part.difference(slot, engine='scad')
                        utils.trimesh_repair(new_node.children[ni].part)
                        new_node.children[ni].part = new_node.children[ni].part.union(self.connectors[idx], engine='scad')
                    except Exception as e:
                        utils.trimesh_repair(new_node.children[pi].part)
                        new_node.children[pi].part = new_node.children[pi].part.difference(slot, engine='scad')
                        utils.trimesh_repair(new_node.children[ni].part)
                        new_node.children[ni].part = new_node.children[ni].part.union(self.connectors[idx],
                                                                                      engine='scad')
                for idx in neg_index:
                    xform = self.connectors[idx].primitive.transform
                    slot = trimesh.primitives.Box(
                        extents=np.ones(3) * (cc.connector_diameter + config.connector_tolerance),
                        transform=xform)
                    try:
                        utils.trimesh_repair(new_node.children[ni].part)
                        new_node.children[ni].part = new_node.children[ni].part.difference(slot, engine='scad')
                        utils.trimesh_repair(new_node.children[pi].part)
                        new_node.children[pi].part = new_node.children[pi].part.union(self.connectors[idx], engine='scad')
                    except Exception as e:
                        utils.trimesh_repair(new_node.children[ni].part)
                        new_node.children[ni].part = new_node.children[ni].part.difference(slot, engine='scad')
                        utils.trimesh_repair(new_node.children[pi].part)
                        new_node.children[pi].part = new_node.children[pi].part.union(self.connectors[idx],
                                                                                      engine='scad')

        return new_tree
