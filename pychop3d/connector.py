import numpy as np
import itertools
import trimesh

from pychop3d import constants
from pychop3d import bsp


class ConnectorPlacer:

    def __init__(self, tree):
        self.connected_components = []
        self.connectors = []
        self.n_connectors = 0
        cross_section_meshes = []
        for n, node in enumerate(tree.nodes):
            if node.cross_section is None:
                continue
            cross_section_meshes.append(node.cross_section.mesh)
            for cc in node.cross_section.connected_components:
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

        for i, connector in enumerate(self.connectors):
            intersections = 0
            for m in cross_section_meshes:
                try:
                    m.intersection(connector, engine='scad')
                    intersections += 1
                except Exception as e:
                    pass
            if intersections != 1:
                self.collisions[i, :] = True
                self.collisions[:, i] = True
        for i, j in itertools.combinations(range(self.n_connectors), 2):
            a = self.connectors[i]
            b = self.connectors[j]
            if (np.any(a.contains(b.vertices + .1 * np.random.rand(3))) or
                    np.any(b.contains(a.vertices + np.random.rand(3) * .1))):
                self.collisions[i, j] = True

    def evaluate_connector_objective(self, state):
        objective = 0
        n_collisions = self.collisions[state, :][:, state].sum()
        objective += constants.CONNECTOR_COLLISION_WEIGHT * n_collisions

        for cc in self.connected_components:
            rc = min(np.sqrt(cc.area) / 2, 10 * cc.connector_diameter)
            sites = cc.get_sites(state)
            ci = 0
            if sites.size > 0:
                ci = len(sites) * np.pi * rc**2
                for i, j in itertools.combinations(range(len(sites)), 2):
                    d = np.sqrt(((sites[i, :3] - sites[j, :3]) ** 2).sum())
                    if d < 2 * rc:
                        ci -= 2 * np.pi * (rc - d/2) ** 2
            objective += cc.area / (10**-5 + max(0, ci))
            if ci < 0:
                objective -= ci / cc.area

        return objective

    def simulated_annealing_connector_placement(self):
        state = np.random.rand(self.n_connectors) > (1 - constants.INITIAL_CONNECTOR_RATIO)
        objective = self.evaluate_connector_objective(state)
        print(f"initial objective: {objective}")
        # initialization
        for i in range(constants.INITIALIZATION_ITERATIONS):
            if not i % (constants.INITIALIZATION_ITERATIONS // 10):
                print('.', end='')
            state, objective = self.sa_iteration(state, objective, 0)

        print(f"\npost initialization objective: {objective}")
        initial_temp = objective / 2
        for i, temp in enumerate(np.linspace(initial_temp, 0, constants.ANNEALING_ITERATIONS)):
            if not i % (constants.ANNEALING_ITERATIONS // 10):
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
                for idx in pos_index:
                    xform = self.connectors[idx].primitive.transform
                    slot = trimesh.primitives.Box(
                        extents=np.ones(3) * (cc.connector_diameter + constants.CONNECTOR_TOLERANCE),
                        transform=xform)
                    new_node.children[0].part = new_node.children[0].part.difference(slot, engine='scad')
                    new_node.children[1].part = new_node.children[1].part.union(self.connectors[idx], engine='scad')
                for idx in neg_index:
                    xform = self.connectors[idx].primitive.transform
                    slot = trimesh.primitives.Box(
                        extents=np.ones(3) * (cc.connector_diameter + constants.CONNECTOR_TOLERANCE),
                        transform=xform)
                    new_node.children[1].part = new_node.children[1].part.difference(slot, engine='scad')
                    new_node.children[0].part = new_node.children[0].part.union(self.connectors[idx], engine='scad')

        return new_tree
