import numpy as np
from trimesh import Trimesh
from trimesh.points import plane_transform
from trimesh.primitives import Sphere

from pychop3d import settings, utils
from pychop3d.bsp_tree import BSPTree, expand_node
from pychop3d.logger import logger


def match_connectors(original_tree: BSPTree, tree: BSPTree):
    """copies connector info from one tree to another (in-place)

    Args:
        original_tree (BSPTree): tree to copy from
        tree (BSPTree): tree to copy to
    """
    for node in tree.nodes:
        if node.cross_section is None:
            continue
        original_node = original_tree.get_node(node.path)
        # create global vector of connectors
        for j, cc in enumerate(node.cross_section.connected_components):
            original_node.cross_section.connected_components[j].index = cc.index
            original_node.cross_section.connected_components[j].sites = cc.sites


class ConnectorPlacer:
    """Manages optimization and placement of connectors"""

    def __init__(self, tree: BSPTree):
        self.connected_components = []
        self.connectors = []
        self.n_connectors = 0
        caps = []
        if len(tree.nodes) < 2:
            raise Exception("input tree needs to have a chop")

        CD = settings.CONNECTOR_DIAMETER
        CT = settings.CONNECTOR_TOLERANCE
        logger.info("Creating connectors...")
        for n, node in enumerate(tree.nodes):
            logger.info("$CONNECTOR_PROGRESS %s", 0.1 * n / len(tree.nodes))
            if node.cross_section is None:
                continue
            # create global vector of connectors
            for cc in node.cross_section.connected_components:
                caps.append(cc.mesh)
                # register the ConnectedComponent's sites with the global array of
                # connectors
                cc.register_sites(len(self.connectors))
                self.connected_components.append(cc)
                for site in cc.sites:
                    conn_m = Sphere(radius=CD / 2)
                    conn_m.apply_transform(
                        np.linalg.inv(plane_transform(site, cc.plane.normal))
                    )
                    self.connectors.append(conn_m)

        self.connectors = np.array(self.connectors)
        self.n_connectors = self.connectors.shape[0]
        if self.n_connectors == 0:
            return
        self.collisions = np.zeros((self.n_connectors, self.n_connectors), dtype=bool)

        logger.info("determining connector-cut intersections")
        mass_centers = np.array([c.center_mass for c in self.connectors])
        intersections = np.zeros(self.n_connectors, dtype=int)
        for cap in caps:
            mask = cap.nearest.on_surface(mass_centers)[1] < CD
            intersections[mask] += 1
        self.collisions[intersections > 1, :] = True
        self.collisions[:, intersections > 1] = True

        logger.info("determining connector mesh protrusions")
        for n, node in enumerate(tree.nodes):
            logger.info("$CONNECTOR_PROGRESS %s", 0.1 + 0.1 * n / len(tree.nodes))
            if node.cross_section is None:
                continue
            origin, normal = node.plane
            origin -= normal * 0.1
            for cc in node.cross_section.connected_components:
                for idx in cc.index:
                    part = node.children[cc.negative].part
                    xform = self.connectors[idx].primitive.transform
                    conn_f = Sphere(radius=(CD + CT + 1) / 2, transform=xform)
                    sliced_conn_f = conn_f.slice_plane(origin, -1 * normal)
                    v_check = part.contains(sliced_conn_f.vertices)
                    protrude = not v_check.all()
                    if protrude:
                        self.collisions[idx, :] = True
                        self.collisions[:, idx] = True

        logger.info("determining connector-connector intersections")
        distances = np.sqrt(
            np.sum((mass_centers[:, None, :] - mass_centers[None, :, :]) ** 2, axis=2)
        )
        mask = (distances > 0) & (distances < CD)
        self.collisions = np.logical_or(self.collisions, mask)

    def evaluate_connector_objective(self, state: np.ndarray) -> float:
        """evaluates connector objective given a state vector"""
        objective = 0
        n_collisions = self.collisions[state, :][:, state].sum()
        objective += settings.CONNECTOR_COLLISION_PENALTY * n_collisions

        for cc in self.connected_components:
            rc = min(np.sqrt(cc.area) / 2, 10 * settings.CONNECTOR_DIAMETER)
            sites = cc.get_sites(state)
            ci = 0
            if len(sites) > 0:
                ci = len(sites) * np.pi * rc**2
                distances = np.sqrt(
                    np.sum((sites[:, None, :] - sites[None, :, :]) ** 2, axis=2)
                )
                mask = (0 < distances) * (distances < 2 * rc)
                ci -= np.sum(np.pi * (rc - distances[mask] / 2) ** 2)
            objective += cc.area / (settings.EMPTY_CC_PENALTY + max(0, ci))
            if ci < 0:
                objective -= ci / cc.area

        return objective

    def get_initial_state(self) -> np.ndarray:
        """gets initial random state vector, enables one connector for each connected
        component"""
        state = np.zeros(self.n_connectors, dtype=bool)
        for cc in self.connected_components:
            i = np.random.choice(cc.index)
            state[i] = True
        return state

    def simulated_annealing_connector_placement(self) -> np.ndarray:
        """run simulated annealing to optimize placement of connectors"""
        state = self.get_initial_state()
        objective = self.evaluate_connector_objective(state)
        logger.info("initial objective: %s", objective)
        # initialization
        for i in range(settings.SA_INITIALIZATION_ITERATIONS):
            state, objective = self.sa_iteration(state, objective, 0)
        logger.info("$CONNECTOR_PROGRESS .3")
        logger.info("post initialization objective: %s", objective)
        initial_temp = objective / 2
        for i, temp in enumerate(np.linspace(initial_temp, 0, settings.SA_ITERATIONS)):
            if (i % (settings.SA_ITERATIONS // 20)) == 0:
                logger.info(
                    "$CONNECTOR_PROGRESS %s", 0.3 + 0.3 * i / settings.SA_ITERATIONS
                )
            state, objective = self.sa_iteration(state, objective, temp)
        logger.info("$CONNECTOR_PROGRESS 1")
        return state

    def sa_iteration(
        self, state: np.ndarray, objective: float, temp: float
    ) -> tuple[np.ndarray, float]:
        """run a single simulated annealing iteration"""
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

    def insert_connectors(
        self, tree: BSPTree, state: np.ndarray, printer_extents: np.ndarray
    ) -> BSPTree:
        """Insert connectors into a tree according to a state vector

        Args:
            tree (BSPTree): tree
            state (np.ndarray): state vector
            printer_extents (np.ndarray): printer dims (mm)

        Returns:
            BSPTree: tree but all the part meshes have connectors
        """
        NI = state.sum()
        CD = settings.CONNECTOR_DIAMETER
        CT = settings.CONNECTOR_TOLERANCE
        nc = 0
        if tree.nodes[0].plane is None:
            new_tree = utils.separate_starter(tree.nodes[0].part, printer_extents)
        else:
            new_tree = BSPTree(tree.nodes[0].part, printer_extents)
        for node in tree.nodes:
            if node.plane is None:
                continue
            new_tree2, result = expand_node(new_tree, node.path, node.plane)
            if result != "success":
                # for debugging
                expand_node(new_tree, node.path, node.plane)
            else:
                new_tree = new_tree2
            new_node = new_tree.get_node(node.path)
            if node.cross_section is None:
                continue
            origin, normal = node.plane
            origin += normal * 0.1
            for cc in node.cross_section.connected_components:
                index = cc.get_indices(state)
                pi = cc.positive
                ni = cc.negative
                for idx in index:
                    nc += 1
                    logger.info("$CONNECTOR_PROGRESS %s", 0.6 + 0.4 * nc / NI)
                    xform = self.connectors[idx].primitive.transform
                    conn_m = self.connectors[idx].slice_plane(origin, -1 * normal)
                    conn_f = Sphere(radius=(CD + CT) / 2, transform=xform)
                    new_node.children[ni].part = insert_connector(
                        new_node.children[ni].part, conn_f, "difference"
                    )
                    new_node.children[pi].part = insert_connector(
                        new_node.children[pi].part, conn_m, "union"
                    )
        return new_tree


def insert_connector(part: Trimesh, connector: Trimesh, operation: str, retries=10):
    """Adds a box / connector to a part using boolean union. Operating under the
    assumption that adding a connector MUST INCREASE the number of vertices of the
    resulting part.

    Args:
        part (Trimesh): part to add connector to
        connector (Trimesh): connector to add to part
        operation (str): 'union' or 'difference'
        retries (int, optional): number of times to retry before raising an error,
            checking to see if number of vertices increases. Defaults to 10.

    Raises:
        Exception: retries exceeded
        ValueError: incorrect operation input

    Returns:
        Trimesh: part with connector inserted
    """
    if operation not in ["union", "difference"]:
        raise ValueError("operation not 'union' or 'difference'")
    utils.trimesh_repair(part)
    for _ in range(retries):
        if operation == "union":
            new_part = part.union(connector)
        else:
            new_part = part.difference(connector)
        if len(new_part.vertices) > len(part.vertices):
            return new_part
    raise Exception("Couldn't insert connector")
