import trimesh
import trimesh
import numpy as np
import copy
import itertools
import json
import os

from pychop3d import constants
from pychop3d import utils
from pychop3d import bsp_mesh
from pychop3d.configuration import Configuration


class BSPNode:

    def __init__(self, part, parent=None, num=None):
        config = Configuration.config
        self.part = part
        self.parent = parent
        self.children = {}
        self.path = []
        self.plane = None
        self.connector_data = None
        self.n_parts = np.prod(np.ceil(self.get_bounding_box_oriented().primitive.extents / config.printer_extents))
        self.terminated = np.all(self.get_bounding_box_oriented().primitive.extents < config.printer_extents)
        # if this isn't the root node
        if self.parent is not None:
            self.path = self.parent.path + [num]

    def split(self, plane):
        part = self.part
        self.plane = plane
        origin, normal = plane
        positive, negative, connector_data = utils.bidirectional_split(part, origin, normal)
        # check for splitting errors
        if None in [positive, negative]:
            return False
        pch, nch = utils.bidirectional_split(part.convex_hull, origin, normal, get_connections=False)
        positive = bsp_mesh.BSPMesh.from_trimesh(positive)
        positive._convex_hull = pch
        negative = bsp_mesh.BSPMesh.from_trimesh(negative)
        negative._convex_hull = nch
        self.connector_data = connector_data
        self.children[0] = BSPNode(positive, parent=self, num=0)
        self.children[1] = BSPNode(negative, parent=self, num=1)
        return True

    def get_bounding_box_oriented(self):
        try:
            return self.part.bounding_box_oriented
        except Exception as e:
            print(e)
            print("REGULAR CONVEX HULL FAILED, USING APPROXIMATION")
            samples, *_ = self.part.nearest.on_surface(constants.CH_SAMPLE_POINTS)
            point_cloud = trimesh.PointCloud(samples)
            return point_cloud.bounding_box_oriented

    def auxiliary_normals(self):
        obb_xform = np.array(self.get_bounding_box_oriented().primitive.transform)
        return obb_xform[:3, :3]

    def get_planes(self, normal):
        config = Configuration.config
        projection = self.part.vertices @ normal
        limits = [projection.min(), projection.max()]
        planes = [(d * normal, normal) for d in np.arange(limits[0], limits[1], config.plane_spacing)][1:]
        return planes

    def different_from(self, other_node, threshold):
        plane_transform = trimesh.points.plane_transform(*self.plane)
        o = other_node.plane[0]
        op = np.array([o[0], o[1], o[2], 1], dtype=float)
        op = (plane_transform @ op)[:3]
        return np.sqrt(np.sum(op ** 2)) > threshold

    def get_connection_objective(self):
        return max([cc['objective'] for cc in self.connector_data])


class BSPTree:

    def __init__(self, part: trimesh.Trimesh):
        config = Configuration.config
        self.nodes = [BSPNode(part)]
        self.a_part = config.objective_weights['part']
        self.a_util = config.objective_weights['utilization']
        self.a_connector = config.objective_weights['connector']
        self.a_fragility = config.objective_weights['fragility']
        self.a_seam = config.objective_weights['seam']
        self.a_symmetry = config.objective_weights['symmetry']

    def get_node(self, path=None):
        node = self.nodes[0]
        if not path:
            return node
        else:
            for i in path:
                node = node.children[i]
        return node

    def expand_node(self, plane, node):
        new_tree = copy.deepcopy(self)
        node = new_tree.get_node(node.path)
        success = node.split(plane)
        new_tree.nodes += list(node.children.values())
        if success:
            return new_tree
        return None

    def get_leaves(self):
        nodes = [self.nodes[0]]
        leaves = []
        while nodes:
            node = nodes.pop()
            if node.children == {}:
                leaves.append(node)
            else:
                nodes += list(node.children.values())
        return leaves

    def terminated(self):
        leaves = self.get_leaves()
        for leaf in leaves:
            if not leaf.terminated:
                return False
        return True

    def largest_part(self):
        return sorted(self.get_leaves(), key=lambda x: x.n_parts)[-1]

    def sufficiently_different(self, node, tree_set, threshold=constants.SUFFICIENTLY_DIFFERENT_THRESHOLD):
        if not tree_set:
            return True
        for tree in tree_set:
            if not self.different_from(tree, node, threshold):
                return False
        return True

    def different_from(self, tree, node, threshold):
        self_node = self.get_node(node.path)
        other_node = tree.get_node(node.path)
        return self_node.different_from(other_node, threshold)

    def nparts_objective(self):
        theta_0 = self.nodes[0].n_parts
        return sum([l.n_parts for l in self.get_leaves()]) / theta_0

    def utilization_objective(self):
        config = Configuration.config
        V = np.prod(config.printer_extents)
        return max([1 - leaf.part.volume / (leaf.n_parts * V) for leaf in self.get_leaves()])

    def connector_objective(self):
        """
        for each leaf, for each connected component of the leaf's cut face, set up a grid of points and check each point
        if a female connector could feasibly fit at that location, then take the convex hull of those points for each
        cut face and compute the quantity in equation 4. Take the maximum of all of these
        """
        return max([n.get_connection_objective() for n in self.nodes if n.connector_data is not None])

    def fragility_objective(self):
        config = Configuration.config
        leaves = self.get_leaves()
        nodes = {}
        for leaf in leaves:
            path = tuple(leaf.parent.path)
            if path not in nodes:
                nodes[path] = leaf.parent

        for node in nodes.values():
            origin, normal = node.plane
            mesh = node.part
            possibly_fragile = np.abs(mesh.vertex_normals @ normal) > config.fragility_objective_th
            if not np.any(possibly_fragile):
                continue
            ray_origins = mesh.vertices[possibly_fragile] - .1 * mesh.vertex_normals[possibly_fragile]
            distance_to_plane = np.abs((ray_origins - origin) @ normal)
            side_mask = (ray_origins - origin) @ normal > 0
            ray_directions = np.ones((ray_origins.shape[0], 1)) * normal[None, :]
            ray_directions[side_mask] *= -1
            hits = mesh.ray.intersects_any(ray_origins, ray_directions)
            if np.any(distance_to_plane[~hits] < 1.5 * config.connector_diameter):
                return np.inf
            if not np.any(hits):
                continue
            locs, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, ray_directions)
            ray_mesh_dist = np.sqrt(np.sum((ray_origins[index_ray] - locs) ** 2, axis=1))
            if np.any((distance_to_plane[index_ray] < 1.5 * config.connector_diameter) *
                      (distance_to_plane[index_ray] < ray_mesh_dist)):
                return np.inf
        return 0

    def seam_objective(self):
        return 0

    def symmetry_objective(self):
        return 0

    def get_objective(self):
        part = self.a_part * self.nparts_objective()
        util = self.a_util * self.utilization_objective()
        connector = self.a_connector * self.connector_objective()
        fragility = self.a_fragility * self.fragility_objective()
        seam = self.a_seam * self.seam_objective()
        symmetry = self.a_symmetry * self.symmetry_objective()
        return part + util + connector + fragility + seam + symmetry

    def preview(self):
        scene = trimesh.scene.Scene()
        for leaf in self.get_leaves():
            leaf.part.visual.face_colors = np.random.rand(3)*255
            scene.add_geometry(leaf.part)
        scene.show()

    def setup_connector_placement(self):
        config = Configuration.config
        self.connected_component = []
        self.sites = []
        self.normals = []
        self.sides = []
        self.connected_component_nodes = []
        self.connected_component_areas = []
        cci = 0
        for n, node in enumerate(self.nodes):
            if node.connector_data is None:
                continue
            for cc in node.connector_data:
                self.connected_component_nodes.append(n)
                self.connected_component.append(np.ones(cc['sites'].shape[0], dtype=int) * cci)
                self.sites.append(cc['sites'])
                self.normals.append(cc['normals'])
                self.sides.append(cc['side'])
                self.connected_component_areas.append(cc['area'])
                cci += 1
        self.connected_component = np.concatenate(self.connected_component, axis=0)
        self.sites = np.concatenate(self.sites, axis=0)
        self.normals = np.concatenate(self.normals, axis=0)
        self.sides = np.concatenate(self.sides, axis=0)
        self.connected_component_nodes = np.array(self.connected_component_nodes)
        connectors = []
        for site, normal in zip(self.sites, self.normals):
            box = trimesh.primitives.Box(extents=np.ones(3) * config.connector_diameter)
            box.apply_transform(np.linalg.inv(trimesh.points.plane_transform(site, normal)))
            box.apply_transform(trimesh.transformations.translation_matrix(normal * (config.connector_diameter / 2 - .1)))
            connectors.append(box)
        self.connectors = np.array(connectors)
        self.connection_matrix = np.zeros((self.sites.shape[0], self.sites.shape[0]), dtype=bool)
        for i, j in itertools.combinations(range(self.sites.shape[0]), 2):
            a = self.connectors[i]
            b = self.connectors[j]
            if (np.any(a.contains(b.vertices + .1 * np.random.rand(3))) or
                    np.any(b.contains(a.vertices + np.random.rand(3) * .1))):
                self.connection_matrix[i, j] = True

    def evaluate_connector_objective(self, state):
        config = Configuration.config
        objective = 0
        n_collisions = self.connection_matrix[state, :][:, state].sum()
        objective += config.connector_collision_penalty * n_collisions

        for cci in range(len(self.connected_component_areas)):
            comp_area = self.connected_component_areas[cci]
            rc = min(np.sqrt(comp_area) / 2, 10 * config.connector_diameter)
            mask = (self.connected_component == cci) * state
            ci = 0
            if mask.sum():
                sites = self.sites[mask]
                ci = mask.sum() * np.pi * rc**2
                for i, j in itertools.combinations(range(mask.sum()), 2):
                    d = np.sqrt(((sites[i, :3] - sites[j, :3]) ** 2).sum())
                    if d < 2 * rc:
                        ci -= 2 * np.pi * (rc - d/2) ** 2
            objective += comp_area / (10**-5 + max(0, ci))
            if ci < 0:
                objective -= ci / comp_area

        return objective

    def simulated_annealing_connector_placement(self):
        config = Configuration.config
        self.setup_connector_placement()
        state = np.random.rand(self.sites.shape[0]) > (1 - config.sa_initial_connector_ratio)
        objective = self.evaluate_connector_objective(state)
        print(f"initial objective: {objective}")
        # initialization
        for i in range(config.sa_initialization_iterations):
            if not i % (config.sa_initialization_iterations // 10):
                print('.', end='')
            state, objective = self.SA_iteration(state, objective, 0)

        print(f"\npost initialization objective: {objective}")
        initial_temp = objective / 2
        for i, temp in enumerate(np.linspace(initial_temp, 0, config.sa_iterations)):
            if not i % (config.sa_iterations // 10):
                print('.', end='')
            state, objective = self.SA_iteration(state, objective, temp)

        print(f"\nfinal objective: {objective}")
        return state

    def SA_iteration(self, state, objective, temp):
        new_state = state.copy()
        if np.random.randint(0, 2):
            e = np.random.randint(0, self.sites.shape[0])
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

    def save(self, filename, state):
        nodes = []
        for node in self.nodes:
            if node.plane is not None:
                this_node = {'path': node.path, 'origin': list(node.plane[0]), 'normal': list(node.plane[1])}
                nodes.append(this_node)
        with open(filename, 'w') as f:
            json.dump({'nodes': nodes, 'state': [bool(s) for s in state]}, f)

    @classmethod
    def from_json(cls, mesh, filename):
        with open(filename) as f:
            data = json.load(f)

        node_data = data['nodes']
        tree = cls(mesh)
        for n in node_data:
            plane = (np.array(n['origin']), np.array(n['normal']))
            node = tree.get_node(n['path'])
            tree = tree.expand_node(plane, node)
        return tree, np.array(data['state'], dtype=bool)

    def export_stl(self, dirname):
        for i, leaf in enumerate(self.get_leaves()):
            leaf.part.export(os.path.join(dirname, f"{i}.stl"))

