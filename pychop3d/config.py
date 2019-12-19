import numpy as np
import trimesh
import json
import os


class Configuration:

    cfg = None

    @classmethod
    def set_configuration(cls, cfg):
        cls.cfg = cfg

    def __init__(self, config_path=None):
        self.do_not_save = ['normals']
        # printer parameters
        self.printer_extents = np.array([200, 200, 200], dtype=float)
        # plane selection parameters
        self.plane_spacing = 20
        self.n_theta = 5
        self.n_phi = 5
        self.normals = self.uniform_normals()
        # plane uniqueness parameters
        self.different_origin_th = .1 * np.sqrt(np.sum(self.printer_extents ** 2))
        self.different_angle_th = np.pi / 10
        # objective parameters
        self.part_weight = 1
        self.utilization_weight = .25
        self.connector_weight = 1
        self.fragility_weight = 1
        self.seam_weight = .1
        self.symmetry_weight = .25
        self.fragility_objective_th = .95
        self.connector_objective_th = 5
        # connector placement parameters
        self.connector_collision_penalty = 10 ** 10
        self.sa_initial_connector_ratio = .1
        self.sa_initialization_iterations = 15_000
        self.sa_iterations = 300_000
        # connector settings
        self.connector_diameter_min = 5
        self.connector_diameter_max = 30
        self.connector_tolerance = 1
        # run settings
        self.mesh = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
        self.directory = "C:\\Users\\Greg\\code\\pychop3d\\debug"
        self.scale = True
        self.beam_width = 2
        # saved data
        self.nodes = None

        # open config JSON
        if config_path is not None:
            with open(config_path) as f:
                config_file = json.load(f)

            for key, value in config_file.items():
                setattr(self, key, value)

    def uniform_normals(self):
        """http://corysimon.github.io/articles/uniformdistn-on-sphere/
        """
        theta = np.arange(0, np.pi, np.pi / self.n_theta)
        phi = np.arccos(1 - np.arccos(1 - np.arange(0, 1, 1 / self.n_phi)))
        theta, phi = np.meshgrid(theta, phi)
        theta = theta.ravel()
        phi = phi.ravel()
        return np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=1)


    def save(self, filename):
        save_data = {}
        for key, value in self.__dict__.items():
            if key in self.do_not_save:
                continue
            if isinstance(value, np.ndarray):
                save_data[key] = [v for v in value]
            else:
                save_data[key] = value
        with open(os.path.join(self.directory, filename), 'w') as f:
            json.dump(save_data, f)


cfg = Configuration()
Configuration.set_configuration(cfg)
