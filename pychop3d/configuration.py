import numpy as np
import yaml
import os


class Configuration:
    """
    This class will hold all of the configuration settings for a run. It will not hold any BSPTree data or mesh data.
    The settings will be saved and loaded from YAML files. Maybe later implement this as a singleton if necessary. For
    now the only object of this class will be stored in the class itself.
    """

    config = None

    def __init__(self, config_path=None):
        self.restore_defaults()

        # open config YAML
        if config_path is not None:
            with open(config_path) as f:
                config_file = yaml.safe_load(f)

            for key, value in config_file.items():
                setattr(self, key, value)

        self.printer_extents = np.array(self.printer_extents, dtype=float)

    def restore_defaults(self):
        self.do_not_save = ['normals']
        # printer parameters
        self.printer_extents = np.array([200, 200, 200], dtype=float)
        # plane selection parameters
        self.plane_spacing = 20
        self.n_theta = 5
        self.n_phi = 5
        self.normals = self.uniform_normals()
        # plane uniqueness parameters
        self.different_origin_th = float(.1 * np.sqrt(np.sum(self.printer_extents ** 2)))
        self.different_angle_th = np.pi / 10
        # objective parameters
        self.objective_weights = {
            'part': 1,
            'utilization': .25,
            'connector': 1,
            'fragility': 1,
            'seam': 0,  # set to zero until implemented
            'symmetry': 0  # set to zero until implemented
        }
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
        self.connector_diameter = 10
        self.connector_tolerance = 1
        # run settings
        self.mesh = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
        self.directory = "C:\\Users\\Greg\\code\\pychop3d\\debug"
        self.scale = True
        self.beam_width = 2

    def uniform_normals(self):
        """http://corysimon.github.io/articles/uniformdistn-on-sphere/
        """
        theta = np.arange(0, np.pi, np.pi / self.n_theta)
        phi = np.arccos(np.arange(0, 1, 1 / self.n_phi))
        theta, phi = np.meshgrid(theta, phi)
        theta = theta.ravel()
        phi = phi.ravel()
        return np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=1)

    def save(self, filename):
        """saves the config file

         from pyyaml docs https://pyyaml.org/wiki/PyYAMLDocumentation:
            safe_dump_all produces only standard YAML tags and cannot represent an arbitrary Python object.

        therefore the save data will convert any numpy array to list first
        """
        save_data = {}
        for key, value in self.__dict__.items():
            if key in self.do_not_save:
                continue
            if isinstance(value, np.ndarray):
                save_data[key] = [float(v) for v in value]
            else:
                save_data[key] = value

        save_path = os.path.join(self.directory, filename)
        with open(save_path, 'w') as f:
            yaml.dump(save_data, f)

        return save_path


config = Configuration()
Configuration.config = config