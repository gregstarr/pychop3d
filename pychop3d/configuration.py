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
            self.load(config_path)

    def load(self, path):
        self.directory = os.path.dirname(path)
        with open(path) as f:
            config_file = yaml.safe_load(f)
        for key, value in config_file.items():
            setattr(self, key, value)
        self.printer_extents = np.array(self.printer_extents, dtype=float)

    @property
    def n_theta(self):
        return self._n_theta

    @n_theta.setter
    def n_theta(self, value):
        self._n_theta = value
        self.normals = self.uniform_normals()

    @property
    def n_phi(self):
        return self._n_phi

    @n_phi.setter
    def n_phi(self, value):
        self._n_phi = value
        self.normals = self.uniform_normals()

    def restore_defaults(self):
        self.name = "chopped"
        self.do_not_save = ['normals']
        # printer parameters
        self.printer_extents = np.array([200, 200, 200], dtype=float)
        # plane selection parameters
        self.plane_spacing = 20
        self._n_theta = 5
        self._n_phi = 5
        self.normals = self.uniform_normals()
        self.add_middle_plane = True
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
        self.connector_objective_th = 10
        self.obb_utilization = False
        # connector placement parameters
        self.connector_collision_penalty = 10 ** 10
        self.empty_cc_penalty = 10**-5
        self.sa_initial_connector_ratio = .1
        self.sa_initialization_iterations = 10_000
        self.sa_iterations = 300_000
        # connector settings
        self.connector_diameter_min = 5
        self.connector_diameter_max = 30
        self.connector_diameter = 5
        self.connector_tolerance = 1
        self.connector_spacing = 10
        # run settings
        self.mesh = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test', 'test_meshes', 'Bunny-LowPoly.stl'))
        self._directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'debug'))
        self.save_path = os.path.abspath(os.path.join(self.directory, 'config.yml'))
        self.scale_factor = -1
        self.beam_width = 5
        self.part_separation = False

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, value):
        self._directory = value
        save_name = os.path.basename(self.save_path)
        self.save_path = os.path.abspath(os.path.join(self.directory, save_name))

    def uniform_normals(self):
        """http://corysimon.github.io/articles/uniformdistn-on-sphere/
        """
        theta = np.arange(0, np.pi, np.pi / self.n_theta)
        phi = np.arccos(np.arange(0, 1, 1 / self.n_phi))
        theta, phi = np.meshgrid(theta, phi)
        theta = theta.ravel()
        phi = phi.ravel()
        return np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=1)

    def save(self, filename=None):
        """saves the config file

         from pyyaml docs https://pyyaml.org/wiki/PyYAMLDocumentation:
            safe_dump_all produces only standard YAML tags and cannot represent an arbitrary Python object.

        therefore the save data will convert any numpy array to list first
        """
        if filename is not None:
            self.save_path = os.path.abspath(os.path.join(self.directory, filename))

        save_data = {}
        for key, value in self.__dict__.items():
            if key in self.do_not_save:
                continue
            if isinstance(value, np.ndarray):
                save_data[key] = [float(v) for v in value]
            else:
                save_data[key] = value

        with open(self.save_path, 'w') as f:
            yaml.dump(save_data, f)

        return self.save_path


config = Configuration()
Configuration.config = config
