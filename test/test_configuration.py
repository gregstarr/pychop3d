"""
desired features of configuration:
    - once you set the configuration, every module has access
    - modify configuration
    - save configuration
    - load configuration
possible features:
    - convenience method to create temporary directories
    - convenience method to create timestamped directory
TODO:
    - should test a variety of functions that use configuration and verify that the output is different
        when using modified configuration
"""

import trimesh
import numpy as np
import tempfile
import yaml
import os

from pychop3d.configuration import Configuration
from pychop3d import bsp
from pychop3d import bsp_mesh

config = Configuration.config


def test_modify_configuration():
    """Verify that modifying the configuration modifies the behavior of the other modules. Create a tree with the
    default part and the default configuration, verify that it will fit in the printer volume, then modify the
    printer volume in the config and verify that a newly created tree will have a different n_parts objective
    """
    print()
    # open and prepare mesh
    mesh = trimesh.load(config.mesh, validate=True)
    chull = mesh.convex_hull
    mesh = bsp_mesh.BSPMesh.from_trimesh(mesh)
    mesh._convex_hull = chull
    # create bsp tree
    tree = bsp.BSPTree(mesh)
    print(f"n parts: {tree.nodes[0].n_parts}")
    assert tree.nodes[0].n_parts == 1
    config.printer_extents = config.printer_extents / 2
    print("modified config")
    print(f"original tree n parts: {tree.nodes[0].n_parts}")
    assert tree.nodes[0].n_parts == 1
    new_tree = bsp.BSPTree(mesh)
    print(f"new tree n parts: {new_tree.nodes[0].n_parts}")
    assert new_tree.nodes[0].n_parts == 2
    config.restore_defaults()


def test_load():
    """load a non-default parameter from a yaml file and verify that the config object matches
    """
    with tempfile.TemporaryDirectory() as tempdir:
        params = {
            'printer_extents': [1, 2, 3],
            'test_key': 'test_value'
        }
        yaml_path = os.path.join(tempdir, "test.yml")
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(params, f)

        new_config = Configuration(yaml_path)
    assert isinstance(new_config.printer_extents, np.ndarray)
    assert np.all(new_config.printer_extents == np.array([1, 2, 3]))
    assert new_config.test_key == 'test_value'
    assert not hasattr(config, 'test_key')


def test_save():
    """modify the config, save it, verify that the modified values are saved and can be loaded
    """
    config.connector_diameter = 100
    with tempfile.TemporaryDirectory() as tempdir:
        config.directory = tempdir
        path = config.save("test_config.yml")

        new_config = Configuration(path)

    assert new_config.connector_diameter == 100
    config.restore_defaults()
