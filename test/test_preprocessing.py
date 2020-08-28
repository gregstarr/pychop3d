import trimesh
import numpy as np
import copy
import os
from pychop3d.configuration import Configuration
from pychop3d import utils
import math

def test_preprocessing_decimate():
    """verify that for preprocessing works with sample Decimate Blender modifier
    """

    Configuration.config = Configuration(os.path.join(os.path.dirname(
        __file__), 'test_data', "preprocessor_config_1.yml"))

    config = Configuration.config
    mesh = trimesh.load(config.mesh, validate=False)
    preprocess_mesh = utils.preprocess(mesh)
    (oldFaces, _) = mesh.faces.shape
    (newFaces, _) = preprocess_mesh.faces.shape

    newFacesExpected = math.floor(oldFaces*0.1) # 8032, decimation -> 803
    newFacesActual = math.floor(newFaces) # decimation -> 802

    assert abs(newFacesExpected - newFacesActual) < 10


test_preprocessing_decimate()
