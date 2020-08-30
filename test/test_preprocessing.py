import trimesh
import numpy as np
import copy
import os
from pychop3d.configuration import Configuration
from pychop3d import utils
import math


def test_preprocessing_decimate(config):
    """verify that for preprocessing works with sample Decimate Blender modifier
    """
    # load wing config, doesn't have preprocessor configured
    config.load(os.path.join(os.path.dirname(__file__), 'test_data', "preprocessor_config_1.yml"))
    mesh = utils.open_mesh()  # load original mesh
    # configure the preprocessor
    config.preprocessor = "test/test_data/decimate_preprocessor.py.template"
    preprocess_mesh = utils.preprocess(mesh)  # preprocess the mesh

    # compare original mesh with preprocessed one
    (oldFaces, _) = mesh.faces.shape
    (newFaces, _) = preprocess_mesh.faces.shape
    newFacesExpected = math.floor(oldFaces*0.1)  # 8032, decimation -> 803
    newFacesActual = math.floor(newFaces)  # decimation -> 802
    assert abs(newFacesExpected - newFacesActual) < 10
