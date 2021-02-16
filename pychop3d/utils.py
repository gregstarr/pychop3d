import numpy as np
import trimesh
import json
import os
import logging

from pychop3d import bsp_tree
from pychop3d import bsp_node
from pychop3d.objective_functions import evaluate_utilization_objective, evaluate_nparts_objective
from pychop3d.configuration import Configuration
from trimesh.interfaces.blender import _blender_executable, exists
from trimesh.interfaces.generic import MeshScript
from trimesh import repair
from trimesh import util


logger = logging.getLogger(__name__)


def preprocess(mesh, debug=True):
    """
    Run a preprocess operation with mesh using Blender.
    """
    config = Configuration.config

    logger.info("starting preprocessing")
    if not exists:
        raise ValueError('No blender available!')
    curr_dir = os.path.dirname(__file__)
    preprocessor_func_fn = find_file(config.preprocessor)
    with open(os.path.join(curr_dir + "/blender_script_templates/preprocessor.py.template"), 'rb') as preprocessor:
        with open(preprocessor_func_fn, 'rb') as preprocessor_func_file:
            preprocessor_func = preprocessor_func_file.read().decode('utf-8')
            script = preprocessor.read().decode(
                'utf-8').replace("$PREPROCESSOR_FUNC", preprocessor_func)

            with MeshScript(meshes=[mesh], script=script, debug=debug) as blend:
                result = blend.run(_blender_executable + ' --background --python $SCRIPT')
                logger.info("finished preprocessing")

            for m in util.make_sequence(result):
                m.face_normals = None
            return result


def separate_starter(mesh):
    """this function takes in a mesh with more than one body and turns it into a tree who's root node is the original
    mesh and the roots children are each of the bodies of the mesh

    :param mesh: mesh with multiple bodies
    :type mesh: `trimesh.Trimesh`
    :return: tree with the bodies as the root node's children
    :rtype: `bsp_tree.BSPTree`
    """
    logger.info("separating starter mesh")
    parts = mesh.split(only_watertight=False)  # split into separate components
    logger.info(f"starter mesh split into {len(parts)} children")
    tree = bsp_tree.BSPTree(mesh)  # create starter tree
    for i, part in enumerate(parts):
        # make a new node for each separate part
        new_node = bsp_node.BSPNode(part, tree.nodes[0], i)
        # make the new node the root node's child
        tree.nodes[0].children.append(new_node)
        tree.nodes.append(new_node)
    # update nparts and utilization objectives
    evaluate_nparts_objective([tree], tuple())
    evaluate_utilization_objective([tree], tuple())
    return tree


def find_file(fn):
    """Looks for a file either as a relative path off of the pytest directory or as an absolute path

    :param fn: input path, either relative or absolute
    :return: path to file
    :raises: FileNotFoundError
    """
    abs_fn = fn
    rel_fn = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', fn))
    if os.path.isfile(abs_fn):
        return abs_fn
    if os.path.isfile(rel_fn):
        return rel_fn
    raise FileNotFoundError(f"Can't find file: {abs_fn}")


def open_mesh():
    """open the mesh according to the configuration and apply any scaling or subdivision
    """
    logger.info("opening")

    config = Configuration.config
    # OPEN MESH
    mesh_fn = find_file(config.mesh)
    mesh = trimesh.load(mesh_fn)

    if hasattr(config, 'preprocessor') and config.preprocessor:
        mesh = preprocess(mesh)

    # REPAIR MESH
    trimesh_repair(mesh)
    # SCALE MESH
    if config.scale_factor > 0:
        mesh.apply_scale(config.scale_factor)
    # SUBDIVIDE MESH

    return mesh


def open_tree(tree_file):
    mesh = open_mesh()
    with open(tree_file) as f:
        data = json.load(f)

    node_data = data['nodes']
    tree = bsp_tree.BSPTree(mesh)
    for n in node_data:
        plane = (np.array(n['origin']), np.array(n['normal']))
        node = tree.get_node(n['path'])
        tree, result_code = bsp_tree.expand_node(tree, node.path, plane)
    return tree


def load_connector_configuration(tree_file):
    with open(tree_file) as f:
        data = json.load(f)
    return np.array(data['state'])


def all_at_goal(trees):
    """convenience / readability function which returns whether a list of trees are all terminated

    :param trees: list of trees to check
    :type trees: list
    :return: bool indicating if the list of trees are all terminated
    :rtype: bool
    """
    for tree in trees:
        if not tree.terminated:
            return False
    return True


def not_at_goal_set(trees):
    """convenience / readability function which returns the not terminated trees from a list

    :param trees: list of input trees
    :type trees: list
    :return: trees from the input list which are not terminated
    :rtype: list
    """
    not_at_goal = []
    for tree in trees:
        if not tree.terminated:
            not_at_goal.append(tree)
    return not_at_goal


def get_unique_normals(non_unique_normals):
    """get unique normals from an (N x 3) array of normal vectors

    :param non_unique_normals: (N x 3) array of normal vectors
    :type non_unique_normals: `numpy.ndarray`
    :return: unique normals from the input array
    :rtype: `numpy.ndarray`
    """
    # round the vectors to avoid having any of the resulting vectors too close
    rounded = np.round(non_unique_normals, 3)
    # treat the array as a 1-D array of tuples
    view = rounded.view(dtype=[('', float), ('', float), ('', float)])
    unique = np.unique(view)  # get the unique tuples
    # reassemble the tuples into a numpy array
    return unique.view(dtype=float).reshape((unique.shape[0], -1))


def make_plane(origin, normal, w=100):
    xform = np.linalg.inv(trimesh.points.plane_transform(origin, normal))
    box = trimesh.primitives.Box(extents=(w, w, .5), transform=xform)
    return box


def trimesh_repair(mesh):
    repair.fill_holes(mesh)
    repair.fix_winding(mesh)
    repair.fix_inversion(mesh)
    repair.fix_normals(mesh)
    mesh.process()


def preview_tree(tree, other_objects=None):
    if other_objects is None:
        other_objects = []
    scene = trimesh.Scene()
    for leaf in tree.leaves:
        leaf.part.visual.face_colors = np.random.rand(3)*255
        scene.add_geometry(leaf.part)
    for ob in other_objects:
        scene.add_geometry(ob)
    scene.camera.z_far = 10_000
    scene.show()


def save_tree(tree, filename, state=None):
    config = Configuration.config

    if state is None:
        state = []

    nodes = []
    for node in tree.nodes:
        if node.plane is not None:
            this_node = {'path': node.path, 'origin': list(
                node.plane[0]), 'normal': list(node.plane[1])}
            nodes.append(this_node)

    with open(os.path.join(config.directory, filename), 'w') as f:
        json.dump({'nodes': nodes, 'state': [bool(s) for s in state]}, f)


def export_tree_stls(tree, fn_info="part"):
    """Saves all of a tree's parts to the directory specified in the configuration

    :param tree: tree to save the parts of
    :type tree: `bsp_tree.BSPTree`
    :param fn_info: string added to the STL filenames after the object name
    :type fn_info: str
    """
    config = Configuration.config
    for i, leaf in enumerate(tree.leaves):
        leaf.part.export(os.path.join(config.directory,
                                      f"{config.name}_{fn_info}_{i}.stl"))
