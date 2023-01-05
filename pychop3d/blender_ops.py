import pathlib

from trimesh.interfaces.blender import _blender_executable, exists
from trimesh.interfaces.generic import MeshScript
from trimesh import util
from pychop3d.logger import logger


def run_blender_op(mesh, func_str, debug=True):
    """
    Run a preprocess operation with mesh using Blender.
    """
    logger.info("starting preprocessing")
    if not exists:
        raise ValueError('No blender available!')

    curr_dir = pathlib.Path(__file__).parent
    script_fn = curr_dir / "blender_template.py.template"
    script = script_fn.read_text().replace("$FUNC", func_str)

    with MeshScript(meshes=[mesh], script=script, debug=debug) as blend:
        result = blend.run(_blender_executable + ' --background --python $SCRIPT')
    logger.info("finished preprocessing")

    for m in util.make_sequence(result):
        m.face_normals = None
    return result


_decimate_str = """
def FUNC(mesh):
    modifier = mesh.modifiers.new('decimate', 'DECIMATE')
    modifier.ratio={ratio}
    modifier.use_collapse_triangulate=True
"""
def decimate(mesh, ratio, debug=True):
    result = run_blender_op(mesh, _decimate_str.format(ratio=ratio), debug)
    return result
