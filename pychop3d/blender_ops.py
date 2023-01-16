"""operations on meshes using blender"""
from pathlib import Path

from trimesh import Trimesh, util
from trimesh.interfaces.blender import _blender_executable, exists
from trimesh.interfaces.generic import MeshScript

from pychop3d.logger import logger


def run_blender_op(mesh: Trimesh, func_str: str, debug: bool = True):
    """Run a preprocess operation with mesh using Blender."""
    logger.info("starting preprocessing")
    if not exists:
        raise ValueError('No blender available!')

    curr_dir = Path(__file__).parent
    script_fn = curr_dir / "blender_template.py.template"
    script = script_fn.read_text().replace("$FUNC", func_str)

    with MeshScript(meshes=[mesh], script=script, debug=debug) as blend:
        result = blend.run(_blender_executable + ' --background --python $SCRIPT')
    logger.info("finished preprocessing")

    for m in util.make_sequence(result):
        m.face_normals = None
    return result


_DECIMATE_STR = """
def FUNC(mesh):
    modifier = mesh.modifiers.new('decimate', 'DECIMATE')
    modifier.ratio={ratio}
    modifier.use_collapse_triangulate=True
"""
def decimate(mesh: Trimesh, ratio: float, debug: bool = True):
    """Decimate a trimesh by a ratio"""
    result = run_blender_op(mesh, _DECIMATE_STR.format(ratio=ratio), debug)
    return result
