import trimesh
import pytest

from pychop3d import utils
from pychop3d import constants


@pytest.fixture(scope='function')
def mesh():
    conf = constants.default_config.copy()
    m, conf = utils.open_mesh(conf)
    yield m

