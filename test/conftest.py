import pytest

from pychop3d.configuration import Configuration


@pytest.fixture(scope='function')
def config():
    config = Configuration.config
    yield config
    config.restore_defaults()
