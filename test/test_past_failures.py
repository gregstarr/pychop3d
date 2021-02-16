import pytest
import glob
import os
import tempfile

from pychop3d import utils

from main import run


config_files = glob.glob(os.path.abspath(os.path.join(os.path.dirname(__file__), 'past_failures', '*.yml')))
xfails = ['bracket', 'electronics_housing']


@pytest.mark.slow
@pytest.mark.parametrize('config_fn', config_files)
def test_past_failure(config, config_fn):
    for xf in xfails:
        if xf in config_fn:
            pytest.skip("Can't figure this one out")
    config.load(config_fn)
    with tempfile.TemporaryDirectory() as tmpdir:
        config.directory = tmpdir
        starter = utils.open_mesh()
        if config.part_separation and starter.body_count > 1:
            starter = utils.separate_starter(starter)
        run(starter)
    print()
