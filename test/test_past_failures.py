import pathlib
import tempfile

import pytest

from pychop3d import run, prepare_starter


config_files = (pathlib.Path(__file__).parent / "past_failures").glob("*.yml")


@pytest.mark.parametrize('config_fn', config_files)
def test_past_failure(config, config_fn):
    config.load(config_fn)
    with tempfile.TemporaryDirectory() as tmpdir:
        config.directory = tmpdir
        starter = prepare_starter()
        run(starter)
    print()
