import tempfile
from pathlib import Path

import numpy as np
import pytest

from pychop3d.main import run

test_data = [
    ("bent_arm.stl", [np.array([200, 200, 200])]),
    ("bracket.stl", [np.array([20, 20, 20])]),
    ("electronics_housing.stl", [np.array([130, 130, 130])]),
    ("gnome.stl", [np.array([100, 100, 100])]),
    ("large_bunny.stl", [np.array([200, 200, 200])]),
    ("loop_hinge.stl", [np.array([100, 100, 100])]),
    ("maya_marker.stl", [np.array([100, 100, 100])]),
    ("moon_wrench.stl", [np.array([60, 60, 60])]),
    ("planetary_gears.stl", [np.array([40, 40, 40])]),
    ("tadpole.stl", [np.array([50, 50, 50])]),
]


@pytest.mark.parametrize("meshfile, printer_extents", test_data)
def test_past_failure(meshfile, printer_extents):
    with tempfile.TemporaryDirectory() as tmpdir:
        run(
            Path(__file__).parent / "past_failures" / meshfile,
            printer_extents,
            "test_chop",
            Path(tmpdir),
        )
