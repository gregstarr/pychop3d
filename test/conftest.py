from pathlib import Path
import pytest


@pytest.fixture()
def bunny_mesh():
    return Path(__file__).parent / "test_meshes" / "Bunny-LowPoly.stl"
