import trimesh
import pytest


@pytest.fixture(scope='function')
def bunny():
    fn = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
    mesh = trimesh.load(fn, validate=True)
    yield mesh
