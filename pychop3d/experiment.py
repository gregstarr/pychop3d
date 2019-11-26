import trimesh
from pychop3d.search import beam_search

# create box
fn = "C:\\Users\\Greg\\Downloads\\Low_Poly_Stanford_Bunny\\files\\Bunny-LowPoly.stl"
mesh = trimesh.load(fn)
mesh.apply_scale(2)

best_tree = beam_search(mesh)

best_tree.preview()
