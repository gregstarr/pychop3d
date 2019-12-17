import trimesh


class BSPMesh(trimesh.Trimesh):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convex_hull = None

    @classmethod
    def from_trimesh(cls, mesh):
        return cls(mesh.vertices, mesh.faces, mesh.face_normals, mesh.vertex_normals, validate=True)

    @property
    def convex_hull(self):
        return self._convex_hull
