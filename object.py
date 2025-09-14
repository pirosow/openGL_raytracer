import os
import numpy as np
from OpenGL.GL import *

class Mesh: #fais que le mesh aie une position et eulers, et appelle le object
    def __init__(self, pos, eulers, dirPath, color=[0, 0, 0], emission_color=[0, 0, 0], emission=0, roughness=0, scale=1):
        dirPath = os.path.join("models/", dirPath)

        print(f"Loading {dirPath}...")

        self.position = np.array(pos, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.scale = np.array([scale, scale, scale], dtype=np.float32)

        files = os.listdir(dirPath)

        objFilePath, mtlFilePath = None, None

        for file in files:
            if file.endswith(".obj"):
                objFilePath = os.path.join(dirPath, file)

        if objFilePath is not None:
            self.total_vertices, self.faces = self.loadObj(objFilePath)

        # after these lines (your existing code)
        self.total_vertices = self.total_vertices.reshape(-1, 8).astype(np.float32)

        self.pos = self.total_vertices[:, 0:3].astype(np.float32)
        self.normals = self.total_vertices[:, 3:6].astype(np.float32)
        self.uvs = self.total_vertices[:, 6:8].astype(np.float32)

        # apply transform (you already do this)
        self.pos, self.normals = self.getWorld()

        self.color = color
        self.emission_color = emission_color
        self.emission = emission
        self.roughness = roughness

    def getWorld(self):
        # transform vertices to world space as you already do
        model_mat4, normal_mat3 = self.make_model_and_normal_matrices(self.position, self.eulers, self.scale, "XYZ")
        RS3 = model_mat4[:3, :3]
        translation = model_mat4[:3, 3]
        world_pos = (RS3 @ self.pos.T).T + translation
        world_normals = (normal_mat3 @ self.normals.T).T
        norms = np.linalg.norm(world_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        world_normals = world_normals / norms

        return world_pos, world_normals

    def rotation_matrix_from_euler(self, rx, ry, rz, order="XYZ"):
        """rx,ry,rz in radians. order string like 'XYZ' (apply X then Y then Z)."""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]], dtype=np.float32)

        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]], dtype=np.float32)

        Rz = np.array([[cz, -sz, 0],
                       [sz, cz, 0],
                       [0, 0, 1]], dtype=np.float32)

        mats = {"X": Rx, "Y": Ry, "Z": Rz}
        R = np.eye(3, dtype=np.float32)
        # rightmost in multiplication is applied first; we reverse to compose correctly
        for axis in reversed(order):
            R = mats[axis] @ R
        return R

    def make_model_and_normal_matrices(self, position, euler_deg, scale=(1.0, 1.0, 1.0), order="XYZ"):
        """
        position: (tx,ty,tz)
        euler_deg: (rx,ry,rz) in degrees
        scale: (sx,sy,sz)
        returns: model_mat (4x4 np.float32), normal_mat (3x3 np.float32)
        """
        tx, ty, tz = position
        rx, ry, rz = np.deg2rad(euler_deg)  # convert degrees -> radians
        sx, sy, sz = scale

        R3 = self.rotation_matrix_from_euler(rx, ry, rz, order)
        S3 = np.diag([sx, sy, sz]).astype(np.float32)

        RS3 = R3 @ S3  # scale then rotate (v' = R * (S * v))
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = RS3
        M[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)

        # normal matrix = transpose(inverse(upper-left 3x3))
        # guard against singular matrix
        try:
            normal_mat = np.linalg.inv(M[:3, :3]).T.astype(np.float32)
        except np.linalg.LinAlgError:
            # fallback to just rotation (if inverse fails)
            normal_mat = R3.astype(np.float32)

        return M, normal_mat

    def loadObj(self, filePath):
        vp = []
        vt = []
        vn = []

        vertices = []
        faces = []

        with open(filePath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                words = line.split(" ")

                while "" in words:
                    words.remove("")

                if len(words) <= 0:
                    continue

                if len(words) <= 0:
                    continue

                if words[0] == "v":
                    vp.append(self.read_vertex(words[1:]))

                elif words[0] == "vt":
                    vt.append(self.read_texcoords(words[1:]))

                elif words[0] == "vn":
                    vn.append(self.read_normals(words[1:]))

                elif words[0] == "f":
                    self.read_faces(words[1:], vp, vn, vt, vertices, faces)

            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.float32)

        return vertices, faces

    #f 1/2/3 2/3/2 4/2/5 3/2/4
    def read_faces(self, faces, vp, vn, vt, vertices, facesList):
        triangles = len(faces) - 2

        for i in range(triangles):
            v1 = self.getVertex(faces[0], vp, vn, vt)
            v2 = self.getVertex(faces[1 + i], vp, vn, vt)
            v3 = self.getVertex(faces[2 + i], vp, vn, vt)

            vertices.append(v1)
            vertices.append(v2)
            vertices.append(v3)

            facesList.append(self.get_face(faces[0]))
            facesList.append(self.get_face(faces[1]))
            facesList.append(self.get_face(faces[2]))

    def get_face(self, face):
        faces = face.split("/")

        return [int(faces[0]) - 1, int(faces[1]) - 1, int(faces[2]) - 1]

    def getVertex(self, face, vp, vn, vt):
        #vp/vt/vn
        #1/2/3

        f = face.split("/")

        v = f[0]

        #1/2/3 or 1//3
        if len(f) == 3:
            t = f[1]
            n = f[2]

            v = vp[int(v) - 1]

            if t != "":
                t = vt[int(t) - 1]

            else:
                t = [0, 0]

            if n != "":
                n = vn[int(n) - 1]

            else:
                n = [0, 0, 1]

        #1
        elif len(f) == 2:
            t = f[1]

            v = vp[int(v) - 1]
            t = vt[int(t) - 1]
            n = [0, 0, 1]

        else:
            v = vp[int(v) - 1]
            t = [0, 0]
            n = [0, 0, 1]

        t = [t[0], 1 - t[1]]

        return v + n + t

    def read_vertex(self, vertex):
        return [
            float(vertex[-3]),
            float(vertex[-2]),
            float(vertex[-1])
        ]

    def read_texcoords(self, coords):
        return [
            float(coords[0]),
            float(coords[1]),
        ]

    def read_normals(self, normals):
        return [
            float(normals[0]),
            float(normals[1]),
            float(normals[2])
        ]

class Rect:
    def __init__(self, size, pos, eulers, color=[0, 0, 0], emission_color=[0, 0, 0], emission=0, roughness=0, scale=1):
        self.position = np.array(pos, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.scale = np.array([scale, scale, scale], dtype=np.float32)

        self.total_vertices = self.make_cube_vertices(size)

        self.total_vertices = self.total_vertices.reshape(-1, 8).astype(np.float32)

        self.pos = self.total_vertices[:, 0:3].astype(np.float32)
        self.normals = self.total_vertices[:, 3:6].astype(np.float32)
        self.uvs = self.total_vertices[:, 6:8].astype(np.float32)

        # apply transform (you already do this)
        self.pos, self.normals = self.getWorld()

        self.color = color
        self.emission_color = np.array(emission_color)
        self.emission = emission
        self.roughness = roughness

    def make_cube_vertices(self, size):
        """
        size: iterable of length 3: (sx, sy, sz)
        returns: numpy array shape (36,8) dtype float32 where each vertex is:
          [px,py,pz, nx,ny,nz, u,v]
        """
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

        # For each face we define 4 corner positions (quad), a single face normal, and 4 UVs
        # We will emit two triangles per quad: (0,1,2) and (0,2,3)
        faces = [
            # +Z front
            ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), (0.0, 0.0, 1.0)),
            # -Z back
            ((hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (0.0, 0.0, -1.0)),
            # +Y top
            ((-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz), (0.0, 1.0, 0.0)),
            # -Y bottom
            ((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz), (0.0, -1.0, 0.0)),
            # +X right
            ((hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (1.0, 0.0, 0.0)),
            # -X left
            ((-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-1.0, 0.0, 0.0)),
        ]

        # simple UVs for each corner of the face
        uv0 = (0.0, 0.0)
        uv1 = (1.0, 0.0)
        uv2 = (1.0, 1.0)
        uv3 = (0.0, 1.0)

        verts = []
        for p0, p1, p2, p3, normal in faces:
            nx, ny, nz = normal
            # triangle 1: p0, p1, p2
            verts.append((*p0, nx, ny, nz, uv0[0], uv0[1]))
            verts.append((*p1, nx, ny, nz, uv1[0], uv1[1]))
            verts.append((*p2, nx, ny, nz, uv2[0], uv2[1]))
            # triangle 2: p0, p2, p3
            verts.append((*p0, nx, ny, nz, uv0[0], uv0[1]))
            verts.append((*p2, nx, ny, nz, uv2[0], uv2[1]))
            verts.append((*p3, nx, ny, nz, uv3[0], uv3[1]))

        arr = np.array(verts, dtype=np.float32)  # shape (36, 8)
        return arr

    def getWorld(self):
        # transform vertices to world space as you already do
        model_mat4, normal_mat3 = self.make_model_and_normal_matrices(self.position, self.eulers, self.scale, "XYZ")
        RS3 = model_mat4[:3, :3]
        translation = model_mat4[:3, 3]
        world_pos = (RS3 @ self.pos.T).T + translation
        world_normals = (normal_mat3 @ self.normals.T).T
        norms = np.linalg.norm(world_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        world_normals = world_normals / norms

        return world_pos, world_normals

    def rotation_matrix_from_euler(self, rx, ry, rz, order="XYZ"):
        """rx,ry,rz in radians. order string like 'XYZ' (apply X then Y then Z)."""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]], dtype=np.float32)

        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]], dtype=np.float32)

        Rz = np.array([[cz, -sz, 0],
                       [sz, cz, 0],
                       [0, 0, 1]], dtype=np.float32)

        mats = {"X": Rx, "Y": Ry, "Z": Rz}
        R = np.eye(3, dtype=np.float32)
        # rightmost in multiplication is applied first; we reverse to compose correctly
        for axis in reversed(order):
            R = mats[axis] @ R
        return R

    def make_model_and_normal_matrices(self, position, euler_deg, scale=(1.0, 1.0, 1.0), order="XYZ"):
        """
        position: (tx,ty,tz)
        euler_deg: (rx,ry,rz) in degrees
        scale: (sx,sy,sz)
        returns: model_mat (4x4 np.float32), normal_mat (3x3 np.float32)
        """
        tx, ty, tz = position
        rx, ry, rz = np.deg2rad(euler_deg)  # convert degrees -> radians
        sx, sy, sz = scale

        R3 = self.rotation_matrix_from_euler(rx, ry, rz, order)
        S3 = np.diag([sx, sy, sz]).astype(np.float32)

        RS3 = R3 @ S3  # scale then rotate (v' = R * (S * v))
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = RS3
        M[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)

        # normal matrix = transpose(inverse(upper-left 3x3))
        # guard against singular matrix
        try:
            normal_mat = np.linalg.inv(M[:3, :3]).T.astype(np.float32)
        except np.linalg.LinAlgError:
            # fallback to just rotation (if inverse fails)
            normal_mat = R3.astype(np.float32)

        return M, normal_mat