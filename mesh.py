import os
import numpy as np
from OpenGL.GL import *

class Mesh: #fais que le mesh aie une position et eulers, et appelle le object
    def __init__(self, pos, eulers, dirPath, color, emission_color, emission, roughness, scale):
        dirPath = os.path.join("models/", dirPath)

        print(f"Loading {dirPath}...")

        self.position = np.array(pos, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.scale = np.array([scale, scale, scale], dtype=np.float32)

        self.vertexStruct = np.dtype([
            ("pos", np.float32, 3), ("_pad0", np.float32),
            ("normal", np.float32, 3), ("_pad1", np.float32),
        ], align=False)  # itemsize == 32

        self.triangleStruct = np.dtype([
            ("indices", np.uint32, 3), #uvec3
            ("pad0", np.uint32),
            ("color", np.float32, 3), #vec3
            ("pad1", np.float32),
            ("emission_color", np.float32, 3), #vec3
            ("pad2", np.float32),
            ("emission", np.float32), #float
            ("roughness", np.float32), #float
            ("pad4", np.float32, 2)
        ], align=False)

        files = os.listdir(dirPath)

        objFilePath, mtlFilePath = None, None

        for file in files:
            if file.endswith(".obj"):
                objFilePath = os.path.join(dirPath, file)

        if objFilePath is not None:
            self.total_vertices, self.faces = self.loadObj(objFilePath)

        self.total_vertices = self.total_vertices.reshape(-1, 8).astype(np.float32)

        self.pos = self.total_vertices[:, 0:3].astype(np.float32)
        self.normals = self.total_vertices[:, 3:6].astype(np.float32)
        self.uvs = self.total_vertices[:, 6:8].astype(np.float32)

        self.pos, self.normals = self.getWorld()

        self.verts = np.zeros(len(self.total_vertices), dtype=self.vertexStruct)
        self.tris = np.zeros(len(self.total_vertices), dtype=self.triangleStruct)

        self.verts["pos"][:, :3] = self.pos
        self.verts["normal"][:, :3] = self.normals

        self.tris["indices"][:, :3] = self.faces
        self.tris["color"][:, :3] = np.array(color, dtype=np.float32)
        self.tris["emission_color"][:, :3] = np.array(emission_color, dtype=np.float32)
        self.tris["emission"] = emission
        self.tris["roughness"] = roughness

        self.verts_object = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.verts_object)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.verts.nbytes, self.verts, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.verts_object)

        self.tris_object = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tris_object)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.tris.nbytes, self.tris, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.tris_object)

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

    def clearMemory(self):
        glDeleteBuffers(1, (self.vbo,))
        glDeleteBuffers(1, (self.ssbo,))