import os
import numpy as np
from OpenGL.GL import *
from object import Mesh

class Scene:
    def __init__(self, objects: list):
        # --- gather per-object arrays into lists (robust v.s. np.append flattening) ---
        pos_list = []
        norm_list = []
        uv_list = []
        vertex_counts = []

        self.colors = []
        self.emissions = []
        self.emission_colors = []
        self.roughnesses = []

        for obj in objects:
            p = np.asarray(obj.pos, dtype=np.float32)
            n = np.asarray(obj.normals, dtype=np.float32)
            u = np.asarray(obj.uvs, dtype=np.float32)

            pos_list.append(p)
            norm_list.append(n)
            uv_list.append(u)
            vertex_counts.append(p.shape[0])  # number of vertices this object contributes

            self.colors.append(np.asarray(obj.color, dtype=np.float32))
            self.emission_colors.append(np.asarray(obj.emission_color, dtype=np.float32))
            self.emissions.append(float(obj.emission))
            self.roughnesses.append(float(obj.roughness))

        # stack into single big arrays (same ordering as original code intended)
        self.pos = np.vstack(pos_list) if pos_list else np.zeros((0,3), dtype=np.float32)
        self.normals = np.vstack(norm_list) if norm_list else np.zeros((0,3), dtype=np.float32)
        self.uvs = np.vstack(uv_list) if uv_list else np.zeros((0,2), dtype=np.float32)

        # --- dtype definitions (keep your existing ones) ---
        self.vertexStruct = np.dtype([
            ("pos", np.float32, 3), ("_pad0", np.float32),
            ("normal", np.float32, 3), ("_pad1", np.float32),
        ], align=False)

        self.triangleStruct = np.dtype([
            ("v0", self.vertexStruct),
            ("v1", self.vertexStruct),
            ("v2", self.vertexStruct),
            ("color", np.float32, 3), ("_pad_color", np.float32),
            ("emission_color", np.float32, 3), ("_pad_emc", np.float32),
            ("emission", np.float32),
            ("roughness", np.float32),
            ("_pad_final", np.float32, 2),
        ], align=False)

        # ----- build triangles by consuming vertices 3-at-a-time -----
        n_vertices = self.pos.shape[0]
        if n_vertices < 3:
            self.tris = np.zeros(0, dtype=self.triangleStruct)
            return

        n_tris = n_vertices // 3
        if n_vertices % 3 != 0:
            print(f"Warning: {n_vertices % 3} leftover vertex/vertices ignored when building triangles")

        # allocate
        self.tris = np.zeros(n_tris, dtype=self.triangleStruct)

        # slice vertex data into v0/v1/v2 for triangles
        v0_pos = self.pos[0::3][:n_tris]
        v1_pos = self.pos[1::3][:n_tris]
        v2_pos = self.pos[2::3][:n_tris]

        v0_norm = self.normals[0::3][:n_tris]
        v1_norm = self.normals[1::3][:n_tris]
        v2_norm = self.normals[2::3][:n_tris]

        # assign nested fields (works with your vertexStruct having 'pos' and 'normal')
        self.tris['v0']['pos'] = v0_pos
        self.tris['v0']['normal'] = v0_norm

        self.tris['v1']['pos'] = v1_pos
        self.tris['v1']['normal'] = v1_norm

        self.tris['v2']['pos'] = v2_pos
        self.tris['v2']['normal'] = v2_norm

        # after building self.pos/self.normals and self.tris and computing n_tris ...

        # build cumulative vertex-start array for objects
        starts = np.concatenate(([0], np.cumsum(vertex_counts)))  # length = num_objects+1

        # triangle start vertex indices in the global vertex array: 0,3,6,...
        tri_start_vertices = (np.arange(n_tris) * 3)

        # get object index for each triangle: triangles whose start vertex lies in object i
        tri_obj_idx = np.searchsorted(starts, tri_start_vertices, side='right') - 1
        tri_obj_idx = np.clip(tri_obj_idx, 0, len(vertex_counts) - 1)  # safety clamp

        # material arrays per object
        colors_arr = np.vstack(self.colors).astype(np.float32)  # shape = (n_objects,3)
        emc_arr = np.vstack(self.emission_colors).astype(np.float32)  # (n_objects,3)
        emissions_arr = np.asarray(self.emissions, dtype=np.float32)  # (n_objects,)
        roughness_arr = np.asarray(self.roughnesses, dtype=np.float32)  # (n_objects,)

        # expand to per-triangle and assign
        self.tris['color'] = colors_arr[tri_obj_idx]
        self.tris['emission_color'] = emc_arr[tri_obj_idx]
        self.tris['emission'] = emissions_arr[tri_obj_idx]
        self.tris['roughness'] = roughness_arr[tri_obj_idx]

        self.tris_object = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tris_object)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.tris.nbytes, self.tris, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.tris_object)

        print("---Scene---")
        print(f"Number of triangles: {len(self.tris)}")
        print(f"Number of vertices: {len(self.tris) * 3}")
        print(f"Number of objects: {len(objects)}")

    def clearMemory(self):
        glDeleteBuffers(1, (self.tris_object,))