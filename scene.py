import os
import numpy as np
from OpenGL.GL import *
from object import Mesh

class Scene:
    def __init__(self, objects: list, slices):
        print("Initalizing scene...")

        self.objects = objects

        # --- gather per-object arrays into lists (robust v.s. np.append flattening) ---
        pos_list = []
        norm_list = []
        uv_list = []
        vertex_counts = []

        self.colors = []
        self.emissions = []
        self.emission_colors = []
        self.surfaces = []

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
            self.surfaces.append([obj.emission, obj.roughness])

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
            ("surface", np.float32, 2),
            ("_pad_final", np.float32, 2),
        ], align=False)

        self.boundingBoxStruct = np.dtype([
            ("numTriangles", np.uint32),
            ("triangleOffset", np.uint32),
            ("childA", np.uint32),
            ("childB", np.uint32),

            ("posMin", np.float32, 3),
            ("_pad2", np.float32),
            ("posMax", np.float32, 3),
            ("_pad3", np.float32),
        ])

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
        emc_arr = np.vstack(self.emission_colors).astype(np.float32)
        surface_arr = np.vstack(self.surfaces).astype(np.float32)

        # expand to per-triangle and assign
        self.tris['color'] = colors_arr[tri_obj_idx]
        self.tris['emission_color'] = emc_arr[tri_obj_idx]
        self.tris['surface'] = surface_arr[tri_obj_idx]

        self.total_triangles = len(self.tris)

        print("\nSlicing bounding boxes...")

        boxes, indices = self.getBoundingBoxes(slices)

        self.boxes = np.zeros(len(boxes), dtype=self.boundingBoxStruct)

        self.numTriangles = []
        self.triangleOffsets = []
        self.posMin = []
        self.posMax = []

        for box in boxes:
            numTris = box["numTriangles"]
            offset = box["triangleOffset"]
            posMin = box["posMin"]
            posMax = box["posMax"]

            self.numTriangles.append(numTris)
            self.triangleOffsets.append(offset)
            self.posMin.append(posMin)
            self.posMax.append(posMax)

        self.numTriangles = np.array(self.numTriangles, dtype=np.uint32)
        self.triangleOffsets = np.array(self.triangleOffsets, dtype=np.uint32)
        self.posMin = np.array(self.posMin, dtype=np.float32)
        self.posMax = np.array(self.posMax, dtype=np.float32)

        self.boxes['numTriangles'] = self.numTriangles
        self.boxes['triangleOffset'] = self.triangleOffsets
        self.boxes['posMin'] = self.posMin
        self.boxes['posMax'] = self.posMax

        self.total_boxes = len(self.boxes)

        self.indices = np.array(indices, dtype=np.uint32)

        """
        for box in boxes:
            numTris = box["numTriangles"]
            offset = box["triangleOffset"]

            for ind in range(offset, numTris + offset):
                self.tris['emission_color'][ind] = self.random_color(offset)
        """

        self.tris_object = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tris_object)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.tris.nbytes, np.ascontiguousarray(self.tris), GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.tris_object)

        self.boundingBoxObject = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.boundingBoxObject)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.boxes.nbytes, np.ascontiguousarray(self.boxes), GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.boundingBoxObject)

        self.indicesObject = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.indicesObject)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.indicesObject)

        print("\n---Scene---")
        print(f"Number of triangles: {len(self.tris)}")
        print(f"Number of vertices: {len(self.tris) * 3}")
        print(f"Number of objects: {len(objects)}")
        print(f"Number of bounding boxes: {len(boxes)}")
        print(f"Average number of triangles per bounding box: {np.round(np.mean(self.numTriangles), 1)}")

    def random_color(self, seed):
        rng = np.random.default_rng(seed)  # Seeded random generator
        color = rng.random(3, dtype=np.float32)  # Random array of size 3 with values in [0,1]
        return color

    def read_ssbo(self, ssbo, dtype, count):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)

        # total size in bytes
        size = dtype.itemsize * count

        # allocate raw byte buffer
        raw = np.empty(size, dtype=np.uint8)

        # copy from GPU into CPU array
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, raw.nbytes, raw)

        # reinterpret as structured dtype
        result = raw.view(dtype=dtype)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        return result

    def getBoundingBoxes(self, slices):
        start = list(range(self.total_triangles))

        boundingBoxes = [start]

        for i in range(slices):
            lastBoundingBoxes = []

            for boundingBox in boundingBoxes:
                b1, b2 = self.sliceBoundingBox(boundingBox)

                if len(b1) > 0:
                    lastBoundingBoxes.append(b1)

                if len(b2) > 0:
                    lastBoundingBoxes.append(b2)

            boundingBoxes = lastBoundingBoxes.copy()

        indices = []

        resBoxes = []

        offset = 0

        for i, box in enumerate(boundingBoxes):
            if len(box) > 0:
                resBox = {}

                resBox["numTriangles"] = np.uint32(len(box))
                resBox["triangleOffset"] = np.uint32(offset)
                resBox["posMin"], resBox["posMax"] = self.getBoundingBoxCorners(box)

                resBoxes.append(resBox)

                for index in box:
                    offset += 1

                    indices.append(index)

        return resBoxes, indices

    def getBoundingBoxCorners(self, box):
        box = list(box)
        if len(box) == 0:
            return (np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32))

        # Initialize posMin/posMax using the first triangle
        i0 = box[0]
        p0 = np.asarray(self.tris['v0']['pos'][i0], dtype=np.float32)
        p1 = np.asarray(self.tris['v1']['pos'][i0], dtype=np.float32)
        p2 = np.asarray(self.tris['v2']['pos'][i0], dtype=np.float32)

        posMin = np.minimum(np.minimum(p0, p1), p2)
        posMax = np.maximum(np.maximum(p0, p1), p2)

        # Expand bounds over remaining triangles
        for idx in box[1:]:
            pv0 = np.asarray(self.tris['v0']['pos'][idx], dtype=np.float32)
            pv1 = np.asarray(self.tris['v1']['pos'][idx], dtype=np.float32)
            pv2 = np.asarray(self.tris['v2']['pos'][idx], dtype=np.float32)

            posMin = np.minimum(posMin, pv0)
            posMin = np.minimum(posMin, pv1)
            posMin = np.minimum(posMin, pv2)

            posMax = np.maximum(posMax, pv0)
            posMax = np.maximum(posMax, pv1)
            posMax = np.maximum(posMax, pv2)

        return posMin, posMax

    def sliceBoundingBox(self, box):
        posMin, posMax = self.getBoundingBoxCorners(box)

        x = abs(posMin[0] - posMax[0])
        y = abs(posMin[1] - posMax[1])
        z = abs(posMin[2] - posMax[2])

        axis = [x, y, z].index(max(x, y, z))

        boundingBox1 = []
        boundingBox2 = []

        center = np.array([0, 0, 0], dtype=np.float32)

        idx = 0

        for index in box:
            idx += 1

            v0 = self.tris['v0']['pos'][index]
            v1 = self.tris['v1']['pos'][index]
            v2 = self.tris['v2']['pos'][index]

            center += v0 / 3
            center += v1 / 3
            center += v2 / 3

        center /= idx

        # calculer les bounding boxes selon l'axe
        for i in box:
            v0 = self.tris['v0']['pos'][i] / 3
            v1 = self.tris['v1']['pos'][i] / 3
            v2 = self.tris['v2']['pos'][i] / 3

            pos = (v0 + v1 + v2)

            if pos[axis] > center[axis]:
                boundingBox2.append(i)

            else:
                boundingBox1.append(i)

        return boundingBox1, boundingBox2

    def clearMemory(self):
        glDeleteBuffers(3, (self.tris_object, self.indicesObject, self.boundingBoxObject))