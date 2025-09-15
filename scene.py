import numpy as np
from OpenGL.GL import *
import time
import math

class Scene:
    def __init__(self, objects: list):
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
            ("childA", np.int32),
            ("childB", np.int32),

            ("posMin", np.float32, 3),
            ("_pad2", np.float32),
            ("posMax", np.float32, 3),
            ("_pad3", np.float32),
        ])

        print("Calculating triangles...")

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

        self.v0_pos_3 = v0_pos / 3
        self.v1_pos_3 = v1_pos / 3
        self.v2_pos_3 = v2_pos / 3

        self.poses = self.v0_pos_3 + self.v1_pos_3 + self.v2_pos_3

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

        start = time.time()

        self.indices, self.totalBoxes, self.leaves = self.getBoundingBoxes()

        print(f"\nTime taken: {round(time.time() - start, 2)} seconds")

    def send(self):
        print("\nSending scene data to gpu...")

        self.boxes = np.zeros(len(self.totalBoxes), dtype=self.boundingBoxStruct)

        self.numTriangles = []
        self.triangleOffsets = []
        self.posMin = []
        self.posMax = []
        self.childA = []
        self.childB = []

        step = max(len(self.totalBoxes) // 100, 1)

        for i, box in enumerate(self.totalBoxes):
            if i % step == 0:
                print(f"\rBox {i + 1}/{len(self.totalBoxes)}...", end="")

            numTris = box["numTriangles"]
            offset = box["triangleOffset"]
            posMin = box["posMin"]
            posMax = box["posMax"]

            if "childA" in list(box.keys()):
                childA = box["childA"]

            else:
                childA = -1

            if "childB" in list(box.keys()):
                childB = box["childB"]

            else:
                childA = -1
                childB = -1

            avgTris = 0
            minTris = math.inf
            maxTris = 0

            self.numTriangles.append(numTris)
            self.triangleOffsets.append(offset)
            self.posMin.append(posMin)
            self.posMax.append(posMax)
            self.childA.append(childA)
            self.childB.append(childB)

        for box in self.leaves:
            length = len(box)

            if length > maxTris:
                maxTris = length

            if length < minTris:
                minTris = length

            avgTris += length

        avgTris /= len(self.leaves)

        self.numTriangles = np.array(self.numTriangles, dtype=np.uint32)
        self.triangleOffsets = np.array(self.triangleOffsets, dtype=np.uint32)
        self.posMin = np.array(self.posMin, dtype=np.float32)
        self.posMax = np.array(self.posMax, dtype=np.float32)
        self.childA = np.array(self.childA, dtype=np.int32)
        self.childB = np.array(self.childB, dtype=np.int32)

        self.boxes['numTriangles'] = self.numTriangles
        self.boxes['triangleOffset'] = self.triangleOffsets
        self.boxes['posMin'] = self.posMin
        self.boxes['posMax'] = self.posMax
        self.boxes['childA'] = self.childA
        self.boxes['childB'] = self.childB

        self.total_boxes = len(self.boxes)

        self.indices = np.array(self.indices, dtype=np.uint32)

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

        print("\n\n---Scene---")
        print(f"Number of triangles: {len(self.tris)}")
        print(f"Number of vertices: {len(self.tris) * 3}")
        print(f"Number of objects: {len(self.objects)}")
        print(f"Number of bounding boxes: {len(self.totalBoxes)}")
        print(f"Avg number of triangles per bounding box: {np.round(avgTris, 1)}")
        print(f"Min number of triangles per bounding box: {minTris}")
        print(f"Max number of triangles per bounding box: {maxTris}")

    def getBoundingBoxes(self):
        start = list((range(self.total_triangles)))

        boundingBoxes = [start]

        totalBoundingBoxes = [{"box": start}]

        childBoundingBoxes = [[start, 0]]

        idx = 0

        print("")

        minus = 0

        slices = math.ceil(math.log(self.total_triangles, 2)) - 1

        for slice in range(slices):
            print(f"\rSlicing {slice + 1}/{slices}... Total number of boxes: {len(totalBoundingBoxes)}", end="")

            lastBoundingBoxes = childBoundingBoxes.copy()
            childBoundingBoxes = []

            length = len(lastBoundingBoxes)

            print("")

            step = max(length // 50, 1)

            for i, fullBoundingBox in enumerate(lastBoundingBoxes):
                boundingBox = fullBoundingBox[0]

                if i % step == 0:
                    print(f"\r{round(i / length * 100, 2)}%...", end="")

                idx += 1

                parentIndex = fullBoundingBox[1]

                b1Index = parentIndex + idx + minus
                b2Index = parentIndex + idx + 1 + minus

                b1, b2 = self.sliceBoundingBox(boundingBox)

                if len(b1) > 0:
                    totalBoundingBoxes[parentIndex]["childA"] = b1Index
                    childBoundingBoxes.append([b1, b1Index])

                else:
                    minus -= 1

                if len(b2) > 0:
                    totalBoundingBoxes[parentIndex]["childB"] = b2Index
                    childBoundingBoxes.append([b2, b2Index])

                else:
                    minus -= 1

            boundingBoxes += childBoundingBoxes.copy()

            for boundingBox in childBoundingBoxes:
                if len(boundingBox[0]) > 0:
                    totalBoundingBoxes.append({"box": boundingBox[0]})

        indices = []

        offset = 0

        print("\rAdding data to boxes...")

        step = max(len(totalBoundingBoxes) // 50, 1)

        for i, box in enumerate(totalBoundingBoxes):
            ind = box["box"]
            length = len(ind)

            if i % step == 0:
                print(f"\rBox {i + 1}/{len(totalBoundingBoxes)}...", end="")

            box["numTriangles"] = np.uint32(length)
            box["triangleOffset"] = np.uint32(offset)
            box["posMin"], box["posMax"] = self.getBoundingBoxCorners(ind)

            offset += length
            indices += ind

        return indices, totalBoundingBoxes, childBoundingBoxes

    def getBoundingBoxCorners(self, box):
        # box: sequence of global triangle indices (may be list or ndarray)
        if len(box) == 0:
            raise ValueError("box must contain at least one triangle")

        indices = np.asarray(box, dtype=np.intp)

        # Assume these are numpy arrays; avoid per-element np.asarray in loop
        v0_all = self.tris['v0']['pos'][indices].astype(np.float32, copy=False)
        v1_all = self.tris['v1']['pos'][indices].astype(np.float32, copy=False)
        v2_all = self.tris['v2']['pos'][indices].astype(np.float32, copy=False)

        # mins for each vertex-array (shape (3,))
        min0 = v0_all.min(axis=0)
        min1 = v1_all.min(axis=0)
        min2 = v2_all.min(axis=0)

        # maxs for each vertex-array (shape (3,))
        max0 = v0_all.max(axis=0)
        max1 = v1_all.max(axis=0)
        max2 = v2_all.max(axis=0)

        # final min/max across all three sets
        posMin = np.minimum(np.minimum(min0, min1), min2)
        posMax = np.maximum(np.maximum(max0, max1), max2)

        return posMin, posMax

    def sliceBoundingBox(self, box):
        posMin, posMax = self.getBoundingBoxCorners(box)

        delta = posMax - posMin

        axis = np.argmax(delta)

        indices = np.array(box)

        v0 = self.v0_pos_3[indices]
        v1 = self.v1_pos_3[indices]
        v2 = self.v2_pos_3[indices]

        poses = self.poses[indices]

        center = np.sum(v0, axis=0)
        center += np.sum(v1, axis=0)
        center += np.sum(v2, axis=0)

        center /= len(box)

        childA = np.where(poses[:, axis] > center[axis])[0]
        childB = np.where(poses[:, axis] <= center[axis])[0]

        boundingBox2 = indices[childA].tolist()
        boundingBox1 = indices[childB].tolist()

        return boundingBox1, boundingBox2

    def clearMemory(self):
        glDeleteBuffers(3, (self.tris_object, self.indicesObject, self.boundingBoxObject))