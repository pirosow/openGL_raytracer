import cython
import numpy as np
import math

class BoundingBoxes:
    def __init__(self):
        pass

    def getBoundingBoxes(self, total_triangles, tris, poses):
        self.total_triangles = total_triangles
        self.tris = tris
        self.poses = poses

        start = np.array(list((range(self.total_triangles))))

        cdef list boundingBoxes = [start]

        cdef list totalBoundingBoxes = [{"box": start}]

        cdef list childBoundingBoxes = [[start, 0]]

        print("")

        cdef int minus = 0
        cdef int idx = 0

        cdef int slices = math.ceil(math.log(self.total_triangles, 2)) - 1

        cdef int step
        cdef int length

        cdef list lastBoundingBoxes

        cdef int iter

        #cdef list boundingBox

        cdef int parentIndex
        cdef int b1Index
        cdef int b2Index

        #cdef list b1
        #cdef list b2

        for slice in range(slices):
            print(f"\rSlicing {slice + 1}/{slices}... Total number of boxes: {len(totalBoundingBoxes):,}", end="")

            lastBoundingBoxes = childBoundingBoxes.copy()
            childBoundingBoxes = []

            length = len(lastBoundingBoxes)

            print("")

            step = max(length // 1000, 10)

            iter = 0

            for fullBoundingBox in lastBoundingBoxes:
                iter += 1

                boundingBox = fullBoundingBox[0]

                if iter % step == 0:
                    print(f"\r{round(iter / length * 100, 2)}%...", end="")

                idx += 1

                parentIndex = fullBoundingBox[1]

                b1Index = parentIndex + idx + minus
                b2Index = parentIndex + idx + 1 + minus

                b1, b2 = self.sliceBoundingBox(boundingBox)

                if len(b1) >= 1:
                    totalBoundingBoxes[parentIndex]["childA"] = b1Index
                    childBoundingBoxes.append([b1, b1Index])

                    totalBoundingBoxes.append({"box": b1})

                else:
                    minus -= 1

                if len(b2) >= 1:
                    totalBoundingBoxes[parentIndex]["childB"] = b2Index
                    childBoundingBoxes.append([b2, b2Index])

                    totalBoundingBoxes.append({"box": b2})

                else:
                    minus -= 1

            boundingBoxes += childBoundingBoxes.copy()

        cdef list indices = []

        cdef int offset = 0

        print("\rAdding data to boxes...")

        step = max(int(len(totalBoundingBoxes) // 1000), 10)

        #cdef list ind

        cdef int i

        for box in totalBoundingBoxes:
            i += 1

            if i % step == 0:
                print(f"\rBox {(i + 1):,}/{len(totalBoundingBoxes):,}...", end="")

            ind = box["box"]
            length = len(ind)

            box["posMin"], box["posMax"] = self.getBoundingBoxCorners(ind)

            keys = box.keys()

            if "childA" not in keys or "childB" not in keys:
                box["numTriangles"] = np.uint32(length)
                box["triangleOffset"] = np.uint32(offset)

                offset += length
                indices.extend(ind.tolist())

            else:
                box["numTriangles"] = np.uint32(0)
                box["triangleOffset"] = np.uint32(0)

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

    def sliceBoundingBox(self, indices):
        # indices: numpy 1D array of triangle indices
        if indices.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # compute bbox corners using poses of triangle centers or vertices
        poses = self.poses[indices]                 # shape (n,3)
        axis = np.argmax(np.ptp(poses, axis=0))        # ptp = max-min per axis
        center = poses.mean(axis=0)

        mask = poses[:, axis] > center[axis]
        childA = indices[mask]      # ndarray
        childB = indices[~mask]     # ndarray

        return childB, childA      # return ndarrays (no tolist)