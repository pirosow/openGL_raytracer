import cython

def loadObj(filePath):
    with open(filePath, "r") as file:
        lines = file.readlines()

    cdef list vp = []
    cdef list vt = []
    cdef list vn = []

    cdef list vertices = []

    cdef int step = max(len(lines) // 100, 10)

    cdef int i = 0

    for line in lines:
        i += 1

        if i % step == 0:
            print(f"\r{round(i / len(lines) * 100, 2)} %", end="")

        line = line.strip()

        words = line.split(" ")

        while "" in words:
            words.remove("")

        if len(words) <= 0:
            continue

        if len(words) <= 0:
            continue

        if words[0] == "v":
            vp.append(read_vertex(words[1:]))

        elif words[0] == "vt":
            vt.append(read_texcoords(words[1:]))

        elif words[0] == "vn":
            vn.append(read_normals(words[1:]))

        elif words[0] == "f":
            read_faces(words[1:], vp, vn, vt, vertices)

    print("")

    return vertices

#f 1/2/3 2/3/2 4/2/5 3/2/4
def read_faces(faces, vp, vn, vt, vertices):
    cdef int triangles = len(faces) - 2

    cdef list v1;
    cdef list v2;
    cdef list v3;

    for i in range(triangles):
        v1 = getVertex(faces[0], vp, vn, vt)
        v2 = getVertex(faces[1 + i], vp, vn, vt)
        v3 = getVertex(faces[2 + i], vp, vn, vt)

        vertices.append(v1)
        vertices.append(v2)
        vertices.append(v3)

def getVertex(face, vp, vn, vt):
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

def read_vertex(vertex):
    return [
        float(vertex[-3]),
        float(vertex[-2]),
        float(vertex[-1])
    ]

def read_texcoords(coords):
    return [
        float(coords[0]),
        float(coords[1]),
    ]

def read_normals(normals):
    return [
        float(normals[0]),
        float(normals[1]),
        float(normals[2])
]