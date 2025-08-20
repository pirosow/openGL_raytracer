import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import ctypes
import pyrr
import os

class App:
    def __init__(self, window_size):
        pg.init()

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)

        pg.display.set_mode(window_size, pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption("OpenGL")

        pg.mouse.set_visible(False)

        glEnable(GL_DEPTH_TEST)

        self.focused = True

        self.sensitivity = 0.1
        self.baseSpeed = 2

        self.w, self.h = window_size

        glViewport(0, 0, self.w, self.h)

        self.clock = pg.time.Clock()

        glClearColor(0.1, 0.1, 0.1, 1)

        glBindVertexArray(glGenVertexArrays(1))

        self.shader = self.createShader("shaders/vertex.txt", "shaders/fragmentColors.txt")

        glUseProgram(self.shader)

        self.plane = Mesh(
            [-20, -20, -10],
            [-90, 0, 180],
            "airplane",
            self.shader
        )

        self.model_location = glGetUniformLocation(self.shader, "model")

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45,
            aspect=window_size[0] / window_size[1],
            near=0.1, far=10000, dtype=np.float32,
        )

        #evoyer la matrice de projection au shader
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )

        self.eye = np.array(
            [-20, 200, -10],
            dtype=np.float32)
        self.orientation = np.array([0, 0, 0], dtype=np.float32)
        self.upVector = np.array([0, 1, 0], dtype=np.float32)
        self.zoom = 0.1

        self.y = 0

        #sauver la position de la vewMatrix
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")

        self.viewMatrix = pyrr.matrix44.create_look_at(self.eye, self.orientation, self.upVector, dtype=np.float32)

        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, self.viewMatrix)

        self.angle = 0
        self.radius = 3

        self.step = 1

        self.stepIncreasing = True

        self.main()

    def getEye(self, y):
        eye = np.array([
            np.cos(np.radians(self.angle)) * self.radius,
            y,
            np.sin(np.radians(self.angle)) * self.radius
        ])

        return eye

    def createShader(self, vertexPath, fragmentPath):
        with open(vertexPath, "r") as f:
            vertex_src = f.read()

        with open(fragmentPath, "r") as f:
            fragment_src = f.read()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

        return shader

    def main(self):
        running = True

        # check events
        while running:
            self.speed = self.baseSpeed

            for event in pg.event.get():
                keys = pg.key.get_pressed()
                if keys[pg.K_ESCAPE]:
                    self.focused = not self.focused

                if keys[pg.KMOD_SHIFT]:
                    self.speed = self.baseSpeed * 5

                if event.type == pg.QUIT:
                    running = False

            pg.event.set_grab(self.focused)
            pg.mouse.set_visible(not self.focused)

            glUseProgram(self.shader)

            self.updateCamera(self.eye, self.orientation, self.upVector, self.zoom)

            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.plane.draw(self.model_location)

            pg.display.flip()

            pg.display.set_caption("OpenGL, fps: " + str(int(self.clock.get_fps())))

            self.clock.tick(60)

        self.quit()

    def updateCamera(self, eye, orientation, upVector, zoom):
        # orientation: [pitch_deg, yaw_deg, roll?]
        pitch = np.radians(orientation[0])
        yaw = np.radians(orientation[1])

        # yaw=0 -> look down -Z; positive yaw rotates right around Y
        fx = np.cos(pitch) * np.sin(yaw)
        fy = np.sin(pitch)
        fz = -np.cos(pitch) * np.cos(yaw)

        forward = np.array([fx, fy, fz], dtype=np.float32)
        forward /= np.linalg.norm(forward)

        # clamp pitch to avoid forward ~ (0,0,0) or extreme flipping
        # optional: orientation[0] = np.clip(orientation[0], -89.0, 89.0)

        # recompute stable camera up:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)

        if np.linalg.norm(right) < 1e-6:
            camera_up = world_up
        else:
            right /= np.linalg.norm(right)
            camera_up = np.cross(right, forward)
            camera_up /= np.linalg.norm(camera_up)

        viewMatrix = pyrr.matrix44.create_look_at(eye, eye + forward * zoom, camera_up)
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, viewMatrix)

        right = np.cross(forward, upVector)
        right /= np.linalg.norm(right)

        pressed = pg.key.get_pressed()

        if pressed[pg.K_w]:
            self.eye += forward * self.speed

        if pressed[pg.K_s]:
            self.eye -= forward * self.speed

        if pressed[pg.K_a]:
            self.eye -= right * self.speed

        if pressed[pg.K_d]:
            self.eye += right * self.speed

        if pressed[pg.K_e]:
            self.eye += upVector * self.speed

        if pressed[pg.K_q]:
            self.eye -= upVector * self.speed

        dx, dy = pg.mouse.get_rel()
        self.orientation[1] += dx * self.sensitivity * self.focused
        self.orientation[0] -= dy * self.sensitivity * self.focused
        self.orientation[0] = np.clip(self.orientation[0], -89, 89)

    def quit(self):
        self.plane.clearMemory()
        self.dragon.clearMemory()

        glDeleteProgram(self.shader)  # program --> shaders

        pg.quit()

class Mesh: #fais que le mesh aie une position et eulers, et appelle le object
    def __init__(self, pos, eulers, dirPath, shader=None):
        dirPath = os.path.join("models/", dirPath)

        print(f"Loading {dirPath}...")

        self.position = np.array(pos, dtype=np.float32)
        self.eulers = np.radians(np.array(eulers, dtype=np.float32))

        if shader:
            self.shader = shader

        files = os.listdir(dirPath)

        objFilePath, mtlFilePath = None, None

        for file in files:
            if file.endswith(".obj"):
                objFilePath = os.path.join(dirPath, file)

            elif file.endswith(".mtl"):
                mtlFilePath = os.path.join(dirPath, file)

        if objFilePath is not None:
            self.total_vertices = self.loadObj(objFilePath)

        else:
            raise "Could not find obj file"

        if mtlFilePath is not None:
            self.textures = self.loadMtl(mtlFilePath, dirPath)

        else:
            self.textures = False

        self.total_vertices_values = list(self.total_vertices.values())

        self.vertex_counts = []

        self.vaos = {

        }

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.total_vertices_values[0].nbytes, self.total_vertices_values[0], GL_DYNAMIC_DRAW)

        for material in list(self.total_vertices.keys()):
            vertices = self.total_vertices[material]

            self.vertex_count = vertices.shape[0]

            self.vertex_counts.append(self.vertex_count)

            vertex_bytes = vertices[0].size * 4

            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            self.vaos[material] = vao

            #Mettre les positions dans location 0 pour le shader
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_bytes, ctypes.c_void_p(0))

            #normals
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_bytes, ctypes.c_void_p(12))

            #uv
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertex_bytes, ctypes.c_void_p(24))

    def loadMtl(self, filePath, dirPath):
        materials = {

        }

        currentMat = ""

        with open(filePath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                words = line.split(" ")

                if "\t" in words[0]:
                    words[0] = words[0][1:]

                if words[0] == "newmtl":
                    currentMat = words[1]

                    materials[words[1]] = ""

                elif words[0] == "map_Kd":
                    imagePath = words[1]

                    materials[currentMat] = self.initTexture(os.path.join(dirPath, imagePath))

        return materials

    def loadObj(self, filePath):
        vp = []
        vt = []
        vn = []

        vertices = {
            "": []
        }

        currentMat = ""

        with open(filePath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                words = line.split(" ")

                while "" in words:
                    words.remove("")

                if len(words) <= 0:
                    continue

                if "usemtl" in words[0]:
                    currentMat = words[1]

                    if currentMat not in list(vertices.keys()):
                        vertices[currentMat] = []

                if len(words) <= 0:
                    continue

                if words[0] == "v":
                    vp.append(self.read_vertex(words[1:]))

                elif words[0] == "vt":
                    vt.append(self.read_texcoords(words[1:]))

                elif words[0] == "vn":
                    vn.append(self.read_normals(words[1:]))

                elif words[0] == "f":
                    self.read_faces(words[1:], vp, vn, vt, vertices[currentMat])

        for mat in list(vertices.keys()):
            if vertices[mat] == []:
                del vertices[mat]

                continue

            vertices[mat] = np.array(vertices[mat], dtype=np.float32)

        return vertices

    def initTexture(self, imgPath):
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)

        return Texture(imgPath, self.shader)

    def draw(self, model_location):
        # creer les transformations et les multiplier
        model_transform = self.get_transform(self.position, self.eulers)

        # envoyer le model transform au shader
        glUniformMatrix4fv(model_location, 1, GL_FALSE, model_transform)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        for material in list(self.total_vertices.keys()):
            glBindVertexArray(self.vaos[material])

            if self.textures:
                if material in list(self.textures.keys()):
                    self.textures[material].use()

                else:
                    glUniform1i(glGetUniformLocation(self.shader, "usingTexture"), 0)

            else:
                try:
                    glUniform1i(glGetUniformLocation(self.shader, "usingTexture"), 0)

                except AttributeError:
                    pass

            vertex_count = len(self.total_vertices[material])

            #envoie les vertices au buffer
            glBufferData(GL_ARRAY_BUFFER, self.total_vertices[material].nbytes, self.total_vertices[material], GL_DYNAMIC_DRAW)

            glDrawArrays(GL_TRIANGLES, 0, vertex_count)

    #f 1/2/3 2/3/2 4/2/5 3/2/4
    def read_faces(self, faces, vp, vn, vt, vertices):
        triangles = len(faces) - 2

        for i in range(triangles):
            v1 = self.getVertex(faces[0], vp, vn, vt)
            v2 = self.getVertex(faces[1 + i], vp, vn, vt)
            v3 = self.getVertex(faces[2 + i], vp, vn, vt)

            vertices.append(v1)
            vertices.append(v2)
            vertices.append(v3)

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

    def get_transform(self, position, eulers):
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)

        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_eulers(
                eulers=eulers,
                dtype=np.float32
            )
        )

        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec=position,
                dtype=np.float32
            )
        )

        return model_transform

    def clearMemory(self):
        glDeleteVertexArrays(1, self.vaos)
        glDeleteBuffers(1, (self.vbo,))

class Texture:
    def __init__(self, path, shader):
        #genere une texture a la location 1
        self.texture = glGenTextures(1)
        #bind la texture
        glBindTexture(GL_TEXTURE_2D, self.texture)

        #dire comment on va afficher la texture
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        #charcher la texture et la donner a OpenGL
        image = pg.image.load(path).convert()

        w, h = image.get_rect().size

        imgData = pg.image.tostring(image, "RGBA")

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, imgData)
        glGenerateMipmap(GL_TEXTURE_2D)

        self.usingTextureLocation = glGetUniformLocation(shader, "usingTexture")


    def use(self):
        glUniform1i(
            self.usingTextureLocation,
            1
        )

        #dire a OpenGL qu'on va utiliser l'emplacement de texture 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def disable(self):
        glUniform1i(
            self.usingTextureLocation,
            0
        )

    def clearMemory(self):
        glDeleteTextures(1, (self.texture,))

if __name__ == "__main__":
    app = App((1000, 720))