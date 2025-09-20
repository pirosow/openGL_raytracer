import pygame as pg
from OpenGL.GL.shaders import compileProgram, compileShader
import math
import time
from screen import Screen
from object import *
import tkinter as tk
from scene import Scene
import os

def read_shader(path):
    with open(path, "r") as f:
        return f.read()

class App:
    def __init__(self, window_size, screen_size, bounces, rays_per_pixel, jitter_amount, lambertian, skyIllumination, tileSize):
        os.system("cls")

        self.dragon = Mesh(
            [-5, -10, 0],
            [270, 0, -90],
            "stanford_mediumdragon",
            [0.92, 0.92, 0.86],
            roughness=1,
            scale=0.25
        )

        self.sphere = Mesh(
            [-25, -20, 20],
            [0, 0, 0],
            color=[1, 1, 1],
            dirPath="sphere",
            roughness=0,
            scale=7
        )

        self.redWall = Rect(
            [8, 5, 0.1],
            [0, 0, 30],
            [0, 0, 0],
            [1, 0.25, 0.3],
            roughness=1,
            scale=10
        )

        self.blueWall = Rect(
            [8, 5, 0.1],
            [0, 0, -30],
            [0, 0, 0],
            [0.3, 0.25, 1],
            roughness=1,
            scale=10
        )

        self.greenWall = Rect(
            [8, 6, 0.1],
            [0, -25, 0],
            [90, 0, 0],
            [0.25, 1, 0.3],
            roughness=1,
            scale=10
        )

        self.backWall = Rect(
            [6, 8, 0.1],
            [-35, 0, 0],
            [0, 90, 0],
            [0.9, 0.9, 0.9],
            roughness=1,
            scale=10
        )

        self.frontWall = Rect(
            [6, 8, 0.1],
            [25, 0, 0],
            [0, 90, 0],
            [0.9, 0.9, 0.9],
            roughness=0,
            scale=10
        )

        self.floor = Rect(
            [8, 6, 0.1],
            [0, 25, 0],
            [90, 0, 0],
            [1, 1, 1],
            roughness=1,
            scale=10
        )

        self.light = Rect(
            [5, 5, 0.25],
            [0, 23.9, 0],
            [-90, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            1.5,
            scale=5
        )

        self.scene = Scene([
            self.dragon,
            self.sphere,
            self.redWall,
            self.blueWall,
            self.greenWall,
            self.frontWall,
            self.floor,
            self.light,
            self.backWall
        ])

        print("Initializing window...")

        pg.init()

        # request a 4.3 core context (SSBOs require GL 4.3+)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 4)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        pg.display.set_mode(screen_size, pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption("OpenGL raytracer")

        self.tileSizeX = window_size[0] // tileSize
        self.tileSizeY = window_size[1] // tileSize

        self.speed = 1

        self.sensitivity = 0.1

        self.canMove = False

        self.w, self.h = window_size
        self.sw, self.sh = screen_size

        self.aspect = self.sw / self.sh

        vert_src = read_shader(os.path.join("shaders", "vertex.glsl"))
        frag_src = read_shader(os.path.join("shaders", "fragment.glsl"))

        glBindVertexArray(glGenVertexArrays(1))

        self.shader = compileProgram(
            compileShader(vert_src, GL_VERTEX_SHADER),
            compileShader(frag_src, GL_FRAGMENT_SHADER)
        )

        self.scene.send()

        self.camPos = np.array([-33.7, 14.8, -21.1], dtype=np.float32)
        self.camDir = np.array([65, -25.4], dtype=np.float32)

        self.camRight, self.camForward, self.camUp = self.get_camera_basis(self.camDir)

        self.numTilesX = int((self.w + self.tileSizeX - 1) / self.tileSizeX)
        self.numTilesY = int((self.h + self.tileSizeY - 1) / self.tileSizeY)

        glUseProgram(self.shader)

        self.screen = Screen(self.w, self.h, self.sw, self.sh)

        self.clock = pg.time.Clock()

        # your camera & uniforms (kept same as your original code)
        self.fov = np.radians(90)
        self.dirStartX = -self.fov / 2 * self.aspect
        self.dirStartY = -self.fov / 2
        self.xStep = self.fov * self.aspect
        self.yStep = self.fov

        self.lambertian = lambertian

        # upload static uniforms (same names as in your shader)
        glUniform1f(glGetUniformLocation(self.shader, "fov"), self.fov)
        glUniform1f(glGetUniformLocation(self.shader, "dirStartX"), self.dirStartX)
        glUniform1f(glGetUniformLocation(self.shader, "dirStartY"), self.dirStartY)
        glUniform1f(glGetUniformLocation(self.shader, "xStep"), self.xStep)
        glUniform1f(glGetUniformLocation(self.shader, "yStep"), self.yStep)

        glUniform3fv(glGetUniformLocation(self.shader, "camPos"), 1, self.camPos)
        glUniform3fv(glGetUniformLocation(self.shader, "camRight"), 1, self.camRight)
        glUniform3fv(glGetUniformLocation(self.shader, "camUp"), 1, self.camUp)
        glUniform3fv(glGetUniformLocation(self.shader, "camForward"), 1, self.camForward)

        glUniform1i(glGetUniformLocation(self.shader, "nBounces"), bounces + 1)
        glUniform1i(glGetUniformLocation(self.shader, "rays_per_pixel"), rays_per_pixel)
        glUniform1f(glGetUniformLocation(self.shader, "jitterAmount"), jitter_amount)

        glUniform1i(glGetUniformLocation(self.shader, "lambertian"), self.lambertian)
        glUniform1f(glGetUniformLocation(self.shader, "skyBrightness"), skyIllumination)
        glUniform1i(glGetUniformLocation(self.shader, "total_frames"), 0)
        glUniform1i(glGetUniformLocation(self.shader, "trisCount"), self.scene.total_triangles)

        glUniform1i(glGetUniformLocation(self.shader, "tileX"), 0)
        glUniform1i(glGetUniformLocation(self.shader, "tileY"), 0)
        glUniform1i(glGetUniformLocation(self.shader, "tileSizeX"), self.tileSizeX)
        glUniform1i(glGetUniformLocation(self.shader, "tileSizeY"), self.tileSizeY)

        glUniform1i(glGetUniformLocation(self.shader, "numTilesX"), self.numTilesX)
        glUniform1i(glGetUniformLocation(self.shader, "numTilesY"), self.numTilesY)
        glUniform1i(glGetUniformLocation(self.shader, "boundingBoxCount"), self.scene.total_boxes)

        time.sleep(0.1)

        self.time_start = time.time()

        # run main loop
        self.main()

    def get_camera_basis(self, dir):
        # dir = (yaw_degrees, pitch_degrees)
        yaw_deg, pitch_deg = dir
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)

        # conventional forward: x = sin(yaw)*cos(pitch), y = sin(pitch), z = cos(yaw)*cos(pitch)
        forward = np.array([sy * cp, sp, cy * cp], dtype=np.float32)
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # compute right so +X points to the camera's right when yaw=0,pitch=0
        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)

        # camera up (orthonormal)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        # return in the order your caller expects: (camRight, camForward, camUp)
        return right, forward, up

    def get_time(self):
        delta = round(time.time() - self.time_start)

        h, rem = divmod(delta, 3600)
        m, s = divmod(rem, 60)

        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"

    def resetFrames(self):
        self.camRight, self.camForward, self.camUp = self.get_camera_basis(self.camDir)

        glUniform3fv(glGetUniformLocation(self.shader, "camPos"), 1, self.camPos)

        self.tileX = 0
        self.tileY = 0

        # reset accumulation
        self.screen.frame_count = 0
        self.screen.accum_index = 0

        for fbo in self.screen.accum_fbo:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.time_start = time.time()

    def main(self):
        running = True

        self.total_frames = 0

        self.frame_count = 0

        last_frame_time = time.time()

        tileX, tileY = 0, 0

        fps = 0
        deltaTime = 0

        time.sleep(1)

        while running:
            self.total_frames += 1

            keys = pg.key.get_pressed()
            delta = pg.mouse.get_rel()

            delta = np.array([delta[0], delta[1] * -1], dtype=np.float32) * self.canMove

            self.camDir += delta * self.sensitivity

            reset = False

            if keys[pg.K_w]:
                self.camPos += self.speed * self.camForward * self.canMove

                reset = True

            if keys[pg.K_s]:
                self.camPos -= self.speed * self.camForward * self.canMove

                reset = True

            if keys[pg.K_d]:
                self.camPos += self.speed * self.camRight * self.canMove

                reset = True

            if keys[pg.K_a]:
                self.camPos -= self.speed * self.camRight * self.canMove

                reset = True

            if keys[pg.K_e]:
                self.camPos += self.speed * self.camUp * self.canMove

                reset = True

            if keys[pg.K_q]:
                self.camPos -= self.speed * self.camUp * self.canMove

                self.resetFrames()

            if delta.any() > 0 or reset:
                glUniform3fv(glGetUniformLocation(self.shader, "camPos"), 1, self.camPos)
                glUniform3fv(glGetUniformLocation(self.shader, "camRight"), 1, self.camRight)
                glUniform3fv(glGetUniformLocation(self.shader, "camUp"), 1, self.camUp)
                glUniform3fv(glGetUniformLocation(self.shader, "camForward"), 1, self.camForward)

                self.resetFrames()

            for event in pg.event.get():
                keys = pg.key.get_pressed()

                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if keys[pg.K_m]:
                        self.canMove = not self.canMove

                        print("\n Can move" if self.canMove else "\nCan't move")

                        pg.mouse.set_visible(not self.canMove)
                        pg.event.set_grab(self.canMove)

                    if keys[pg.K_l]:
                        self.lambertian = not self.lambertian

                        print(f"\nSet lambertian lighting to {self.lambertian}")

                        glUniform1i(glGetUniformLocation(self.shader, "lambertian"), self.lambertian)

                        self.resetFrames()

                    if keys[pg.K_c]:
                        print("\nCamera info:")
                        print(f"Camera position: {self.camPos}")
                        print(f"Camera rotation: {self.camDir}")

                    if keys[pg.K_r]:
                        self.camDir = np.round(self.camDir / 5) * 5

                        self.resetFrames()

                    if event.key == pg.K_ESCAPE:
                        running = False

            # choose prev and next accumulation indices
            prev_idx = self.screen.accum_index
            next_idx = 1 - prev_idx

            # bind previous accumulation as prevFrame (texture unit 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.screen.accum_tex[prev_idx])

            glUniform1i(glGetUniformLocation(self.shader, "prevFrame"), 0)
            glUniform1i(glGetUniformLocation(self.shader, "frameNumber"), int(self.screen.frame_count))

            glUniform1i(glGetUniformLocation(self.shader, "tileX"), tileX)
            glUniform1i(glGetUniformLocation(self.shader, "tileY"), tileY)

            # render into the NEXT accumulation FBO (do not render into the texture we're reading from)
            glBindFramebuffer(GL_FRAMEBUFFER, self.screen.accum_fbo[next_idx])
            glViewport(0, 0, self.w, self.h)

            glBindVertexArray(self.screen.vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)

            # blit the result we just produced to the default framebuffer so the window shows it
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.screen.accum_fbo[next_idx])
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            glBlitFramebuffer(0, 0, self.w, self.h, 0, 0, self.sw, self.sh, GL_COLOR_BUFFER_BIT, GL_NEAREST)

            self.screen.accum_index = next_idx

            pg.display.flip()

            pg.display.set_caption("OpenGL raytracer! Fps: " + str(round(fps)) + " Frame: " + str(
                self.screen.frame_count) + " Frame render time: " + str(
                round(deltaTime * 1000)) + "ms" + " Total render time: " + self.get_time())

            tileX += 1

            if tileX >= self.numTilesX:
                tileY += 1
                tileX = 0

                if tileY >= self.numTilesY:
                    tileY = 0

                    self.screen.frame_count += 1

                    deltaTime = time.time() - last_frame_time

                    if deltaTime > 0:
                        fps = 1 / deltaTime

                    else:
                        fps = 0

                    last_frame_time = time.time()

                    glUniform1i(glGetUniformLocation(self.shader, "total_frames"), self.total_frames)

        screen = pg.display.get_surface()
        size = screen.get_size()
        buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
        screen_surf = pg.image.fromstring(buffer, size, "RGBA")
        screen_surf = pg.transform.rotate(screen_surf, 180)

        if time.time() - self.time_start > 10 * 60:
            pg.image.save(screen_surf, f"render_{self.get_time()}.png")

        # cleanup
        self.screen.delete()
        self.scene.clearMemory()
        glDeleteProgram(self.shader)
        pg.quit()

if __name__ == "__main__":
    rays_per_pixel = 1
    bounces = 7
    jitter_amount = 0.001
    lambertian = True
    skyBrightness = 1
    window_size = 1080
    tileSize = 1

    window = tk.Tk()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    aspect = screen_width / screen_height

    if window_size < screen_height:
        screen_size = np.array((int(screen_width // 1.15), int(screen_height // 1.15)))

    else:
        screen_size = np.array([window_size * aspect, window_size], dtype=int)

    window.destroy()

    window_size = np.array([window_size * aspect, window_size], dtype=int)

    App(window_size, screen_size, bounces, rays_per_pixel, jitter_amount, lambertian, skyBrightness, tileSize)