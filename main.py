import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import os
import math
import random
import time
from screen import Screen
from object import *
import tkinter as tk
from scene import Scene

def read_shader(path):
    with open(path, "r") as f:
        return f.read()

class App:
    def __init__(self, window_size, screen_size, bounces, rays_per_pixel, jitter_amount, lambertian, skyIllumination):
        pg.init()

        # request a 4.3 core context (SSBOs require GL 4.3+)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 4)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

        pg.display.set_mode(screen_size, pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption("OpenGL raytracer")

        self.speed = 1

        self.sensitivity = 0.1

        self.canMove = False

        self.w, self.h = window_size
        self.sw, self.sh = screen_size

        self.aspect = self.sw / self.sh

        glViewport(0, 0, self.w, self.h)
        glDisable(GL_DEPTH_TEST)

        vert_src = read_shader(os.path.join("shaders", "vertex.glsl"))
        frag_src = read_shader(os.path.join("shaders", "fragment.glsl"))

        glBindVertexArray(glGenVertexArrays(1))

        self.shader = compileProgram(
            compileShader(vert_src, GL_VERTEX_SHADER),
            compileShader(frag_src, GL_FRAGMENT_SHADER)
        )

        glUseProgram(self.shader)

        self.screen = Screen(self.w, self.h, self.sw, self.sh)

        self.knight = Mesh(
            [0, -24, 0],
            [270, 0, -90],
            "knight",
            [0.75, 0.75, 0.75],
            scale=7
        )

        self.redWall = Rect(
            [5, 5, 0.1],
            [0, 0, -25],
            [0, 0, 0],
            [1, 0, 0],
            scale=10
        )

        self.blueWall = Rect(
            [5, 5, 0.1],
            [0, 0, 25],
            [0, 0, 0],
            [0, 0, 1],
            scale=10
        )

        self.greenWall = Rect(
            [5, 5, 0.1],
            [25, 0, 0],
            [0, 90, 0],
            [0, 1, 0],
            scale=10
        )

        self.ground = Rect(
            [5, 5, 0.1],
            [0, -25, 0],
            [90, 0, 0],
            [1, 1, 1],
            scale=10
        )

        self.floor = Rect(
            [5, 5, 0.1],
            [0, 25, 0],
            [90, 0, 0],
            [1, 1, 1],
            scale=10
        )

        self.light = Rect(
            [5, 5, 0.25],
            [0, 23.9, 0],
            [-90, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            2,
            scale=4
        )

        self.scene = Scene([
            self.knight,
            self.redWall,
            self.blueWall,
            self.greenWall,
            self.ground,
            self.floor,
            self.light
        ])

        self.clock = pg.time.Clock()

        # your camera & uniforms (kept same as your original code)
        self.fov = np.radians(60)
        self.dirStartX = -self.fov / 2 * self.aspect
        self.dirStartY = -self.fov / 2
        self.xStep = self.fov * self.aspect
        self.yStep = self.fov

        self.lambertian = lambertian

        # camera pos and dir (kept same)
        self.camPos = np.array([-73, 0, -1], dtype=np.float32)
        self.camDir = np.array([90, 0], dtype=np.float32)

        self.camRight, self.camForward, self.camUp = self.get_camera_basis(self.camDir)

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
        glUniform1i(glGetUniformLocation(self.shader, "trisCount"), len(self.knight.total_vertices))

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
        # reset accumulation
        self.screen.frame_count = 0
        self.screen.accum_index = 0
        for fbo in self.screen.accum_fbo:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glViewport(0, 0, self.w, self.h)
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.time_start = time.time()

    def main(self):
        running = True

        self.total_frames = 0

        last_frame_time = time.time()

        while running:
            self.total_frames += 1

            keys = pg.key.get_pressed()
            delta = pg.mouse.get_rel()

            delta = np.array([delta[0], delta[1] * -1], dtype=np.float32) * self.canMove

            self.camDir += delta * self.sensitivity

            self.camRight, self.camForward, self.camUp = self.get_camera_basis(self.camDir)

            if keys[pg.K_w]:
                self.camPos += self.speed * self.camForward * self.canMove

                self.resetFrames()

            if keys[pg.K_s]:
                self.camPos -= self.speed * self.camForward * self.canMove

                self.resetFrames()

            if keys[pg.K_d]:
                self.camPos += self.speed * self.camRight * self.canMove

                self.resetFrames()

            if keys[pg.K_a]:
                self.camPos -= self.speed * self.camRight * self.canMove

                self.resetFrames()

            if keys[pg.K_e]:
                self.camPos += self.speed * self.camUp * self.canMove

                self.resetFrames()

            if keys[pg.K_q]:
                self.camPos -= self.speed * self.camUp * self.canMove

                self.resetFrames()

            if delta.any() > 0:
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
            loc_prev = glGetUniformLocation(self.shader, "prevFrame")

            if loc_prev != -1:
                glUniform1i(loc_prev, 0)

            # set frameNumber (0-based) — shader expects this
            loc_fn = glGetUniformLocation(self.shader, "frameNumber")
            if loc_fn != -1:
                glUniform1i(loc_fn, int(self.screen.frame_count))

            # set frame seed (if your shader reads 'frame' for random)
            loc_f = glGetUniformLocation(self.shader, "frame")
            if loc_f != -1:
                glUniform1f(loc_f, random.uniform(0.0, 1.0))

            glUniform3fv(glGetUniformLocation(self.shader, "camPos"), 1, self.camPos)
            glUniform3fv(glGetUniformLocation(self.shader, "camRight"), 1, self.camRight)
            glUniform3fv(glGetUniformLocation(self.shader, "camUp"), 1, self.camUp)
            glUniform3fv(glGetUniformLocation(self.shader, "camForward"), 1, self.camForward)
            glUniform1i(glGetUniformLocation(self.shader, "total_frames"), self.total_frames)

            # render into the NEXT accumulation FBO (do not render into the texture we're reading from)
            glBindFramebuffer(GL_FRAMEBUFFER, self.screen.accum_fbo[next_idx])
            glViewport(0, 0, self.w, self.h)

            glBindVertexArray(self.screen.vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)

            # blit the result we just produced to the default framebuffer so the window shows it
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.screen.accum_fbo[next_idx])
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            glBlitFramebuffer(0, 0, self.w, self.h, 0, 0, self.sw, self.sh, GL_COLOR_BUFFER_BIT, GL_NEAREST)

            # swap and increment
            self.screen.accum_index = next_idx
            self.screen.frame_count += 1

            pg.display.flip()

            deltaTime = time.time() - last_frame_time

            fps = 1 / deltaTime

            last_frame_time = time.time()

            pg.display.set_caption("OpenGL raytracer! Fps: " + str(round(fps)) + " Frame: " + str(self.screen.frame_count) + " Frame render time: " + str(round(deltaTime * 1000)) + "ms" + " Total render time: " + self.get_time())

        screen = pg.display.get_surface()
        size = screen.get_size()
        buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
        screen_surf = pg.image.fromstring(buffer, size, "RGBA")
        screen_surf = pg.transform.rotate(screen_surf, 180)

        if time.time() - self.time_start > 1 * 60:
            pg.image.save(screen_surf, f"render_{self.get_time()}.png")

        # cleanup
        self.screen.delete()
        self.scene.clearMemory()
        glDeleteProgram(self.shader)
        pg.quit()

if __name__ == "__main__":
    rays_per_pixel = 1
    bounces = 100
    jitter_amount = 0.0005
    lambertian = True
    skyBrightness = 0
    window_size = np.array([1000, 700])

    window = tk.Tk()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    screen_size = (int(screen_width // 1.15), int(screen_height // 1.15))
    window.destroy()

    App(window_size, screen_size, bounces, rays_per_pixel, jitter_amount, lambertian, skyBrightness)