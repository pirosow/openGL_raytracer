import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import ctypes
import os
import math
import random
import time

SHADER_DIR = "shaders"

def read_shader(path):
    with open(path, "r") as f:
        return f.read()

class Screen:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # fullscreen quad x,y,z,u,v
        self.vertices = np.array([
            -1, -1, 0, 0, 0,
            -1,  1, 0, 0, 1,
             1,  1, 0, 1, 1,

            -1, -1, 0, 0, 0,
             1, -1, 0, 1, 0,
             1,  1, 0, 1, 1
        ], dtype=np.float32)

        # VAO / VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # pos (location 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

        # uv (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))

        # create two accumulation textures + fbos (ping-pong)
        self.accum_tex = [glGenTextures(1), glGenTextures(1)]
        self.accum_fbo = [glGenFramebuffers(1), glGenFramebuffers(1)]

        for i in range(2):
            glBindTexture(GL_TEXTURE_2D, self.accum_tex[i])
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

            glBindFramebuffer(GL_FRAMEBUFFER, self.accum_fbo[i])
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.accum_tex[i], 0)

        # make sure fbos are complete
        for fbo in self.accum_fbo:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        # clear the accumulation textures to zero (safe initial state)
        for fbo in self.accum_fbo:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glViewport(0, 0, width, height)
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT)

        # reset binding
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.frame_count = 0
        self.accum_index = 0   # index of texture containing the "previous accumulation"

    def delete(self):
        glDeleteTextures(self.accum_tex)
        glDeleteBuffers(1, (self.vbo,))
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteFramebuffers(2, self.accum_fbo)

class App:
    def __init__(self, window_size, bounces, rays_per_pixel, jitter_amount):
        pg.init()

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)

        pg.display.set_mode(window_size, pg.OPENGL | pg.DOUBLEBUF)
        pg.display.set_caption("OpenGL - Progressive (ping-pong)")

        self.w, self.h = window_size
        self.aspect = self.w / self.h

        glViewport(0, 0, self.w, self.h)
        glDisable(GL_DEPTH_TEST)

        self.screen = Screen(self.w, self.h)

        vert_src = read_shader(os.path.join(SHADER_DIR, "vertex.glsl"))
        frag_src = read_shader(os.path.join(SHADER_DIR, "fragment.glsl"))

        self.shader = compileProgram(
            compileShader(vert_src, GL_VERTEX_SHADER),
            compileShader(frag_src, GL_FRAGMENT_SHADER)
        )

        glUseProgram(self.shader)

        self.clock = pg.time.Clock()

        # your camera & uniforms (kept same as your original code)
        self.fov = np.radians(60)
        self.dirStartX = -self.fov / 2 * self.aspect
        self.dirStartY = -self.fov / 2
        self.xStep = self.fov * self.aspect
        self.yStep = self.fov

        # camera pos and dir (kept same)
        self.camPos = np.array([-40, 50, -85.0], dtype=np.float32)
        self.camDir = [5, -15]

        # your get_camera_basis returns (forward,right,up) — keep assignment as you had it to avoid changing logic
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

        # run main loop
        self.main()

    def get_camera_basis(self, dir):
        # dir = (yaw_degrees, pitch_degrees)
        yaw_deg, pitch_deg = dir
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        cy = math.cos(yaw);
        sy = math.sin(yaw)
        cp = math.cos(pitch);
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

    def main(self):
        running = True
        clock = pg.time.Clock()

        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    if event.key == pg.K_r:
                        # reset accumulation
                        self.screen.frame_count = 0
                        self.screen.accum_index = 0
                        for fbo in self.screen.accum_fbo:
                            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
                            glViewport(0, 0, self.w, self.h)
                            glClearColor(0.0,0.0,0.0,0.0)
                            glClear(GL_COLOR_BUFFER_BIT)
                        glBindFramebuffer(GL_FRAMEBUFFER, 0)
                        print("Accumulation reset")

            # choose prev and next accumulation indices
            prev_idx = self.screen.accum_index
            next_idx = 1 - prev_idx

            glUseProgram(self.shader)

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

            # render into the NEXT accumulation FBO (do not render into the texture we're reading from)
            glBindFramebuffer(GL_FRAMEBUFFER, self.screen.accum_fbo[next_idx])
            glViewport(0, 0, self.w, self.h)

            glBindVertexArray(self.screen.vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)

            # blit the result we just produced to the default framebuffer so the window shows it
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.screen.accum_fbo[next_idx])
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            glBlitFramebuffer(0, 0, self.w, self.h, 0, 0, self.w, self.h, GL_COLOR_BUFFER_BIT, GL_NEAREST)

            # swap and increment
            self.screen.accum_index = next_idx
            self.screen.frame_count += 1

            pg.display.flip()

            self.clock.tick(300)

            pg.display.set_caption("OpenGL raytracer! Fps: " + str(round(self.clock.get_fps())) + " Frame: " + str(self.screen.frame_count))

        screen = pg.display.get_surface()
        size = screen.get_size()
        buffer = glReadPixels(0, 0, *size, GL_RGBA, GL_UNSIGNED_BYTE)
        screen_surf = pg.image.fromstring(buffer, size, "RGBA")
        screen_surf = pg.transform.rotate(screen_surf, 180)
        pg.image.save(screen_surf, "render.png")

        # cleanup
        self.screen.delete()
        glDeleteProgram(self.shader)
        pg.quit()

if __name__ == "__main__":
    rays_per_pixel = 1
    bounces = 20
    jitter_amount = 0.0001

    App((1440, 900), bounces, rays_per_pixel, jitter_amount)