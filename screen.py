import numpy as np
from OpenGL.GL import *

class Screen:
    def __init__(self, width, height, sw, sh):
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
            glViewport(0, 0, sw, sh)
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