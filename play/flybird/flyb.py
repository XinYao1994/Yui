"""
success:       score       [reward = +1].
block_failure: score       [reward = -score].
"""
import taichi as ti
import random
import numpy as np
import time
import sys
import math

ti.init(arch=ti.cpu)

image_w, image_h = 512, 512

size = 0.02
block_count = 3
a = 0.005
g = -0.0001
map_speed = -0.006
block_w = 0.02
space_h = size * 20
block_range = [0.1, 0.9 - space_h]

class FlapBrid(object):
    '''
    def __init__(self):
        self.blocks = ti.Vector(4, ti.f32, shape=(block_count))
        self.action_space = ['s', 'd'] # "u"
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self._build_flapbrid()
    '''

    def __init__(self, RENDER=True):
        if RENDER:
            self.gui = ti.GUI("mirai", res = (image_w, image_h), background_color=0x201840)
        self.blocks = ti.Vector(4, ti.f32, shape=(block_count))
        self.action_space = ['s', 'd'] # "u"
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self._build_flapbrid()

    def CreateABlock(self, idx):
        _block = self.blocks[idx]
        t1 = random.random() * (block_range[1] - block_range[0])
        _block[0] = t1
        _block[1] = t1 + space_h
        _block[2] = 1.0
        _block[3] = 0.0

    def _build_flapbrid(self):
        self.pos = [0.3, 0.5]
        self.speed = 0.0
        self.score = 0
        for idx in range(block_count):
            self.CreateABlock(idx)
            self.blocks[idx][2] = 1.0 + 0.33 * idx

    def reset(self):
        self.op = 0
        self.pos = [0.3, 0.5]
        self.speed = 0.0
        self.score = 0
        for idx in range(block_count):
            self.CreateABlock(idx)
            self.blocks[idx][2] = 1.0 + 0.33 * idx
        # return state
        for idx in range(block_count):
            if self.blocks[idx][3] == 0.0:
                break
        state = np.array([self.pos[0] - self.blocks[idx][2], self.pos[1] - 0.5*self.blocks[idx][1] - 0.5*self.blocks[idx][0]])
        return state

    def step(self, action):
        panity = 1
        if action == 1: # self.gui.get_event((ti.GUI.PRESS, ti.GUI.UP))
            self.speed += a
        if self.speed > 0.1:
            self.speed = 0.1
        if self.speed < -0.1:
            self.speed = -0.1
        self.pos[1] += self.speed
        self.speed += g
        self.MapMove()
        # state_ = ??
        for idx in range(block_count):
            if self.blocks[idx][3] == 0.0:
                break
        state_ = np.array([self.pos[0] - self.blocks[idx][2], self.pos[1] - 0.5*self.blocks[idx][1] - 0.5*self.blocks[idx][0]])
        dis_ = (state_[0]**2 + state_[1]**2)**0.5
        if state_[1] > 0:
            panity = 2500
        if self.check():
            reward = -50 - self.score - math.log(1/(dis_*panity))
            done = True
        else:
            reward = 10 + math.log(1/(dis_*panity))
            done = False
        return state_, reward, done

    def check(self):
        if self.pos[1] < -size or self.pos[1] > 1 + size:
            return True
        for idx in range(block_count):
            _block = self.blocks[idx]
            if self.pos[0] + size >= _block[2] and self.pos[0] - size <= _block[2] + block_w: # 
                if self.pos[1] - size < _block[0] or self.pos[1] + size > _block[1]: # 
                    return True
        return False

    def MapMove(self):
        for idx in range(block_count):
            _block = self.blocks[idx]
            _block[2] += map_speed
            if _block[2] + block_w < self.pos[0] - size and _block[3] != -1: # and
                self.score += 1
                _block[3] = -1
            if _block[2]  < -block_w:
                self.CreateABlock(idx)

    def render(self):
        self.gui.clear(0x201840)
        self.gui.circle(self.pos, 0xFFFFFF, size*image_h)
        for idx in range(block_count):
            _block = self.blocks[idx]
            self.gui.rect([_block[2], 0], [_block[2]+block_w, _block[0]])
            self.gui.rect([_block[2], 1], [_block[2]+block_w, _block[1]])
        self.gui.text(f"{self.score}", (0.05, 0.95))
        self.gui.show()


