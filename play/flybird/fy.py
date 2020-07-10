import taichi as ti
import random

ti.init(arch=ti.cpu)

image_w, image_h = 256, 256

pos = [0.3, 0.5]
speed = -0.0
size = 0.05
space_h = size * 8
block_range = [0.1, 0.9 - space_h]
map_speed = -0.006
g = -0.0001
block_w = 0.05
block_count = 3
blocks = ti.Vector(4, ti.f32, shape=(block_count))
score = 0

def CreateABlock(idx):
    _block = blocks[idx]
    t1 = random.random() * (block_range[1] - block_range[0])
    _block[0] = t1
    _block[1] = t1 + space_h
    _block[2] = 1.0

def check():
    if pos[1] < -size:
        return True
    for idx in range(block_count):
        _block = blocks[idx]
        if pos[0] + size >= _block[2]: # and pos[0] + size 
            if pos[1] - size < _block[0]: # or pos[0]
                return True
    return False

def MapMove():
    for idx in range(block_count):
        global score
        _block = blocks[idx]
        _block[2] += map_speed
        if _block[2] + block_w < pos[0] - size: # and
            score += 1
            _block[3] = -1
        if _block[2]  < -block_w:
            CreateABlock(idx)

for idx in range(block_count):
    CreateABlock(idx)
    blocks[idx][2] = 1.0 + 0.33 * idx

gui = ti.GUI("mirai", res = (image_w, image_h), background_color=0x201840)

for frame in range(10000):
    if gui.get_event((ti.GUI.PRESS, ti.GUI.SPACE)):
        speed += 0.01
    if speed > 0.1:
        speed = 0.1
    if speed < -0.1:
        speed = -0.1
    pos[1] += speed
    if pos[1] + size > 1:
        pos[1] = 1.0 - size
        if speed > 0: speed = 0
    speed += g
    MapMove()

    if check():
        break
    gui.clear(0x201840)
    gui.circle(pos, 0xFFFFFF, size*image_h)
    for idx in range(block_count):
        _block = blocks[idx]
        gui.rect([_block[2], 0], [_block[2]+block_w, _block[0]])
        gui.rect([_block[2], 1], [_block[2]+block_w, _block[1]])
    gui.text(f"{score}", (0.05, 0.95))
    gui.show()

while not gui.get_event(ti.GUI.ESCAPE):
    gui.clear(0x401820)
    gui.text(f"game over {score}", (0.3, 0.6))
    gui.show()
                 










