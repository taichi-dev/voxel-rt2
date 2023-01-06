
from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=3.3)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
# scene.set_background_color((0.5, 0.5, 0.4)) # (0.5, 0.5, 0.4)
scene.set_directional_light((-0.8, 1.3, -1), 0.025, (1.0, 0.949, 0.937)) # (1, 0.8, 0.6)
scene.set_use_physical_sky(True)
scene.set_use_clouds(True)

scale = 4
offset = ivec3(-60, 0, -60)
brick_noise = vec3(0.05)
wood_noise = vec3(0.08)
stone_noise = vec3(0.08)
pillar_noise = vec3(0.2)
metal_noise = vec3(0.01)

@ti.func
def create_air(pos, size):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]),
                       (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, 0, vec3(0., 0., 0.))

@ti.func
def create_brick(pos, size):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]),
                       (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, 10, vec3(130.0, 87.0, 73.0) / 255.0 - 0.15 - brick_noise * ti.random())

@ti.func
def create_wood(pos, size):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]),
                       (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, 31, vec3(183.0, 150.0, 91.0) / 255.0 + wood_noise * ti.random())

@ti.func
def create_dark_wood(pos, size):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]),
                       (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, 31, vec3(183.0, 150.0, 91.0) * 0.5 / 255.0 + wood_noise * ti.random())

@ti.func
def create_stone(pos, size):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]),
                       (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, 21, vec3(246.0, 237.0, 226.0) / 255.0 + stone_noise * ti.random())

@ti.func
def create_pillar(pos, is_corner):
    for x in range(pos[0], pos[0] + scale):
        for z in range(pos[2], pos[2] + scale):
            color = vec3(246.0, 237.0, 226.0) / 255.0 - pillar_noise * (1.0 if (x+z) % 2 == 0 else 0.0)
            for y in range(pos[1], pos[1] + scale*4):
                scene.set_voxel(ivec3(x,y,z), 21, color)

    create_stone(pos + ivec3(0, 7, -2)*scale//2, ivec3(scale, scale // 2, scale))
    create_stone(pos + ivec3(0, 6, -1)*scale//2, ivec3(scale, scale // 2, scale // 2))

    create_stone(pos + ivec3(0, 7,  2)*scale//2, ivec3(scale, scale // 2, scale))
    create_stone(pos + ivec3(0, 6,  2)*scale//2, ivec3(scale, scale // 2, scale // 2))

    if is_corner:
        create_stone(pos + ivec3(-2, 7, 0)*scale//2, ivec3(scale, scale // 2, scale))
        create_stone(pos + ivec3(-1, 6, 0)*scale//2, ivec3(scale // 2, scale // 2, scale))

        create_stone(pos + ivec3(2, 7, 0)*scale//2, ivec3(scale, scale // 2, scale))
        create_stone(pos + ivec3(2, 6, 0)*scale//2, ivec3(scale // 2, scale // 2, scale))



@ti.func
def create_metal(pos, size):
    for I in ti.grouped(
            ti.ndrange((pos[0], pos[0] + size[0]), (pos[1], pos[1] + size[1]),
                       (pos[2], pos[2] + size[2]))):
        scene.set_voxel(I, 50, vec3(0.9, 0.9, 0.9) + metal_noise * ti.random())





#  _____9_____
# |    ___    |
# | O |   | O |
# |   |   |   |
# |   |   |   |
# |   |   |   | 
# |   |   |   |
# | O |   | O |
# |   |   |   |
# |   |   |   |
# |   |   |   |
# |   |   |   | 25
# | O |   | O |
# |   |   |   |
# |   |   |   |
# |   |   |   |
# |   |   |   |
# | O |   | O |
# |   |   |   |
# |   |   |   |
# |   |   |   |
# |   |   |   |
# | O |_3_| O |
# |___________|
#
# pillar height: 4
# full height: 8

@ti.kernel
def initialize_voxels():

    # outer walls
    create_brick(ivec3(-1, 1, -1)*scale + offset, ivec3(11, 9, 27)*scale) # initial prism
    create_air(ivec3(0, 2, 0)*scale + offset, ivec3(9, 4, 25)*scale) # lower gap

    create_air(ivec3(2, 6, 3)*scale + offset, ivec3(5, 4, 19)*scale) # upper gap
    
    # floor
    create_stone(ivec3(0, 1, 0)*scale + offset, ivec3(9, 1, 25)*scale)
    create_metal(ivec3(3, 1, 2)*scale + offset, ivec3(3, 1, 21)*scale)

    # frames
    create_brick(ivec3(2, 2, 1)*scale + offset, ivec3(5, 4, 1)*scale) # frame
    create_air(ivec3(3, 2, 1)*scale + offset, ivec3(3, 3, 1)*scale) # frame gap

    create_brick(ivec3(2, 2, 23)*scale + offset, ivec3(5, 4, 1)*scale) # same thing ^
    create_air(ivec3(3, 2, 23)*scale + offset, ivec3(3, 3, 1)*scale)

    # doors
    create_wood(ivec3(3, 2, 0)*scale + offset, ivec3(3, 3, 1)*scale)
    create_dark_wood(ivec3(4, 2, 0)*scale + offset, ivec3(1, 2, 1)*scale)
    create_air(ivec3(8, 4, 1)*scale//2 + offset, ivec3(2, 4, 1)*scale//2)

    create_wood(ivec3(3, 2, 24)*scale + offset, ivec3(3, 3, 1)*scale)
    create_dark_wood(ivec3(4, 2, 24)*scale + offset, ivec3(1, 2, 1)*scale)
    create_air(ivec3(8, 4, 48)*scale//2 + offset, ivec3(2, 4, 1)*scale//2)


    # pillars
    for i in range(0, 5):
        create_pillar(ivec3(1, 2, 2 + i*5)*scale + offset, i == 0 or i == 4)
        create_pillar(ivec3(7, 2, 2 + i*5)*scale + offset, i == 0 or i == 4)

    


initialize_voxels()

scene.finish()
