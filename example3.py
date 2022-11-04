from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=30)
scene.set_directional_light((-1, 1, -1), 0.2, (0.1, 0.1, 0.1)) # (1, 0.8, 0.6)
scene.set_floor(0, (1.0, 1.0, 1.0))

n = 50


@ti.kernel
def initialize_voxels():
    for i in range(n):
        for j in range(n):
            scene.set_voxel(vec3(0, i, j), 1, vec3(0.9, 0.3, 0.3)) # left wall
            scene.set_voxel(vec3(n, i, j), 1, vec3(0.3, 0.9, 0.3)) # right wall
            scene.set_voxel(vec3(i, n, j), 11, vec3(1, 1, 1)) # ceiling
            scene.set_voxel(vec3(i, 0, j), 11, vec3(0.0, 0.0, 0.0)) # floor
            scene.set_voxel(vec3(i, j, 0), 51, vec3(1, 1, 1)) # back wall
            # scene.set_voxel(vec3(i, j, n), 1, vec3(0.3, 0.3, 0.9)) # front wall
            # scene.set_voxel(vec3(i, j, 0), 51 if j < n/2 else 51, vec3(1, 1, 1))

    for i in range(-n // 8, n // 8):
        for j in range(-n // 8, n // 8):
            scene.set_voxel(vec3(i + n // 2, n-1, j + n // 2), 2, vec3(1, 1, 1))

    # for i_ in range(n // 8 * 3):
    #     i = i_ * 2
    #     for j in range(n // 4 * 3):
    #         scene.set_voxel(
    #             vec3(j + n // 8, n // 4 + ti.sin(
    #                 (i + j) / n * 30) * 0.05 * n + i / 10, -i + n // 8 * 7), 1,
    #             vec3(0.3, 0.3, 0.9))


initialize_voxels()
scene.finish()
