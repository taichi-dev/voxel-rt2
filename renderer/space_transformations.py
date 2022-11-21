import taichi as ti
from taichi.math import *
import numpy as np
from renderer.math_utils import *

@ti.func
def linearize_depth(depth, inv_proj_mat):
    return 1.0 / ((depth * 2.0 - 1.0) * inv_proj_mat[3, 2] + inv_proj_mat[3, 3])

@ti.func
def delinearize_depth(lindepth, proj_mat):
    return ((-lindepth * proj_mat[2, 2] + proj_mat[2, 3]) / -lindepth) * -0.5 + 0.5

@ti.func
def screen_to_view(uv, depth, inv_proj_mat): # here, depth is non-linear
    pos = ti.Vector([uv.x, uv.y, depth, 1.0])
    pos.xyz = pos.xyz * 2.0 - 1.0 # bring into ND space
    pos = inv_proj_mat @ pos
    pos.xyz /= pos.w
    return pos.xyz

@ti.func
def view_to_screen(view_pos, proj_mat):
    pos = proj_mat @ ti.Vector([view_pos.x, view_pos.y, view_pos.z, 1.0])
    pos.xyz /= pos.w
    return pos.xyz * 0.5 + 0.5

@ti.func
def view_to_world(pos, inv_view_mat, is_position = 1.0):
    return (inv_view_mat @ ti.Vector([pos.x, pos.y, pos.z, is_position])).xyz

@ti.func
def world_to_view(pos, view_mat, is_position = 1.0):
    return (view_mat @ ti.Vector([pos.x, pos.y, pos.z, is_position])).xyz