import math
import taichi as ti
from taichi.math import *
import numpy as np

from renderer.math_utils import *

@ti.dataclass
class StorageReservoir:
    M : ti.f16
    W : ti.f16
    F : vec3
    rc_pos : vec3
    rc_normal_and_NEE_dir : ti.u32
    rc_incident_dir : ti.types.vector(2, ti.f16)
    rc_incident_L : vec3
    rc_mat_info : ti.u32
    cached_jacobian_term : ti.f16
    lobes : ti.i8


@ti.dataclass
class Sample:
    F : vec3
    rc_pos : vec3 # when rc vertex is an escape vertex, this is a direction instead

    rc_normal : vec3 # when zero, that means rc vertex is an escape vertex

    rc_incident_dir : vec3 # when zero, that means path terminated here

    rc_incident_L : vec3 # when rc vertex is an escape vertex, this is the sky/NEE colour

    rc_NEE_dir : vec3 # when zero, that means NEE visibility is zero

    rc_mat_info : ti.u32 # stores mat_id an albedo sequentially in the bits

    cached_jacobian_term : ti.f32
    lobes : ti.i32 # lobe indices bsdf sampling at x1 and x_rc

@ti.dataclass
class Reservoir:
    z:Sample
    M:float
    weight:float

    @ti.func
    def init(self):
        self.z = Sample(F=vec3(0,0,0), \
                        rc_pos=vec3(0,0,0), \
                        rc_normal=vec3(0,0,0), \
                        rc_incident_dir=vec3(0,0,0), \
                        rc_incident_L=vec3(0,0,0), \
                        rc_NEE_dir=vec3(0,0,0), \
                        rc_mat_info=0, \
                        cached_jacobian_term=1.0, lobes = 0)
        self.M = 0.0
        self.weight = 0.0

    @ti.func
    def update_cached_jacobian_term(self, x1):
        dir_x1_to_x2 = (self.z.rc_pos - x1)
        self.z.cached_jacobian_term = dir_x1_to_x2.dot(dir_x1_to_x2)/abs(dir_x1_to_x2.normalized().dot(self.z.rc_normal))

    @ti.func
    def input_sample(self, in_w, in_z, force_add = False):
        self.M += 1

        selected = False
        if(in_w > 0.):
            self.weight += in_w
            selected = ti.random() * self.weight <= in_w or force_add
            if selected:
                self.z = in_z
        return selected

    @ti.func
    def merge(self, in_r, in_w, force_add = False):
        self.M += in_r.M

        selected = False
        if(in_w > 0.):
            self.weight += in_w
            selected = ti.random() * self.weight <= in_w or force_add
            if selected:
                self.z = in_r.z
        return selected

    @ti.func
    def finalize(self):
        p_hat = luminance(self.z.F)
        if p_hat < 1e-6:
            self.weight = 0.0
        else:
            self.weight = self.weight / (p_hat*self.M)

    @ti.func
    def finalize_without_M(self):
        p_hat = luminance(self.z.F)
        if p_hat < 1e-6:
            self.weight = 0.0
        else:
            self.weight = self.weight / (p_hat)

    @ti.func
    def encode(self):
        enc = StorageReservoir()
        enc.M = ti.cast(self.M, ti.f16)
        enc.W = ti.cast(self.weight, ti.f16)
        enc.F = self.z.F
        enc.rc_pos = self.z.rc_pos

        oct_rc_normal = encode_unit_vector_3x16(self.z.rc_normal)
        oct_rc_NEE_dir = encode_unit_vector_3x16(self.z.rc_NEE_dir)
        enc.rc_normal_and_NEE_dir = encode_u32_arb(ti.Vector([oct_rc_normal.x, oct_rc_normal.y, \
                                                              oct_rc_NEE_dir.x, oct_rc_NEE_dir.y]), \
                                                   ti.Vector([8, 8, 8, 8]))
        
        enc.rc_incident_dir = encode_unit_vector_3x16(self.z.rc_incident_dir)
        enc.rc_incident_L = self.z.rc_incident_L
        enc.rc_mat_info = self.z.rc_mat_info
        enc.cached_jacobian_term = ti.cast(self.z.cached_jacobian_term, ti.f16)
        enc.lobes = ti.cast(self.z.lobes, ti.i8)

        return enc
    
    @ti.func
    def decode(self, enc):
        self.M = ti.cast(enc.M, ti.f32)
        self.weight = ti.cast(enc.W, ti.f32)
        self.z.F = enc.F
        self.z.rc_pos = enc.rc_pos

        data = decode_u32_arb(enc.rc_normal_and_NEE_dir, ti.Vector([8, 8, 8, 8]))
        self.z.rc_normal = decode_unit_vector_3x16(data.xy)
        self.z.rc_NEE_dir = decode_unit_vector_3x16(data.zw)

        self.z.rc_incident_dir = decode_unit_vector_3x16(enc.rc_incident_dir)
        self.z.rc_incident_L = enc.rc_incident_L
        self.z.rc_mat_info = enc.rc_mat_info
        self.z.cached_jacobian_term = ti.cast(enc.cached_jacobian_term, ti.f32)
        self.z.lobes = ti.cast(enc.lobes, ti.i32)



        
# NOTES
# when doing reconnection shift:
#   multiply Jacobian determinant by ratio of dstPdf / srcPdf for both x_1 and x_rc
#   when computing shifted integrand, it is brdf of first bounce times brdf/pdf of rc bounce times rc_L