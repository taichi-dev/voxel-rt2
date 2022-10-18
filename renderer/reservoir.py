import math
import taichi as ti
from taichi.math import *
import numpy as np

@ti.dataclass
class StorageSample:
    rc_pos : vec3
    rc_normal : ti.types.vector(2, ti.f16)
    rc_incident_dir : ti.types.vector(2, ti.f16)
    rc_incident_L : vec3
    rc_NEE_dir : ti.types.vector(2, ti.f16)
    rc_mat_info : ti.u32
    cached_jacobian_term : ti.f16 # When I do encoding for reservoirs, put cached_jacobian_term and lobes into one uint32


@ti.dataclass
class Sample:
    L : vec3
    rc_pos : vec3 # when rc vertex is an escape vertex, this is a direction instead
    rc_normal : vec3 # when zero, that means rc vertex is an escape vertex
    rc_incident_dir : vec3 # when zero, that means path terminated here
    rc_incident_L : vec3
    rc_NEE_dir : vec3 # when zero, that means NEE visibility is zero
    rc_mat_info : ti.u32
    cached_jacobian_term : ti.f32
    lobes : ti.i32 # lobe indices bsdf sampling at x1 and x_rc


@ti.dataclass
class Reservoir:
    z:Sample
    M:float
    weight:float

    @ti.func
    def init(self):
        self.z = Sample(L=vec3(0,0,0), \
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
    def merge(self, in_r, in_w, force_add = False):
        self.M += in_r.M

        self.weight += in_w
        selected = ti.random() * self.weight <= in_w or force_add
        if selected:
            self.z = in_r.z
        return selected

    @ti.func
    def finalize(self, computed_integrand):
        p_hat = vec3(0.33, 0.33, 0.33).dot(computed_integrand)
        if p_hat < 1e-6:
            self.weight = 0.0
        else:
            self.weight = self.weight / p_hat
# NOTES
# when doing reconnection shift:
#   account for ratio of dstPdf / srcPdf for both x_1 and x_rc
#   when computing shifted integrand, it is brdf/pdf of first bounce times brdf/pdf of rc bounce times rc_L