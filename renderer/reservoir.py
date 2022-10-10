import math
import taichi as ti
from taichi.math import *
import numpy as np

@ti.dataclass
class StorageSample:
    L : vec3
    rc_pos : vec3
    rc_normal : ti.types.vector(2, ti.f16)
    rc_incident_dir : vec3
    rc_incident_L : vec3
    cached_jacobian_term : ti.f16


@ti.dataclass
class Sample:
    L : vec3
    rc_pos : vec3
    rc_normal : vec3
    rc_incident_dir : vec3
    rc_incident_L : vec3
    cached_jacobian_term : ti.f32

@ti.dataclass
class Reservoir:
    z:Sample
    M:float
    weight:float

    @ti.func
    def init(self):
        self.z = Sample(L=vec3(0,0,0), rc_pos=vec3(0,0,0), rc_normal=vec3(0,0,0), rc_incident_dir=vec3(0,0,0), rc_incident_L=vec3(0,0,0), cached_jacobian_term=1.0)
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
        
