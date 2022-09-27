import math
import taichi as ti
from taichi.math import *
import numpy as np

@ti.dataclass
class StorageSample:
    L : vec3
    gbuff_coord : ti.u32
    x2 : vec3
    n2 : ti.types.vector(2, ti.f16)
    x3 : vec3
    x2_Li : vec3


@ti.dataclass
class Sample:
    L : vec3
    x1 : vec3
    n1 : vec3
    x2 : vec3
    n2 : vec3
    x3 : vec3
    x2_Li : vec3

@ti.dataclass
class Reservoir:
    z:Sample
    M:float
    weight:float

    @ti.func
    def init(self):
        self.z = Sample(L=vec3(0,0,0), x1=vec3(0,0,0), n1=vec3(0,0,0), x2=vec3(0,0,0), n2=vec3(0,0,0), x3=vec3(0,0,0), x2_Li=vec3(0,0,0))
        self.M = 0.0
        self.weight = 0.0

    @ti.func
    def merge(self, in_r, in_w, mis_weight):
        self.M += in_r.M

        self.weight += in_w
        if(ti.random() * self.weight <= in_w):
            self.z = in_r.z

    @ti.func
    def finalize(self, computed_integrand):
        p_hat = vec3(0.33, 0.33, 0.33).dot(computed_integrand)
        if p_hat < 1e-6:
            self.weight = 0.0
        else:
            self.weight = self.weight / p_hat
        
