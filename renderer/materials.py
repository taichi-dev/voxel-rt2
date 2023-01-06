import taichi as ti
from renderer.math_utils import (eps, inf, sqr)
from renderer.bsdf import DisneyBSDF

import numpy as np
import csv

# All of our hardcoded materials here



######   Basic   ######
# 0 = Air
# 1 = Simple rough surface
# 2 = Emissive
#
######   Concrete   ######
# 10 = Rough concrete
# 11 = Smooth concrete
#
######   Stone   ######
# 20 = Silicate (Like jade)
# 21 = Smooth ceramic
# 22 = Rough ceramic
#
######   Wood   ######
# 30 = Bark
# 31 = Wood plank
# 32 = Gloss coated wood plank
#
######   Plastic   ######
# 40 = Smooth plastic
# 41 = Rough plastic
#
######   Metals   ######
# 50 = Rough metal
# 51 = Smooth metal
# 52 = Mirror
# 53 = Brushed metal (anisotropic)
# 54 = Car paint (partially metallic)
#
######   Misc   ######
# 80 = Plant
# 81 = Light skin
# 82 = Cloth

@ti.data_oriented
class MaterialList:
    @ti.kernel
    def init_all_to_default(self):
        for i in self.mat_list:
            self.mat_list[i] = self.bsdf.disney_material(base_col=ti.math.vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.0 \
                                        ,metallic=0.0 \
                                        ,specular=0.04 \
                                        ,specular_tint=0.0 \
                                        ,roughness=0.9 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0 \
                                        ,ior_minus_one=0.0)
    
    @ti.kernel
    def load_from_csv(self, data : ti.types.ndarray(element_dim=1)):
        for i in data:
            mat_values = data[i]
            index = ti.cast(mat_values[0], ti.i32)
            base_col = ti.math.vec3([mat_values[1], mat_values[2], mat_values[3]])
            subsurface = mat_values[4]
            metallic = mat_values[5]
            specular = mat_values[6]
            specular_tint = mat_values[7]
            roughness = mat_values[8]
            anisotropic = mat_values[9]
            sheen = mat_values[10]
            sheen_tint = mat_values[11]
            clearcoat = mat_values[12]
            clearcoat_gloss = mat_values[13]
            ior_minus_one = mat_values[14]
            self.mat_list[index] = self.bsdf.disney_material( \
                                         base_col=base_col \
                                        ,subsurface=subsurface \
                                        ,metallic=metallic \
                                        ,specular=specular \
                                        ,specular_tint=specular_tint \
                                        ,roughness=roughness \
                                        ,anisotropic=anisotropic \
                                        ,sheen=sheen \
                                        ,sheen_tint=sheen_tint \
                                        ,clearcoat=clearcoat \
                                        ,clearcoat_gloss=clearcoat_gloss \
                                        ,ior_minus_one=ior_minus_one)

    def __init__(self) -> None:
        self.bsdf = DisneyBSDF()
        self.mat_list = self.bsdf.disney_material.field(shape=(128,))

        self.init_all_to_default()

        materials_array = []
        with open('default_material_set.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in reader:
                if i > 0:
                    materials_array.append([float(x) for x in row])
                i += 1
        materials_array = np.array(materials_array).astype(np.float32)

        self.load_from_csv(materials_array)
