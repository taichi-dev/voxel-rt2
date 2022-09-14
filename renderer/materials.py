import taichi as ti
from renderer.math_utils import (eps, inf, vec3, sqr)
from renderer.bsdf import DisneyBSDF

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
# 31 = Gloss coated wood plank
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
# 54 = Gloss coated, rough metal
# 55 = Car paint (partially metallic)
#
######   Misc   ######
# 80 = Plant
# 81 = Light skin
# 82 = Dark skin
# 83 = Cloth

@ti.data_oriented
class MaterialList:
    def __init__(self) -> None:
        self.bsdf = DisneyBSDF()
        self.mat_list = self.bsdf.disney_material.field(shape=(128,))

        self.simple_rough_surface = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.0 \
                                        ,metallic=0.0 \
                                        ,specular=0.0 \
                                        ,specular_tint=0.0 \
                                        ,roughness=1.0 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0)

        self.mat_list[1] = self.simple_rough_surface

        self.mat_list[2] = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.0 \
                                        ,metallic=0.0 \
                                        ,specular=0.0 \
                                        ,specular_tint=0.0 \
                                        ,roughness=1.0 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0)

        self.mat_list[10] = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.0 \
                                        ,metallic=0.0 \
                                        ,specular=0.3 \
                                        ,specular_tint=0.0 \
                                        ,roughness=0.6 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0)

        self.mat_list[11] = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.0 \
                                        ,metallic=0.0 \
                                        ,specular=0.3 \
                                        ,specular_tint=0.0 \
                                        ,roughness=0.2 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0)

        self.mat_list[20] = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.9 \
                                        ,metallic=0.0 \
                                        ,specular=0.5 \
                                        ,specular_tint=0.2 \
                                        ,roughness=0.04 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0)

        self.mat_list[21] = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.5 \
                                        ,metallic=0.0 \
                                        ,specular=0.6 \
                                        ,specular_tint=0.0 \
                                        ,roughness=0.6 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=1.0 \
                                        ,clearcoat_gloss=0.99)

        self.mat_list[22] = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.5 \
                                        ,metallic=0.0 \
                                        ,specular=0.6 \
                                        ,specular_tint=0.0 \
                                        ,roughness=0.6 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0)

        self.mat_list[30] = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.3 \
                                        ,metallic=0.0 \
                                        ,specular=0.2 \
                                        ,specular_tint=0.0 \
                                        ,roughness=0.6 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.4 \
                                        ,sheen_tint=0.5 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0)

        self.mat_list[31] = self.bsdf.disney_material(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.3 \
                                        ,metallic=0.0 \
                                        ,specular=0.5 \
                                        ,specular_tint=0.0 \
                                        ,roughness=0.5 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.4 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.0 \
                                        ,clearcoat_gloss=0.0)