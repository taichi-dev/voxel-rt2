from cmath import isinf
import math
import taichi as ti
from taichi.math import *
import numpy as np
from renderer.math_utils import (eps, inf, sqr, saturate, sample_cosine_weighted_hemisphere)
from taichi.math import (mix, reflect, refract)


# references:
#   - https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
#   - https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
#   - https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf

@ti.data_oriented
class DisneyBSDF:
    def __init__(self) -> None:
        # struct for all disney bsdf parameters
        self.disney_material = ti.types.struct(base_col=vec3 \
                                                ,subsurface=float \
                                                ,metallic=float \
                                                ,specular=float \
                                                ,specular_tint=float \
                                                ,roughness=float \
                                                ,anisotropic=float \
                                                ,sheen=float \
                                                ,sheen_tint=float \
                                                ,clearcoat=float \
                                                ,clearcoat_gloss=float \
                                                ,ior_minus_one=float)

    @ti.func
    def disneySubsurface(self, mat, n_dot_l, n_dot_v, l_dot_h, F_L, F_V):
        
        Fss90 = l_dot_h*l_dot_h*mat.roughness
        Fss = mix(1.0, Fss90, F_L) * mix(1.0, Fss90, F_V)
        ss = 1.25 * (Fss * (1. / (n_dot_l + n_dot_v) - .5) + .5)
        
        return (1./np.pi) * ss * mat.base_col

    @ti.func
    def disney_diffuse(self, mat, n_dot_l, n_dot_v, l_dot_h): # diffuse, subsurface AND sheen

        R_R = 2.0 * mat.roughness * sqr(l_dot_h)
        F_L = pow(1.0 - n_dot_l, 5.0)
        F_V = pow(1.0 - n_dot_v, 5.0)

        f_lambert = mat.base_col / np.pi
        f_retro   = f_lambert * R_R * (F_L + F_V + F_L*F_V*(R_R - 1.0))

        f_d = f_lambert * (1.0 - 0.5*F_L) * (1.0 - 0.5*F_V) + f_retro

        albedo_lum = mat.base_col.dot(vec3([0.2125, 0.7154, 0.0721]))
        sheen_col = mat.base_col / albedo_lum if albedo_lum > 0. else vec3([1.0, 1.0, 1.0])
        sheen_schlick = pow(1.0 - l_dot_h, 5.0)
        sheen = mat.sheen * mix(vec3([1.0, 1.0, 1.0]), sheen_col, mat.sheen_tint) * sheen_schlick

        ss = self.disneySubsurface(mat, n_dot_l, n_dot_v, l_dot_h, F_L, F_V)

        return mix(f_d, ss, mat.subsurface) + sheen

    @ti.func
    def GTR2_anisotropic(self, n_dot_h, h_dot_x, h_dot_y, ax, ay):
        return 1.0 / (np.pi * ax*ay * sqr( sqr(h_dot_x/ax) + sqr(h_dot_y/ay) + sqr(n_dot_h) ))#

    @ti.func
    def smithG_GGX_aniso(self, n_dot_v, v_dot_x, v_dot_y, ax, ay):
        return 1.0 / (n_dot_v + ti.sqrt( sqr(v_dot_x*ax) + sqr(v_dot_y*ay) + sqr(n_dot_v) ))

    @ti.func
    def disney_fresnel(self, mat, l_dot_h):
        albedo_lum = mat.base_col.dot(vec3([0.2125, 0.7154, 0.0721]))
        spec_tint = mat.base_col / albedo_lum if albedo_lum > 0. else vec3([1.0, 1.0, 1.0])
        spec_col = mix(mat.specular *.08 * mix(vec3([1.0, 1.0, 1.0]), spec_tint, mat.specular_tint), mat.base_col, mat.metallic)
        F_L = pow(1.0 - l_dot_h, 5.0)
        return mix(spec_col, vec3([1.0, 1.0, 1.0]), F_L)


    @ti.func
    def disney_specular(self, mat, \
                        n_dot_l, n_dot_v, \
                        l_dot_h, n_dot_h, \
                        h_dot_x, h_dot_y, \
                        l_dot_x, l_dot_y, \
                        v_dot_x, v_dot_y, \
                        tang, bitang): # specular REFLECTION
        
        aspect = ti.sqrt(1.0 - 0.9*mat.anisotropic)

        ax = max(sqr(mat.roughness) / aspect, 1e-3)
        ay = max(sqr(mat.roughness) * aspect, 1e-3)

        D = self.GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay)
        G = self.smithG_GGX_aniso(n_dot_l, l_dot_x, l_dot_y, ax, ay) \
        * self.smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay)
        F = self.disney_fresnel(mat, l_dot_h)

        return D * G* F# / max(4.0 * n_dot_l * n_dot_v, 1e-5)

    @ti.func
    def sclick_fresnel(v_dot_h, n1, n2):
        F_0 = sqr((n1-n2)/(n1+n2))
        return F_0 + (1 - F_0) * pow(1.0 - v_dot_h, 5.0)

    @ti.func
    def GTR1(self, n_dot_h, alpha):
        
        a2 = alpha*alpha
        t = 1 + (a2-1)*n_dot_h*n_dot_h
        D = (a2-1) / (np.pi*ti.log(a2)*t)

        if (alpha >= 1): D = 1/np.pi

        return D

    @ti.func
    def smithG_GGX(self, n_dot_v, alpha):
        a2 = alpha*alpha
        b = n_dot_v*n_dot_v
        return 1.0 / (n_dot_v + ti.sqrt(a2 + b - a2*b))

    @ti.func
    def disney_clearcoat(self, mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h):
        alpha = mix(0.1, 0.001, mat.clearcoat_gloss)
        D = self.GTR1(abs(n_dot_h), alpha)
        F = mix(0.04, 1.0, pow(1.0 - l_dot_h, 5.0))
        G = self.smithG_GGX(n_dot_l, .25) * self.smithG_GGX(n_dot_v, .25)
        return mat.clearcoat * D * F * G


    @ti.func
    def disney_evaluate(self, mat, v, n, l, tang, bitang):
        n_dot_l = n.dot(l)
        n_dot_v = (n.dot(v))

        bsdf = vec3([0.0, 0.0, 0.0])

        if n_dot_l > 0 and n_dot_v > 0:

            h = l + v
            h = h.normalized()

            l_dot_h = (l.dot(h))
            n_dot_h = (n.dot(h))

            h_dot_x = (h.dot(tang))
            h_dot_y = (h.dot(bitang))

            l_dot_x = (l.dot(tang))
            l_dot_y = (l.dot(bitang))

            v_dot_x = (v.dot(tang))
            v_dot_y = (v.dot(bitang))

            bsdf += self.disney_diffuse(mat, n_dot_l, n_dot_v, l_dot_h) * (1.0 - mat.metallic)
            bsdf += self.disney_specular(mat, \
                            n_dot_l, n_dot_v, \
                            l_dot_h, n_dot_h, \
                            h_dot_x, h_dot_y, \
                            l_dot_x, l_dot_y, \
                            v_dot_x, v_dot_y, \
                            tang, bitang)
            bsdf += self.disney_clearcoat(mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h)
        return bsdf

    @ti.func
    def pdf_diffuse(self, mat, n, l):
        pdf = saturate(l.dot(n)) / np.pi
        return pdf

    @ti.func
    def sample_diffuse(self, mat, n):
        cosine_weighted_dir = sample_cosine_weighted_hemisphere(n)
        pdf = saturate(cosine_weighted_dir.dot(n)) / np.pi

        return cosine_weighted_dir, pdf
    
    @ti.func
    def pdf_clearcoat(self, mat, v, n, l, tang, bitang):
        alpha = mix(0.1, 0.001, mat.clearcoat_gloss)
        h = (v + l).normalized()
        n_dot_h = abs(n.dot(h))
        v_dot_h = v.dot(h)
        D = self.GTR1(n_dot_h, alpha)
        pdf = D * n_dot_h / (4.0 * v_dot_h)
        return pdf

    @ti.func
    def sample_clearcoat(self, mat, v, n, tang, bitang):
        u = ti.Vector([ti.random(), ti.random()])
        alpha = mix(0.1, 0.001, mat.clearcoat_gloss)
        a2 = sqr(alpha)
        cosTheta = ti.sqrt(max(1e-4, (1.0 - pow(a2, 1.0 - u.x))/(1.0 - a2)))
        sinTheta = ti.sqrt(max(1e-4, 1.0 - cosTheta * cosTheta))

        phi = 2.0 * np.pi * u.y

        m = vec3([sinTheta * ti.cos(phi), cosTheta, sinTheta * ti.sin(phi)])

        m = m.x * tang + m.z * bitang + m.y * n

        if(m.dot(v) < 0.0): m *= -1.0

        sampled_dir = reflect(-v, m)

        # calculate pdf
        n_dot_h = abs(n.dot(m))
        v_dot_h = v.dot(m)
        D = self.GTR1(n_dot_h, alpha)
        pdf = D * n_dot_h / (4.0 * v_dot_h)
        return sampled_dir, pdf

    @ti.func
    def GGX_VNDF_aniso(self, mat, v, n, tang, bitang, ax, ay):
        # multiply view vector by inverse TBN matrix
        v_t = ti.math.mat3(tang, n, bitang) @ v

        u = ti.Vector([ti.random(), ti.random()])
        # ggx vndf
        V = vec3([v_t.x * ax, v_t.y, v_t.z * ay]).normalized()

        t1 = (V.cross(vec3([0.0, 1.0, 0.0])).normalized()) if V.y < 0.9999 else vec3([1.0, 0.0, 0.0])
        t2 = t1.cross(V)

        a = 1.0 / (1.0 + V.y)
        r = ti.sqrt(u.x)
        phi = (u.y / a) * np.pi if u.y < a else np.pi + (u.y - a) / (1.0 - a) * np.pi
        p1 = r * ti.cos(phi)
        p2 = r * ti.sin(phi) * (1.0 if u.y < a else V.y)

        m = p1 * t1 + p2 * t2 + ti.sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * V
        m = vec3([ax * m.x, m.y, ay * m.z]).normalized()

        # convert to world space
        m = m.x * tang + m.z * bitang + m.y * n

        if(m.dot(v) < 0.0): m *= -1.0

        return m

    @ti.func
    def pdf_specular(self, mat, v, n, l, tang, bitang):

        aspect = ti.sqrt(1.0 - 0.9*mat.anisotropic)

        ax = max(sqr(mat.roughness) / aspect, 1e-3)
        ay = max(sqr(mat.roughness) * aspect, 1e-3)

        h = (v + l).normalized()

        # compute pdf
        n_dot_l = abs(n.dot(l))
        n_dot_v = (n.dot(v))
        l_dot_h = abs(l.dot(h))
        n_dot_h = (n.dot(h))
        h_dot_x = (h.dot(tang))
        h_dot_y = (h.dot(bitang))
        v_dot_x = (v.dot(tang))
        v_dot_y = (v.dot(bitang))
        D = self.GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay)
        G = self.smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay)
        pdf = G * l_dot_h * D / n_dot_l

        return pdf

    @ti.func
    def sample_specular(self, mat, v, n, tang, bitang):

        aspect = ti.sqrt(1.0 - 0.9*mat.anisotropic)

        ax = max(sqr(mat.roughness) / aspect, 1e-3)
        ay = max(sqr(mat.roughness) * aspect, 1e-3)

        m = self.GGX_VNDF_aniso(mat, v, n, tang, bitang, ax, ay)

        sampled_dir = reflect(-v, m)

        # compute pdf
        n_dot_l = abs(n.dot(sampled_dir))
        n_dot_v = (n.dot(v))
        l_dot_h = abs(sampled_dir.dot(m))
        n_dot_h = (n.dot(m))
        h_dot_x = (m.dot(tang))
        h_dot_y = (m.dot(bitang))
        v_dot_x = (v.dot(tang))
        v_dot_y = (v.dot(bitang))
        D = self.GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay)
        G = self.smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay)
        pdf = G * l_dot_h * D / n_dot_l

        return sampled_dir, pdf

    @ti.func
    def disney_evaluate_lobewise(self, mat, v, n, l, tang, bitang, lobe_id, specular_mult = 1.0):
        n_dot_l = n.dot(l)
        n_dot_v = (n.dot(v))

        bsdf = vec3([0.0, 0.0, 0.0])

        if n_dot_l > 0 and n_dot_v > 0:

            h = l + v
            h = h.normalized()

            l_dot_h = (l.dot(h))
            n_dot_h = (n.dot(h))

            h_dot_x = (h.dot(tang))
            h_dot_y = (h.dot(bitang))

            l_dot_x = (l.dot(tang))
            l_dot_y = (l.dot(bitang))

            v_dot_x = (v.dot(tang))
            v_dot_y = (v.dot(bitang))

            # 9 means all lobes
            if lobe_id == 0 or lobe_id == 9:
                bsdf += self.disney_diffuse(mat, n_dot_l, n_dot_v, l_dot_h) * (1.0 - mat.metallic)
            if lobe_id == 1 or lobe_id == 9:
                bsdf += self.disney_specular(mat, \
                                n_dot_l, n_dot_v, \
                                l_dot_h, n_dot_h, \
                                h_dot_x, h_dot_y, \
                                l_dot_x, l_dot_y, \
                                v_dot_x, v_dot_y, \
                                tang, bitang)*specular_mult
            if lobe_id == 2 or lobe_id == 9:
                bsdf += self.disney_clearcoat(mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h)*specular_mult
        return bsdf

    @ti.func
    def disney_get_lobe_probabilities(self, mat):
        diffuse_w = (1.0 - mat.metallic) * clamp(1.0 - mat.specular, 0.4, 0.9)
        specular_w = 1.0 - diffuse_w
        clearcoat_w = mat.clearcoat*0.7

        w_sum = diffuse_w + specular_w + clearcoat_w

        diffuse_w /= w_sum
        specular_w /= w_sum
        clearcoat_w /= w_sum

        return diffuse_w, specular_w, clearcoat_w

    @ti.func
    def pdf_disney_lobewise(self, mat, v, n, l, tang, bitang, lobe_id):

        diffuse_w, specular_w, clearcoat_w = self.disney_get_lobe_probabilities(mat)

        pdf = 1.0
        if(lobe_id == 0):
            pdf *= self.pdf_diffuse(mat, n, l) * diffuse_w
        elif(lobe_id == 1):
            pdf *= self.pdf_specular(mat, v, n, l, tang, bitang) * specular_w
        else:
            pdf *= self.pdf_clearcoat(mat, v, n, l, tang, bitang) * clearcoat_w

        if isinf(pdf) or isnan(pdf):
            pdf = 1.0
            
        return pdf

    @ti.func
    def pdf_disney(self, mat, v, n, l, tang, bitang):

        diffuse_w, specular_w, clearcoat_w = self.disney_get_lobe_probabilities(mat)

        pdf = 0.0
        pdf += self.pdf_diffuse(mat, n, l) * diffuse_w
        pdf += self.pdf_specular(mat, v, n, l, tang, bitang) * specular_w
        pdf += self.pdf_clearcoat(mat, v, n, l, tang, bitang) * clearcoat_w

        return pdf

    @ti.func
    def sample_disney(self, mat, v, n, tang, bitang):

        # set lobe probabilities
        diffuse_w, specular_w, clearcoat_w = self.disney_get_lobe_probabilities(mat)

        # choose a lobe
        sample_dir = vec3([1.0, 1.0, 1.0])
        brdf = vec3([0.0, 0.0, 0.0])
        pdf = 1.0
        rand = ti.random()
        chosen_lobe = -1
        if(rand < diffuse_w):
            sample_dir, pdf = self.sample_diffuse(mat, n)
            chosen_lobe = 0
        elif(rand < diffuse_w + specular_w):
            sample_dir, pdf = self.sample_specular(mat, v, n, tang, bitang)
            chosen_lobe = 1
        else:
            sample_dir, pdf = self.sample_clearcoat(mat, v, n, tang, bitang)
            chosen_lobe = 2

        
        # compute brdf inputs
        n_dot_l = n.dot(sample_dir)
        n_dot_v = (n.dot(v))

        h = sample_dir + v
        h = h.normalized()

        l_dot_h = (sample_dir.dot(h))
        n_dot_h = (n.dot(h))

        h_dot_x = (h.dot(tang))
        h_dot_y = (h.dot(bitang))

        l_dot_x = (sample_dir.dot(tang))
        l_dot_y = (sample_dir.dot(bitang))

        v_dot_x = (v.dot(tang))
        v_dot_y = (v.dot(bitang))

        v_dot_h = (v.dot(h))

        if(chosen_lobe == 0):
            brdf += self.disney_diffuse(mat, n_dot_l, n_dot_v, l_dot_h) * (1.0 - mat.metallic)
            pdf *= diffuse_w
        elif(chosen_lobe == 1):
            brdf += self.disney_specular(mat, \
                                n_dot_l, n_dot_v, \
                                l_dot_h, n_dot_h, \
                                h_dot_x, h_dot_y, \
                                l_dot_x, l_dot_y, \
                                v_dot_x, v_dot_y, \
                                tang, bitang)
            pdf *= specular_w
        else:
            brdf += self.disney_clearcoat(mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h)
            pdf *= clearcoat_w

        if isinf(pdf) or isnan(pdf):
            pdf = 1.0

        return sample_dir, brdf, pdf, chosen_lobe

    @ti.func
    def translucent_specular(self, mat, \
                        n_dot_l, n_dot_v, \
                        l_dot_h, n_dot_h, \
                        h_dot_x, h_dot_y, \
                        l_dot_x, l_dot_y, \
                        v_dot_x, v_dot_y, v_dot_h, \
                        tang, bitang, n1, n2): # specular REFLECTION
        
        aspect = ti.sqrt(1.0 - 0.9*mat.anisotropic)

        ax = max(sqr(mat.roughness) / aspect, 1e-3)
        ay = max(sqr(mat.roughness) * aspect, 1e-3)

        D = self.GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay)
        G = self.smithG_GGX_aniso(n_dot_l, l_dot_x, l_dot_y, ax, ay) \
        * self.smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay)
        F = self.sclick_fresnel(v_dot_h, n1, n2)

        return D * G* F# / max(4.0 * n_dot_l * n_dot_v, 1e-5)

    @ti.func
    def translucent_transmission(self, mat, \
                         n_dot_l, n_dot_v, \
                         l_dot_h, n_dot_h, \
                         h_dot_x, h_dot_y, \
                         l_dot_x, l_dot_y, \
                         v_dot_x, v_dot_y, v_dot_h, \
                         tang, bitang, n1, n2):
        aspect = ti.sqrt(1.0 - 0.9*mat.anisotropic)

        ax = max(sqr(mat.roughness) / aspect, 1e-3)
        ay = max(sqr(mat.roughness) * aspect, 1e-3)

        D = self.GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay)
        G = self.smithG_GGX_aniso(n_dot_l, l_dot_x, l_dot_y, ax, ay) \
            * self.smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay)
        F = self.sclick_fresnel(v_dot_h, n1, n2)

        eta = n1/n2

        a = (abs(l_dot_h) * abs(v_dot_h)) / (abs(n_dot_l) * abs(n_dot_v))
        b = (1.0 / sqr(l_dot_h + eta * v_dot_h))
        return mat.base_col * a * b * (1.0 - F) * G * D

    @ti.func
    def evaluate_translucent_bsdf(self, mat, v, n, l, tang, bitang, n1):
        n2 = 1.0 + mat.ior_minus_one
        eta = n1 / n2

        n_dot_l = n.dot(l)
        n_dot_v = (n.dot(v))

        bsdf = vec3([0.0, 0.0, 0.0])

        in_upper_hemisphere = n_dot_l > 0.0 and n_dot_v > 0.0

        if n_dot_l > 0 and n_dot_v > 0:

            h = l + v
            h = h.normalized()

            l_dot_h = (l.dot(h))
            n_dot_h = (n.dot(h))

            h_dot_x = (h.dot(tang))
            h_dot_y = (h.dot(bitang))

            l_dot_x = (l.dot(tang))
            l_dot_y = (l.dot(bitang))

            v_dot_x = (v.dot(tang))
            v_dot_y = (v.dot(bitang))

            v_dot_h = v.dot(h)

            if(in_upper_hemisphere):
                bsdf += self.translucent_specular(mat, \
                                n_dot_l, n_dot_v, \
                                l_dot_h, n_dot_h, \
                                h_dot_x, h_dot_y, \
                                l_dot_x, l_dot_y, \
                                v_dot_x, v_dot_y, v_dot_h, \
                                tang, bitang, n1, n2)
                bsdf += self.disney_clearcoat(mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h)
            else:
                bsdf += self.translucent_transmission(mat, \
                         n_dot_l, n_dot_v, \
                         l_dot_h, n_dot_h, \
                         h_dot_x, h_dot_y, \
                         l_dot_x, l_dot_y, \
                         v_dot_x, v_dot_y, v_dot_h, \
                         tang, bitang, n1, n2)
        return bsdf

    @ti.func
    def sample_translucent(self, mat, v, n, tang, bitang, n1):
        # set lobe probabilities
        translucent_w = 1.0
        clearcoat_w = mat.clearcoat*0.5
        
        n2 = 1.0 + mat.ior_minus_one
        eta = n1 / n2

        w_sum = clearcoat_w + translucent_w

        clearcoat_w /= w_sum
        translucent_w /= w_sum

        # choose a lobe
        sample_dir = vec3([1.0, 1.0, 1.0])
        bsdf = vec3([0.0, 0.0, 0.0])
        pdf = 1.0
        rand = ti.random()
        chosen_lobe = -1
        if(rand < clearcoat_w):
            sample_dir, pdf = self.sample_clearcoat(mat, v, n, tang, bitang)
            chosen_lobe = 2
        else:
            aspect = ti.sqrt(1.0 - 0.9*mat.anisotropic)

            ax = max(sqr(mat.roughness) / aspect, 1e-3)
            ay = max(sqr(mat.roughness) * aspect, 1e-3)

            m = self.GGX_VNDF_aniso(mat, v, n, tang, bitang, ax, ay)

            F = self.sclick_fresnel(v.dot(m), n1, n2)

            # compute pdf
            
            n_dot_v = (n.dot(v))
            n_dot_h = (n.dot(m))
            h_dot_x = (m.dot(tang))
            h_dot_y = (m.dot(bitang))
            v_dot_x = (v.dot(tang))
            v_dot_y = (v.dot(bitang))
            D = self.GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay)
            G = self.smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay)
            
            if(ti.random() < F):
                sampled_dir = reflect(-v, m)

                n_dot_l = abs(n.dot(sampled_dir))
                l_dot_h = abs(sampled_dir.dot(m))

                pdf = F * G * l_dot_h * D / n_dot_l
                chosen_lobe = 3
            else:
                sampled_dir = refract(-v, m, eta)

                n_dot_l = abs(n.dot(sampled_dir))
                l_dot_h = abs(sampled_dir.dot(m))

                pdf = (1.0 - F) * G * l_dot_h * D / n_dot_l
                chosen_lobe = 4
        
        # compute brdf inputs
        n_dot_l = n.dot(sample_dir)
        n_dot_v = (n.dot(v))

        h = sample_dir + v
        h = h.normalized()

        l_dot_h = (sample_dir.dot(h))
        n_dot_h = (n.dot(h))

        h_dot_x = (h.dot(tang))
        h_dot_y = (h.dot(bitang))

        l_dot_x = (sample_dir.dot(tang))
        l_dot_y = (sample_dir.dot(bitang))

        v_dot_x = (v.dot(tang))
        v_dot_y = (v.dot(bitang))

        v_dot_h = (v.dot(h))

        if(chosen_lobe == 2):
            bsdf += self.disney_clearcoat(mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h)
            pdf *= clearcoat_w
        elif(chosen_lobe == 3):
            brdf += self.translucent_specular(mat, \
                                n_dot_l, n_dot_v, \
                                l_dot_h, n_dot_h, \
                                h_dot_x, h_dot_y, \
                                l_dot_x, l_dot_y, \
                                v_dot_x, v_dot_y, v_dot_h, \
                                tang, bitang, n1, n2)
            pdf *= translucent_w
        else:
            brdf += self.translucent_transmission(mat, \
                         n_dot_l, n_dot_v, \
                         l_dot_h, n_dot_h, \
                         h_dot_x, h_dot_y, \
                         l_dot_x, l_dot_y, \
                         v_dot_x, v_dot_y, v_dot_h, \
                         tang, bitang, n1, n2)
            pdf *= translucent_w

        return sample_dir, bsdf, pdf
