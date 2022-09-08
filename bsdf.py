import math
import taichi as ti
import numpy as np
from math_utils import (eps, inf, vec3, mix, sqr, pow5, saturate, out_dir)

# references:
#   - https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
#   - https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf

# struct for all disney bsdf parameters
disney_material_params = ti.types.struct(base_col=vec3 \
                                        ,subsurface=float \
                                        ,metallic=float \
                                        ,specular=float \
                                        ,specular_tint=float \
                                        ,roughness=float \
                                        ,anisotropic=float \
                                        ,sheen=float \
                                        ,sheen_tint=float \
                                        ,clearcoat=float \
                                        ,clearcoat_gloss=float)


@ti.func
def disneySubsurface(mat, n_dot_l, n_dot_v, l_dot_h, F_L, F_V):
    
    Fss90 = l_dot_h*l_dot_h*mat.roughness
    Fss = mix(1.0, Fss90, F_L) * mix(1.0, Fss90, F_V)
    ss = 1.25 * (Fss * (1. / (n_dot_l + n_dot_v) - .5) + .5)
    
    return (1./np.pi) * ss * mat.base_col

@ti.func
def disney_diffuse(mat, n_dot_l, n_dot_v, l_dot_h): # diffuse, subsurface AND sheen

    R_R = 2.0 * mat.roughness * sqr(l_dot_h)
    F_L = pow5(1.0 - n_dot_l)
    F_V = pow5(1.0 - n_dot_v)

    f_lambert = mat.base_col / np.pi
    f_retro   = f_lambert * R_R * (F_L + F_V + F_L*F_V*(R_R - 1.0))

    f_d = f_lambert * (1.0 - 0.5*F_L) * (1.0 - 0.5*F_V) + f_retro

    albedo_lum = mat.base_col.dot(vec3([0.2125, 0.7154, 0.0721]))
    sheen_col = mat.base_col / albedo_lum if albedo_lum > 0. else vec3([1.0, 1.0, 1.0])
    sheen_schlick = pow5(1.0 - l_dot_h)
    sheen = mat.sheen * mix(vec3([1.0, 1.0, 1.0]), sheen_col, mat.sheen_tint) * sheen_schlick

    ss = disneySubsurface(mat, n_dot_l, n_dot_v, l_dot_h, F_L, F_V)

    return mix(f_d, ss, mat.subsurface) + sheen

@ti.func
def GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay):
    return 1.0 / (np.pi * ax*ay * sqr( sqr(h_dot_x/ax) + sqr(h_dot_y/ay) + sqr(n_dot_h) ))#

@ti.func
def smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay):
    return 1.0 / (n_dot_v + ti.sqrt( sqr(v_dot_x*ax) + sqr(v_dot_y*ay) + sqr(n_dot_v) ))

@ti.func
def disney_fresnel(mat, l_dot_h):
    albedo_lum = mat.base_col.dot(vec3([0.2125, 0.7154, 0.0721]))
    spec_tint = mat.base_col / albedo_lum if albedo_lum > 0. else vec3([1.0, 1.0, 1.0])
    spec_col = mix(mat.specular *.08 * mix(vec3([1.0, 1.0, 1.0]), spec_tint, mat.specular_tint), mat.base_col, mat.metallic)
    F_L = pow5(1.0 - l_dot_h)
    return mix(spec_col, vec3([1.0, 1.0, 1.0]), F_L)


@ti.func
def disney_specular(mat, \
                    n_dot_l, n_dot_v, \
                    l_dot_h, n_dot_h, \
                    h_dot_x, h_dot_y, \
                    l_dot_x, l_dot_y, \
                    v_dot_x, v_dot_y, \
                    tang, bitang): # specular REFLECTION
    
    aspect = ti.sqrt(1.0 - 0.9*mat.anisotropic)

    ax = max(sqr(mat.roughness) / aspect, 1e-3)
    ay = max(sqr(mat.roughness) * aspect, 1e-3)

    D = GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay)
    G = smithG_GGX_aniso(n_dot_l, l_dot_x, l_dot_y, ax, ay) \
      * smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay)
    F = disney_fresnel(mat, l_dot_h)

    return D * G* F# / max(4.0 * n_dot_l * n_dot_v, 1e-5)

@ti.func
def GTR1(n_dot_h, alpha):
    
    a2 = alpha*alpha
    t = 1 + (a2-1)*n_dot_h*n_dot_h
    D = (a2-1) / (np.pi*ti.log(a2)*t)

    if (alpha >= 1): D = 1/np.pi

    return D

@ti.func
def smithG_GGX(n_dot_v, alpha):
    a2 = alpha*alpha
    b = n_dot_v*n_dot_v
    return 1.0 / (n_dot_v + ti.sqrt(a2 + b - a2*b))

@ti.func
def disney_clearcoat(mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h):
    alpha = mix(0.1, 0.001, mat.clearcoat_gloss)
    D = GTR1(abs(n_dot_h), alpha)
    F = mix(0.04, 1.0, pow5(1.0 - l_dot_h))
    G = smithG_GGX(n_dot_l, .25) * smithG_GGX(n_dot_v, .25)
    return mat.clearcoat * D * F * G


@ti.func
def disney_evaluate(mat, v, n, l, tang, bitang):
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

        bsdf += disney_diffuse(mat, n_dot_l, n_dot_v, l_dot_h) * (1.0 - mat.metallic)
        bsdf += disney_specular(mat, \
                        n_dot_l, n_dot_v, \
                        l_dot_h, n_dot_h, \
                        h_dot_x, h_dot_y, \
                        l_dot_x, l_dot_y, \
                        v_dot_x, v_dot_y, \
                        tang, bitang)
        bsdf += disney_clearcoat(mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h)
    return bsdf

@ti.func
def reflect(view, normal):
    view = -view
    return view - 2.0 * normal.dot(view) * normal

@ti.func
def sample_diffuse(mat, n):
    cosine_weighted_dir = out_dir(n)
    pdf = saturate(cosine_weighted_dir.dot(n)) / np.pi

    return cosine_weighted_dir, pdf

@ti.func
def sample_clearcoat(mat, v, n, tang, bitang):
    u = ti.Vector([ti.random(), ti.random()])
    alpha = mix(0.1, 0.001, mat.clearcoat_gloss)
    a2 = sqr(alpha)
    cosTheta = ti.sqrt(max(1e-4, (1.0 - ti.pow(a2, 1.0 - u.x))/(1.0 - a2)))
    sinTheta = ti.sqrt(max(1e-4, 1.0 - cosTheta * cosTheta))

    phi = 2.0 * np.pi * u.y

    m = vec3([sinTheta * ti.cos(phi), cosTheta, sinTheta * ti.sin(phi)])

    m = m.x * tang + m.z * bitang + m.y * n

    if(m.dot(v) < 0.0): m *= -1.0

    sampled_dir = reflect(v, m)

    # calculate pdf
    n_dot_h = abs(n.dot(m))
    v_dot_h = v.dot(m)
    D = GTR1(n_dot_h, alpha)
    pdf = D * n_dot_h / (4.0 * v_dot_h)
    return sampled_dir, pdf

@ti.func
def sample_specular(mat, v, n, tang, bitang):

    aspect = ti.sqrt(1.0 - 0.9*mat.anisotropic)

    ax = max(sqr(mat.roughness) / aspect, 1e-3)
    ay = max(sqr(mat.roughness) * aspect, 1e-3)

    u = ti.Vector([ti.random(), ti.random()])
    # ggx vndf
    V = vec3([v.x * ax, v.y, v.z * ay]).normalized()

    t1 = (V.cross(vec3([0.0, 1.0, 0.0])).normalized()) if V.y < 0.9999 else vec3([1.0, 0.0, 0.0])
    t2 = t1.cross(V)

    a = 1.0 / (1.0 + V.y)
    r = ti.sqrt(u.x)
    phi = (u.y / a * np.pi) if u.y < a else (np.pi + (u.y - a) * np.pi / (1.0 - a))
    p1 = r * ti.cos(phi)
    p2 = r * ti.sin(phi) * (1.0 if u.y < a else V.y)

    m = p1 * t1 + p2 * t2 + ti.sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * V
    m = vec3([ax * m.x, m.y, ay * m.z]).normalized()

    # convert to world space
    m = m.x * tang + m.z * bitang + m.y * n

    if(m.dot(v) < 0.0): m *= -1.0

    sampled_dir = reflect(v, m)

    # compute pdf
    n_dot_l = abs(n.dot(sampled_dir))
    n_dot_v = (n.dot(v))
    l_dot_h = abs(sampled_dir.dot(m))
    n_dot_h = (n.dot(m))
    h_dot_x = (m.dot(tang))
    h_dot_y = (m.dot(bitang))
    v_dot_x = (v.dot(tang))
    v_dot_y = (v.dot(bitang))
    D = GTR2_anisotropic(n_dot_h, h_dot_x, h_dot_y, ax, ay)
    G = smithG_GGX_aniso(n_dot_v, v_dot_x, v_dot_y, ax, ay)
    pdf = G * l_dot_h * D / n_dot_l

    return sampled_dir, pdf

@ti.func
def sample_disney(mat, v, n, tang, bitang):
    # set lobe probabilities
    diffuse_w = (1.0 - mat.metallic) * ti.math.clamp(1.4 - mat.specular, 0.4, 0.9)
    specular_w = 1.0 - diffuse_w
    clearcoat_w = mat.clearcoat*0.7

    w_sum = diffuse_w + specular_w + clearcoat_w

    diffuse_w /= w_sum
    specular_w /= w_sum
    clearcoat_w /= w_sum

    # choose a lobe
    sample_dir = vec3([1.0, 1.0, 1.0])
    brdf = vec3([0.0, 0.0, 0.0])
    pdf = 0.0
    rand = ti.random()
    chosenLobe = -1
    if(rand < diffuse_w):
        sample_dir, pdf = sample_diffuse(mat, n)
        chosenLobe = 0
    elif(rand < diffuse_w + specular_w):
        sample_dir, pdf = sample_specular(mat, v, n, tang, bitang)
        chosenLobe = 1
    else:
        sample_dir, pdf = sample_clearcoat(mat, v, n, tang, bitang)
        chosenLobe = 2
    
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

    if(chosenLobe == 0):
        brdf += disney_diffuse(mat, n_dot_l, n_dot_v, l_dot_h) * (1.0 - mat.metallic)
        pdf *= diffuse_w
    elif(chosenLobe == 1):
        brdf += disney_specular(mat, \
                               n_dot_l, n_dot_v, \
                               l_dot_h, n_dot_h, \
                               h_dot_x, h_dot_y, \
                               l_dot_x, l_dot_y, \
                               v_dot_x, v_dot_y, \
                               tang, bitang)
        pdf *= specular_w
    else:
        brdf += disney_clearcoat(mat, n_dot_l, n_dot_v, n_dot_h, l_dot_h)
        pdf *= clearcoat_w

    return sample_dir, brdf, pdf