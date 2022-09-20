import math
import taichi as ti
import numpy as np

eps = 1e-4
inf = 1e10
vec3 = ti.types.vector(3, float)

@ti.func
def saturate(x):
    return min(max(x, 0.0), 1.0)

@ti.func
def sqr(x):
    return x*x

@ti.func
def sample_cosine_weighted_hemisphere(n):
    # Shirley, et al, 2019. Sampling Transformation Zoo. Chapter 16, Ray Tracing Gems, p240
    u = ti.Vector([ti.random(), ti.random()])
    a = 1.0 - 2.0 * u[0]
    b = ti.sqrt(1.0 - a * a)
    phi = 2.0 * np.pi * u[1]
    return ti.Vector([n.x + b * ti.cos(phi), n.y + a, n.z + b * ti.sin(phi)])

@ti.func
def make_tangent_space(n):
    h = ti.math.vec3(1.0, 0.0, 0.0) if ti.abs(n.y) > 0.9 else ti.math.vec3(0.0, 1.0, 0.0)
    y = n.cross(h).normalized()
    x = n.cross(y)
    return ti.math.mat3(x, y, n).transpose()

@ti.func
def sample_cone(cos_theta_max):
    u0 = ti.random()
    u1 = ti.random()
    cos_theta = (1.0 - u0) + u0 * cos_theta_max
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
    phi = 2.0 * np.pi * u1
    x = sin_theta * ti.cos(phi)
    y = sin_theta * ti.sin(phi)
    z = cos_theta
    return ti.Vector([x, y, z])

@ti.func
def sample_cone_oriented(cos_theta_max, n):
    mat_dir = make_tangent_space(n) @ sample_cone(cos_theta_max)
    return ti.Vector([mat_dir[0], mat_dir[1], mat_dir[2]])

@ti.func
def interleave_bits_z3(v: ti.u32):
    # https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    x = (v | (v << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


@ti.func
def morton(p):
    return (
        interleave_bits_z3(p.x)
        | (interleave_bits_z3(p.y) << 1)
        | (interleave_bits_z3(p.z) << 2)
    )


@ti.func
def rgb32f_to_rgb8(c):
    c = ti.math.clamp(c, 0.0, 1.0)
    r = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])
    for i in ti.static(range(3)):
        r[i] = ti.cast(c[i] * 255, ti.u8)
    return r


@ti.func
def rgb8_to_rgb32f(c):
    r = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        r[i] = ti.cast(c[i], ti.f32) / 255.0
    return r


@ti.func
def ray_aabb_intersection(box_min, box_max, o, d):
    near_int = -inf
    far_int = inf

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    intersect = near_int <= far_int
    return intersect, near_int, far_int


def np_normalize(v):
    # https://stackoverflow.com/a/51512965/12003165
    return v / np.sqrt(np.sum(v**2))


def np_rotate_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # https://stackoverflow.com/a/6802723/12003165
    axis = np_normalize(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
            [0, 0, 0, 1],
        ]
    )

# Uchimura 2017, "HDR theory and practice"
# Math: https://www.desmos.com/calculator/gslcdxvipg
# Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
@ti.func
def uchimura(x):
    P = 1.0   # max display brightness
    a = 1.0   # contrast
    m = 0.22  # linear section start
    l = 0.4   # linear section length
    c = 1.33  # black
    b = 0.0   # pedestal

    l0 = ((P - m) * l) / a
    S0 = m + l0
    S1 = m + a * l0
    C2 = (a * P) / (P - S1)
    CP = -C2 / P

    w0 = 1.0 - ti.math.smoothstep(0.0, m, x)
    w2 = ti.math.step(m + l0, x)
    w1 = 1.0 - w0 - w2

    T = m * ti.pow(x / m, ti.math.vec3(c)) + b
    S = P - (P - S1) * ti.exp(CP * (x - S0))
    L = m + a * (x - m)

    return T * w0 + L * w1 + S * w2

