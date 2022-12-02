import math
import taichi as ti
import numpy as np

eps = 1e-6
inf = np.inf
uvec2 = ti.types.vector(2, ti.u32)

@ti.func
def saturate(x):
    return min(max(x, 0.0), 1.0)

@ti.func
def sqr(x):
    return x*x

@ti.func
def is_vec_zero(x):
    return x.dot(x) < 1e-7

@ti.func
def sample_cosine_weighted_hemisphere(n):
    # Shirley, et al, 2019. Sampling Transformation Zoo. Chapter 16, Ray Tracing Gems, p240
    u = ti.Vector([ti.random(), ti.random()])
    a = 1.0 - 2.0 * u[0]
    b = ti.sqrt(1.0 - a * a)
    a *= 1.0 - 1e-5
    b *= 1.0 - 1e-5 # Grazing angle precision fix
    phi = 2.0 * np.pi * u[1]
    return ti.Vector([n.x + b * ti.cos(phi), n.y + b * ti.sin(phi), n.z + a]).normalized()

@ti.func
def make_orthonormal_basis(n):
    h = ti.math.vec3(1.0, 0.0, 0.0) if ti.abs(n.y) > 0.9 else ti.math.vec3(0.0, 1.0, 0.0)
    y = n.cross(h).normalized()
    x = n.cross(y)
    return x, y

@ti.func
def make_tangent_space(n):
    x, y = make_orthonormal_basis(n)
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
def cone_sample_pdf(cos_theta_max, cos_theta):
    return 1.0/(2.0*np.pi*(1.0 - cos_theta_max)) if cos_theta >= cos_theta_max else 0.0
    # This is actually wrong, since it assumes the light direction is alligned along the Y axis
    

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

@ti.func
def luminance(x):
    return ti.Vector([0.2125, 0.7154, 0.0721]).dot(x)

@ti.func
def smoothstep(edge0, edge1, x):
    t = ti.math.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

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

@ti.func
def Pack2x8(x):
    floored = ti.floor(255.0 * x + 0.5)
    return ti.cast(floored.dot(ti.Vector([1.0 / 65535.0, 256.0 / 65535.0])), ti.f16)

@ti.func
def Unpack2x8(pack):
    packed = ti.cast(pack, ti.f32) 
    packed *= 65535.0 / 256.0
    y_comp = ti.floor(packed)
    x_comp = packed - y_comp
    return ti.Vector([256.0 / 255.0, 1.0 / 255.0]) * ti.Vector([x_comp, y_comp])

# octahedral encoding
@ti.func
def encode_unit_vector_3x16(vec):
    vec.xy /= abs(vec.x) + abs(vec.y) + abs(vec.z)

    encoded = ((1.0 - abs(vec.yx)) * ti.Vector([1.0 if vec.x >= 0.0 else -1.0, 1.0 if vec.y >= 0.0 else -1.0]) if vec.z <= 0.0 else vec.xy) * 0.5 + 0.5
    return ti.cast(ti.Vector([encoded.x, encoded.y]), ti.f16)

@ti.func
def decode_unit_vector_3x16(a):
    encoded = ti.cast(a, ti.f32) * 2.0 - 1.0
    vec = ti.Vector([encoded.x, encoded.y, 1.0 - abs(encoded.x) - abs(encoded.y)])
    t = max(-vec.z, 0.0)
    vec.xy += ti.Vector([-t if vec.x >= 0.0 else t, -t if vec.y >= 0.0 else t])
    return vec.normalized()

@ti.func
def hash3(x : ti.u32, y : ti.u32, z : ti.u32):
	x += x >> 11
	x ^= x << 7
	x += y
	x ^= x << 3
	x += z ^ (x >> 14)
	x ^= x << 6
	x += x >> 15
	x ^= x << 5
	x += x >> 12
	x ^= x << 9
	return x

@ti.func
def encode_material(mat_id, albedo):
    shift = ti.cast(ti.Vector([0, 8, 16, 24]), ti.u32)
    data = ti.cast(ti.Vector([mat_id, albedo.r*255, albedo.g*255, albedo.b*255]), ti.u32)
    shifted = data << shift
    return shifted[0] | shifted[1] | shifted[2] | shifted[3]

@ti.func
def decode_material(mat_list, enc : ti.u32):
    shift = ti.cast(ti.Vector([0, 8, 16, 24]), ti.u32)
    unshifted = enc >> shift
    unshifted = unshifted & 255
    albedo = unshifted.yzw / 255.0
    
    disney_mat = mat_list[ti.cast(unshifted[0], ti.i32)]
    disney_mat.base_col = albedo
    return disney_mat, unshifted[0]

# Encode 4 floats in (0,1) in a uint32 with arbitrary precision for each
@ti.func
def encode_u32_arb(data : ti.types.vector(4, ti.f32), size : ti.types.vector(4, ti.i32)):
    mult = pow(2.0, ti.cast(size, ti.f32)) - 1.0
    shift = ti.cast(ti.Vector([0, size.x, size.x + size.y, size.x + size.y + size.z]), ti.u32)
    shifted = ti.cast(data*mult + 0.5, ti.u32) << shift
    return shifted[0] | shifted[1] | shifted[2] | shifted[3]

@ti.func
def decode_u32_arb(enc : ti.u32, size : ti.types.vector(4, ti.i32)):
    max_value = ti.cast(pow(2, size) - 1, ti.u32)
    shift = ti.cast(ti.Vector([0, size.x, size.x + size.y, size.x + size.y + size.z]), ti.u32)
    unshifted = enc >> shift
    unshifted = unshifted & max_value
    return ti.cast(unshifted, ti.f32) / ti.cast(max_value, ti.f32)