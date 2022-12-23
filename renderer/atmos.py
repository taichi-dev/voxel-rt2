import math
import taichi as ti
from taichi.math import *
import numpy as np
from renderer.math_utils import *

@ti.func
def rsi(pos, dir, r):
    b     = pos.dot(dir)
    discr   = b*b - pos.dot(pos) + r*r
    discr     = ti.sqrt(discr)

    return vec2(-1.0, -1.0) if discr < 0.0 else vec2(-b, -b) + vec2(-discr, discr)


@ti.func
def rayleigh_phase(cos_theta):
    return 3.0/(16.0*np.pi)*(1.0 + cos_theta*cos_theta)

@ti.func
def mie_phase(cos_theta, g):
    # Henyey-Greenstein phase
    return (1-g*g)/(4.0*np.pi*pow(1.0 + g*g - 2*g*cos_theta, 1.5))

@ti.func
def get_unit_vec(rand):
    rand.x *= np.pi * 2.0; rand.y = rand.y * 2.0 - 1.0
    ground = vec2(sin(rand.x), cos(rand.x)) * sqrt(1.0 - rand.y * rand.y)
    return vec3(ground.x, ground.y, rand.y).normalized()


@ti.data_oriented
class Atmos:
    def __init__(self):
        # Constants
        self.air_num_density       = 2.5035422e25
        self.ozone_peak = 8e-6
        self.ozone_num_density     = self.air_num_density * 0.012588 * self.ozone_peak
        self.ozone_cross_sec      = vec3(4.51103766177301e-21, 3.2854797958699e-21, 1.96774621921165e-22) * 0.0001

        self.rayleigh_coeff = vec3(0.00000519673, 0.0000121427, 0.0000296453) # scattering coeff
        self.mie_coeff = 8.6e-6 # scattering coeff
        self.ozone_coeff    = self.ozone_cross_sec*self.ozone_num_density
        self.extinc_mat = ti.Matrix([[self.rayleigh_coeff.x, self.rayleigh_coeff.y, self.rayleigh_coeff.z], \
                                     [self.mie_coeff*1.11, self.mie_coeff*1.11, self.mie_coeff*1.11]      , \
                                     [self.ozone_coeff.x, self.ozone_coeff.y, self.ozone_coeff.z]], ti.f32).transpose()
        self.scatter_mat = ti.Matrix([[self.rayleigh_coeff.x, self.rayleigh_coeff.y, self.rayleigh_coeff.z], \
                                      [self.mie_coeff, self.mie_coeff, self.mie_coeff]      , \
                                      [0.0, 0.0, 0.0]], ti.f32).transpose()

        self.scale_height_rayl = 8500.0
        self.scale_height_mie  = 1200.0

        self.scale_heights = ti.Vector([self.scale_height_rayl, self.scale_height_mie])

        self.mie_g = 0.75

        self.planet_r = 6371e3
        self.atmos_height  = 110e3

        self.trans_LUT = ti.Vector.field(3, dtype=ti.f16, shape=(256, 128))

        self.skybox_fres = ti.Vector([1.0/5120.0, 1.0/2560.0])
        self.skybox_res = ti.Vector([5120, 2560])
        self.skybox_scattering = ti.Vector.field(3, dtype=ti.f32, shape=(self.skybox_res.x, self.skybox_res.y))
        self.skybox_transmittance = ti.Vector.field(3, dtype=ti.f32, shape=(self.skybox_res.x, self.skybox_res.y))
        #############

    @ti.func
    def sample_skybox(self, ray_dir):
        texcoord = self.project_sky((ray_dir + vec3(ti.random(), ti.random(), ti.random()) * 0.01).normalized())
        fcoord = ti.Vector([texcoord.x * self.skybox_res.x - 0.5, texcoord.y * self.skybox_res.y - 0.5])
        icoord = ti.cast(fcoord, ti.i32)
        f = fract(fcoord)

        bl = self.skybox_scattering[icoord.x, icoord.y]
        br = self.skybox_scattering[(icoord.x + 1) % self.skybox_res.x, icoord.y]
        tl = self.skybox_scattering[icoord.x, (icoord.y + 1) % self.skybox_res.y]
        tr = self.skybox_scattering[(icoord.x + 1) % self.skybox_res.x, (icoord.y + 1) % self.skybox_res.y]

        scatt = mix(mix(bl, br, f.x), mix(tl, tr, f.x), f.y)

        bl = self.skybox_transmittance[icoord.x, icoord.y]
        br = self.skybox_transmittance[(icoord.x + 1) % self.skybox_res.x, icoord.y]
        tl = self.skybox_transmittance[icoord.x, (icoord.y + 1) % self.skybox_res.y]
        tr = self.skybox_transmittance[(icoord.x + 1) % self.skybox_res.x, (icoord.y + 1) % self.skybox_res.y]

        trans = mix(mix(bl, br, f.x), mix(tl, tr, f.x), f.y)

        return scatt, trans

    @ti.func
    def sample_skybox_transmittance(self, ray_dir):
        texcoord = self.project_sky(ray_dir)
        fcoord = ti.Vector([texcoord.x * self.skybox_res.x - 0.5, texcoord.y * self.skybox_res.y - 0.5])
        icoord = ti.cast(fcoord, ti.i32)
        f = fract(fcoord)

        bl = self.skybox_transmittance[icoord.x, icoord.y]
        br = self.skybox_transmittance[icoord.x + 1, icoord.y]
        tl = self.skybox_transmittance[icoord.x, icoord.y + 1]
        tr = self.skybox_transmittance[icoord.x + 1, icoord.y + 1]

        trans = mix(mix(bl, br, f.x), mix(tl, tr, f.x), f.y)

        return trans

    @ti.kernel
    def compute_skybox(self, sun_dir : vec3, sun_col : vec3, sun_cone_cos_theta_max : ti.f32):
        cam_pos = vec3(0.0, self.planet_r + 1000.0, 0.0)

        for u, v in self.skybox_scattering:
            texcoord = ti.Vector([u, v]) * self.skybox_fres

            ray_dir = self.unproject_sky(texcoord)
            in_scatter, transmittance = self.atmospheric_scattering(cam_pos, ray_dir, sun_dir, sun_col, sun_cone_cos_theta_max, 0)
            self.skybox_scattering[u, v] = in_scatter
            self.skybox_transmittance[u, v] = transmittance

    @ti.func
    def atmospheric_scattering(self, ray_pos, ray_dir, sun_dir, sun_col, sun_cone_cos_theta_max, depth : ti.template(), steps = 128):
        #
        # r(p) = ray_pos + ray_dir * lambda
        #
        fsteps = 1.0 / ti.cast(steps, ti.f32)

        air_lambdas = rsi(ray_pos, ray_dir, self.planet_r + self.atmos_height)
        planet_lamdas = rsi(ray_pos, ray_dir, self.planet_r)
        air_lambdas.y = min(air_lambdas.y, planet_lamdas.x) if planet_lamdas.x > 0.0 else air_lambdas.y

        step_delta = (air_lambdas.y - max(air_lambdas.x, 0.0)) * fsteps
        ray_step = ray_dir*step_delta
        ray_pos = ray_pos + ray_step*(0.5) # Midpoint rule gives better results than left rule

        transmittance = vec3(1., 1., 1.)
        in_scatter_col = vec3(0., 0., 0.)
        multiple_scattering = vec3(0., 0., 0.)

        if ti.static(depth <= 1):

            for i in range(0, steps):
                h = self.get_elevation(ray_pos)
                density = self.get_density(h)

                step_od = self.extinc_mat @ (density * step_delta)
                step_transmittance = saturate(exp(-step_od))

                # improved scattering integration (weighting) by Sébastian Hillaire
                visible_scattering = transmittance * saturate((1.0 - step_transmittance)/step_od)

                
                sun_ray_transmittance = self.read_trans_lut(ray_pos.normalized().dot(sun_dir), h)

                # pick direct light sample
                DIRECT_SAMPLE_COUNT = 8
                for j in range(0, DIRECT_SAMPLE_COUNT):
                    sample_dir = sample_cone_oriented(sun_cone_cos_theta_max, sun_dir)
                    cos_theta = ray_dir.dot(sample_dir)
                    phases = ti.Vector([rayleigh_phase(cos_theta), mie_phase(cos_theta, self.mie_g)])

                    
                    in_scatter_col += self.rayleigh_coeff * sun_col * sun_ray_transmittance * visible_scattering * phases.x * density.x * step_delta / ti.cast(DIRECT_SAMPLE_COUNT, ti.f32)
                    in_scatter_col += self.mie_coeff * sun_col * sun_ray_transmittance * visible_scattering * phases.y * density.y * step_delta / ti.cast(DIRECT_SAMPLE_COUNT, ti.f32)
                
                # step_scattering = self.scatter_mat @ (density * step_delta)
                # step_albedo = step_scattering / step_od
                ms_energy = 3.3 # 0.75 * 0.84 * step_albedo / (1.0 - 0.84 * step_albedo)
                # in_scatter_col += ambient_scatter * visible_scattering * step_scattering

                # multiple scattering sample
                MS_SAMPLE_COUNT = 8
                for j in range(0, MS_SAMPLE_COUNT):
                    sample_dir = get_unit_vec(vec2((j + ti.random()) / ti.cast(MS_SAMPLE_COUNT, ti.f32), fract(j * 1.618033988749)))
                    cos_theta = ray_dir.dot(sample_dir)
                    phases = ti.Vector([1.0, mie_phase(cos_theta, self.mie_g)])

                    ambient_scatter, ambient_trans = self.atmospheric_scattering(ray_pos, sample_dir, sun_dir, sun_col, sun_cone_cos_theta_max, depth+1, 8)
                    
                    # Not using rayleigh phase here because it looks better for this bad multiple scattering
                    in_scatter_col += ms_energy * self.rayleigh_coeff * ambient_scatter * visible_scattering * density.x * step_delta / ti.cast(MS_SAMPLE_COUNT, ti.f32)
                    in_scatter_col += ms_energy * self.mie_coeff * ambient_scatter * visible_scattering * phases.y * density.y * step_delta / ti.cast(MS_SAMPLE_COUNT, ti.f32)

                

                transmittance *= step_transmittance

                ray_pos += ray_step


            if planet_lamdas.x > 0.0:
                transmittance *= 0.0

        return in_scatter_col, transmittance

    # Skybox parameterization from https://sebh.github.io/publications/egsr2020.pdf
    @ti.func
    def project_sky(self, ray_dir):

        projected_dir = ray_dir.normalized().xz

        horizon_angle  = np.pi * 0.5
        azimuth  = np.pi + atan2(projected_dir.x, -projected_dir.y)
        elevation = horizon_angle - acos(ray_dir.y)

        coord = ti.Vector([0.0, 0.0])
        coord.x = azimuth / (np.pi * 2.0)
        coord.y = 0.5 + 0.5 * sign(elevation) * ti.sqrt(2.0 / np.pi * abs(elevation))

        return coord * (1.0 - self.skybox_fres) + (0.5 * self.skybox_fres)

    @ti.func
    def unproject_sky(self, uv):
        coord = (uv - 0.5 * self.skybox_fres) / (1.0 - 1.0 * self.skybox_fres)
        coord.y = -sqr(1.0 - 2.0 * coord.y) if (coord.y < 0.5) else sqr(2.0 * coord.y - 1.0)

        azimuth  = coord.x * 2.0 * np.pi - np.pi
        elevation = coord.y * 0.5 * np.pi

        cos_elevation = cos(elevation)
        sin_elevation = sin(elevation)
        cos_azimuth  = cos(azimuth)
        sin_azimuth  = sin(azimuth)

        return vec3(cos_elevation * sin_azimuth, sin_elevation, -cos_elevation * cos_azimuth).normalized()

    @ti.func
    def read_trans_lut(self, cos_theta, h):
        sample_uv = ti.cast(ti.Vector([min((cos_theta*0.5 + 0.5) * 256, 255), min((h/self.atmos_height) * 128, 127)]), ti.i32)
        return self.trans_LUT[sample_uv.x, sample_uv.y]

    @ti.kernel
    def generate_transmittance_lut(self):
        for x, y in self.trans_LUT:
            cos_theta = (x/256.0)*2.0 - 1.0
            h = self.atmos_height*y/128.0
            
            theta = acos(cos_theta)
            sin_theta = sin(theta)

            ray_dir = vec3(sin_theta, cos_theta, 0.0)
            ray_pos = vec3(0.0, self.planet_r + h, 0.0)
            self.trans_LUT[x, y] = ti.cast(self.get_ray_transmittance(ray_pos, ray_dir), ti.f16)

    @ti.func
    def get_ray_transmittance(self, ray_pos, ray_dir):
        # assumes ray starts INSIDE the atmosphere
        # this is fine since the camera will always be at sea level for the skybox rendering
        steps = 128
        fsteps = 1.0 / 128.0
        step_delta = rsi(ray_pos, ray_dir, self.planet_r + self.atmos_height).y * fsteps
        ray_step = ray_dir*step_delta
        ray_pos = ray_pos + ray_step*(0.5 * (max(ray_dir.y, 0.0) * 0.5 + 0.5))

        od  = vec3(0.0, 0.0, 0.0)
        for i in range(0, steps):
            elevation = self.get_elevation(ray_pos)

            densities = self.get_density(elevation)
            
            od += densities * step_delta
            ray_pos += ray_step
        od = self.extinc_mat @ od
        transmittance = ti.exp(-(od))

        if rsi(ray_pos, ray_dir, self.planet_r).x > 0.0:
            transmittance *= 0.0
        return transmittance

    @ti.func
    def get_ozone_density(self, h):
        # A curve that roughly fits measured data for ozone distribution.

        h_km = h * 0.001 # elevation in km

        peak_height = 25.0
        h_peak_relative_sqr = h_km - peak_height # Square of difference between peak location
        h_peak_relative_sqr = h_peak_relative_sqr*h_peak_relative_sqr

        peak_density = 1. # density at the peak
        
        d = (peak_density - 0.375) * exp(-h_peak_relative_sqr / 49.0) # main peak
        d += 0.375 * exp(-h_peak_relative_sqr / 256.0) # "tail", makes falloff of peak more gradual
        d += max(0.0, -0.000015 * pow(h_km - 15.0, 3.0)) # density becomes almost constant at low altitudes
                                                         # could modify the coefficients to model the small increase 
                                                         # in ozone at the very bottom that happens due to pollution

        return d * 4.

    @ti.func
    def get_density(self, h):
        h = max(h, 0.0)
        return vec3(exp(-h/self.scale_height_rayl), exp(-h/self.scale_height_mie), self.get_ozone_density(h))

    @ti.func
    def get_elevation(self, pos):
        return ti.sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z) - self.planet_r
