import math
import taichi as ti
from taichi.math import *
import numpy as np

from renderer.bsdf import DisneyBSDF
from renderer.materials import MaterialList
from renderer.voxel_world import VoxelWorld
from renderer.raytracer import VoxelOctreeRaytracer
from renderer.math_utils import *
from renderer.space_transformations import *
from renderer.reservoir import *

USE_RESTIR_PT = False

MAX_RAY_DEPTH = 4
use_directional_light = True

RADIANCE_CLAMP = 3200.0

@ti.func
def firefly_filter(v):
    return clamp(v, 0.0, RADIANCE_CLAMP)

@ti.data_oriented
class Renderer:
    def __init__(self, dx, image_res, up, voxel_edges, exposure=3):
        self.image_res = image_res
        self.inv_image_res = ti.Vector([1.0 / image_res[0], 1.0 / image_res[1]])
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0
        self.current_frame = 0

        self.color_buffer = ti.Vector.field(3, dtype=ti.f16)
        self.history_buffer = ti.Vector.field(4, dtype=ti.f16)
        self.fov = ti.field(dtype=ti.f32, shape=())

        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_cone_cos_theta_max = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_weight = ti.field(dtype=ti.f32, shape=())

        self.cast_voxel_hit = ti.field(ti.i32, shape=())
        self.cast_voxel_index = ti.Vector.field(3, ti.i32, shape=())

        self.exposure = exposure

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.prev_camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.floor_height = ti.field(dtype=ti.f32, shape=())
        self.floor_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.floor_material = ti.field(dtype=ti.i32, shape=())

        self.background_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        # By interleaving with 16x8 blocks,
        # each thread block will process 16x8 pixels in a batch instead of a 32 pixel row in a batch
        # Thus we pay less divergence penalty on hard paths (e.g. voxel edges)
        ti.root.dense(ti.ij, (image_res[0] // 16, image_res[1] // 8)).dense(ti.ij, (16, 8)).place(self.color_buffer)
        ti.root.dense(ti.ij, (image_res[0] // 32, image_res[1] // 32)).dense(ti.ijk, (32, 32, 2)).place(self.history_buffer)

        self.voxel_grid_res = 128
        self.world = VoxelWorld(dx, self.voxel_grid_res, voxel_edges)
        self.voxel_raytracer = VoxelOctreeRaytracer(self.voxel_grid_res)

        self._rendered_image = ti.Texture(ti.f32, 4, self.image_res)
        self.set_up(*up)
        self.set_fov(np.deg2rad(50.0))

        self.floor_height[None] = 0
        self.floor_color[None] = (1, 1, 1)
        self.floor_material[None] = 1

        self.bsdf = DisneyBSDF()
        self.mats = MaterialList()

        # matrices
        self.proj_mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.proj_mat_inv = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.view_mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.view_mat_inv = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.prev_proj_mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.prev_view_mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        

        # Reservoir storage
        self.spatial_reservoirs = StorageReservoir.field()
        ti.root.dense(ti.ij, (image_res[0] // 32, image_res[1] // 32)).dense(ti.ijk, (32, 32, 2)).place(self.spatial_reservoirs)

        # Gbuffer
        self.gbuff_mat_id = ti.field(dtype=ti.u32)
        self.gbuff_normals = ti.Vector.field(2, dtype=ti.f16)
        # self.gbuff_depth = ti.field(dtype=ti.f32)
        self.gbuff_position = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.ij, (image_res[0] // 32, image_res[1] // 32)).dense(ti.ij, (32, 32)).place(self.gbuff_mat_id)
        ti.root.dense(ti.ij, (image_res[0] // 32, image_res[1] // 32)).dense(ti.ij, (32, 32)).place(self.gbuff_normals)
        ti.root.dense(ti.ij, (image_res[0] // 32, image_res[1] // 32)).dense(ti.ij, (32, 32)).place(self.gbuff_position)
        # self.gbuff_mat_id = ti.field(dtype=ti.u32, shape=(image_res[0], image_res[1]))
        # self.gbuff_normals = ti.Vector.field(2, dtype=ti.f16, shape=(image_res[0], image_res[1]))
        # self.gbuff_position = ti.Vector.field(3, dtype=ti.f32, shape=(image_res[0], image_res[1])) 

    def set_directional_light(self, direction, light_cone_angle, light_color):
        self.light_direction[None] = ti.Vector(direction).normalized()
        # Theta is half-angle of the light cone
        self.light_cone_cos_theta_max[None] = math.cos(light_cone_angle * 0.5)
        self.light_color[None] = light_color
        self.light_weight = 3.0 # light brightness multiplier times cone sampling pdf

    @ti.func
    def _sdf_march(self, p, d):
        dist = (self.floor_height[None] - p[1]) / d[1]
        return dist

    @ti.func
    def _sdf_normal(self, p):
        return ti.Vector([0.0, 1.0, 0.0])  # up

    @ti.func
    def _sdf_color(self, p):
        return self.floor_color[None]
    
    @ti.func
    def world_to_voxel(self, pos):
        return self.world.voxel_inv_size * pos - self.world.voxel_grid_offset

    @ti.func
    def voxel_to_world(self, pos):
        return self.world.voxel_size * (pos + self.world.voxel_grid_offset)

    @ti.func
    def _trace_sdf(
        self, pos, dir, closest_hit_dist: ti.template(),
        normal: ti.template(), color: ti.template(), is_light: ti.template(), mat_id : ti.template()
    ):
        # Ray marching
        ray_march_dist = self._sdf_march(pos, dir)
        if ray_march_dist > eps and ray_march_dist < closest_hit_dist:
            hit_pos = pos + dir * ray_march_dist
            sdf_normal = self._sdf_normal(hit_pos)
            if length(hit_pos - dot(hit_pos, sdf_normal)) < 10.0:
                closest_hit_dist = ray_march_dist
                normal = sdf_normal
                if normal.dot(dir) > 0:
                    normal = -normal
                color = self._sdf_color(hit_pos)
                is_light = self.floor_material[None] == 2
                mat_id = self.floor_material[None]

    @ti.func
    def _trace_voxel(
        self, eye_pos, d, colors: ti.template(),
        closest_hit_dist: ti.template(), normal: ti.template(), color: ti.template(), is_light: ti.template(), mat_id : ti.template(),
        shadow_ray: ti.template()
    ):
        # Re-scale the ray origin from world space to voxel grid space [0, voxel_grid_res)
        eye_pos_scaled = self.world_to_voxel(eye_pos)

        hit_distance, voxel_index, hit_normal, iters = self.voxel_raytracer.raytrace(
            eye_pos_scaled, d, eps, inf)

        # If the ray hits a voxel, get the surface data
        if hit_distance * self.world.voxel_size < closest_hit_dist:
            # Re-scale from the voxel grid space back to world space
            closest_hit_dist = hit_distance * self.world.voxel_size
            if ti.static(not shadow_ray):
                voxel_uv = ti.math.clamp(eye_pos_scaled + hit_distance * d - voxel_index, 0.0, 1.0)
                voxel_index += self.world.voxel_grid_offset
                # Get surface data
                color, is_light, mat_id = self.world.voxel_surface_color(
                    voxel_index, voxel_uv, colors)
                normal = hit_normal

        return voxel_index, iters

    @ti.func
    def next_hit(self, pos, d, max_dist, colors: ti.template(), shadow_ray: ti.template()):
        # Hit Data
        closest_hit_dist = max_dist
        normal = ti.Vector([0.0, 0.0, 0.0])
        albedo = ti.Vector([0.0, 0.0, 0.0])
        hit_light = 0
        mat_id = 0

        # Intersect with implicit SDF
        self._trace_sdf(pos, d, closest_hit_dist, normal, albedo, hit_light, mat_id)

        # Then intersect with voxel grid
        vx_idx, iters = self._trace_voxel(
            pos, d, colors, closest_hit_dist, normal, albedo, hit_light, mat_id, shadow_ray)

        # Highlight the selected voxel
        if ti.static(not shadow_ray):
            if self.cast_voxel_hit[None]:
                cast_vx_idx = self.cast_voxel_index[None]
                if all(cast_vx_idx == vx_idx):
                    albedo = ti.Vector([1.0, 0.65, 0.0])
                    # For light sources, we actually invert the material to make it
                    # more obvious
                    hit_light = 1 - hit_light

        return closest_hit_dist, normal, albedo, hit_light, iters, mat_id

    @ti.kernel
    def set_camera_pos(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.camera_pos[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_up(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.up[None] = ti.Vector([x, y, z]).normalized()

    @ti.kernel
    def set_look_at(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.look_at[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_fov(self, fov: ti.f32):
        self.fov[None] = fov
    
    @ti.kernel
    def set_proj_mat(self, M: ti.types.ndarray()):
        for i in ti.static(range(4)):  # a parallel for loop
            for j in ti.static(range(4)):
                self.proj_mat[None][j, i] = M[i, j]
        self.proj_mat_inv[None] = inverse(self.proj_mat[None])
            

    @ti.kernel
    def set_view_mat(self, M: ti.types.ndarray()):
        for i in ti.static(range(4)):  # a parallel for loop
            for j in ti.static(range(4)):
                self.view_mat[None][j, i] = M[i, j]
        self.view_mat_inv[None] = inverse(self.view_mat[None])

    @ti.kernel
    def copy_prev_matrices(self):
        self.prev_proj_mat[None] = self.proj_mat[None]
        self.prev_view_mat[None] = self.view_mat[None]
        self.prev_camera_pos[None] = self.camera_pos[None]

    @ti.func
    def get_cast_dir(self, u, v):
        # half_fov = self.fov[None] * 0.5
        # d = (self.look_at[None] - self.camera_pos[None]).normalized()
        # fu = (
        #     2 * half_fov * (u + 0.5) / self.image_res[1]
        #     - half_fov * self.aspect_ratio
        #     - 1e-5
        # )
        # fv = 2 * half_fov * (v + 0.5) / self.image_res[1] - half_fov - 1e-5
        # du = d.cross(self.up[None]).normalized()
        # dv = du.cross(d).normalized()
        # d = (d + fu * du + fv * dv).normalized()
        # return d
        texcoord = (ti.Vector([u, v]) + 0.5) * self.inv_image_res
        d = screen_to_view(texcoord, 1.0, self.proj_mat_inv[None]).normalized()
        d = view_to_world(d, self.view_mat_inv[None], 0.0)
        return d

    def prepare_data(self):
        self.world.update_data()
        self.voxel_raytracer._update_lods(
            self.world.voxel_material, ti.Vector(self.world.voxel_grid_offset))

    @ti.func
    def generate_new_sample(self, u: ti.f32, v: ti.f32):
        d = self.get_cast_dir(u, v)
        pos = self.camera_pos[None]

        contrib = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])
        rc_incident_L = ti.Vector([0.0, 0.0, 0.0])
        rc_throughput = ti.Vector([1.0, 1.0, 1.0])

        hit_light = 0
        hit_background = 0

        input_sample_reservoir = Reservoir()
        input_sample_reservoir.init()

        return d, pos, contrib, throughput, hit_light, hit_background, input_sample_reservoir

    @ti.func
    def power_heuristic(self, a, b):
        a_sqr = a*a
        p_sum = max(a_sqr + b*b, 1e-4)
        return a_sqr/p_sum

    @ti.kernel
    def render(self, colors: ti.types.texture(3)):

        # Render
        ti.loop_config(block_dim=128)
        for u, v in self.color_buffer:
            (
                d,
                pos,
                contrib,
                throughput,
                hit_light,
                hit_background,
                input_sample_reservoir
            ) = self.generate_new_sample(u, v)

            # values to populate gbuffer with
            primary_normal = ti.cast(ti.Vector([0.0, 0.0]), ti.f16)
            primary_pos = vec3(0,0,0)
            primary_mat_info = ti.cast(0, ti.u32)

            # values to extract from PT
            throughput_after_rc = ti.Vector([1.0, 1.0, 1.0]) 
            first_bounce_lobe_id = 0
            first_bounce_brdf = ti.Vector([0.0, 0.0, 0.0]) 
            first_bounce_invpdf = 1.0
            first_vertex_NEE = ti.Vector([0.0, 0.0, 0.0])
            first_bounce_dir = ti.Vector([0.0, 0.0, 0.0]) 
            first_light_sample_bsdf_pdf = 1.0 # first vertex 
            first_light_sample_dir = ti.Vector([0.0, 0.0, 0.0])
            rc_bounce_lobe_id = 0

            is_sky_ray = False

            # Tracing begin
            for depth in range(MAX_RAY_DEPTH):
                closest, normal, albedo, hit_light, iters, mat_id = self.next_hit(
                    pos, d, inf, colors, shadow_ray=False)
                hit_mat = self.mats.mat_list[mat_id]
                hit_pos = pos + closest * d

                # Write to gbuffer and reservoir
                if(depth == 0):
                    primary_normal = encode_unit_vector_3x16(normal)
                    primary_pos = hit_pos
                    primary_mat_info = encode_material(mat_id, albedo)
                    
                elif(depth == 1):
                    input_sample_reservoir.z.rc_pos = hit_pos
                    input_sample_reservoir.z.rc_normal = normal
                    input_sample_reservoir.z.rc_mat_info = encode_material(mat_id, albedo)
                    first_bounce_dir = d
                elif(depth == 2):
                    input_sample_reservoir.z.rc_incident_dir = d

                # Enable this to debug iteration counts
                if ti.static(False):
                    if depth == 1:
                        worst_case_iters = ti.simt.subgroup.reduce_max(iters)
                        best_case_iters = ti.simt.subgroup.reduce_min(iters)
                        self.color_buffer[u, v] += ti.Vector(
                            [worst_case_iters / 64.0, best_case_iters / 64.0, 0.0])

                if not hit_light and closest < inf:
                    pos = hit_pos + normal * eps
                    hit_mat.base_col = albedo
                    view = -d

                    tang, bitang = make_orthonormal_basis(normal)

                    NEE_visible = 0.0
                    if ti.static(use_directional_light):
                        light_dir = sample_cone_oriented(
                            self.light_cone_cos_theta_max[None], self.light_direction[None])
                        dot = light_dir.dot(normal)

                        light_sample_bsdf_pdf = self.bsdf.pdf_disney(hit_mat, view, normal, light_dir, tang, bitang)
                        light_sample_dir = light_dir

                        if depth == 0:
                            first_light_sample_bsdf_pdf = light_sample_bsdf_pdf
                            first_light_sample_dir = light_sample_dir
                        if dot > 0:
                            hit_light_ = 0
                            dist, _, _, hit_light_, iters, smat = self.next_hit(
                                pos, light_dir, inf, colors, shadow_ray=True
                            )
                            if dist >= inf:
                                NEE_visible = 1.0
                                # far enough to hit directional light
                                if depth == 1:
                                    input_sample_reservoir.z.rc_NEE_dir = light_dir

                                light_sample_mis_weight = 1.0

                                if depth > 0:
                                    light_sample_light_pdf = cone_sample_pdf(self.light_cone_cos_theta_max[None], 1.0)
                                    light_sample_mis_weight = self.power_heuristic(light_sample_light_pdf, light_sample_bsdf_pdf)

                                light_bsdf = self.bsdf.disney_evaluate(hit_mat, view, normal, light_dir, tang, bitang)
                                NEE_contrib = firefly_filter(light_sample_mis_weight * light_bsdf * self.light_weight * self.light_color[None] * dot)
                                if depth == 0:
                                    first_vertex_NEE += throughput * NEE_contrib
                                else:
                                    contrib += throughput * NEE_contrib
                                    
                                if depth >= 2:
                                    input_sample_reservoir.z.rc_incident_L += throughput_after_rc * NEE_contrib
                    
                    # Sample next bounce
                    d, bsdf, pdf, lobe_id = self.bsdf.sample_disney(hit_mat, view, normal, tang, bitang)
                    # d = sample_cosine_weighted_hemisphere(normal)
                    # pdf = saturate(d.dot(normal)) / np.pi
                    # bsdf = self.bsdf.disney_evaluate(hit_mat, view, normal, d, tang, bitang)
                    # lobe_id = 0

                    # Apply weight to throughput (Not on first bounce)
                    bounce_weight = bsdf * saturate(d.dot(normal))
                    if depth == 0:
                        first_bounce_invpdf = 1.0/pdf
                        first_bounce_lobe_id = lobe_id
                    else:
                        bounce_weight /= pdf

                        # MIS weight
                        bsdf_sample_light_pdf = cone_sample_pdf(self.light_cone_cos_theta_max[None], self.light_direction[None].dot(d))
                        bounce_weight *= self.power_heuristic(pdf, NEE_visible*bsdf_sample_light_pdf)

                        if depth == 1:
                            rc_bounce_lobe_id = lobe_id
                        if depth >= 2:
                            throughput_after_rc *= bounce_weight
                    throughput *= bounce_weight
                    
                else:
                    if closest == inf:
                        # Hit background
                        hit_sun = 1.0 if self.light_direction[None].dot(d) >= self.light_cone_cos_theta_max[None] else 0.0
                        sky_emission = firefly_filter(self.background_color[None] + self.light_weight * self.light_color[None] * hit_sun)
                        contrib += throughput * sky_emission
                        if depth == 0:
                            # first_vertex_NEE += throughput * sky_emission
                            primary_pos = vec3(0.0, 0.0, 0.0)
                            is_sky_ray = True
                        elif depth == 1:
                            input_sample_reservoir.z.rc_pos = d
                            input_sample_reservoir.z.rc_incident_L = sky_emission
                        
                        if depth >= 2:
                            input_sample_reservoir.z.rc_incident_L += throughput_after_rc * sky_emission
                    else:
                        # hit light voxel, terminate tracing
                        if depth > 0:
                            contrib += throughput * albedo

                        if depth >= 2:
                            input_sample_reservoir.z.rc_incident_L += throughput_after_rc * albedo
                    break

                # RR is disabled for now since it complicates ReSTIR PT a litte bit
                # Russian roulette (with ReSTIR PT, we won't do RR on first bounce)
                # max_c = clamp(throughput.max(), 0.0, 1.0)
                # if ti.random() > max_c:
                #     break
                # else:
                #     throughput /= max_c

            # primary_pos_view = world_to_view(primary_pos, self.view_mat[None])
            
            # Write to gbuffer
            self.gbuff_normals[u, v]  = primary_normal
            # self.gbuff_depth[u, v] = view_to_screen(primary_pos_view.xyz, self.proj_mat[None]).z
            self.gbuff_position[u, v] = primary_pos
            self.gbuff_mat_id[u, v] = primary_mat_info

            # Finish populating reservoir fields
            input_sample_reservoir.z.F = contrib
            input_sample_reservoir.z.lobes = rc_bounce_lobe_id*10 + first_bounce_lobe_id
            input_sample_reservoir.M = 1.0
            input_sample_reservoir.update_cached_jacobian_term(primary_pos)
            
            # MIS for primary vertex (we do it separate from other bounces because they will be used on GRIS weights)
            if not is_sky_ray:
                if ti.static(use_directional_light):
                    bsdf_sample_bsdf_pdf = 1.0 / first_bounce_invpdf
                    bsdf_sample_light_pdf = cone_sample_pdf(self.light_cone_cos_theta_max[None], self.light_direction[None].dot(first_bounce_dir))
                    if is_vec_zero(first_vertex_NEE):
                        bsdf_sample_light_pdf = 0.0
                    
                    bsdf_sample_mis_weight = self.power_heuristic(bsdf_sample_bsdf_pdf, bsdf_sample_light_pdf)

                    light_sample_light_pdf = cone_sample_pdf(self.light_cone_cos_theta_max[None], 1.0)
                    # first_light_sample_bsdf_pdf = already defined!
                    light_sample_mis_weight = self.power_heuristic(light_sample_light_pdf, first_light_sample_bsdf_pdf)

                    if ti.static(not USE_RESTIR_PT):
                        input_sample_reservoir.z.F *= bsdf_sample_mis_weight
                    p_hat = luminance(input_sample_reservoir.z.F)
                    input_sample_reservoir.weight = bsdf_sample_mis_weight * p_hat * first_bounce_invpdf

                    if ti.static(not USE_RESTIR_PT):
                        first_vertex_NEE *= light_sample_mis_weight
                    light_sample_weight = light_sample_mis_weight * luminance(first_vertex_NEE) # rcp pdf already applied in first_vertex_NEE
                
                    # Since our only NEE light is an angular light source, we will consider NEE vertices as just escape vertices.
                    # So, the way this implementation of ReSTIR PT is set up, it will not work for NEE on arbitrary light sources.
                    # This should be fine, since in our voxel scene, we won't have any explicit lights other than the sun,
                    # and emissive materials are handled through BSDF sampling.
                    # Hopefully nobody wants voxel-rt2 to support other types of explicit light sources.
                    # If so, handling would be required for these types of light sources,
                    # probably storing light_pdf in the reservoir sample.
                    light_sample = Sample(F=first_vertex_NEE, \
                                        rc_pos=first_light_sample_dir, \
                                        rc_normal=vec3(0,0,0), \
                                        rc_incident_dir=vec3(0,0,0), \
                                        rc_incident_L=self.light_weight * self.light_color[None], \
                                        rc_NEE_dir=vec3(0,0,0), \
                                        rc_mat_info=0, \
                                        cached_jacobian_term=1.0, lobes = 99) # 9 indicates all lobes will be used

                    input_sample_reservoir.input_sample(light_sample_weight, light_sample)
                else:
                    p_hat = luminance(contrib)
                    input_sample_reservoir.weight = p_hat * first_bounce_invpdf
                input_sample_reservoir.finalize_without_M()
            else:
                input_sample_reservoir.weight = 1.0


            self.spatial_reservoirs[u, v, 0] = input_sample_reservoir.encode()

            if ti.static(not USE_RESTIR_PT):
                primary_mat, primary_mat_id = decode_material(self.mats.mat_list, primary_mat_info)
                emission = primary_mat.base_col if primary_mat_id == 2 else vec3(0.0, 0.0, 0.0)

                self.color_buffer[u, v] = input_sample_reservoir.z.F * first_bounce_invpdf + first_vertex_NEE + emission
            # self.color_buffer[u, v] += contrib

    @ti.kernel
    def _render_to_image(self, img : ti.types.rw_texture(num_dimensions=2, num_channels=4, channel_format=ti.f32, lod=0), samples: ti.i32):
        for i, j in self.color_buffer:
            uv = ti.Vector([i, j]) / self.image_res

            darken = 1.0 - self.vignette_strength * \
                max(ti.math.distance(uv, self.vignette_center) -
                    self.vignette_radius, 0.0)

            ldr_color = ti.pow(
                uchimura(self.color_buffer[i, j] * darken * self.exposure), 1.0 / 2.2)
            img.store(ti.Vector([i, j]), ti.Vector([ldr_color.r, ldr_color.g, ldr_color.b, 1.0]))

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    # returns 1: computed integrand after it is shifted from src_reservoir into this reservoir's domain
    # returns 2: Jacobian determinant of this transformation
    @ti.func
    def shift(self, dst_pos, dst_normal, dst_material, \
                    src_pos, src_normal, src_material, \
                    src_reservoir):
        # RC vertex info
        rc_is_escape_vertex = is_vec_zero(src_reservoir.z.rc_normal)
        rc_is_last_vertex = is_vec_zero(src_reservoir.z.rc_incident_dir)
        rc_is_NEE_visible = not is_vec_zero(src_reservoir.z.rc_NEE_dir)

        dir_to_rc_vertex = src_reservoir.z.rc_pos if rc_is_escape_vertex else (src_reservoir.z.rc_pos - dst_pos).normalized()
        src_dir_to_rc_vertex = src_reservoir.z.rc_pos if rc_is_escape_vertex else (src_reservoir.z.rc_pos - src_pos).normalized()

        # NdotL at primary and rc vertices
        passed_checks = 1.0
        if dst_normal.dot(dir_to_rc_vertex) < 1e-5 or \
           (not rc_is_escape_vertex and src_reservoir.z.rc_normal.dot(-dir_to_rc_vertex) < 1e-5):
           passed_checks = 0.0

        rc_tang, rc_bitang = make_orthonormal_basis(src_reservoir.z.rc_normal)
        rc_mat, rc_mat_id = decode_material(self.mats.mat_list, src_reservoir.z.rc_mat_info)

        # Rc vertex weights
        rc_brdf = vec3(0., 0., 0.)
        dst_rc_pdf = 1.0
        src_rc_pdf = 1.0
        if not rc_is_last_vertex and not rc_is_escape_vertex:
            rc_brdf = self.bsdf.disney_evaluate_lobewise(rc_mat, \
                                                        -dir_to_rc_vertex, \
                                                         src_reservoir.z.rc_normal, \
                                                         src_reservoir.z.rc_incident_dir, \
                                                         rc_tang, rc_bitang, \
                                                         src_reservoir.z.lobes // 10)
            rc_brdf *= saturate(src_reservoir.z.rc_normal.dot(src_reservoir.z.rc_incident_dir))
            dst_rc_pdf = self.bsdf.pdf_disney_lobewise(rc_mat, \
                                             -dir_to_rc_vertex, \
                                              src_reservoir.z.rc_normal, \
                                              src_reservoir.z.rc_incident_dir, \
                                              rc_tang, rc_bitang, \
                                              src_reservoir.z.lobes // 10)
            
            src_rc_pdf = self.bsdf.pdf_disney_lobewise(rc_mat, \
                                             -src_dir_to_rc_vertex, \
                                              src_reservoir.z.rc_normal, \
                                              src_reservoir.z.rc_incident_dir, \
                                              rc_tang, rc_bitang, \
                                              src_reservoir.z.lobes // 10)

        rc_nee_brdf = vec3(0., 0., 0.)
        # rc_nee_pdf = 1.0
        if rc_is_NEE_visible:
            rc_nee_brdf = self.bsdf.disney_evaluate(rc_mat, \
                                                            -dir_to_rc_vertex, \
                                                             src_reservoir.z.rc_normal, \
                                                             src_reservoir.z.rc_NEE_dir, \
                                                             rc_tang, rc_bitang)
            rc_nee_brdf *= saturate(src_reservoir.z.rc_normal.dot(src_reservoir.z.rc_NEE_dir))
            # rc_nee_pdf = cone_sample_pdf(self.light_cone_cos_theta_max[None], self.light_direction[None].dot(src_reservoir.z.rc_NEE_dir))

        # primary dst vertex weights
        dst_tang, dst_bitang = make_orthonormal_basis(dst_normal)
        view = (self.camera_pos[None] - dst_pos).normalized()
        primary_brdf = self.bsdf.disney_evaluate_lobewise(dst_material, \
                                                          view, \
                                                          dst_normal, \
                                                          dir_to_rc_vertex, \
                                                          dst_tang, dst_bitang, \
                                                          src_reservoir.z.lobes % 10) # Evaluating the source reservoir's
                                                                                      # BSDF lobe since that is part of
                                                                                      # the source sample.
        primary_brdf *= saturate(dst_normal.dot(dir_to_rc_vertex))
        # dst_primary_pdf = self.bsdf.pdf_disney_lobewise(dst_material, \
        #                                        view, \
        #                                        dst_normal, \
        #                                        dir_to_rc_vertex, \
        #                                        dst_tang, dst_bitang, \
        #                                        src_reservoir.z.lobes % 10)
        
        # primary src vertex weights
        # src_tang, src_bitang = make_orthonormal_basis(src_normal)
        # src_primary_pdf = self.bsdf.pdf_disney_lobewise(src_material, \
        #                                        (self.camera_pos[None] - src_pos).normalized(), \
        #                                        src_normal, \
        #                                        src_dir_to_rc_vertex, \
        #                                        src_tang, src_bitang, \
        #                                        src_reservoir.z.lobes % 10)
        
        # compute shifted integrand
        contrib = vec3(0., 0., 0.)
        if not rc_is_escape_vertex and not rc_is_last_vertex:
            # RC BSDF MIS weight
            rc_bsdf_sample_light_pdf = cone_sample_pdf(self.light_cone_cos_theta_max[None], \
                                                    self.light_direction[None].dot(src_reservoir.z.rc_incident_dir))
            rc_bsdf_mis_weight = self.power_heuristic(dst_rc_pdf, rc_bsdf_sample_light_pdf * (1.0 if rc_is_NEE_visible else 0.0))

            contrib += firefly_filter(rc_bsdf_mis_weight * rc_brdf / dst_rc_pdf * src_reservoir.z.rc_incident_L) # rc bounce
        if rc_is_escape_vertex:
            contrib += firefly_filter(src_reservoir.z.rc_incident_L) # sky emission
        if ti.static(use_directional_light):
            if rc_is_NEE_visible and not rc_is_escape_vertex:
                rc_light_sample_bsdf_pdf = self.bsdf.pdf_disney(rc_mat, \
                                                               -dir_to_rc_vertex, 
                                                                src_reservoir.z.rc_normal, 
                                                                src_reservoir.z.rc_NEE_dir, 
                                                                rc_tang, rc_bitang)
                rc_light_sample_light_pdf = cone_sample_pdf(self.light_cone_cos_theta_max[None], 1.0)
                rc_light_sample_mis_weight = self.power_heuristic(rc_light_sample_light_pdf, rc_light_sample_bsdf_pdf)
                contrib += firefly_filter(rc_light_sample_mis_weight * rc_nee_brdf * self.light_weight * self.light_color[None]) # NEE at rc vertex
        contrib += vec3(0., 0., 0.) if rc_mat_id != 2 else rc_mat.base_col # emission at rc vertex

        contrib *= primary_brdf

        # compute jacobian
        jacobian = 1.0 # limit case of escape vertices converges to one

        # jacobian *= dst_rc_pdf / max(src_rc_pdf, 1e-4)
        # jacobian *= dst_primary_pdf / max(src_primary_pdf, 1e-4)

        if not rc_is_escape_vertex:
            jacobian = src_reservoir.z.cached_jacobian_term
            dir_y1_to_x2 = src_reservoir.z.rc_pos - dst_pos
            jacobian *= abs(dir_y1_to_x2.normalized().dot(src_reservoir.z.rc_normal)) / dir_y1_to_x2.dot(dir_y1_to_x2)
            
        
        if jacobian < 0.0 or isnan(jacobian) or isinf(jacobian): 
            jacobian = 0.0
            if max(jacobian, 1.0 / jacobian) > 11.0: # 11 rejection threshold
                contrib = vec3(0.0, 0.0, 0.0)
        
        # rc_dist = 1.0
        # if not rc_is_escape_vertex:
        #     rc_dist = length(src_reservoir.z.rc_pos - dst_pos)
        # if rc_dist < 0.02: # is_vec_zero(primary_brdf) or is_vec_zero(rc_brdf) or 
        #     contrib = vec3(0., 0., 0.)


        return contrib, jacobian*passed_checks


    @ti.kernel
    def spatial_GRIS(self, pass_id : ti.i32, max_radius : ti.f32, max_taps : ti.i32, pass_total : ti.i32, colors: ti.types.texture(3)):

        # Render
        ti.loop_config(block_dim=256)
        for u, v in self.gbuff_position:

            texcoord = (ti.Vector([u, v]) + 0.5) * self.inv_image_res
            
            start_index = ti.cast(ti.random() * max_taps, ti.u32)
            offset_mask = ti.cast(max_taps - 1, ti.u32)

            
            # Generate random offset seeds
            seed_x = ti.cast(u, ti.u32) >> 3 if pass_id == 0 else 2
            seed_y = ti.cast(v, ti.u32) >> 3 if pass_id == 0 else 2
            seed = hash3(seed_x, seed_y, ti.cast(self.current_frame*2, ti.u32) + pass_id)

            angle_shift = ti.cast(((seed & 0x007FFFFF) | 0x3F800000), ti.f32)/4294967295.0 * np.pi
            radius_shift = ti.random()

            # Get center reservoir
            center_reservoir_enc = self.spatial_reservoirs[u, v, pass_id % 2]
            center_reservoir = Reservoir()
            center_reservoir.decode(center_reservoir_enc)

            # Initialize output reservoir
            output_reservoir = Reservoir()
            output_reservoir.init()

            # Variables of center pixel
            # center_depth = self.gbuff_depth[u, v]
            # center_x1 = screen_to_view(texcoord, center_depth, self.proj_mat_inv[None])
            # center_dist = length(center_x1)
            # center_x1 = view_to_world(center_x1, self.view_mat_inv[None]) # get in world space
            center_x1 = self.gbuff_position[u, v]
            center_dist = distance(center_x1, self.camera_pos[None])
            # center_depth = linearize_depth(center_depth, self.proj_mat_inv[None]) # linearize for use in depth comparison
            center_n1 = decode_unit_vector_3x16(self.gbuff_normals[u, v])

            if is_vec_zero(center_x1):
                self.color_buffer[u, v] = center_reservoir.z.F
                continue

            center_mat, center_mat_id = decode_material(self.mats.mat_list, self.gbuff_mat_id[u, v])

           
            view = (self.camera_pos[None] - center_x1).normalized()
            tang, bitang = make_orthonormal_basis(center_n1)

            valid_samples = 0

            # For MIS weighting of RIS weights, we are doing defensive pairwise MIS.
            # Algorithm from http://benedikt-bitterli.me/Data/dissertation.pdf 
            # Using the generalized form from the ReSTIR PT paper.
            canonical_mis_weight = 1.0

            for i in range(max_taps):
                # neighbor_index = ti.cast(start_index + i, ti.u32) & offset_mask
                # offset = ti.cast(self.reuse_offsets[neighbor_index] * max_radius, ti.i32)
                golden_angle = 2.399963229728
                
                angle = (i + angle_shift) * golden_angle
                offset_radius = ti.sqrt(float(i + radius_shift) / ti.cast(max_taps, ti.f32)) * max_radius

                offset = ti.cast(ti.Vector([ti.cos(angle) * offset_radius, ti.sin(angle) * offset_radius]), ti.i32)

                if offset.x == 0 and offset.y == 0:
                    continue
                
                tap_coord = ti.Vector([u, v]) + offset
                tap_texcoord = (tap_coord + 0.5) * self.inv_image_res

                neighbour_n1 = decode_unit_vector_3x16(self.gbuff_normals[tap_coord.x, tap_coord.y])
                # neighbour_depth = self.gbuff_depth[tap_coord.x, tap_coord.y]
                # neighbour_x1 = screen_to_view(tap_texcoord, neighbour_depth, self.proj_mat_inv[None])
                # neighbour_x1 = view_to_world(neighbour_x1, self.view_mat_inv[None]) # get in world space
                # neighbour_depth = linearize_depth(neighbour_depth, self.proj_mat_inv[None]) # linearize for use in depth comparison
                neighbour_x1 = self.gbuff_position[tap_coord.x, tap_coord.y]
                neighbour_dist = distance(neighbour_x1, self.camera_pos[None])
                neighbour_reservoir_enc = self.spatial_reservoirs[tap_coord.x, tap_coord.y, pass_id % 2]
                neighbour_reservoir = Reservoir()
                neighbour_reservoir.decode(neighbour_reservoir_enc)

                neighbour_primary_lobe_id = neighbour_reservoir.z.lobes % 10

                if abs(neighbour_dist - center_dist) > 0.1*center_dist or center_n1.dot(neighbour_n1) < 0.5:
                    continue

                neighbour_mat, neighbour_mat_id = decode_material(self.mats.mat_list, self.gbuff_mat_id[tap_coord.x, tap_coord.y])

                # Center reservoir's sample shifted into neighbour pixel's domain
                center_integrand, c_jacobian = self.shift(neighbour_x1, neighbour_n1, neighbour_mat, \
                                                          center_x1   , center_n1   , center_mat, \
                                                          center_reservoir)
                
                # Neighbour reservoir's sample shifted into center pixel's domain
                shifted_integrand, jacobian = self.shift(center_x1   , center_n1   , center_mat, \
                                                         neighbour_x1, neighbour_n1, neighbour_mat, \
                                                         neighbour_reservoir)
                
                # Update canonical sample's MIS weight
                center_p_hat = luminance(center_integrand) * c_jacobian
                canonical_weight = center_p_hat * neighbour_reservoir.M
                canonical_weight /= center_p_hat * neighbour_reservoir.M + luminance(center_reservoir.z.F) * center_reservoir.M / ti.cast(max_taps, ti.f32)
                canonical_mis_weight += 1 - canonical_weight

                p_hat = luminance(shifted_integrand)

                # if jacobian < 1e-3: 
                #     continue

                # neighbour sample's MIS weight
                p_hat_from_neighbour = luminance(shifted_integrand) / jacobian

                neighbour_mis_weight = p_hat_from_neighbour * neighbour_reservoir.M
                neighbour_mis_weight /= p_hat_from_neighbour * neighbour_reservoir.M + p_hat * center_reservoir.M / ti.cast(max_taps, ti.f32)
                if isinf(neighbour_mis_weight) or isnan(neighbour_mis_weight):
                    neighbour_mis_weight = 0.0
                
                # Update neighbour reservoir sample to be the shifted one
                neighbour_reservoir.z.F = shifted_integrand
                # neighbour_reservoir.z.lobes = neighbour_reservoir.z.lobes // 10 + center_reservoir.z.lobes % 10

                selected = output_reservoir.merge(neighbour_reservoir, neighbour_reservoir.weight * p_hat * jacobian * neighbour_mis_weight)
                
                valid_samples += 1

            # Shade RIS pass chosen sample (if this RIS sample is occluded, we will use canonical sample)
            force_add_canonical = False
            dir_to_rc_vertex = output_reservoir.z.rc_pos if is_vec_zero(output_reservoir.z.rc_normal) else (output_reservoir.z.rc_pos - center_x1).normalized()
            dist, _, _, hit_light_, iters, smat = self.next_hit(center_x1 + center_n1*0.007*center_dist, dir_to_rc_vertex, inf, colors, shadow_ray=True)
            actual_dist = inf if is_vec_zero(output_reservoir.z.rc_normal) else distance(center_x1, output_reservoir.z.rc_pos)

            if dist < inf and abs(dist - actual_dist) > 0.1*actual_dist:
                output_reservoir.weight = 0.0
                force_add_canonical = True
            
            
            center_p_hat = luminance(center_reservoir.z.F)
            selected = output_reservoir.merge(center_reservoir, center_reservoir.weight * center_p_hat * canonical_mis_weight, force_add_canonical)

            output_reservoir.finalize_without_M()
            output_reservoir.weight /= (valid_samples + 1)


            if pass_id == pass_total - 1:
                emission = center_mat.base_col if center_mat_id == 2 else vec3(0.0, 0.0, 0.0)
                self.color_buffer[u, v] = output_reservoir.z.F * clamp(output_reservoir.weight, 0.0, 50.0) + emission

            output_reservoir.update_cached_jacobian_term(center_x1)
            self.spatial_reservoirs[u, v, (pass_id + 1) % 2] = output_reservoir.encode()

    @ti.func
    def reproject(self, world_pos):
        pos = ti.Vector([world_pos.x, world_pos.y, world_pos.z, 1.0])
        # pos.xyz += self.camera_pos[None] - self.prev_camera_pos[None]
        pos = self.prev_view_mat[None] @ pos
        pos = self.prev_proj_mat[None] @ pos
        pos.xyz /= pos.w
        pos.xyz = pos.xyz * 0.5 + 0.5
        return pos.xyz

    @ti.kernel
    def temporal_filter(self):

        ti.loop_config(block_dim=512)
        for u, v in self.gbuff_position:
            center_x1 = self.gbuff_position[u, v]

            if is_vec_zero(center_x1):
                continue

            center_dist = distance(center_x1, self.camera_pos[None])
            # center_depth = linearize_depth(center_depth, self.proj_mat_inv[None]) # linearize for use in depth comparison
            center_n1 = decode_unit_vector_3x16(self.gbuff_normals[u, v])


            current = self.color_buffer[u, v]

            history = ti.Vector([current.x, current.y, current.z, 1.0])

            # reproject
            reprojected_pos = self.reproject(center_x1)
            reprojected_coord = ti.cast(ti.Vector([reprojected_pos.x * self.image_res[0], reprojected_pos.y * self.image_res[1]]), ti.i32)

            # previous frame pos is on screen
            if all(clamp(reprojected_pos.xy, 0.0, 1.0) == reprojected_pos.xy):
                history = self.history_buffer[reprojected_coord.x, reprojected_coord.y, 0]
                history.w = min(history.w + 1.0, 50.0)
                history.xyz = mix(history.xyz, current, 1.0 / history.w)
            
            self.history_buffer[u, v, 1] = history
            self.color_buffer[u, v] = history.xyz

        ti.loop_config(block_dim=512)
        for u, v in self.gbuff_position:
            self.history_buffer[u, v, 0] = self.history_buffer[u, v, 1]








    
    def accumulate(self):
        self.render(self.world.voxel_color_texture)
        if ti.static(USE_RESTIR_PT):
            self.spatial_GRIS(0, 16.0, 12, 1, self.world.voxel_color_texture)
            # self.spatial_GRIS(1, 16.0, 6, 2, self.world.voxel_color_texture)
        self.temporal_filter()
        self.current_spp += 1
        self.current_frame += 1

    def fetch_image(self):
        self._render_to_image(self._rendered_image, self.current_spp)
        return self._rendered_image

    @ti.func
    def set_voxel(self, idx, mat, color):
        self.world.voxel_material[idx] = ti.cast(mat, ti.i8)
        self.world.voxel_color[idx] = rgb32f_to_rgb8(color)

    @ti.func
    def get_voxel(self, ijk):
        mat = self.world.voxel_material[ijk]
        color = self.world.voxel_color[ijk]
        return mat, rgb8_to_rgb32f(color)
