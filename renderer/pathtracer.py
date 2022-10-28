import math
import taichi as ti
from taichi.math import *
import numpy as np

from renderer.bsdf import DisneyBSDF
from renderer.materials import MaterialList
from renderer.voxel_world import VoxelWorld
from renderer.raytracer import VoxelOctreeRaytracer
from renderer.math_utils import *
from renderer.reservoir import *

USE_RESTIR_PT = True

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
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0
        self.current_frame = 0

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.fov = ti.field(dtype=ti.f32, shape=())

        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_cone_cos_theta_max = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_multiplier = 3.0

        self.cast_voxel_hit = ti.field(ti.i32, shape=())
        self.cast_voxel_index = ti.Vector.field(3, ti.i32, shape=())

        self.exposure = exposure

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
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

        # Hardcoded spatial reuse offsets (Temporary)
        self.reuse_offsets = ti.Vector.field(2, dtype=ti.f32, shape=(9))
        self.reuse_offsets[0] = ti.Vector([-0.490245, -0.860320])
        self.reuse_offsets[0] = ti.Vector([0.529266, -0.580959])
        self.reuse_offsets[0] = ti.Vector([0.039021, 0.558722])
        self.reuse_offsets[0] = ti.Vector([-0.451224, -0.301598])
        self.reuse_offsets[0] = ti.Vector([0.568287, -0.022237])
        self.reuse_offsets[0] = ti.Vector([0.078042, -0.882556])
        self.reuse_offsets[0] = ti.Vector([-0.412203, 0.257125])
        self.reuse_offsets[0] = ti.Vector([0.607308, 0.536486])
        self.reuse_offsets[0] = ti.Vector([0.117063, -0.323834])
        

        # Reservoir storage
        # TODO: Make this use the packed reservoirs and encode/decode on read/write
        self.spatial_reservoirs = Reservoir.field(shape=(image_res[0], image_res[1], 2))

        # Auxiliaries
        self.nee_and_emission_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(image_res[0], image_res[1]))

        # Gbuffer
        self.gbuff_mat_id = ti.field(dtype=ti.u32, shape=(image_res[0], image_res[1]))
        self.gbuff_normals = ti.Vector.field(2, dtype=ti.f16, shape=(image_res[0], image_res[1]))
        self.gbuff_position = ti.Vector.field(3, dtype=ti.f32, shape=(image_res[0], image_res[1])) # TODO: use depth instead and reconstruct pos

    def set_directional_light(self, direction, light_cone_angle, light_color):
        self.light_direction[None] = ti.Vector(direction).normalized()
        # Theta is half-angle of the light cone
        self.light_cone_cos_theta_max[None] = math.cos(light_cone_angle * 0.5)
        self.light_color[None] = light_color

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
        eye_pos_scaled = self.world.voxel_inv_size * eye_pos - self.world.voxel_grid_offset

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

    @ti.func
    def get_cast_dir(self, u, v):
        half_fov = self.fov[None] * 0.5
        d = (self.look_at[None] - self.camera_pos[None]).normalized()
        fu = (
            2 * half_fov * (u + ti.random(ti.f32)) / self.image_res[1]
            - half_fov * self.aspect_ratio
            - 1e-5
        )
        fv = 2 * half_fov * (v + ti.random(ti.f32)) / self.image_res[1] - half_fov - 1e-5
        du = d.cross(self.up[None]).normalized()
        dv = du.cross(d).normalized()
        d = (d + fu * du + fv * dv).normalized()
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
            first_vertex_NEE_and_emission = ti.Vector([0.0, 0.0, 0.0])
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
                    primary_normal = encodeUnitVector3x16(normal)
                    primary_pos = hit_pos
                    primary_mat_info = encode_material(mat_id, albedo)
                    
                elif(depth == 1):
                    input_sample_reservoir.z.rc_pos = hit_pos
                    input_sample_reservoir.z.rc_normal = normal
                    input_sample_reservoir.z.rc_mat_info = encode_material(mat_id, albedo)
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

                    if ti.static(use_directional_light):
                        light_dir = sample_cone_oriented(
                            self.light_cone_cos_theta_max[None], self.light_direction[None])
                        dot = light_dir.dot(normal)
                        if dot > 0:
                            hit_light_ = 0
                            dist, _, _, hit_light_, iters, smat = self.next_hit(
                                pos, light_dir, inf, colors, shadow_ray=True
                            )
                            if dist >= inf:
                                # far enough to hit directional light
                                input_sample_reservoir.z.rc_NEE_dir = light_dir

                                light_bsdf = self.bsdf.disney_evaluate(hit_mat, view, normal, light_dir, tang, bitang)
                                NEE_contrib = firefly_filter(light_bsdf * self.light_color[None] * self.light_multiplier * dot)
                                if depth == 0:
                                    first_vertex_NEE_and_emission += throughput * NEE_contrib
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
                        if depth == 1:
                            rc_bounce_lobe_id = lobe_id
                        if depth >= 2:
                            throughput_after_rc *= bounce_weight
                    throughput *= bounce_weight
                    
                else:
                    if closest == inf:
                        # Hit background
                        sky_emission = firefly_filter(self.background_color[None])
                        contrib += throughput * sky_emission
                        if depth == 0:
                            # first_vertex_NEE_and_emission += throughput * sky_emission
                            primary_pos = vec3(0.0, 0.0, 0.0)
                            is_sky_ray = True
                        elif depth == 1:
                            input_sample_reservoir.z.rc_pos = d
                            input_sample_reservoir.z.rc_incident_L = sky_emission
                        
                        if depth >= 2:
                            input_sample_reservoir.z.rc_incident_L += throughput_after_rc * sky_emission
                    else:
                        # hit light voxel, terminate tracing
                        if depth == 0:
                            first_vertex_NEE_and_emission += albedo
                        else:
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

            
            # Write to gbuffer
            self.gbuff_normals[u, v]  = primary_normal
            self.gbuff_position[u, v] = primary_pos
            self.gbuff_mat_id[u, v] = primary_mat_info

            # Finish populating reservoir fields
            input_sample_reservoir.z.F = contrib
            input_sample_reservoir.z.lobes = rc_bounce_lobe_id*10 + first_bounce_lobe_id
            input_sample_reservoir.M = 1.0
            input_sample_reservoir.update_cached_jacobian_term(primary_pos)


            p_hat = luminance(contrib)
            input_sample_reservoir.weight = p_hat * first_bounce_invpdf # MIS weight here is just 1.0

            input_sample_reservoir.finalize() # set reservoir W to now hold the initial 
                                              # sample's unbiased contribution weight 
            
            if is_sky_ray:
                input_sample_reservoir.weight = 1.0

            self.spatial_reservoirs[u, v, 0] = input_sample_reservoir

            if ti.static(not USE_RESTIR_PT):
                self.color_buffer[u, v] += input_sample_reservoir.z.F * input_sample_reservoir.weight + first_vertex_NEE_and_emission
            self.nee_and_emission_buffer[u, v] = first_vertex_NEE_and_emission
            # self.color_buffer[u, v] += contrib

    @ti.kernel
    def _render_to_image(self, img : ti.types.rw_texture(num_dimensions=2, num_channels=4, channel_format=ti.f32, lod=0), samples: ti.i32):
        for i, j in self.color_buffer:
            uv = ti.Vector([i, j]) / self.image_res

            darken = 1.0 - self.vignette_strength * \
                max(ti.math.distance(uv, self.vignette_center) -
                    self.vignette_radius, 0.0)

            ldr_color = ti.pow(
                uchimura(self.color_buffer[i, j] * darken * self.exposure / samples), 1.0 / 2.2)
            img.store(ti.Vector([i, j]), ti.Vector([ldr_color.r, ldr_color.g, ldr_color.b, 1.0]))

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    # returns 1: computed integrand after it is shifted from src_reservoir into this reservoir's domain
    # returns 2: Jacobian determinant of this transformation
    @ti.func
    def shift(self, dst_pos, dst_normal, dst_material, \
                    src_pos, src_normal, src_material, \
                    dst_reservoir, src_reservoir):
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
            dst_rc_pdf = self.bsdf.pdf_disney(rc_mat, \
                                             -dir_to_rc_vertex, \
                                              src_reservoir.z.rc_normal, \
                                              src_reservoir.z.rc_incident_dir, \
                                              rc_tang, rc_bitang, \
                                              src_reservoir.z.lobes // 10)
            
            src_rc_pdf = self.bsdf.pdf_disney(rc_mat, \
                                             -src_dir_to_rc_vertex, \
                                              src_reservoir.z.rc_normal, \
                                              src_reservoir.z.rc_incident_dir, \
                                              rc_tang, rc_bitang, \
                                              src_reservoir.z.lobes // 10)

        rc_nee_brdf = vec3(0., 0., 0.)
        if rc_is_NEE_visible:
            rc_nee_brdf = self.bsdf.disney_evaluate_lobewise(rc_mat, \
                                                            -dir_to_rc_vertex, \
                                                             src_reservoir.z.rc_normal, \
                                                             src_reservoir.z.rc_NEE_dir, \
                                                             rc_tang, rc_bitang, \
                                                             src_reservoir.z.lobes // 10)
            rc_nee_brdf *= saturate(src_reservoir.z.rc_normal.dot(src_reservoir.z.rc_NEE_dir))

        # primary dst vertex weights
        dst_tang, dst_bitang = make_orthonormal_basis(dst_normal)
        view = (self.camera_pos[None] - dst_pos).normalized()
        primary_brdf = self.bsdf.disney_evaluate_lobewise(dst_material, \
                                                          view, \
                                                          dst_normal, \
                                                          dir_to_rc_vertex, \
                                                          dst_tang, dst_bitang, \
                                                          src_reservoir.z.lobes % 10) # Evaluating the source reservoir's
                                                                                      # BSDF lobe for primary vertex gives correct 
                                                                                      # results. While using the destination's lobe
                                                                                      # gives incorrect/noisy/bad results.
                                                                                      # TODO: Understand why...
        primary_brdf *= saturate(dst_normal.dot(dir_to_rc_vertex))
        dst_primary_pdf = self.bsdf.pdf_disney(dst_material, \
                                               view, \
                                               dst_normal, \
                                               dir_to_rc_vertex, \
                                               dst_tang, dst_bitang, \
                                               dst_reservoir.z.lobes % 10)
        
        # primary src vertex weights
        src_tang, src_bitang = make_orthonormal_basis(src_normal)
        src_primary_pdf = self.bsdf.pdf_disney(src_material, \
                                               (self.camera_pos[None] - src_pos).normalized(), \
                                               src_normal, \
                                               src_dir_to_rc_vertex, \
                                               src_tang, src_bitang, \
                                               src_reservoir.z.lobes % 10)
        # compute shifted integrand
        contrib = vec3(0., 0., 0.)
        if not rc_is_escape_vertex and not rc_is_last_vertex:
            contrib += rc_brdf / dst_rc_pdf * src_reservoir.z.rc_incident_L # rc bounce
        if rc_is_escape_vertex:
            contrib += src_reservoir.z.rc_incident_L # sky emission
        contrib += rc_nee_brdf * self.light_color[None] * self.light_multiplier # NEE at rc vertex
        contrib += vec3(0., 0., 0.) if rc_mat_id != 2 else rc_mat.base_col # emission at rc vertex

        contrib *= primary_brdf

        # compute jacobian
        jacobian = 1.0 # limit case of escape vertices converges to one

        # jacobian *= dst_rc_pdf / src_rc_pdf
        # jacobian *= dst_primary_pdf / src_primary_pdf

        if not rc_is_escape_vertex:
            jacobian = src_reservoir.z.cached_jacobian_term
            dir_y1_to_x2 = src_reservoir.z.rc_pos - dst_pos
            jacobian *= abs(dir_y1_to_x2.normalized().dot(src_reservoir.z.rc_normal)) / dir_y1_to_x2.dot(dir_y1_to_x2)
            jacobian = clamp(jacobian, 0.0, 8.0)
        
        return contrib, jacobian*passed_checks


    @ti.kernel
    def spatial_GRIS(self, pass_id : ti.i32, max_radius : ti.f32, max_taps : ti.i32, pass_total : ti.i32):

        # Render
        ti.loop_config(block_dim=128)
        for u, v in self.color_buffer:
            
            start_index = ti.cast(ti.random() * max_taps, ti.u32)
            offset_mask = ti.cast(max_taps - 1, ti.u32)

            
            # Generate random offset seeds
            seed_x = ti.cast(u, ti.u32) >> 3 if pass_id == 0 else 2
            seed_y = ti.cast(v, ti.u32) >> 3 if pass_id == 0 else 2
            seed = hash3(seed_x, seed_y, ti.cast(self.current_frame*2, ti.u32) + pass_id)

            angle_shift = ti.cast(((seed & 0x007FFFFF) | 0x3F800000), ti.f32)/4294967295.0 * np.pi
            radius_shift = ti.random()

            # Get center reservoir
            center_reservoir = self.spatial_reservoirs[u, v, pass_id % 2]

            # Initialize output reservoir
            output_reservoir = Reservoir()
            output_reservoir.init()

            # Variables of center pixel
            center_x1 = self.gbuff_position[u, v]
            center_n1 = decodeUnitVector3x16(self.gbuff_normals[u, v])

            if is_vec_zero(center_x1):
                self.color_buffer[u, v] += center_reservoir.z.F
                continue

            center_mat, center_mat_id = decode_material(self.mats.mat_list, self.gbuff_mat_id[u, v])

           
            view = (self.camera_pos[None] - center_x1).normalized()
            tang, bitang = make_orthonormal_basis(center_n1)

            center_dist = distance(self.camera_pos[None], center_x1)

            valid_samples = 0

            for i in range(max_taps):
                # neighbor_index = ti.cast(start_index + i, ti.u32) & offset_mask
                # offset = ti.cast(self.reuse_offsets[neighbor_index] * max_radius, ti.i32)
                golden_angle = 2.399963229728
                
                angle = (i + angle_shift) * golden_angle
                offset_radius = ti.sqrt(float(i + radius_shift) / ti.cast(max_taps, ti.f32)) * max_radius

                offset = ti.cast(ti.Vector([ti.cos(angle) * offset_radius, ti.sin(angle) * offset_radius]), ti.i32)
                # force_add = False
                # if i == 0:
                #     offset = ti.Vector([0, 0])
                #     force_add = True
                if offset.x == 0 and offset.y == 0:
                    continue
                
                tap_coord = ti.Vector([u, v]) + offset

                neighbour_n1 = decodeUnitVector3x16(self.gbuff_normals[tap_coord.x, tap_coord.y])
                neighbour_x1 = self.gbuff_position[tap_coord.x, tap_coord.y]
                neighbour_reservoir = self.spatial_reservoirs[tap_coord.x, tap_coord.y, pass_id % 2]

                neighbour_dist = distance(self.camera_pos[None], neighbour_x1)

                neighbour_primary_lobe_id = neighbour_reservoir.z.lobes % 10

                if abs(neighbour_dist - center_dist) > 0.1*center_dist or center_n1.dot(neighbour_n1) < 0.5:
                    continue

                neighbour_mat, neighbour_mat_id = decode_material(self.mats.mat_list, self.gbuff_mat_id[tap_coord.x, tap_coord.y])

                shifted_integrand, jacobian = self.shift(center_x1   , center_n1   , center_mat, \
                                                    neighbour_x1, neighbour_n1, neighbour_mat, \
                                                    output_reservoir, neighbour_reservoir)
                
                p_hat = luminance(shifted_integrand) * jacobian

                if jacobian < 1e-3: 
                    continue

                neighbour_reservoir.z.F = shifted_integrand
                neighbour_reservoir.z.lobes = neighbour_reservoir.z.lobes // 10 + center_reservoir.z.lobes % 10

                selected = output_reservoir.merge(neighbour_reservoir, neighbour_reservoir.weight * p_hat * neighbour_reservoir.M)
                
                valid_samples += 1

            center_p_hat = luminance(center_reservoir.z.F)
            selected = output_reservoir.merge(center_reservoir, center_reservoir.weight * center_p_hat * center_reservoir.M)

            # mis_weight = p_chosen / (p_sum)
            output_reservoir.finalize()
            # output_reservoir.weight *= mis_weight # biased, constant RMIS weight
            if pass_id == pass_total - 1:
                self.color_buffer[u, v] += output_reservoir.z.F * clamp(output_reservoir.weight, 0.0, 10.0) + self.nee_and_emission_buffer[u, v]

            output_reservoir.update_cached_jacobian_term(center_x1)
            self.spatial_reservoirs[u, v, (pass_id + 1) % 2] = output_reservoir

                
                

    
    def accumulate(self):
        self.render(self.world.voxel_color_texture)
        if ti.static(USE_RESTIR_PT):
            self.spatial_GRIS(0, 16.0, 12, 1)
            # self.spatial_GRIS(1, 16.0, 6, 2)
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
