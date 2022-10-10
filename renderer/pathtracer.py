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

        # Reservoir storage
        # TODO: Make this use the packed reservoirs and encode/decode on read/write
        self.spatial_reservoirs = Reservoir.field(shape=(image_res[0], image_res[1], 2))

        # Auxiliaries
        self.nee_and_emission_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(image_res[0], image_res[1]))

        # Gbuffer
        self.gbuff_mat_id = ti.Vector.field(2, dtype=ti.i32, shape=(image_res[0], image_res[1]))
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
        first_bounce_lobe_id = 0
        first_bounce_brdf = ti.Vector([0.0, 0.0, 0.0])
        first_bounce_invpdf = 1.0
        first_vertex_NEE_and_emission = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        hit_light = 0
        hit_background = 0

        input_sample_reservoir = Reservoir()
        input_sample_reservoir.init()

        return d, pos, contrib, first_bounce_lobe_id, first_bounce_brdf, first_bounce_invpdf, first_vertex_NEE_and_emission, throughput, hit_light, hit_background, input_sample_reservoir

    @ti.kernel
    def render(self, colors: ti.types.texture(3)):

        # Render
        ti.loop_config(block_dim=128)
        for u, v in self.color_buffer:
            (
                d,
                pos,
                contrib,
                first_bounce_lobe_id,
                first_bounce_brdf, # not really used by restir
                first_bounce_invpdf,
                first_vertex_NEE_and_emission,
                throughput,
                hit_light,
                hit_background,
                input_sample_reservoir
            ) = self.generate_new_sample(u, v)

            x1 = vec3(0.0, 0.0, 0.0)

            # Tracing begin
            for depth in range(MAX_RAY_DEPTH):
                closest, normal, albedo, hit_light, iters, mat_id = self.next_hit(
                    pos, d, inf, colors, shadow_ray=False)
                hit_mat = self.mats.mat_list[mat_id]
                hit_pos = pos + closest * d

                # Write to gbuffer and reservoir
                if(depth == 0):
                    self.gbuff_normals[u, v]  = encodeUnitVector3x16(normal)
                    self.gbuff_position[u, v] = hit_pos
                    x1 = hit_pos
                    self.gbuff_mat_id[u, v] = ti.Vector([mat_id, ti.cast(256*256*255*albedo.r + 256*255*albedo.g + 255*albedo.b, ti.i32)])
                elif(depth == 1):
                    input_sample_reservoir.z.rc_pos = hit_pos
                    input_sample_reservoir.z.rc_normal = normal
                elif(depth == 2):
                    input_sample_reservoir.z.rc_incident_dir = hit_pos # dont need normal of fourth vertex for reconnection shift

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
                                light_bsdf = self.bsdf.disney_evaluate(hit_mat, view, normal, light_dir, tang, bitang)
                                NEE_contrib = firefly_filter(throughput * light_bsdf * self.light_color[None] * np.pi * dot)
                                if depth == 0:
                                    first_vertex_NEE_and_emission += NEE_contrib
                                else:
                                    contrib += NEE_contrib
                    
                    # Sample next bounce
                    d, bsdf, pdf, lobe_id = self.bsdf.sample_disney(hit_mat, view, normal, tang, bitang)
                    # d = sample_cosine_weighted_hemisphere(normal)
                    # pdf = saturate(d.dot(normal)) / np.pi
                    # bsdf = self.bsdf.disney_evaluate(hit_mat, view, normal, d, tang, bitang)
                    # lobe_id = 0

                    # Apply weight to throughput (Not on first bounce)
                    if depth == 0:
                        first_bounce_invpdf = 1.0/pdf
                        first_bounce_brdf = bsdf * saturate(d.dot(normal))
                    else:
                        throughput *= bsdf * saturate(d.dot(normal)) / pdf
                    # throughput *= bsdf * saturate(d.dot(normal)) / pdf
                    
                else:
                    if closest == inf:
                        # Hit background
                        sky_emission = firefly_filter(throughput * self.background_color[None])
                        contrib += sky_emission
                        if depth == 0:
                            first_vertex_NEE_and_emission += sky_emission
                    else:
                        # hit light voxel, terminate tracing
                        if depth == 0:
                            first_vertex_NEE_and_emission += albedo
                        else:
                            contrib += throughput * albedo
                    break

                # Russian roulette (with ReSTIR PT, we won't do RR on first bounce)
                max_c = clamp(throughput.max(), 0.0, 1.0)
                if ti.random() > max_c:
                    break
                else:
                    throughput /= max_c

            # Finish populating reservoir fields
            input_sample_reservoir.z.L = contrib
            input_sample_reservoir.M = 1.0
            input_sample_reservoir.update_cached_jacobian_term(x1)

            F = contrib * first_bounce_brdf
            p_hat = vec3(0.33, 0.33, 0.33).dot(F)
            input_sample_reservoir.weight = p_hat * first_bounce_invpdf # MIS weight here is just 1.0

            input_sample_reservoir.finalize(F) # set reservoir w to now hold the initial 
                                               # sample's unbiased contribution weight 
            self.spatial_reservoirs[u, v, 0] = input_sample_reservoir

            # self.color_buffer[u, v] += F * input_sample_reservoir.weight + first_vertex_NEE_and_emission
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

    @ti.kernel
    def spatial_GRIS(self, pass_id : ti.u32, max_radius : ti.f32, max_taps : ti.i32, pass_total : ti.i32):

        # Render
        ti.loop_config(block_dim=128)
        for u, v in self.color_buffer:
            
            # Generate random offset seeds
            seed_x = ti.cast(u, ti.u32) >> 3
            seed_y = ti.cast(v, ti.u32) >> 3
            seed = hash3(seed_x, seed_y, ti.cast(self.current_frame^3, ti.u32) + pass_id)

            angle_shift = ti.random() # ti.cast(((seed & 0x007FFFFF) | 0x3F800000), ti.f32)/4294967295.0 * np.pi
            radius_shift = ti.random()

            # Initialize output reservoir
            output_reservoir = Reservoir()
            output_reservoir.init()

            # Variables of center pixel
            chosen_F = vec3(0.0, 0.0, 0.0)
            center_x1 = self.gbuff_position[u, v]
            center_n1 = decodeUnitVector3x16(self.gbuff_normals[u, v])

            mat_info = self.gbuff_mat_id[u, v]
            disney_mat = self.mats.mat_list[mat_info.x]
            disney_mat.base_col = ti.cast(ti.Vector([mat_info.y/(256*256), (mat_info.y/256) % 256, mat_info.y % 256]), ti.f32) / 255.0

            view = (self.camera_pos[None] - center_x1).normalized()
            tang, bitang = make_orthonormal_basis(center_n1)

            center_dist = distance(self.camera_pos[None], center_x1)

            valid_samples = 0

            for i in range(max_taps):
                golden_angle = 2.399963229728
                
                angle = (i + angle_shift) * golden_angle
                offset_radius = ti.sqrt(float(i + radius_shift) / ti.cast(max_taps, ti.f32)) * max_radius

                offset = ti.Vector([ti.cast(ti.cos(angle) * offset_radius, ti.i32), ti.cast(ti.sin(angle) * offset_radius, ti.i32)])
                force_add = False
                if i == 0:
                    offset = ti.Vector([0, 0])
                    force_add = True
                
                tap_coord = ti.Vector([u, v]) + offset

                neighbour_normal = decodeUnitVector3x16(self.gbuff_normals[tap_coord.x, tap_coord.y])
                neighbour_position = self.gbuff_position[tap_coord.x, tap_coord.y]
                neighbour_reservoir = self.spatial_reservoirs[tap_coord.x, tap_coord.y, pass_id % 2]

                dir_to_rc_vertex = (neighbour_reservoir.z.rc_pos - center_x1)
                neighbour_dist = distance(self.camera_pos[None], neighbour_position)

                # dir_to_rc_vertex.dot(dir_to_rc_vertex) < 0.015*0.015*center_dist*center_dist 
                if i > 0 and (abs(neighbour_dist - center_dist) > 0.1*center_dist):
                    continue

                if i > 0 and (center_n1.dot(dir_to_rc_vertex) < 1e-5 or \
                              neighbour_reservoir.z.rc_normal.dot(-dir_to_rc_vertex) < 1e-5 or \
                              dir_to_rc_vertex.dot(dir_to_rc_vertex) < 0.015*0.015*center_dist*center_dist or \
                              center_n1.dot(neighbour_normal) < 0.5): 
                    continue

                dir_to_rc_vertex = dir_to_rc_vertex.normalized()


                shifted_integrand = neighbour_reservoir.z.L
                brdf = self.bsdf.disney_evaluate(disney_mat, \
                                                               view, center_n1, dir_to_rc_vertex, tang, bitang)
                brdf *= saturate(dir_to_rc_vertex.dot(center_n1))
                shifted_integrand *= brdf
            

                p_hat = vec3(0.33, 0.33, 0.33).dot(shifted_integrand)

                jacobian = 1.0 # TODO: actually do jacobian

                if i > 0:
                    jacobian = neighbour_reservoir.z.cached_jacobian_term
                    dir_y1_to_x2 = neighbour_reservoir.z.rc_pos - center_x1
                    jacobian *= abs(dir_y1_to_x2.normalized().dot(neighbour_reservoir.z.rc_normal)) / dir_y1_to_x2.dot(dir_y1_to_x2)
                    jacobian = clamp(jacobian, 0.0, 8.0)


                selected = output_reservoir.merge(neighbour_reservoir, neighbour_reservoir.weight * jacobian * p_hat, force_add)

                if(selected):
                    chosen_F = shifted_integrand
                
                valid_samples += 1

            mis_weight = 1.0 / valid_samples
            output_reservoir.finalize(chosen_F)
            output_reservoir.weight *= mis_weight # biased, constant RMIS weight
            if pass_id == pass_total - 1:
                self.color_buffer[u, v] += chosen_F * clamp(output_reservoir.weight, 0.0, 100.0) + self.nee_and_emission_buffer[u, v]

            output_reservoir.update_cached_jacobian_term(center_x1)
            self.spatial_reservoirs[u, v, (pass_id + 1) % 2] = output_reservoir

                
                

    
    def accumulate(self):
        self.render(self.world.voxel_color_texture)
        self.spatial_GRIS(0, 16.0, 6, 2)
        self.spatial_GRIS(1, 16.0, 6, 2)
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
