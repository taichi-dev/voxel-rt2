import math
import taichi as ti

from renderer.voxel_world import VoxelWorld
from renderer.raytracer import VoxelOctreeRaytracer
from renderer.math_utils import *

MAX_RAY_DEPTH = 4
use_directional_light = True

DIS_LIMIT = 100


@ti.data_oriented
class Renderer:
    def __init__(self, dx, image_res, up, voxel_edges, exposure=3):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0

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
        self.set_fov(0.23)

        self.floor_height[None] = 0
        self.floor_color[None] = (1, 1, 1)

    def set_directional_light(self, direction, light_cone_angle, light_color):
        direction_norm = (
            direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2
        ) ** 0.5
        self.light_direction[None] = (
            direction[0] / direction_norm,
            direction[1] / direction_norm,
            direction[2] / direction_norm,
        )
        # Theta is half-angle of the light cone
        self.light_cone_cos_theta_max[None] = math.cos(light_cone_angle * 0.5)
        self.light_color[None] = light_color

    @ti.func
    def _sdf_march(self, p, d):
        dist = inf
        if d[1] < -eps:
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
        normal: ti.template(), color: ti.template(), is_light: ti.template()
    ):
        # Ray marching
        ray_march_dist = self._sdf_march(pos, dir)
        if ray_march_dist < DIS_LIMIT and ray_march_dist < closest_hit_dist:
            closest_hit_dist = ray_march_dist
            hit_pos = pos + dir * closest_hit_dist
            normal = self._sdf_normal(hit_pos)
            color = self._sdf_color(hit_pos)
            is_light = 0

    @ti.func
    def _trace_voxel(
        self, eye_pos, d, colors: ti.template(),
        closest_hit_dist: ti.template(), normal: ti.template(), color: ti.template(), is_light: ti.template(),
        shadow_ray: ti.template()
    ):
        # Return data for the voxel hit
        iters = 0
        voxel_index = ti.math.ivec3(0)

        # Re-scale the ray origin from world space to voxel grid space [0, voxel_grid_res)
        bbox_min = self.world.voxel_inv_size * \
            self.world.bbox[0] - self.world.voxel_grid_offset
        bbox_max = self.world.voxel_inv_size * \
            self.world.bbox[1] - self.world.voxel_grid_offset
        eye_pos_scaled = self.world.voxel_inv_size * \
            eye_pos - self.world.voxel_grid_offset
        inter, scene_near, scene_far = ray_aabb_intersection(
            bbox_min, bbox_max, eye_pos_scaled, d
        )

        # If the ray hits the bounding box, trace the voxel grid
        if inter:
            # Setting the near to voxel grid AABB intersection distance,
            # so that we don't waste time outside of the voxel grid
            hit_distance, voxel_pos, voxel_index, iters = self.voxel_raytracer.raytrace(
                eye_pos_scaled, d, scene_near, scene_far)

            # If the ray hits a voxel, get the surface data
            if hit_distance < scene_far:
                # Re-scale from the voxel grid space back to world space
                closest_hit_dist = hit_distance * self.world.voxel_size
                if ti.static(not shadow_ray):
                    voxel_index += self.world.voxel_grid_offset
                    voxel_uv = ti.math.fract(voxel_pos)
                    # Get surface data
                    color, is_light = self.world.voxel_surface_color(
                        voxel_index, voxel_uv, colors)
                    # Get the distance from hit point to the surface (for computing normals)
                    surface_normal = ti.Vector([0.0, 0.0, 0.0])
                    dis = min(voxel_uv, 1.0 - voxel_uv)
                    if dis[0] <= dis[1] and dis[0] < dis[2]:
                        surface_normal[0] = -ti.math.sign(d[0])
                    elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                        surface_normal[1] = -ti.math.sign(d[1])
                    else:
                        surface_normal[2] = -ti.math.sign(d[2])
                    normal = surface_normal

        return voxel_index, iters

    @ti.func
    def next_hit(self, pos, d, max_dist, colors: ti.template(), shadow_ray: ti.template()):
        # Hit Data
        closest_hit_dist = max_dist
        normal = ti.Vector([0.0, 0.0, 0.0])
        albedo = ti.Vector([0.0, 0.0, 0.0])
        hit_light = 0

        # First intersect with voxel grid
        vx_idx, iters = self._trace_voxel(
            pos, d, colors, closest_hit_dist, normal, albedo, hit_light, shadow_ray)

        # Then intersect with implicit SDF
        self._trace_sdf(pos, d, closest_hit_dist, normal, albedo, hit_light)

        # Highlight the selected voxel
        if ti.static(not shadow_ray):
            if self.cast_voxel_hit[None]:
                cast_vx_idx = self.cast_voxel_index[None]
                if all(cast_vx_idx == vx_idx):
                    albedo = ti.Vector([1.0, 0.65, 0.0])
                    # For light sources, we actually invert the material to make it
                    # more obvious
                    hit_light = 1 - hit_light
        return closest_hit_dist, normal, albedo, hit_light, iters

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
        fov = self.fov[None]
        d = (self.look_at[None] - self.camera_pos[None]).normalized()
        fu = (
            2 * fov * (u + ti.random(ti.f32)) / self.image_res[1]
            - fov * self.aspect_ratio
            - 1e-5
        )
        fv = 2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5
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
        c = ti.Vector([1.0, 1.0, 1.0])

        depth = 0
        hit_light = 0
        hit_background = 0

        return d, pos, contrib, throughput, c, depth, hit_light, hit_background

    @ti.kernel
    def render(self, n_spp: ti.i32, colors: ti.types.texture(3)):
        # Render
        ti.loop_config(block_dim=128)
        for u, v in self.color_buffer:
            spp_completed = 0

            (
                d,
                pos,
                contrib,
                throughput,
                c,
                depth,
                hit_light,
                hit_background,
            ) = self.generate_new_sample(u, v)

            # Tracing begin
            while spp_completed < n_spp:
                sample_complete = False

                depth += 1
                closest, normal, c, hit_light, iters = self.next_hit(
                    pos, d, inf, colors, shadow_ray=False)
                hit_pos = pos + closest * d
                # Enable this to debug iteration counts
                if ti.static(False):
                    if depth == 1:
                        worst_case_iters = ti.simt.subgroup.reduce_max(iters)
                        best_case_iters = ti.simt.subgroup.reduce_min(iters)
                        self.color_buffer[u, v] += ti.Vector(
                            [worst_case_iters / 64.0, best_case_iters / 64.0, 0.0])
                if not hit_light and normal.norm() != 0 and closest < 1e8:
                    d = sample_cosine_weighted_hemisphere(normal)
                    pos = hit_pos + normal * eps
                    throughput *= c

                    if ti.static(use_directional_light):
                        light_dir = sample_cone_oriented(
                            self.light_cone_cos_theta_max[None], self.light_direction[None])
                        dot = light_dir.dot(normal)
                        if dot > 0:
                            hit_light_ = 0
                            iters_ = 0
                            dist, _, _, hit_light_, iters_ = self.next_hit(
                                pos, light_dir, inf, colors, shadow_ray=True
                            )
                            if dist > DIS_LIMIT:
                                # far enough to hit directional light
                                contrib += throughput * \
                                    self.light_color[None] * dot
                else:  # hit background or light voxel, terminate tracing
                    hit_background = 1
                    sample_complete = True

                # Russian roulette
                max_c = throughput.max()
                if ti.random() > max_c:
                    throughput = [0, 0, 0]
                    sample_complete = True
                else:
                    throughput /= max_c

                if depth >= MAX_RAY_DEPTH:
                    sample_complete = True

                # Tracing end
                if sample_complete:
                    if hit_light:
                        contrib += throughput * c
                    else:
                        if depth == 1 and hit_background:
                            # Direct hit to background
                            contrib = self.background_color[None]
                    self.color_buffer[u, v] += contrib
                    spp_completed += 1
                    (
                        d,
                        pos,
                        contrib,
                        throughput,
                        c,
                        depth,
                        hit_light,
                        hit_background,
                    ) = self.generate_new_sample(u, v)

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

    def accumulate(self, n_spp):
        self.render(n_spp, self.world.voxel_color_texture)
        self.current_spp += n_spp

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
