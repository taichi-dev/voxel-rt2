import math
import taichi as ti
import numpy as np

from math_utils import (eps, inf, out_dir, vec3, saturate, ray_aabb_intersection)
import bsdf

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
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.voxel_color = ti.Vector.field(3, dtype=ti.u8)
        self.voxel_material = ti.field(dtype=ti.i8)

        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_direction_noise = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.cast_voxel_hit = ti.field(ti.i32, shape=())
        self.cast_voxel_index = ti.Vector.field(3, ti.i32, shape=())

        self.voxel_edges = voxel_edges
        self.exposure = exposure

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.floor_height = ti.field(dtype=ti.f32, shape=())
        self.floor_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.background_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.voxel_dx = dx
        self.voxel_inv_dx = 1 / dx
        # Note that voxel_inv_dx == voxel_grid_res iff the box has width = 1
        self.voxel_grid_res = 128
        voxel_grid_offset = [-self.voxel_grid_res // 2 for _ in range(3)]

        ti.root.dense(ti.ij, image_res).place(self.color_buffer)
        ti.root.dense(ti.ijk, self.voxel_grid_res).place(
            self.voxel_color, self.voxel_material, offset=voxel_grid_offset
        )

        shape = (self.voxel_grid_res, self.voxel_grid_res, self.voxel_grid_res)
        self.voxel_color_texture = ti.Texture(ti.u8, 4, shape)

        self.n_lods = int(math.log2(self.voxel_grid_res))
        lod_map_size = 0
        for i in range(self.n_lods):
            lod_res = self.voxel_grid_res >> i
            lod_map_size += lod_res * lod_res * lod_res
        self.occupancy = ti.field(dtype=ti.i32, shape=((lod_map_size // 32) + 1))

        self._rendered_image = ti.Vector.field(3, float, image_res)
        self.set_up(*up)
        self.set_fov(0.23)

        self.floor_height[None] = 0
        self.floor_color[None] = (1, 1, 1)

    def set_directional_light(self, direction, light_direction_noise, light_color):
        direction_norm = (
            direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2
        ) ** 0.5
        self.light_direction[None] = (
            direction[0] / direction_norm,
            direction[1] / direction_norm,
            direction[2] / direction_norm,
        )
        self.light_direction_noise[None] = light_direction_noise
        self.light_color[None] = light_color

    @ti.func
    def inside_grid(self, ipos):
        return (
            ipos.min() >= -self.voxel_grid_res // 2
            and ipos.max() < self.voxel_grid_res // 2
        )

    @ti.func
    def linearize_index(self, ipos, lod):
        base_idx = 0
        if lod > 0:
            n = self.voxel_grid_res
            n = n * n * n
            #    Given the result must be integer
            #    And b >= 1, then
            #    n * sum(0..b, (0.5 ** b))
            # => n * (1 - 0.5 ** b) / 0.5
            # => (2.0 * n) * (1.0 - 0.5 ** (b+1))
            # => 2.0 * n - 2.0 * n * (0.5 ** (b+1))
            # => 2.0 * n - n / (2 ** b)
            # => (n << 1) - (n >> b)
            base_idx = (n << 1) - (n >> (lod - 1))
        lod_res = self.voxel_grid_res >> lod
        voxel_pos = ipos + (lod_res >> 1)
        # We are not memory bound here, no need to use morton code
        base_idx += (
            voxel_pos.z * (lod_res * lod_res) + voxel_pos.y * lod_res + voxel_pos.x
        )
        return base_idx

    @ti.func
    def query_occupancy(self, ipos, lod):
        ret = 0
        idx = self.linearize_index(ipos, lod)
        ret = self.occupancy[idx >> 5] & (1 << (idx & 31))
        return ret != 0

    @ti.func
    def query_density(self, ipos):
        inside = self.inside_grid(ipos)
        ret = 0
        if inside:
            ret = self.voxel_material[ipos]
        else:
            ret = 0
        return ret

    @ti.func
    def _to_voxel_index(self, pos):
        p = pos * self.voxel_inv_dx
        voxel_index = ti.floor(p).cast(ti.i32)
        return voxel_index

    @ti.func
    def voxel_surface_color(self, pos, colors: ti.template()):
        p = pos * self.voxel_inv_dx
        p -= ti.floor(p)
        voxel_index = self._to_voxel_index(pos)

        boundary = self.voxel_edges
        count = 0
        for i in ti.static(range(3)):
            if p[i] < boundary or p[i] > 1 - boundary:
                count += 1

        f = 0.0
        if count >= 2:
            f = 1.0

        voxel_color = ti.Vector([0.0, 0.0, 0.0])
        is_light = 0
        if self.inside_particle_grid(voxel_index):
            half_res = (
                ti.Vector(
                    [self.voxel_grid_res, self.voxel_grid_res, self.voxel_grid_res]
                )
                >> 1
            )
            data = colors.fetch((voxel_index + half_res).zyx, 0)
            voxel_color = data.rgb
            voxel_material = ti.cast(data.a * 255.0, ti.i32)
            if voxel_material == 2:
                is_light = 1

        return voxel_color, is_light #  * (1.3 - 1.2 * f)

    @ti.func
    def ray_march(self, p, d):
        dist = inf
        if d[1] < -eps:
            dist = (self.floor_height[None] - p[1]) / d[1]
        return dist

    @ti.func
    def sdf_normal(self, p):
        return ti.Vector([0.0, 1.0, 0.0])  # up

    @ti.func
    def sdf_color(self, p):
        return self.floor_color[None]

    @ti.func
    def dda_voxel(self, eye_pos, d, colors: ti.template()):
        for i in ti.static(range(3)):
            if abs(d[i]) < 1e-6:
                d[i] = 1e-6
        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        iters = 0

        bbox_min = self.voxel_inv_dx * self.bbox[0]
        bbox_max = self.voxel_inv_dx * self.bbox[1]
        eye_pos_scaled = self.voxel_inv_dx * eye_pos
        inter, scene_near, scene_far = ray_aabb_intersection(
            bbox_min, bbox_max, eye_pos_scaled, d
        )
        hit_distance = inf
        hit_light = 0
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        voxel_index = ti.Vector([0, 0, 0])
        if inter:
            current_lod = 0

            voxel_pos = eye_pos_scaled + d * max(0, scene_near + 5.0 * eps)
            ipos_lod0 = ti.cast(ti.floor(voxel_pos), ti.i32)

            while iters < 512:
                ipos = ipos_lod0 >> current_lod
                sample = self.query_occupancy(ipos, current_lod)
                while sample and current_lod > 0:
                    # If we hit something, traverse down the LODs
                    # Until we reach LOD 0 or reach a empty cell
                    current_lod = current_lod - 1
                    ipos = ipos_lod0 >> current_lod
                    sample = self.query_occupancy(ipos, current_lod)

                lod_size = ti.cast(1 << current_lod, ti.f32)
                cell_base = ti.cast(ipos << current_lod, ti.f32)
                it, near, far = ray_aabb_intersection(
                    ti.math.vec3(0.0),
                    ti.math.vec3(lod_size),
                    eye_pos_scaled - cell_base,
                    d,
                )

                if near > scene_far:
                    break

                if sample and it:
                    # If at LOD = 0, we get a voxel hit
                    hit_distance = self.voxel_dx * near
                    voxel_index = ipos
                    break
                else:
                    # Move beyond the hit boundary
                    voxel_pos = eye_pos_scaled + d * (far + 10.0 * eps)
                    # Snap to cloest grid boundary
                    ipos_lod0 = ti.cast(ti.floor(voxel_pos), ti.i32)
                    # No point going over the top lods
                    current_lod = min(max(0, self.n_lods - 2), current_lod + 1)

                iters += 1

            if hit_distance < inf:
                pos = self.voxel_dx * voxel_pos
                c, hit_light = self.voxel_surface_color(pos, colors)
                dis = ti.math.fract(voxel_pos)
                dis = min(dis, 1.0 - dis)
                mm = ti.Vector([0, 0, 0])
                if dis[0] <= dis[1] and dis[0] < dis[2]:
                    mm[0] = 1
                elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                    mm[1] = 1
                else:
                    mm[2] = 1
                normal = -mm * rsign

        return hit_distance, normal, c, hit_light, voxel_index, iters

    @ti.func
    def inside_particle_grid(self, ipos):
        pos = ipos * self.voxel_dx
        return (
            self.bbox[0][0] <= pos[0]
            and pos[0] < self.bbox[1][0]
            and self.bbox[0][1] <= pos[1]
            and pos[1] < self.bbox[1][1]
            and self.bbox[0][2] <= pos[2]
            and pos[2] < self.bbox[1][2]
        )

    @ti.func
    def next_hit(self, pos, d, t, colors: ti.template()):
        closest = inf
        normal = ti.Vector([0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0])
        hit_light = 0
        closest, normal, c, hit_light, vx_idx, iters = self.dda_voxel(pos, d, colors)

        ray_march_dist = self.ray_march(pos, d)
        if ray_march_dist < DIS_LIMIT and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.sdf_normal(pos + d * closest)
            c = self.sdf_color(pos + d * closest)

        # Highlight the selected voxel
        if self.cast_voxel_hit[None]:
            cast_vx_idx = self.cast_voxel_index[None]
            if all(cast_vx_idx == vx_idx):
                c = ti.Vector([1.0, 0.65, 0.0])
                # For light sources, we actually invert the material to make it
                # more obvious
                hit_light = 1 - hit_light
        return closest, normal, c, hit_light, iters

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

    @ti.kernel
    def _make_texture(
        self,
        colors: ti.types.rw_texture(
            num_dimensions=3, num_channels=4, channel_format=ti.u8, lod=0
        ),
    ):
        for ijk in ti.grouped(self.voxel_color):
            color = ti.cast(self.voxel_color[ijk], ti.f32) / 255.0
            material = ti.cast(self.voxel_material[ijk], ti.f32) / 255.0
            half_res = (
                ti.Vector(
                    [self.voxel_grid_res, self.voxel_grid_res, self.voxel_grid_res]
                )
                >> 1
            )
            colors.store(
                ijk.zyx + half_res, ti.Vector([color[0], color[1], color[2], material])
            )

    @ti.kernel
    def _update_lods(self):
        # Generate LOD 0
        half_size = self.voxel_grid_res >> 1
        for i, j, k in ti.ndrange(
            (-half_size, half_size), (-half_size, half_size), (-half_size, half_size)
        ):
            if self.query_density(ti.Vector([i, j, k])) > 0:
                idx = self.linearize_index(ti.Vector([i, j, k]), 0)
                bit = 1 << (idx & 31)
                ti.atomic_or(self.occupancy[idx >> 5], bit)
        # Generate LOD 1~N
        for ll in ti.static(range(self.n_lods - 1)):
            lod = ll + 1
            half_size = self.voxel_grid_res >> (lod + 1)
            for i, j, k in ti.ndrange(
                (-half_size, half_size),
                (-half_size, half_size),
                (-half_size, half_size),
            ):
                empty = True
                for subi, subj, subk in ti.static(ti.ndrange(2, 2, 2)):
                    empty = empty and (
                        not self.query_occupancy(
                            ti.Vector([i * 2 + subi, j * 2 + subj, k * 2 + subk]),
                            lod - 1,
                        )
                    )
                if not empty:
                    idx = self.linearize_index(ti.Vector([i, j, k]), lod)
                    bit = 1 << (idx & 31)
                    ti.atomic_or(self.occupancy[idx >> 5], bit)

    def prepare_data(self):
        shape = (self.voxel_grid_res, self.voxel_grid_res, self.voxel_grid_res)
        self._make_texture(self.voxel_color_texture)
        self._update_lods()

    @ti.func
    def generate_new_sample(self, u: ti.f32, v: ti.f32):
        d = self.get_cast_dir(u, v)
        pos = self.camera_pos[None]
        t = 0.0

        contrib = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])
        c = ti.Vector([1.0, 1.0, 1.0])

        depth = 0
        hit_light = 0
        hit_background = 0

        return d, pos, t, contrib, throughput, c, depth, hit_light, hit_background

    @ti.kernel
    def render(self, n_spp: ti.i32, colors: ti.types.texture(3)):

        hardcoded_mat = bsdf.disney_material_params(base_col=vec3([1.0,1.0,1.0]) \
                                        ,subsurface=0.0 \
                                        ,metallic=0.2 \
                                        ,specular=0.5 \
                                        ,specular_tint=0.0 \
                                        ,roughness=0.4 \
                                        ,anisotropic=0.0 \
                                        ,sheen=0.0 \
                                        ,sheen_tint=0.0 \
                                        ,clearcoat=0.8 \
                                        ,clearcoat_gloss=0.99)

        # Render
        ti.loop_config(block_dim=64)
        for u, v in self.color_buffer:
            spp_completed = 0

            (
                d,
                pos,
                t,
                contrib,
                throughput,
                albedo,
                depth,
                hit_light,
                hit_background,
            ) = self.generate_new_sample(u, v)

            # Tracing begin
            while spp_completed < n_spp:
                sample_complete = False

                depth += 1
                closest, normal, albedo, hit_light, iters = self.next_hit(pos, d, t, colors)
                hit_pos = pos + closest * d
                # if depth == 1:
                #     worst_case_iters = ti.simt.subgroup.reduce_max(iters)
                #     best_case_iters = ti.simt.subgroup.reduce_min(iters)
                #     self.color_buffer[u, v] += ti.Vector([worst_case_iters / 64.0, best_case_iters / 64.0, 0.0])
                if not hit_light and normal.norm() != 0 and closest < 1e8:
                    pos = hit_pos + normal * eps
                    hardcoded_mat.base_col = albedo
                    view = -d

                    tang = normal.cross(vec3([0.0, 1.0, 1.0])).normalized()
                    bitang = tang.cross(normal)

                    d, brdf, pdf = bsdf.sample_disney(hardcoded_mat, view, normal, tang, bitang)

                    # d = out_dir(normal)

                    # brdf = bsdf.disney_evaluate(hardcoded_mat, view, normal, d, tang, bitang)
                    # pdf = saturate(d.dot(normal)) / np.pi
                    

                    if ti.static(use_directional_light):
                        dir_noise = (
                            ti.Vector(
                                [
                                    ti.random() - 0.5,
                                    ti.random() - 0.5,
                                    ti.random() - 0.5,
                                ]
                            )
                            * self.light_direction_noise[None]
                        )
                        light_dir = (
                            self.light_direction[None] + dir_noise
                        ).normalized()
                        dot = light_dir.dot(normal)
                        if dot > 0:
                            hit_light_ = 0
                            dist, _, _, hit_light_, iters = self.next_hit(
                                pos, light_dir, t, colors
                            )
                            if dist > DIS_LIMIT:
                                # far enough to hit directional light
                                light_brdf = bsdf.disney_evaluate(hardcoded_mat, view, normal, light_dir, tang, bitang)
                                contrib += throughput * light_brdf * self.light_color[None] * np.pi * dot

                    # Apply mask to throughput
                    throughput *= ti.math.clamp(brdf * saturate(d.dot(normal)) / pdf, 0.0, 999999.0)

                else:  # hit background or light voxel, terminate tracing
                    hit_background = 1
                    contrib += throughput * self.background_color[None]
                    sample_complete = True

                # Russian roulette
                # max_c = throughput.max()
                # if ti.random() > max_c:
                #     throughput = [0, 0, 0]
                #     sample_complete = True
                # else:
                #     throughput /= max_c

                if depth >= MAX_RAY_DEPTH:
                    sample_complete = True

                # Tracing end
                if sample_complete:
                    if hit_light:
                        contrib += throughput * albedo
                    else:
                        if depth == 1 and hit_background:
                            # Direct hit to background
                            contrib = self.background_color[None]
                    self.color_buffer[u, v] += contrib
                    spp_completed += 1
                    (
                        d,
                        pos,
                        t,
                        contrib,
                        throughput,
                        albedo,
                        depth,
                        hit_light,
                        hit_background,
                    ) = self.generate_new_sample(u, v)

    @ti.kernel
    def _render_to_image(self, samples: ti.i32):
        for i, j in self.color_buffer:
            u = 1.0 * i / self.image_res[0]
            v = 1.0 * j / self.image_res[1]

            darken = 1.0 - self.vignette_strength * max(
                (
                    ti.sqrt(
                        (u - self.vignette_center[0]) ** 2
                        + (v - self.vignette_center[1]) ** 2
                    )
                    - self.vignette_radius
                ),
                0,
            )

            for c in ti.static(range(3)):
                self._rendered_image[i, j][c] = ti.sqrt(
                    self.color_buffer[i, j][c] * self.exposure / samples # * darken
                )

    @ti.kernel
    def recompute_bbox(self):
        for d in ti.static(range(3)):
            self.bbox[0][d] = 1e9
            self.bbox[1][d] = -1e9
        for I in ti.grouped(self.voxel_material):
            if self.voxel_material[I] != 0:
                for d in ti.static(range(3)):
                    ti.atomic_min(self.bbox[0][d], (I[d] - 1) * self.voxel_dx)
                    ti.atomic_max(self.bbox[1][d], (I[d] + 2) * self.voxel_dx)

    def reset_framebuffer(self):
        self.current_spp = 0
        self.color_buffer.fill(0)

    def accumulate(self, n_spp):
        self.render(n_spp, self.voxel_color_texture)
        self.current_spp += n_spp

    def fetch_image(self):
        self._render_to_image(self.current_spp)
        return self._rendered_image

    @staticmethod
    @ti.func
    def to_vec3u(c):
        c = ti.math.clamp(c, 0.0, 1.0)
        r = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i] * 255, ti.u8)
        return r

    @staticmethod
    @ti.func
    def to_vec3(c):
        r = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i], ti.f32) / 255.0
        return r

    @ti.func
    def set_voxel(self, idx, mat, color):
        self.voxel_material[idx] = ti.cast(mat, ti.i8)
        self.voxel_color[idx] = self.to_vec3u(color)

    @ti.func
    def get_voxel(self, ijk):
        mat = self.voxel_material[ijk]
        color = self.voxel_color[ijk]
        return mat, self.to_vec3(color)
