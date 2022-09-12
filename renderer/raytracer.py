from turtle import distance
import taichi as ti
import math
from renderer.math_utils import eps, inf, ray_aabb_intersection


@ti.data_oriented
class VoxelOctreeRaytracer:
    def __init__(self, voxel_grid_res) -> None:
        self.voxel_grid_res = voxel_grid_res
        self.n_lods = int(math.log2(self.voxel_grid_res))
        lod_map_size = 0
        for i in range(self.n_lods):
            lod_res = self.voxel_grid_res >> i
            lod_map_size += lod_res * lod_res * lod_res
        self.occupancy = ti.field(
            dtype=ti.i32, shape=((lod_map_size // 32) + 1))

    @ti.func
    def linearize_index(self, ipos, lod):
        base_idx = 0
        if lod > 0:
            n = ti.static(self.voxel_grid_res ** 3)
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
        # We are not memory bound here, no need to use morton code
        base_idx += (
            ipos.z * (lod_res * lod_res) + ipos.y * lod_res + ipos.x
        )
        return base_idx

    @ti.func
    def query_occupancy(self, ipos, lod):
        ret = 0
        idx = self.linearize_index(ipos, lod)
        ret = self.occupancy[idx >> 5] & (1 << (idx & 31))
        return ret != 0

    @ti.kernel
    def _update_lods(self, voxels: ti.template(), offset: ti.types.vector(3, ti.i32)):
        # Generate LOD 0
        for i, j, k in ti.ndrange(self.voxel_grid_res, self.voxel_grid_res, self.voxel_grid_res):
            if voxels[ti.Vector([i, j, k]) + offset] > 0:
                idx = self.linearize_index(ti.Vector([i, j, k]), 0)
                bit = 1 << (idx & 31)
                ti.atomic_or(self.occupancy[idx >> 5], bit)
        # Generate LOD 1~N
        for lod in ti.static(range(1, self.n_lods)):
            size_lod = self.voxel_grid_res >> lod
            for i, j, k in ti.ndrange(size_lod, size_lod, size_lod):
                empty = True
                for subi, subj, subk in ti.static(ti.ndrange(2, 2, 2)):
                    empty = empty and (
                        not self.query_occupancy(
                            ti.Vector(
                                [i * 2 + subi, j * 2 + subj, k * 2 + subk]),
                            lod - 1,
                        )
                    )
                if not empty:
                    idx = self.linearize_index(ti.Vector([i, j, k]), lod)
                    bit = 1 << (idx & 31)
                    ti.atomic_or(self.occupancy[idx >> 5], bit)

    @ti.func
    def raytrace(self, origin, direction, ray_min_t, ray_max_t):
        # We assume ray_min_t and ray_max_t are within the voxel volume
        iters = 0
        current_lod = 0

        hit_distance = max(0.0, ray_min_t + eps * 5.0)
        voxel_pos = origin + direction * hit_distance
        ipos_lod0 = ti.cast(ti.floor(voxel_pos), ti.i32)
        inv_dir = 1.0 / ti.abs(direction)

        while iters < 512:
            if hit_distance > ray_max_t:
                break

            sample = self.query_occupancy(ipos_lod0 >> current_lod, current_lod)
            while sample and current_lod > 0:
                # If we hit something, traverse down the LODs
                # Until we reach LOD 0 or reach a empty cell
                current_lod = current_lod - 1
                sample = self.query_occupancy(ipos_lod0 >> current_lod, current_lod)

            if sample:
                # If hit, exit loop
                break
            
            # Find parametric edge distances (time to hit)
            cell_size = 1 << current_lod
            dist = voxel_pos - ti.cast(ipos_lod0 & (-cell_size), ti.f32)
            for i in ti.static(range(3)):
                if direction[i] > 0.0:
                    dist[i] = cell_size - dist[i]
            t = (dist + eps) * inv_dir
            
            # If empty, advance to the nearest edge & increment lod
            min_t = ti.min(t.x, t.y, t.z)
            hit_distance += min_t
            voxel_pos = origin + direction * hit_distance
            ipos_lod0 = ti.cast(ti.floor(voxel_pos), ti.i32)
            current_lod = min(max(0, self.n_lods - 2), current_lod + 1)

            iters += 1

        return hit_distance, voxel_pos, ipos_lod0, iters
