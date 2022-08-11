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
        iters = 0

        hit_distance = inf
        voxel_index = ti.Vector([0, 0, 0])

        current_lod = 0

        voxel_pos = origin + direction * max(0, ray_min_t + 5.0 * eps)

        while iters < 512:
            ipos_lod0 = ti.cast(ti.floor(voxel_pos), ti.i32)
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
                origin - cell_base,
                direction,
            )

            if near > ray_max_t:
                break

            if sample and it:
                # If at LOD = 0, we get a voxel hit
                hit_distance = near
                voxel_index = ipos
                break
            else:
                # Move beyond the hit boundary
                voxel_pos = origin + direction * (far + 10.0 * eps)
                # No point going over the top lods
                current_lod = min(max(0, self.n_lods - 2), current_lod + 1)

            iters += 1

        return hit_distance, voxel_pos, voxel_index, iters
