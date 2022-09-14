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
            # base_idx = (n << 1) - (n >> (lod - 1))
            #          = (n << 1) - ((n << 1) >> lod)
            base_idx = ti.static(n << 1) - (ti.static(n << 1) >> lod)
        lod_res = self.voxel_grid_res >> lod
        # We are not memory bound here, no need to use morton code
        base_idx += (
            ipos.z * (lod_res * lod_res) + ipos.y * lod_res + ipos.x
        )
        return base_idx

    @ti.func
    def query_occupancy(self, ipos, lod):
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
        # Initialize result to be empty
        hit_distance = inf
        ipos_lod0 = ti.Vector([-1, -1, -1])
        hit_normal = ti.Vector([0, 0, 0])
        iters = 0

        # Check the bounding box of the entire volume
        bbox_intersect, bbox_near, bbox_far = ray_aabb_intersection(
            ti.math.vec3(0.0, 0.0, 0.0),
            ti.math.vec3(self.voxel_grid_res, self.voxel_grid_res, self.voxel_grid_res),
            origin, direction)

        if bbox_intersect and ray_min_t < bbox_far and ray_max_t > bbox_near:
            # Move the ray onto / into the bbox
            hit_distance = max(bbox_near, ray_min_t)

            # We are on or inside the bbox
            # Prepare DDA data
            initial_p = origin + direction * (hit_distance + eps)
            ipos_lod0 = ti.math.clamp(ti.cast(ti.floor(initial_p), ti.i32), 0, self.voxel_grid_res - 1)
            inv_dir = 1.0 / ti.abs(direction)
            current_lod = 0
            far = min(ray_max_t, bbox_far) - eps

            # Compute initial normal in case we hit boundry voxels
            initial_dist = ti.abs(initial_p - self.voxel_grid_res * 0.5)
            max_dist = ti.max(initial_dist.x, initial_dist.y, initial_dist.z)
            hit_normal = ti.cast(max_dist == initial_dist, ti.i32)
            
            while iters < 512:
                if hit_distance > far:
                    hit_distance = inf
                    break

                ipos = ti.Vector([0, 0, 0])
                sample = 0
                while True:
                    # If we hit something, traverse down the LODs
                    # Until we reach LOD 0 or reach a empty cell
                    ipos = ipos_lod0 >> current_lod
                    sample = self.query_occupancy(ipos, current_lod)
                    if sample and current_lod > 0:
                        current_lod -= 1
                    else:
                        break

                if sample:
                    # If hit, exit loop
                    break
                
                # Find parametric edge distances (time to hit)
                cell_size = 1 << current_lod
                cell_base = ipos * cell_size
                voxel_pos = origin + direction * hit_distance
                frac_pos = voxel_pos - ti.cast(cell_base, ti.f32)
                dist = frac_pos
                for i in ti.static(range(3)):
                    if direction[i] > 0.0:
                        dist[i] = cell_size - dist[i]
                t = dist * inv_dir
                
                # If empty, find intersection point with the nearest plane of current cell
                min_t = ti.min(t.x, t.y, t.z)
                # Compute the integer hit point within current cell
                edge_frac_pos = ti.math.clamp(
                    ti.cast(ti.floor(frac_pos + min_t * direction), ti.i32),
                    0, cell_size - 1)
                # Advance floating point values
                hit_distance += min_t
                # The hit normal is the normal of the nearest plane
                hit_normal = ti.cast(t == min_t, ti.i32) * ti.cast(ti.math.sign(direction), ti.i32)
                # Next cell would be `hit_point + hit_normal`. All integer thus water-tight
                ipos_lod0 = cell_base + edge_frac_pos + hit_normal
                current_lod = current_lod + 1

                iters += 1

        # Flip normals if it's backwards
        if direction.dot(hit_normal) > 0:
            hit_normal = -hit_normal

        return hit_distance, ipos_lod0, ti.cast(hit_normal, ti.f32), iters
