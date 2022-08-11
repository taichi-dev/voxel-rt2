import taichi as ti

@ti.data_oriented
class VoxelWorld:
    def __init__(self, voxel_size, voxel_grid_res, voxel_edges):
        self.voxel_color = ti.Vector.field(3, dtype=ti.u8)
        self.voxel_material = ti.field(dtype=ti.i8)

        self.voxel_size = voxel_size
        self.voxel_inv_size = 1 / voxel_size
        # Note that voxel_inv_size == voxel_grid_res iff the box has width = 1
        self.voxel_grid_res = voxel_grid_res
        self.voxel_grid_offset = [-self.voxel_grid_res // 2 for _ in range(3)]

        ti.root.dense(ti.ijk, self.voxel_grid_res).place(
            self.voxel_color, self.voxel_material, offset=self.voxel_grid_offset
        )

        shape = (self.voxel_grid_res, self.voxel_grid_res, self.voxel_grid_res)
        self.voxel_color_texture = ti.Texture(ti.u8, 4, shape)

        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        
        self.voxel_edges = voxel_edges

    @ti.func
    def inside_grid(self, ipos):
        return (
            ipos.min() >= -self.voxel_grid_res // 2
            and ipos.max() < self.voxel_grid_res // 2
        )

    @ti.func
    def voxel_surface_color(self, voxel_index, voxel_uv, colors: ti.template()):
        boundary = self.voxel_edges
        count = 0
        for i in ti.static(range(3)):
            if voxel_uv[i] < boundary or voxel_uv[i] > 1 - boundary:
                count += 1

        f = 0.0
        if count >= 2:
            f = 1.0

        voxel_color = ti.Vector([0.0, 0.0, 0.0])
        is_light = 0
        if self.inside_grid(voxel_index):
            data = colors.fetch((voxel_index - self.voxel_grid_offset).zyx, 0)
            voxel_color = data.rgb
            voxel_material = ti.cast(data.a * 255.0, ti.i32)
            if voxel_material == 2:
                is_light = 1

        return voxel_color * (1.3 - 1.2 * f), is_light

    @ti.kernel
    def _recompute_bbox(self):
        for d in ti.static(range(3)):
            self.bbox[0][d] = 1e9
            self.bbox[1][d] = -1e9
        for I in ti.grouped(self.voxel_material):
            if self.voxel_material[I] != 0:
                for d in ti.static(range(3)):
                    ti.atomic_min(self.bbox[0][d], (I[d] - 1) * self.voxel_size)
                    ti.atomic_max(self.bbox[1][d], (I[d] + 2) * self.voxel_size)

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

    def update_data(self):
        self._recompute_bbox()
        self._make_texture(self.voxel_color_texture)
