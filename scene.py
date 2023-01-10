import time
import os
from datetime import datetime
import numpy as np
import taichi as ti
from renderer import Renderer
from renderer.math_utils import np_normalize, np_rotate_matrix
import __main__


VOXEL_DX = 1 / 64
SCREEN_RES = (1920, 1080)
UP_DIR = (0, 1, 0)
HELP_MSG = """
====================================================
Camera:
* Drag with your left mouse button to rotate
* Press W/A/S/D/Q/E to move
* P to screenshot
====================================================
"""

from taichi.lang.impl import _ti_core

class Camera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((0.4, 0.5, 2.0))
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = None

    @property
    def mouse_exclusive_owner(self):
        return True

    def update_camera(self, delta_time):
        res = self._update_by_wasd(delta_time)
        res = self._update_by_mouse() or res
        return res

    def _update_by_mouse(self):
        win = self._window
        if not self.mouse_exclusive_owner or not win.is_pressed(ti.ui.LMB):
            self._last_mouse_pos = None
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
            return False
        # Makes camera rotation feels right
        dx, dy = self._last_mouse_pos - mouse_pos
        self._last_mouse_pos = mouse_pos

        out_dir = self._lookat_pos - self._camera_pos
        leftdir = self._compute_left_dir(np_normalize(out_dir))

        scale = 3
        rotx = np_rotate_matrix(self._up, dx * scale)
        roty = np_rotate_matrix(leftdir, dy * scale)

        out_dir_homo = np.array(list(out_dir) + [0.0])
        new_out_dir = np.matmul(np.matmul(roty, rotx), out_dir_homo)[:3]
        self._lookat_pos = self._camera_pos + new_out_dir

        return True

    def _update_by_wasd(self, delta_time):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        lut = [
            ("w", tgtdir),
            ("a", leftdir),
            ("s", -tgtdir),
            ("d", -leftdir),
            ("e", [0, -1, 0]),
            ("q", [0, 1, 0]),
        ]
        dir = np.array([0.0, 0.0, 0.0])
        pressed = False
        for key, d in lut:
            if win.is_pressed(key):
                pressed = True
                dir += np.array(d)
        if not pressed:
            return False
        dir *= delta_time
        self._lookat_pos += dir
        self._camera_pos += dir
        return True

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)


class Scene:
    def __init__(self, voxel_edges=0.06, exposure=3):
        ti.init(arch=ti.vulkan, offline_cache=True)
        print(HELP_MSG)
        self.window = ti.ui.Window("Taichi Voxel Renderer", SCREEN_RES, vsync=False)
        self.camera = Camera(self.window, up=UP_DIR)
        self.renderer = Renderer(
            dx=VOXEL_DX,
            image_res=SCREEN_RES,
            up=UP_DIR,
            voxel_edges=voxel_edges,
            exposure=exposure,
        )

        self.renderer.set_camera_pos(*self.camera.position)
        self.renderer.set_directional_light((1, 1, 1), 0.1, (0.0, 0.0, 0.0)) # set default values
        if not os.path.exists("screenshot"):
            os.makedirs("screenshot")

    @staticmethod
    @ti.func
    def round_idx(idx_):
        idx = ti.cast(idx_, ti.f32)
        return ti.Vector([ti.round(idx[0]), ti.round(idx[1]), ti.round(idx[2])]).cast(
            ti.i32
        )

    @ti.func
    def set_voxel(self, idx, mat, color):
        self.renderer.set_voxel(self.round_idx(idx), mat, color)

    @ti.func
    def get_voxel(self, idx):
        mat, color = self.renderer.get_voxel(self.round_idx(idx))
        return mat, color

    def set_floor(self, height, color, material=1):
        self.renderer.floor_height[None] = height
        self.renderer.floor_color[None] = color
        self.renderer.floor_material[None] = material

    def set_directional_light(self, direction, direction_noise, color):
        self.renderer.set_directional_light(direction, direction_noise, color)

    def set_background_color(self, color):
        self.renderer.background_color[None] = color

    def set_use_physical_sky(self, use):
        if use:
            self.renderer.use_physical_atmosphere[None] = 1
        else:
            self.renderer.use_physical_atmosphere[None] = 0
    
    def set_use_clouds(self, use):
        if use:
            self.renderer.atmos.use_clouds[None] = 1
        else:
            self.renderer.atmos.use_clouds[None] = 0

    def finish(self):
        self.renderer.prepare_data()

        canvas = self.window.get_canvas()
        gui = self.window.get_gui()
        samples = 0
        samples_per_frame = 1
        last_1k_samples_time = 0.0
        enable_gui = True
        current_fov = self.renderer.fov[None]
        initial_t = time.time()
        last_t = initial_t

        aspect = SCREEN_RES[0]/SCREEN_RES[1]

        # taichi built-in camera. We use this only so that we can easily get the OpenGL standard
        # projection and view matrices without needing to do any extra work.
        tcamera = ti.ui.Camera()
        tcamera.up(0, 1, 0)
        tcamera.z_far(10.0)
        tcamera.z_near(0.01)

        camera_is_moving = False
        first_show = True

        # _ti_core.wait_for_debugger()

        sample_idx = 1
        if self.renderer.use_physical_atmosphere[None] == 1:
            print("Computing clouds")
        max_samples = 32

        # split rendering of skybox into slices as to not timeout
        slice_idx = 0
        max_slices = 32
        while self.window.running:
            

            should_reset_framebuffer = False
            self.renderer.set_max_samples(999999999.0)
            self.renderer.set_render_scale(1.0)

            t = time.time()
            if self.camera.update_camera(t - last_t):
                self.renderer.set_camera_pos(*self.camera.position)
                look_at = self.camera.look_at
                self.renderer.set_look_at(*look_at)
                self.renderer.set_max_samples(50.0)
                self.renderer.set_render_scale(0.5)
                if not camera_is_moving:
                    camera_is_moving = True
                    should_reset_framebuffer = True
            else:
                if camera_is_moving:
                    camera_is_moving = False
                    should_reset_framebuffer = True
            
            self.renderer.set_camera_is_moving(camera_is_moving)
            last_t = t
            

            # update built-in camera
            tcamera.position(self.camera._camera_pos[0], self.camera._camera_pos[1], self.camera._camera_pos[2])
            tcamera.lookat(self.camera._lookat_pos[0], self.camera._lookat_pos[1], self.camera._lookat_pos[2])
            tcamera.fov(np.rad2deg(current_fov))
            self.renderer.set_proj_mat(tcamera.get_projection_matrix(aspect))
            self.renderer.set_view_mat(tcamera.get_view_matrix())
            

            if should_reset_framebuffer:
                self.renderer.reset_framebuffer()

            if sample_idx < max_samples+1 and self.renderer.use_physical_atmosphere[None] == 1:
                self.renderer.accumulate_clouds(max_samples)
                print(sample_idx,"/",max_samples," cloud samples")
                self.window.show()
                sample_idx += 1
            elif sample_idx == max_samples+1 and slice_idx < max_slices and self.renderer.use_physical_atmosphere[None] == 1:
                print(slice_idx+1,"/",max_slices," skybox progress")
                self.renderer.compute_atmosphere(slice_idx, max_slices)
                if slice_idx == max_slices - 1:
                    print("Done atmosphere & clouds")
                slice_idx += 1
            else:
                for i in range(samples_per_frame):
                    self.renderer.accumulate()
                print("pos ", self.camera.position, "look_at ", self.camera.look_at)


            img = self.renderer.fetch_image()

            self.renderer.copy_prev_matrices()

            if self.window.is_pressed("p"):
                timestamp = datetime.today().strftime("%Y-%m-%d-%H%M%S")
                dirpath = os.getcwd()
                main_filename = os.path.split(__main__.__file__)[1]
                fname = os.path.join(
                    dirpath, "screenshot", f"{main_filename}-{timestamp}.jpg"
                )
                ti.tools.image.imwrite(img, fname)
                print(f"Screenshot has been saved to {fname}")
            canvas.set_image(img)
            if samples > 1024:
                last_1k_samples_time = time.time() - initial_t
                print("1024 samples took", last_1k_samples_time)
                samples -= 1024
                initial_t = time.time()
            samples += samples_per_frame

            if self.window.is_pressed("g"):
                enable_gui = not enable_gui

            if enable_gui:
                with gui.sub_window("Settings", x=0.05, y=0.05, width=0.25, height=0.2) as g:
                    g.text("Press G to show/hide GUI")
                    g.text(f"Last 1024 samples took {last_1k_samples_time:.3f}s")
                    new_fow = np.deg2rad(g.slider_float("Verticle FOV", np.rad2deg(current_fov), 1.0, 90.0))
                    if new_fow != current_fov:
                        current_fov = new_fow
                        self.renderer.fov[None] = current_fov

            self.window.show()

            if first_show:
                # ti.profiler.print_scoped_profiler_info()
                first_show = False
