# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
Rendering engine for f1tenth gym env based on pyglet and OpenGL
Author: Hongrui Zheng
"""

# opengl stuff
try:
    import pyglet
    from pyglet.gl import *
except ImportError:
    pass

# other
import numpy as np
from PIL import Image
import yaml

# helpers
from .collision_models import get_vertices

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

class EnvRenderer(pyglet.window.Window):
    """
    A window class inherited from pyglet.window.Window, handles the camera/projection interaction,
    resizing window, and rendering the environment
    """
    def __init__(self, width, height, *args, **kwargs):
        # ウィンドウ／OpenGL の初期化
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        super().__init__(width, height, config=conf, resizable=True, vsync=False, *args, **kwargs)
        glClearColor(9/255, 32/255, 87/255, 1.)

        # カメラ初期設定
        self.left   = -width/2
        self.right  =  width/2
        self.bottom = -height/2
        self.top    =  height/2
        self.zoom_level    = 1.2
        self.zoomed_width  = width
        self.zoomed_height = height

        # 描画用バッチ
        self.batch = pyglet.graphics.Batch()

        # 地図点群用VertexListを保持しておくためのリスト
        self._map_vertex_lists = []

        # 現在の地図点群座標
        self.map_points = None
        # 車両ポーズ／頂点情報
        self.poses    = None
        self.vertices = None

        # FPS表示
        self.fps_display = pyglet.window.FPSDisplay(self)

    def update_map(self, map_path, map_ext):
        """
        Update the map being drawn by the renderer. 
        - 古い地図をクリア
        - 新しい YAML・PNG を読み込み、点群を batch に登録
        """
        # 1) 既存の地図点群をすべて削除
        for vlist in self._map_vertex_lists:
            vlist.delete()
        self._map_vertex_lists.clear()

        # 2) YAML から解像度／原点を読み込み
        with open(map_path + '.yaml', 'r') as yaml_stream:
            meta = yaml.safe_load(yaml_stream)
            res    = meta['resolution']
            origin = meta['origin']
            ox, oy = origin[0], origin[1]

        # 3) PNG 画像を読み込んで上下反転
        img = np.array(
            Image.open(map_path + map_ext)
                 .transpose(Image.FLIP_TOP_BOTTOM)
        ).astype(np.float64)
        h, w = img.shape[0], img.shape[1]

        # 4) 画素→ワールド座標への変換
        xs = (np.arange(w) * res + ox)
        ys = (np.arange(h) * res + oy)
        mx, my = np.meshgrid(xs, ys)
        mz = np.zeros_like(mx)
        coords = np.vstack((mx.flatten(), my.flatten(), mz.flatten()))

        # 5) 障害物（画素値==0）だけマスク
        mask = (img.flatten() == 0.0)
        pts  = coords[:, mask].T  # shape=(N,3)
        # 任意のスケール（もともとのコードは50倍していたようなので継承）
        map_points = 50. * pts

        # 6) 新しい点群を batch に登録し、VertexList を保存
        for p in map_points:
            vlist = self.batch.add(
                1, GL_POINTS, None,
                ('v3f/stream', [p[0], p[1], p[2]]),
                ('c3B/stream', [183, 193, 222])
            )
            self._map_vertex_lists.append(vlist)

        # 7) 内部状態を更新
        self.map_points = map_points

    def on_resize(self, width, height):
        """
        Callback function on window resize, overrides inherited method, and updates camera values on top of the inherited on_resize() method.

        Potential improvements on current behavior: zoom/pan resets on window resize.

        Args:
            width (int): new width of window
            height (int): new height of window

        Returns:
            None
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        (width, height) = self.get_size()
        self.left = -self.zoom_level * width/2
        self.right = self.zoom_level * width/2
        self.bottom = -self.zoom_level * height/2
        self.top = self.zoom_level * height/2
        self.zoomed_width = self.zoom_level * width
        self.zoomed_height = self.zoom_level * height

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Callback function on mouse drag, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (int): Relative X position from the previous mouse position.
            dy (int): Relative Y position from the previous mouse position.
            buttons (int): Bitwise combination of the mouse buttons currently pressed.
            modifiers (int): Bitwise combination of any keyboard modifiers currently active.

        Returns:
            None
        """

        # pan camera
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top -= dy * self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        Callback function on mouse scroll, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            scroll_x (float): Amount of movement on the horizontal axis.
            scroll_y (float): Amount of movement on the vertical axis.

        Returns:
            None
        """

        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1

        # If zoom_level is in the proper range
        if .01 < self.zoom_level * f < 10:

            self.zoom_level *= f

            (width, height) = self.get_size()

            mouse_x = x/width
            mouse_y = y/height

            mouse_x_in_world = self.left + mouse_x*self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y*self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

    def on_close(self):
        """
        Callback function when the 'x' is clicked on the window, overrides inherited method. Also throws exception to end the python program when in a loop.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: with a message that indicates the rendering window was closed
        """

        super().on_close()
        raise Exception('Rendering window was closed.')

    def on_draw(self):
        """
        Function when the pyglet is drawing. The function draws the batch created that includes the map points, the agent polygons, and the information text, and the fps display.
        
        Args:
            None

        Returns:
            None
        """

        # if map and poses doesn't exist, raise exception
        if self.map_points is None:
            raise Exception('Map not set for renderer.')
        if self.poses is None:
            raise Exception('Agent poses not updated for renderer.')

        # Initialize Projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()

        # Clear window with ClearColor
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set orthographic projection matrix
        glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        # Draw all batches
        self.batch.draw()
        self.fps_display.draw()
        # Remove default modelview matrix
        glPopMatrix()

    def update_obs(self, obs):
        """
        Updates the renderer with the latest observation from the gym environment, including the agent poses, and the information text.

        Args:
            obs (dict): observation dict from the gym env

        Returns:
            None
        """

        self.ego_idx = obs['ego_idx']
        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        poses_theta = obs['poses_theta']

        num_agents = len(poses_x)
        if self.poses is None:
            self.cars = []
            for i in range(num_agents):
                if i == self.ego_idx:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices), ('c3B', [172, 97, 185, 172, 97, 185, 172, 97, 185, 172, 97, 185]))
                    self.cars.append(car)
                else:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices), ('c3B', [99, 52, 94, 99, 52, 94, 99, 52, 94, 99, 52, 94]))
                    self.cars.append(car)

        poses = np.stack((poses_x, poses_y, poses_theta)).T
        for j in range(poses.shape[0]):
            vertices_np = 50. * get_vertices(poses[j, :], CAR_LENGTH, CAR_WIDTH)
            vertices = list(vertices_np.flatten())
            self.cars[j].vertices = vertices
        self.poses = poses

        # self.score_label.text = 'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'.format(laptime=obs['lap_times'][0], count=obs['lap_counts'][obs['ego_idx']])

    def get_rgb_array(self):
        """
        現在のフレームバッファから RGB 配列を取得して返す。
        """
        # 描画後に描画バッファから画像データを取得
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        # 'RGB' チャンネルで読み込み。pitch は幅 * 3（チャネル数）となる
        arr = np.frombuffer(image_data.get_data('RGB', image_data.width * 3), dtype=np.uint8)
        arr = arr.reshape((image_data.height, image_data.width, 3))
        # OpenGL は下から上へ描画しているため、配列を上下反転させる
        arr = np.flipud(arr)
        return arr
