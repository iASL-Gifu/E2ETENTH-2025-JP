import gymnasium as gym
import numpy as np
import math
from pyglet.gl import GL_POINTS
from pyglet.text import Label


from f1tenth_gym.maps.map_manager import MapManager

class F110Wrapper(gym.Wrapper):
    """
    F110Env のラッパークラス。
    環境のインターフェースを簡略化し、追加機能を提供する。

    - step() の出力を整形
    - reset() のオプション処理を強化
    - レンダリング処理の簡略化
    - ラップタイム取得機能の追加
    """

    def __init__(self, env, map_manager: MapManager):
        super().__init__(env)
        self.ego_idx = env.ego_idx
        self.map_manager = map_manager

        self._waypoint_vlists = []  # ← 追加：ウェイポイント用のVertexListを保持
        self.speed = 0.0

        self.env.add_render_callback(self.render_callback)
        # Waypoint描画機能をレンダリングコールバックとして追加
        self.env.add_render_callback(self.render_waypoints)

    def step(self, action, num_points=10):
        """
        環境を1ステップ進める。

        返り値:
        - observation: 観測データ
        - reward: ステップごとの報酬
        - terminated: エピソード終了フラグ
        - truncated: エピソード途中終了フラグ
        - info: 追加情報（ラップタイムなど）
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 0.0

        ## waypointを作成
        current_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        waypoint = self.map_manager.get_future_waypoints(current_pos, num_points=num_points)
        info['waypoint'] = waypoint
        info['current_pos'] = current_pos
        vel_x = obs['linear_vels_x'][0]
        vel_y = obs['linear_vels_y'][0]
        vel = np.sqrt(vel_x**2 + vel_y**2)
        info['velocity'] = vel

        self.speed = vel

        # spin
        if abs(obs['poses_theta'][0]) > 100.0:
            truncated = True
            
        # 1 lap 終了
        if obs['lap_counts'][0] == 1:
            terminated = True

        ## 報酬
        ### 時間経過で報酬を減らす
        reward -= self.env.timestep

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, index: int=0):
        """
        環境をリセットする。

        返り値:
        - observation: 初期観測データ
        - info: 追加情報（オプションで初期状態指定可能）
        """
        positions = []
        if self.map_manager.waypoints is not None:
            num_waypoints = len(self.map_manager.waypoints)
            num_agents = self.env.num_agents

            # 各エージェントに対して均等にWaypointを割り当て
            index_increment = num_waypoints / num_agents

            for i in range(num_agents):
                # 浮動小数点のインデックスを整数インデックスに変換
                waypoint_index = int(i * index_increment + index) % num_waypoints 
                next_waypoint_index = (waypoint_index + 1) % num_waypoints

                x, y = self.map_manager.waypoints[waypoint_index][:2]
                next_x, next_y = self.map_manager.waypoints[next_waypoint_index][:2]
            
                # 角度を計算（ラジアン）
                dx = next_x - x
                dy = next_y - y
                t = math.atan2(dy, dx)

                positions.append([x, y, t])

        else:
            # Waypointsが存在しない場合のデフォルトの位置
            positions = [[0, 0, 0] for _ in range(self.env.num_agents)]

        options = {
            "poses": np.array(positions)
        }
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def update_map(self, map_name, map_ext):
        """
        環境のマップを更新する。

        引数:
        - map_name: 新しいマップの名前
        - map_ext: 新しいマップの拡張子
        """
        self.map_manager.update_map(map_name)
        map_path = self.map_manager.map_yaml_path
        max_ext = self.map_manager.map_ext
        self.env.update_map(map_path, max_ext)

    def render(self, mode="human"):
        """
        環境のレンダリングを簡単に呼び出せるようにする。
        """
        return self.env.render(mode=mode)

    def get_lap_time(self):
        """
        Ego車両の現在のラップタイムを取得する。

        返り値:
        - float: 現在のラップタイム
        """
        return self.env.lap_times[self.ego_idx]

    def close(self):
        """
        環境を閉じる（リソース解放）。
        """
        self.env.close()

    def render_callback(self, env_renderer):
        # custom extra drawing function for camera update
        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)

        l = 800
        e.left = left - l
        e.right = right + l
        e.top = top + l
        e.bottom = bottom - l

        # 初回のみ Label を生成する
        if not hasattr(self, 'speed_label'):
            self.speed_label = Label('Speed: 0.00 m/s',
                                    font_name='Times New Roman',
                                    font_size=14,
                                    x=left, y=top - 30,
                                    anchor_x='left', anchor_y='top',
                                    color=(255, 255, 255, 255),
                                    batch=e.batch)

        # テキストの内容だけ更新する
        self.speed_label.text = f'Speed: {self.speed:.2f} m/s'

        # 座標も更新する（カメラが動く場合）
        self.speed_label.x = left
        self.speed_label.y = top - 30


    def render_waypoints(self, renderer):
        """
        Waypoint を描画するコールバック。
        古い頂点リストは毎回削除してから、新しいものを登録します。
        """
        # 1) Waypoints 未設定時は何もしない
        if self.map_manager.waypoints is None:
            return

        # 2) 前回のウェイポイントをクリア
        for vlist in self._waypoint_vlists:
            vlist.delete()
        self._waypoint_vlists.clear()

        # 3) 座標変換・スケーリング
        points = np.vstack((
            self.map_manager.waypoints[:, 0],
            self.map_manager.waypoints[:, 1]
        )).T
        scaled = 50. * points

        # 4) 新しいウェイポイントを登録
        for i, (x, y) in enumerate(scaled):
            # ターゲット waypoint を赤、それ以外を灰にするならここで色分け
            color = [255, 0, 0] if i == 0 else [200, 200, 200]
            v = renderer.batch.add(
                1, GL_POINTS, None,
                ('v3f/stream', [x, y, 0.]),
                ('c3B/stream', color)
            )
            self._waypoint_vlists.append(v)
            