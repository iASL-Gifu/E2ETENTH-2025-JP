import gymnasium as gym
import numpy as np
import os
import time

# gl関連のimportはそのまま
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl

# 定数
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

# 旧コード同様のSimulator, Integratorのimport
from .base_classes import Simulator, Integrator

class F110Env(gym.Env):
    """
    Gymnasium用のF1TENTH環境

    環境は以下のように初期化してください:
        gymnasium.make('f110_gym:F110-v0', **kwargs)

    kwargsの例:
        seed (int, default=12345)
        map (str, default='vegas') : 'berlin', 'skirk', 'levine' など、もしくはカスタムマップのyamlファイルパス
        map_ext (str, default='png')
        params (dict, default= { ... } )
        num_agents (int, default=2)
        timestep (float, default=0.01)
        ego_idx (int, default=0)
        integrator (Enum, default=Integrator.RK4)
    """
    metadata = {'render_modes': ['human', 'human_fast', 'rgb_array']}

    # クラス変数としてレンダラ等を保持
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):
        super().__init__()  # Gymnasiumの初期化呼び出し
        
        # 各パラメータの抽出（存在しなければデフォルト値）
        self.seed = kwargs.get('seed', 12345)
        self.map_name = kwargs.get('map', 'vegas')
        if self.map_name == 'berlin':
            self.map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps', 'berlin.yaml')
        elif self.map_name == 'skirk':
            self.map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps', 'skirk.yaml')
        elif self.map_name == 'levine':
            self.map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps', 'levine.yaml')
        else:
            ## .yaml
            self.map_path = self.map_name + '.yaml'
            
        self.map_ext = kwargs.get('map_ext', '.png')
        self.params = kwargs.get('params', {
            'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 
            'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 
            'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 
            's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 
            'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 
            'v_max': 20.0, 'width': 0.31, 'length': 0.58
        })
        self.num_beams = kwargs.get('num_beams', 20)
        self.num_agents = kwargs.get('num_agents', 2)
        self.timestep = kwargs.get('timestep', 0.01)
        self.ego_idx = kwargs.get('ego_idx', 0)
        self.integrator = kwargs.get('integrator', Integrator.RK4)

        # 各種状態変数の初期化
        self.start_thresh = 0.5  # 10cm
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))
        self.near_start = True
        self.num_toggles = 0
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # Simulatorの初期化とマップ設定
        self.sim = Simulator(self.params, self.num_beams, self.num_agents, self.seed, time_step=self.timestep, integrator=self.integrator)
        self.sim.set_map(self.map_path, self.map_ext)
        self.render_obs = None

    def _check_done(self):
        """
        現在のロールアウトが終了したかを判定
        """
        left_t = 2
        right_t = 2
        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0
        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)
        return bool(done), self.toggle_list >= 8

    def _update_state(self, obs_dict):
        """
        観測データに基づき状態を更新する
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        """
        ステップ関数

        Gymnasium仕様に合わせて、以下の5要素を返します:
          - observation
          - reward
          - terminated: 自然終了の場合にTrue
          - truncated: タイムリミット等による中断の場合にTrue（今回は常にFalseとする）
          - info
        """
        obs = self.sim.step(action)
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        F110Env.current_obs = obs

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
        }
        
        reward = self.timestep
        self.current_time += self.timestep

        self._update_state(obs)
        done, toggle_list = self._check_done()
        info = {'checkpoint_done': toggle_list}
        
        # Gymnasiumではdoneをterminatedとtruncatedに分ける
        terminated = done
        truncated = False  # 必要に応じてタイムリミット判定などを実装可能
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        環境のリセット

        Gymnasium仕様では、reset()は (observation, info) を返します。
        初期状態を設定するためのパラメータ（例えば poses ）は、options経由で受け取ることができます。
        """
        super().reset(seed=seed)
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # options から初期ポーズを取得（なければデフォルト値を使用）
        if options is not None and "poses" in options:
            poses = options["poses"]
        else:
            poses = np.zeros((self.num_agents, 3))  # 必要に応じて適切な初期値に変更

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([
            [np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])],
            [np.sin(-self.start_thetas[self.ego_idx]),  np.cos(-self.start_thetas[self.ego_idx])]
        ])

        self.sim.reset(poses)

        # 初回のステップで状態を更新
        action = np.zeros((self.num_agents, 2))
        obs, reward, terminated, truncated, info = self.step(action)
        
        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
        }
        
        return obs, info

    def update_map(self, map_path, map_ext):
        """
        シミュレーションで用いるマップを更新する
        """
        self.sim.set_map(map_path, map_ext) ## .yamlのパスを指定
        self.map_path = map_path
        self.map_ext  = map_ext

        # .yamlを覗いたpathを取得
        if map_path.endswith('.yaml'):
            map_name, ext = os.path.splitext(map_path)
        if F110Env.renderer is not None:
            # EnvRenderer.update_map(マップyamlのベースパス, 拡張子)
            F110Env.renderer.update_map(map_name, self.map_ext)

    def update_params(self, params, index=-1):
        """
        車両パラメータを更新する
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        レンダリング時に追加の描画関数を呼び出す
        """
        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        環境をレンダリングする。
        
        mode:
        - 'human': ウィンドウに描画し、実時間に近い速度で表示
        - 'human_fast': 高速レンダリング（実時間制御なし）
        - 'rgb_array': 描画結果を RGB の NumPy 配列として返す
        """
        assert mode in ['human', 'human_fast', 'rgb_array']
        
        if F110Env.renderer is None:
            from .rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self.map_name, self.map_ext)
            
        F110Env.renderer.update_obs(self.render_obs)
        
        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)
        
        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        
        if mode == 'human':
            F110Env.renderer.flip()
            time.sleep(0.005)
        elif mode == 'human_fast':
            F110Env.renderer.flip()
        elif mode == 'rgb_array':
            # 描画後の RGB 配列を取得して返す
            return F110Env.renderer.get_rgb_array()


    def close(self):
        """
        レンダラーやその他リソースの解放を行う
        """
        if F110Env.renderer is not None:
            F110Env.renderer.close()
            F110Env.renderer = None
