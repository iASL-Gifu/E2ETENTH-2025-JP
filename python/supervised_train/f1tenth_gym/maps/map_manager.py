import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

MAP_DICT = {
    0: 'Austin', 1: 'BrandsHatch', 2: 'Budapest', 3: 'Catalunya', 4: 'Hockenheim',
    5: 'IMS', 6: 'Melbourne', 7: 'MexicoCity', 8: 'Monza', 9: 'MoscowRaceway',
    10: 'Nuerburgring', 11: 'Oschersleben', 12: 'Sakhir', 13: 'SaoPaulo',
    14: 'Sepang', 15: 'Silverstone', 16: 'Sochi', 17: 'Spa', 18: 'Spielberg',
    19: 'YasMarina', 20: 'Zandvoort'
}

## 8:2で分割した場合の学習用マップ
TRAIN_MAPS = [
    'Austin', 'Budapest', 'Hockenheim', 'Melbourne', 'MexicoCity',
    'MoscowRaceway', 'Nuerburgring', 'Oschersleben', 'Sakhir', 'Sepang',
    'Silverstone', 'Sochi', 'Spa', 'Spielberg', 'YasMarina'
]

TEST_MAPS = [
    'BrandsHatch', 'Catalunya', 'IMS', 'Monza', 'SaoPaulo', 'Zandvoort'
]


class MapManager:
    def __init__(
        self,
        map_name: str,
        map_dir: str = os.path.dirname(__file__),
        map_ext: str = '.png',
        line_type: str = 'center',
        speed: float = 5.0,
        downsample: int = 1,
        use_dynamic_speed: bool = False,
        a_lat_max: float = 3.0,
        smooth_sigma: float = 2.0
    ):
        # 基本設定をメンバに保持
        self.map_dir          = map_dir
        self.map_ext          = map_ext
        self.line_type = line_type
        self.speed            = speed
        self.downsample       = downsample
        self.use_dynamic_speed = use_dynamic_speed
        self.a_lat_max        = a_lat_max
        self.smooth_sigma     = smooth_sigma

        # 初回ロード
        self._set_map_name(map_name)
        self._load_map_data()

    def _set_map_name(self, map_name: str):
        """マップ名から各種パスを再設定"""
        self.map_name         = map_name
        self.map_base_dir     = os.path.join(self.map_dir, map_name)
        self.map_path         = os.path.join(self.map_base_dir, map_name + "_map")
        self.map_img_path     = os.path.join(self.map_base_dir, map_name + "_map" + self.map_ext)
        self.map_yaml_path    = os.path.join(self.map_base_dir, map_name + '_map.yaml')
        self.center_line_path = os.path.join(self.map_base_dir, map_name + '_centerline.csv')
        self.race_line_path   = os.path.join(self.map_base_dir, map_name + '_raceline.csv')

    def _compute_speeds(self, wpts: np.ndarray) -> np.ndarray:
        N = len(wpts)
        if not self.use_dynamic_speed:
            return np.full((N,1), self.speed, dtype=np.float32)

        # 曲率計算
        curvature = np.zeros(N, dtype=np.float32)
        for i in range(1, N-1):
            v1 = wpts[i]   - wpts[i-1]
            v2 = wpts[i+1] - wpts[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_t = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
            theta = np.arccos(cos_t)
            curvature[i] = theta / n2

        # 曲率スムージング
        curvature_smooth = gaussian_filter1d(curvature, sigma=self.smooth_sigma)

        # カーブクラス分け
        bins = [0.01, 0.02, 0.2]
        curve_classes = np.digitize(curvature_smooth, bins=bins)
        self.curve_classes = curve_classes

        # クラスごとのベース速度
        curve_base_speed = [self.speed, 0.8*self.speed, 0.7*self.speed, 0.5*self.speed]

        # 微調整係数を曲率に応じて計算（正規化: 大きい曲率 → 小さい係数）
        max_curv = np.max(curvature_smooth)
        min_curv = np.min(curvature_smooth)
        epsilon = 1e-6
        norm_curvature = (curvature_smooth - min_curv) / (max_curv - min_curv + epsilon)
        adjustment_factor = 1.0 - 0.5 * norm_curvature  # 0.5〜1.0の補正係数

        # 最終速度計算
        speeds = np.array([
            curve_base_speed[cls] * adjustment_factor[i]
            for i, cls in enumerate(curve_classes)
        ], dtype=np.float32)

        return speeds.reshape(-1, 1)


    def _load_map_data(self):
        """ウェイポイント読み込み→ダウンサンプル→速度付与→累積距離計算"""
        if self.line_type == 'center':
            wpts = np.genfromtxt(self.center_line_path, delimiter=',', usecols=(0, 1))
            wpts = wpts[::self.downsample]
            speeds = self._compute_speeds(wpts)  # centerline は速度計算が必要

        elif self.line_type == 'race':
            # s; x; y; psi; kappa; vx; ax
            data = np.genfromtxt(self.race_line_path, delimiter=';', usecols=(1, 2, 5))  # x, y, vx_mps
            data = data[::self.downsample]
            wpts = data[:, :2]
            speeds = data[:, 2].reshape(-1, 1)

        else:
            raise ValueError(f"Invalid line_type '{self.line_type}' (expected 'center' or 'race')")

        print(f"Loading waypoint type: {self.line_type}")
        self.waypoints = np.column_stack((wpts, speeds))  # shape=(N, 3)

        # 各セグメント長・累積距離を計算
        diffs = np.diff(self.waypoints[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.cum_dis = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        self.total_dis = float(self.cum_dis[-1])

    # 以降のメソッドは変更なし
    def update_map(
        self,
        new_map_name: str,
        speed: float = None,
        downsample: int = None,
        use_dynamic_speed: bool = None,
        line_type: str = None
    ):
        if speed is not None:          self.speed            = speed
        if downsample is not None:     self.downsample       = downsample
        if use_dynamic_speed is not None: self.use_dynamic_speed = use_dynamic_speed
        if line_type is not None:      self.line_type = line_type

        self._set_map_name(new_map_name)
        self._load_map_data()

    def get_trackline_segment(self, point):
        wpts_array = self.waypoints[:, :2]
        point_array = np.array(point).reshape(1,2)
        dists = np.linalg.norm(point_array - wpts_array, axis=1)
        i = np.argmin(dists)
        if i == 0:
            return 0, dists
        elif i == len(dists)-1:
            return len(dists)-2, dists
        if dists[i+1] < dists[i-1]: return i, dists
        return i-1, dists

    def interp_pts(self, idx, dists):
        d_ss = self.cum_dis[idx+1] - self.cum_dis[idx]
        d1, d2 = dists[idx], dists[idx+1]
        if d1 < 0.01:
            return 0, 0
        if d2 < 0.01:
            return dists[idx], 0
        s = (d_ss + d1 + d2)/2
        area = max((s*(s-d_ss)*(s-d1)*(s-d2))**0.5, 0)
        h = (2*area)/d_ss
        x = (d1**2 - h**2)**0.5
        return x, h

    def get_future_waypoints(self, current_point, num_points=10):
        idx, _ = self.get_trackline_segment(current_point)
        future_indices = [(idx + i) % len(self.waypoints) for i in range(num_points)]
        return self.waypoints[future_indices]

    def calc_progress(self, point):
        idx, dists = self.get_trackline_segment(point)
        x, _ = self.interp_pts(idx, dists)
        s = (self.cum_dis[idx] + x) / self.total_dis * 100
        return s
