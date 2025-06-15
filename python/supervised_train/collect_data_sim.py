import os
import numpy as np
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager
from f1tenth_gym.maps.map_manager import TRAIN_MAPS as MAP_DICT
from src.planner.purePursuit import PurePursuitPlanner

@hydra.main(config_path="config", config_name="collect_data_sim", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # 実行ごとに固有のランIDディレクトリを作成
    base_out = cfg.output_dir
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(base_out, run_id)
    os.makedirs(out_root, exist_ok=True)

    # マップごとのディレクトリを事前に作成 (これは残しておくと全体構造が分かりやすい)
    for map_name in MAP_DICT:
        os.makedirs(os.path.join(out_root, map_name), exist_ok=True)

    # 環境とプランナーの初期化
    map_cfg     = cfg.envs.map
    map_manager = MapManager(
        map_name=MAP_DICT[0],
        map_ext=map_cfg.ext,
        speed=map_cfg.speed,
        downsample=map_cfg.downsample,
        use_dynamic_speed=map_cfg.use_dynamic_speed,
        a_lat_max=map_cfg.a_lat_max,
        smooth_sigma=map_cfg.smooth_sigma
    )
    env = make_env(env_cfg=cfg.envs, map_manager=map_manager, param=cfg.vehicle)

    wheelbase = cfg.planner.wheelbase
    lookahead = cfg.planner.lookahead
    planner = PurePursuitPlanner(
        wheelbase=wheelbase,
        map_manager=map_manager,
        lookahead=lookahead,
        gain=cfg.planner.gain,
        max_reacquire=cfg.planner.max_reacquire,
    )

    render_flag = cfg.render
    render_mode = cfg.render_mode
    num_sets      = cfg.num_sets # 新しい設定項目
    num_steps     = cfg.num_steps
    num_waypoints = cfg.get('num_waypoints', 10)

    map_counters = {m: 0 for m in MAP_DICT}

    # 各マップを num_sets 回繰り返すループ構造に変更
    for set_idx in range(num_sets): # セット数のループ
        print(f"\n--- Data Collection Set: {set_idx + 1}/{num_sets} ---")
        for map_id, name in enumerate(MAP_DICT): # 各マップのループ
            env.update_map(map_name=name, map_ext=map_cfg.ext)
            obs, info = env.reset()

            count = map_counters[name]
            map_counters[name] += 1

            # --- 保存先パスを更新：マップ名の下にさらにランIDのディレクトリを作成 ---
            # 例: output_dir/20240101_123456/map_name/run0/
            run_output_dir = os.path.join(out_root, name, f"run{count}")
            os.makedirs(run_output_dir, exist_ok=True)

            # --- データの初期化 ---
            positions = []
            scans = []
            waypoints = []
            prev_actions = []
            actions = []

            prev_action = np.zeros((1, 2), dtype='float32')
            current_pos = info.get('current_pos', np.array([0.0, 0.0], dtype='float32'))
            truncated = False

            print(f"Collecting data for Map: {name}, Run: {count + 1}...")

            for step in range(num_steps):
                steer, speed = planner.plan(obs)
                action = np.array([steer, speed], dtype='float32').reshape(1, 2)
                scan = np.array(obs['scans']).astype('float32').squeeze(axis=0)

                wpts = map_manager.get_future_waypoints(
                    current_pos, num_points=num_waypoints
                ).astype('float32')
                if wpts.shape[0] < num_waypoints:
                    pad = np.repeat(wpts[-1][None, :], num_waypoints - wpts.shape[0], axis=0)
                    wpts = np.vstack([wpts, pad])
                wpts = wpts.reshape(1, num_waypoints, 3)

                # --- データの保存 ---
                positions.append(current_pos)
                scans.append(scan)
                waypoints.append(wpts)
                prev_actions.append(prev_action)
                actions.append(action)

                next_obs, reward, terminated, truncated, info = env.step(action)
                if truncated:
                    print(f"Episode terminated or truncated at step {step + 1}.")
                    break

                obs = next_obs
                prev_action = action
                current_pos = info.get('current_pos', current_pos)

                if render_flag:
                    env.render(mode=render_mode) if render_mode else env.render()

            # `terminated` または `truncated` で早期終了しなかった場合のみデータを保存
            if not truncated:
                # --- データの書き込み ---
                np.save(os.path.join(run_output_dir, "scans.npy"), np.array(scans))

                # actions を steers と speeds に分割して保存
                all_actions = np.array(actions)
                all_steers = all_actions[:, 0, 0] # actions の最初の要素がsteer
                all_speeds = all_actions[:, 0, 1] # actions の二番目の要素がspeed

                # 速度を正規化
                max_speed = map_cfg.speed
                if max_speed != 0:
                    all_speeds_normalized = all_speeds / max_speed
                else:
                    print("[WARN] map_cfg.speed is 0, cannot normalize speed. Saving unnormalized speed.")
                    all_speeds_normalized = all_speeds

                np.save(os.path.join(run_output_dir, "steers.npy"), all_steers)
                np.save(os.path.join(run_output_dir, "speeds.npy"), all_speeds_normalized) 

                # 必要であれば、positions, waypoints, prev_actions も保存
                # np.save(os.path.join(run_output_dir, "positions.npy"), np.array(positions))
                # np.save(os.path.join(run_output_dir, "waypoints.npy"), np.array(waypoints))
                # np.save(os.path.join(run_output_dir, "prev_actions.npy"), np.array(prev_actions))
                
                print(f"Data for Map: {name}, Run: {count + 1} saved to: {run_output_dir}")
            else:
                print(f"Data for Map: {name}, Run: {count + 1} not saved due to early termination or truncation.")

if __name__ == '__main__':
    main()