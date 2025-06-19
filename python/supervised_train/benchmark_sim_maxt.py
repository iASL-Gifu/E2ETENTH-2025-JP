import os
import csv
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from f1tenth_gym.maps.map_manager import MapManager
from f1tenth_gym.maps.map_manager import TEST_MAPS as MAP_DICT
from src.envs.envs import make_env
from src.models.models import load_maxt_model 

@hydra.main(config_path="config", config_name="benchmark_sim", version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # --- 環境／プランナ／エージェント等の初期化 ---
    map_manager = MapManager(
        map_name=cfg.envs.map.name,
        map_ext=cfg.envs.map.ext,
        speed=cfg.envs.map.speed,
        downsample=cfg.envs.map.downsample,
        use_dynamic_speed=cfg.envs.map.use_dynamic_speed,
        a_lat_max=cfg.envs.map.a_lat_max,
        smooth_sigma=cfg.envs.map.smooth_sigma
    )
    env = make_env(cfg.envs, map_manager, cfg.vehicle)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- モデルの読み込み ---
    model = load_maxt_model(size='tiny').to(device)
    model_path = os.path.join(cfg.ckpt_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 評価モードに設定

    is_rnn = "Lstm" in cfg.model_name
    
    # --- ベンチマーク結果の保存ディレクトリ ---
    benchmark_dir = cfg.benchmark_dir
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    for ep in range(len(MAP_DICT)):
        map_name = MAP_DICT[ep]
        print(f"Evaluating on map: {map_name}")
        
        # マップごとのディレクトリとCSVファイルの準備
        map_dir = os.path.join(benchmark_dir, map_name)
        os.makedirs(map_dir, exist_ok=True)
        csv_file = os.path.join(map_dir, f"{map_name}_trajectory.csv")
        lap_file = os.path.join(map_dir, f"{map_name}_lap_times.csv")
        
        # CSVファイルの初期化
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "velocity"])

        # ラップタイムのCSV初期化
        if not os.path.exists(lap_file):
            with open(lap_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Lap Number", "Lap Time"])

        env.update_map(map_name, map_ext=cfg.envs.map.ext)
        
        # is_use_prev_action は未定義のため、一旦Falseと仮定します
        is_use_prev_action = False
        prev_action = torch.zeros((1, 2), device=device) if is_use_prev_action else None

        # --- 評価ループ ---
        obs, info = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                scan = obs['scans'][0]
                scan_tensor = torch.from_numpy(scan).float().to(device) / cfg.envs.max_beam_range
                scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0)

                if is_rnn:
                    pass
                else:
                    # --- ★★★ここからが修正箇所★★★ ---
                    
                    # モデルの出力は {'output': {'C1': tensor, 'C2': tensor, ...}} の形式
                    result_dict = model(scan_tensor)
                    predictions_dict = result_dict['output']

                    # 1. 各スケールの予測を平均して、単一の予測テンソルに集約
                    if predictions_dict:
                        # テンソルのリストをスタックし(num_scales, 1, 2)、dim=0で平均を取る -> (1, 2)
                        avg_prediction_tensor = torch.stack(list(predictions_dict.values())).mean(dim=0)
                    else:
                        avg_prediction_tensor = torch.zeros((1, 2), device=device)

                    # 2. テンソルから steer と speed を抽出
                    action_numpy = avg_prediction_tensor.squeeze(0).cpu().numpy()
                    steer = action_numpy[0]
                    speed = action_numpy[1]
                    
                    # 3. 速度のスケーリングと最終的なアクションの決定
                    final_speed = speed * cfg.envs.map.speed
                    action = [steer, final_speed]
                    actions = [action] # f1tenth_gymはリスト形式を期待

                    # --- ★★★修正ここまで★★★ ---

            # 環境のステップ
            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
            
            # 終了条件の判定
            # 注意: f1tenth_gym_rosでは obs['lap_counts'][0] は存在しない場合があります。
            # info['lap_count'] など、お使いの環境に合わせたキーをご利用ください。
            if 'lap_counts' in obs and obs['lap_counts'][0] >= 1:
                terminated = True
            if 'collisions' in obs and obs['collisions'][0]:
                truncated = True
            done = terminated or truncated

            # --- CSVへの書き込み ---
            # (infoのキーも環境によって異なる場合があるため、ご確認ください)
            current_pos = info.get('pose_x', [0,0])[0:2] if 'pose_x' in info else [0,0]
            velocity = info.get('linear_vel_x', 0.0)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_pos[0], current_pos[1], velocity])

            if done:
                lap_time = obs.get('lap_times', [0.0])[0]
                if truncated:
                    print(f"  Collision detected. Lap time set to 0.")
                    with open(lap_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([1, 0.0]) # 失敗したラップとして記録
                elif terminated:
                    print(f"  Lap finished. Time: {lap_time:.4f}s")
                    with open(lap_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([1, lap_time])
                break

            if cfg.render:
                env.render(mode=cfg.render_mode)

            obs = next_obs
    env.close()

if __name__ == "__main__":
    main()