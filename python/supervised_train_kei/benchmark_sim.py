import os
import csv
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from f1tenth_gym.maps.map_manager import MapManager
from f1tenth_gym.maps.map_manager import TEST_MAPS as MAP_DICT
from src.envs.envs import make_env
from src.models.models import load_cnn_model 

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
    model = load_cnn_model(
        model_name=cfg.model_name, 
        input_dim=cfg.input_dim, 
        output_dim=cfg.output_dim
    ).to(device)
    model_path = os.path.join(cfg.ckpt_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 評価モードに設定
    model.to(device)

    is_rnn = "Lstm" in cfg.model_name
    is_use_prev_action = "Action" in cfg.model_name
    
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

        hidden_state = None
        ## 0で初期化
        prev_action =  torch.zeros((1, 2), device=device) if is_use_prev_action else None


        # --- 評価ループ ---
        obs, info = env.reset()
        done = False

        # 初期スキャンの登録
        scan = obs['scans'][0]

        while not done:
            # 行動選択（エージェントとプランナ）
            actions = []
            with torch.no_grad():
                # スキャンをグラフに変換
                scan = obs['scans'][0]
                scan_tensor = torch.from_numpy(scan).float().to(device) / cfg.envs.max_beam_range # [1080]
        
                # モデルの入力形状 (B, C, L) に合わせる
                scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, 1080]

                if is_rnn:

                    if is_use_prev_action:
                        action, hidden_state = model(scan_tensor, pre_action=prev_action, hidden=prev_hidden_state)
                        prev_action = action
                    else:
                        action, hidden_state = model(scan_tensor, hidden=prev_hidden_state)

                    prev_hidden_state = hidden_state
                else:
                
                    if is_use_prev_action:
                        action = model(scan_tensor, prev_action)
                        prev_action = action
                    else:
                        action = model(scan_tensor)

                steer, speed = action[0].tolist()


                
                # action = model(scan_tensor).squeeze(0).cpu().numpy()
                steer = steer 
                speed = speed * cfg.envs.map.speed # speed
                action = [steer, speed]
                actions.append(action) 

            # 環境のステップ
            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
            terminated = obs['lap_counts'][0] == 1
            truncated = obs["collisions"][0]
            done = terminated or truncated


            # --- CSVへの書き込み ---
            current_pos = info['current_pos']   # [x, y]
            velocity = info['velocity']         # float
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_pos[0], current_pos[1], velocity])

            if done:
                if truncated:
                    with open(lap_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([0, 0.0])
                elif terminated:
                    lap_time = obs['lap_times']
                    with open(lap_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([1, lap_time])
                
                break

            if cfg.render:
                env.render(cfg.render_mode)

            obs = next_obs
    env.close()

if __name__ == "__main__":
    main()