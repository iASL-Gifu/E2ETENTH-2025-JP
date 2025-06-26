import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# PyG (PyTorch Geometric) のバッチオブジェクトを使用
from torch_geometric.data import Batch

from src.models.models import load_gnn_model 
from src.data.dataset.dataset import HybridLoader 
from src.data.dataset.transform import SeqToSeqTransform, StreamAugmentor
from src.data.graph.graph import build_batch_graph_cuda
# コンパイルされたCUDAモジュールも必要
import lidar_graph_cuda

@hydra.main(config_path="config", config_name="train_gnn", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    GNNモデルの学習を実行するメイン関数。
    LiDARスキャンデータをCUDAでグラフに変換してからモデルに入力する。
    """
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    
    # --- データセットとデータローダーの準備 ---
    # ユーザーの想定通り、sequence_length=1 に設定されていることが前提
    if cfg.sequence_length != 1:
        raise ValueError("This script assumes `sequence_length=1` to process data as a graph.")

    transform_random = SeqToSeqTransform(
        range_max=cfg.range_max,
        downsample_num=cfg.input_dim,
        augment=True,
        flip_prob=cfg.flip_prob,
        noise_std=cfg.noise_std
    )

    transform_stream = SeqToSeqTransform(
        range_max=cfg.range_max,
        downsample_num=cfg.input_dim,
        augment=False
    )

    augmentor_stream = StreamAugmentor(
        augment=True,
        flip_prob=cfg.flip_prob,
        noise_std=cfg.noise_std
    )

    train_loader = HybridLoader(
        root_dir=data_path,
        sequence_length=cfg.sequence_length,
        total_batch_size=cfg.batch_size,  
        random_ratio=cfg.random_ratio,    
        transform_random=transform_random,
        transform_stream=transform_stream,
        augmentor_stream=augmentor_stream,
        num_workers_random=cfg.num_workers 
    )
    
    # --- モデル、損失関数、最適化手法の準備 ---
    # 注: load_gnn_model はPyTorch GeometricのBatchオブジェクトを受け付けるモデルを返す必要があります
    model = load_gnn_model(
        model_name=cfg.model_name, 
        input_dim=cfg.input_dim, 
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        pool_method=cfg.pool_method,
    ).to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # --- チェックポイントと早期終了の準備 ---
    save_path = cfg.ckpt_path
    os.makedirs(save_path, exist_ok=True)
    early_stop_epochs = cfg.early_stop_epochs
    patience_counter = 0
    top_k = 3
    top_k_checkpoints = [] 

    # -------------------
    # 学習ループ
    # -------------------
    print(f"\n--- Start Training: {cfg.model_name} ---")
    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader), 
            desc=f"Epoch {epoch+1}/{cfg.num_epochs}"
        )
        
        for i, batch in progress_bar:
            scan_seq = batch['scan_seq'].to(device) # Shape: [B, 1, NumPoints]
            target_seq = batch['target_action_seq'].to(device) # Shape: [B, 1, OutputDim]
            
            # --- 変更点: グラフ構築 ---
            # 1. 時系列の次元を削除 (B, 1, NumPoints) -> (B, NumPoints)
            scan_batch = scan_seq.squeeze(1)

            # 2. CUDA関数でバッチグラフを構築
            #    cfgからグラフ構築用のパラメータを取得
            graph_batch = build_batch_graph_cuda(
                scan_data_batch=scan_batch,
                distance_threshold=cfg.distance_threshold,
                max_neighbors=cfg.max_neighbors
            )
            
            # 3. 構築したグラフを学習デバイスに配置
            #    build_batch_graph_cuda はGPU上でテンソルを作成するが、念のためdeviceに送る
            graph_batch = graph_batch.to(device)

            # --- モデルへの入力と順伝播 ---
            # モデルへはグラフバッチオブジェクトを渡す
            action = model(graph_batch)

            # --- 損失計算 ---
            # ターゲットも時系列の次元を削除 (B, 1, D) -> (B, D)
            target = target_seq.squeeze(1)

            # モデル出力とターゲットの形状が一致しているはず
            loss = criterion(action, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{running_loss / (i + 1):.4f}")

        avg_loss = running_loss / len(train_loader)
        tqdm.write(f"[Epoch {epoch+1}/{cfg.num_epochs}] Loss: {avg_loss:.4f}")

        # --- モデルの保存と早期終了の判定 ---
        if len(top_k_checkpoints) < top_k or avg_loss < top_k_checkpoints[-1][0]:
            checkpoint_path = os.path.join(save_path, f'model_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"  [✔] Saved new top-k model (loss={avg_loss:.4f})")
            top_k_checkpoints.append((avg_loss, epoch + 1, checkpoint_path))
            top_k_checkpoints.sort(key=lambda x: x[0])
            if len(top_k_checkpoints) > top_k:
                worst_checkpoint = top_k_checkpoints.pop()
                if os.path.exists(worst_checkpoint[2]):
                    os.remove(worst_checkpoint[2])
                    tqdm.write(f"  [🗑] Removed old model: {os.path.basename(worst_checkpoint[2])}")
            patience_counter = 0
        else:
            patience_counter += 1
            best_loss = top_k_checkpoints[0][0] if top_k_checkpoints else float('inf')
            tqdm.write(f"  [!] No improvement over best loss ({best_loss:.4f}). Patience: {patience_counter}/{early_stop_epochs}")
            if patience_counter >= early_stop_epochs:
                tqdm.write("[⏹] Early stopping triggered.")
                break

    print("\n--- Training Finished ---")
    print("Top models saved:")
    for loss_val, epoch_num, path in top_k_checkpoints:
        print(f"  - Epoch: {epoch_num}, Loss: {loss_val:.4f}, Path: {path}")

if __name__ == "__main__":
    main()