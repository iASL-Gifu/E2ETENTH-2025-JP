import torch
from torch.utils.data import DataLoader
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.models import load_gnn_model
from src.data.dataset.lidar_dataset import LidarSeqDataset
from src.data.dataset.transform import E2ESeqTransform
from src.data.graph.graph import build_batch_graph_cuda
from src.utils.timers import Timer

@hydra.main(config_path="config", config_name="train_gnn", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Hydraによって設定ファイルを読み込み、GNNモデルの学習を行うメイン関数。
    時系列・非時系列モデルを単一のロジックで扱う。
    """
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- ハイパーパラメータ等の設定 ---
    sequence_length = cfg.sequence_length
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    lr = cfg.lr
    early_stop_epochs = cfg.early_stop_epochs
    
    # --- データセットとデータローダーの準備 ---
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    transform = E2ESeqTransform(range_max=cfg.range_max, base_num=1081, downsample_num=cfg.input_dim)
    train_dataset = LidarSeqDataset(root_dir=data_path, sequence_length=sequence_length, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # --- モデルの動的ロード ---
    model = load_gnn_model(model_name=cfg.model_name,
                           input_dim=cfg.input_dim,
                           hidden_dim=cfg.hidden_dim,
                           output_dim=cfg.output_dim,
                           pool_method=cfg.pool_method)
    model.to(device)
    print(f"Loaded model: {cfg.model_name}")

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- モデル保存とEarly Stoppingの準備 ---
    patience_counter = 0
    top_k = 3
    top_k_checkpoints = []
    save_path = cfg.ckpt_path
    os.makedirs(save_path, exist_ok=True)
    
    # -------------------
    # 学習ループ
    # -------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            scan_seq = batch['scan_seq'].to(device) # [B, T, N]
            
            # ------------------------------------------------------------------
            # ★★★ 修正箇所: 処理を統一 ★★★
            # 常にグラフのリストを作成する。
            # sequence_length=1 の場合は、要素数1のリストが生成される。
            # ------------------------------------------------------------------
            data_sequence = []
            for t in range(scan_seq.size(1)): # scan_seq.size(1) は sequence_length
                scans_at_t = scan_seq[:, t, :] # 時刻tのLiDARデータ [B, N]
                
                with Timer(timer_name="build_batch_graph_cuda"):
                    graph_batch_at_t = build_batch_graph_cuda(
                        scans_at_t, 
                        distance_threshold=cfg.distance_threshold,
                        max_edges=cfg.max_edges,
                    )
                    data_sequence.append(graph_batch_at_t)

            # モデルへの入力は、常にグラフのリスト
            output = model(data_sequence)
            
            # モデルの出力テンソルの次元数に応じてターゲットの形状を変える
            if output.dim() == 3:
                # --- 出力が3次元 (B, T, F) の場合: 時系列モデルと判断 ---
                target_steer = batch['steer_seq']
                target_speed = batch['speed_seq']
                target = torch.stack([target_steer, target_speed], dim=2).to(device)
            
            elif output.dim() == 2:
                # --- 出力が2次元 (B, F) の場合: 非時系列モデルと判断 ---
                target_steer = batch['steer_seq'][:, -1]
                target_speed = batch['speed_seq'][:, -1]
                target = torch.stack([target_steer, target_speed], dim=1).to(device)

            else:
                # 想定外の形状の場合はエラーを出す
                raise ValueError(f"Unsupported output shape: {output.shape}")


            # 損失計算とパラメータ更新
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # --- モデルの保存 & Early Stopping ---
        if len(top_k_checkpoints) < top_k or avg_loss < top_k_checkpoints[-1][0]:
            checkpoint_path = os.path.join(save_path, f'model_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [✔] Saved new top-k model (loss={avg_loss:.4f})")
            
            top_k_checkpoints.append((avg_loss, epoch + 1, checkpoint_path))
            top_k_checkpoints.sort(key=lambda x: x[0])
            
            if len(top_k_checkpoints) > top_k:
                worst_checkpoint = top_k_checkpoints.pop()
                if os.path.exists(worst_checkpoint[2]):
                    os.remove(worst_checkpoint[2])
                    print(f"  [🗑] Removed old model: {os.path.basename(worst_checkpoint[2])}")
            patience_counter = 0
        else:
            patience_counter += 1
            best_loss = top_k_checkpoints[0][0] if top_k_checkpoints else float('inf')
            print(f"  [!] No improvement over best loss ({best_loss:.4f}). Patience: {patience_counter}/{early_stop_epochs}")
            if patience_counter >= early_stop_epochs:
                print("[⏹] Early stopping triggered.")
                break

    print("\n--- Training Finished ---")
    print("Top models saved:")
    for loss_val, epoch_num, path in top_k_checkpoints:
        print(f"  - Epoch: {epoch_num}, Loss: {loss_val:.4f}, Path: {path}")


if __name__ == "__main__":
    main()