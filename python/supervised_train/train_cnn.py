import torch
from torch.utils.data import DataLoader
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# 外部ファイルのimport (パスが通っている必要があります)
from src.models.models import load_cnn_model  
from src.data.dataset.lidar_dataset import LidarSeqDataset
from src.data.dataset.transform import E2ESeqTransform

@hydra.main(version_base=None, config_path="config", config_name="train_cnn")
def main(cfg: DictConfig) -> None:
    """
    Hydraによって設定ファイルを読み込み、モデルの学習を行うメイン関数。
    """
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ... (ハイパーパラメータやデータローダーの設定は変更なし) ...
    sequence_length = cfg.sequence_length
    downsample_scan_num = cfg.input_dim
    range_max = cfg.range_max
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    lr = cfg.lr
    early_stop_epochs = cfg.early_stop_epochs
    patience_counter = 0
    top_k = 3
    top_k_checkpoints = [] 
    save_path = "checkpoints"
    os.makedirs(save_path, exist_ok=True)
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    transform = E2ESeqTransform(range_max=range_max, base_num=1081, downsample_num=downsample_scan_num)
    train_dataset = LidarSeqDataset(root_dir=data_path, sequence_length=sequence_length, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = load_cnn_model(model_name=cfg.model_name, input_dim=cfg.input_dim, output_dim=cfg.output_dim).to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------
    # 学習ループ
    # -------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            scan_seq = batch['scan_seq'].to(device)
            
            # まずモデルの出力を得る
            output = model(scan_seq)
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

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

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