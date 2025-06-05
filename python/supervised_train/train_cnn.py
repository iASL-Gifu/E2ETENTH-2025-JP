import torch
from torch.utils.data import DataLoader
import os
from omegaconf import OmegaConf

from src.models.models import load_cnn_model  
from src.data.dataset.lidar_dataset import LidarSeqDataset
from src.data.dataset.transform import E2ESeqTransform

config = OmegaConf.load("./config/train_cnn.yaml") 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -------------------
# ハイパーパラメータと設定
# -------------------
sequence_length = config.sequence_length
downsample_scan_num = config.input_dim
range_max = config.range_max
batch_size = config.batch_size
num_epochs = config.num_epochs
lr = config.lr

# 早期終了パラメータ
early_stop_epochs = config.early_stop_epochs
patience_counter = 0

# --- Top-Kチェックポイントのための設定 ---
top_k = 3
top_k_checkpoints = [] 

# モデル保存ディレクトリ
save_path = config.ckpt_path
os.makedirs(save_path, exist_ok=True)

# -------------------
# データセットとローダー
# -------------------
transform = E2ESeqTransform(range_max=range_max, base_num=1081, downsample_num=downsample_scan_num)
train_dataset = LidarSeqDataset(root_dir=config.data_path, sequence_length=sequence_length, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------
# モデルと損失関数
# -------------------
model = load_cnn_model(model_name=config.model_name ,input_dim=config.input_dim, output_dim=config.output_dim).to(device)
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
        
        # steer_seq, speed_seqの形状は (batch_size, sequence_length)
        target_steer = batch['steer_seq'][:, -1] # 各バッチのシーケンスの最後の値を取得
        target_speed = batch['speed_seq'][:, -1] # 各バッチのシーケンスの最後の値を取得
        target = torch.stack([target_steer, target_speed], dim=1).to(device)
        
        # モデルはシーケンス全体を入力とし、最後のタイムステップに対する予測値を出力することを想定
        output = model(scan_seq)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    # --- 早期終了とTop-Kチェックポイント (この部分は変更なし) ---
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