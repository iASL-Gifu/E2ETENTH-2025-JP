import torch
from torch.utils.data import DataLoader
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# 外部ファイルのimportを新しいものに更新
from src.models.models import load_cnn_model # ご自身のモデル読み込み関数に置き換えてください
from src.data.dataset.lidar_dataset import LidarSeqToSeqDataset
from src.data.dataset.transform import SeqToSeqTransform

@hydra.main(config_path="config", config_name="train_cnn", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Hydraによって設定ファイルを読み込み、モデルの学習を行うメイン関数。
    prev_actionの使用有無や、モデルの出力形式に柔軟に対応する。
    """
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- データセットとデータローダーの準備 ---
    # 新しいTransformとDatasetを使用
    transform = SeqToSeqTransform(
        range_max=cfg.range_max, 
        base_num=1081, # Lidarの元の点群数に合わせて調整
        downsample_num=cfg.input_dim
    )
    # to_absolute_pathで絶対パスに変換
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    train_dataset = LidarSeqToSeqDataset(
        root_dir=data_path, 
        sequence_length=cfg.sequence_length, 
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # --- モデル、損失関数、最適化手法の準備 ---
    # モデル名は設定ファイルから取得
    model = load_cnn_model(
        model_name=cfg.model_name, 
        input_dim=cfg.input_dim, 
        output_dim=cfg.output_dim
    ).to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # --- チェックポイントと早期終了の準備 (変更なし) ---
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

        for batch in train_loader:
            # 必要なデータをすべてデバイスに送る
            scan_seq = batch['scan_seq'].to(device)
            prev_action_seq = batch['prev_action_seq'].to(device)
            target_seq = batch['target_action_seq'].to(device)

            # --- モデルへの入力を動的に切り替え ---
            if cfg.model_use_prev_action:
                # prev_action を使うモデルの場合
                output = model(scan_seq, prev_action_seq)
            else:
                # prev_action を使わないモデルの場合 (LiDARシーケンスのみ入力)
                output = model(scan_seq)

            # --- モデルの出力形状に合わせて教師データの形状を整形 ---
            if output.dim() == 3 and output.shape[1] > 1:
                # 出力が時系列 [B, SeqLen, F] の場合 -> 教師データも時系列に
                target = target_seq
            elif output.dim() == 2:
                # 出力が非時系列 [B, F] の場合 -> 教師データはシーケンスの最後のフレーム
                target = target_seq[:, -1, :]
            else:
                # その他の想定外の形状
                raise ValueError(f"Unsupported output shape: {output.shape}")

            # 損失計算と逆伝播
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{cfg.num_epochs}] Loss: {avg_loss:.4f}")

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