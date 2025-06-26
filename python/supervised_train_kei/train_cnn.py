import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime


from src.models.SAC import Actor
from src.data.dataset.dataset import HybridLoader 
from src.data.dataset.transform import SeqToSeqTransform, StreamAugmentor
from src.models.layers.state_manager import RnnStateManager 


def load_sac_checkpoint(path, device):
    """SACチェックポイント全体をロードする"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    print(f"Loaded SAC checkpoint with keys: {list(checkpoint.keys())}")
    return checkpoint


def create_actor_from_checkpoint(checkpoint, downsample_num, action_dim, hidden_dim, device):
    """チェックポイントからActorモデルを作成"""
    model = Actor(lidar_dim=downsample_num,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                freeze_feature_layers=True)
    
    if 'actor_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['actor_state_dict'])
        print("Loaded actor from 'actor_state_dict'")
    elif 'actor' in checkpoint:
        model.load_state_dict(checkpoint['actor'])
        print("Loaded actor from 'actor'")
    else:
        available_keys = list(checkpoint.keys())
        raise KeyError(f"No actor state found in checkpoint. Available keys: {available_keys}")
        
    model.to(device)
    return model


def save_sac_checkpoint(original_checkpoint, model, optimizer, epoch, loss, save_path):
    """SAC形式でチェックポイントを保存"""
    # 元のSACチェックポイント構造をコピー
    updated_checkpoint = {}
    
    # 元のチェックポイントの内容をコピー（深いコピーではなく参照コピー）
    for key, value in original_checkpoint.items():
        if key != 'actor_state_dict' and key != 'actor_optimizer':
            updated_checkpoint[key] = value
    
    # Actor関連を更新
    updated_checkpoint['actor_state_dict'] = model.state_dict()
    updated_checkpoint['actor_optimizer'] = optimizer.state_dict()
    
    # ファインチューニング情報を追加
    updated_checkpoint['fine_tune_epoch'] = epoch
    updated_checkpoint['fine_tune_loss'] = loss
    updated_checkpoint['original_checkpoint_path'] = original_checkpoint.get('checkpoint_path', 'unknown')
    
    torch.save(updated_checkpoint, save_path)
    return updated_checkpoint


@hydra.main(config_path="config", config_name="train_cnn", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("=== SAC Fine-tuning Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("=====================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    print(f"Data path: {data_path}")
    
    # --- データセットとデータローダーの準備 ---
    print("\n--- Preparing Data Loaders ---")
    
    # 1. RandomDataset用の変換Transform（データ拡張あり）
    transform_random = SeqToSeqTransform(
        range_max=cfg.range_max,
        downsample_num=cfg.input_dim,
        augment=True,  # 確率的なデータ拡張を有効化
        flip_prob=cfg.flip_prob,  # configから設定可能に
        noise_std=cfg.noise_std
    )

    # 2. StreamDataset用のベース変換Transform（データ拡張なし）
    #    正規化やクリッピングなど、常に適用する決定的な処理のみ行います。
    transform_stream = SeqToSeqTransform(
        range_max=cfg.range_max,
        downsample_num=cfg.input_dim,
        augment=False  # 確率的なデータ拡張は無効化
    )

    # 3. StreamDataset用のAugmentorを作成（エピソード単位の拡張あり）
    #    エピソード単位での確率的なデータ拡張を行います。
    augmentor_stream = StreamAugmentor(
        augment=True,
        flip_prob=cfg.flip_prob,
        noise_std=cfg.noise_std
    )

    # 4. HybridLoaderを新しいインタフェースで初期化
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
    
    print(f"Data loader prepared with batch size: {cfg.batch_size}")
    
    # --- SAC チェックポイントをロード ---
    print("\n--- Loading SAC Checkpoint ---")
    ckpt_path = "/home/ktr/rl_ws/ckpts/sac/TAL/2025-06-16/19-13-34/best_model_reward_291.18.pth"
    print(ckpt_path)
    
    try:
        # 元のSACチェックポイント全体をロード
        original_checkpoint = load_sac_checkpoint(ckpt_path, device)
        
        # Actorモデルを作成
        downsample_num = 100
        action_dim = 2
        hidden_dim = 256
        
        model = create_actor_from_checkpoint(
            original_checkpoint, 
            downsample_num, 
            action_dim, 
            hidden_dim, 
            device,
            
        )
        
        print(f"Successfully loaded Actor model from SAC checkpoint")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Error loading SAC checkpoint: {e}")
        raise
    
    # --- 損失関数と最適化手法の準備 ---
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    print(f"Using SmoothL1Loss with Adam optimizer (lr={cfg.lr})")

    # --- チェックポイントと早期終了の準備 ---
    save_path = cfg.ckpt_path
    os.makedirs(save_path, exist_ok=True)
    
    # 日付時間フォルダを作成
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    datetime_save_path = os.path.join(save_path, f"fine_tuning_{timestamp}")
    os.makedirs(datetime_save_path, exist_ok=True)
    
    early_stop_epochs = cfg.early_stop_epochs
    patience_counter = 0
    top_k = 3
    top_k_checkpoints = []
    
    print(f"Checkpoints will be saved to: {datetime_save_path}")
    print(f"Early stopping patience: {early_stop_epochs}")

    # -------------------
    # ファインチューニングループ
    # -------------------
    print(f"\n=== Start Fine-tuning: {cfg.model_name} ===")
    
    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            # --- バッチデータの取得 ---
            scan_seq = batch['scan_seq'].to(device)  # 2D LiDARのスキャン
            
            target_seq = batch['target_action_seq'].to(device)
            
            # --- モデルに入力する ---        
            _, _, action = model.sample(scan_seq)

            # ターゲットの形状を調整
            if action.dim() == 2:
                target = target_seq[:, -1, :]  # 最後のタイムステップのアクション
            else:
                raise ValueError(f"Unsupported output shape: {action.shape}")

            # 損失計算とバックプロパゲーション
            loss = criterion(action, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1

        # エポック終了時の処理
        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1:3d}/{cfg.num_epochs}] Loss: {avg_loss:.6f} (Batches: {batch_count})")

        # Top-k チェックポイント保存（SAC形式で保存）
        if len(top_k_checkpoints) < top_k or avg_loss < top_k_checkpoints[-1][0]:
            checkpoint_filename = f'finetuned_epoch_{epoch+1:03d}_loss_{avg_loss:.6f}.pth'
            checkpoint_path = os.path.join(datetime_save_path, checkpoint_filename)  # 日付時間フォルダ下に保存
            
            try:
                # SAC形式でチェックポイントを保存
                updated_checkpoint = save_sac_checkpoint(
                    original_checkpoint=original_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=avg_loss,
                    save_path=checkpoint_path
                )
                
                print(f"  [✔] Saved fine-tuned SAC model: {checkpoint_filename}")
                print(f"      📁 Location: {datetime_save_path}")
                
                # Top-k管理
                top_k_checkpoints.append((avg_loss, epoch + 1, checkpoint_path))
                top_k_checkpoints.sort(key=lambda x: x[0])  # 損失でソート
                
                # 古いチェックポイントを削除
                if len(top_k_checkpoints) > top_k:
                    worst_checkpoint = top_k_checkpoints.pop()
                    if os.path.exists(worst_checkpoint[2]):
                        os.remove(worst_checkpoint[2])
                        print(f"  [🗑] Removed old model: {os.path.basename(worst_checkpoint[2])}")
                
                patience_counter = 0  # 改善があったのでカウンターリセット
                
            except Exception as e:
                print(f"  [❌] Error saving checkpoint: {e}")
                
        else:
            patience_counter += 1
            best_loss = top_k_checkpoints[0][0] if top_k_checkpoints else float('inf')
            print(f"  [!] No improvement over best loss ({best_loss:.6f}). Patience: {patience_counter}/{early_stop_epochs}")
            
            # 早期終了チェック
            if patience_counter >= early_stop_epochs:
                print(f"[⏹] Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"\n=== Fine-tuning Completed ===")
    print("Top fine-tuned models saved:")
    for i, (loss_val, epoch_num, path) in enumerate(top_k_checkpoints, 1):
        print(f"  {i}. Epoch: {epoch_num:3d}, Loss: {loss_val:.6f}")
        print(f"     Path: {path}")
    
    # 最終統計
    if top_k_checkpoints:
        best_loss = top_k_checkpoints[0][0]
        best_epoch = top_k_checkpoints[0][1]
        print(f"\nBest model: Epoch {best_epoch}, Loss: {best_loss:.6f}")
    
    print("Fine-tuning finished successfully!")


if __name__ == "__main__":
    main()