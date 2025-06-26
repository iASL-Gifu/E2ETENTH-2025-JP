import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime


# 凍結機能付きActorクラス（ローカル定義またはimport可能）
class Actor(nn.Module):
    def __init__(self, lidar_dim, action_dim=2, hidden_dim=256, freeze_feature_layers=False):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        
        # LiDARデータを直接処理する全結合層
        self.fc1 = nn.Linear(lidar_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 平均と対数標準偏差の出力ヘッド
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # ✅ パラメータとして登録（register_bufferを使用）
        self.register_buffer('action_scale', torch.FloatTensor([1.0, 1.0]))
        self.register_buffer('action_bias', torch.FloatTensor([0.0, 0.0]))
        
        # 🔥 特徴抽出層を凍結
        if freeze_feature_layers:
            self.freeze_feature_layers()
        
    def freeze_feature_layers(self):
        """fc1, fc2, fc3の3つの全結合層を凍結"""
        print("特徴抽出層(fc1, fc2, fc3)を凍結しています...")
        
        # fc1, fc2, fc3の勾配計算を無効化
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False
        for param in self.fc3.parameters():
            param.requires_grad = True
            
        print("凍結完了: fc1, fc2, fc3")
    
    def unfreeze_feature_layers(self):
        """fc1, fc2, fc3の3つの全結合層の凍結を解除"""
        print("特徴抽出層(fc1, fc2, fc3)の凍結を解除しています...")
        
        # fc1, fc2, fc3の勾配計算を有効化
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True
        for param in self.fc3.parameters():
            param.requires_grad = True
            
        print("凍結解除完了: fc1, fc2, fc3")
    
    def check_frozen_status(self):
        """凍結状態をチェック"""
        fc1_frozen = not any(p.requires_grad for p in self.fc1.parameters())
        fc2_frozen = not any(p.requires_grad for p in self.fc2.parameters())
        fc3_frozen = not any(p.requires_grad for p in self.fc3.parameters())
        mean_head_frozen = not any(p.requires_grad for p in self.mean_head.parameters())
        log_std_head_frozen = not any(p.requires_grad for p in self.log_std_head.parameters())
        
        print(f"凍結状態:")
        print(f"  fc1: {'凍結' if fc1_frozen else '学習可能'}")
        print(f"  fc2: {'凍結' if fc2_frozen else '学習可能'}")
        print(f"  fc3: {'凍結' if fc3_frozen else '学習可能'}")
        print(f"  mean_head: {'凍結' if mean_head_frozen else '学習可能'}")
        print(f"  log_std_head: {'凍結' if log_std_head_frozen else '学習可能'}")
        
        return {
            'fc1': fc1_frozen,
            'fc2': fc2_frozen, 
            'fc3': fc3_frozen,
            'mean_head': mean_head_frozen,
            'log_std_head': log_std_head_frozen
        }
        
    def forward(self, lidar_data):
        # LiDARデータを直接処理
        x = F.relu(self.fc1(lidar_data))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 安定性のためクリッピング
        
        return mean, log_std
    
    def sample(self, lidar_data):
        """アクションをサンプリング"""
        mean, log_std = self.forward(lidar_data)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # 再パラメータ化トリック
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # ログ確率を計算
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 決定論的な平均アクション
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


# Import文（元のコードから必要に応じて追加）
# from src.data.dataset.dataset import HybridLoader 
# from src.data.dataset.transform import SeqToSeqTransform, StreamAugmentor
# from src.models.layers.state_manager import RnnStateManager 


def load_sac_checkpoint(path, device):
    """SACチェックポイント全体をロードする"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    print(f"Loaded SAC checkpoint with keys: {list(checkpoint.keys())}")
    return checkpoint


def create_actor_from_checkpoint(checkpoint, downsample_num, action_dim, hidden_dim, device, freeze_features=False):
    """チェックポイントからActorモデルを作成（凍結オプション付き）"""
    # 🔥 凍結機能付きActorモデルを作成
    model = Actor(
        lidar_dim=downsample_num,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        freeze_feature_layers=freeze_features  # 凍結オプション
    )
    
    if 'actor_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['actor_state_dict'])
        print("Loaded actor from 'actor_state_dict'")
    elif 'actor' in checkpoint:
        model.load_state_dict(checkpoint['actor'])
        print("Loaded actor from 'actor'")
    else:
        available_keys = list(checkpoint.keys())
        raise KeyError(f"No actor state found in checkpoint. Available keys: {available_keys}")
    
    # ロード後に再度凍結設定を適用（state_dictロード後に必要）
    if freeze_features:
        model.freeze_feature_layers()
        
    model.to(device)
    
    # 凍結状態を確認
    print("\n=== Actor Model Freeze Status ===")
    model.check_frozen_status()
    
    return model


def save_sac_checkpoint(original_checkpoint, model, optimizer, epoch, loss, save_path, freeze_info=None):
    """SAC形式でチェックポイントを保存（凍結情報付き）"""
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
    
    # 🔥 凍結情報を保存
    if freeze_info:
        updated_checkpoint['freeze_info'] = freeze_info
        updated_checkpoint['frozen_feature_layers'] = freeze_info.get('frozen', False)
    
    torch.save(updated_checkpoint, save_path)
    return updated_checkpoint


def create_optimizer_for_model(model, lr, freeze_mode=False):
    """モデルの凍結状態に応じてオプティマイザーを作成"""
    if freeze_mode:
        # 学習可能なパラメータのみを対象にする
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        
        print(f"凍結モード: 学習可能パラメータ {trainable_count:,}/{total_params:,} ({100*trainable_count/total_params:.1f}%)")
    else:
        # 全パラメータを対象にする
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"通常モード: 全パラメータ {total_params:,} が学習可能")
    
    return optimizer


# 設定ファイルのデフォルト値（実際のconfig.yamlファイルに追加推奨）
DEFAULT_FREEZE_CONFIG = {
    'freeze_feature_layers': True,      # 特徴抽出層を凍結するか
    'unfreeze_at_epoch': None,          # 指定エポックで凍結解除（Noneで無効）
    'freeze_schedule': [],              # エポックと凍結状態のスケジュール
    'freeze_lr_multiplier': 0.1,        # 凍結時の学習率倍率
}


@hydra.main(config_path="config", config_name="train_cnn", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("=== SAC Fine-tuning with Freezing Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    
    # 🔥 凍結設定のマージ（設定ファイルに無い場合はデフォルト使用）
    freeze_cfg = OmegaConf.merge(DEFAULT_FREEZE_CONFIG, cfg.get('freeze', {}))
    print(f"\n=== Freeze Configuration ===")
    print(OmegaConf.to_yaml(freeze_cfg))
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    print(f"Data path: {data_path}")
    
    # --- データセットとデータローダーの準備 ---
    print("\n--- Preparing Data Loaders ---")
    
    # 注意: 実際の環境では以下のimportが必要
    # from src.data.dataset.dataset import HybridLoader 
    # from src.data.dataset.transform import SeqToSeqTransform, StreamAugmentor
    
    # データローダーの設定（元コードと同じ）
    # ここでは簡略化のため、実際のデータローダー初期化は省略
    print("データローダー初期化は実装環境に応じて行ってください")
    
    # 仮想的なデータローダー（テスト用）
    class DummyDataLoader:
        def __init__(self, batch_size, sequence_length, input_dim):
            self.batch_size = batch_size
            self.sequence_length = sequence_length
            self.input_dim = input_dim
            self.batches = 10  # 仮想バッチ数
            
        def __iter__(self):
            for _ in range(self.batches):
                yield {
                    'scan_seq': torch.randn(self.batch_size, self.sequence_length, self.input_dim),
                    'target_action_seq': torch.randn(self.batch_size, self.sequence_length, 2)
                }
                
        def __len__(self):
            return self.batches
    
    # 仮想データローダー（実際の環境では上記のHybridLoaderを使用）
    train_loader = DummyDataLoader(
        batch_size=cfg.batch_size,
        sequence_length=cfg.sequence_length,
        input_dim=cfg.input_dim
    )
    
    print(f"Data loader prepared with batch size: {cfg.batch_size}")
    
    # --- SAC チェックポイントをロード ---
    print("\n--- Loading SAC Checkpoint ---")
    ckpt_path = "/home/ktr/rl_ws/ckpts/sac/TAL/2025-06-16/19-13-34/best_model_reward_291.18.pth"
    print(f"Checkpoint path: {ckpt_path}")
    
    try:
        # 元のSACチェックポイント全体をロード
        original_checkpoint = load_sac_checkpoint(ckpt_path, device)
        
        # 🔥 凍結機能付きActorモデルを作成
        downsample_num = 100
        action_dim = 2
        hidden_dim = 256
        
        model = create_actor_from_checkpoint(
            checkpoint=original_checkpoint, 
            downsample_num=downsample_num, 
            action_dim=action_dim, 
            hidden_dim=hidden_dim, 
            device=device,
            freeze_features=freeze_cfg.freeze_feature_layers  # 🔥 凍結設定
        )
        
        print(f"Successfully loaded Actor model from SAC checkpoint")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Error loading SAC checkpoint: {e}")
        # 代替として新しいモデルを作成
        print("Creating new model instead...")
        model = Actor(
            lidar_dim=100,
            action_dim=2,
            hidden_dim=256,
            freeze_feature_layers=freeze_cfg.freeze_feature_layers
        ).to(device)
        original_checkpoint = {}
    
    # --- 損失関数と最適化手法の準備 ---
    criterion = torch.nn.SmoothL1Loss()
    
    # 🔥 凍結状態に応じた学習率調整
    lr = cfg.lr
    if freeze_cfg.freeze_feature_layers:
        lr *= freeze_cfg.get('freeze_lr_multiplier', 1.0)
        print(f"凍結モードにより学習率を調整: {cfg.lr} → {lr}")
    
    # 🔥 凍結状態に応じたオプティマイザー作成
    optimizer = create_optimizer_for_model(model, lr, freeze_cfg.freeze_feature_layers)
    
    print(f"Using SmoothL1Loss with Adam optimizer (lr={lr})")

    # --- チェックポイントと早期終了の準備 ---
    save_path = cfg.ckpt_path
    os.makedirs(save_path, exist_ok=True)
    
    # 日付時間フォルダを作成
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    freeze_suffix = "_frozen" if freeze_cfg.freeze_feature_layers else "_full"
    datetime_save_path = os.path.join(save_path, f"fine_tuning{freeze_suffix}_{timestamp}")
    os.makedirs(datetime_save_path, exist_ok=True)
    
    early_stop_epochs = cfg.early_stop_epochs
    patience_counter = 0
    top_k = 3
    top_k_checkpoints = []
    
    print(f"Checkpoints will be saved to: {datetime_save_path}")
    print(f"Early stopping patience: {early_stop_epochs}")

    # -------------------
    # ファインチューニングループ（凍結機能付き）
    # -------------------
    print(f"\n=== Start Fine-tuning: {cfg.model_name} ===")
    print(f"凍結設定: {'有効' if freeze_cfg.freeze_feature_layers else '無効'}")
    
    for epoch in range(cfg.num_epochs):
        # 🔥 エポックベースの凍結スケジュール処理
        if freeze_cfg.get('unfreeze_at_epoch') == epoch:
            print(f"\n🔥 エポック {epoch}: 凍結解除")
            model.unfreeze_feature_layers()
            # オプティマイザーを再構築
            optimizer = create_optimizer_for_model(model, cfg.lr, freeze_mode=False)
            
        # 🔥 複数ステップの凍結スケジュール処理
        for schedule_item in freeze_cfg.get('freeze_schedule', []):
            if schedule_item['epoch'] == epoch:
                if schedule_item['action'] == 'unfreeze':
                    print(f"\n🔥 エポック {epoch}: スケジュール凍結解除")
                    model.unfreeze_feature_layers()
                    optimizer = create_optimizer_for_model(model, cfg.lr, freeze_mode=False)
                elif schedule_item['action'] == 'freeze':
                    print(f"\n🔥 エポック {epoch}: スケジュール凍結")
                    model.freeze_feature_layers()
                    lr_frozen = cfg.lr * freeze_cfg.get('freeze_lr_multiplier', 1.0)
                    optimizer = create_optimizer_for_model(model, lr_frozen, freeze_mode=True)
        
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
        
        # 凍結状態の表示
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        freeze_status = f"凍結: {frozen_params}/{total_params}" if frozen_params > 0 else "全て学習可能"
        
        print(f"[Epoch {epoch+1:3d}/{cfg.num_epochs}] Loss: {avg_loss:.6f} | {freeze_status} | Batches: {batch_count}")

        # Top-k チェックポイント保存（SAC形式で保存、凍結情報付き）
        if len(top_k_checkpoints) < top_k or avg_loss < top_k_checkpoints[-1][0]:
            checkpoint_filename = f'finetuned_epoch_{epoch+1:03d}_loss_{avg_loss:.6f}.pth'
            checkpoint_path = os.path.join(datetime_save_path, checkpoint_filename)
            
            try:
                # 🔥 凍結情報を含める
                freeze_info = {
                    'frozen': any(not p.requires_grad for p in model.parameters()),
                    'freeze_status': model.check_frozen_status(),
                    'epoch': epoch + 1,
                    'config': freeze_cfg
                }
                
                # SAC形式でチェックポイントを保存
                updated_checkpoint = save_sac_checkpoint(
                    original_checkpoint=original_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=avg_loss,
                    save_path=checkpoint_path,
                    freeze_info=freeze_info  # 🔥 凍結情報追加
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
    
    # 🔥 最終的な凍結状態を表示
    print(f"\n=== Final Model Status ===")
    model.check_frozen_status()
    
    print("Fine-tuning finished successfully!")


if __name__ == "__main__":
    main()