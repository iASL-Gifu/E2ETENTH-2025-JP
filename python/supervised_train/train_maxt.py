import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models.models import load_maxt_model
from src.data.dataset.dataset import HybridLoader
from src.data.dataset.transform import SeqToSeqTransform, StreamAugmentor
from src.models.layers.state_manager import RnnStateManager

@hydra.main(config_path="config", config_name="train_maxt", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = hydra.utils.to_absolute_path(cfg.data_path)

    # --- データセットとデータローダーの準備 --- 
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
    model = load_maxt_model(size="tiny").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    is_rnn = False
    if is_rnn:
        state_manager = RnnStateManager(device)

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
    print(f"\n--- Start Training: MaxTransformer 1d ---")
    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        if is_rnn:
            state_manager.reset_states(cfg.batch_size)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=False)
        
        for i, batch in enumerate(pbar):
            scan_seq = batch['scan_seq'].to(device)
            ground_truth = batch['target_action_seq'][:, -1, :2].to(device) 

            if is_rnn:
                pass
            else:
                
                result_dict = model(scan_seq, target=ground_truth)
                # 辞書から計算済みの損失を取得
                current_batch_loss = result_dict['loss']
            
            optimizer.zero_grad()
            current_batch_loss.backward()
            optimizer.step()
            running_loss += current_batch_loss.item()
            
            current_avg_loss = running_loss / (i + 1)
            pbar.set_postfix(loss=f"{current_avg_loss:.4f}")

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