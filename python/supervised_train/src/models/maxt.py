import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from typing import Union, Dict, Optional, Any
import time

from .layers.maxt1d import MaxT1d
from .layers.fpn1d import PAFPN1d
from .layers.head import _MultiScaleHead


def get_model_cfg(
    model_size: str, 
    backbone_stages: int = 4, 
    fpn_stages: int = 4,
) -> DictConfig:
    """ 
    モデルのサイズとステージ数から設定を生成する。
    Stemの出力をステージ数に含めず、C0をStage0の出力に紐付けるように修正。
    headにストライド情報を追加。
    """
    if fpn_stages > backbone_stages:
        raise ValueError("fpn_stages cannot be greater than backbone_stages")

    if model_size == 'tiny':
        dims = [32, 64, 128]
        num_blocks = [1, 1, 1]
        neck_depth = 0.33
        dim_head = 32
    elif model_size == 'small':
        dims = [48, 96, 192]
        num_blocks = [1, 1, 1]
        neck_depth = 0.67
        dim_head = 24
    elif model_size == 'base':
        dims = [64, 128, 256]
        num_blocks = [1, 1, 1]
        neck_depth = 1.0
        dim_head = 32
    elif model_size == 'nano':
        dims = [24, 48, 96]  
        num_blocks = [1, 1, 1]
        neck_depth = 0.25          
        dim_head = 24              
    elif model_size == 'pico':
        dims = [16, 32, 64]   
        num_blocks = [1, 1, 1]
        neck_depth = 0.25
        dim_head = 16              
    else: 
        raise ValueError(f"Unknown model_size: {model_size}")
    
    output_stage_indices = list(range(1, backbone_stages + 1)) # [1, 2, ..., backbone_stages]
    backbone_out_keys = [f'C{i}' for i in output_stage_indices] # [C1, C2, ..., C_backbone_stages]

    active_dims = dims[:backbone_stages]
    active_num_blocks = num_blocks[:backbone_stages]
    
    fpn_in_keys = backbone_out_keys[-fpn_stages:]
    fpn_in_channels_map = {key: active_dims[int(key[1:]) - 1] for key in fpn_in_keys}

    target_seq_len = 1024 # cfg.model.backbone.target_seq_len と一致させる
    
    strides_map: Dict[str, int] = {}
    
    # Stem後の初期ストライドは 2 (シーケンス長が半分になるため)
    current_stride = 2 
    for i, key in enumerate(backbone_out_keys):
        # 各Stage1dもシーケンス長を半分にするため、ストライドが倍になる
        # MaxT1dのC1はStage0の出力、C2はStage1の出力...に対応
        # Stage i の出力は、Stem + (i+1)個のDownsampleLayerによって生成される
        # C1 (Stage0の出力) -> 2 * 2 = 4
        # C2 (Stage1の出力) -> 4 * 2 = 8
        # ...
        current_stride *= 2 
        strides_map[key] = current_stride
    
    cfg = {
        'model': {
            'backbone': {
                'features_only': True, 'input_channels': 1, 'target_seq_len': target_seq_len, # target_seq_lenを明示的に使用
                'partition_ratio': 16, 'drop_path_rate': 0.1, 'dims': active_dims,
                'num_blocks': active_num_blocks, 
                'out_indices': output_stage_indices, 
                'out_keys': backbone_out_keys,       
                'dim_head': dim_head,
            },
            'neck': { 
                'depthwise': False, 'act': 'silu', 'depth': neck_depth,
                'in_keys': fpn_in_keys, 
                'in_channels': fpn_in_channels_map,
            },
            'head': { 
                'name': 'MultiScaleRegressionHead',
                'out_features': 2, 
                'mid_features_ratio': 0.5,
                'in_channels': fpn_in_channels_map,
                'strides_map': strides_map,
            }
        }
    }
    
    return OmegaConf.create(cfg)

class LidarRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        
        self.backbone = MaxT1d(cfg.model.backbone)
        self.neck = PAFPN1d(cfg.model.neck)
        self.head = _MultiScaleHead(cfg.model.head)
        
        # 損失関数をクラス内で定義
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        常に辞書を返すように統一。
        - target が与えられた場合 (学習時): {'output': ..., 'loss': ...} を返す
        - target が None の場合 (評価時): {'output': ...} のみを返す
        """
        # --- 1. 予測値の計算 ---
        all_features = self.backbone(x)
        neck_input = {key: all_features[key] for key in self.neck.in_keys}
        neck_features = self.neck(neck_input)
        
        # predictions_dict は {'C1': tensor, 'C2': tensor, ...} の形式
        predictions_dict = self.head(neck_features)

        # --- 2. 戻り値の構築 ---
        
        # 評価時は、予測値を 'output' キーに入れて返す
        if target is None:
            return {'output': predictions_dict}

        # --- 学習時は、以下で損失を計算して追加 ---
        
        total_loss = 0.0
        num_predictions = len(predictions_dict)

        if num_predictions > 0:
            for pred_tensor in predictions_dict.values():
                total_loss += self.criterion(pred_tensor, target)
            calculated_loss = total_loss / num_predictions
        else:
            calculated_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        # 学習時は、予測値と損失の両方を辞書で返す
        return {
            'output': predictions_dict,
            'loss': calculated_loss
        }
    
if __name__ == "__main__":
    print("--- LidarRegressor Model Test ---")

    # モデルサイズの選択
    model_size = 'tiny'
    backbone_stages = 4
    fpn_stages = 3

    print(f"Testing with model_size='{model_size}', backbone_stages={backbone_stages}, fpn_stages={fpn_stages}")

    try:
        # 1. 設定の生成
        cfg = get_model_cfg(model_size, backbone_stages, fpn_stages)
        print("\n--- Generated Config ---")
        print(OmegaConf.to_yaml(cfg))
        print("-------------------------")

        # 2. モデルの初期化
        model = LidarRegressor(cfg)
        print("\n--- Model Initialized ---")
        
        # モデルをGPUに移動（利用可能な場合）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # 推論モードに設定
        print(f"Model moved to: {device}")

        # パラメータ数の表示をM表示に修正
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params / 1_000_000:.2f} M")

        # 3. ダミーデータの生成
        batch_size = 1
        input_channels = 1
        # get_model_cfgでtarget_seq_lenが1024に設定されているため、それに合わせる
        target_seq_len = cfg.model.backbone.target_seq_len 
        dummy_input = torch.randn(batch_size, input_channels, target_seq_len).to(device)
        print(f"\n--- Dummy Input Shape: {dummy_input.shape} ---")

        # 4. フォワードパスの実行 (初回実行)
        print("\n--- Running Forward Pass (Initial Check) ---")
        with torch.no_grad():
            output = model(dummy_input)

        # 5. 出力形状と対応するストライドの確認
        print("\n--- Model Output Shape and Stride ---")
        print(f"Output keys: {output}")
        head_strides_map = model.head.strides_map # _MultiScaleHeadからストライドマップを取得
        for k, v in output.items():
            stride = head_strides_map.get(k, 'N/A') # 各出力キーに対応するストライドを取得
            print(f"  {k}: Shape {v.shape}, Stride: {stride}")

        # --- 推論速度の測定 ---
        print("\n--- Measuring Inference Speed ---")
        num_warmup = 10  # ウォームアップ回数
        num_runs = 100   # 測定回数

        # ウォームアップ
        print(f"Warming up for {num_warmup} runs...")
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # GPUを使用している場合は同期
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 測定開始
        print(f"Measuring inference speed over {num_runs} runs...")
        start_time = time.perf_counter() # 高精度な時間計測

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)

        # GPUを使用している場合は同期
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_inference = total_time / num_runs * 1000 # ミリ秒に変換
        fps = 1000 / avg_time_per_inference if avg_time_per_inference > 0 else float('inf')

        print(f"\nTotal inference time for {num_runs} runs: {total_time:.4f} seconds")
        print(f"Average time per inference: {avg_time_per_inference:.4f} ms")
        print(f"Frames per second (FPS): {fps:.2f}")

    except Exception as e:
        print(f"\n--- An Error Occurred ---")
        print(f"Error: {e}")

    print("\n--- Test Finished ---")
