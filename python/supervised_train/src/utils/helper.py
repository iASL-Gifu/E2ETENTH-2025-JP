import torch

def get_model_size_in_m(model: torch.nn.Module) -> str:
    """
    モデルのパラメータ数を計算し、M表記（100万単位）の文字列で返す。
    
    Args:
        model (torch.nn.Module): サイズを計算するモデル。

    Returns:
        str: M表記にフォーマットされたモデルサイズ (例: "1.23M")。
    """
    # model.parameters()がトレーニング可能なパラメータのみを返すように、requires_grad=Trueのものをフィルタリングしても良い
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_m = total_params / 1_000_000
    return f"{params_in_m:.2f}M"