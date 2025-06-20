try:
    from torch import compile as th_compile
    print("[*] torch.compile is available.")
except ImportError:
    th_compile = None
    print("[!] torch.compile is NOT available. (PyTorch < 2.0 or missing dependencies)")

from .gnn import LidarGCN, LidarGAT, LidarGcnLstmNet, LidarGatLstmNet

from .cnn import (
    TinyLidarNet,
    TinyLidarLstmNet,
    TinyLidarConvLstmNet,
    TinyLidarActionNet,
    TinyLidarActionLstmNet,
    TinyLidarActionConvLstmNet,
    TinyLidarConvTransformerNet
)
from .maxt import LidarRegressor, get_model_cfg 

def load_cnn_model(model_name, input_dim, output_dim, compile_model: bool = False):
    """
    CNNベースのモデルをロードし、オプションでtorch.compileでコンパイルする。
    Args:
        model_name (str): ロードするCNNモデルの名前。
        input_dim (int): モデルの入力次元。
        output_dim (int): モデルの出力次元。
        compile_model (bool): Trueの場合、モデルをtorch.compileでコンパイルする。
    Returns:
        torch.nn.Module: ロードまたはコンパイルされたモデルインスタンス。
    """
    model = None
    if model_name == 'TinyLidarNet':  
        model = TinyLidarNet(input_dim, output_dim)
    elif model_name == 'TinyLidarLstmNet':
        model = TinyLidarLstmNet(input_dim, output_dim, lstm_hidden_dim=128, lstm_layers=1)
    elif model_name == 'TinyLidarConvLstmNet':
        model = TinyLidarConvLstmNet(input_dim, output_dim)
    elif model_name == 'TinyLidarActionNet':
        model = TinyLidarActionNet(input_dim, output_dim, action_dim=2)
    elif model_name == 'TinyLidarActionLstmNet':
        model = TinyLidarActionLstmNet(input_dim, output_dim, lstm_hidden_dim=128, lstm_layers=1)
    elif model_name == 'TinyLidarActionConvLstmNet':
        model = TinyLidarActionConvLstmNet(input_dim, output_dim)
    elif model_name == 'TinyLidarConvTransformerNet':
        model = TinyLidarConvTransformerNet(input_dim, output_dim, d_model=256,
                                           nhead=4,
                                           num_encoder_layers=2,
                                           dim_feedforward=256,
                                           dropout=0.1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # torch.compile の適用
    if compile_model and th_compile:
        print(f"[*] Compiling {model_name} with torch.compile...")
        model = th_compile(model)
        print(f"[+] {model_name} successfully compiled!")
    elif compile_model and not th_compile:
        print(f"[!] Warning: torch.compile requested for {model_name} but not available.")

    return model

def load_gnn_model(model_name, input_dim, hidden_dim, output_dim, pool_method='mean'):

    if model_name == 'LidarGCN':
        return LidarGCN(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        pool_method=pool_method)
    elif model_name == 'LidarGCNLstm':
        return LidarGcnLstmNet(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        pool_method=pool_method)
    elif model_name == 'LidarGAT':
        return LidarGAT(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        heads=8,
                        dropout_rate=0.5,
                        pool_method=pool_method)
    elif model_name == 'LidarGATLstm':
        return LidarGatLstmNet(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        heads=8,
                        dropout_rate=0.5,
                        pool_method=pool_method)
    else:
        raise ValueError(f"Unknown GNN model name: {model_name}")
    

def load_maxt_model(size: str, backbone_stages: int = 4, fpn_stages: int = 4, compile_model: bool = False):
    """
    MAxTモデルをロードし、オプションでtorch.compileでコンパイルする関数。
    Args:
        size (str): モデルのサイズ ('tiny', 'small', 'base' など)
        backbone_stages (int): バックボーンのステージ数。
        fpn_stages (int): FPNのステージ数。
        compile_model (bool): Trueの場合、モデルをtorch.compileでコンパイルする。
    Returns:
        torch.nn.Module: ロードまたはコンパイルされたモデルインスタンス。
    """
    cfg = get_model_cfg(size, backbone_stages, fpn_stages)
    model = LidarRegressor(cfg)

    # torch.compile の適用
    if compile_model and th_compile:
        print(f"[*] Compiling MaxT Model (size: {size}) with torch.compile...")
        model = th_compile(model)
        print(f"[+] MaxT Model (size: {size}) successfully compiled!")
    elif compile_model and not th_compile:
        print(f"[!] Warning: torch.compile requested for MaxT Model (size: {size}) but not available.")
    
    return model