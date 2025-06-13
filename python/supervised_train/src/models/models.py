# from .cnn import TinyLidarNet, TinyLidarLstmNet, TinyLidarConvLstmNet, TinyLidarActionNet, TinyLidarActionLstmNet, TinyLidarActionConvLstmNet, TinyLidarConvTransformerNet
from .cnn import (
    TinyLidarNet,
    TinyLidarLstmNet,
    TinyLidarConvLstmNet,
    TinyLidarActionNet,
    TinyLidarActionLstmNet,
    TinyLidarActionConvLstmNet,
    TinyLidarConvTransformerNet
)
from .gnn import LidarGCN, LidarGAT, LidarGcnLstmNet, LidarGatLstmNet

def load_cnn_model(model_name, input_dim, output_dim):

    if model_name == 'TinyLidarNet':  
        return TinyLidarNet(input_dim, output_dim)
    elif model_name == 'TinyLidarLstmNet':
        return TinyLidarLstmNet(input_dim, output_dim, lstm_hidden_dim=128, lstm_layers=1)
    elif model_name == 'TinyLidarConvLstmNet':
        return TinyLidarConvLstmNet(input_dim, output_dim)
    elif model_name == 'TinyLidarActionNet':
        return TinyLidarActionNet(input_dim, output_dim, action_dim=2)
    elif model_name == 'TinyLidarActionLstmNet':
        return TinyLidarActionLstmNet(input_dim, output_dim, lstm_hidden_dim=128, lstm_layers=1)
    elif model_name == 'TinyLidarActionConvLstmNet':
        return TinyLidarActionConvLstmNet(input_dim, output_dim)
    elif model_name == 'TinyLidarConvTransformerNet':
        return TinyLidarConvTransformerNet(input_dim, output_dim, d_model=256,
                                           nhead=4,
                                           num_encoder_layers=2,
                                           dim_feedforward=256,
                                           dropout=0.1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

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