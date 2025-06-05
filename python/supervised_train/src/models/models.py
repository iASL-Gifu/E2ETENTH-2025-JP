from .cnn import TinyLidarNet, TinyLidarLstmNet

def load_cnn_model(model_name, input_dim, output_dim):

    if model_name == 'TinyLidarNet':  # <-- '!=' を '==' に修正
        return TinyLidarNet(input_dim, output_dim)
    elif model_name == 'TinyLidarLstmNet':
        return TinyLidarLstmNet(input_dim, output_dim, lstm_hidden_dim=128, lstm_layers=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")