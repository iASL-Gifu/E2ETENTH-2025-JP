from .cnn import TinyLidarNet

def load_cnn_model(model_name, input_dim, output_dim):

    if model_name != 'TinyLidarNet':
        return TinyLidarNet(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")