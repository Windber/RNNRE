import torch.nn.init as init

def gru_init_(layer):
    input_size = layer.input_size
    hidden_size = layer.hidden_size
    init.xavier_uniform_(layer.weight_hh.data)
    init.xavier_uniform_(layer.weight_ih.data)
    init.uniform_(layer.bias_hh.data, 0, 0)
    init.uniform_(layer.bias_ih.data, 0, 0)
def linear_init_(layer):
    input_size = layer.in_features
    output_size = layer.out_features
    init.xavier_uniform_(layer.weight.data)
    init.uniform_(layer.bias.data, 0, 0)