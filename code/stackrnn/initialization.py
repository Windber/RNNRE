import torch.nn.init as init

def rnn_init_(layer):
    init.xavier_uniform_(layer.weight_hh.data)
    init.xavier_uniform_(layer.weight_ih.data)
    init.uniform_(layer.bias_hh.data, 0, 0)
    init.uniform_(layer.bias_ih.data, 0, 0)
def linear_init_(layer):
    init.xavier_uniform_(layer.weight.data)
    init.uniform_(layer.bias.data, 0, 0)