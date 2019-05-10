import torch
import torch.nn as nn
from stackrnn.stackrnntask import StackRNNTask, StackRNN
from basicrnn.rnntask import RNNTask, RNN
from stackrnn.nlfunction import *
from stackrnn.stackrnncell import StackRNNCell
from stackrnn.callback import Save_data
basic = {
    "batch_size": 100,
    "epochs": 50,
    "testfile_num": 2,
    "lr": 1e-3,
    "load": True,
    "device": torch.device("cpu"),
    "verbose": False,
    "debug": False,
    "onlytest": False,
    "verbose_batch": 0,
    "initialization": True,
    }
anbn = {
    "alphabet": {"a": [2], "b": [3], "s": [0], "e": [1], "#": [1]},
    "classes": {"3": [1, 1, 0], "6": [0, 1, 1], "4": [0, 0, 1], "1": [1, 0, 0]},
    "input_size": 4,
    "hidden_size": 3,
    "output_size": 3,
    "train_path": r"../data_predicttask/countlanguage",
    "test_path": r"../data_predicttask/countlanguage",
    "data_name": "anbn",

    }

gru = {
    "model_name": "GRU",
    "cell_class": nn.GRUCell,
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
   "task_class": RNNTask,
    "model_class": RNN,
    }
tomitaConfig = {
    "alphabet": {"0": [0], "1": [1], "s": [2], "e": [3], "#": [3]},
    "classes": {"0": 0, "1": 1},
    "train_path": r"../data/tomita",
    "test_path": r"../data/tomita",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
    }
tomitaConfig.update(basic)
t4config = {
    "task_name": "T4@GRU",
    "data_name": "T4",
    "model_name": "GRU",
    "task_class": RNNTask,
    "model_class": RNN,
    "load_model": r"",
            }
t5config = {
    "task_name": "T5@GRU",
    "data_name": "T5",
    "model_name": "GRU",
    "task_class": RNNTask,
    "model_class": RNN,
    "load_model": r"T5@GRU_0.00_1.00@240021",
    
            }
t7config = {
    "task_name": "T7@GRU",
    "data_name": "T7",
    "model_name": "GRU",
    "task_class": RNNTask,
    "model_class": RNN,
    "load_model": r"T5@GRU_0.00_1.00@240021",
    
            }
t4config.update(tomitaConfig)
t5config.update(tomitaConfig)
t7config.update(tomitaConfig)

anbngruConfig = {
    "task_name": "anbn@GRU",
     "load_model": r"anbn@LSTM_1.00_0.80@1118",
    }


anbnLSTMconfig = {
    "task_name": "anbn@LSTM",
    "data_name": "anbn",
    "model_name": "LSTM",
    "cell_class": nn.LSTMCell,
     "load_model": r"anbn@LSTM_1.00_0.80@1118",
    }

anbngruConfig.update(anbn, **gru)
anbnLSTMconfig.update(anbn)
anbncnconfig = {
    "input_size": 5,
    "hidden_size": 4,
    "output_size": 4,
    "alphabet": {"a": [2], "b": [3], "c": [4], "s": [0], "e": [1]},
    "classes": {"1": [1, 0, 0, 0], "3": [1, 1, 0, 0], "4": [0, 0, 1, 0], "6": [0, 1, 1, 0], "8": [0, 0, 0, 1]},
    "train_path": r"../data_predicttask/countlanguage",
    "test_path": r"../data_predicttask/countlanguage",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
    "task_name": "anbncn@GRU",
    "data_name": "anbncn",
    "model_name": "GRU",
    "cell_class": nn.GRU,
    "task_class": RNNTask,
    "model_class": RNN,
    "load_model": r"anbn@LSTM_1.00_0.80@1118",
    }
anbncnconfig.update(basic)
sd = Save_data(file_name="stackrnn/sdata/hotmap")

dyck1config = {
    "input_size": 4,
    "hidden_size": 3,
    "output_size": 3,
    "alphabet": {"(": [2], ")": [3], "s": [0], "e": [1]},
    "classes": {"1": [1, 0, 0], "3": [1, 1, 0], "6": [0, 1, 1]},
    "train_path": r"../data_predicttask/countlanguage",
    "test_path": r"../data_predicttask/countlanguage",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
    "task_name": "dyck1@LSTM",
    "data_name": "dyck1",
    "model_name": "LSTM",
    "cell_class": nn.LSTMCell,
    "task_class": RNNTask,
    "model_class": RNN,
    "load_model": r"anbn@LSTM_1.00_0.80@1118",
    
    }
dyck1config.update(basic)
anbnStackRNNconfig = {
    "input_size": 4,
    "hidden_size": 2,
    "output_size": 3,
    "alphabet": {"a": [2], "b": [3], "s": [0], "e": [1], "#": [1]},
    "classes": {"3": [1, 1, 0], "6": [0, 1, 1], "4": [0, 0, 1], "1": [1, 0, 0]},
    "train_path": r"../data_predicttask/countlanguage",
    "test_path": r"../data_predicttask/countlanguage",
    "task_name": "anbn@Stack",     
    "data_name": "anbn",
    "model_name": "Stack",
    "task_class": StackRNNTask,
    "model_class": StackRNN,
    'cell_class': StackRNNCell,
    "read_size": 2,
    "n_args": 2,
    "load_model": r'anbn@Stack_0.50_0.24@0345',
    "load_path": r"stackrnn/smodel",
    "saved_path": r"stackrnn/smodel",
    "sigmoid_type": HardSigmoid,
    "class_weight": [0.5, 0.5],
    "loss_num": 2,
    "loss_weight": [0.01],
    "linear_layers": [8],
    "batch_callback": [],
    "step_callback":[]
            }
anbnStackRNNconfig.update(basic)

if __name__ == "__main__":
    config_dict = anbnStackRNNconfig
    task = config_dict["task_class"](config_dict)
    task.experiment()
    
