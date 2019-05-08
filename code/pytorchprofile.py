import torch
import torch.nn as nn
from stackrnn.stackrnntask import StackRNNTask
from basicrnn.rnntask import RNNTask, RNN
from stackrnn.nlfunction import *
from stackrnn.stackrnncell import StackRNNCell
from stackrnn.callback import Save_data
basicConfig = {
    "batch_size": 10,
    "epochs": 10,
    "testfile_num": 5,
    "lr": 1e-4,
    "load": True,
    "device": torch.device("cpu"),
    "verbose": False,
    "threshold": 0.05,
    "debug": False,
    "onlytest": False,
    "verbose_batch": 0,
    "initialization": False,
    }
tomitaConfig = {
    "alphabet": {"0": [0], "1": [1], "s": [2], "e": [3], "#": [3]},
    "classes": {"0": 0, "1": 1},
    "train_path": r"../data/tomita",
    "test_path": r"../data/tomita",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
    }
tomitaConfig.update(basicConfig)
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

anbnconfig = {
    "input_size": 4,
    "hidden_size": 3,
    "output_size": 3,
    "alphabet": {"a": [2], "b": [3], "s": [0], "e": [1], "#": [1]},
    "classes": {"3": [1, 1, 0], "6": [0, 1, 1], "4": [0, 0, 1], "1": [1, 0, 0]},
    "train_path": r"../data_predicttask/countlanguage",
    "test_path": r"../data_predicttask/countlanguage",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
    "task_name": "anbn@LSTM",
    "data_name": "anbn",
    "model_name": "LSTM",
    "cellclass": nn.LSTMCell,
    "task_class": RNNTask,
    "model_class": RNN,
    "load_model": r"anbn@LSTM_0.00_0.00@2055",
    }
anbnconfig.update(basicConfig)

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
    "cell": nn.GRU,
    "task_class": RNNTask,
    "model_class": RNN,
    "load_model": r"anbncn@LSTM_1.00_0.88@1822",
    }
anbncnconfig.update(basicConfig)
sd = Save_data(file_name="stackrnn/sdata/hotmap")

dyck2config = {
    "task_name": "dyck2@Stack",     
    "data_name": "dyck2",
    "model_name": "Stack",
    "task_class": StackRNNTask,
    "model_class": StackRNNCell,
    "alphabet": {"(": [0], ")": [1], "s": [2], "e": [3], "#": [3], "[": [4], "]": [5]},
    "classes": {"0": 0, "1": 1},
    "read_size": 4,
    "n_args": 2,
    "train_path": r"../data/dyck2",
    "test_path": r"../data/dyck2",
    "load_model": r"dyck2@Stack_0.07_0.94@250128",
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
dyck2config.update(basicConfig)

if __name__ == "__main__":
    config_dict = anbnconfig
    task = config_dict["task_class"](config_dict)
    task.experiment()
    
