import torch
from stackrnn.stackrnntask import StackRNNTask
from basicrnn.rnntask import RNNTask, RNN
from stackrnn.nlfunction import *
from stackrnn.stackrnncell import StackRNNCell
BasicConfig = {
    "batch_size": 100,
    "input_size": 6,
    "hidden_size": 8,
    "output_size": 2,
    "epochs": 1,
    "testfile_num": 1,
    "onlytest": False,
    "lr": 1e-3,
    "load": False,
    "device": torch.device("cpu"),
    "verbose": False,
    "threshold": 0.05,
    "load_path": r"smodel",
    "saved_path": r"smodel",
    }
Tomita4Config = {
    "task_name": "T4@GRU",
    "data_name": "T4",
    "model_name": "GRU",
    "task_class": RNNTask,
    "model_class": RNN,
    "alphabet": {"0": [0], "1": [1], "s": [2], "e": [3], "#": [3]},
    "classes": {"0": 0, "1": 1},
    "train_path": r"../../data/tomita",
    "test_path": r"../../data/tomita",
    "load_model": r"",
    
            }

config_dyck2 = {
    "task_name": "dyck2@Stack",
    "data_name": "dyck2",
    "model_name": "Stack",
    "task_class": StackRNNTask,
    "model_class": StackRNNCell,
    "alphabet": {"(": [0], ")": [1], "s": [2], "e": [3], "#": [3], "[": [4], "]": [5]},
    "classes": {"0": 0, "1": 1},
    "read_size": 8,
    "n_args": 3,
    "train_path": r"../data/dyck2",
    "test_path": r"../data/dyck2",
    "load_model": r"",
    "sigmoid": HardSigmoid,
    
            }
config_dyck2.update(BasicConfig)
if __name__ == "__main__":
    config_dict = config_dyck2
    task = config_dict["task_class"](config_dict)
    task.experiment()