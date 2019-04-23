import torch
from stackrnn.stackrnntask import StackRNNTask
from basicrnn.rnntask import GRUTask, GRU
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
    "debug": True,
    "verbose_batch": 100,
    }
t4config = {
    "task_name": "T4@GRU",
    "data_name": "T4",
    "model_name": "GRU",
    "task_class": GRUTask,
    "model_class": GRU,
    "alphabet": {"0": [0], "1": [1], "s": [2], "e": [3], "#": [3]},
    "classes": {"0": 0, "1": 1},
    "train_path": r"../data/tomita",
    "test_path": r"../data/tomita",
    "load_model": r"",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
    
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
    "load_path": r"stackrnn/smodel",
    "saved_path": r"stackrnn/smodel",
    "sigmoid": HardSigmoid,
    
            }
config_dyck2.update(BasicConfig)
t4config.update(BasicConfig)
if __name__ == "__main__":
    config_dict = t4config
    task = config_dict["task_class"](config_dict)
    task.experiment()