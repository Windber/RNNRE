import torch
from stackrnn.stackrnntask import StackRNNTask
from basicrnn.rnntask import GRUTask, GRU
from stackrnn.nlfunction import *
from stackrnn.stackrnncell import StackRNNCell
basicConfig = {
    "batch_size": 100,
    "input_size": 6,
    "hidden_size": 4,
    "output_size": 2,
    "epochs": 50,
    "testfile_num": 3,
    "lr": 1e-4,
    "load": True,
    "device": torch.device("cpu"),
    "verbose": False,
    "threshold": 0.05,
    "debug": False,
    "onlytest": False,
    "verbose_batch": 0,
    "initialization": True,
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
    "task_class": GRUTask,
    "model_class": GRU,
    "load_model": r"",
            }
t5config = {
    "task_name": "T5@GRU",
    "data_name": "T5",
    "model_name": "GRU",
    "task_class": GRUTask,
    "model_class": GRU,
    "load_model": r"T5@GRU_0.00_1.00@240021",
    
            }
t7config = {
    "task_name": "T7@GRU",
    "data_name": "T7",
    "model_name": "GRU",
    "task_class": GRUTask,
    "model_class": GRU,
    "load_model": r"T5@GRU_0.00_1.00@240021",
    
            }
t4config.update(tomitaConfig)
t5config.update(tomitaConfig)
t7config.update(tomitaConfig)
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
    "load_model": r"dyck2@Stack_0.05_0.96@241924",
    "load_path": r"stackrnn/smodel",
    "saved_path": r"stackrnn/smodel",
    "sigmoid_type": HardSigmoid,
    "class_weight": [0.2219, 0.7781],
    "loss_num": 2,
    "loss_weight": [0.01],
    "linear_layers": [8],
    
            }
dyck2config.update(basicConfig)

if __name__ == "__main__":
    config_dict = dyck2config
    task = config_dict["task_class"](config_dict)
    task.experiment()
    
