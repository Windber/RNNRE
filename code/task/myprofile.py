'''
@author: lenovo
'''
import copy
import torch
from torch.optim import SGD, Adam, Adagrad, RMSprop
BaseConfig = {"input_size": 4,
              "hidden_size": 2,
              "output_size": 2,
              "batch_size": 100,
              "epoch": 100,
              "lr": 1e-3,
              "verbose": False,
              "device": torch.device("cpu"),
              "threshold": 0.05,
              "Optimizer": RMSprop
              }

Tomita1Config = {"trpath_prefix": "../../data/tomita/",
                 "tepath_prefix": "../../data/tomita/",
                 "load_path_prefix":  "../../savedmodel/BestModel_",
                "load": False,
                 "task": "T1",
                 "save_path_prefix": "../../savedmodel/BestModel_",
                 }
Tomita2Config = {"trpath_prefix": "../../data/tomita/",
                 "tepath_prefix": "../../data/tomita/",
                 "load_path_prefix":  "../../savedmodel/BestModel_",
                 "load": False,
                 "task": "T2",
                 "save_path_prefix": "../../savedmodel/BestModel_",
                 }
Tomita3Config = {"trpath_prefix": "../../data/tomita/",
                 "tepath_prefix": "../../data/tomita/",
                 "load_path_prefix":  "../../savedmodel/BestModel_",
                 "load": False,
                 "task": "T3",
                 "save_path_prefix": "../../savedmodel/BestModel_",
                 }
Tomita4Config = {"trpath_prefix": "../../data/tomita/",
                 "tepath_prefix": "../../data/tomita/",
                 "load_path_prefix":  "../../savedmodel/BestModel_",
                 "load": False,
                 "task": "T4",
                 "save_path_prefix": "../../savedmodel/BestModel_",
                 }
Tomita5Config = {"trpath_prefix": "../../data/tomita/",
                 "tepath_prefix": "../../data/tomita/",
                 "load_path_prefix":  "../../savedmodel/BestModel_",
                 "load": False,
                 "task": "T5",
                 "save_path_prefix": "../../savedmodel/BestModel_",
                 }
Tomita6Config = {"trpath_prefix": "../../data/tomita/",
                 "tepath_prefix": "../../data/tomita/",
                 "load_path_prefix":  "../../savedmodel/BestModel_",
                 "load": False,
                 "task": "T6",
                 "save_path_prefix": "../../savedmodel/BestModel_",
                 }
Tomita7Config = {"trpath_prefix": "../../data/tomita/",
                 "tepath_prefix": "../../data/tomita/",
                 "load_path_prefix":  "../../savedmodel/BestModel_",
                 "load": False,
                 "task": "T7",
                 "save_path_prefix": "../../savedmodel/BestModel_",
                 }
Tomita1Config.update(BaseConfig)
Tomita2Config.update(BaseConfig)
Tomita3Config.update(BaseConfig)
Tomita4Config.update(BaseConfig)
Tomita5Config.update(BaseConfig)
Tomita6Config.update(BaseConfig)
Tomita7Config.update(BaseConfig)



