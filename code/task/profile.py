'''
@author: lenovo
'''
import copy
import torch
BaseConfig = {"input_size": 4,
              "hidden_size": 2,
              "output_size": 2,
              "batch_size": 5,
              "epoch": 10,
              "device": torch.device("cpu")
              }

Tomita1Config = {"trpath": "../../data/tomita_dealed/T1_test1",
                 "tepath": "../../data/tomita_dealed/T1_test1",
                 "load_path": None,
                 "task": "Tomita1"
                 }
Tomita1Config.update(BaseConfig)

