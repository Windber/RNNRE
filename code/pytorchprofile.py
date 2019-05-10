import torch
import torch.nn as nn
from stackrnn.stackrnntask import StackRNNTask, StackRNN
from basicrnn.rnntask import RNNTask, RNN
from stackrnn.nlfunction import *
from stackrnn.stackrnncell import StackRNNCell
from stackrnn.callback import Save_data
import countlanguage
basic = {
    "batch_size": 100,
    "epochs": 50,
    "testfile_num": 5,
    "lr": 1e-3,
    "device": torch.device("cpu"),
    "verbose": False,
    "debug": False,
    "onlytest": False,
    "verbose_batch": 0,
    "initialization": True,
    }

rnn = {
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
   "task_class": RNNTask,
    "model_class": RNN,
    }

gru = {
    "model_name": "GRU",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
   "task_class": RNNTask,
    "model_class": RNN,
    "cell_class": nn.GRUCell,
    }

lstm = {
    "model_name": "LSTM",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel",
   "task_class": RNNTask,
    "model_class": RNN,
    "cell_class": nn.LSTMCell,
    }

srn = {
    "model_name": "SRN",
    "cell_class": nn.RNNCell,
    }

srn.update(rnn)
lstm.update(rnn)
gru.update(rnn)

tomita = {
    "input_size": 4,
    "hidden_size": 2,
    "output_size": 3,
    "alphabet": {"0": [0], "1": [1], "s": [2], "e": [3]},
    "classes": {'1': [1, 0, 0], '5': [1, 0, 1], '7': [1, 1, 1]},
    "train_path": r"../data_predicttask/tomita",
    "test_path": r"../data_predicttask/tomita",
    "load_path": r"basicrnn/smodel",
    "saved_path": r"basicrnn/smodel"
    }

t1 = {
    "data_name": "t1",
    }
t2 = {
    "data_name": "t2",
    }
t3 = {
    "data_name": "t3",
    }
t4 = {
    "data_name": "t4",
    }
t5 = {
    "data_name": "t5",
    }
t6 = {
    "data_name": "t6",
    }
t7 = {
    "data_name": "t7",
    }
t1.update(tomita)
t2.update(tomita)
t3.update(tomita)
t4.update(tomita)
t5.update(tomita)
t6.update(tomita)
t7.update(tomita)

t1srnConfig = {
    "task_name": "t1@srn",
    "load": False,
    "load_model": r"t1@srn_0.03_0.01@1322",
            }
t1srnConfig.update(basic)
t1srnConfig.update(t1)
t1srnConfig.update(srn)

t1gruConfig = {
    "task_name": "t1@GRU",
    "load": False,
    "load_model": r"t1@GRU_0.03_0.01@1322",
            }
t1gruConfig.update(basic)
t1gruConfig.update(t1)
t1gruConfig.update(gru)

t1lstmConfig = {
    "task_name": "t1@LSTM",
    "load": False,
    "load_model": r"t1@LSTM_0.03_0.01@1322",
            }
t1lstmConfig.update(basic)
t1lstmConfig.update(t1)
t1lstmConfig.update(lstm)

t2srnConfig = {
    "task_name": "t2@srn",
    "load": False,
    "load_model": r"t2@srn_0.03_0.01@1322",
            }
t2srnConfig.update(basic)
t2srnConfig.update(t2)
t2srnConfig.update(srn)

t2gruConfig = {
    "task_name": "t2@GRU",
    "load": False,
    "load_model": r"t2@GRU_0.03_0.01@1322",
            }
t2gruConfig.update(basic)
t2gruConfig.update(t2)
t2gruConfig.update(gru)

t2lstmConfig = {
    "task_name": "t2@LSTM",
    "load": False,
    "load_model": r"t2@LSTM_0.03_0.01@1322",
            }
t2lstmConfig.update(basic)
t2lstmConfig.update(t2)
t2lstmConfig.update(lstm)

t3srnConfig = {
    "task_name": "t3@srn",
    "load": False,
    "load_model": r"t3@srn_0.03_0.01@1322",
            }
t3srnConfig.update(basic)
t3srnConfig.update(t3)
t3srnConfig.update(srn)

t3gruConfig = {
    "task_name": "t3@GRU",
    "load": False,
    "load_model": r"t3@GRU_0.03_0.01@1322",
            }
t3gruConfig.update(basic)
t3gruConfig.update(t3)
t3gruConfig.update(gru)

t3lstmConfig = {
    "task_name": "t3@LSTM",
    "load": False,
    "load_model": r"t3@LSTM_0.03_0.01@1322",
            }
t3lstmConfig.update(basic)
t3lstmConfig.update(t3)
t3lstmConfig.update(lstm)


t4srnConfig = {
    "task_name": "t4@srn",
    "load": False,
    "load_model": r"t4@srn_0.03_0.01@1322",
            }
t4srnConfig.update(basic)
t4srnConfig.update(t4)
t4srnConfig.update(srn)

t4gruConfig = {
    "task_name": "t4@GRU",
    "load": False,
    "load_model": r"t4@GRU_0.03_0.01@1322",
            }
t4gruConfig.update(basic)
t4gruConfig.update(t4)
t4gruConfig.update(gru)

t4lstmConfig = {
    "task_name": "t4@LSTM",
    "load": False,
    "load_model": r"t4@LSTM_0.03_0.01@1322",
            }
t4lstmConfig.update(basic)
t4lstmConfig.update(t4)
t4lstmConfig.update(lstm)

t5srnConfig = {
    "task_name": "t5@srn",
    "load": False,
    "load_model": r"t5@srn_0.03_0.01@1322",
            }
t5srnConfig.update(basic)
t5srnConfig.update(t5)
t5srnConfig.update(srn)

t5gruConfig = {
    "task_name": "t5@GRU",
    "load": False,
    "load_model": r"t5@GRU_0.03_0.01@1322",
            }
t5gruConfig.update(basic)
t5gruConfig.update(t5)
t5gruConfig.update(gru)

t5lstmConfig = {
    "task_name": "t5@LSTM",
    "load": False,
    "load_model": r"t5@LSTM_0.03_0.01@1322",
            }
t5lstmConfig.update(basic)
t5lstmConfig.update(t5)
t5lstmConfig.update(lstm)

t6srnConfig = {
    "task_name": "t6@srn",
    "load": False,
    "load_model": r"t6@srn_0.03_0.01@1322",
            }
t6srnConfig.update(basic)
t6srnConfig.update(t6)
t6srnConfig.update(srn)

t6gruConfig = {
    "task_name": "t6@GRU",
    "load": False,
    "load_model": r"t6@GRU_0.03_0.01@1322",
            }
t6gruConfig.update(basic)
t6gruConfig.update(t6)
t6gruConfig.update(gru)

t6lstmConfig = {
    "task_name": "t6@LSTM",
    "load": False,
    "load_model": r"t6@LSTM_0.03_0.01@1322",
            }
t6lstmConfig.update(basic)
t6lstmConfig.update(t6)
t6lstmConfig.update(lstm)

t7srnConfig = {
    "task_name": "t7@srn",
    "load": False,
    "load_model": r"t7@srn_0.03_0.01@1322",
            }
t7srnConfig.update(basic)
t7srnConfig.update(t7)
t7srnConfig.update(srn)

t7gruConfig = {
    "task_name": "t7@GRU",
    "load": False,
    "load_model": r"t7@GRU_0.03_0.01@1322",
            }
t7gruConfig.update(basic)
t7gruConfig.update(t7)
t7gruConfig.update(gru)

t7lstmConfig = {
    "task_name": "t7@LSTM",
    "load": False,
    "load_model": r"t7@LSTM_0.03_0.01@1322",
            }
t7lstmConfig.update(basic)
t7lstmConfig.update(t7)
t7lstmConfig.update(lstm)


countlanguage = {
    "train_path": r"../data_predicttask/countlanguage",
    "test_path": r"../data_predicttask/countlanguage",
    }
anbn = {
    "alphabet": {"a": [2], "b": [3], "s": [0], "e": [1], "#": [1]},
    "classes": {"3": [1, 1, 0], "6": [0, 1, 1], "4": [0, 0, 1], "1": [1, 0, 0]},
    "input_size": 4,
    "hidden_size": 3,
    "output_size": 3,
    "data_name": "anbn",
    }
anbn.update(countlanguage)

anbncn = {
    "input_size": 5,
    "hidden_size": 4,
    "output_size": 4,    
    "alphabet": {"a": [2], "b": [3], "c": [4], "s": [0], "e": [1]},
    "classes": {"1": [1, 0, 0, 0], "3": [1, 1, 0, 0], "4": [0, 0, 1, 0], "6": [0, 1, 1, 0], "8": [0, 0, 0, 1]},
    "data_name": "anbncn",
    }
anbncn.update(countlanguage)

anbncngruconfig = {
    "task_name": "anbncn@gru",
    "load_model": r"anbn@LSTM_1.00_0.80@1118",
    "load": False,
    }
anbncngruconfig.update(basic)
anbncngruconfig.update(gru)
anbncngruconfig.update(anbncn)

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
    "load_model": r"anbn@Stack_0.66_0.43@1747",
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
    config_dict = t4srnConfig
    task = config_dict["task_class"](config_dict)
    task.experiment()
    
