import torch
import torch.nn as nn
from stackrnn.stackrnntask import StackRNNTask, StackRNN
from stackrnn.rnntask import RNNTask, RNN, PHLSTMCell, MyLSTMCell
from stackrnn.nlfunction import *
from stackrnn.stackrnncell import StackRNNCell
from stackrnn.callback import Save_data, Sdforlstm, Save_loss

def append(ori, *dess):
    key = ori.keys()
    for des in dess:
        for k in des.keys():
            if k not in key:
                ori[k] = des[k]
                
import sys
sd = None
# sd = Sdforlstm(path="stackrnn/sdata/", task='anbn@phlstm')
# sd = Sdforstacksrn(path="stackrnn/sdata/", task='dyck2@srn')
# sd = Sdforstacksrn(path="stackrnn/sdata/", task='dyck2@srn')
task = sys.argv[1] + sys.argv[2] 
#sd = Save_loss(path='stackrnn/sdata/', task=task)
sdl = [sd] if sd is not None else []
basic = {
    "batch_size": 100,
    "epochs": 20,
    "testfile_num": 2,
    "lr": 1e-1,
    "device": torch.device("cpu"),
    "verbose": False,
    "debug": False,
    "verbose_batch": 0,
    "initialization": True,
    'validate': False,
    "load_path": r"stackrnn/smodel/",
    "saved_path": r"stackrnn/smodel/",
    'callback': sdl,
    'load': True,
    'onlytest': True,
    'load_last': False,

    }

rnn = {
   "task_class": RNNTask,
    "model_class": RNN,
    }

gru = {
    "model_name": "GRU",
    "cell_class": nn.GRUCell,
    }

lstm = {
    "model_name": "LSTM",
    "cell_class": nn.LSTMCell,
    }

phlstm = {
    "model_name": "PHLSTM",
    "cell_class": PHLSTMCell,
    }
srn = {
    "model_name": "SRN",
    "cell_class": nn.RNNCell,
    }

append(srn, rnn)
append(gru, rnn)
append(lstm, rnn)
append(phlstm, rnn)
stackrnn = {
    "task_class": StackRNNTask,
    "model_class": StackRNN,
    'cell_class': StackRNNCell,
    "read_size": 2,
    "n_args": 2,
    'hidden_size': 2,
    "lr": 1e-3,
    "alpha": 0.002,
    'customalization': True,
    'weight_decay': 0,
    'bias_customalization': False,
}

stacksrn = {
    "model_name": "stacksrn",
    'controller_cell_class': nn.RNNCell,
    }

stackgru = {
    "model_name": "stackgru",
    'controller_cell_class': nn.GRUCell,
    }

stacklstm = {
    "model_name": "stacklstm",
    'controller_cell_class': PHLSTMCell,
    }

append(stacksrn, stackrnn)
append(stackgru, stackrnn)
append(stacklstm, stackrnn)

tomita = {
    "input_size": 4,
    "hidden_size": 2,
    "output_size": 3,
    "alphabet": {"0": [0], "1": [1], "s": [2], "e": [3]},
    "classes": {'1': [1, 0, 0], '2': [0, 1, 0], '3': [1, 1, 0], '4': [0, 0, 1], '5': [1, 0, 1], '6': [0, 1, 1], '7': [1, 1, 1]},
    "train_path": r"../data_predicttask/tomita/",
    "test_path": r"../data_predicttask/tomita/",
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

append(t1, tomita)
append(t2, tomita)
append(t3, tomita)
append(t4, tomita)
append(t5, tomita)
append(t6, tomita)
append(t7, tomita)

cl = {
    "train_path": r"../data_predicttask/countlanguage/",
    "test_path": r"../data_predicttask/countlanguage/",
    }

anbn = {
    "alphabet": {"a": [2], "b": [3], "s": [0], "e": [1], "#": [1]},
    "classes": {"3": [1, 1, 0], "6": [0, 1, 1], "4": [0, 0, 1], "1": [1, 0, 0]},
    "input_size": 4,
    "hidden_size": 3,
    "output_size": 3,
    "data_name": "anbn",
    }
append(anbn, cl)

anbncn = {
    "input_size": 5,
    "hidden_size": 4,
    "output_size": 4,    
    "alphabet": {"a": [2], "b": [3], "c": [4], "s": [0], "e": [1]},
    "classes": {"1": [1, 0, 0, 0], "3": [1, 1, 0, 0], "4": [0, 0, 1, 0], "6": [0, 1, 1, 0], "8": [0, 0, 0, 1]},
    "data_name": "anbncn",
    }
append(anbncn, cl)

dyck1 = {
    "input_size": 4,
    "hidden_size": 3,
    "output_size": 3,
    "alphabet": {"(": [2], ")": [3], "s": [0], "e": [1]},
    "classes": {"1": [1, 0, 0], "3": [1, 1, 0], "6": [0, 1, 1]},
    'data_name': 'dyck1'
}
append(dyck1, cl)

cfl = {
    "train_path": r"../data_predicttask/cfl/",
    "test_path": r"../data_predicttask/cfl/",
    }

dyck2 = {
    "input_size": 6,
    "output_size": 5,
    "alphabet": {"(": [2], ")": [3], "s": [0], "e": [1], '[': [4], ']': [5]},
    "classes": {"1": [1, 0, 0, 0, 0], "b": [1, 1, 0, 1, 0], "e": [0, 1, 1, 1, 0], '1a': [0, 1, 0, 1, 1]},
    'data_name': 'dyck2'
}
append(dyck2, cfl)

t1srnConfig = {
    "task_name": "t1@srn",
    "load": False,
    "load_model": r"t1@srn_0.03_0.01@1322",
            }

append(t1srnConfig, basic, t1, srn)

t1gruConfig = {
    "task_name": "t1@GRU",
    "load": False,
    "load_model": r"t1@GRU_0.03_0.01@1322",
            }

append(t1gruConfig, basic, t1, gru)

t1lstmConfig = {
    "task_name": "t1@LSTM",
    "load_model": r"t1@LSTM_1.00_0.02@1312",
            }

append(t1lstmConfig, basic, t1, lstm)

t2srnConfig = {
    "task_name": "t2@srn",
    "load": False,
    "load_model": r"t2@srn_0.03_0.01@1322",
            }

append(t2srnConfig, basic, t2, srn)

t2gruConfig = {
    "task_name": "t2@GRU",
    "load": False,
    "load_model": r"t2@GRU_0.03_0.01@1322",
            }

append(t2gruConfig, basic, t2, gru)

t2lstmConfig = {
    "task_name": "t2@LSTM",
    "load": False,
    "load_model": r"t2@LSTM_0.03_0.01@1322",
            }

append(t2lstmConfig, basic, t2, lstm)

t3srnConfig = {
    "task_name": "t3@srn",
    "load": False,
    "load_model": r"t3@srn_0.03_0.01@1322",
            }

append(t3srnConfig, basic, t3, srn)

t3gruConfig = {
    "task_name": "t3@GRU",
    "load": False,
    "load_model": r"t3@GRU_0.03_0.01@1322",
            }
append(t3gruConfig, basic, t3, gru)

t3lstmConfig = {
    "task_name": "t3@LSTM",
    "load": False,
    "load_model": r"t3@LSTM_0.03_0.01@1322",
            }

append(t3lstmConfig, basic, t3, lstm)

t4srnConfig = {
    "task_name": "t4@srn",
    "load": False,
    "load_model": r"t4@srn_0.03_0.01@1322",
            }

append(t4srnConfig, basic, t4, srn)

t4gruConfig = {
    "task_name": "t4@GRU",
    "load": False,
    "load_model": r"t4@GRU_0.03_0.01@1322",
            }

append(t4gruConfig, basic, t4, gru)

t4lstmConfig = {
    "task_name": "t4@LSTM",
    "load": False,
    "load_model": r"t4@LSTM_0.03_0.01@1322",
            }

append(t4lstmConfig, basic, t4, lstm)

t5srnConfig = {
    "task_name": "t5@srn",
    "load": False,
    "load_model": r"t5@srn_0.03_0.01@1322",
            }
append(t5srnConfig, basic, t5, srn)

t5gruConfig = {
    "task_name": "t5@GRU",
    "load": False,
    "load_model": r"t5@GRU_0.03_0.01@1322",
            }
append(t5gruConfig, basic, t5, gru)

t5lstmConfig = {
    "task_name": "t5@LSTM",
    "load": False,
    "load_model": r"t5@LSTM_0.03_0.01@1322",
            }
append(t5lstmConfig, basic, t5, lstm)

t6srnConfig = {
    "task_name": "t6@srn",
    "load": False,
    "load_model": r"t6@srn_0.03_0.01@1322",
            }
append(t6srnConfig, basic, t6, srn)

t6gruConfig = {
    "task_name": "t6@GRU",
    "load": False,
    "load_model": r"t6@GRU_0.03_0.01@1322",
            }
append(t6gruConfig, basic, t6, gru)

t6lstmConfig = {
    "task_name": "t6@LSTM",
    "load": False,
    'onlytest': True,
    "load_model": r"t6@LSTM_0.03_0.01@1322",
            }
append(t6lstmConfig, basic, t6, lstm)

t7srnConfig = {
    "task_name": "t7@srn",
    "load": False,
    "load_model": r"t7@srn_0.03_0.01@1322",
            }
append(t7srnConfig, basic, t7, srn)

t7gruConfig = {
    "task_name": "t7@GRU",
    "load": False,
    "load_model": r"t7@GRU_0.03_0.01@1322",
            }
append(t7gruConfig, basic, t7, gru)

t7lstmConfig = {
    "task_name": "t7@LSTM",
    "load": False,
    "load_model": r"t7@LSTM_0.03_0.01@1322",
            }
append(t7lstmConfig, basic, t7, lstm)

anbnsrnConfig = {
    "task_name": "anbn@srn",
    "load_model": r"anbn@LSTM_1.00_0.80@1118",
    "load": False,
    }

anbngruConfig = {
    "task_name": "anbn@gru",
    "load_model": r"anbn@LSTM_1.00_0.80@1118",
    "load": False,
    }

anbnlstmConfig = {
    "task_name": "anbn@lstm",
    "load_model": r"anbn@LSTM_1.00_0.80@1118",
    }

anbnphlstmConfig = {
    "task_name": "anbn@phlstm",
    "load_model": r"finaltrain_anbnphlstm",
    'lr': 100,
    }

append(anbnsrnConfig, basic, anbn, srn)
append(anbngruConfig, basic, anbn, gru)
append(anbnlstmConfig, basic, anbn, lstm)
append(anbnphlstmConfig, basic, anbn, phlstm)


anbncnsrnConfig = {
    "task_name": "anbncn@srn",
    "load_model": r"anbncn@LSTM_1.00_0.80@1118",
    "load": False,
    }

anbncngruConfig = {
    "task_name": "anbncn@gru",
    "load_model": r"anbncn@LSTM_1.00_0.80@1118",
    "load": False,
    }

anbncnlstmConfig = {
    "task_name": "anbncn@lstm",
    "load_model": r"anbncn@LSTM_1.00_0.80@1118",
    "load": False,
    }

anbncnphlstmConfig = {
    "task_name": "anbncn@phlstm",
    "load_model": r"anbncn@phlstm_1.00_0.00@0636",
    'lr': 1e-3,
    }

append(anbncnsrnConfig, basic, anbncn, srn)
append(anbncngruConfig, basic, anbncn, gru)
append(anbncnlstmConfig, basic, anbncn, lstm)
append(anbncnphlstmConfig, basic, anbncn, phlstm)

dyck1srnConfig = {
    "task_name": "dyck1@srn",
    "load_model": r"dyck1@LSTM_1.00_0.80@1118",
    "load": False,
    }

dyck1gruConfig = {
    "task_name": "dyck1@gru",
    "load_model": r"dyck1@LSTM_1.00_0.80@1118",
    "load": False,
    }

dyck1lstmConfig = {
    "task_name": "dyck1@lstm",
    "load_model": r"dyck1@LSTM_1.00_0.80@1118",
    "load": False,
    }

dyck1phlstmConfig = {
    "task_name": "dyck1@phlstm",
    "load_model": r"dyck1@phlstm_1.00_0.00@0827",
    'lr': 1e-4,
    }
append(dyck1srnConfig, basic, dyck1, srn)
append(dyck1gruConfig, basic, dyck1, gru)
append(dyck1phlstmConfig, basic, dyck1, phlstm)

dyck2srnConfig = {
    "task_name": "dyck2@srn",
    "load_model": r"dyck2@LSTM_1.00_0.80@1118",
    }

dyck2gruConfig = {
    "task_name": "dyck2@gru",
    "load_model": r"dyck2@LSTM_1.00_0.80@1118",
    }

dyck2phlstmConfig = {
    "task_name": "dyck2@phlstm",
    "load_model": r"dyck2@LSTM_1.00_0.80@1118",
    }

append(dyck2srnConfig, basic, dyck2, srn)
append(dyck2gruConfig, basic, dyck2, gru)
append(dyck2phlstmConfig, basic, dyck2, phlstm)


anbnstackgruConfig = {
    "task_name": "anbn@stackgru",     
    "load_model": r'finaltrain_anbnstackgru',
            }

anbncnstackgruConfig = {
    "task_name": "anbncn@stackgru",     
    "load_model": r'anbncn@stackgru_1.00_1.00@honey',
            }

dyck1stackgruConfig = {
    "task_name": "dyck1@stackgru",     
    "load_model": r'dyck1@stackgru_1.00_1.00@honey',
            }

t4stackgruConfig = {
    "task_name": "t4@stackgru",     
    "load_model": r't4@stackgru_1.00_1.00@honey',
            }
append(anbnstackgruConfig, basic, anbn, stackgru)
append(anbncnstackgruConfig, basic, anbncn, stackgru)
append(dyck1stackgruConfig, basic, dyck1, stackgru)
append(t4stackgruConfig, basic, t4, stackgru)

dyck2stacksrnConfig = {
    "task_name": "dyck2@stacksrn",     
    "load_model": r'dyck2@stacksrn_1.00_0.00@0005',
            }

dyck2stackgruConfig = {
    "task_name": "dyck2@stackgru",     
    "load_model": r'dyck2@stackgru_0.88_0.20@1018',
            }

dyck2stacklstmConfig = {
    "task_name": "dyck2@stacklstm",     
    "load_model": r'dyck2@stacklstm_1.00_0.01@2216',
            }

append(dyck2stacksrnConfig, basic, dyck2, stacksrn)
append(dyck2stackgruConfig, basic, dyck2, stackgru)
append(dyck2stacklstmConfig, basic, dyck2, stacklstm)

if __name__ == "__main__":
            config_dict = eval( task + 'Config')
            task = config_dict["task_class"](config_dict)
            task.experiment()
#     for t in ['t1', 't2', 't3', 't4', 't5', 't6', 't7']:
#         for r in ['srn', 'gru', 'lstm']:
#     
#             config_dict = eval(t + r + 'Config')
#             sd = Save_loss(path='stackrnn/sdata/', task=t+r)
#             config_dict['callback'] = [sd]
#             task = config_dict["task_class"](config_dict)
#             task.experiment()
            
