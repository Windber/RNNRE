import numpy as np
import pickle
class Call_back:
    def __init__(self, kwargs):
        self.params = kwargs
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
        
    def step_cb(self, model, *args):
        pass
    def batch_cb(self, model, *args):
        pass
    def epoch_cb(self, model, *args):
        pass
    def eepoch_cb(self, model, *args):
        pass
class Save_loss(Call_back):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.eepoch = list()
    def eepoch_cb(self, model, *args):
        f = open(self.path + self.task + '_loss', 'wb')
        pickle.dump(self.eepoch, f)
        f.close()
    def epoch_cb(self, model, *args):
        self.eepoch.append(args[0])
class Save_data(Call_back):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.epoch = list()
        self.batch = list()
        self.epoch_count = 1
    def __call__(self, model, *args):
        pass
    def epoch_cb(self, model, *args):
        f = open(self.path + self.task + '_test' + str(self.epoch_count), 'wb')
        pickle.dump(self.epoch, f)
        self.epoch = list()
        self.epoch_count += 1
        
class Sdforlstm(Save_data):
    def batch_cb(self, model, *args):
        arr = np.array(self.batch)
        arr = np.transpose(arr, (1, 0, 2))
        self.epoch.append(arr)
        self.batch = list()

    def step_cb(self, model, *args):
        h = args[0][0].detach().numpy()
        c = args[0][1].detach().numpy()
        batch_size = model.batch_size
        element_size = 0
        li = [h, c]
        for i in li:
            tmp = i.shape[1] if len(i.shape) > 1 else 1
            element_size += tmp
        arr = np.ndarray((batch_size, element_size))
        index = 0
        for i in li:
            tmp = i.shape[1] if len(i.shape) > 1 else 1
            arr[:, index: index + tmp] = i
            index += tmp
        self.batch.append(arr)

class Sdforsrn(Save_data):
    def batch_cb(self, model, *args):
        arr = np.array(self.batch)
        arr = np.transpose(arr, (1, 0, 2))
        self.epoch.append(arr)
        self.batch = list()

    def step_cb(self, model, *args):
        h = args[0].detach().numpy()
        batch_size = model.batch_size
        element_size = 0
        li = [h]
        for i in li:
            tmp = i.shape[1] if len(i.shape) > 1 else 1
            element_size += tmp
        arr = np.ndarray((batch_size, element_size))
        index = 0
        for i in li:
            tmp = i.shape[1] if len(i.shape) > 1 else 1
            arr[:, index: index + tmp] = i
            index += tmp
        self.batch.append(arr)
class Sdforstacksrn(Save_data):
    def batch_cb(self, model, *args):
        arr = np.array(self.batch)
        arr = np.transpose(arr, (1, 0, 2))
        y = args[0].detach().numpy()
        arr = np.concatenate([arr, y], axis=2)
        self.epoch.append(arr)
        self.batch = list()

    def step_cb(self, model, *args):
        x = args[0].detach().numpy()
        yp = args[1].detach().numpy()
        read = args[2].detach().numpy()
        hidden = args[3].detach().numpy()
        s1 = model._s1.detach().numpy()
        s2 = model._s2.detach().numpy()
        u = model._u.detach().numpy()
        v1 = model._v1.detach().numpy()
        v2 = model._v2.detach().numpy()
        batch_size = model.batch_size
        element_size = 0
        li = [ u, hidden, x, s1, s2, read, v1, v2, yp]
        for i in li:
            tmp = i.shape[1] if len(i.shape) > 1 else 1
            element_size += tmp
        arr = np.ndarray((batch_size, element_size))
        index = 0
        for i in li:
            tmp = i.shape[1] if len(i.shape) > 1 else 1
            arr[:, index: index + tmp] = i
            index += tmp
        self.batch.append(arr)
if __name__ == "__main__":
    sd = Save_data(a=0.03, b= 2)
    print(sd.a)