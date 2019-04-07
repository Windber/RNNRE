'''
@author: lenovo
'''
import torch
import torch.nn as nn
import pandas as pd
import time
import sys
from profile import *
class RNN(torch.nn.Module):
    def __init__(self, params):
        super(RNN, self).__init__()
        self.input_size = params["input_size"]
        self.hidden_size = params["hidden_size"]
        self.output_size = params["output_size"]
        self.device = params["device"]
        self.rnn = nn.RNNCell(self.input_size, self.hidden_size)
        self.hidden = None
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.rnn.to(self.device)
        self.linear.to(self.device)
        #self.sm = nn.Softmax()
    def init(self, batch_size):
        self.hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
    def forward(self, x):
        self.hidden = self.rnn(x, self.hidden)
        #output = self.sm(self.linear(self.hidden))
        output = self.linear(self.hidden)
        return output

class Task:
    def __init__(self, params):
        self.trpath = params["trpath"]
        self.tepath = params["tepath"]
        self.input_size = params["input_size"]
        self.hidden_size = params["hidden_size"]
        self.output_size = params["output_size"]
        self.device = params["device"]
        self.model = RNN(params)
        self.model.to(self.device)
        self.trtx, self.trty, self.tetx, self.tety = self.get_data()
        self.cel = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters())
        self.batch_size = params["batch_size"]
        self.load_path = params["load_path"]
        self.min_eloss = 10000
        self.state = None
        self.epoch = params["epoch"]
        self.task = params["task"]
        if self.load_path:
            self.state, self.min_eloss = torch.load(self.load_path)
            self.model.load_state_dict(self.state)
    def init(self):
        #self.batch_size = batch_size
        self.model.init(self.batch_size)
    def get_data(self):
        trdata = pd.read_csv(self.trpath, header=None, index_col=None)
        trdx = trdata[0].values.tolist()
        trdy = trdata[1].values.tolist()
        xmap = {"0": [0], "1": [1], "s": [2], "e": [3], "#": [3]}
        ymap = {"0": 0, "1": 1}
        trdx = [list(map(lambda x: xmap[x], s)) for s in trdx]
        trdy = [list(map(lambda x: ymap[x], s)) for s in trdy]
        trtx = torch.Tensor(trdx).long()
        trty = torch.Tensor(trdy).long().to(self.device)
        input_size = 4
        total = len(trdx)
        steps = len(trdx[0])
        trtx = torch.zeros(total, steps, input_size).scatter_(2, trtx, 1.).to(self.device)
        
        tedata = pd.read_csv(self.tepath, header=None, index_col=None)
        tedx = tedata[0].values.tolist()
        tedy = tedata[1].values.tolist()
        xmap = {"0": [0], "1": [1], "s": [2], "e": [3], "#": [3]}
        ymap = {"0": 0, "1": 1}
        tedx = [list(map(lambda x: xmap[x], s)) for s in tedx]
        tedy = [list(map(lambda x: ymap[x], s)) for s in tedy]
        tetx = torch.Tensor(tedx).long()
        tety = torch.Tensor(tedy).long().to(self.device)
        input_size = 4
        total = len(tedx)
        steps = len(tedx[0])
        tetx = torch.zeros(total, steps, input_size).scatter_(2, tetx, 1.).to(self.device)
        
        return trtx, trty, tetx, tety
    def train(self):
        epoch = 2
        batchs = self.trtx.shape[0] // self.batch_size
        for e in range(epoch):
            queue = torch.randperm(self.trtx.shape[0])
            epoch_loss = 0
            for b in range(batchs):
                batch_start = b * self.batch_size
                batch_end = (b + 1) * self.batch_size
                xs = self.trtx[queue[batch_start: batch_end]]
                ys = self.trty[batch_start: batch_end]
                epoch_loss += self.perbatch(xs, ys)
            epoch_loss = epoch_loss / batchs
            if epoch_loss <= self.min_eloss:
                self.state = self.model.state_dict()
                self.min_eloss = epoch_loss
            print("Train epoch %d Loss: %f" % (e, epoch_loss))
        torch.save([self.state, self.min_eloss], "../../savedmodel/" + self.task + "_BestModel@" + time.strftime("%d%H%M"))
    def perbatch(self, xs, ys, istraining=True):
        batch_loss = 0
        steps = xs.shape[1]
        self.model.init(self.batch_size)
        for s in range(steps):
            x = xs[:, s, :]
            yp = self.model(x)
            batch_loss += self.cel(yp, ys[:, s])
        if istraining:
            self.optim.zero_grad()
            batch_loss.backward()
            self.optim.step()
        #print("batch_loss: %f" % (batch_loss.item()))
        return batch_loss
    def test(self):
        batchs = self.tetx.shape[0] // self.batch_size
        steps = self.tetx.shape[1]
        queue = torch.randperm(self.tetx.shape[0])
        epoch_loss = 0
        for b in range(batchs):
            batch_start = b * self.batch_size
            batch_end = (b + 1) * self.batch_size
            xs = self.tetx[queue[batch_start: batch_end]]
            ys = self.tety[batch_start: batch_end]
            epoch_loss += self.perbatch(xs, ys)
        print("Test Loss: %f" % ( epoch_loss / batchs))
if __name__ == '__main__':
    configname = sys.argv[1]
    config = globals()[configname]
    start = time.time()
    t = Task(config)
    t.train()
    t.test()
    print("time: %f" % (time.time()-start))