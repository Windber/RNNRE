'''
@author: lenovo
'''
import torch
import torch.nn as nn
import pandas as pd
import time
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNNCell(self.input_size, self.hidden_size)
        self.hidden = None
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.rnn.to(device)
        self.linear.to(device)
        #self.sm = nn.Softmax()
    def init(self, batch_size):
        self.hidden = torch.zeros(batch_size, self.hidden_size).to(device)
    def forward(self, x):
        self.hidden = self.rnn(x, self.hidden)
        #output = self.sm(self.linear(self.hidden))
        output = self.linear(self.hidden)
        return output

class Task:
    def __init__(self, load_path=None):
        self.trpath = "../../data/tomita_dealed/T1_test1"
        self.tepath = "../../data/tomita_dealed/T1_test1"
        input_size = 4
        hidden_size = 2
        output_size = 2
        self.model = RNN(input_size, hidden_size, output_size)
        self.model.to(device)
        self.trtx, self.trty, self.tetx, self.tety = self.get_data()
        self.cel = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters())
        self.batch_size = 5
        self.load_path = load_path
        self.min_eloss = 10000
        self.state = None
        self.epoch = 10
        self.task = "Tomita1"
        if self.load_path:
            self.state, self.min_eloss = torch.load(self.load_path)
            self.model.load_state_dict(self.state)
    def init(self, batch_size):
        self.batch_size = batch_size
        self.model.init(batch_size)
    def get_data(self):
        trdata = pd.read_csv(self.trpath, header=None, index_col=None)
        trdx = trdata[0].values.tolist()
        trdy = trdata[1].values.tolist()
        xmap = {"0": [0], "1": [1], "s": [2], "e": [3], "#": [3]}
        ymap = {"0": 0, "1": 1}
        trdx = [list(map(lambda x: xmap[x], s)) for s in trdx]
        trdy = [list(map(lambda x: ymap[x], s)) for s in trdy]
        trtx = torch.Tensor(trdx).long()
        trty = torch.Tensor(trdy).long().to(device)
        input_size = 4
        total = len(trdx)
        steps = len(trdx[0])
        trtx = torch.zeros(total, steps, input_size).scatter_(2, trtx, 1.).to(device)
        
        tedata = pd.read_csv(self.tepath, header=None, index_col=None)
        tedx = tedata[0].values.tolist()
        tedy = tedata[1].values.tolist()
        xmap = {"0": [0], "1": [1], "s": [2], "e": [3], "#": [3]}
        ymap = {"0": 0, "1": 1}
        tedx = [list(map(lambda x: xmap[x], s)) for s in tedx]
        tedy = [list(map(lambda x: ymap[x], s)) for s in tedy]
        tetx = torch.Tensor(tedx).long()
        tety = torch.Tensor(tedy).long().to(device)
        input_size = 4
        total = len(tedx)
        steps = len(tedx[0])
        tetx = torch.zeros(total, steps, input_size).scatter_(2, tetx, 1.).to(device)
        
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
    device = torch.device("cuda: 0")
    start = time.time()
    t = Task()
    t.train()
    t.test()
    print("time: %f" % (time.time()-start))