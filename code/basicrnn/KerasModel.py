import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.experimental import PeepholeLSTMCell
import pandas as pd
import tensorflow.keras.regularizers as regularizers
import numpy as np
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
import smtplib
import time
from email.mime.text import MIMEText
class Task:
    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
        self.get_data()
    def __getattr__(self, name):
        if name in self.params.keys():
            return self.params[name]
        else:
            raise AttributeError
    def get_data(self):
        cw = self.class_weight
        if self.debug:
            trpath =self.train_path + self.task_name + "_test1"
        else:
            trpath =self.train_path + self.task_name + "_train"
        tepath_prefix = self.test_path + self.task_name + "_test"
        testn = self.test_num
        inpalpha = self.alphabet
        labalpha = self.classes
        train_df = pd.read_csv(trpath, dtype={1: str}, index_col=None, header=None)
        train_lx = train_df.iloc[:, 0].values.tolist()
        train_ly = train_df.iloc[:, 1].values.tolist()
        train_x = [list(map(lambda x: inpalpha[x], i)) for i in train_lx]
        train_y = [list(map(lambda x: labalpha[x], i)) for i in train_ly]
        trx_np = np.array(train_x)
        try_np = np.array(train_y)
        self.trx = tf.one_hot(trx_np, 4)
        self.trl = tf.one_hot(try_np, 2)
#         tryw_np = np.vectorize(cw.get)(try_np)
        self.tex_l = list()
        self.tel_l = list()
        for i in range(testn):
            test_df = pd.read_csv(tepath_prefix + str(i+1), dtype={1: str}, index_col=None, header=None)
            test_lx = test_df.iloc[:, 0].values.tolist()
            test_ly = test_df.iloc[:, 1].values.tolist()
            test_x = [list(map(lambda x: inpalpha[x], i)) for i in test_lx]
            test_y = [list(map(lambda x: labalpha[x], i)) for i in test_ly]
            tex_np = np.array(test_x)
            tey_np = np.array(test_y)
            tex = tf.one_hot(tex_np, 4)
            tel = tf.one_hot(tey_np, 2)
            self.tex_l.append(tex)
            self.tel_l.append(tel)
    def experiment(self):
        RNN = keras.layers.RNN
        if not self.onlytest:
            cell = PeepholeLSTMCell(self.hidden_size, input_shape=(32, 4))
            model = keras.Sequential([RNN(cell, return_sequences=True),
                                           layers.Activation('tanh'),
                                           layers.Dense(self.hidden_size),
                                     keras.layers.Dense(self.output_size),
                                     keras.layers.Activation('softmax')])
            model.compile(optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'],
                         sample_weight_mode="temporal")
            count = 0
            while count == 0 or his.history['loss'][0] > self.loss_thred and count < self.epochs:
                his = model.fit(self.trx, self.trl, epochs=1, validation_data=(self.tex_l[self.valid_test], self.tel_l[self.valid_test]), steps_per_epoch=200, validation_steps=20)
                count += 1
            task_name = self.task_name + "@" + self.cell_name + "@" + time.strftime("%d%H%M%S")
            #model.save_weights(self.save_path + task_name + ".h5", True, "h5")
        for timestep, tex, tel in zip(self.test_timestep, self.tex_l, self.tel_l):
            cell = PeepholeLSTMCell(self.hidden_size, input_shape=(timestep, 4))
            emodel = keras.Sequential([RNN(cell, return_sequences=True),
                                       layers.Activation('tanh'),
                                       layers.Dense(self.hidden_size),
                                     layers.Dense(self.output_size),
                                     keras.layers.Activation('softmax')])
            if self.onlytest:
                emodel.load_weights(self.load_path)
            else:
                emodel.set_weights(model.get_weights())
            emodel.compile(optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
            a = emodel.evaluate(tex, tel, steps=5)
            print(a)
        self.emailto()
    def emailto(self):
        text = "Task " + self.task_name + " completed."
        msg = MIMEText(text, 'plain', 'utf-8')
        smtp = smtplib.SMTP()
        smtp.connect("smtp.126.com")
        smtp.login("guolipengyeah", "217000mh")
        smtp.sendmail("guolipengyeah@126.com", ["guolipengyeah@126.com"], msg.as_string())
        smtp.quit()
