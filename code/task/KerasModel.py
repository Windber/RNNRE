import tensorflow as tf
import tensorflow.keras as keras

import pandas as pd

import numpy as np
hidden_dim = 4
thred = 0.01
Tomita = 1
RNN = keras.layers.LSTM
output_dim = 2
trpath = '../../data/tomita/T' + str(Tomita) + '_train'
tepath_prefix = '../../data/tomita/T' + str(Tomita) + '_test'
testn = 4
inpalpha = {'s': 2, 'e': 3, '#': 3, '0': 0, '1': 1}
labalpha = {'0': 0, '1': 1}
ts = [32, 64, 128, 256]
#RNN = keras.layers.LSTM
train_df = pd.read_csv(trpath, dtype={1: str}, index_col=None, header=None)
train_lx = train_df.iloc[:, 0].values.tolist()
train_ly = train_df.iloc[:, 1].values.tolist()
train_x = [list(map(lambda x: inpalpha[x], i)) for i in train_lx]
train_y = [list(map(lambda x: labalpha[x], i)) for i in train_ly]
trx_np = np.array(train_x)
try_np = np.array(train_y)
trx = tf.one_hot(trx_np, 4)
trl = tf.one_hot(try_np, 2)

tex_l = list()
tel_l = list()
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
    tex_l.append(tex)
    tel_l.append(tel)
model = keras.Sequential([RNN(hidden_dim, input_shape=(32, 4), return_sequences=True),
                         keras.layers.Dense(output_dim),
                         keras.layers.Activation('softmax')])

model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
his = model.fit(trx, trl, steps_per_epoch=200, epochs=1)
while his.history['loss'][0] > thred:
    his = model.fit(trx, trl, steps_per_epoch=200, epochs=1)


for timestep, tex, tel in zip(ts, tex_l, tel_l):
    emodel = keras.Sequential([RNN(hidden_dim, input_shape=(timestep, 4), return_sequences=True),
                             keras.layers.Dense(output_dim),
                             keras.layers.Activation('softmax')])
    emodel.set_weights(model.get_weights())
    emodel.compile(optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    a = emodel.evaluate(tex, tel, steps=5)
    print(a)
