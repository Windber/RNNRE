import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.experimental import PeepholeLSTMCell
import pandas as pd
import tensorflow.keras.regularizers as regularizers
import numpy as np
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
hidden_dim = 6
thred = 0.01
#Tomita = 6
Task = "Anbn"
reg = regularizers.l1_l2(l1=0.01, l2=0.1)
cell = PeepholeLSTMCell(hidden_dim, input_shape=(32, 4), kernel_regularizer=reg, dropout=0.3, recurrent_dropout=0.3)
RNN = keras.layers.RNN
output_dim = 2
#trpath = '../../data/tomita/T' + str(Tomita) + '_train'
#tepath_prefix = '../../data/tomita/T' + str(Tomita) + '_test'
# Abnbn cw
cw = {0: 0.5, 1: 0.5}
trpath = "../../data/countlanguage/" + Task + "_train"
tepath_prefix = "../../data/countlanguage/" + Task + "_test"
testn = 4
inpalpha = {'s': 2, 'e': 3, '#': 3, 'a': 0, 'b': 1}
#inpalpha = {'s': 2, 'e': 3, '#': 3, '(': 0, ')': 1}
labalpha = {'0': 0, '1': 1}
#ts = [32, 256, 256, 512]
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
tryw_np = np.vectorize(cw.get)(try_np)
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
model = keras.Sequential([RNN(cell, return_sequences=True),
                          layers.Dropout(0.3),
                               layers.Activation('tanh'),
                               layers.Dense(hidden_dim, kernel_regularizer=reg),
                               layers.Dropout(0.3),
                         keras.layers.Dense(output_dim, kernel_regularizer=reg),
                         keras.layers.Activation('softmax')])

model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'],
             )#sample_weight_mode="temporal")
count = 0
#his = model.fit(trx, trl, steps_per_epoch=200, epochs=1, sample_weight=[tryw_np])
while count == 0 or his.history['loss'][0] > thred and count < 50:
    his = model.fit(trx, trl, epochs=1, validation_data=(tex_l[1], tel_l[1]), steps_per_epoch=200, validation_steps=20, callbacks=[callbacks.EarlyStopping(monitor='val_acc', patience=10)])
    #his = model.fit(trx, trl, steps_per_epoch=200, epochs=1, sample_weight=[tryw_np])
    count += 1
task_name = "Anbn@LSTM"
model.save("smodel/Keras_" + task_name + ".h5")
for timestep, tex, tel in zip(ts, tex_l, tel_l):
    cell = PeepholeLSTMCell(hidden_dim, input_shape=(timestep, 4))
    emodel = keras.Sequential([RNN(cell, return_sequences=True),
                               layers.Dropout(0.3),
                               layers.Activation('tanh'),
                               layers.Dense(hidden_dim),
                               layers.Dropout(0.3),
                             layers.Dense(output_dim),
                             keras.layers.Activation('softmax')])
    emodel.set_weights(model.get_weights())
    emodel.compile(optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    a = emodel.evaluate(tex, tel, steps=5)
    print(a)
import smtplib
from email.mime.text import MIMEText
text = "Task " + Task + " completed."
msg = MIMEText(text, 'plain', 'utf-8')
smtp = smtplib.SMTP()
smtp.connect("smtp.126.com")
smtp.login("guolipengyeah", "217000mh")
smtp.sendmail("guolipengyeah@126.com", ["guolipengyeah@126.com"], msg.as_string())
smtp.quit()
