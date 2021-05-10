from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import Transformer as tfr

num_shape = 512

ion_w2v = KeyedVectors.load_word2vec_format("ion_w2v")
amino_w2v = KeyedVectors.load_word2vec_format("amino_w2v")
data = pd.read_csv("data.csv")
print(len(data["mz"]))
print(len(data["seq"]))
mz_mx_len = 0
seq_mx_len = 0
for i in data["mz"]:
    mz_array = np.fromstring(i[1:-1], dtype=float, sep=' ')
    mz_mx_len = max(mz_mx_len,len(mz_array))

for i in data["seq"]:
    seq_mx_len = max(seq_mx_len,len(i))
seq_mx_len += 2
zero = []
for i in range(num_shape):
    zero.append(0)
input_array = []
output_array = []
cnt = 0
total = len(data["mz"])
load_count = np.zeros((101))
mx_val = 0
for i in data["mz"]:
    mz_array = np.fromstring(i[1:-1], dtype=float, sep=' ')
    vv = []
    for p in mz_array:
        rval = p * 10
        rval = round(rval)
        mx_val = max(mx_val,rval)
        vv.append(rval)
    diff = mz_mx_len - len(mz_array)
    for p in range(diff):
        vv.append(0)
    input_array.append(vv)
    if cnt%50 == 0:
        per = int(round((cnt*100/total)))
        if per%10 == 0 and load_count[per] == 0:
            print("{} % process....".format(round(cnt*100/total),-1))
            load_count[per] = 1
    cnt = cnt + 1

print("Succes input array")
time.sleep(1)

load_count = np.zeros((101))
cnt = 0
for i in data["seq"]:
    vv = []
    vv.append(1)
    for p in i:
        vv.append(ord(p)-ord('A') + 3)
    vv.append(2)
    diff = seq_mx_len - len(i) - 2
    for p in range(diff):
        vv.append(0)
    output_array.append(vv)
    if cnt%50 == 0:
        per = int(round((cnt*100/total)))
        if per %10 == 0 and load_count[per] == 0:
            print("{} % process....".format(round(cnt * 100 / total), -1))
            load_count[per] = 1
    cnt = cnt + 1

print("Success output array")

print("input size : {} , output size :{}".format(len(input_array),len(output_array)))

print("go into Transformer")
time.sleep(1)

print("max_value : {}".format(mx_val))
dmodel = 512
num_layer = 6
num_head = 8
dff = 2048
dropout = 0.3
input_size = 20000
output_size = 30
epoch = 20
BATCH_SIZE = 20
BUFFER_SIZE = 20000
input_array = np.array(input_array)
output_array = np.array(output_array)

ATCH_SIZE = 64
BUFFER_SIZE = 20000
print(seq_mx_len)
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': input_array,
        'dec_inputs': output_array[:, :-1]
    },
    {
        'outputs': output_array[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()
model = tfr.transformer(vocab_size=input_size,
                        num_layers=num_layer,
                        dff=dff,
                        d_model=dmodel,
                        num_heads=num_head,
                        dropout=dropout)
learning_rate = tfr.CustomSchedule(dmodel)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, seq_mx_len - 1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
  print(y_true)
  y_true = tf.reshape(y_true, shape=(-1, seq_mx_len - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
model.fit(dataset,epochs=epoch)