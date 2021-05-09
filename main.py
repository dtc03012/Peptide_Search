from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import time

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

zero = []
for i in range(num_shape):
    zero.append(0)
input_array = []
output_array = []
cnt = 0
total = len(data["mz"])
load_count = np.zeros((101))
for i in data["mz"]:
    mz_array = np.fromstring(i[1:-1], dtype=float, sep=' ')
    vv = []
    for p in mz_array:
        rval = p * 10
        rval = round(rval)
        try:
            xx = np.array(ion_w2v.get_vector(str(rval)))
        except:
            print("{} 번째 mz 값 없음".format(rval))
        vv.append([xx])
    diff = mz_mx_len - len(vv)
    for p in range(diff):
        vv.append([zero])
    input_array.append([vv])
    if cnt%50 == 0:
        per = int(round((cnt*100/total)))
        if load_count[per] == 0:
            print("{} % process....".format(round(cnt*100/total),-1))
            load_count[per] = 1
    cnt = cnt + 1

print("Succes input array")
time.sleep(1)

load_count = np.zeros((101))
cnt = 0
for i in data["seq"]:
    vv = []
    for p in i:
        xx = amino_w2v.get_vector(p)
        vv.append([xx])
    diff = seq_mx_len - len(vv)
    for p in range(diff):
        vv.append([zero])
    output_array.append([vv])
    if cnt%50 == 0:
        per = int(round((cnt*100/total)))
        if load_count[per] == 0:
            print("{} % process....".format(round(cnt * 100 / total), -1))
            load_count[per] = 1
    cnt = cnt + 1

print("Success output array")

print("input size : {} , output size :{}".format(len(input_array),len(output_array)))

