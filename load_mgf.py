import pandas as pd
import numpy as np
import sqlite3 as sql
from pyteomics import mgf
mgf_file1 = mgf.read("/home/kimtaehoon/desktop/Peptide_Source/01625b_GA1-TUM_first_pool_1_01_01-3xHCD-1h-R1.mgf")
mgf_file2 = mgf.read("/home/kimtaehoon/desktop/Peptide_Source/01625b_GA2-TUM_first_pool_9_01_01-3xHCD-1h-R1.mgf")

conn = sql.connect("dtc")
cur = conn.cursor()

def parsing(x):
    raw_file = ""
    scan_number = ""
    type = 0
    for i in x:
        if i == '.':
            break
        raw_file += i
    x = reversed(x)
    for i in x:
        if i == '=':
            break
        if i.isdigit() :
            scan_number += i
    return raw_file , scan_number

output = pd.DataFrame(columns=["mz","seq"])
for i in mgf_file1:
    mz = i["m/z array"]
    title = i["params"]["title"]
    raw_file , scan_number = parsing(title)
    cur.execute("select sequence from pepdv where raw_file = '{}'and scan_number = '{}'".format(raw_file,scan_number))
    xx = cur.fetchone()
    if xx == None:
        continue
    sequence = xx[0]
    new_data = {
        "mz" : mz,
        "seq" : sequence
    }
    output = output.append(new_data,ignore_index=True)

for i in mgf_file2:
    mz = i["m/z array"]
    title = i["params"]["title"]
    raw_file , scan_number = parsing(title)
    cur.execute("select sequence from pepdv where raw_file = '{}'and scan_number = '{}'".format(raw_file,scan_number))
    xx = cur.fetchone()
    if xx == None:
        continue
    sequence = xx[0]
    new_data = {
        "mz": mz,
        "seq": sequence
    }
    output = output.append(new_data, ignore_index=True)

output.to_csv("data.csv",index=False)





