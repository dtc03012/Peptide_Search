import pandas as pd
import sqlite3 as sql


# reader = mgf.read("/home/kimtaehoon/desktop/Peptide_Source/01625b_GA2-TUM_first_pool_9_01_01-3xHCD-1h-R1.mgf")

conn = sql.connect("dtc")
cur = conn.cursor()
database = pd.read_excel('database.xlsx',sheet_name='Sheet1')

for i in range(len(database)):
    Raw_file = database.loc[i]["Raw file"]
    Scan_number = database.loc[i]["Scan number"]
    Scan_number = str(Scan_number)
    Sequence = database.loc[i]["Sequence"]
    Length = database.loc[i]["Length"]
    Length = str(Length)
    Missed_cleavages = database.loc[i]["Missed cleavages"]
    Missed_cleavages = str(Missed_cleavages)
    Charge = database.loc[i]["Charge"]
    Charge = str(Charge)
    mz = database.loc[i]["m/z"]
    Mass = database.loc[i]["Mass"]
    Score = database.loc[i]["Score"]

    cur.execute('insert into pepdv(raw_file,scan_number,sequence,length,missed_cleavages,charge,mz,mass,score) values(?,?,?,?,?,?,?,?,?)',(Raw_file,Scan_number,Sequence,Length,Missed_cleavages,Charge,mz,Mass,Score))

conn.commit()
conn.close()