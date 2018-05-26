import pandas as pd
import numpy as np

f=pd.read_csv("../v2.0/data/prefile/outputfile.csv")
print(f.head(5))
for index, row in f.iterrows():
    if row["y_true"]==0:
        print("aaa")
