import numpy as np
import pandas as pd
import re

"""
# 将文件处理成textRCNN需要的格式
f = open("data/train_set.txt")
f1 = open("v2.0/data/train_RCNN.csv", "w")
f1.write("context,label\n")
for eachline in f:
    seg = eachline.split("__label__")
    try:
        context = seg[0]
        label = seg[1]
        context = re.sub(r"[0-9]", '', context)
        label = label.strip()
        if label == "rumor":
            sentiment = 0
        else:
            sentiment = 1
        
    except:
        print(eachline)
    f1.write(context.strip()+','+str(sentiment)+'\n')
f.close()
f1.close()
"""
df = pd.read_csv("data/test_set.csv",sep='\001')
print(df.head(5))
f1 = open("v2.0/data/test_RCNN.csv", "w")
f1.write("_id,context,label\n")
print(df.head(5))
i=0
for index,row in df.iterrows():
    _id=row["_id"]
    context = str(row["context"])
    label = row["label"]
    label = label.strip()
    if label == "__label__rumor":
        sentiment = 0
        i+=1
    else:
        sentiment = 1
    f1.write(str(_id)+','+context.strip()+','+str(sentiment)+'\n')
f1.close()
print(i)

