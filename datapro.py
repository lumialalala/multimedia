# -*- coding: utf-8 -*-
"""
Created on Fri May  4 20:59:08 2018
谣言分类
@author: Administrator
"""

import pandas as pd
import numpy as np
import re
import jieba
import glob


def clean_str(string):
    string = re.sub(r"[A-Za-z0-9(),!?.\'\"，。！\‘\’\“\”]", '', string)
    return string


# 文本分词，并去除停用词,返回str类型
def seg_str(string):
    # 使用结巴分词
    string = clean_str(string)
    seg = list(jieba.cut(string, cut_all=False))
    stopwords_list = []
    final = ""
    f = open("./data/哈工大停用词表.txt", encoding="utf-8")
    # 去除停用词，停用词表为哈工大停用词表https://github.com/goto456/stopwords.git
    for line in f:
        stopwords_list.append(line.strip())
    for i in seg:
        if i not in stopwords_list:
            final += i
            final += ' '
        else:
            pass
    return final


def load_data(file, label):
    f = open(file, "rb")
    line = f.readline()
    ind = 0
    texts = []
    _ids = []
    while line:
        if ind % 3 == 0:
            line = line.decode("utf-8")
            _id = line.split('|')[0]
            _ids.append(_id)
        elif ind % 3 == 2:
            line = line.decode("utf-8")
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            line = seg_str(line)
            texts.append(line)
        else:
            pass
        line = f.readline()
        ind += 1
    f.close()
    length = int(ind / 3)
    labels = [label] * length
    d = {"_id": _ids, "context": texts, "label": labels}
    df = pd.DataFrame(d)
    # 3783条非谣言
    return df


def concat_df(df1, df2):
    frames = [df1, df2]
    df = pd.concat(frames)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def load_train_data():
    # 3783条非谣言
    # 3748条谣言
    f = open("data/train_set.txt", "w")
    df1 = load_data("tweets/train_nonrumor.txt", "__label__nonrumor")
    df2 = load_data("tweets/train_rumor.txt", "__label__rumor")
    df = concat_df(df1, df2)
    for index, row in df.iterrows():
        _id = row["_id"]
        context = row["context"]
        label = row["label"]
        context = context.encode("gbk", 'ignore')
        context = context.decode("gbk", 'ignore')
        f.write(_id + " " + context + " " + label + "\n")
    f.close()


def load_test_data():
    df1 = load_data("tweets/test_nonrumor.txt", "__label__nonrumor")
    df2 = load_data("tweets/test_rumor.txt", "__label__rumor")
    df = concat_df(df1, df2)
    f = open("data/test_set.csv", "w")
    f.write("_id,context,labels")
    for index, row in df.iterrows():
        _id = row["_id"]
        context = row["context"]
        label = row["label"]
        context = context.encode("gbk", 'ignore')
        context = context.decode("gbk", 'ignore')
        f.write(_id + " " + context + " " + label + "\n")
    f.close()


#load_train_data()
load_test_data()
