# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:46:49 2018
word2vec
@author: Administrator
"""
import jieba
from gensim.models import word2vec
import pandas as pd
import nltk
import re
import yaml
from collections import Counter

class w2v(object):
    def __init__(self, conf_file):
        self.conf_file = open(conf_file)
        self.conf = yaml.load(self.conf_file)
        self.file_train = self.conf["trainfile"]
        self.file_test = self.conf["testfile"]
        self.file_w2vcorpus = self.conf["outputfile"]["w2vcorpus"]
        self.file_w2vmodel = self.conf["outputfile"]["w2vmodel"]
        self.file_stopwords = self.conf["outputfile"]["stopwords"]
        self.file_lfw = self.conf["outputfile"]["lowfreqwords"]
        self.file_train_replaced = self.conf["outputfile"]["file_train_replaced"]
        self.file_test_replaced = self.conf["outputfile"]["file_test_replaced"]

    def clean_str(self,string):
        string = re.sub(r"[A-Za-z(),!?.\'\"，。！\‘\’\“\”]", '', string)
        return string


    # 文本分词，并去除停用词,返回str类型
    def seg_str(self,string):
        # 使用结巴分词
        string = self.clean_str(string)
        seg = list(jieba.cut(string, cut_all=False))
        stopwords_list = []
        final = ""
        f = open(self.file_stopwords)
        # 去除停用词，停用词表为哈工大停用词表https://github.com/goto456/stopwords.git
        for line in f:
            stopwords_list.append(line.strip())
        for i in seg:
            if i not in stopwords_list:
                final += i
                final += ' '
            else:
                pass
        f.close()
        return final


    # 统计低频词，频率小于2的都算作低频词
    def get_lfwords(self):
        f_lfw = open(self.file_lfw, 'w')
        records = pd.read_csv(self.file_train)
        records = records.dropna()
        words = []
        for index, record in records.iterrows():
            context = record["context"]
            context_list = context.split()
            for word in context_list:
                words.append(word)
        word_counts = Counter(words)
        for word in set(word_counts):
            if word_counts[word] < 2:
                f_lfw.write(word + '\n')
        f_lfw.close()
        return word_counts


    # 使用“lfwords” 替换低频词
    def replace_lfw(self,file_origin, file_replaced):
        word_counts = self.get_lfwords()
        records = pd.read_csv(file_origin)
        records = records.dropna()
        file_replace = open(file_replaced,"w")
        temp = "_id,content,label\n"
        temp1 = "content,label\n"
        if "_id" in records.columns:
            file_replace.write(temp)
        else:
            file_replace.write(temp1)
        for index, record in records.iterrows():
            context = record["context"].strip()
            label = int(record["label"])
            seg = context.split()
            context_replaced = ""
            for word in seg:
                if word_counts[word]<0:
                    context_replaced += "lfwords"
                    context_replaced += ' '
                else:
                    context_replaced += word
                    context_replaced += ' '
            if "_id" in record:
                file_replace.write(str(record["_id"])+','+context_replaced + ',' + str(label) + '\n')
            else:
                file_replace.write(context_replaced + ',' + str(label) + '\n')
        file_replace.close()


    def replace_train(self):
        self.replace_lfw(self.file_train, self.file_train_replaced)
        records = pd.read_csv(self.file_train_replaced)
        records = records.dropna()
        print("dropna之后的df长度",records.shape)
        w2vcorpus = open(self.file_w2vcorpus, 'w')
        for index, record in records.iterrows():
            context = record["content"]
            w2vcorpus.write(context + '\n')
        w2vcorpus.close()


    def replace_test(self):
        self.replace_lfw(self.file_test, self.file_test_replaced)


    # 训练w2v模型
    def train_w2v(self):
        print("w2v模型开始训练")
        sentence = word2vec.LineSentence(self.file_w2vcorpus)
        w2vmodel = word2vec.Word2Vec(sentence, size=100, window=5, min_count=0, workers=20, iter=100)
        w2vmodel.save(self.file_w2vmodel)
        # model=word2vec.Word2Vec.load(w2vmodel)
        print("w2v模型训练完成")
        return w2vmodel

    #get_lfwords()
    #replace_train()
    #replace_test()
    #train_w2v()
