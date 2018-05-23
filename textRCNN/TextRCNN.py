# -*- coding: utf-8 -*-
from evaluate import Evaluate
from RCNNmodel import RCNN
from w2v import w2v
import yaml
import pandas as pd
import numpy as np
import pickle
from keras.utils import plot_model


class TextRCNN(object):
    def __init__(self,conf_file):
        self.conf_file = conf_file
        conf_open=open(conf_file)
        self.conf = yaml.load(conf_open)
        #第一次巡练模型存储的模型权重地址
        self.file_RCNNmodel = self.conf["outputfile"]["file_RCNNmodel"]
        self.file_test_replaced = self.conf["outputfile"]["file_test_replaced"]
        self.prefile = self.conf["outputfile"]["prefile"]


    def get_class(self):
        conf_file = self.conf_file
        rcnnclass = RCNN(self.conf_file)
        return rcnnclass

    def get_model(self):
        rcnnclass = self.get_class()
        model = rcnnclass.RCNNmodel
        return model

    def load_train_data(self):
        rcnnclass = self.get_class()
        train_data, target = rcnnclass.load_train_data()
        return train_data,target

    #第一次训练
    def train(self):
        rcnnclass = self.get_class()
        rcnnclass.train_model()
        rcnnclass.test_model()

    #再次训练从上一轮开始，每轮训练结束后保存模型权重
    def re_train(self):
        rcnnclass = self.get_class()
        train_data, train_target = rcnnclass.load_train_data()
        test_data, test_target = rcnnclass.load_test_data()
        for i in range(1,10):
            j = i + 1
            print("第",str(j),"轮训练中")
            if i==1:
                old_modelpath = self.file_RCNNmodel
                new_modelpath = self.file_RCNNmodel+'_'+str(j)
                pre_file = self.prefile + '_' + str(j)
                new_model = rcnnclass.re_train(old_modelpath,train_data,train_target,new_modelpath)
                df_pre=self.test(new_model,test_data)
                df_pre.to_csv(pre_file, sep=',', index=False)
            else:
                old_modelpath = self.file_RCNNmodel + '_' + str(i)
                new_modelpath = self.file_RCNNmodel + '_' + str(j)
                pre_file = self.prefile + '_' + str(j)
                new_model = rcnnclass.re_train(old_modelpath, train_data,train_target, new_modelpath)
                df_pre = self.test(new_model, test_data)
                df_pre.to_csv(pre_file, sep=',', index=False)
            print("第", str(j),"轮训练结束")

    def test(self,model,test_data):
        re=model.predict(test_data)
        df = pd.read_csv(self.file_test_replaced)
        _id = df["_id"]
        label = df["label"]
        content = df["content"]
        re1 = re[:, 1]
        y_pre = np.ones_like(_id)
        dfnew = pd.DataFrame({"_id": _id, "content": content, "y_pre": y_pre, "score": re1, "y_true": label})
        return dfnew

    def evaluate(self):
        evalu = Evaluate(self.conf_file)
        evalu.get_best()

    def w2vprocess(self):
        w2vmodel=w2v(self.conf_file)
        w2vmodel.get_lfwords()
        w2vmodel.replace_train()
        w2vmodel.replace_test()
        w2vmodel.train_w2v()
    
    def plot_model(self):
        RCNNclass = self.get_class()
        RCNNclass.plot_model()

conf_file="../conf/config2.yaml"
model=TextRCNN(conf_file)
print("w2v开始训练")
#model.w2vprocess()
print("w2v训练完成，开始训练RCNN模型")
#model.train()
print("RCNN模型第一轮训练完成，开始retrain")
#model.re_train()
print("retrain完成，开始评估")
model.evaluate()
#model.plot_model()


