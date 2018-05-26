# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn import metrics
import yaml
import pickle


class Evaluate(object):
    def __init__(self, conf_file):
        self.conf_file = open(conf_file)
        self.conf = yaml.load(self.conf_file)
        self.prefile = self.conf["outputfile"]["prefile"]
        self.bestrefile = self.conf["outputfile"]["bestrefile"]

    def roc_auc(self, y_true, score):
        fpr, tpr, thresholds = roc_curve(y_true, score, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return fpr, tpr, auc;

    def get_metrics(self, y_true, y_pre):
        rp = np.array(metrics.precision_recall_fscore_support(y_true, y_pre)).T
        print(type(rp),rp.shape,rp)
        avg = (rp[0, :] * rp[0, -1] + rp[1, :] * rp[1, -1]) / np.sum(rp[:, -1])
        avg[-1] = np.sum(rp[:, -1])
        rp = np.vstack((rp, np.array([avg])))
        df = pd.DataFrame(rp, columns=['precision', 'recall', 'f1-score', 'support'],
                          index=['负例(0)', '正例(1)', '平均'])
        return df

    def evaluate(self, y_true, y_pre, score):
        df_metrics = self.get_metrics(y_true, y_pre)
        fpr, tpr, auc = self.roc_auc(y_true, score)
        auc = auc
        return df_metrics, auc

    def print_metrics(self, df_metrics, auc):
        print('-' * 50)
        print('测试集评估结果:')
        print('-' * 50)
        print('-' * 50)
        print(df_metrics)
        print('-' * 50)
        print('auc:', auc)

    def best_thres(self, y_true, y_pre, score):
        thresholds = np.arange(0.0, 1.0, 0.01)
        acclist = []
        for i in thresholds:
            y_pre[score <= i] = 0
            acc = metrics.accuracy_score(y_true, y_pre)
            acclist.append(acc)
        ind = np.argmax(acclist)
        thres = thresholds[ind]
        #print("threshold:", thres)
        return thres

    def get_params(self, df_metrics, auc):
        pre_n = df_metrics["precision"][0]
        pre_p = df_metrics["precision"][1]
        pre_avg = df_metrics["precision"][2]
        re_n = df_metrics["recall"][0]
        re_p = df_metrics["recall"][1]
        re_avg = df_metrics["recall"][2]
        f1_n = df_metrics["f1-score"][0]
        f1_p = df_metrics["f1-score"][1]
        f1_avg = df_metrics["f1-score"][2]
        neg_num = df_metrics["support"][0]
        pos_num = df_metrics["support"][1]
        sample_num = df_metrics["support"][1]
        dict_params = {"pre_n":pre_n, "pre_p": pre_p, "pre_avg": pre_avg, "re_n": re_n, "re_p": re_p, "re_avg": re_avg,
             "f1_n": f1_n, "f1_p": f1_p, "f1_avg": f1_avg, "neg_num": neg_num, "pos_num": pos_num,
             "sample_num": sample_num, "auc": auc}
        return dict_params

    def get_re(self, prefile):
        df = pd.read_csv(prefile)
        df = df[df["y_true"] != -1]
        df = df.reset_index(drop=True)
        y_true = df["y_true"]
        score = df["score"]
        y_pre = df["y_pre"]
        y_precopy = np.array(y_pre)
        thres = self.best_thres(y_true, y_pre, score)
        y_precopy[score < thres] = 0
        df_metrics, auc = self.evaluate(y_true, y_precopy, score)
        dict_params = self.get_params(df_metrics, auc)
        return dict_params, thres, df_metrics,auc

    def get_best(self):
        dict_best, thres_best,df_metrics_best,auc_best = self.get_re(self.prefile)
        flag=1
        for i in range(2, 11):
            prefile = self.prefile + '_' + str(i)
            dict_params, thres, df_metrics,auc = self.get_re(prefile)
            if dict_params["auc"] > dict_best["auc"]:
                dict_best = dict_params
                thres_best = thres
                df_metrics_best = df_metrics
                auc_best = auc
                flag = i
        file_bestre = open(self.bestrefile, 'ab')
        epoch = "epoch "+str(flag)
        pickle.dump([epoch, dict_best,thres_best],file_bestre)
        file_bestre.close()
        print("epoch: ",str(flag))
        print("thres_best: ",str(thres_best))
        print(df_metrics_best)
        print("auc_best: ",auc_best)
