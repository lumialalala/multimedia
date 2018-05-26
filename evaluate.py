import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn import metrics
import yaml

def roc_auc(y_true, score):
    fpr,tpr,thresholds=roc_curve(y_true,score, pos_label=1)
    auc=metrics.auc(fpr,tpr)
    return fpr, tpr, auc;

def get_metrics(y_true, y_pre):
    rp = np.array(metrics.precision_recall_fscore_support(y_true, y_pre)).T
    avg = (rp[0, :] * rp[0, -1] + rp[1, :] * rp[1, -1]) / np.sum(rp[:, -1])
    avg[-1] = np.sum(rp[:, -1])
    rp = np.vstack((rp, np.array([avg])))
    df = pd.DataFrame(rp, columns=['precision', 'recall', 'f1-score', 'support'],
                      index=['负例(0)', '正例(1)', '平均'])
    return df

def evaluate(y_true, y_pre, score):
    df_metrics = get_metrics(y_true,y_pre)
    fpr, tpr, auc = roc_auc(y_true,score)
    print('-' * 50)
    print('测试集评估结果:')
    print('-' * 50)
    print(df_metrics)
    print('-' * 50)
    print('auc:', auc)


def best_thres(y_true,y_pre,score):
    thresholds=np.arange(0.0,1.0,0.01)
    acclist=[]
    for i in thresholds:
        y_pre[score<=i]=0
        acc=metrics.accuracy_score(y_true,y_pre) 
        acclist.append(acc)
    ind=np.argmax(acclist)
    thres=thresholds[ind]
    print("acc",acclist[ind])
    print("threshold:" ,thres)
    return thres

def get_re():
    df=pd.read_csv(prefile)
    print("处理之前df长度", df.shape)
    df=df[df["y_true"]!=-1] 
    df=df.reset_index(drop=True)
    print("处理之后df长度",df.shape)
    y_true=df["y_true"]
    score=df["score"]
    y_pre=df["y_pre"]
    for i in y_true:
        if i==0:
            print(i)
    y_precopy=np.array(y_pre)
    thres=best_thres(y_true,y_pre,score)
    y_precopy[score<thres]=0
    evaluate(y_true,y_precopy,score)

f=open("conf/config2.yaml")
conf=yaml.load(f)
#thres=conf["thres"]
prefile=conf["outputfile"]["prefile"]
get_re()
