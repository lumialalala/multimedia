import fastText as fasttext
import numpy as np
import pandas as pd
import yaml

def train_model():
    classifier=fasttext.train_supervised(input=ft_trainset,ws=25,lr=0.1,epoch=30,pretrainedVectors="data/w2vmodel.bin",dim=300)
    classifier.save_model(model)
    return classifier
def load_model():
    #首次训练模型，第二次及以后load模型
    classifier=train_model()
    #classifier=fasttext.load_model(model)
    #classifier.quantize(input=ft_trainset, qnorm=True,retrain=False,cutoff=100000)
    #classifier.save_model("model_2.ftz")
    return classifier
def predict():
    classifier=load_model()
    df=pd.read_csv(ft_testset,sep='\001')
    print("读入df的长度:",df.shape[0])
    output=open(prefile,"w")
    output.write("_id,context,y_true,y_pre,score\n")
    df = df.dropna()
    print("dropna之后df的长度",df.shape)
    for index,row in df.iterrows():
        _id=str(row["_id"])
        context=row['context']
        label=row['label']
        if label == "__label__nonrumor":
            y_true=1
        else:
            y_true=0
        re=classifier.predict(context,k=2)    
        labels=re[0]
        probas=re[1]
        for i in range(0,2):
            if labels[i]=='__label__nonrumor':
                score=probas[i]
                y_pre=1
                result=''.join(_id+','+context+','+str(y_true)+','+str(y_pre)+','+str(score)+'\n')
                output.write(result)
    output.close()

f=open("conf/config.yaml")
conf=yaml.load(f)
ft_trainset=conf["outputfile"]["ft_train"]
ft_testset=conf["outputfile"]["ft_test"]
prefile=conf["outputfile"]["prefile"]
model=conf["model"]
predict()
#load_model()
