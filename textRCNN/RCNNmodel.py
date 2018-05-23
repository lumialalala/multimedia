# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from gensim import models
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import yaml
import pickle
from keras.utils import plot_model


class RCNN(object):
    def __init__(self, conf_file):
        self.conf_file = open(conf_file)
        self.conf = yaml.load(self.conf_file)
        self.prefile = self.conf["outputfile"]["prefile"]
        self.file_his = self.conf["outputfile"]["hisfile"]
        self.file_w2vmodel = self.conf["outputfile"]["w2vmodel"]
        self.file_train_replaced = self.conf["outputfile"]["file_train_replaced"]
        self.file_test_replaced = self.conf["outputfile"]["file_test_replaced"]
        self.file_RCNNmodel = self.conf["outputfile"]["file_RCNNmodel"]
        self.file_pre = self.conf["outputfile"]["prefile"]

    def RCNNmodel(self):
        word2vec = models.Word2Vec.load(self.file_w2vmodel)
        MAX_TOKENS = word2vec.syn0_lockf.shape[0]
        print("MAX_TOKENS: ", MAX_TOKENS)
        embedding_dim = 100
        hidden_dim_1 = 200
        hidden_dim_2 = 100
        NUM_CLASSES = 2
        embeddings = np.zeros((word2vec.syn0_lockf.shape[0] + 2, embedding_dim), dtype="float32")
        for w in word2vec.wv.vocab:
            embeddings[word2vec.wv.vocab[w].index] = word2vec.wv[w]
        document = Input(shape=(None,), dtype="int32")
        left_context = Input(shape=(None,), dtype="int32")
        right_context = Input(shape=(None,), dtype="int32")
        embedder = Embedding(MAX_TOKENS + 2, embedding_dim, weights=[embeddings], trainable=True)
        doc_embedding = embedder(document)
        l_embedding = embedder(left_context)
        r_embedding = embedder(right_context)
        forward = LSTM(hidden_dim_1, return_sequences=True)(l_embedding)
        backward = LSTM(hidden_dim_1, return_sequences=True, go_backwards=True)(r_embedding)
        backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)
        together = concatenate([forward, doc_embedding, backward], axis=2)
        semantic = TimeDistributed(Dense(hidden_dim_2, activation="tanh"))(together)
        pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)
        output = Dense(NUM_CLASSES, input_dim=hidden_dim_2, activation="softmax")(pool_rnn)
        model = Model(inputs=[document, left_context, right_context], outputs=output)
        model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    # sentence length: 30
    def load_data(self, df):
        texts = df["content"]
        target = df["label"]
        tokenlist = []
        left_list = []
        right_list = []
        word2vec = models.Word2Vec.load(self.file_w2vmodel)
        MAX_TOKENS = len(word2vec.wv.vocab)
        MAX_LEN = 30
        for text in texts:
            tokens = text.split()
           #lfwindex = word2vec.wv.vocab["lfwords"]
            tokens = [word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else MAX_TOKENS for token in tokens]
            doc_as_array = np.array(tokens)
            left_context_as_array = np.array([MAX_TOKENS+1] + tokens[:-1])
            right_context_as_array = np.array(tokens[1:] + [MAX_TOKENS+1])
            tokenlist.append(doc_as_array)
            left_list.append(left_context_as_array)
            right_list.append(right_context_as_array)
        tokenlist = pad_sequences(tokenlist, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post',
                                  value=MAX_TOKENS+1)
        left_list = pad_sequences(left_list, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post',
                                  value=MAX_TOKENS+1)
        right_list = pad_sequences(right_list, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post',
                                   value=MAX_TOKENS+1)
        target = to_categorical(target)
        doc_as_array = np.array(tokenlist)
        left_context_as_array = np.array(left_list)
        right_context_as_array = np.array(right_list)
        return [doc_as_array, left_context_as_array, right_context_as_array], target

    def load_train_data(self):
        df = pd.read_csv(self.file_train_replaced)
        train_data, target = self.load_data(df)
        return train_data,target

    def load_test_data(self):
        df = pd.read_csv(self.file_test_replaced)
        test_data, target = self.load_data(df)
        label = df["label"]
        return test_data, label

    def train_model(self):
        train_data, target = self.load_train_data()
        model = self.RCNNmodel()
        history = model.fit(train_data,target,epochs=1, verbose=2)
        file_his = open(self.file_his, 'wb')
        pickle.dump([history.epoch,history.history],file_his)
        file_his.close()
        model.save_weights(self.file_RCNNmodel)
        #print("history: ", history)
        #print(history.epoch)
        #print(history.history)
        #print("history: ", history)
        #model.save(self.file_RCNNmodel)
        #print("train finished")

    #retrain 加载上一轮训练的epoch的权重，然后继续训练，每次训练一轮，保存新模型。每次提供一个旧模型的路径，再提供一个新模型的存储地址（在TextRCNN中提供）
    def re_train(self, old_modelpath ,train_data,target,new_modelpath):
        model = self.RCNNmodel()
        model.load_weights(old_modelpath)
        history = model.fit(train_data,target,epochs=1, verbose=2)
        model.save_weights(new_modelpath)
        file_his = open(self.file_his, 'ab')
        pickle.dump([history.epoch,history.history],file_his)
        file_his.close()
        return model
        #print("history: ", history)
        #print(history.epoch)
        #print(history.history)
        #print("history: ", history)
        #print("saving model to:", self.file_newmodel)
        #model.save(self.file_newmodel)
        #print("retraining finished")


    def test_model(self):
        print("testing:")
        test_data, label = self.load_test_data()
        model = self.RCNNmodel()
        model.load_weights(self.file_RCNNmodel)
        re = model.predict(test_data)
        df = pd.read_csv(self.file_test_replaced)
        _id = df["_id"]
        label = df["label"]
        content = df["content"]
        re1 = re[:, 1]
        y_pre = np.ones_like(_id)
        print("saving to outputfile")
        dfnew = pd.DataFrame({"_id":_id, "content": content, "y_pre": y_pre, "score": re1, "y_true": label})
        dfnew.to_csv(self.file_pre, sep=',', index=False)

    def plot_model(self):
        model = self.RCNNmodel()
        plot_model(model, to_file = "model.png")

    #train_model()
    #re_train(file_RCNNmodel)
    #test_model(file_newmodel)
