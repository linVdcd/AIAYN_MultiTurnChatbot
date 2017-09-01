# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os
import jieba
import tensorflow as tf
import numpy as np
import  random
from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from test_model import Graph
from nltk.translate.bleu_score import corpus_bleu

def encoding(de2idx,x):
    size = len(x)
    i=0
    X = np.zeros((1, hp.maxlen))
    while i < size:
        try:
            X[:, i] = de2idx[x[i]]
            i += 1
        except Exception as e:
            print(x[i] + ' can not fund in the dict')
            X[:, i] = de2idx['<UNK>']
            i += 1
            continue
    X[:, i] = 3
    return X




class API():
    def __init__(self):
        self.g = Graph(is_training=False)
        print("Graph loaded")
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self.sess, tf.train.latest_checkpoint(hp.logdir))
        print('restored')
        self.de2idx, self.idx2de = load_de_vocab()
        self.en2idx, self.idx2en = load_en_vocab()



        hp.batch_size = 1
        self.beam_size = hp.topk



    def query(self,query):
        input1 = query[2].decode('utf-8')
        x1 = encoding(self.de2idx, ' '.join(jieba.cut(query[0].decode('utf-8'))).split(' '))
        x2 = encoding(self.de2idx, ' '.join(jieba.cut(query[1].decode('utf-8'))).split(' '))

        input1 = ' '.join(jieba.cut(input1)).split(' ')
        x3 = encoding(self.de2idx, input1)

        preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
        _preds = self.sess.run(self.g.topk, {self.g.x1: x1, self.g.x2: x2, self.g.x3: x3, self.g.y: preds})
        hyp = []
        result = []

        for i in range(self.beam_size):
            if _preds[1][0, 0, i] == 3:
                result.append([[_preds[1][0, 0, i]], _preds[0][0, 0, i]])
            else:
                hyp.append([[_preds[1][0, 0, i]], _preds[0][0, 0, i]])

        rs = []
        for j in range(1, hp.maxlen):
            if len(result) == self.beam_size:
                break
            X1 = np.zeros((len(hyp), hp.maxlen))
            X2 = np.zeros((len(hyp), hp.maxlen))
            X3 = np.zeros((len(hyp), hp.maxlen))
            preds = np.zeros((len(hyp), hp.maxlen), np.int32)
            for i in range(len(hyp)):
                X1[i, :] = x1[0, :]
                X2[i, :] = x2[0, :]
                X3[i, :] = x3[0, :]
                seq = hyp[i][0]
                preds[i, :j] = seq
            _preds = self.sess.run(self.g.top, {self.g.x1: X1, self.g.x2: X2, self.g.x3: X3, self.g.y: preds})

            tmphyp = []

            s = []
            for i in range(len(hyp)):
                for k in range(1):

                    seq = hyp[i][0] + [_preds[1][i, j, k]]

                    if _preds[1][i, j, k] == 3:
                        result.append([seq, hyp[i][1]])
                        rs.append(hyp[i][1])
                    else:
                        tmphyp.append([seq, hyp[i][1] + _preds[0][i, j, k]])
                        s.append(hyp[i][1] + _preds[0][i, j, k])
            index = list(reversed(np.argsort(s)))
            # tmphyp =tmphyp[index]
            hyp = []

            for i in range(len(s)):
                try:
                    hyp.append(tmphyp[index[i]])
                except Exception as e:
                    break
                    # hyp = tmphyp
        index = list(reversed(np.argsort(rs)))

        output = []
        for i in range(0, len(index)):
            if self.en2idx['<UNK>'] in result[index[i]][0]:
                continue
            output.append(result[index[i]][0])
        # oi = random.randint(0, len(output) - 1)
        for i in range(0, len(output)):
            try:
                seq = output[i]
                # print(result[index[i]][1])
                got = " ".join(self.idx2en[idx] for idx in seq)
                if i == 0:
                    out = got

                    # print(got)
            except Exception as e:
                continue
        return out.split(' ')[:-1]










