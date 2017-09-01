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
from train import Graph
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

def eval():



    g = Graph(is_training=False)
    print("Graph loaded")


    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    x1 = encoding(de2idx,[])
    x2 = encoding(de2idx,[])
    hp.batch_size = 1
    beam_size=hp.topk
    #     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]

    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
            # fi = codecs.open('corpora/input','r','utf-8')
            # inputs = fi.readlines()
            # fi.close()
            #
            # fo =codecs.open('chatbotres.txt','w','utf-8')


            while 1:
                #fo.write('input> '+input1+'\n')
                input1=raw_input('input>').decode('utf-8')

                if input1=='':
                    print('换个话题')
                    x1=encoding(de2idx,'')
                    x2 = encoding(de2idx,'')

                    continue

                input1 = ' '.join(jieba.cut(input1)).split(' ')
                x3 = encoding(de2idx,input1)


                ## Autoregressive inference
                # preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                # for j in range(hp.maxlen):
                #     _preds = sess.run(g.preds,{g.x1:x1,g.x2:x2,g.x3:x3,g.y:preds})
                #     preds[:,j] = _preds[:,j]
                # got = " ".join(idx2en[idx] for idx in preds[0]).split("</S>")[0].strip()
                # print('res:'+got)
                #
                # x1 = x3
                # x2 = encoding(de2idx,got.split(' '))




                preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                _preds = sess.run(g.topk, {g.x1: x1,g.x2:x2,g.x3:x3, g.y: preds})
                hyp=[]
                result=[]

                for i in range(beam_size):
                    if _preds[1][0,0,i]==3:
                        result.append([[_preds[1][0,0,i]],_preds[0][0,0,i]])
                    else:
                        hyp.append([[_preds[1][0,0,i]],_preds[0][0,0,i]])

                rs = []
                for j in range(1,hp.maxlen):
                    if len(result)==beam_size:
                        break
                    X1 = np.zeros((len(hyp),hp.maxlen))
                    X2 = np.zeros((len(hyp), hp.maxlen))
                    X3 = np.zeros((len(hyp), hp.maxlen))
                    preds = np.zeros((len(hyp), hp.maxlen), np.int32)
                    for i in range(len(hyp)):
                        X1[i,:] = x1[0,:]
                        X2[i, :] = x2[0, :]
                        X3[i, :] = x3[0, :]
                        seq = hyp[i][0]
                        preds[i,:j]=seq
                    _preds = sess.run(g.top, {g.x1: X1,g.x2:X2,g.x3:X3, g.y: preds})

                    tmphyp=[]

                    s = []
                    for i in range(len(hyp)):
                        for k in range(1):

                            seq =  hyp[i][0]+[_preds[1][i,j,k]]

                            if _preds[1][i,j,k]==3:
                                result.append([seq,hyp[i][1]])
                                rs.append(hyp[i][1])
                            else:
                                tmphyp.append([seq,hyp[i][1]+_preds[0][i,j,k]])
                                s.append(hyp[i][1]+_preds[0][i,j,k])
                    index = list(reversed(np.argsort(s)))
                    #tmphyp =tmphyp[index]
                    hyp=[]

                    for i in range(len(s)):
                        try:
                            hyp.append(tmphyp[index[i]])
                        except Exception as e:
                            break
                    #hyp = tmphyp
                index = list(reversed(np.argsort(rs)))


                output = []
                for i in range(0,len(index)):
                    if en2idx['<UNK>'] in result[index[i]][0]:
                        continue
                    output.append(result[index[i]][0])
                oi = random.randint(0, len(output) - 1)
                for i in range(0,len(output)):
                    try:
                        seq = output[i]
                        #print(result[index[i]][1])
                        got = " ".join(idx2en[idx] for idx in seq)
                        if i==0:
                            out = got

                        #print(got)
                    except Exception as e:
                        continue
                print('res>'+''.join(out.split(' ')[:-1]))
                x1 = x3
                x2 = encoding(de2idx,out.split(' ')[:-1])


if __name__ == '__main__':
    eval()
    print("Done")

