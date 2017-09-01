# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Index
    x1_list,x2_list,x3_list, y_list, Sources, Targets =[],[], [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        source_sent = source_sent.split('\t')

        x1 = [de2idx.get(word, 1) for word in (source_sent[0] + u" </S>").split()]  # 1: OOV, </S>: End of Text
        x2 = [de2idx.get(word, 1) for word in (source_sent[1] + u" </S>").split()]  # 1: OOV, </S>: End of Text
        x3 = [de2idx.get(word, 1) for word in (source_sent[2] + u" </S>").split()]  # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x1), len(y)) <=hp.maxlen and len(y)>=hp.minlen and len(x2) <=hp.maxlen and len(x3) <=hp.maxlen:
            x1_list.append(np.array(x1))
            x2_list.append(np.array(x2))
            x3_list.append(np.array(x3))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X1 = np.zeros([len(x1_list), hp.maxlen], np.int32)
    X2 = np.zeros([len(x2_list), hp.maxlen], np.int32)
    X3 = np.zeros([len(x3_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x1,x2,x3, y) in enumerate(zip(x1_list,x2_list,x3_list, y_list)):
        X1[i] = np.lib.pad(x1, [0, hp.maxlen-len(x1)], 'constant', constant_values=(0, 0))
        X2[i] = np.lib.pad(x2, [0, hp.maxlen - len(x2)], 'constant', constant_values=(0, 0))
        X3[i] = np.lib.pad(x3, [0, hp.maxlen - len(x3)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))

    return X1,X2,X3, Y, Sources, Targets

def load_train_data():
    de_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    
    X1,X2,X3, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X1,X2,X3, Y
    
def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [line for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n")]
    en_sents = [line for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n")]
        
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X1, X2, X3, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X1) // hp.batch_size
    
    # Convert to tensor
    X1 = tf.convert_to_tensor(X1, tf.int32)
    X2 = tf.convert_to_tensor(X2, tf.int32)
    X3 = tf.convert_to_tensor(X3, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X1,X2,X3, Y])
            
    # create batch queues
    x1,x2,x3, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x1,x2,x3, y, num_batch # (N, T), (N, T), ()
