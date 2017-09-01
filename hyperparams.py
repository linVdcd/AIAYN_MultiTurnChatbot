# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'corpora/100w.p'
    target_train = 'corpora/100w.r'
    source_test = 'corpora/xhj.p'
    target_test = 'corpora/xhj.r'
    
    # training
    batch_size = 64 # alias = N
    lr = 0.00001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir_8' # log directory
    
    # model
    maxlen = 14 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 5 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.05
    minlen=6
    topk=10
    
    

