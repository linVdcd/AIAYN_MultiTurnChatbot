# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from modules import *
import os, codecs
from tqdm import tqdm


class Graph():
    def __init__(self, is_training=True):
        if is_training:
            self.x1, self.x2, self.x3, self.y, self.num_batch = get_batch_data()  # (N, T)
        else:  # inference
            self.x1 = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.x2 = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.x3 = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

        # define decoder inputs
        self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>

        # Load vocabulary
        de2idx, idx2de = load_de_vocab()
        en2idx, idx2en = load_en_vocab()

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc1, self.enc2, self.enc3 = embedding_en(self.x1, self.x2, self.x3,
                                                           vocab_size=len(de2idx),
                                                           num_units=hp.hidden_units,
                                                           scale=True,
                                                           scope="enc_embed")

            ## Positional Encoding
            pe1, pe2, pe3 = embedding_en(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x1)[1]), 0), [tf.shape(self.x1)[0], 1]),
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x2)[1]), 0),
                        [tf.shape(self.x2)[0], 1]),
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x3)[1]), 0),
                        [tf.shape(self.x3)[0], 1]),
                vocab_size=hp.maxlen,
                num_units=hp.hidden_units,
                zero_pad=False,
                scale=False,
                scope="enc_pe")
            self.enc1 += pe1
            self.enc2 += pe2
            self.enc3 += pe3

            ## Dropout
            self.enc1 = tf.layers.dropout(self.enc1,
                                          rate=hp.dropout_rate,
                                          training=tf.convert_to_tensor(is_training))
            self.enc2 = tf.layers.dropout(self.enc2,
                                          rate=hp.dropout_rate,
                                          training=tf.convert_to_tensor(is_training))
            self.enc3 = tf.layers.dropout(self.enc3,
                                          rate=hp.dropout_rate,
                                          training=tf.convert_to_tensor(is_training))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc1, self.enc2, self.enc3 = multihead_attention_en(queries1=self.enc1, queries2=self.enc2,
                                                                             queries3=self.enc3,
                                                                             keys1=self.enc1, keys2=self.enc2,
                                                                             keys3=self.enc3,
                                                                             num_units=hp.hidden_units,
                                                                             num_heads=hp.num_heads,
                                                                             dropout_rate=hp.dropout_rate,
                                                                             is_training=is_training,
                                                                             causality=False)

                    ### Feed Forward
                    self.enc1 = feedforward(self.enc1, num_units=[4 * hp.hidden_units, hp.hidden_units])
                    self.enc2 = feedforward(self.enc2, num_units=[4 * hp.hidden_units, hp.hidden_units], reuse=True)
                    self.enc3 = feedforward(self.enc3, num_units=[4 * hp.hidden_units, hp.hidden_units], reuse=True)
            self.enc = self.enc1 + self.enc2 + self.enc3

        # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            self.dec = embedding(self.decoder_inputs,
                                 vocab_size=len(en2idx),
                                 num_units=hp.hidden_units,
                                 scale=True,
                                 scope="dec_embed")

            ## Positional Encoding
            self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                                          [tf.shape(self.decoder_inputs)[0], 1]),
                                  vocab_size=hp.maxlen,
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=False,
                                  scope="dec_pe")

            ## Dropout
            self.dec = tf.layers.dropout(self.dec,
                                         rate=hp.dropout_rate,
                                         training=tf.convert_to_tensor(is_training))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.dec,
                                                   num_units=hp.hidden_units,
                                                   num_heads=hp.num_heads,
                                                   dropout_rate=hp.dropout_rate,
                                                   is_training=is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    ## Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.enc,
                                                   num_units=hp.hidden_units,
                                                   num_heads=hp.num_heads,
                                                   dropout_rate=hp.dropout_rate,
                                                   is_training=is_training,
                                                   causality=False,
                                                   scope="vanilla_attention")

                    ## Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])

        # Final linear projection
        self.logits = tf.layers.dense(self.dec, len(en2idx))
        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
        self.topk = tf.nn.top_k(tf.log(tf.nn.softmax(self.logits)), hp.topk)
        self.top = tf.nn.top_k(tf.log(tf.nn.softmax(self.logits)), 1)






if __name__ == '__main__':
    # Load vocabulary
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    # Construct graph
    g = Graph("train");
    print("Graph loaded")

    # Start session
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop(): break
            for step in range(g.num_batch):
                sess.run(g.train_op)
                if step % 20 == 0:
                    print('epoch:' + str(epoch) + '----' + 'step:' + str(step) + '----loss:' + str(
                        sess.run(g.mean_loss)) + '----acc:' + str(sess.run(g.acc)))

            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")


