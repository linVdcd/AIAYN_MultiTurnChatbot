# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs


def embedding_en(inputs1,inputs2,inputs3,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs1 = tf.nn.embedding_lookup(lookup_table, inputs1)
        outputs2 = tf.nn.embedding_lookup(lookup_table, inputs2)
        outputs3 = tf.nn.embedding_lookup(lookup_table, inputs3)
        if scale:
            outputs1 = outputs1 * (num_units ** 0.5)
            outputs2 = outputs2 * (num_units ** 0.5)
            outputs3 = outputs3 * (num_units ** 0.5)

    return outputs1,outputs2,outputs3


def multihead_attention_en(queries1,queries2,queries3,
                        keys1,keys2,keys3,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries1.get_shape().as_list[-1]

        # Linear projections
        Q1 = tf.layers.dense(queries1, num_units, activation=tf.nn.relu,name='d1')  # (N, T_q, C)
        K1 = tf.layers.dense(keys1, num_units, activation=tf.nn.relu,name = 'd2')  # (N, T_k, C)
        V1 = tf.layers.dense(keys1, num_units, activation=tf.nn.relu,name = 'd3')  # (N, T_k, C)

        Q2 = tf.layers.dense(queries2, num_units, activation=tf.nn.relu, name='d1',reuse=True)  # (N, T_q, C)
        K2 = tf.layers.dense(keys2, num_units, activation=tf.nn.relu, name='d2',reuse=True)  # (N, T_k, C)
        V2 = tf.layers.dense(keys2, num_units, activation=tf.nn.relu, name='d3',reuse=True)  # (N, T_k, C)

        Q3 = tf.layers.dense(queries3, num_units, activation=tf.nn.relu, name='d1',reuse=True)  # (N, T_q, C)
        K3 = tf.layers.dense(keys3, num_units, activation=tf.nn.relu, name='d2',reuse=True)  # (N, T_k, C)
        V3 = tf.layers.dense(keys3, num_units, activation=tf.nn.relu, name='d3',reuse=True)  # (N, T_k, C)


        # Split and concat
        Q_1 = tf.concat(tf.split(Q1, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_1 = tf.concat(tf.split(K1, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_1 = tf.concat(tf.split(V1, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        Q_2 = tf.concat(tf.split(Q2, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_2 = tf.concat(tf.split(K2, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_2 = tf.concat(tf.split(V2, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        Q_3 = tf.concat(tf.split(Q3, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_3 = tf.concat(tf.split(K3, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_3 = tf.concat(tf.split(V3, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)



        # Multiplication
        outputs1 = tf.matmul(Q_1, tf.transpose(K_1, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs2 = tf.matmul(Q_2, tf.transpose(K_2, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs3 = tf.matmul(Q_3, tf.transpose(K_3, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs1 = outputs1 / (K_1.get_shape().as_list()[-1] ** 0.5)
        outputs2 = outputs2 / (K_2.get_shape().as_list()[-1] ** 0.5)
        outputs3 = outputs3 / (K_3.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks1 = tf.sign(tf.abs(tf.reduce_sum(keys1, axis=-1)))  # (N, T_k)
        key_masks1 = tf.tile(key_masks1, [num_heads, 1])  # (h*N, T_k)
        key_masks1 = tf.tile(tf.expand_dims(key_masks1, 1), [1, tf.shape(queries1)[1], 1])  # (h*N, T_q, T_k)

        key_masks2 = tf.sign(tf.abs(tf.reduce_sum(keys2, axis=-1)))  # (N, T_k)
        key_masks2 = tf.tile(key_masks2, [num_heads, 1])  # (h*N, T_k)
        key_masks2 = tf.tile(tf.expand_dims(key_masks2, 1), [1, tf.shape(queries2)[1], 1])  # (h*N, T_q, T_k)

        key_masks3 = tf.sign(tf.abs(tf.reduce_sum(keys3, axis=-1)))  # (N, T_k)
        key_masks3 = tf.tile(key_masks3, [num_heads, 1])  # (h*N, T_k)
        key_masks3 = tf.tile(tf.expand_dims(key_masks3, 1), [1, tf.shape(queries3)[1], 1])  # (h*N, T_q, T_k)



        paddings1 = tf.ones_like(outputs1) * (-2 ** 32 + 1)
        outputs1 = tf.where(tf.equal(key_masks1, 0), paddings1, outputs1)  # (h*N, T_q, T_k)

        paddings2 = tf.ones_like(outputs2) * (-2 ** 32 + 1)
        outputs2 = tf.where(tf.equal(key_masks2, 0), paddings2, outputs2)  # (h*N, T_q, T_k)

        paddings3 = tf.ones_like(outputs3) * (-2 ** 32 + 1)
        outputs3 = tf.where(tf.equal(key_masks3, 0), paddings3, outputs3)  # (h*N, T_q, T_k)
        # Causality = Future blinding
        if causality:
            diag_vals3 = tf.ones_like(outputs3[0, :, :])  # (T_q, T_k)
            tril3 = tf.contrib.linalg.LinearOperatorTriL(diag_vals3).to_dense()  # (T_q, T_k)
            masks3 = tf.tile(tf.expand_dims(tril3, 0), [tf.shape(outputs3)[0], 1, 1])  # (h*N, T_q, T_k)

            diag_vals1 = tf.ones_like(outputs1[0, :, :])  # (T_q, T_k)
            tril1 = tf.contrib.linalg.LinearOperatorTriL(diag_vals1).to_dense()  # (T_q, T_k)
            masks1 = tf.tile(tf.expand_dims(tril1, 0), [tf.shape(outputs1)[0], 1, 1])  # (h*N, T_q, T_k)

            diag_vals2 = tf.ones_like(outputs2[0, :, :])  # (T_q, T_k)
            tril2 = tf.contrib.linalg.LinearOperatorTriL(diag_vals2).to_dense()  # (T_q, T_k)
            masks2 = tf.tile(tf.expand_dims(tril2, 0), [tf.shape(outputs2)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings1 = tf.ones_like(masks1) * (-2 ** 32 + 1)
            outputs1 = tf.where(tf.equal(masks1, 0), paddings1, outputs1)  # (h*N, T_q, T_k)

            paddings2 = tf.ones_like(masks2) * (-2 ** 32 + 1)
            outputs2 = tf.where(tf.equal(masks2, 0), paddings2, outputs2)  # (h*N, T_q, T_k)

            paddings3 = tf.ones_like(masks3) * (-2 ** 32 + 1)
            outputs3 = tf.where(tf.equal(masks3, 0), paddings3, outputs3)  # (h*N, T_q, T_k)




        # Activation
        outputs1 = tf.nn.softmax(outputs1)  # (h*N, T_q, T_k)
        outputs2 = tf.nn.softmax(outputs2)  # (h*N, T_q, T_k)
        outputs3 = tf.nn.softmax(outputs3)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks1 = tf.sign(tf.abs(tf.reduce_sum(queries1, axis=-1)))  # (N, T_q)
        query_masks1 = tf.tile(query_masks1, [num_heads, 1])  # (h*N, T_q)
        query_masks1 = tf.tile(tf.expand_dims(query_masks1, -1), [1, 1, tf.shape(keys1)[1]])  # (h*N, T_q, T_k)
        outputs1 *= query_masks1  # broadcasting. (N, T_q, C)

        query_masks2 = tf.sign(tf.abs(tf.reduce_sum(queries2, axis=-1)))  # (N, T_q)
        query_masks2 = tf.tile(query_masks2, [num_heads, 1])  # (h*N, T_q)
        query_masks2 = tf.tile(tf.expand_dims(query_masks2, -1), [1, 1, tf.shape(keys2)[1]])  # (h*N, T_q, T_k)
        outputs2 *= query_masks2  # broadcasting. (N, T_q, C)

        query_masks3 = tf.sign(tf.abs(tf.reduce_sum(queries3, axis=-1)))  # (N, T_q)
        query_masks3 = tf.tile(query_masks3, [num_heads, 1])  # (h*N, T_q)
        query_masks3 = tf.tile(tf.expand_dims(query_masks3, -1), [1, 1, tf.shape(keys3)[1]])  # (h*N, T_q, T_k)
        outputs3 *= query_masks3  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs1 = tf.layers.dropout(outputs1, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs1 = tf.matmul(outputs1, V_1)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs1 = tf.concat(tf.split(outputs1, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs1 += queries1

        # Normalize
        outputs1 = normalize(outputs1)  # (N, T_q, C)

        # Dropouts
        outputs2 = tf.layers.dropout(outputs2, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs2 = tf.matmul(outputs2, V_2)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs2 = tf.concat(tf.split(outputs2, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs2 += queries2

        # Normalize
        outputs2 = normalize(outputs2)  # (N, T_q, C)

        # Dropouts
        outputs3 = tf.layers.dropout(outputs3, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs3 = tf.matmul(outputs3, V_3)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs3 = tf.concat(tf.split(outputs3, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs3 += queries3

        # Normalize
        outputs3 = normalize(outputs3)  # (N, T_q, C)

    return outputs1,outputs2,outputs3
def multihead_attention(queries,
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
    
    

            