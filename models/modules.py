"""
Jul. 2021 by Weijie

building blocks for Transformer
"""

import numpy as np
import tensorflow as tf


def TransformerEncoder(inputs,
                       num_layers,
                       src_masks,
                       num_heads,
                       nhid_ffn,
                       d_model,
                       dropout_rate,
                       training,
                       causality=False):
    """ return encoder outputs with shape (N, T1, d_model)

    :param num_layers: number of blocks of self-attention blocks
    :param src_masks:
    :param num_heads : number of heads of self-attention
    :param nhid_ffn : number of units of ffn hidden layers.
    :param dropout_rate:
    :param training : used in  dropout
    :return:
    """
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        enc = inputs

        for i in range(num_layers):
            with tf.variable_scope("num_layer_".format(i), reuse=tf.AUTO_REUSE):
                enc = multihead_attention(queries=enc,
                                          keys=enc,
                                          values=enc,
                                          key_masks=src_masks,
                                          d_model=d_model,
                                          dropout_rate=dropout_rate,
                                          num_heads=num_heads,
                                          training=training,
                                          causality=causality) # same shape with queries
                #print(enc.shape)
                enc = ffn(enc, num_units=nhid_ffn, d_model=d_model, scope='positionalwise_feedforward_'.format(i))
        memory = enc
    return memory

def TransformerDecoder(inputs, num_layers, ):
    """ not defined yet

    :param num_layers:
    :return:
    """
    return

def layer_normalization(inputs, epsilon=1e-8, scope='layer_normalization'):
    """ Apply layer normalization
    :param inputs: A tensor with 2 or more dimensions, where the first dimesnion has 'batch_size'.
    :param epsilon: A float number. usually very small number for preventinb zero-division error
    :param scope: optional scope for 'variable_scope'.
    :return: A tensor with the same shape and data type as 'inputs'
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def get_token_embeddings(vocab_size, num_units, name, zero_pad=True):
    """Constructs token embedding matrix.

    :param vocab_size: The number of different one-hot vector. V.
    :param num_units: embedding dimensionality. E.
    :param zero_pad: If True, all the values of the first row (id = 0) should be constant zero
                      To apply query/key masks easily, zero pad is turned on.
    :return: weight variables: (V, E)
    """
    with tf.variable_scope('shared_weight_matrix'):
        embeddings = tf.get_variable(dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     name=name)
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)
    return embeddings


def scaled_dot_product_attention(Q, K, V,
                                 key_masks,
                                 causality=True,
                                 dropout_rate=0.5,
                                 training=True,
                                 scope='scaled_dot_product_attention'):
    """

    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.

    output: A 3d tensor with shape [N, T_q, d_v]
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs

    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"
    :return:

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """

    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    # elif type in ("q", "query", "queries"):
    #     # Generate masks
    #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    #
    #     # Apply masks to inputs
    #     outputs = inputs*masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def multihead_attention(queries, keys, values,
                        key_masks,
                        d_model,
                        dropout_rate,
                        num_heads=8,
                        training=True,
                        causality=False,
                        scope='multihead_attention'):
    """Apply multihead attention

    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    :return:  A 3d tensor with shape of (N, T_q, d_model)
    """

    #print("queires {}, keys {}, values {}".format(queries.shape,keys.shape,values.shape))
    #d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True)
        K = tf.layers.dense(keys, d_model, use_bias=True)
        V = tf.layers.dense(values, d_model, use_bias=True)

        # split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # attention

        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, d_model)
        # residual connection

        outputs += queries

        outputs = layer_normalization(outputs)# same shape with queries

    return outputs




def ffn(inputs, num_units, d_model, scope='positionalwise_feedforward'):
    """positional-wise feed forward net.

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: the number of units of the hidden layer
    scope: Optional scope for `variable_scope`.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    """
    #d_model = inputs.get_shape().as_list()[-1] # the output units should be of the same size with inputs
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units, activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, d_model, activation=tf.nn.relu)

        outputs += inputs
        outputs = layer_normalization(outputs)
    return outputs



def label_smoothing(inputs, epsilon=0.1):
    """Apply label smoothing.

    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
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
    """
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)




def positional_encoding(inputs,
                        maxlen=200,
                        masking=True,
                        scope='positional_encoding'):
    """Sinusodial Positional encoding

    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    """
    E = inputs.get_shape().as_list()[-1]
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        # first part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)
        ])
        # second part: apply the cosine to even columns and sin to odds
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        # look up
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
    return tf.to_float(outputs)



def noam_scheme(init_lr, global_step, warm_up=4000.):
    """Noam scheme learning rate decay

    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases until it reaches init_lr.

    :return:
    """

    pass

def attention(H, dropout_rate):
    """利用Attention机制对concat所有encoder的输出
    inputs
    H : the output of the encoder [bsz, T, E]

    outputs: concat all the timesteps: [bsz, E]
    """
    # 获得最后一层LSTM的神经元数量
    # hiddenSize = config.model.hiddenSizes[-1]
    _, sequence_length, hiddenSize = H.get_shape().as_list()#self._nb_hidden
    # 初始化一个权重向量，是可训练的参数
    W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
    # 对LSTM的输出用激活函数做非线性转换
    M = tf.tanh(H)
    # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
    # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
    newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

    # 对newM做维度转换成[batch_size, time_step]
    restoreM = tf.reshape(newM, [-1, sequence_length])

    # 用softmax做归一化处理[batch_size, time_step]
    alpha = tf.nn.softmax(restoreM)

    # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, sequence_length, 1]))

    # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
    sequeezeR = tf.reshape(r, [-1, hiddenSize])

    sentenceRepren = tf.tanh(sequeezeR)

    # 对Attention的输出可以做dropout处理
    output = tf.nn.dropout(sentenceRepren, 1 - dropout_rate)

    return output

def target_aware_attention_decoder(candidates, H,
                                   key_masks,
                                   dropout_rate,
                                   training=True,
                                   causality=False):
    """target-aware attention decoder defined (utilized) in GeoSAN (kdd 2020.)
    # 原文是对每一个step的输出和所有的candidate做匹配，
    # 这里我们先对所有的steps做attention,然后只用该attention和candidate做匹配

    :param candidates: candidate representations: [bsz, (1+k), E]
    :param H: output of the encoder [bsz, T, E]
    :param value:
    :return: [bsz, E]
    """

    assert candidates.get_shape().as_list()[-1] == H.get_shape().as_list()[-1],\
        "The embedding size should be the same!"

    #Fl = attention(H, dropout_rate) #[bsz, E]
    #_, number_candidates, hidden_szie = candidates.get_shape().as_list()
    #W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
    attn_outputs = multihead_attention(candidates, H, H,
                                       key_masks,
                                       dropout_rate,
                                       training,
                                       causality)

    attn_outputs = attention(attn_outputs, dropout_rate)

    return attn_outputs











