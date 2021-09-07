"""
created by Weijie

Jul 2021
"""

import tensorflow as tf

# scores [bsz, 1+k]: 1 positive sample and k negative samples
def build_binaryCEloss(pos_scores, neg_scores): # [bsz, n], [bsz, p]
    loss = - tf.log_sigmoid(pos_scores) + tf.reduce_sum(tf.nn.softplus(neg_scores) / neg_scores.get_shape().as_list()[2], axis=-1)
#    loss = - tf.reduce_sum(tf.log_sigmoid(pos_scores)/pos_scores.shape.as_list()[1], axis=-1) + \
#           tf.reduce_sum(tf.nn.softplus(neg_scores) / neg_scores.shape.as_list()[1], axis=-1)
    return loss


def build_ranking_loss(pos_scores, neg_scores, batch_size):
    """ loss defined in 
    "Session-based Recommendations with Recurrent Neural Networks"
    """
    # normalized to 1 firstly
    pos_scores_mask = 1.0 - tf.cast(tf.math.equal(pos_scores, 0), tf.float32)
    neg_scores_mask = 1.0 - tf.cast(tf.math.equal(neg_scores, 0), tf.float32)
    loss = []
    for i in range(batch_size):
        neg_score_sum = tf.reduce_sum(neg_scores[i] * neg_scores_mask[i]) / (tf.reduce_sum(neg_scores_mask[i]) + 1.0)
        obj = tf.reduce_sum(tf.sigmoid(neg_score_sum - pos_scores[i]) * pos_scores_mask[i]) / (tf.reduce_sum(pos_scores_mask[i])+ 1.0)
        reg = tf.reduce_sum(tf.sigmoid(neg_scores[i] ** 2) * neg_scores_mask[i]) / (tf.reduce_sum(neg_scores_mask[i]) + 1.0)
        loss.append(obj + reg)
    return tf.reduce_mean(loss)

def _build_binaryCEloss(pos_scores, neg_scores):
    pos_scores = tf.where(tf.is_nan(pos_scores), tf.zeros_like(pos_scores), pos_scores)
    neg_scores = tf.where(tf.is_nan(neg_scores), tf.zeros_like(neg_scores), neg_scores)

    loss = - tf.log_sigmoid(pos_scores[:, 0]) \
           + tf.reduce_sum(tf.nn.softplus(neg_scores) / neg_scores.shape.as_list()[1], axis=-1)
    return tf.reduce_mean(loss)

def build_weighted_binaryCEloss(scores, temperature):
    pos_scores = scores[:, 0]
    neg_scores = scores[:, 1:]
    weight = tf.nn.softmax(neg_scores / temperature, axis=-1)

    loss = - tf.log_sigmoid(pos_scores) + tf.reduce_sum(tf.nn.softplus(neg_scores) * weight, axis=-1)
    return tf.reduce_mean(loss)

def build_weighted_prob_binaryCEloss(scores, probs, temperature):
    pos_scores = scores[:, 0]
    neg_scores = scores[:, 1:]
    weight = tf.nn.softmax(neg_scores / temperature - tf.log(probs), axis=-1)

    loss = - tf.log_sigmoid(pos_scores) + tf.reduce_sum(tf.nn.softplus(neg_scores) * weight, axis=-1)
    return tf.reduce_mean(loss)

import numpy as np
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def softplus(x):
    return np.log(1 + np.exp(x))
