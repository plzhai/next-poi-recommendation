"""
written by Weijie

Aug 2021
codes for open datasets: brightkite, gowalla
"""

import tensorflow as tf
import numpy as np
import math
import sampler
from models.modules import get_token_embeddings, positional_encoding, TransformerEncoder, \
    target_aware_attention_decoder, attention, multihead_attention
import pickle
import os
import utils
import time
from loss import build_binaryCEloss, build_weighted_binaryCEloss, build_ranking_loss
from collections import Counter, namedtuple

class GeoSAN:
    """ adapted codes for KDD '20 paper
    "Geography-Aware Sequential Location Recommendation"
    """

    def __init__(self, config):
        # 1. settings for the inputs

        self._sequence_length = config['model_params']['sequence_length']  # the time steps, i.e., T.
        self._prediction_type = config['model_params']['prediction_type'] # or sequential
        self._num_negtive_samples = config['model_params']['num_negative_samples']
        # self._selected_features = config['model_params']['selected_feature_names']
        self.dataset_name = config['dataset']['name']

        self.n_user = config['dataset']['n_user']
        self.n_poi = config['dataset']['n_loc']
        self.n_time = config['dataset']['n_time']
        self.n_region = config['dataset']['n_region']
        self.user_embedding_size = config['dataset']['user_embedding_size']
        self.poi_embedding_size = config['dataset']['poi_embedding_size']
        self.time_embedding_size = config['dataset']['time_embedding_size']
        self.reg_embedding_size = config['dataset']['reg_embedding_size']

        self.use_user_embedding = True
        # 2. settings for the network

        self.nhid_ffn = config['model_params']['transformer']['nhid_ffn']
        self.num_encoder_blocks = config['model_params']['transformer']['num_encoder_blocks']
        self.d_model = config['model_params']['transformer']['d_model']
        self.nhead_enc = config['model_params']['transformer']['nhead_enc']
        # self.src_square_mask = config['model_params']['transformer']['mask']
        # self.src_binary_mask

        self._scaled = config['model_params']['transformer']['scaled']
        self.dropout_rate = config['model_params']['dropout_rate']  # default 0.5
        self._use_attention_as_decoder = config['model_params']['use_attention_as_decoder']  # default False
        self._weighted_loss = config['model_params']['loss']['weighted']  # default False
        self._temperature = config['model_params']['loss']['temperature']

        # 3. settings for training

        self._learning_rate = config['model_params']['learning_rate']
        self._nb_epoch = config['model_params']['nb_epoch']
        self._batch_size = config['model_params']['batch_size']
        self._train_max_patience = config['model_params']['max_patience']
        self._path_log = config['model_params']['path_log']
        self._dev_size = config['model_params']['dev_size']
        self._path_model = config['model_params']['path_model']
        self.training = True
        self.train_sampler = sampler.__getattribute__(config['dataset']['NegativeSampler'])(self.n_poi)

        # training log
        self.print_freq = 100
        self.training_dynamics = {"steps": [], 'top1_trn': [], 'hr_trn': [], 'ndcg_trn': [], 'top1_tst': [], 'hr_tst': [], 'ndcg_tst': []}

        # build model, loss and optimizer
        self.init_placeholder()
        self.scores = self.build_model()

        self.pos_scores = self.scores[:, :, 0]
        self.neg_scores = self.scores[:, :, 1:]

        if self._prediction_type == 'next':# scores :[bsz, 1+k]
            loss = build_binaryCEloss(self.pos_scores, self.neg_scores)# [bsz, 1]
            self.loss = tf.reduce_mean(loss)
        else:# scores : [bsz, time_steps, 1+k]
            keep = tf.cast(1. - self.mask, tf.float32)
            loss = build_binaryCEloss(self.pos_scores, self.neg_scores)
            self.loss = tf.reduce_sum(tf.multiply(loss, keep)) / tf.reduce_sum(keep)
        # if not self._weighted_loss:
        #    self.loss = build_binaryCEloss(self.pos_scores, self.neg_scores, self._batch_size)
        # else:
        #    self.loss = build_weighted_binaryCEloss(self.scores, self._temperature)
        self.build_optimizer()
        self.params = utils.count_params()
        print('params amount: {}'.format(self.params))

        """placeholder
        self.input_feature_ph_dict
        self.input_feature_ph_dict_target
        """
    def init_placeholder(self):
        self.user_id = tf.placeholder(tf.int32, shape=(None, self._sequence_length))
        # POI IDs  and target POI
        self.poi_id = tf.placeholder(tf.int32, shape=(None, self._sequence_length))
        self.time_id = tf.placeholder(tf.int32, shape=(None, self._sequence_length))
        self.reg_id = tf.placeholder(tf.int32, shape=(None, self._sequence_length))
        # mask matrix
        self.mask = tf.placeholder(tf.float32, shape=(None, self._sequence_length))
        #self.ds = tf.placeholder(tf.int32, shape=(None,))
        #self.ds = tf.reduce_sum(tf.cast(1. - self.mask, tf.float32), 1)

        if self._prediction_type == 'next':
            self.trg = tf.placeholder(tf.int32, shape=(None,1))
            self.trg_neg = tf.placeholder(tf.int32, shape=(None, self._num_negtive_samples))
            self.src_trg = tf.concat([self.trg, self.trg_neg], axis=-1) #[bsz, 1+k]
        else:# sequential
            self.trg = tf.placeholder(tf.int32, shape=(None, self._sequence_length))
            trg_ = tf.expand_dims(self.trg, axis=-1)# [bsz, steps, 1]
            self.trg_neg = tf.placeholder(tf.int32, shape=(None,self._sequence_length, self._num_negtive_samples))
            self.src_trg = tf.concat([trg_, self.trg_neg], axis=-1) #[bsz, time_steps, 1+k]

        # poi embedding matrix
        self.poi_embedding = get_token_embeddings(
                    self.n_poi,  # the first dimension is for zero padding
                    self.poi_embedding_size,
                    name='feature_weight_poi')
        # user embedding matrix
        self.user_embedding = get_token_embeddings(
            self.n_user,  # the first dimension is for zero padding
            self.user_embedding_size,
            name='feature_weight_user')
        self.time_embedding = get_token_embeddings(self.n_time, self.time_embedding_size, name='feature_weight_time')
        self.region_embedding = get_token_embeddings(self.n_region, self.reg_embedding_size, name='feature_weight_region')




    def build_model(self):
        # one hot vectors for all the discrete features
        with tf.variable_scope('input_embedding'):
            user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user_id)
            poi_emb = tf.nn.embedding_lookup(self.poi_embedding, self.poi_id)
            time_emb = tf.nn.embedding_lookup(self.time_embedding, self.time_id)
            reg_emb = tf.nn.embedding_lookup(self.region_embedding, self.reg_id)

        # 为最终的非线性内积设置权重
        #nonlinear_inner_weight = tf.get_variable("nonlinear_inner_weight",
        #                                         shape=[self.d_model, self.poi_embedding_size],
        #                                         initializer=tf.contrib.layers.xavier_initializer())

        # concat all features
        # shape: [bsz, T, concat-all-dimensions]
        if self.use_user_embedding:# default True
            input_features = tf.concat([user_emb, poi_emb, time_emb, reg_emb], axis=-1)
        else:
            input_features = tf.concat([poi_emb, time_emb, reg_emb], axis=-1)

        # embedding the concat features into vector with shape d_model
        input_features = tf.layers.dense(input_features, self.d_model)

        if self._prediction_type == 'next':
            loc_emb_trg = tf.nn.embedding_lookup(self.poi_embedding, self.src_trg) # #[bsz, 1+k, d_model]
        else:
            loc_emb_trg = tf.nn.embedding_lookup(self.poi_embedding, self.src_trg) # #[bsz, time_steps, 1+k, d_model]


        if self._scaled:
            input_features *= (self.d_model ** 0.5)  # [bsz, T, d_model]
        # positional encoding (pe) : x = x + pe (in GeoSan)
        src = input_features + positional_encoding(input_features)  # [bsz, T, d_model]
        # print(src.shape)
        # shape [bsz, time_steps, d_model]
        src = TransformerEncoder(inputs=src,
                                 num_layers=self.num_encoder_blocks,
                                 src_masks=self.mask,
                                 num_heads=self.nhead_enc,
                                 nhid_ffn=self.nhid_ffn,  #
                                 d_model=self.d_model,
                                 dropout_rate=self.dropout_rate,
                                 training=self.training,
                                 causality=False)
        # src_target shape[bsz, 1+k, user_dim] with 1 positive sample and k negative samples

        if self._use_attention_as_decoder:
            # use the multi_head_attention as decoder as in GeoSAN: [bsz, E]

            attn_outputs = target_aware_attention_decoder(src, src,
                                                          key_masks=self.src_mask,
                                                          dropout_rate=self.dropout_rate,
                                                          training=self.training,
                                                          causality=True)
            attn_outputs = tf.expand_dims(attn_outputs, 1)  # [bsz, 1, E]
            #attn_outputs = tf.matmul(attn_outputs, nonlinear_inner_weight)  # [bsz, 1, poi_embedding:64]
            scores = tf.reduce_sum(tf.multiply(attn_outputs, loc_emb_trg), axis=-1)  # [bsz, n+p]

            scores = tf.nn.softmax(scores, -1)  # [bsz, n+p]
        else:
            if self._prediction_type == 'next':
                enc_outputs = attention(src, dropout_rate=self.dropout_rate)  # [bsz, E]
                self.enc_outputs = tf.layers.dense(enc_outputs, self.poi_embedding_size, use_bias=False)
                #self.enc_outputs = tf.matmul(enc_outputs, nonlinear_inner_weight)  # [bsz, poi_embedding:50]
                enc_outputs = tf.expand_dims(self.enc_outputs, 1)  # [bsz, 1, E]
                scores = tf.reduce_sum(tf.multiply(enc_outputs, loc_emb_trg), axis=-1)  # [bsz, k+1]
                scores = tf.expand_dims(scores, 1) # [bsz, 1, k+1]
            else:
                # src: [bsz, time_steps, E], loc_emb_trg : [bsz, time_steps, k+1, poi_embedding]
                # scores should be [bsz, time_steps, k+1]
                #self.enc_outputs = tf.matmul(src, nonlinear_inner_weight) # [bsz,time_steps, poi_embedding]
                self.enc_outputs = tf.layers.dense(src, self.poi_embedding_size, use_bias=False)
                enc_outputs = tf.expand_dims(self.enc_outputs, -2)  # [bsz,time_steps, 1, poi_embedding]
                if len(loc_emb_trg.shape) < 4:
                    loc_emb_trg = tf.expand_dims(loc_emb_trg, axis=1)# [bsz, 1, k+1, poi_embedding]
                scores = tf.reduce_sum(tf.multiply(enc_outputs, loc_emb_trg), axis=-1)  # [bsz, time_steps, k+1]

        return scores  # , candidates_scores

    def build_optimizer(self):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # train op
        decayed_lr = tf.train.exponential_decay(self._learning_rate,
                                                global_step, 1000,
                                                0.85, staircase=True)
        optimizer = tf.train.AdamOptimizer(decayed_lr)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        # grads_and_vars = optimizer.compute_gradients(self.total_loss)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, name='train_op', global_step=global_step)

        # path log
        writer = tf.summary.FileWriter(self._path_log, tf.get_default_graph())
        writer.close()

        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        # init all variable
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit(self, trainset, testset):
        self.saver = tf.train.Saver()
        # feature_names = list(data_train_dict.keys())
        train_data_count = len(trainset)
        print('there are {} training samples...'.format(train_data_count))

        nb_train = int(math.ceil(train_data_count / float(self._batch_size)))

        max_dev_acc = 0.  # 全局最小dev loss, for early stopping)
        #dev_acc = 0.
        current_patience = 0  # for early stopping
        losses = []
        print('start to train...')
        for step in range(self._nb_epoch):
            print("==> training...")
            print('Epoch %d / %d:' % (step + 1, self._nb_epoch), flush=True)

            # shuffle train data
            # data_list = [data_train_dict['label']]
            # [data_list.append(data_train_dict[name]) for name in self._feature_names]
            #             # shuffle_matrix(*data_list, seed=seed)
            # train
            #train_loss = 0
            for i in range(nb_train):
                t0 = time.time()
                feed_dict = dict()

                batch_data_zip = trainset[i * self._batch_size: (i + 1) * self._batch_size] if \
                    (i + 1) * self._batch_size <= train_data_count else \
                    trainset[i*self._batch_size:train_data_count]
                batch_data = utils.pad_sequence([[list(check[0:-1]) for check in user[0]] for user in batch_data_zip], [0,0,0,0], self._sequence_length) #[bsz, time_steps, 4]

                if self._prediction_type == 'next':
                    batch_data_trg = np.array([user[-1][1] for user in batch_data_zip]).reshape((len(batch_data_zip), 1))
                    feed_dict.update({self.trg: batch_data_trg})
                    # uniform sampler
                    trg_neg_id = np.array([self.train_sampler.sample(1, self._num_negtive_samples)[0] for i in range(len(batch_data))]).reshape((len(batch_data), self._num_negtive_samples))
                    feed_dict.update({self.trg_neg: trg_neg_id})
                else:
                    batch_data_trg = utils.pad_sequence([[check[1] for check in user[-2]] for user in batch_data_zip], 0, self._sequence_length)
                    feed_dict.update({self.trg: batch_data_trg})
                    trg_neg_id = np.array([self.train_sampler.sample(self._sequence_length, self._num_negtive_samples)[0] for i in range(len(batch_data))])
                    feed_dict.update({self.trg_neg: trg_neg_id})

                feed_dict.update({self.user_id: batch_data[:, :, 0]})
                feed_dict.update({self.poi_id: batch_data[:, :, 1]})
                feed_dict.update({self.time_id: batch_data[:, :, 2]})
                feed_dict.update({self.reg_id: batch_data[:, :, 3]})
                mask = (batch_data[:, :, 0]==0) # [bsz, time_steps]
                #ds = (batch_data[:, :, 0]!=0).sum(-1)
                feed_dict.update({self.mask: mask})

                # mask feed
                # batch_mask = data_train_dict['mask'][batch_indices]
                # feed_dict.update({self.mask_ph: batch_mask})

                _, loss = self.sess.run(
                    [self.train_op, self.loss], feed_dict=feed_dict)
                losses.append(loss)

                t1 = time.time()
                if (i + 1) % 10 == 0:
                    #acc = self.compute_accuracy(scores)
                    print("Train: [{0}][{1}/{2}]\t"
                          "BT {batch_time:.3f}\t"
                          "loss {loss:.3f}".format(step + 1, i + 1, nb_train, batch_time=t1 - t0, loss=np.mean(losses)))
                    losses = []

                # 计算一个epoch后在开发集上的loss

                if (i + 1) % self.print_freq == 0 or i == nb_train - 1:
                    print("==> testing...")
                    t_ = time.time()
                    self.training_dynamics['steps'].append(i * self.print_freq / nb_train + step)
                    top1_trn, hr_trn, ndcg_trn = self.evaluate(trainset, num_neg=1000)
                    print("Test on Trainset: [{0}][{1}/{2}]\t"
                          "BT {inference_time:.3f}\t"
                          "Top1 {Top1:.3f}\t"
                          "HR@5 {HR5:.3f}\t"
                          "HR@10 {HR10:.3f}\t"
                          "NDCG@5 {NDCG5:.3f}\t"
                          "NDCG@10 {NDCG10:.3f}".format(step + 1, i + 1, nb_train, inference_time=time.time() - t_, Top1=top1_trn,
                                                        HR5=hr_trn[4], HR10=hr_trn[9], NDCG5=ndcg_trn[4], NDCG10=ndcg_trn[9]))

                    t_ = time.time()
                    top1_tst, hr_tst, ndcg_tst = self.evaluate(testset, num_neg=1000)
                    print("Test on Testset: [{0}][{1}/{2}]\t"
                          "BT {inference_time:.3f}\t"
                          "Top1 {Top1:.3f}\t"
                          "HR@5 {HR5:.3f}\t"
                          "HR@10 {HR10:.3f}\t"
                          "NDCG@5 {NDCG5:.3f}\t"
                          "NDCG@10 {NDCG10:.3f}".format(step + 1, i + 1, nb_train, inference_time=time.time() - t_, Top1=top1_tst,
                                                 HR5=hr_tst[4], HR10=hr_tst[9], NDCG5=ndcg_tst[4], NDCG10=ndcg_tst[9]))

                    self.training_dynamics['top1_trn'].append(top1_trn)
                    self.training_dynamics['hr_trn'].append(hr_trn)
                    self.training_dynamics['ndcg_trn'].append(ndcg_trn)
                    self.training_dynamics['top1_tst'].append(top1_tst)
                    self.training_dynamics['hr_tst'].append(hr_tst)
                    self.training_dynamics['ndcg_tst'].append(ndcg_tst)


                    # 根据dev上的表现保存模型
                    if not self._path_model:
                        continue
                    dev_acc = top1_tst
                    if dev_acc > max_dev_acc:
                        max_dev_acc = dev_acc
                        current_patience = 0
                        # save model
                        save_path = os.path.join(self._path_log, "{}_epoch_{}_sampler_{}".format(self.dataset_name, step, self.train_sampler))
                        utils.serialize(self.training_dynamics, save_path)
                        #self.saver.save(self.sess, self._path_model)
                        print('model has saved to %s!' % self._path_model, flush=True)
                    else:
                        current_patience += 1
                        print('no improvement, current patience: %d / %d' %
                              (current_patience, self._train_max_patience), flush=True)
                        if self._train_max_patience and current_patience >= self._train_max_patience:
                            print('\nfinished training! (early stopping, max patience: %d)'
                                  % self._train_max_patience, flush=True)
                            return
                        print('\nfinished training!', flush=True)
        return

    def evaluate(self, dataset, num_neg=1000):
        self.training = False
        train_data_count = len(dataset)
        print('there are {} training samples...'.format(train_data_count))

        nb_train = int(math.ceil(train_data_count / float(self._batch_size)))

        cnt = Counter()
        top1 = 0
        array = np.zeros(num_neg + 1)

        poi_embedding = self.sess.run(self.poi_embedding, feed_dict={})#[n_loc, embedding]
        for i in range(nb_train):
            feed_dict = dict()

            batch_data_zip = dataset[i * self._batch_size: (i + 1) * self._batch_size] if \
                (i + 1) * self._batch_size <= train_data_count else \
                dataset[i * self._batch_size:train_data_count]
            batch_data = utils.pad_sequence([[list(check[0:-1]) for check in user[0]] for user in batch_data_zip],
                                            [0, 0, 0, 0], self._sequence_length)  # [bsz, time_steps, 4]

            batch_data_trg_pos = np.array([user[-1][1] for user in batch_data_zip]).\
                reshape((len(batch_data_zip), 1)) # [bsz, 1]
            batch_data_trg_neg = np.array([self.train_sampler.sample(1, num_neg)[0] for i in range(len(batch_data))]).reshape((len(batch_data), num_neg)) #[bsz, n_neg_loc]
            batch_data_trg_id = np.concatenate([batch_data_trg_pos, batch_data_trg_neg], axis=1) #[bsz, 1+n_neg_loc]
            #feed_dict.update({self.trg: batch_data_trg})
            #feed_dict.update({self.trg_neg: batch_data_trg_neg})

            feed_dict.update({self.user_id: batch_data[:, :, 0]})
            feed_dict.update({self.poi_id: batch_data[:, :, 1]})
            feed_dict.update({self.time_id: batch_data[:, :, 2]})
            feed_dict.update({self.reg_id: batch_data[:, :, 3]})
            mask = (batch_data[:, :, 0] == 0)  # [bsz, time_steps]
            ds = (batch_data[:, :, 0] != 0).sum(-1)  # [bsz,]
            feed_dict.update({self.mask: mask})

            # mask feed
            # batch_mask = data_train_dict['mask'][batch_indices]
            # feed_dict.update({self.mask_ph: batch_mask})
            # scores: [bsz, 1+k]

            enc_outputs = self.sess.run(self.enc_outputs, feed_dict=feed_dict)
            if self._prediction_type == 'sequential':# enc_outputs [bsz, time_steps, embedding]
                enc_outputs = enc_outputs[np.arange(len(ds)), ds-1, :]# [bsz, embedding]
            enc_outputs = np.expand_dims(enc_outputs, axis=1)#[bsz, 1, embedding]
            batch_poi_embedding = np.array([poi_embedding[ids] for ids in batch_data_trg_id])  # [bsz, 1+num_neg, embedding]

            scores = np.multiply(enc_outputs, batch_poi_embedding).sum(-1)  # [bsz, n_loc]
            idx = scores.argsort(axis=-1).T[::-1].T
            order = idx.argsort(axis=-1)[:, 0]
            cnt.update(list(order))
            top1 += (scores.argmax(-1) == 0).sum()
        for k, v in cnt.items():
            array[k] = v
        # hit rate and NDCG
        hr = array.cumsum()
        ndcg = 1 / np.log2(np.arange(0, num_neg + 1) + 2)
        ndcg = ndcg * array
        ndcg = ndcg.cumsum() / hr.max()
        hr = hr / hr.max()
        self.training = True
        return top1/train_data_count, hr[:10], ndcg[:10]
