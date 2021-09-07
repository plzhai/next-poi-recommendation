"""
created by Weijie
contact to zhaipenglong.zpl@alibaba.inc.com

Jul 2021
"""

import tensorflow as tf
import numpy as np
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import os

def split_train_dev(data_dict, data_dict_target, dev_size):
    user_id = list(data_dict_target.keys())
    count = len(data_dict_target)
    train_count = int(count * (1.0 - dev_size))
    train_user_id = user_id[0:train_count]
    dev_user_id = user_id[train_count:]

    data_train_dict = {}
    data_dev_dict = {}
    data_train_dict_target = {}
    data_dev_dict_target = {}
    for uid in train_user_id:
        data_train_dict[uid] = data_dict[uid]
        data_train_dict_target[uid] = data_dict_target[uid]
    for uid in dev_user_id:
        data_dev_dict[uid] = data_dict[uid]
        data_dev_dict_target[uid] = data_dict_target[uid]

    return data_train_dict, data_dev_dict, data_train_dict_target, data_dev_dict_target


def clip(nums):
    nums = np.array(nums)
    where_are_nan = np.isnan(nums)
    where_are_inf = np.isinf(nums)
    nums[where_are_nan] = 0
    nums[where_are_inf] = 0
    return nums

def compute_accuracy(scores, target_pos_length):
    click = 0
    for score in scores:
        # hr = hr + (eff_index < int(pos.sum())).sum() / (pos.sum()+0.0001)
        click = click + 1 if score.argmax() < target_pos_length else click

    return click / len(scores)


def count_params():
    count = 0
    for param in tf.trainable_variables():
        t = 1.
        shape = param.shape
        for s in shape:
            t *= s
        count += t
    return int(count)

def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)

def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)

def get_visited_locs(dataset):
    user_visited_locs = {}
    for u in range(len(dataset.user_seq)):
        seq = dataset.user_seq[u]
        user = seq[0][0]
        user_visited_locs[user] = set()
        for i in reversed(range(len(seq))):
            if not seq[i][4]:
                break
        user_visited_locs[user].add(seq[i][1])
        seq = seq[:i]
        for check_in in seq:
            user_visited_locs[user].add(check_in[1])
    return user_visited_locs
def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def pad_sequence(sequence, pad_value, sequence_length):
    return np.array([seq + [pad_value] * (sequence_length - len(seq)) if len(seq) < sequence_length else seq[0:sequence_length] for seq in sequence])


