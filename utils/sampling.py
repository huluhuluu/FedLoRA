#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8


import numpy as np
import random

def dataset_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def dataset_noniid(dataset, num_users, num_shards=200):
    """
    Sample non-I.I.D client data from dataset
        method: divide dataset into 'num_shards' parts and then assign each part to 'num_users' users 
                if the number of shards is less than the number of users, some users will not have data
                if the number of shards is greater than the number of users, some shards users will have more shard
    :param dataset:
    :param num_users:
    :param shards: the number of slices of the data set will be split
    :return:
    """
    # the number of slices of the data set will be split
    num_case = len(dataset) // num_shards
    # index of shards
    idx_shard = [i for i in range(num_shards)]
    # index of user
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_case)

    labels = np.array(dataset["label"])

    # Match the length of idxs
    labels = labels[0: num_shards*num_case]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    
    # divide and assign
    while idx_shard:
        random.shuffle(idx_shard)  # shuffle the order of the shards
        for i in range(num_users):
            if idx_shard:  # if there are still shards to be assigned
                rand = idx_shard.pop()  # get a shard randomly and remove it from the list
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_case : (rand+1) * num_case]), axis=0)
            else:  # finish the assignment
                break
    return dict_users



