#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
# set full print
import torch
torch.set_printoptions(profile="full")
cache_path = '/data/huluhulu/fedbertlora/cache'

# get label counts
def get_label_count(dataset_train, dataset_test):
    # get label counts
    label_counts_train = {}
    label_counts_test = {}

    # traverse all train data
    for i in range(len(dataset_train)):
        if label_counts_train.get(dataset_train[i]['label']) is None:
            label_counts_train[dataset_train[i]['label']] = 1
        else:
            label_counts_train[dataset_train[i]['label']] += 1
    
    # traverse all test data
    for i in range(len(dataset_test)):
        if label_counts_test.get(dataset_test[i]['label']) is None:
            label_counts_test[dataset_test[i]['label']] = 1
        else:
            label_counts_test[dataset_test[i]['label']] += 1
    

    # print label counts res
    for key in label_counts_train.keys():
        print('label: {},train count: {} test count: {}'.format(
            key, 
            label_counts_train[key], 
            label_counts_test[key] if label_counts_test.get(key) is not None else 0))

# get label counts after sampling
# dataset: dataset_train
# users: sampled index for users
def get_label_count_sample(dataset, users, verbose = True):
    label_counts = {}
    users = list(users)
    for i in range(len(users)):
        ind = int(users[i])
        if label_counts.get(dataset[ind]['label']) is None:
            label_counts[dataset[ind]['label']] = 1
        else:
            label_counts[dataset[ind]['label']] += 1
    
    sum_label_count = 0
    for key in label_counts.keys():
        if verbose:
            print('label: {}, count: {}'.format(key, label_counts[key]))
        sum_label_count += 1
    print('label num: {}, data num {}'.format(sum_label_count, len(users)))
