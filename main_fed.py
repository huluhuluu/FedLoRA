#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import sys
import numpy as np
import torch

from utils.sampling import dataset_iid, dataset_noniid
from utils.options import args_parser
from utils import options
from models.Update import LocalUpdate
from models.Nets import BertLora
from models.Fed import FedAvg
from models.Fed import FedSVD, FedABOptim
from models.test import test_
from datasets import load_dataset
import torch.nn.modules

# References code
# https://github.com/alexriggio/BERT-LoRA-TensorRT/tree/main/notebooks
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/pytorch-2-0-bert-text-classification.ipynb

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

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Dataset id from huggingface.co/dataset
    dataset_id = "PolyAI/banking77" 

    # get data from huggingface.co/dataset
    raw_dataset = load_dataset(dataset_id, cache_dir='./cache/data')

    dataset_train = raw_dataset['train']
    dataset_test = raw_dataset['test']
    labels = dataset_train.features["label"].names
    # print(len(dataset_train), len(dataset_test)) # 10003 3080
    # get_label_count(dataset_train, dataset_test)
    
    # sample users
    dataset_train = dataset_train.shuffle(seed=42)
    if args.iid:
        dict_users = dataset_iid(dataset_train, args.num_users)
    else:
        dict_users = dataset_noniid(dataset_train, args.num_users)
    
    # get_label_count_sample(dataset_train, dict_users[0])
    # non iid: label num:3
    
    # build model
    net_glob = BertLora(num_labels = len(labels), rank=args.lora_rank, args = args)
    # print(net_glob.model)
   
    # copy A,B weights
    lora_glob = net_glob.get_lora_parameters()
    # print(lora_glob[0][0][0].shape) # q_lora_A
    # print(lora_glob[0][0][1].shape) # q_lora_B
    
    # training
    loss_train, loss_test = [], []
    acc_train, acc_test = [], []

    for iter in range(args.epochs):
        # train
        loss_locals, acc_locals = [], []
        lora_locals = []

        # num of selected users and selected users
        m = max(int(args.frac * args.num_users), 1) 
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # train using the local dataset
        for idx in idxs_users:
            # use global lora parameters
            net_glob.put_lora_parameters(lora_glob)

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if args.verbose:
                # print iter and idx
                print('Round {:3d}, User {:3d}'.format(iter, idx))
                get_label_count_sample(dataset_train, dict_users[idx], verbose = False)
            
            lora, loss, acc = local.train(net = net_glob)

            lora_locals.append(copy.deepcopy(lora))
            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))

        # modify

        # update global weights
        if args.reduce_method == options.FedAvg:
            lora_glob = FedAvg(lora_locals)
        elif args.reduce_method == options.FedSVD:
            lora_glob = FedSVD(lora_locals, args.lora_rank)
        elif args.reduce_method == options.FedABOptim:
            lora_glob = FedABOptim(lora_locals)

        # copy weight to net_glob
        net_glob.put_lora_parameters(lora_glob)
        
        # compute loss and accuracy every args.interval rounds
        if iter % args.interval == 0:
            # compute loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            # compute accuracy
            acc_avg = sum(acc_locals) / len(acc_locals)
            acc_train.append(acc_avg)

            ## test
            net_glob.put_lora_parameters(lora_glob)
            acc_test_local, loss_test_local = test_(net_glob.model, dataset_test, args)
            acc_test.append(acc_test_local)
            loss_test.append(loss_test_local)

            # print loss, train acc, test acc
            print('Round {:3d}, Average loss {:.3f}, Train accuracy {:.3f}, Test accuracy {:.3f}'.
                  format(iter,loss_avg, acc_avg, acc_test_local))
    # plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot([i * args.interval for i in range(len(loss_train))], loss_train)
    plt.title('Training Loss')
    plt.xlabel('round')
    plt.ylabel('loss')

    # plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot([i * args.interval for i in range(len(loss_train))], acc_test)
    plt.title('Test Accuracy')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.savefig('./save/fed_{}_C{}_iid{}.png'.format(args.epochs, args.frac, args.iid))
    
    file_name = './save/fed_{}_C{}_iid{}.txt'.format(args.epochs, args.frac, args.iid)
    # write loss to file
    with open(file_name, 'w') as file:
        # write command arg to file
        file.write('Command line arg: ')
        file.write(' '.join(sys.argv))
        file.write('\n')
        for iter, (a, l) in enumerate(zip(acc_test, loss_train)):
            file.write('Round {:3d}, Average loss {:.3f} Accuracy {:.3f}\n'.format(iter * args.interval, l, a))
        file.write('\n')
