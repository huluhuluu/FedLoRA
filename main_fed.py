#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from utils.sampling import dataset_iid, dataset_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg,FedAvgSyn,FedAvgMod
from models.test import test_
from datasets import load_dataset

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # Dataset id from huggingface.co/dataset
    dataset_id = "banking77" 
    # get raw dataset
    raw_dataset = load_dataset(dataset_id, cache_dir = './data')
    dataset_train = raw_dataset['train']
    dataset_test = raw_dataset['test']
    
    # sample users
    if args.iid:
        dict_users = dataset_iid(dataset_train, args.num_users)
    else:
        dict_users = dataset_noniid(dataset_train, args.num_users)
    
    img_size = dataset_train[0][0].shape

    # build model
    net_glob = CNNCifar(args=args).to(args.device)
    
    print(net_glob)

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    accuracy = []


    for iter in range(args.epochs):
        # train
        net_glob.train()
        loss_locals = []
        w_locals = []

        # num of selected users
        m = max(int(args.frac * args.num_users), 1) # choose

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # train using the local dataset
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # modify

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        # compute loss and accuracy every 10 rounds
        if iter%10 == 0:
            # compute loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            # test
            acc_test, loss_test = test_(net_glob, dataset_test, args)
            accuracy.append(acc_test)
        
            print('Round {:3d}, Average loss {:.3f} Accuracy {:.3f}'.format(iter, loss_avg, acc_test))

    # plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss_train)), loss_train)
    plt.title('Training Loss')
    plt.xlabel('round')
    plt.ylabel('loss')

    # plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(len(accuracy)), accuracy)
    plt.title('Test Accuracy')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.savefig('./save/fed_{}_C{}_iid{}.png'.format(args.epochs, args.frac, args.iid))
    
    file_name = './save/fed_{}_C{}_iid{}.txt'.format(args.epochs, args.frac, args.iid)
    # write loss to file
    with open('file_name', 'w') as file:
        for iter, (a, l) in enumerate(zip(accuracy, loss_train)):
            file.write('Round {:3d}, Average loss {:.3f} Accuracy {:.3f}\n'.format(iter, l, a))
        file.write('\n')
