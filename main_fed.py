#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
import torch

from utils.sampling import dataset_iid, dataset_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import BertLora
from models.Fed import FedAvg
from models.test import test_
from datasets import load_dataset
import torch.nn.modules

# https://github.com/Desein-Yang/LoRA_Lightning
# https://github.com/alexriggio/BERT-LoRA-TensorRT/tree/main/notebooks
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/pytorch-2-0-bert-text-classification.ipynb

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
    
    # sample users
    if args.iid:
        dict_users = dataset_iid(dataset_train, args.num_users)
    else:
        dict_users = dataset_noniid(dataset_train, args.num_users)
    
    # build model
    net_glob = BertLora(num_labels = len(labels), args = args)
    # print(net_glob.model)
   
    # copy A,B weights
    lora_glob = net_glob.get_lora_parameters()
    # print(lora_glob[0][0][0].shape) # q_lora_A
    # print(lora_glob[0][0][1].shape) # q_lora_B
    
    # training
    loss_train = []
    accuracy = []

    for iter in range(args.epochs):
        # train
        loss_locals = []
        lora_locals = []

        # num of selected users and selected users
        m = max(int(args.frac * args.num_users), 1) 
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # train using the local dataset
        for idx in idxs_users:
            # use global lora parameters
            net_glob.put_lora_parameters(lora_glob)

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # exit()
            lora, loss = local.train(net = net_glob)

            lora_locals.append(copy.deepcopy(lora))
            loss_locals.append(copy.deepcopy(loss))

        # modify

        # update global weights
        lora_glob = FedAvg(lora_locals)

        # copy weight to net_glob
        net_glob.put_lora_parameters(lora_glob)
        
        # compute loss and accuracy every 10 rounds
        if iter%1 == 0:
            # compute loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            # # test
            net_glob.put_lora_parameters(lora_glob)
            acc_test, loss_test = test_(net_glob.model, dataset_test, args)
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
    with open(file_name, 'w') as file:
        for iter, (a, l) in enumerate(zip(accuracy, loss_train)):
            file.write('Round {:3d}, Average loss {:.3f} Accuracy {:.3f}\n'.format(iter, l, a))
        file.write('\n')
