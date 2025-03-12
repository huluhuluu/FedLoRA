#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import sys
import numpy as np
import os

from utils.helper import get_label_count_sample, get_label_count, cache_path
from utils.sampling import dataset_iid, dataset_noniid_shard, dataset_noniid_dirichlet
from utils.options import args_parser
from utils import options
from models.Update import LocalUpdate
from models.Nets import BertLora
from models.Fed import FedAVG, LoraGA, FedACA
from models.Fed import FedSVD, FedABOptim
from models.test import test_
from datasets import load_dataset
import torch.nn.modules

# References code
# https://github.com/alexriggio/BERT-LoRA-TensorRT/tree/main/notebooks
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/pytorch-2-0-bert-text-classification.ipynb

# python main_fed.py --model bert-base-uncased --dataset agnews --dirichlet_noniid --num_users 1000 --local_bs 32 --local_ep 1 --lr 0.1 --interval 1 --epochs 100 --gpu 0 --init gauss --alpha 1.0
# python main_fed.py --model gpt2-medium --dataset agnews --dirichlet_noniid --num_users 1000 --local_bs 32 --local_ep 1 --lr 0.1 --interval 1 --epochs 100 --gpu 0 --init gauss --alpha 1.0
# openai-community/gpt2-medium
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # get data from huggingface.co/dataset
    if args.dataset == 'banking77':
        raw_dataset = load_dataset("PolyAI/banking77", cache_dir = cache_path + '/data')
    elif args.dataset == 'agnews':
        raw_dataset = load_dataset("fancyzhx/ag_news", cache_dir = cache_path + '/data')
    else:
        # glue sub-dataset
        raw_dataset = load_dataset("nyu-mll/glue", args.dataset, cache_dir = cache_path + '/data')
    
    print(raw_dataset)
    
    dataset_train = raw_dataset['train']
    if args.dataset == 'banking77' or args.dataset == '20news' or args.dataset == 'agnews' or args.dataset == 'sst2':
        dataset_test = raw_dataset['test']
    else:
        dataset_test = raw_dataset['validation']
    
    labels = dataset_train.features["label"].names
    # print(len(dataset_train), len(dataset_test)) # 10003 3080
    # get_label_count(dataset_train, dataset_test)
    
    # sample users
    dataset_train = dataset_train.shuffle(seed=42)
    if args.iid:
        dict_users = dataset_iid(dataset_train, args.num_users)
    elif args.shard_noniid:
        dict_users = dataset_noniid_shard(dataset_train, args.num_users)
    elif args.dirichlet_noniid:
        dict_users = dataset_noniid_dirichlet(dataset_train, args.num_users, args.alpha)
    else: # default iid 
        dict_users = dataset_iid(dataset_train, args.num_users)
    
    # get_label_count_sample(dataset_train, dict_users[0])
    # print('---------------------------------')
    # get_label_count_sample(dataset_train, dict_users[1])
    
    # build model
    net_glob = BertLora(num_labels = len(labels), rank=args.lora_rank, args = args)
    # print(net_glob.model)
    # print(len(net_glob.name_lora_para))
    
    # copy A,B weights
    lora_glob = net_glob.get_lora_parameters()
    # print(len(lora_glob)) # 12
    # print(lora_glob[0][0][0].shape) # q_lora_A torch.Size([8, 768])
    # print(lora_glob[0][0][1].shape) # q_lora_B torch.Size([768, 8])
    
    # training
    loss_train, loss_test = [], []
    acc_train, acc_test = [], []
    valid_users = []
    for ind in range(args.num_users):
        if len(dict_users[ind]) == 0: # skip empty user
            continue
        valid_users.append(ind)
    print('valid users:', len(valid_users))

    for iter in range(args.epochs):
        print('--------------------------------------')
        # train
        loss_locals, acc_locals = [], []
        lora_locals = []

        # num of selected users and selected users
        m = max(int(args.frac * args.num_users), 1) 
        idxs_users = np.random.choice(range(len(valid_users)), m, replace=False)
        idxs_users = [valid_users[i] for i in idxs_users]
        idx_ = 0
        # train using the local dataset
        num_ = []
        for idx in idxs_users:
            if len(dict_users[idx]) == 0: # skip empty user
                print('Round {:3d} User{:d} {:3d} is empty and skiped'.format(iter, idx_, idx))
                continue
            num_.append(len(dict_users[idx]))
            if args.verbose:
                # print iter and idx
                print('Round {:3d}, User{:d} {:3d}'.format(iter, idx_, idx))
                get_label_count_sample(dataset_train, dict_users[idx], verbose = False)
            idx_ += 1

            # use global lora parameters
            net_glob.put_lora_parameters(lora_glob)

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            lora, loss, acc, fail_flag = local.train(net = net_glob, printflag= (iter<=1 and idx == idxs_users[0]))
            # judge errors beteween lora and lora_glob
            # ||lora - lora_glob|| (12, 2)
            loss_lora = []
            for layer_ind in range(len(lora)): # traverse all layers
                for qv_ind in range(len(lora[layer_ind])): # traverse query and value
                    for ab_index in range(len(lora[layer_ind][qv_ind])):
                        loss_lora.append(torch.norm(lora[layer_ind][qv_ind][ab_index] - lora_glob[layer_ind][qv_ind][ab_index]).item())
            
            # print('lora para loss:', loss_lora)
            # print('sum lora para loss:', sum(loss_lora))
            
            if fail_flag:
                print('invalid local model, skip this user')
                continue
            lora_locals.append(copy.deepcopy(lora))
            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))

        if iter == 0:
            if args.init == options.LORAGA:
                lora_glob = LoraGA(lora_locals, lora_glob, args)
            elif args.init == options.SVD:
                lora_glob = FedSVD(lora_locals, args.lora_rank, args, num_)
            elif args.init == options.ACA:
                lora_glob = FedACA(lora_locals, args.lora_rank, args, num_)
            else:
                lora_glob, _, _, _, _ = FedAVG(lora_locals, args)
        else:
            # update global weights
            if args.reduce_method == options.FedAvg:
                lora_glob, _, _, _, _ = FedAVG(lora_locals, args)
            elif args.reduce_method == options.FedSVD:
                lora_glob = FedSVD(lora_locals, args.lora_rank, args, num_)
            elif args.reduce_method == options.FedABOptim:
                lora_glob = FedABOptim(lora_locals)
            else:
                lora_glob, _, _, _, _ = FedAVG(lora_locals, args)
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
        
        if iter <= -1:
            # print lora parameters
            lora_glob = net_glob.get_lora_parameters()
            # 12, 2, 2
            # [[[q_lora_a, q_lora_b], [v_lora_a, v_lora_b]], ...]
            for layer_ind in range(len(lora_glob)):
                print(f'layer {layer_ind}   | q_lora_a: {torch.linalg.norm(lora_glob[layer_ind][0][0], ord='fro')} \t' +
                                            f'q_lora_b: {torch.linalg.norm(lora_glob[layer_ind][0][1], ord='fro')} \t' +
                                            f'v_lora_a: {torch.linalg.norm(lora_glob[layer_ind][1][0], ord='fro')} \t'+
                                            f'v_lora_b: {torch.linalg.norm(lora_glob[layer_ind][1][1], ord='fro')}')
            print('--------------------------------------')
            if iter == 1:
                print('\t\t\t\t')
                exit(0)
    
    # path
    path_name = './log/{model}_{path}_{dataset}_epoch{epochs}_C{frac}_{iid}{alpha}_localep{local_ep}_ep{ep}_lr{lr}'.format(
                    model = args.model,
                    path = args.init,
                    dataset = args.dataset,
                    epochs = args.epochs,
                    frac = args.frac,
                    iid = 'iid' if args.iid else ('noniid_dirichlet' if not args.shard_noniid else 'noniid_shard'),
                    alpha = '_alpha{}'.format(args.alpha) if args.dirichlet_noniid else '',
                    ep = args.local_ep,
                    local_ep = args.local_ep,
                    lr = args.lr
                    )
    # if path_name exists, add a number to the end: 0, 1, 2, ...
    ind = 0
    while True:
        if os.path.exists(path_name + '.png'):
            path_name = path_name + f'_{ind}'
            ind += 1
        else:
            break
    
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
    plt.savefig(path_name + '.png')

    # write loss to file
    with open(path_name + '.txt', 'w') as file:
        # write command arg to file
        file.write('Command line arg: ')
        file.write(' '.join(sys.argv))
        file.write('\n')
        for iter, (a, l) in enumerate(zip(acc_test, loss_train)):
            file.write('Round {:3d}, Average loss {:.3f} Accuracy {:.3f}\n'.format(iter * args.interval, l, a))
        file.write('\n')
