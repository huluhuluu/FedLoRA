#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import sys
import numpy as np
from pathlib import Path
from utils.helper import get_label_count_sample, get_label_count
from utils.sampling import dataset_iid, dataset_noniid_shard, dataset_noniid_dirichlet
from utils.options import args_parser
from utils import options
from models.Update import LocalUpdate
from models.Nets import LoraNet
from models.Fed import FedAVG, LoraGA, FedACA
from models.Fed import FedSVD, FedABOptim
from models.test import test_
from datasets import load_dataset
import torch.nn.modules

# Generate output path
def get_output_path(args):
    """Generate unique output path for saving results."""
    # Build base filename
    data_type = 'iid' if args.iid else f'noniid_alpha{args.alpha}'
    model_name = Path(args.model).name  # Extract model name from path
    
    filename = (
        f"{model_name}_{args.init}_{args.task}_"
        f"epoch{args.epochs}_C{args.frac}_{data_type}_"
        f"local_ep{args.local_ep}_lr{args.lr}"
    )
    
    # Create log directory
    log_dir = Path('./log')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Find unique path
    base_path = log_dir / filename
    output_path = base_path
    idx = 0
    while output_path.with_suffix('.png').exists():
        output_path = log_dir / f"{filename}_{idx}"
        idx += 1
    
    return output_path
    
def main():
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset from local path
    raw_dataset = load_dataset(args.dataset)
    
    print(raw_dataset)
    
    dataset_train = raw_dataset['train']
    # use 'test' split if available, otherwise use 'validation'
    if 'test' in raw_dataset:
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
    
    # build model
    net_glob = LoraNet(num_labels = len(labels), rank=args.lora_rank, args = args)
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

        # Aggregate LoRA parameters
        method = args.init if iter == 0 else args.reduce_method
        
        if method == options.LORAGA and iter == 0:
            lora_glob = LoraGA(lora_locals, lora_glob, args.lr, args.lora_rank)
        elif method == options.SVD:
            lora_glob = FedSVD(lora_locals, args.lora_rank, num_, autorank=args.autorank, theta=args.theta)
        elif method == options.ACA:
            lora_glob = FedACA(lora_locals, args.lora_rank, num_, autorank=args.autorank)
        elif method == options.FedABOptim:
            lora_glob = FedABOptim(lora_locals, args.lora_rank, num_)
        else:  # FedAvg or default
            lora_glob, _, _, _, _ = FedAVG(lora_locals)
        
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

    output_path = get_output_path(args)
    
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
    plt.savefig(str(output_path) + '.png')

    # write loss to file
    with open(str(output_path) + '.txt', 'w') as file:
        # write command arg to file
        file.write('Command line arg: ')
        file.write(' '.join(sys.argv))
        file.write('\n')
        for iter, (train_acc, test_acc, l) in enumerate(zip(acc_train, acc_test, loss_train)):
            file.write('Round {:3d}, Average loss {:.3f}, Train accuracy {:.3f}, Test accuracy {:.3f}\n'.format(iter * args.interval, l, train_acc, test_acc))
        file.write('\n')

if __name__ == '__main__':
    main()