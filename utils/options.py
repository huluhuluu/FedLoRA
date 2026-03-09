#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import argparse

FedAvg, FedSVD, FedABOptim = "fedavg", "fedsvd", "fedaboptim"
GAUSS, ACA, SVD, LORAGA, PISSA, EVA, OLORA, KAMING = "gaussian", "aca", "svd", "loraga", "pissa", 'eva', 'olora', 'kaming'

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.9)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--reduce_method', type=str, default=FedAvg, choices=[FedAvg, FedSVD, FedABOptim], help="method to reduce A and B, choose from fedavg, fedsvd, fedaboptim")
    parser.add_argument('--init', type=str, default=KAMING, choices=[GAUSS, ACA, SVD, LORAGA, PISSA, EVA, OLORA, KAMING], help="initialization method, choose from gaussian, aca, svd, loraga, pissa, eva, olora, kaming")
    parser.add_argument('--interval', type=int, default=1, help="loss and accuracy curve will be plotted every (interval) rounds")
    
    # other arguments
    parser.add_argument('--dataset', type=str, default='./data/banking77', help='path to dataset directory')
    parser.add_argument('--model', type=str, default='./models/bert-base-uncased', help='path to pretrained model directory')
    parser.add_argument('--task', type=str, default='banking77', help='task name for dataset (e.g., banking77, agnews, sst2, mnli)')
    parser.add_argument('--iid', action='store_true', help='i.i.d')
    parser.add_argument('--shard_noniid', action='store_true', help='shard_noniid')
    parser.add_argument('--dirichlet_noniid', action='store_true', help='dirichlet_noniid')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha of dirichlet distribution')

    parser.add_argument('--autorank', action='store_true', help='use svd singular value guide rank or not')
    parser.add_argument('--theta', type=float, default=0.98, help='theta of svd singular value sum')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # lora arguments
    parser.add_argument('--lora_rank', type=int, default=8, help='rank of lora (default: 8)')
    
    # sequence length
    parser.add_argument('--seq_len', type=int, default=512, help='max sequence length for tokenizer (default: 512)')
    
    args = parser.parse_args()
    return args
