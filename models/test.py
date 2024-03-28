#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import DatasetSplit

def test_(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(datatest, [i for i in range(len(datatest))]), batch_size=args.bs, shuffle=False)
    l = len(data_loader)
    for idx, (text, attention_mask, target) in enumerate(data_loader):
        if args.gpu != -1:
            text, attention_mask, target = text.to(args.device), attention_mask.to(args.device),  target.to(args.device)
        with torch.no_grad():
            logits = net_g(text, attention_mask=attention_mask).logits  # get logits
            # sum up batch loss
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = logits.argmax(dim=1)  # get the index of the max log-probability
            # print(logits)
            # print(y_pred)
            # print(target)
            # exit()
            correct += y_pred.eq(target).long().sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
