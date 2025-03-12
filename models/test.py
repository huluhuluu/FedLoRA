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
    net_g.to(args.device)
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(datatest, args.dataset, [i for i in range(len(datatest))]), batch_size=args.bs, shuffle=False)

    correct_label, answer_label = set(), set()
    correct_label_dict, answer_label_dict = {}, {}
    logits_list = []
    for idx, (input_ids, attention_mask, target) in enumerate(data_loader):
        if args.gpu != -1:
            input_ids, attention_mask, target = input_ids.to(args.device), attention_mask.to(args.device),  target.to(args.device)
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        with torch.no_grad():
            logits = net_g(input_ids, attention_mask=attention_mask).logits  # get logits
            
            # sum up batch loss
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = logits.argmax(dim=1)  # get the index of the max log-probability

            # record the correct prediction label
            for i in range(len(y_pred)):
                if y_pred[i] == target[i]:
                    correct_label.add(target[i].item())
                    if correct_label_dict.get(target[i].item()) is None:
                        correct_label_dict[target[i].item()] = 1
                    else:
                        correct_label_dict[target[i].item()] += 1
                answer_label.add(y_pred[i].item())
            logits_list.append(logits)
            correct += y_pred.eq(target).long().sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    if args.verbose:
        print('Correct label:', correct_label)
        print('Answer label:', answer_label)
        # print correct dict
        print('Correct label dict:', correct_label_dict)
        print('\nTest set: Average loss: {:.4f} \tAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
