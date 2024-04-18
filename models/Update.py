#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import metrics
from transformers import AutoTokenizer

# Model id to load the tokenizer
model_id = "bert-base-uncased"
# Load Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./cache/tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/data/huluhulu/fedbertlora/cache/tokenizer')

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(idx) for idx in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        text, label = self.dataset[self.idxs[item]] ['text'] , self.dataset[self.idxs[item]]['label']
        text_encoded = tokenizer(text, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = text_encoded['input_ids'].squeeze()
        attention_mask = text_encoded['attention_mask'].squeeze()
        return input_ids, attention_mask, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.model.train()
        net.model.to(self.args.device)
        # train and update
        optimizer = torch.optim.SGD(net.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        epoch_loss = []
        acc_ = [0 for i in range(self.args.local_ep)]
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct = 0

            correct_label = set()
            answer_label = set()
            for batch_idx, (text, attention_mask, label) in enumerate(self.ldr_train):
                text, attention_mask, label = text.to(self.args.device), attention_mask.to(self.args.device), label.to(self.args.device)

                net.model.zero_grad()

                logits = net.model(text, attention_mask=attention_mask).logits  # Get logits from model output
                loss = self.loss_func(logits, label)

                # deepcopy logits and label
                logits_copy = logits.detach().clone()
                label_copy = label.detach().clone()
                y_pred = logits_copy.argmax(dim=1)  # get the index of the max log-probability
                correct += y_pred.eq(label_copy).long().sum().item() # get correct num
                # record the correct prediction label
                for i in range(len(y_pred)):
                    if y_pred[i] == label_copy[i]:
                        correct_label.add(label_copy[i].item())
                    answer_label.add(y_pred[i].item())
                
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            # add accuracy to acc_
            acc_[iter] = correct / len(self.ldr_train.dataset)
            if self.args.verbose:
                # print('correct labels:', correct_label)
                # print('answer labels:', answer_label)
                print('Update Epoch: {} \tAverage Loss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    iter, sum(batch_loss) / len(batch_loss), 100. * correct / len(self.ldr_train.dataset)))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.get_lora_parameters(), sum(epoch_loss) / len(epoch_loss), acc_[-1]

