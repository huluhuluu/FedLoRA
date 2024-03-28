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
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./cache/tokenizer')

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
        # train and update
        optimizer = torch.optim.SGD(net.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (text, attention_mask, label) in enumerate(self.ldr_train):
                text, attention_mask, label = text.to(self.args.device), attention_mask.to(self.args.device), label.to(self.args.device)

                net.model.zero_grad()

                logits = net.model(text, attention_mask=attention_mask).logits  # Get logits from model output
                loss = self.loss_func(logits, label)
                
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(text), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.get_lora_parameters(), sum(epoch_loss) / len(epoch_loss)

