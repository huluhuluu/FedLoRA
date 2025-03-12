#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from utils.helper import cache_path

# Load Tokenizer
tokenizer = None
task_to_keys = {
    "banking77": ("text", None),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    # "sst2": ("sentence", None),
    "sst2": ("text", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "20news": ("text", None),
    "agnews": ("text", None),
}

class DatasetSplit(Dataset):
    def __init__(self, dataset, task, idxs):
        self.dataset = dataset
        self.task = task
        self.idxs = [int(idx) for idx in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        global tokenizer, task_to_keys
        if task_to_keys[self.task][1] is not None:
            text_encoded = tokenizer(self.dataset[self.idxs[item]][task_to_keys[self.task][0]], self.dataset[self.idxs[item]][task_to_keys[self.task][1]],  truncation=True, padding='max_length', return_tensors='pt')
        else:
            text_encoded = tokenizer(self.dataset[self.idxs[item]][task_to_keys[self.task][0]], truncation=True, padding='max_length', return_tensors='pt')
        
        input_ids = text_encoded['input_ids'].squeeze()
        attention_mask = text_encoded['attention_mask'].squeeze()
        return input_ids, attention_mask, self.dataset[self.idxs[item]]['label']


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        global tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = cache_path + '/tokenizer')
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.args.dataset ,idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, printflag = False):
        net.model.train()
        net.model.to(self.args.device)
        # train and update
        optimizer = torch.optim.SGD(net.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # adam eps = 1e-8
        # optimizer = torch.optim.Adam(net.model.parameters(), lr=self.args.lr, eps=1e-8)
        
        epoch_loss = []
        acc_ = [0 for i in range(self.args.local_ep)]
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct = 0

            correct_label = set()
            answer_label = set()
            for batch_idx, (input_ids, attention_mask, label) in enumerate(self.ldr_train):
                input_ids, attention_mask, label = input_ids.to(self.args.device), attention_mask.to(self.args.device), label.to(self.args.device)

                net.model.zero_grad()

                logits = net.model(input_ids, attention_mask=attention_mask).logits  # Get logits from model output
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

            # show gradient's f norm
            if iter == 0 and printflag:
                for name, param in net.model.named_parameters():
                    if param.requires_grad:
                        print(name, torch.linalg.norm(param.grad, ord='fro'))
            if self.args.verbose:
                # print('correct labels:', correct_label)
                # print('answer labels:', answer_label)
                print('Update Epoch: {} \tAverage Loss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    iter, sum(batch_loss) / len(batch_loss), 100. * correct / len(self.ldr_train.dataset)))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.get_lora_parameters(), sum(epoch_loss) / len(epoch_loss), acc_[-1], epoch_loss[-1] > epoch_loss[0] * 1.1

