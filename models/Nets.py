#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from utils.helper import cache_path

class BertLora():
    def __init__(self, num_labels, rank = 16, lora_alpha = 16, args = None):
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir= cache_path + '/tokenizer')

        # get pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels, cache_dir= cache_path + '/model')
        # Define LoRA Config
        lora_config = LoraConfig(
            r = rank,
            lora_alpha = lora_alpha,
            target_modules= ["query", "value"],
            lora_dropout = 0.05,
            bias = "none"
        )
        # apply lora to bert model
        self.model = get_peft_model(self.model, lora_config)

        # print trainbale parameter
        self.model.print_trainable_parameters()
        self.args = args
        self.model.to(self.args.device)

        # get name of lora layer
        self.name_lora_para, name_lora = [], [] 
        for name in self.model.state_dict():
            if 'lora' in name and 'weight' in name:
                name_lora.append(name)
        # reshpae name list to [ [ [q_lora_a, q_lora_b], [v_lora_a, v_lora_b] ], ...]
        for ind in range(0, len(name_lora)//4):
            self.name_lora_para.append([name_lora[4*ind:4*ind+2], name_lora[4*ind+2:4*ind+4]])

    # get lora parameters
    # [[[q_lora_a, q_lora_b], [v_lora_a, v_lora_b]], ...]
    def get_lora_parameters(self):
        # get lora parameters from model
        lora_parameters = []
        model_dict = self.model.state_dict()
        # traverse all query and value in moudle(
        for layer in self.name_lora_para:
            # get lora parameters from query and value
            # print(layer)
            q_lora = [model_dict[layer[0][0]].detach().clone(), model_dict[layer[0][1]].detach().clone()]
            v_lora = [model_dict[layer[1][0]].detach().clone(), model_dict[layer[1][1]].detach().clone()]
            lora_parameters.append([q_lora, v_lora])
        return lora_parameters
    
    # apply lora parameters to model
    # lora_parameters = [[[q_lora_a, q_lora_b], [v_lora_a, v_lora_b]], ...]
    def put_lora_parameters(self, lora_parameters):
        # traverse all parameters in model
        ind = 0
        for name, module in self.model.named_modules():
            # get layer index, query or value index, and matrix a or b index
            layer_ind, qv_ind, ab_index = ind // 4, (ind % 4) // 2, (ind % 4) % 2
            # print(ind, layer_ind, qv_ind, ab_index, name)
            # update lora parameters
            if name + '.default.weight' == self.name_lora_para[layer_ind][qv_ind][ab_index]:
                # find the corresponding module
                # print(ind, layer_ind, qv_ind, ab_index, name)
                new_weight = torch.nn.Parameter(lora_parameters[layer_ind][qv_ind][ab_index].detach().clone())
                module['default'].weight = new_weight
                ind += 1
            
            # break if all lora parameters are updated
            if ind >= len(self.name_lora_para)*4:
                break