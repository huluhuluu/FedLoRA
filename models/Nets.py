#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, LoraModel, get_peft_model
from utils.helper import cache_path
from models.Fed import SVD, ACA
class BertLora():
    def __init__(self, num_labels, rank = 16, lora_alpha = 16, args = None, contrast_flag = False):
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir= cache_path + '/tokenizer')

        # get pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels, cache_dir= cache_path + '/model')
        # Define LoRA Config
        if args.init in ['gaussian', 'olora', 'eva']:
            lora_config = LoraConfig(
                r = rank,
                lora_alpha = lora_alpha,
                target_modules= ["query", "value"],
                lora_dropout = 0.05,
                bias = "none",
                init_lora_weights=args.init
            )
        else:
            lora_config = LoraConfig(
                r = rank,
                lora_alpha = lora_alpha,
                target_modules= ["query", "value"],
                lora_dropout = 0.05,
                bias = "none"
            )
        if not contrast_flag:
            # apply lora to bert model
            self.model = get_peft_model(self.model, lora_config)

            # print trainbale parameter
            self.model.print_trainable_parameters()
        else:
            # freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
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
        rank = []
        model_dict = self.model.state_dict()
        # traverse all query and value in moudle(
        for layer in self.name_lora_para:
            # get lora parameters from query and value
            # print(layer)
            q_lora = [model_dict[layer[0][0]].detach().clone(), model_dict[layer[0][1]].detach().clone()]
            v_lora = [model_dict[layer[1][0]].detach().clone(), model_dict[layer[1][1]].detach().clone()]
            lora_parameters.append([q_lora, v_lora])
            rank.append([(q_lora[0].shape),(v_lora[0].shape)])
        
        # print("ranks: ", rank)
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

    def allocate_rank_model(self, layer_num, piece, order, num_labels, args):
        begin_layer, end_layer = layer_num // piece * order, layer_num // piece * (order + 1)
        print("begin_layer: ", begin_layer, "end_layer: ", end_layer)
        allocate_rank = args.lora_rank * piece
        
        # Define LoRA Config
        lora_config = LoraConfig(
            r = allocate_rank,
            lora_alpha = allocate_rank * 2,
            target_modules= ["query", "value"],
            lora_dropout = 0.05,
            bias = "none"
        )
        # apply lora to bert model
        for idx, layer in enumerate(self.model.bert.encoder.layer):
            if begin_layer <= idx and idx < end_layer:
                adapter_name = f"default" 
                layer.attention = LoraModel(layer.attention, lora_config, adapter_name)

        # get name of lora layer
        self.name_lora_para, name_lora = [], [] 
        for name in self.model.state_dict():
            if 'lora' in name and 'weight' in name:
                name_lora.append(name)
        # reshpae name list to [ [ [q_lora_a, q_lora_b], [v_lora_a, v_lora_b] ], ...]
        for ind in range(0, len(name_lora)//4):
            self.name_lora_para.append([name_lora[4*ind:4*ind+2], name_lora[4*ind+2:4*ind+4]])
        print(self.name_lora_para)
        print(self.model)

    def reshape_model(self, layer_num, piece, reshape_ranks, args):
        for p in range(piece):
            begin_layer, end_layer = layer_num // piece * p, layer_num // piece * (p + 1)
            print("begin_layer: ", begin_layer, "end_layer: ", end_layer)
            allocate_rank = reshape_ranks[p]
            # Define LoRA Config
            lora_config = LoraConfig(
                r = allocate_rank,
                lora_alpha = allocate_rank * 2,
                target_modules= ["query", "value"],
                lora_dropout = 0.05,
                bias = "none"
            )
            # apply lora to bert model
            for idx, layer in enumerate(self.model.bert.encoder.layer):
                if begin_layer <= idx and idx < end_layer:
                    adapter_name = f"default" 
                    layer.attention = LoraModel(layer.attention, lora_config, adapter_name)

        # get name of lora layer
        self.name_lora_para, name_lora = [], [] 
        for name in self.model.state_dict():
            if 'lora' in name and 'weight' in name:
                name_lora.append(name)
        # reshpae name list to [ [ [q_lora_a, q_lora_b], [v_lora_a, v_lora_b] ], ...]
        for ind in range(0, len(name_lora)//4):
            self.name_lora_para.append([name_lora[4*ind:4*ind+2], name_lora[4*ind+2:4*ind+4]])
        print(self.name_lora_para)
        print(self.model)