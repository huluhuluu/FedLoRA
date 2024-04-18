#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import copy
import os

class BertLora():
    def __init__(self, num_labels, rank = 16, lora_alpha = 16, args = None):
        # Model id to load the tokenizer
        model_id = "bert-base-uncased"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./cache/tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/data/huluhulu/fedbertlora/cache/tokenizer')

        # get pretrained model
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels, cache_dir='./cache/model')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels, cache_dir='/data/huluhulu/fedbertlora/cache/model')
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
    
    # get lora parameters
    # [[[q_lora_a, q_lora_b], [v_lora_a, v_lora_b]], ...]
    def get_lora_parameters(self):
        # get lora parameters from model
        lora_parameters = []
        # traverse all layers in bert.encoder.layer
        for layer in self.model.base_model.model.bert.encoder.layer:
            q_lora = []
            v_lora = []
            # get lora parameters from query 
            for name, module in layer.attention.self.named_children():
                if name == 'query':
                    q_lora = [module.lora_A['default'].weight.detach().clone(), module.lora_B['default'].weight.detach().clone()]
            # get lora parameters from value
            for name, module in layer.attention.self.named_children():
                if name == 'value':
                    v_lora = [module.lora_A['default'].weight.detach().clone(), module.lora_B['default'].weight.detach().clone()]
            # append lora parameters to list
            if q_lora is not None and v_lora is not None:
                lora_parameters.append([q_lora, v_lora])
        return lora_parameters
    
    # apply lora parameters to model
    # lora_parameters = [[[q_lora_a, q_lora_b], [v_lora_a, v_lora_b]], ...]
    def put_lora_parameters(self, lora_parameters):
        # Apply lora parameters to the model
        for layer, lora_param in zip(self.model.base_model.model.bert.encoder.layer, lora_parameters):
            q_lora, v_lora = lora_param
            # Apply q_lora parameters
            for name, module in layer.attention.self.named_children():
                if name == 'query':
                    # transforme tensor to parameter
                    q_lora_A_weight = torch.nn.Parameter(q_lora[0].detach().clone())
                    q_lora_B_weight = torch.nn.Parameter(q_lora[1].detach().clone())
                    # assign parameter to model's weight
                    module.lora_A['default'].weight = q_lora_A_weight
                    module.lora_B['default'].weight = q_lora_B_weight
            # Apply v_lora parameters
            for name, module in layer.attention.self.named_children():
                if name == 'value':
                    # transforme tensor to parameter
                    v_lora_A_weight = torch.nn.Parameter(v_lora[0].detach().clone())
                    v_lora_B_weight = torch.nn.Parameter(v_lora[1].detach().clone())
                    # assign parameter to model's weight
                    module.lora_A['default'].weight = v_lora_A_weight
                    module.lora_B['default'].weight = v_lora_B_weight
