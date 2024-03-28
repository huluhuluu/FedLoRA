#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import torch

# lora = [ [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 0
#          [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 1
#          [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 2
#                                                         ...]      # client n
# the lora parameters' shape is (client_num, layer_num, 2, 2, lora_size)
# in this function we average the lora parameters of all clients
def FedAvg(lora):
    # copy client 0's lora parameters
    lora_avg = copy.deepcopy(lora[0])
    
    # aggregation the lora parameters of all clients
    for i in range(1, len(lora)):   # traverse all clients
        for j in range(len(lora_avg)):  # traverse all layers
            for k in range(len(lora_avg[j])):   # traverse all modules
                for l in range(len(lora_avg[j][k])):    # traverse all parameters
                    lora_avg[j][k][l] = torch.add(lora_avg[j][k][l], lora[i][j][k][l])

    # average the lora parameters
    for j in range(len(lora_avg)):
        for k in range(len(lora_avg[j])):
            for l in range(len(lora_avg[j][k])):
                lora_avg[j][k][l] = torch.div(lora_avg[j][k][l], len(lora))
    return lora_avg
