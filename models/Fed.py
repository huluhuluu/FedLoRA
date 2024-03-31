#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import torch
from torch import optim

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


def SVD(W, r: int):
    # 进行奇异值分解
    U, S, V = torch.linalg.svd(W, full_matrices=False)

    # print("U:\n", U)
    # print("S:", S, S.shape)
    # print("V:\n", V)

    B = U[:, :r] * torch.sqrt(S[:r])
    A = torch.sqrt(S[:r]).unsqueeze(1) * V[:r, :]
    W_approx = torch.mm(B, A)

    # loss = torch.norm(W - W_approx) / W.numel()
    loss = torch.norm(W - W_approx)
    print(f"重构误差: {loss:.10f}", )

    return A, B, loss


def FedSVD(lora, r: int):
    # 1. W_{i} = B_{i}A_{i}
    # 2. W = avg(W_{i})
    # 3. A, B = SVD(W, r)

    layer_weight_list = []
    # initialize
    # get weight shape
    device = lora[0][0][0][0].device
    for layer in lora[0]:
        # a 8*768, b 768*8
        q_shape = torch.Size((layer[0][1].shape[0], layer[0][0].shape[1]))
        q_dtype = layer[0][0].dtype
        v_shape = torch.Size((layer[1][1].shape[0], layer[1][0].shape[1]))
        v_dtype = layer[1][0].dtype
        layer_weight_list.append(
            [
                torch.zeros(q_shape, dtype=q_dtype, device=device),
                torch.zeros(v_shape, dtype=v_dtype, device=device)
            ]
        )

    for client in lora:
        for layer_index, layer in enumerate(client):
            layer_weight_list[layer_index][0] += torch.mm(
                layer[0][1], layer[0][0])
            layer_weight_list[layer_index][1] += torch.mm(
                layer[1][1], layer[1][0])

    # avg
    for layer_weight in layer_weight_list:
        layer_weight[0] /= len(lora)
        layer_weight[1] /= len(lora)

    # SVD
    result = copy.deepcopy(lora[0])
    total_loss = 0
    for layer_index, layer_weight in enumerate(layer_weight_list):
        q_a, q_b, loss = SVD(layer_weight[0], r)
        total_loss += loss
        v_a, v_b, loss = SVD(layer_weight[1], r)
        total_loss += loss
        result[layer_index][0] = (q_a, q_b)
        result[layer_index][1] = (v_a, v_b)

    print(f"total average loss is {total_loss:.10f}")

    return result


def FedABOptimHandler(W, A, B):
    # 定义优化器
    W = W.clone().detach()
    A = A.clone().detach().requires_grad_(True)
    B = B.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([A, B], lr=0.1)
    num_epoch = 100

    for epoch in range(num_epoch):
        loss = torch.norm(W - torch.mm(B, A), p=2) / W.numel()
        # 梯度归零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()
        if epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.10f}')
        # if loss < 1e-8:
        #     break

    # 检查最终的 A 和 B
    print(f'Final Loss: {loss.item():.10f}')
    return A, B


def FedABOptim(lora):
    # 1. W_{i} = B_{i}A_{i}
    # 2. W = avg(W_{i})
    # 3. optim || W - BA ||
    # init B=avg(B_{i}), A=avg(A_{i})

    layer_weight_list = []
    AB_list = []

    # initialize
    # get weight shape
    device = lora[0][0][0][0].device
    for layer in lora[0]:
        # a 8*768, b 768*8
        q_shape = torch.Size((layer[0][1].shape[0], layer[0][0].shape[1]))
        q_dtype = layer[0][0].dtype
        v_shape = torch.Size((layer[1][1].shape[0], layer[1][0].shape[1]))
        v_dtype = layer[1][0].dtype
        layer_weight_list.append(
            [
                torch.zeros(q_shape, dtype=q_dtype, device=device),
                torch.zeros(v_shape, dtype=v_dtype, device=device)
            ]
        )
        # q_a, q_b, v_a, v_b
        AB_list.append(
            [
                torch.zeros_like(layer[0][0], dtype=layer[0][0].dtype, device=device),
                torch.zeros_like(layer[0][1], dtype=layer[0][1].dtype, device=device),
                torch.zeros_like(layer[1][0], dtype=layer[1][0].dtype, device=device),
                torch.zeros_like(layer[1][1], dtype=layer[1][1].dtype, device=device),
            ]
        )

    for client in lora:
        for layer_index, layer in enumerate(client):
            layer_weight_list[layer_index][0] += torch.mm(
                layer[0][1], layer[0][0])
            layer_weight_list[layer_index][1] += torch.mm(
                layer[1][1], layer[1][0])

            AB_list[layer_index][0] += layer[0][0]
            AB_list[layer_index][1] += layer[0][1]
            AB_list[layer_index][2] += layer[1][0]
            AB_list[layer_index][3] += layer[1][1]

    for layer_weight in layer_weight_list:
        layer_weight[0] /= len(lora)
        layer_weight[1] /= len(lora)

    for layer in AB_list:
        for i in range(len(layer)):
            layer[i] /= len(lora)

    # ABOptim
    result = copy.deepcopy(lora[0])
    for layer_index, layer_weight in enumerate(layer_weight_list):
        q_a, q_b, v_a, v_b = AB_list[layer_index]
        q_a, q_b = FedABOptimHandler(layer_weight[0], q_a, q_b)
        v_a, v_b = FedABOptimHandler(layer_weight[1], v_a, v_b)

        result[layer_index][0] = (q_a, q_b)
        result[layer_index][1] = (v_a, v_b)

    return result
