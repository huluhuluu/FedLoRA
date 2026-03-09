#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import torch
from torch import optim
import matplotlib.pyplot as plt
import os

# lora = [ [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 0
#          [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 1
#          [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 2
#                                                         ...]      # client n
# the lora parameters' shape is (client_num, layer_num, 2, 2, lora_size)
# in this function we average the lora parameters of all clients
# get average of matrix A, B, W(AB)
def FedAVG(lora):
    """
    Federated averaging of LoRA parameters.
    
    Args:
        lora: list of client LoRA parameters, each client has shape (layer_num, 2, 2)
              where 2 is for (query, value) and (A, B) matrices
    
    Returns:
        lora_avg: averaged LoRA parameters
        A_avg: averaged A matrices
        B_avg: averaged B matrices
        W_avg: averaged W = B * A matrices
        loss_list: reconstruction loss for each layer
    """
    # get avg A value, B value and average A*B value
    # A_avg = [ [q_lora_a, v_lora_a], ...]      # different layers
    # B_avg = [ [q_lora_b, v_lora_b], ...]      # different layers
    # W_avg = [ [q_lora_b * q_lora_b, v_lora_a * v_lora_b], ...]      # different layers
    # lora_avg shape is parameter lora shape
    A_avg, B_avg, W_avg, lora_avg = [], [], [], []

    # init A_avg, B_avg and W_avg
    for ind, layer in enumerate(lora[0]):
        A_avg.append([copy.deepcopy(layer[0][0]), copy.deepcopy(layer[1][0]) ] )
        B_avg.append([copy.deepcopy(layer[0][1]), copy.deepcopy(layer[1][1]) ] )
        W_avg.append([torch.mm(layer[0][1], layer[0][0]), torch.mm(layer[1][1], layer[1][0])])

    # aggregation the A_avg, B_avg and W_avg for all clients
    for ind, client in enumerate(lora[1:]): # traverse all clients
        for layer_ind, layer in enumerate(client): # traverse all layers
            for qv_ind, qv in enumerate(layer): # traverse all modules
                A_avg[layer_ind][qv_ind] = torch.add(A_avg[layer_ind][qv_ind], qv[0])
                B_avg[layer_ind][qv_ind] = torch.add(B_avg[layer_ind][qv_ind], qv[1])
                W_avg[layer_ind][qv_ind] = torch.add(W_avg[layer_ind][qv_ind], torch.mm(qv[1], qv[0]))

    # average the A_avg, B_avg and W_avg:
    for j in range(len(A_avg)):
        lora_para = []
        for k in range(len(A_avg[0])):   # traverse all modulesd
            A_avg[j][k] = torch.div(A_avg[j][k], len(lora))
            B_avg[j][k] = torch.div(B_avg[j][k], len(lora))
            W_avg[j][k] = torch.div(W_avg[j][k], len(lora))
            
            lora_para.append([A_avg[j][k], B_avg[j][k]])

        # get lora avg parameter
        lora_avg.append(lora_para)

    # compute loss |W_avg - B_avg * A_avg|
    loss_list, sum_loss = [], 0
    for j in range(len(W_avg)): # traverse all layers
        qv_loss_list, qv_loss = [], 0
        for k in range(len(W_avg[0])):
            # compute loss |W_avg - B_avg * A_avg|
            qv_loss = torch.norm(W_avg[j][k] - torch.mm(B_avg[j][k], A_avg[j][k]))
            # sum loss
            sum_loss += qv_loss.item()
            qv_loss_list.append(qv_loss.item())
        loss_list.append(qv_loss_list)
    # print('total fedavg loss: ', sum_loss)

    return lora_avg, A_avg, B_avg, W_avg, loss_list


def LoraGA(lora, lora_glob, lr, lora_rank):
    """
    Initialize LoRA parameters using LoraGA method.
    
    Args:
        lora: list of client LoRA parameters
        lora_glob: global LoRA parameters
        lr: learning rate used in local training
        lora_rank: rank of LoRA
    
    Returns:
        res: initialized LoRA parameters
    """
    dW, NW_avg = [], []

    # init W_avg
    for ind, layer in enumerate(lora[0]):
        NW_avg.append([torch.mm(layer[0][1], layer[0][0]), torch.mm(layer[1][1], layer[1][0])])

    # aggregation the NW_avg for all clients
    for ind, client in enumerate(lora[1:]): # traverse all clients
        for layer_ind, layer in enumerate(client): # traverse all layers
            for qv_ind, qv in enumerate(layer): # traverse all modules
                NW_avg[layer_ind][qv_ind] = torch.add(NW_avg[layer_ind][qv_ind], torch.mm(qv[1], qv[0]))
    
    # average NW_avg:
    for j in range(len(NW_avg)):
        for k in range(len(NW_avg[0])):   # traverse all modulesd
            NW_avg[j][k] = torch.div(NW_avg[j][k], len(lora))
    
    # NW_avg = W_glob - lr * dW => dW = (W_glob - NW_avg)/lr
    # compute dW
    for layer_ind, layer in enumerate(lora_glob):
        # layer: [ [(q_lora_a, q_lora_b)], [(v_lora_a, v_lora_b)] ]
        # NW_avg[layer_ind]: [ [W_q, W_v], ...]
        dW.append([
                    torch.div(torch.sub(torch.mm(layer[0][1], layer[0][0]), NW_avg[layer_ind][0]), lr),
                    torch.div(torch.sub(torch.mm(layer[1][1], layer[1][0]), NW_avg[layer_ind][1]), lr)
                ])

    # init lora parameters with LoraGA method
    res = []
    for layer_ind, layer in enumerate(dW):
        para = []
        # layer: [dW_q, dW_v]
        for qv_ind, qv in enumerate(layer):
            # qv: dW_q / dW_v        
            U, S, V = torch.linalg.svd(qv, full_matrices=False)
            d_out, r, gamma = qv.shape[0], lora_rank, 16

            # reconstruct B and A
            B =  U[:, r:2*r] * (d_out ** (1/4.0)) / (gamma ** (1/2.0))
            A = V[:r, :] * (d_out ** (1/4.0)) / (gamma ** (1/2.0))
            para.append([A, B])
        res.append(para) 

    return res

def SVD(W, r, autorank=False, theta=0.98):
    """
    Reconstruct matrix W = B * A using SVD decomposition.
    
    Args:
        W: matrix to decompose
        r: target rank
        autorank: if True, automatically determine rank based on singular value threshold
        theta: threshold ratio for autorank
    
    Returns:
        A: reconstructed A matrix
        B: reconstructed B matrix
        loss: reconstruction loss
        r: actual rank used
    """
    U, S, V = torch.linalg.svd(W, full_matrices=False)
    
    # use svd singular value guide rank
    if autorank:
        # get rank of SVD
        singular_val_sum, tmp_sum, rank = S.sum().item(), 0, 0
        
        for i in range(W.shape[0]):
            tmp_sum += S[i].item()
            if tmp_sum / singular_val_sum > theta:
                rank = i + 1
                break
        r = rank
    
    # reconstruct B and A
    B = U[:, :r] * torch.sqrt(S[:r])
    A = torch.sqrt(S[:r]).unsqueeze(1) * V[:r, :]
    W_approx = torch.mm(B, A)

    loss = torch.norm(W - W_approx)

    return A, B, loss, r


def ACA(W, k, autorank=False, eps=1e-5):
    """
    Reconstruct matrix W = B * A using Adaptive Cross Approximation (ACA) method.
    
    Args:
        W: matrix to decompose
        k: target rank
        autorank: if True, automatically determine rank based on epsilon threshold
        eps: threshold for autorank
    
    Returns:
        A: reconstructed A matrix
        B: reconstructed B matrix
        loss: reconstruction loss
        iter: actual rank used
    """
    with torch.no_grad():
        point, R, device = [], [W], W.device
        m, n, F_data = W.shape[0], W.shape[1], torch.norm(W)
        iter_count, max_iter = 0, min(m, n) if autorank else k
        for _ in range(max_iter):
            # 1. choose the pivot xk, yk
            
            # non-zero matrix
            nonzero_indices = torch.nonzero(R[-1], as_tuple=True)
            non_zero = R[-1][nonzero_indices]
            # max non-zero value
            max_value = torch.max(non_zero)
            # max non-zero index
            max_index_nonzero = torch.argmax(non_zero)
            # max non-zero index in the original matrix
            xk, yk = nonzero_indices[0][max_index_nonzero], nonzero_indices[1][max_index_nonzero]
            
            point.append((xk, yk))

            # 2. recursive
            uk, vk = R[-1][:,yk], R[-1][xk,:] / max_value
            R.append(R[-1] - torch.mm(uk.reshape(-1,1), vk.reshape(1,-1)))
            iter_count += 1

            # 3. epsilon
            if autorank and torch.norm(R[-1]) < eps * F_data:
                break

        # 3. construct the matrix B, A
        C = torch.zeros((iter_count, W.shape[0]), dtype=W.dtype).to(device)
        G = torch.zeros((iter_count, W.shape[1]), dtype=W.dtype).to(device)
        for ind, (xk, yk) in enumerate(point):
            C[ind,:] = W[:,yk]
            G[ind,:] = W[xk,:]
        
        M = torch.zeros((iter_count, iter_count), dtype=W.dtype).to(device)
        for i in range(iter_count):
            for j in range(iter_count):
                xi, yj = point[i][0], point[j][1]
                M[i][j] = W[xi][yj]

        B, A = C.T, torch.mm(torch.inverse(M), G).to(device)
        loss = torch.norm(W - torch.mm(B, A))
        return A, B, loss, iter_count


def ABOptim(W, r, init_A=None, init_B=None, lr=0.1, num_epoch=100):
    """
    Reconstruct matrix W = B * A using optimization method.
    
    Args:
        W: matrix to decompose
        r: target rank
        init_A: initial A matrix (optional)
        init_B: initial B matrix (optional)
        lr: learning rate for optimizer
        num_epoch: number of optimization epochs
    
    Returns:
        A: optimized A matrix
        B: optimized B matrix
        loss: final reconstruction loss
        r: rank used
    """
    W = W.clone().detach()
    # if no initial values provided, use random initialization
    if init_A is None or init_B is None:
        init_A = torch.randn(r, W.shape[1], dtype=W.dtype, device=W.device) * 0.01
        init_B = torch.randn(W.shape[0], r, dtype=W.dtype, device=W.device) * 0.01
    
    A = init_A.clone().detach().requires_grad_(True)
    B = init_B.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([A, B], lr=lr)

    for epoch in range(num_epoch):
        loss = torch.norm(W - torch.mm(B, A), p=2) / W.numel()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return A, B, loss, r


def FedReconstruct(lora, r, num_, reconstruct_fn, need_init_AB=False, **reconstruct_kwargs):
    """
    Generic federated reconstruction aggregation method.
    
    Args:
        lora: list of client LoRA parameters
        r: LoRA rank
        num_: list of data counts for each client
        reconstruct_fn: reconstruction function with signature 
                       (W, r, **kwargs) -> (A, B, loss, rank)
        need_init_AB: whether to pass initial A, B to reconstruction function
        **reconstruct_kwargs: additional arguments passed to reconstruct_fn
    
    Returns:
        result: reconstructed LoRA parameters
    """
    layer_weight_list = []
    AB_list = []  # for storing aggregated A, B (optional)
    
    # initialize: get weight shape
    device = lora[0][0][0][0].device
    for layer in lora[0]:
        q_shape = torch.Size((layer[0][1].shape[0], layer[0][0].shape[1]))
        q_dtype = layer[0][0].dtype
        v_shape = torch.Size((layer[1][1].shape[0], layer[1][0].shape[1]))
        v_dtype = layer[1][0].dtype
        layer_weight_list.append([
            torch.zeros(q_shape, dtype=q_dtype, device=device),
            torch.zeros(v_shape, dtype=v_dtype, device=device)
        ])
        if need_init_AB:
            AB_list.append([
                torch.zeros_like(layer[0][0], dtype=layer[0][0].dtype, device=device),
                torch.zeros_like(layer[0][1], dtype=layer[0][1].dtype, device=device),
                torch.zeros_like(layer[1][0], dtype=layer[1][0].dtype, device=device),
                torch.zeros_like(layer[1][1], dtype=layer[1][1].dtype, device=device),
            ])

    # aggregate W_i = B_i * A_i
    for client_index, client in enumerate(lora):
        for layer_index, layer in enumerate(client):
            layer_weight_list[layer_index][0] += torch.mm(layer[0][1], layer[0][0])
            layer_weight_list[layer_index][1] += torch.mm(layer[1][1], layer[1][0])
            if need_init_AB:
                AB_list[layer_index][0] += layer[0][0]
                AB_list[layer_index][1] += layer[0][1]
                AB_list[layer_index][2] += layer[1][0]
                AB_list[layer_index][3] += layer[1][1]

    # average
    for layer_weight in layer_weight_list:
        layer_weight[0] /= len(lora)
        layer_weight[1] /= len(lora)
    
    if need_init_AB:
        for layer in AB_list:
            for i in range(len(layer)):
                layer[i] /= len(lora)

    # reconstruct using reconstruct_fn
    result = copy.deepcopy(lora[0])
    total_loss = 0
    rank_list, loss_list = [], []
    
    for layer_index, layer_weight in enumerate(layer_weight_list):
        if need_init_AB:
            q_a, q_b, v_a, v_b = AB_list[layer_index]
            q_a, q_b, loss1, r1 = reconstruct_fn(layer_weight[0], r, init_A=q_a, init_B=q_b, **reconstruct_kwargs)
            v_a, v_b, loss2, r2 = reconstruct_fn(layer_weight[1], r, init_A=v_a, init_B=v_b, **reconstruct_kwargs)
        else:
            q_a, q_b, loss1, r1 = reconstruct_fn(layer_weight[0], r, **reconstruct_kwargs)
            v_a, v_b, loss2, r2 = reconstruct_fn(layer_weight[1], r, **reconstruct_kwargs)
        
        total_loss += loss1.item() + loss2.item()
        result[layer_index][0] = (q_a, q_b)
        result[layer_index][1] = (v_a, v_b)
        loss_list.append([loss1.item(), loss2.item()])
        rank_list.append([r1, r2])
    
    return result


# ============ Convenience wrapper functions ============

def FedSVD(lora, r, num_, autorank=False, theta=0.98):
    """
    Federated reconstruction using SVD method.
    
    Args:
        lora: list of client LoRA parameters
        r: LoRA rank
        num_: list of data counts for each client
        autorank: whether to automatically determine rank
        theta: threshold ratio for autorank
    
    Returns:
        result: reconstructed LoRA parameters
    """
    return FedReconstruct(lora, r, num_, SVD, need_init_AB=False, autorank=autorank, theta=theta)


def FedACA(lora, r, num_, autorank=False, eps=1e-5):
    """
    Federated reconstruction using ACA (Adaptive Cross Approximation) method.
    
    Args:
        lora: list of client LoRA parameters
        r: LoRA rank
        num_: list of data counts for each client
        autorank: whether to automatically determine rank
        eps: threshold for autorank
    
    Returns:
        result: reconstructed LoRA parameters
    """
    return FedReconstruct(lora, r, num_, ACA, need_init_AB=False, autorank=autorank, eps=eps)


def FedABOptim(lora, r, num_, lr=0.1, num_epoch=100):
    """
    Federated reconstruction using optimization method with aggregated A, B as initialization.
    
    Args:
        lora: list of client LoRA parameters
        r: LoRA rank
        num_: list of data counts for each client
        lr: learning rate for optimizer
        num_epoch: number of optimization epochs
    
    Returns:
        result: reconstructed LoRA parameters
    """
    return FedReconstruct(lora, r, num_, ABOptim, need_init_AB=True, lr=lr, num_epoch=num_epoch)
