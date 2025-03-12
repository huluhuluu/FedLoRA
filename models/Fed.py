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
def FedAVG(lora, args):
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

# lora = [ [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 0
#          [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 1
#          [[(q_lora_a, q_lora_b), (v_lora_a, v_lora_b)], ...]      # client 2
#                                                         ...]      # client n
# the lora parameters' shape is (client_num, layer_num, 2, 2, lora_size)
# in this function we average the lora parameters of all clients
# use loraga method to init lora parameters
def LoraGA(lora, lora_glob, args):
    # get avg A value, B value and average A*B value
    # A_avg = [ [q_lora_a, v_lora_a], ...]      # different layers
    # B_avg = [ [q_lora_b, v_lora_b], ...]      # different layers
    # W_avg = [ [q_lora_b * q_lora_b, v_lora_a * v_lora_b], ...]      # different layers
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
                    torch.div(torch.sub(torch.mm(layer[0][1], layer[0][0]), NW_avg[layer_ind][0]), args.lr),
                    torch.div(torch.sub(torch.mm(layer[1][1], layer[1][0]), NW_avg[layer_ind][1]), args.lr)
                ])

    # init lora parameters with LoraGA method
    res = []
    for layer_ind, layer in enumerate(dW):
        para = []
        # layer: [dW_q, dW_v]
        for qv_ind, qv in enumerate(layer):
            # qv: dW_q / dW_v        
            U, S, V = torch.linalg.svd(qv, full_matrices=False)
            d_out, r, gamma = qv.shape[0], args.lora_rank, 16

            # reconstruct B and A
            B =  U[:, r:2*r] * (d_out ** (1/4.0)) / (gamma ** (1/2.0))
            A = V[:r, :] * (d_out ** (1/4.0)) / (gamma ** (1/2.0))
            para.append([A, B])
        res.append(para) 

    return res

def analyse_svd(W, U, S, V):
    # resolve file path conflicts
    png_path = 'save/analyse/svd.png'
    txt_path = 'save/analyse/svd.txt'
    # if exists, add a number to the file name
    if os.path.exists(png_path):
        index = 1
        while index < 5:
            new_png_path = f'save/analyse/svd_{index}.png'
            new_txt_path = f'save/analyse/svd_{index}.txt'
            if not os.path.exists(new_png_path):
                png_path = new_png_path
                txt_path = new_txt_path
                break
            index += 1
        if index == 5:
            return
    # statistics
    errors, ratios, s = [], [], []
    singular_val_sum = 0
    for rank in range(1, W.shape[0]//5 + 1):
        # decompose W
        B = U[:, :rank] * torch.sqrt(S[:rank])
        A = torch.sqrt(S[:rank]).unsqueeze(1) * V[:rank, :]
        W_approx = torch.mm(B, A)
        # compute loss
        loss = torch.norm(W - W_approx)
        # compute singular value sum
        singular_val_sum += S[rank].item()

        errors.append(loss.item())
        ratios.append(singular_val_sum)
        s.append(S[rank].item())
    # plot
    plt.figure(figsize=(15, 5))
    
    # fig1
    plt.subplot(2, 2, 1)
    plt.plot(range(len(errors)), errors)
    plt.title('Reconstruction Error')
    plt.xlabel('r')
    plt.ylabel('Error')
    # fig2
    plt.subplot(2, 2, 2)
    ratios = [ s_ / singular_val_sum for s_ in ratios ]
    plt.plot(range(len(ratios)), ratios)
    plt.title('Singular Values Ratio')
    plt.xlabel('r')
    plt.ylabel('Ratio')
    # fig 3
    plt.subplot(2, 2, 3)
    plt.plot(range(len(s)), s)
    plt.title('Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # fig4
    plt.subplot(2, 2, 4)
    plt.plot(range(len(errors)), errors, label='Errors')
    plt.plot(range(len(ratios)), ratios, label='Ratios')
    plt.plot(range(len(s)), s, label='Singular Values')
    plt.title('Combined Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(png_path)
    # save to txt
    with open(txt_path, 'w') as f:
        for i in range(len(errors)):
            f.write(f'Rank: {i+1:.3d}, Error: {errors[i]:.10f}, Ratio: {ratios[i]:.10f}, Singular Value: {s[i]:.10f}\n')

def SVD(W, r: int, args):
    # SVD
    U, S, V = torch.linalg.svd(W, full_matrices=False)

    # print("U:\n", U)
    # print("S:", S[0:15], S.shape)
    # print("V:\n", V)
    # analyse_svd(W, U, S, V)
    
    # use svd singular value guide rank
    if args.autorank:
        # get rank of SVD
        singular_val_sum, tmp_sum, rank = S.sum().item(), 0, 0
        
        for i in range(W.shape[0]):
            tmp_sum += S[i].item()
            if tmp_sum / singular_val_sum > args.theta:
                rank = i + 1
                break
        r = rank
    
    # reconstruct B and A
    B = U[:, :r] * torch.sqrt(S[:r])
    A = torch.sqrt(S[:r]).unsqueeze(1) * V[:r, :]
    W_approx = torch.mm(B, A)

    # loss = torch.norm(W - W_approx) / W.numel()
    loss = torch.norm(W - W_approx)
    # if args.verbose:
    #     print(f"reconstruct loss: {loss:.10f}", )

    return A, B, loss, r

def FedSVD(lora, r: int, args, num_ : list):
    _, _, _, _, loss_avg = FedAVG(lora, args)
    # print('fedavg loss list:', loss_avg)
    # 1. W_{i} = B_{i}A_{i}
    # 2. W = avg(W_{i})
    # 3. A, B = SVD(W, r)
    sum_ = sum(num_)
    
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

    for client_index, client in enumerate(lora):
        client_weight = num_[client_index] * 1.0 / sum_
        for layer_index, layer in enumerate(client):
            layer_weight_list[layer_index][0] +=  torch.mm(
                layer[0][1], layer[0][0])
            layer_weight_list[layer_index][1] +=  torch.mm(
                layer[1][1], layer[1][0])

    # avg
    for layer_weight in layer_weight_list:
        layer_weight[0] /= len(lora)
        layer_weight[1] /= len(lora)

    # SVD
    result = copy.deepcopy(lora[0])
    total_loss = 0
    rank_list, loss_svd = [], []
    for layer_index, layer_weight in enumerate(layer_weight_list):
        q_a, q_b, loss1, r1 = SVD(layer_weight[0], r, args)
        v_a, v_b, loss2, r2 = SVD(layer_weight[1], r, args)
        total_loss += loss1.item() + loss2.item()
        result[layer_index][0] = (q_a, q_b)
        result[layer_index][1] = (v_a, v_b)
        loss_svd.append([loss1.item(), loss2.item()])
        rank_list.append([r1, r2])
    
    # print(f"total fedsvd loss is {total_loss:.10f}")
    # print(f"fedsvd loss list:", loss_svd)
    # print(f"rank list: {rank_list}")
    # print avg rank
    rank = sum([r1 + r2 for r1, r2 in rank_list]) // ( len(rank_list) * 2 )
    # print(f"avg rank: {rank}")
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
            pass
            # print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.10f}')
        # if loss < 1e-8:
        #     break

    # 检查最终的 A 和 B
    # print(f'Final Loss: {loss.item():.10f}')
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

# reconstruct matrix W = B * A
# using ACA(Adaptive Cross Approximation) method
def ACA(W: torch.Tensor, k : int, args, eps = 1e-5):
    with torch.no_grad():
        point, R, device = [], [W], W.device
        m, n, F_data = W.shape[0], W.shape[1], torch.norm(W)
        iter, max_iter = 0, min(m, n) if args.autorank else k
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
            iter += 1

            # 3. epsilon
            if args.autorank and torch.norm(R[-1]) < eps * F_data:
                break

        # 3. construct the matrix B, A
        C = torch.zeros((iter, W.shape[0]), dtype=W.dtype).to(device)
        G = torch.zeros((iter, W.shape[1]), dtype=W.dtype).to(device)
        for ind, (xk, yk) in enumerate(point):
            C[ind,:] = W[:,yk]
            G[ind,:] = W[xk,:]
        
        M = torch.zeros((iter, iter), dtype=W.dtype).to(device)
        for i in range(iter):
            for j in range(iter):
                xi, yj = point[i][0], point[j][1]
                M[i][j] = W[xi][yj]

        B, A = C.T, torch.mm(torch.inverse(M), G).to(device)
        l = torch.norm(W - torch.mm(B, A))
        return A, B, l, iter

def FedACA(lora, r: int, args, num_ : list):
    _, _, _, _, loss_avg = FedAVG(lora, args)
    # print('fedavg loss list:', loss_avg)
    # 1. W_{i} = B_{i}A_{i}
    # 2. W = avg(W_{i})
    # 3. A, B = SVD(W, r)
    sum_ = sum(num_)
    
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

    for client_index, client in enumerate(lora):
        client_weight = num_[client_index] * 1.0 / sum_
        for layer_index, layer in enumerate(client):
            layer_weight_list[layer_index][0] +=  torch.mm(
                layer[0][1], layer[0][0])
            layer_weight_list[layer_index][1] +=  torch.mm(
                layer[1][1], layer[1][0])

    # avg
    for layer_weight in layer_weight_list:
        layer_weight[0] /= len(lora)
        layer_weight[1] /= len(lora)

    # ACA
    result = copy.deepcopy(lora[0])
    total_loss = 0
    rank_list, loss_svd = [], []
    for layer_index, layer_weight in enumerate(layer_weight_list):
        q_a, q_b, loss1, r1 = ACA(layer_weight[0], r, args)
        v_a, v_b, loss2, r2 = ACA(layer_weight[1], r, args)
        total_loss += loss1.item() + loss2.item()
        result[layer_index][0] = (q_a, q_b)
        result[layer_index][1] = (v_a, v_b)
        loss_svd.append([loss1.item(), loss2.item()])
        rank_list.append([r1, r2])
    
    # print(f"total fedsvd loss is {total_loss:.10f}")
    # print(f"fedsvd loss list:", loss_svd)
    # print(f"rank list: {rank_list}")
    # print avg rank
    rank = sum([r1 + r2 for r1, r2 in rank_list]) // ( len(rank_list) * 2 )
    # print(f"avg rank: {rank}")
    return result