#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
# set full print
import torch
torch.set_printoptions(profile="full")

# get label counts
def get_label_count(dataset_train, dataset_test):
    # get label counts
    label_counts_train = {}
    label_counts_test = {}

    # traverse all train data
    for i in range(len(dataset_train)):
        if label_counts_train.get(dataset_train[i]['label']) is None:
            label_counts_train[dataset_train[i]['label']] = 1
        else:
            label_counts_train[dataset_train[i]['label']] += 1
    
    # traverse all test data
    for i in range(len(dataset_test)):
        if label_counts_test.get(dataset_test[i]['label']) is None:
            label_counts_test[dataset_test[i]['label']] = 1
        else:
            label_counts_test[dataset_test[i]['label']] += 1
    

    # print label counts res
    for key in label_counts_train.keys():
        print('label: {},train count: {} test count: {}'.format(
            key, 
            label_counts_train[key], 
            label_counts_test[key] if label_counts_test.get(key) is not None else 0))

# get label counts after sampling
# dataset: dataset_train
# users: sampled index for users
def get_label_count_sample(dataset, users, verbose = True):
    label_counts = {}
    users = list(users)
    for i in range(len(users)):
        ind = int(users[i])
        if label_counts.get(dataset[ind]['label']) is None:
            label_counts[dataset[ind]['label']] = 1
        else:
            label_counts[dataset[ind]['label']] += 1
    
    sum_label_count = 0
    for key in label_counts.keys():
        if verbose:
            print('label: {}, count: {}'.format(key, label_counts[key]))
        sum_label_count += 1
    print('label num: {}, data num {}'.format(sum_label_count, len(users)))


def analyse_svd(W, U, S, V):
    """
    Analyze SVD decomposition and save results.
    """
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

