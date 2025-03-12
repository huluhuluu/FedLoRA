# Introduction
This repo uses fedavg to fine tune a bert model with lora method.

## References
### paper
Fedavg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

Bert:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

Lora:[LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9) 


### code
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561


# how to use profile
1. set local_bs in profile.py and test gpu id
2. replace ```/home/user/miniconda3/envs/zjh_fed/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py``` to modeling_bert.py
3. run profile.py