# Introduction
This repo uses fedavg to fine tune a bert model with lora method.

## References
### paper
Fedavg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

Bert:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

Lora:[LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9) 


### code
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561


# debug command
validate lora
python main_fed.py --iid --num_users 1 --local_ep 1 --local_bs 32 --lr 0.01 --epochs 50

validate fed
python main_fed.py --iid --num_users 100 --local_bs 32 --lr 0.001 --epochs 80


- todo

data parallel

python main_fed.py --iid --num_users 63 --local_ep 1 --local_bs 16 --lr 0.01 --epochs 80