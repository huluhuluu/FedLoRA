# FedLoRA

## 项目简介

本项目在 [Shaoxiong Ji](https://github.com/shaoxiongji) 的 [Federated-Learning](https://github.com/shaoxiongji/Federated-Learning) 项目基础上进行改进，实现了 **FedLoRA**：将联邦学习（Federated Learning）与 LoRA（Low-Rank Adaptation）技术相结合，用于对 BERT、GPT-2 等预训练语言模型进行参数高效的联邦微调。

### 核心特性

- **参数高效**：只训练和传输 LoRA 低秩矩阵（A、B），大幅减少通信开销
- **多种聚合算法**：支持 FedAvg、FedSVD、FedABOptim、FedACA 等多种联邦聚合方法
- **多种初始化方法**：支持 Gaussian、Kaiming、LoRAGA、PiSSA、SVD、ACA 等初始化策略
- **灵活的数据分布**：支持 IID、Shard Non-IID、Dirichlet Non-IID 等数据分布方式
- **丰富的数据集**：支持 Banking77、AGNews、SST2 以及 GLUE 多个子任务

## 文件结构说明

```
FedLoRA/
├── main_fed.py              # 主训练入口，包含联邦学习训练循环
├── contrast_rank.py         # 对比学习相关实验脚本
├── contrast_rank_piece.py   # 分片对比学习实验脚本
├── run.sh                   # 训练启动脚本示例
├── requirements.txt         # 项目依赖列表
│
├── models/
│   ├── Fed.py               # 联邦聚合算法实现（FedAvg, FedSVD, FedABOptim, LoraGA, FedACA）
│   ├── Nets.py              # BERT/GPT-2 LoRA 模型定义和参数管理
│   ├── Update.py            # 本地客户端训练逻辑和数据加载
│   └── test.py              # 测试集评估函数
│
└── utils/
    ├── options.py           # 命令行参数解析和配置定义
    ├── sampling.py          # 数据采样策略（IID/Non-IID 分布）
    ├── helper.py            # 辅助工具函数（标签统计等）
    └── draw.py              # 训练曲线绘图工具
```

## 快速开始

```bash
# 基本运行示例
python main_fed.py \
    --model bert-base-uncased \
    --dataset agnews \
    --dirichlet_noniid \
    --num_users 1000 \
    --local_bs 32 \
    --local_ep 1 \
    --lr 0.1 \
    --epochs 300 \
    --gpu 0 \
    --alpha 10.0
```

更多参数说明请参考 `utils/options.py`。

## 参考文献

### 论文
- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9)

### 代码参考
- Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561
- https://github.com/alexriggio/BERT-LoRA-TensorRT/tree/main/notebooks
- https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/pytorch-2-0-bert-text-classification.ipynb

## 相关项目

- **性能分析工具**: [HF-LLM-Profiler](https://github.com/huluhuluu/HF-LLM-Profiler) - HuggingFace 大语言模型性能分析工具
