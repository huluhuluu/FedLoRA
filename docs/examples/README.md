# 配置入口

配置统一放在 [experiments/configs](../../experiments/configs/)。

- agnews_gpt2_smoke.json：双客户端启动示例。
- agnews_gpt2_shared_smoke.json：四卡联邦、临时借两卡评估。
- agnews_gpt2_centralized_smoke.json：相同模型与任务的标准中心式训练。
- agnews_gpt2.json：完整数据的联邦配置。
- cluster.example.json：不等卡数资源清单，需修改地址和网卡。
- compare.example.json：中心式初始化、联邦初始化与聚合的分类对比矩阵。
- suite.example.json：兼容的简单参数网格。

字段见 [配置](../configuration.md)，扩展见 [custom](../custom.md)，对比运行见 [实验](../experiments.md)。
