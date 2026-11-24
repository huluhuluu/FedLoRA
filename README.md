# FedLoRA

FedLoRA 是面向算法实验的联邦 LoRA 模拟器，同时提供标准中心式训练作为对比。一次实验选择一个模型、一个数据集和一个任务；逻辑客户端配置与实际 GPU 资源分别设置。

支持的功能：

- 分类、文本对分类与因果语言建模；任务插件统一接管批处理、损失和指标。
- Transformers AutoModel 与 PEFT 命名 A/B 状态，适配 GPT-2、BERT、Qwen 等兼容架构的目标模块。
- IID、标签 Shard、Dirichlet 数据划分；固定分组、资源配置分组、标签分布分组。
- 均匀或组间均衡选择；扁平、分层 FedAvg 与支持异构 rank/alpha 的 QR/SVD 聚合。
- 公共初始化插件；共享、私有或冻结头；客户端独立预算、优化器状态和模拟硬件。
- 标准中心式全局 batch 梯度更新，有界流式归约，持续的优化器、采样器、调度器与 AMP 状态。
- 串行、单机多卡、多机多卡；驻留模型与动态任务分发。
- 部分 GPU 临时评估、其余 GPU 继续训练；版本化快照、有界队列与评估完成后归还 worker。
- 联邦虚拟时间、通信体积、实测标定；JSON 配置、分类对比矩阵、检查点与续跑。

从 [install.md](install.md) 开始。四卡示例均限制数据和训练预算，需要根据本机修改资源路径：

~~~bash
.venv/bin/python main.py --config experiments/configs/agnews_gpt2_shared_smoke.json --validate-only
.venv/bin/python main.py --config experiments/configs/agnews_gpt2_shared_smoke.json
.venv/bin/python main.py --config experiments/configs/agnews_gpt2_centralized_smoke.json
~~~

~~~text
main.py                     统一 JSON 命令行入口
main_fed.py                 兼容入口
algorithm/                  划分、分组、选择、初始化、聚合与注册
  custom/                   算法与任务插件示例
tasks/                      任务批处理、模型类型、损失、统计与指标
models/                     PEFT 状态、本地训练和评估
training/                   公共训练步骤、标准中心式循环
simulation/                 联邦轮次、客户端持久状态、虚拟时间
execution/                  动态任务池、评估调度、本地与跨节点 worker
utils/                      配置、数据、产物、缓存和绘图
tools/                      集群启动、时间标定、对比矩阵
experiments/configs/        可追踪的配置模板
experiments/runs/           自动生成并由 Git 忽略的实验目录
tests/                      离线小模型测试
docs/                       模块、配置和扩展说明
~~~

详细说明见 [文档索引](docs/README.md)。每个 worker 需要容纳一个完整骨干；当前提供 LoRA 训练，未启用客户端内部 DDP/FSDP、量化、异步联邦轮次或编译执行。中心式使用 CPU 梯度协调与持久优化器，性能边界见 [中心式训练](docs/centralized.md)。
