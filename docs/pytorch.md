# PyTorch 执行约束

项目使用 PyTorch 2.x 的 autocast、GradScaler、spawn 多进程和 torch.distributed。依赖基线为 Torch 2.7.1 + CUDA 12.6、Transformers 4.57.3、PEFT 0.18.1；环境版本会写入每次实验的 environment.json。安装见 [install.md](../install.md)。

升级依赖时需要同时验证适配器导出/回写、异构秩切换、分类头冻结、断点续跑和初始化进程组后的加载行为。包的最低版本要求满足，并不保证所有可选分布式接口都兼容。无需引入 torchao、Ray 或额外协调服务。

## 客户端并行与模型并行

多个联邦客户端可以各自在一张 GPU 上训练，共享的是每轮公共适配器状态。它们有不同数据和优化器，因此不能把全部 worker 的梯度放进同一 DDP 通信组。

当前 distributed 后端使用 Gloo 点对点传递 CPU 适配器或梯度状态，GPU 只负责各自的本地训练。模型完整复制在每个 worker 上，显存预算至少包含一个骨干、训练适配器、客户端优化器与激活（中心式优化器在 CPU 协调器）。数据按 CPU mmap/批次读取，不把完整训练集复制到每张卡。

标准中心式在同一模型版本内合并 microbatch 梯度，优化器、调度器和缩放保存在协调器；它没有使用 NCCL DDP，详见 [中心式训练](centralized.md)。

后续若一个客户端需要多卡，应为其设备集合单独创建通信组，定义分片状态到 AdapterState 的聚合和回写边界，并隔离其他客户端。FSDP2/DTensor 的状态不能直接交给当前 CPU 字典聚合器。当前每个任务只使用一个设备。

## 精度与资源

bf16 适用于具备相应能力的设备；fp16 使用损失缩放，CPU fp16 训练被拒绝。聚合在 CPU float32 上计算，再恢复输出张量类型。减少精度不会改变 LoRA 的 scale 定义。

每进程 cpu_threads 默认 2，DataLoader 不额外创建常驻数据子进程。本地 worker 持续复用骨干，任务结束后只导出适配器，不调用整模型 deepcopy。缓存目录固定在仓库 .cache/，检查点仅保留最新一份。保留所有客户端优化器状态会随客户端数和秩增加 CPU 内存与检查点体积；不需要持续动量时使用 reset。

Gloo 默认的阻塞对象接收也会对空闲等待计时。执行协议使用独立的消息长度头与载荷，消息头采用长等待，在途任务超时由协调器管理；载荷仍受进程组传输超时约束。相关行为见 [PyTorch 2.7.1 Gloo 源码](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/csrc/distributed/c10d/ProcessGroupGloo.cpp)。
