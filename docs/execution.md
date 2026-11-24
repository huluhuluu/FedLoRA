# 单卡、多卡和多机

一个 worker 对应一个设备，驻留公共预训练骨干并读取同一数据集。数据通过 CPU Arrow/mmap 和小批次读取，不把完整数据集放进显存。协调器仅持有 CPU 训练状态。

## 动态任务分发

联邦先选择逻辑客户端，再放入待执行队列。balanced_count 按队列顺序派发给空闲 worker；balanced_work 按预算×batch_size×rank 的估计工作量优先派发较重任务。任务数不再预先固定到某张卡；worker 完成任务即可接收下一个。该启发式不探测显存容量。

中心式将一个全局 batch 拆成固定 microbatch，派发给空闲 worker 计算同一版本的梯度；全部梯度返回后统一更新。详见 [中心式训练](centralized.md)。

评估临时借用指定 worker，其余 worker 继续从训练队列取任务。评估结束立即归还，不重启模型进程，详见 [评估](evaluation.md)。逻辑分组与 GPU 归属无关，客户端私有头和优化器状态随逻辑 ID 移动。

## 本地执行

serial 使用 execution.devices 的第一个设备；local 使用 spawn 创建持久进程，devices 如 ["cuda:0","cuda:1"]。每个任务开始前完整回写适配器，不能沿用上一客户端的公共权重。进程队列传输序列化的 CPU 状态，避免大量命名张量各自占用共享内存文件描述符。

## 跨节点执行

distributed 使用 Gloo 点对点请求/结果。rank 0 是 CPU 协调器，自己的 GPU 由本地子进程承担；其他 rank 常驻训练服务。根节点逐任务派发，各 rank 独立返回，评估 worker 无需参加训练屏障。冻结骨干不逐轮传输。只有 rank 0 写实验产物。

联邦返回客户端适配器供统一聚合；中心式返回 microbatch 梯度供全局优化器更新。当前采用 CPU 对象通信，未优化为 NCCL Tensor 通信；客户端数、梯度大小和链路带宽可能限制吞吐。

等卡数集群可在每台机器执行 torchrun，例如每台两卡：

~~~bash
python -m torch.distributed.run --nnodes=2 --node-rank=0 \
  --nproc-per-node=2 --master-addr=192.168.1.10 --master-port=29517 \
  main.py --config experiments/configs/agnews_gpt2_shared_smoke.json \
  --set execution.backend=distributed \
  --set 'evaluation.devices=[]' --set 'evaluation.worker_ids=[2,3]'
~~~

另一节点将 node-rank 改为 1。worker_ids 是全局 rank；torchrun 时每个进程使用 LOCAL_RANK 对应的 CUDA 设备，devices 不决定外部 torchrun 启动的全局 rank 数量。CPU 分布式使用 execution.devices=["cpu"]。

## 不等卡数集群

[资源示例](../experiments/configs/cluster.example.json) 描述 node-a 两卡、node-b 一卡。在各节点分别执行：

~~~bash
# node-a
python -m tools.launch --config experiments/configs/agnews_gpt2_shared_smoke.json \
  --resources experiments/configs/cluster.example.json --node node-a \
  --set 'evaluation.devices=[]' --set 'evaluation.worker_ids=[2]'

# node-b
python -m tools.launch --config experiments/configs/agnews_gpt2_shared_smoke.json \
  --resources experiments/configs/cluster.example.json --node node-b \
  --set 'evaluation.devices=[]' --set 'evaluation.worker_ids=[2]'
~~~

launch 计算 WORLD_SIZE 与节点 rank 区间，用 CUDA_VISIBLE_DEVICES 映射设备，并仅启动当前节点。它不自动登录其他机器。支持不连续设备编号，CPU 节点可重复 cpu 条目。同一节点不能重复使用同一 CUDA 设备。

所有节点需要一致的代码、依赖、配置、骨干和数据，且协调端口与 Gloo TCP 互通。可在资源文件指定 interface 或设置 GLOO_SOCKET_IFNAME；不要在跨节点环境使用仅本机可达的 loopback。

## 边界

每个 worker 需要容纳完整骨干，单个联邦客户端仅占一个设备。当前不启用编译执行或任务内模型分片。没有自动丢弃失败客户端或显存降级；失败后使用最近完整检查点续跑。标准中心式使用共享梯度协调，不是客户端内部 DDP/FSDP。

设备编号相对于当前 CUDA_VISIBLE_DEVICES；未设置该环境变量时对应物理编号。节点启动和对比工具使用相同的映射规则。单个节点使用 CPU worker 或 CUDA worker，不混用两种设备类型。

协调器准备的 DataBundle 通过 Arrow/mmap 元数据交给本地 worker，避免每个进程重复加载、划分和扫描标签。模型骨干仍由各 worker 独立驻留。

execution.timeout_seconds 默认为 null；显式设置后按每个在途任务计时。worker 进程退出会直接报告错误，不等待固定的十分钟。空闲消息头使用独立的长等待，不按任务超时判死；连接错误会终止当前运行。失败时会回收本节点拥有的子进程，不自动重试任务。

串行模式同步执行当前计算，超时检查不会中途抢占该计算；需要限制等待时间时优先使用 local 或 distributed。
