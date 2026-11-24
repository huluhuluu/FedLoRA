# 标准中心式训练

experiment.mode=centralized 使用标准全局 minibatch 训练：同一批次的所有 microbatch 从相同模型版本计算梯度，按任务的样本数或有效 token 数加权合并，再执行一次全局优化器更新。这里没有客户端本地多步训练后的参数平均。

~~~json
{
  "experiment": {"mode": "centralized"},
  "model": {"adapter": {"rank": 4, "alpha": 8}},
  "training": {
    "batch_size": 8,
    "micro_batch_size": 1,
    "budget": {"unit": "steps", "value": 100},
    "optimizer": {"name": "adamw", "lr": 0.0002},
    "max_grad_norm": 1,
    "scheduler": {"name": "cosine", "warmup_steps": 10}
  },
  "timing": {"mode": "off"}
}
~~~

batch_size 是全局 batch，micro_batch_size 决定固定拆分；两者不随空闲 GPU 数变化。末尾不足 batch 的样本仍会处理。预算可按 steps 或 epochs 指定，每个 epoch 重新确定性打乱；数据游标跨 step 保持。

## 状态与执行

训练默认只更新 LoRA 和公共可训练头，骨干冻结；也可冻结头。初始化仍使用 initialization 插件。标准梯度更新没有联邦参数聚合，aggregation.method 须省略或为 null；clients、partition、grouping 和 selection 不参与该流程。

入口是 main.py，循环在 training/centralized.py，worker 梯度计算在 training/loop.py。协调器持有 FP32 训练参数与一个持续的 SGD/AdamW 优化器，worker 在 GPU 上计算梯度。FP16 使用统一损失缩放，合并后检查溢出；跳过更新时不推进模型版本和调度器。

调度器支持 constant、linear、cosine 和 warmup_steps。默认衰减范围是首次运行的总步数；续跑保留这一范围。需要提前规划更长训练时设置 training.scheduler.total_steps。

串行、本地多进程和跨节点使用同一计算契约。microbatch 的索引、种子和合并顺序不依赖可用 worker 数，因此评估借卡仅改变任务放置。浮点精度、硬件与算子仍可能造成数值差异。

## 评估与续跑

评估触发单位为 step。评估可占用部分 worker，其余 worker 处理下一全局 batch 的 microbatch；该 batch 的梯度全部完成后才能执行更新。全局优化器跨 step 持续，不因评估重建。

~~~bash
python main.py --config experiments/configs/agnews_gpt2_centralized_smoke.json
python main.py --config experiments/configs/agnews_gpt2_centralized_smoke.json \
  --resume experiments/runs/RUN/checkpoint_last.pt --set training.budget.value=8
~~~

检查点保存模型版本、全局参数、优化器、调度器、损失缩放和数据游标；续跑不能使用 best_adapter.pt。中心式记录真实耗时，不计算联邦终端虚拟时间。

## 性能边界

当前通过 CPU FP32 参数和 Gloo/进程队列传输梯度，优先服务 LoRA 算法对比与动态借卡。它不是 NCCL DDP，较多训练参数或跨节点低带宽会使协调器成为瓶颈。每个 worker 都持有完整骨干，未实现骨干分片或全参数训练。

协调器按固定 microbatch 顺序流式归约梯度，只缓冲有限数量的在途结果；梯度缓存随 worker 数量控制，不随整个全局 batch 无限制增长。同一参数快照在 worker 间按需下发，连续计算同一参考的梯度时复用已加载的模型参数。
