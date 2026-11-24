# 评估频率、GPU 和版本

evaluation 与训练共享同一 worker 池。没有评估时所有设备均可训练；到达触发点后，指定 worker 在当前任务结束时开始评估，其余 worker 继续训练。某个评估分片完成且没有更多分片时，该 worker 立即回到训练队列。

~~~json
{
  "execution": {"backend": "local", "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]},
  "evaluation": {
    "mode": "async",
    "trigger": {"unit": "round", "every": 2},
    "devices": ["cuda:2", "cuda:3"],
    "batch_size": 2,
    "at_end": true,
    "pending_policy": "latest",
    "save_best": true,
    "metric": "loss",
    "direction": "min"
  }
}
~~~

联邦触发单位为 round，中心式为 step。也可直接写 every、unit；trigger 在加载时转换为这两个字段。旧版 output.eval_every 仅在输入时转换。

## 资源

本地使用 devices 选择 execution.devices 中的设备。跨节点建议使用 worker_ids，即全局 rank 编号，避免不同节点重复的 cuda:0 名称产生歧义；devices 与 worker_ids 二选一。未指定时，async 默认借最后一个 worker，sync 默认使用 worker 0。串行后端使用 sync 或 off。

mode=sync 等待本次评估后继续；async 将评估和下一批训练重叠；off 关闭评估并跳过评估数据与资源准备。即便只有一个 worker，也可使用 local+async，但计算只能交替进行。训练与评估固定共享同一资源池。

评估分片按数据索引划分，合并损失和指标的原始统计后统一 finalize，不能直接平均各卡的 accuracy、perplexity 或 F1。私有头按触发时选中客户端的头部快照逐个评估，再等权平均客户端指标。

## 快照与队列

每次评估复制 CPU 适配器与必要私有头，带 model_version 和触发 round/step；冻结骨干不复制。worker 切换任务时恢复正确状态，评估不会继续读取正在更新的模型。

最多保留一个执行中的快照和一个待执行快照。pending_policy=latest 用新请求替换待执行请求，并写 evaluation_replaced 事件；已经执行的评估不会被取消。block 保留每次请求，队列满时训练等待。at_end=true 保证最终版本完成评估后退出；关闭后仍会等待已经提交的评估结束。

## 结果

evaluations.csv 和 evaluations.jsonl 是异步评估结果的依据，包含模型版本、触发位置、worker 与完成时间。rounds.csv 中仅同步返回的指标一定有值，不能按评估完成时的训练轮次误认模型版本。

save_best=true 按 metric/direction 保存 best_adapter.pt，绑定实际评估快照。该文件用于模型参数与指标，不含可续跑的优化器/数据流；训练续跑使用 checkpoint_last.pt。metric 使用任务 finalize 返回的指标名，如 loss、accuracy、perplexity 或插件的 macro_f1。

output.trace_dispatch=true 时生成 dispatch.jsonl，记录任务派发/完成、worker、时间和版本，便于观察借卡与归还；默认关闭以控制产物体积。这些是控制端派发与接收时间，不代表 GPU kernel 的实际起止。

当指标无法表示为有限数值时，例如极大 loss 对应的 perplexity，JSON 记录为 null、CSV 留空；该指标不会用于更新最佳快照。
