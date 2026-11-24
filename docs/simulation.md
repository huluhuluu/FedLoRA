# 模拟时间与标定

真实执行耗时和模拟终端耗时分别记录。减少实际 GPU 数量会增加客户端排队，但不会自动放大虚拟同步轮次时间。

timing.mode：

- off：关闭虚拟时间，相关字段为 0。
- analytic：使用 assumed_step_seconds × 实际计算批次数 × time_multiplier。
- profiled：按照模型路径、任务、precision、rank、batch_size、seq_len、hardware_profile 和 compute_signature 精确匹配标定项，再乘计算批次数和 time_multiplier。
- 插件名称：通过 timing 注册自定义计算时间模型。

每个客户端时间为：

~~~text
compute
+ download_bytes × 8 / (download_mbps × 1e6)
+ upload_bytes × 8 / (upload_mbps × 1e6)
+ 2 × latency_ms / 1000
~~~

上传/下发只统计共享参数，私有头保存在模拟器内，不作为全局网络载荷；模拟器内部的进程传输开销属于真实执行时间。这里的网络模型是 independent_links，没有模拟链路竞争、拥塞和共享总带宽。

扁平同步轮次取客户端最大完成时间，再加 assumed_aggregation_seconds。分层聚合先取各组客户端最大时间并加组内聚合，再取组间最大值并加跨组聚合。聚合时间目前是显式假设值。include_evaluation 当前必须为 false，评估成本没有被伪装成已经标定的虚拟成本。

## 从真实训练标定

使用目标模型、精度、batch、序列长度和秩进行有界训练，然后执行：

~~~bash
python -m tools.profile experiments/runs/RUN
~~~

工具从 events.jsonl 中的实际训练时间/计算批次数取中位数，写入该实验目录的 timing_profile.json。实测 hardware_profile 使用真实 GPU 型号或 execution.hardware_profile 标签，不能把任意逻辑客户端的慢硬件标签当作实测机器。

profiled 模式需要设置 profile_file 和客户端的 system.hardware_profile：

~~~json
{
  "timing": {
    "mode": "profiled",
    "profile_file": "../runs/RUN/timing_profile.json",
    "assumed_aggregation_seconds": 0,
    "network": "independent_links",
    "include_evaluation": false
  },
  "clients": {
    "defaults": {"system": {"hardware_profile": "NVIDIA RTX A6000"}}
  }
}
~~~

标签以实际 profile 条目为准。条目缺失或重复会报错，避免悄悄使用其他模型/秩的速度。标定记录格式为 schema_version、source 和 entries；每个 entry 包含匹配字段、step_seconds、observations、compute_signature；签名覆盖训练参数布局、精度、优化器、裁剪与 dropout 等配置，避免误用不同工作负载的标定。

AMP 溢出跳过更新仍消耗计算，因此按批次数计时，optimizer_steps 单独记录成功更新数。中心式不调用联邦虚拟时钟，tools.profile 仅接受联邦实验。

训练计时包括分词、batch 搬运、前后向和优化器，不包括骨干加载。应使用足够样本和训练步数、稳定的运行负载来标定；仅一步的 smoke 不能代表稳定吞吐。不同 CPU、算子实现、模型注意力方式和并发负载会影响标定有效性。analytic 示例中的数值是用户假设，不是硬件测量结果。
