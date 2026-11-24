# 算法与聚合

algorithm/registry.py 按 kind/name 注册对象。内置算法如下：

| kind | method | 输入与输出 |
| --- | --- | --- |
| partition | iid、shard、dirichlet | 数据集、客户端 → 客户端索引 |
| initialization | kaiming、gaussian | PEFT 初始 A/B；公共初始化 |
| grouping | none、predefined、resource_profile、distribution | 客户端配置、标签分布 → 逻辑分组 |
| selection | uniform、group_balanced | 非空客户端、分组 → 客户端 ID |
| aggregation | fedavg、svd | 当前版本更新、旧全局适配器 → 新适配器 |

所有划分均无放回且保留尾部样本。样本少于客户端数时允许空客户端，选择阶段会过滤空客户端。Dirichlet 不通过反复重试强制最小样本数，因此较小 alpha 可能产生空客户端。

组间均衡选择每轮随机化组顺序；当配额小于组数时，也不会长期只选择排在前面的固定组。

## 初始化

kaiming 使用 PEFT 默认的 A 初始化与零 B；gaussian 使用高斯 A 与零 B。初始化是一个公共参考，不在每个客户端重新采样。每个客户端在开始训练前从当前全局状态构造自己的秩和缩放布局；更高秩补充的 A 行由客户端训练种子产生，新增 B 列为零。

插件可以提供 initialize(state, net, data, context)，在 worker 0 上执行一次并返回公共适配器。不能修改冻结骨干，否则同一个 A/B 不再代表相同的模型增量。PiSSA、OLoRA、EVA 等涉及骨干重参数化或数据标定的初始化没有作为内置别名启用，需要实现完整的公共参考语义。

## FedAvg

对 A、B 和共享头分别做加权平均。weighting=samples 的权重是客户端数据划分大小，uniform 为等权。A/B 平均与模型增量 B@A 的平均通常不同，这是 FedAvg 适配器算法的定义。

FedAvg 要求模块、键、秩、alpha 和实际缩放一致。不能直接把不同客户端的 A/B 补零后当作等布局 FedAvg。

## SVD

每个模块的有效增量为 scale × B@A。对客户端加权增量求和后做低秩近似。实现将 sqrt(weight×scale) 分配到 A 和 B，拼接因子，通过薄 QR 把 SVD 缩小到聚合秩空间，不生成完整稠密权重矩阵。服务器输出按自己的 alpha/rank 还原因子。

SVD 支持异构客户端 rank/alpha、server_rank 上限和 energy 能量阈值；energy 未设置时不自动改变服务器秩。全零 B 保留原有可训练 A。客户端从服务器高秩状态接收较低秩适配器时存在截断误差，这是低秩压缩的一部分；较高秩用零增量补齐。

私有参数和冻结参数保持旧全局模板，只有 global 参数参与聚合。共享头独立做普通加权平均，不参加 LoRA 分解。

## 分层聚合

~~~json
{
  "aggregation": {
    "mode": "hierarchical",
    "within_group": {"method": "svd", "params": {"weighting": "samples"}},
    "across_groups": {"method": "svd", "params": {"weighting": "samples"}}
  }
}
~~~

先按逻辑组聚合，再按组中实际参与者的样本总量聚合。组内或组间低秩截断会改变结果；它不等价于一次全局 SVD。分组与物理节点没有绑定关系，组内更新不会先在每个 GPU 上独立截断。

当前分布式后端收集 CPU 适配器状态，由 rank 0 聚合。它没有采用“先平均每张卡、再平均卡结果”的方式，因此不受每卡客户端数量不等造成的权重偏差影响。聚合 metrics 和实际工作时间写入 events.jsonl。

## 中心式对比

中心式复用相同初始化插件、任务、模型和评估。它采用标准梯度优化，不调用 FedAvg/SVD 参数聚合。需要比较初始化时，在中心式 case 中变化 initialization.method；需要比较联邦聚合时，在联邦 case 中变化 aggregation.method。对比入口见 [实验](experiments.md)。
