# 自定义扩展

最小可运行示例位于 algorithm/custom/example.py。将模块加入 plugins，并在对应算法段选择注册名称：

~~~json
{
  "plugins": ["algorithm.custom.example"],
  "selection": {"method": "priority", "params": {"clients_per_round": 2}},
  "clients": {"overrides": [{"client_ids": [3], "priority": 10}]}
}
~~~

插件也可放在自己的可导入 Python 包中。注册名称不能覆盖已有名称；构造参数来自 params，不需要修改 main.py 或协调器。

## 接口

| kind | 方法 |
| --- | --- |
| partition | partition(dataset, clients, context) → dict[int, list[int]] |
| grouping | group(clients, observations, context) → dict[str, list[int]] |
| selection | select(clients, groups, context) → list[int] |
| initialization | peft_value() → PEFT 初始化值；可选 initialize(state, net, data, context) → AdapterState |
| aggregation | aggregate(updates, previous, context) → AggregationResult |
| timing | seconds(workload, client) → 计算秒数；构造参数 timing_config、model |

context 提供 round_id、global_version、seed 和独立 NumPy rng。使用 context.rng 或 context.seed，避免依赖进程号、Python hash 或调用顺序的全局随机状态。

划分必须完整、唯一地覆盖训练索引。分组必须完整、唯一地覆盖客户端。selection 收到的 clients 已过滤无数据客户端。observations 含每个客户端的 num_samples 和 label_distribution；资源配置在 clients[cid].system 中。

聚合示例：

~~~python
from algorithm.registry import register
from algorithm.Fed import FedAvg

@register("aggregation", "my_average")
class MyAverage(FedAvg):
    def aggregate(self, updates, previous, context):
        result = super().aggregate(updates, previous, context)
        result.metrics["participants"] = len(updates)
        return result
~~~

ClientUpdate 包含 client_id、round_id、global_version、AdapterState、num_samples、metrics、Workload、optimizer_state、scaler_state 和 worker_id。模型张量按规范名称访问。不要将 private/frozen 参数纳入全局聚合，也不要原地改写 previous 或输入更新。新骨干布局需要先检查模块、维度和公共参考。

## 自定义初始化

继承 Kaiming 或 Gaussian 可复用 PEFT 初始策略，再实现 initialize。新实验仅在 worker 0 上执行一次，返回值作为公共初始状态下发；续跑直接恢复检查点，不重复初始化。data 是只读 DataBundle；net 提供 model、tokenizer、task、export 和 prepare，可用于有数据的标定。示例中的 scaled_kaiming 只缩放初始 A，保留零 B。

如需 PiSSA 等会改变骨干的算法，需要同时设计公共骨干参考和各 worker 的同步，不能只返回 A/B 并保持其他 worker 的旧骨干。本接口约定初始化不修改冻结权重。

## 有状态算法

需要续跑时同时提供 state_dict() 和 load_state_dict(state)。状态必须由普通字典、列表、标量和 CPU Tensor 组成，兼容 torch.load(weights_only=True)。协调器保存和恢复这些状态。无状态算法无需增加方法。

## 增加任务

将任务注册在显式插件内，无需修改入口或模型后端。可运行示例是 [task_example.py](../algorithm/custom/task_example.py)，它复用分类任务并通过合并混淆矩阵计算 macro F1。

~~~json
{
  "plugins": ["algorithm.custom.task_example"],
  "task": {"type": "classification_f1"},
  "data": {"text_fields": ["text"], "label_field": "label"},
  "evaluation": {"metric": "macro_f1", "direction": "max", "save_best": true}
}
~~~

~~~python
from tasks import register_task
from tasks.classification import Task as ClassificationTask

@register_task("my_classification")
class MyTask(ClassificationTask):
    metric_names = ("loss", "accuracy")
~~~

新任务继承 tasks/base.py 的 BaseTask，或复用现有任务。接口如下：

| 成员 | 用途 |
| --- | --- |
| model_class、peft_task_type | Transformers 模型类与 PEFT 类型 |
| label_kind、input_fields() | 标签语义与需要的数据文本列 |
| validate_data(bundle) | 加载 GPU 模型前的数据校验 |
| model_kwargs() | 模型加载所需的额外参数 |
| head_modules(model) | 默认需要导出的任务头 |
| collate(tokenizer, rows) | 输入、标签和 mask |
| compute_loss(output, batch) | 从模型输出和 batch 计算标量损失 |
| statistics(output, batch)、finalize(totals) | 可加和原始统计与最终指标 |
| metric_names | 实验 CSV 中记录的指标列 |

statistics 须提供 loss_sum、weight，并可返回额外数值计数；分片合并后 finalize 至少返回 loss。自定义非线性指标应先累计充分统计，不能先在每个分片计算 F1 再平均。compute_loss 同时用于联邦、中心式和评估；训练权重必须与 statistics 的 weight 一致。

## 增加模型支持

已有 Transformers 架构通常只需调整 model.path、task.type、target_modules；无需按 BERT、Qwen 名称改训练循环。分类与 causal_lm 是两个独立任务，生成评测或其他监督格式需要实现相应任务插件。

新增适配器后端需在 execution/worker.py 接入，并实现模型/任务、export、prepare、invalidate、parameter_map 和 autocast 契约。parameter_map 将规范导出键映射到实际可训练参数，供中心式梯度读取。所有额外状态都应明确 global/private/frozen 角色。

扩展文件只包含算法或任务逻辑；进程管理、实验目录、随机种子和资源缓存继续复用框架。

任务批次、算法观察值和 AdapterState 输入视为只读。需要更新状态时返回新的状态对象，避免原地修改公共快照；执行器会缓存这些不可变参考。时间模型也可通过 state_dict/load_state_dict 保存自己的状态，与虚拟时钟一起续跑。
