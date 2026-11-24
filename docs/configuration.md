# JSON 配置

完整默认值定义在 utils/config.py 的 DEFAULT。省略字段会合并默认值；字典递归合并，列表整体替换。路径相对于配置文件目录解析，输出配置保存为绝对路径。--set 支持点号字段，右侧先按 JSON 解析，普通字符串也可直接使用。

~~~bash
python main.py --config experiments/configs/agnews_gpt2_smoke.json \
  --set simulation.rounds=5 \
  --set 'execution.devices=["cuda:0","cuda:2"]' \
  --set aggregation.method=svd
~~~

## 顶层字段

| 字段 | 含义 |
| --- | --- |
| schema_version | 当前为 1 |
| experiment | name、seed；mode=federated/centralized |
| runtime | precision=fp32/bf16/fp16；cpu_threads 每进程线程数 |
| task | type=classification/causal_lm 或插件名称；model.task 是兼容别名 |
| model | source=local/hub、path、revision、head_policy、adapter，可选 head_modules/reset_head |
| data | source、path、loader、task（字段预设）、split、seq_len、子集上限、text_fields、label_field |
| training | 中心式全局 batch、microbatch、预算、优化器、调度器，见 [中心式训练](centralized.md) |
| clients | 联邦客户端数量、默认值、配置档、分配和覆盖，见下文 |
| partition | method=iid/shard/dirichlet，params 传给算法构造函数 |
| initialization | method=kaiming/gaussian 或插件；初始化作用于公共参考 |
| grouping | none/predefined/resource_profile/distribution；update_every=0 表示只在起始分组 |
| selection | uniform/group_balanced；params.clients_per_round 为选中数量上限 |
| aggregation | flat 或 hierarchical，见算法文档 |
| simulation | rounds 为总轮数；同步轮次；on_empty_round=skip/error |
| execution | backend、设备、任务分配和资源清单，见执行文档 |
| evaluation | mode、trigger/every/unit、devices/worker_ids、batch_size、at_end、有界队列与最佳参数，见 [评估](evaluation.md) |
| timing | off/analytic/profiled 或插件，见模拟文档 |
| plugins | 显式导入的 Python 模块列表 |
| output | root、checkpoint_every（默认 10，0 表示仅保存最终状态）、trace_dispatch |

默认客户端数为 2，默认使用 CPU 串行、fp32、全局可训练头。model.path 和 data.path 需要由配置提供。head_policy 可选 global_trainable、private_trainable、frozen；后两者的通信和评估语义见模型文档。

data.loader 默认 huggingface；DatasetDict 保存目录可设 saved_dataset，也会自动识别 dataset_dict.json。local 源只读本地目录；hub 源可按 ID 加载资源，缓存仍写仓库。分类标签列默认 label，可通过 label_field 指定原始列名；文本列可用 text_fields 指定一个或两个字段。内置字段映射包括 AG News、Banking77、SST-2、CoLA、MNLI、MRPC、QNLI、QQP、RTE、WNLI，前提是用户准备了对应 split 和标签。因果语言建模使用第一个 text_fields，默认 text。

max_train_samples、max_eval_samples 默认 null，即全量；指定时由种子确定无放回子集，不改变源文件。seq_len 是 tokenizer 最大长度，也是 padding 长度。

## 客户端配置

~~~json
{
  "clients": {
    "count": 4,
    "defaults": {
      "adapter": {"rank": 8, "alpha": 16},
      "train": {
        "batch_size": 2,
        "budget": {"unit": "steps", "value": 10},
        "optimizer": {"name": "adamw", "lr": 0.0002, "state_policy": "reset"}
      },
      "system": {
        "hardware_profile": "reference", "time_multiplier": 1,
        "upload_mbps": 100, "download_mbps": 100, "latency_ms": 0
      }
    },
    "profiles": {"small": {"adapter": {"rank": 4, "alpha": 8}}},
    "assignments": [{"profile": "small", "client_ids": [0, 1]}],
    "overrides": [{"client_ids": [3], "train": {"budget": {"unit": "steps", "value": 5}}}]
  }
}
~~~

合并顺序为 defaults → profile → overrides。客户端 ID 是从 0 开始的整数。assignments 不允许重复分配，同一客户端可以通过 overrides 继续覆盖。客户端配置还可保存在单独文件，通过 clients.file 引用，文件内容就是 clients 对象。

预算单位为整数 steps 或 epochs。steps 可跨本地数据多个 epoch；epochs 包含最后一个不足 batch 的批次。num_samples 聚合权重使用该客户端的划分大小，不使用重复处理的样本数。SGD 支持 momentum；AdamW 支持 weight_decay；train.max_grad_norm 可选。state_policy=retain 保留客户端优化器状态，reset 每次重建优化器。FP16 损失缩放始终随客户端延续，避免短任务反复从初始缩放开始。SGD/AdamW 的其他参数传给对应 PyTorch 优化器；momentum 仅用于 SGD。

system 描述模拟终端，与实际承载任务的 GPU 独立。time_multiplier 越大，模拟计算越慢；网络速率单位 Mbps，latency_ms 按下发和上传各计一次。

## 算法与执行配置

划分参数：shard.num_shards 默认 200；dirichlet.alpha 默认 1。分组参数：predefined.groups 为名称到客户端列表；distribution.num_groups 默认 2，iterations 默认 20。资源分组按 clients.profile 分组。

flat 聚合直接配置 method/params。hierarchical 使用 within_group 和 across_groups 两个完整算法描述；分组必须覆盖每个客户端一次。SVD 可设置 server_rank 和 energy，energy 表示保留的奇异值平方能量比例，server_rank 是上限。FedAvg 要求公共 A/B 布局和缩放相同。

execution.backend=serial/local/distributed。devices 是统一设备列表，serial 使用第一个设备，local 使用全部设备；也可通过 resources_file 与 local_node 引用资源清单。distributed 根据设备类型与 LOCAL_RANK 使用节点内设备。placement=balanced_count/balanced_work；每个任务使用一个设备。execution.timeout_seconds 默认为 null，只在显式设置时限制在途任务的等待时间，初始化和空闲等待不计入任务时间。execution.hardware_profile 可设置标定标签，默认取 GPU 型号或 cpu。

中心式配置使用 model.adapter.rank/alpha；联邦配置按 clients 中的 rank/alpha 构建客户端布局。中心式 aggregation.method 必须省略或为 null，并设置 timing.mode=off。标准中心式不会调用客户端划分、选择和聚合。

output 默认 experiments/runs。每次运行创建新目录，只保留一个最新检查点。分布式资源文件不是实验配置继承文件；它描述物理节点和设备。配置校验不能确认模型文件完整或显存足够，正式加载由模型后端处理。

旧版 model.task、execution.device、output.eval_every 在输入时转换为统一字段。旧版单值开关仅作输入兼容，不进入解析后的配置；新配置无需填写 compile、adapter.backend、scope、protocol、client_training、evaluation.placement/max_pending 或 timing.network/include_evaluation。

evaluation.mode=off 时无需提供评估 split，不创建评估任务，也不校验评估设备和最佳指标。中心式模式不校验未使用的客户端、分组和轮次配置。
