# 实验与检查点

配置模板位于 experiments/configs/，个人配置可放入 Git 忽略的 experiments/local/。每次运行在 output.root 创建 UTC 时间、实验名与随机后缀组成的目录。

~~~text
RUN/
  config.resolved.json       合并默认值、覆盖值与绝对路径
  environment.json           Python/依赖、Git HEAD 与 dirty、父检查点
  clients.json               联邦客户端配置，仅联邦
  partitions.npz             联邦数据索引，仅联邦
  adapter_layout.json        适配器布局与角色
  workers.json               worker 设备与评估 worker 编号
  rounds.csv                 round/step 的训练、时间与载荷指标
  evaluations.csv            已完成评估的指标、模型版本与触发位置
  evaluations.jsonl          评估时间、版本、worker 和完整指标
  events.jsonl               工作量、分组、聚合、队列替换等事件
  dispatch.jsonl             可选任务派发/完成轨迹
  checkpoint_last.pt         最近完整保存的可续跑训练状态
  best_adapter.pt            可选最佳评估快照，仅模型状态
  status.json                running/completed/failed 与错误
  timing_profile.json        显式标定后生成，仅联邦
~~~

只保存最新训练检查点，采用同目录临时文件原子替换，不保存逐轮完整骨干。output.checkpoint_every 按 round/step 控制间隔，最后总会保存。异步评估完成且执行资源关闭后再把 status 写为 completed。

联邦检查点包含下一轮、全局版本、A/B 和头、客户端优化器/AMP 状态、划分、分组、虚拟时钟及算法状态。中心式检查点包含下一步、全局参数、优化器、调度器、统一损失缩放和数据游标。两者不混用，不包含骨干、tokenizer 或原始数据；torch.load 使用 weights_only=True。

## 续跑

使用同一配置与 --resume，增加 simulation.rounds 或 training.budget.value。允许改变实验名、输出、评估策略、CPU 线程数和执行资源；优化器、任务、数据和训练语义须一致。中心式保留原有调度器范围，详见 [中心式训练](centralized.md)。

续跑创建新目录并记录父检查点，指标只包含本次运行。best_adapter.pt 对应最佳已评估版本，不能作为续跑检查点。选择、初始化、轮次和 microbatch 使用独立派生种子；更换精度、硬件或算子仍可能引入浮点差异。插件需保存必要状态。

## 对比矩阵

[compare.example.json](../experiments/configs/compare.example.json) 分别组织中心式初始化与联邦初始化/聚合对比。全局 grid 与每个 case 的 grid 合并，case 字段优先：

~~~json
{
  "base": "agnews_gpt2_shared_smoke.json",
  "grid": {"experiment.seed": [42, 43]},
  "cases": [
    {
      "name": "centralized",
      "overrides": {
        "experiment": {"mode": "centralized"},
        "timing": {"mode": "off"},
        "evaluation": {"unit": "step"}
      },
      "grid": {"initialization.method": ["kaiming", "gaussian"]}
    },
    {
      "name": "federated",
      "grid": {
        "initialization.method": ["kaiming", "gaussian"],
        "aggregation.method": ["fedavg", "svd"]
      }
    }
  ],
  "output_root": "../runs"
}
~~~

~~~bash
python -m tools.run_suite --suite experiments/configs/compare.example.json --prepare-only
python -m tools.run_suite --suite experiments/configs/compare.example.json
~~~

prepare-only 校验全部组合并生成配置，不启动训练。默认每个组合顺序运行，避免争抢 GPU；suite 目录保存 suite.json、configs/、runs/ 和 summary.csv。汇总取最后训练记录和最后完成评估，并保留评估模型版本。省略 cases 时仍支持旧版全局网格。

从联邦切换到中心式 case 时，工具清除继承的联邦聚合；显式为中心式指定参数聚合会报错。中心式转回联邦时，未显式指定聚合则使用默认 FedAvg。

serial/local 与单节点 distributed 可直接运行。多节点先 prepare-only，再使用生成的 configs 在各节点通过 tools.launch 启动；工具不自动远程调度整个矩阵。

## 指标与绘图

train_loss 按实际损失权重汇总，分类按样本、语言建模按有效预测 token。联邦聚合的 samples 权重取客户端划分大小。optimizer_steps 记录成功更新数，工作量 batches 包括 AMP 跳过更新时已经完成的计算。

异步评估以 evaluations.csv/JSONL 的 model_version 为准；rounds.csv 中相邻训练记录不一定有评估值。physical_round_seconds 包含该次循环的训练、分发与同步等待，不含骨干启动；异步评估的独立耗时另行记录，不能简单相加当作总墙钟时间。虚拟时间和通信张量体积见 [模拟时间](simulation.md)。

~~~bash
python -m utils.draw experiments/runs/RUN --metric eval_loss
python -m tools.profile experiments/runs/FED_RUN
~~~

绘图保存在实验目录；eval_* 自动读取 evaluations.csv，中心式横轴为 step，联邦为 round。.venv、.cache、实验输出、模型和个人配置由 Git 忽略；仓库追踪代码、测试、功能文档和配置模板。

output.checkpoint_every 默认 10；设为 0 时只保存最终状态。保存发生在协调器线程，需要按实验规模选择间隔，避免逐步落盘影响任务派发。

旧版检查点需要保留同目录的 config.resolved.json，以校验和迁移原有的配置身份；新检查点可按训练语义直接校验。续跑不重新执行初始化算法。
