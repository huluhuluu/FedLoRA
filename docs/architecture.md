# 架构与模块

main.py 读取配置、加载显式插件和任务、准备数据、创建执行器，随后按 experiment.mode 选择联邦 Engine 或 CentralizedTrainer。一次运行固定模型和任务。

## 任务、状态与算法

tasks/ 定义模型类型、数据校验、collate、compute_loss、statistics 和 finalize。训练和评估都调用任务的损失方法；分类按样本、语言建模按有效预测 token 汇总。自定义指标先合并原始统计，再计算最终值。

models/peft_adapter.py 加载骨干并导出、回写命名 LoRA/头部参数。AdapterState 描述 A/B 名称、维度、rank、alpha、缩放、方向与 global/private/frozen 角色。算法依赖这些元数据，不依赖模型名称或张量位置。

algorithm/ 提供划分、分组、选择、初始化、聚合和插件注册。algorithm/interfaces.py 定义客户端更新、工作量与聚合结果。算法不管理 GPU 进程或实验文件。

## 两种训练流程

simulation/engine.py 维护逻辑客户端、全局版本和同步轮次，依次选择、训练、聚合。客户端优化器和私有头随逻辑 ID 保存。simulation/timing.py 独立计算联邦虚拟时间。

training/centralized.py 维护一个全局模型状态、优化器、调度器与数据流。每个全局 batch 拆成固定 microbatch，worker 返回同一版本的梯度，协调器按损失权重合并后只更新一次。training/loop.py 复用前向、梯度、优化器和计算签名逻辑。详见 [中心式训练](centralized.md)。

## 执行与评估

execution/pool.py 管理待执行任务与每 worker 一个在途任务；local.py 使用进程队列，distributed.py 使用 Gloo 点对点请求和结果。每个设备的骨干持续驻留，下一任务开始前回写指定适配器。逻辑分组与物理 GPU 归属无关。

execution/evaluation.py 保存不可变的版本快照，并临时借用配置中的 worker。评估分片先合并统计，评估结束的 worker 可立即继续训练。没有全体训练 worker 必须共同参加的评估屏障。详见 [评估](evaluation.md)。

## 配置与产物

utils/config.py 合并 JSON、展开客户端和解析路径；utils/data.py 读取只读源并生成确定性子集；utils/experiment.py 统一指标、版本记录和原子检查点。tools/ 提供节点启动、标定及顺序对比矩阵。骨干、数据、缓存和实验输出各有明确归属，不复制完整模型作为逐轮产物。

资源创建者负责构造失败时回收已经创建的进程和通信组；close 只释放资源，不再执行评估。训练循环在正常结束时等待已提交评估，入口在资源关闭后记录最终状态。任务和初始化共用 algorithm/interfaces.py 中的种子上下文。
