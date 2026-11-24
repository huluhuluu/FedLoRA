# 任务、模型和 A/B 状态

## 任务

tasks/classification.py 使用 AutoModelForSequenceClassification。单文本和文本对字段由数据配置确定，输出 loss 和 accuracy；准确率范围为 0–1。有效标签通过数据集 ClassLabel 或训练标签映射确定，评估 split 不能带未知标签或 -1 占位标签。

tasks/causal_lm.py 使用 AutoModelForCausalLM，将输入拷贝为 labels，attention_mask=0 的位置设置为 -100。EOS 即使与 PAD 共用 ID，只要是有效输入就保留监督。指标按移位后的有效 token 数累计，输出 loss 与 perplexity。批次需要至少一个有效的下一 token 预测。

task.type 通过注册表选择任务，model.task 保留兼容；data.task 仅决定常见数据字段预设。标签列可用 data.label_field 指定。任务插件可复用已有模型加载逻辑。

任务对象负责数据校验、collate、compute_loss、statistics、finalize；训练器只消费统一的模型输入和输出。它不把所有任务强行解释为分类，也不在模型层拼接特定数据集的文本。

## 模型后端

模型适配器后端使用 PEFT LoRA。模型、tokenizer 从同一来源及 model.revision 加载，local 来源设置 local_files_only，并且权重采用 safetensors。训练使用右侧 padding。tokenizer 缺少 PAD 时使用已有 EOS，若两者都没有则需要用户补齐 tokenizer。

target_modules 可以省略，使用 PEFT 对模型架构的映射，也可显式指定。GPT-2 的 c_attn 是融合的 Q/K/V Conv1D，BERT 常用 query/value，其他模型通常使用 q_proj/v_proj；模块名需要以实际模型为准。backend 会读取 LoRA 模块的真实缩放与方向，不假设每四个张量为一层。

fp32 使用普通计算；bf16/fp16 使用对应骨干精度和 autocast，初始和重建的 LoRA A/B 及可训练头均固定使用 FP32 参数；fp16 CUDA 训练使用 GradScaler，中心式由协调器统一维护缩放。无需为每个客户端复制整套骨干。当前不启用量化、DoRA、embedding LoRA、torch.compile 或客户端内部模型分片。

## 参数状态

AdapterState 包含：

- tensors：规范化 PEFT 名称到独立 CPU 张量的映射。
- blocks：模块名称到 LoRABlockSpec，明确 A/B 键、rank、alpha、scale、输入/输出维度、模块类型和权重方向。
- extra_roles：非 A/B 张量的 global/private/frozen 角色。
- base_reference：骨干来源与任务参考，聚合必须一致。
- layout_version：状态布局版本。

export 使用 get_peft_model_state_dict，返回独立 CPU 快照；prepare 检查规范键与形状，并回写 PEFT 导出的实时张量映射。PEFT 导出键会移除适配器的内部名称，算法无需依赖 default/client 这些临时名称。布局改变时先建立替代适配器，再切换和释放旧适配器；联邦优化器在切换后创建并按客户端策略恢复。中心式 worker 不创建优化器，由 parameter_map 将规范键映射到实际可训练参数并返回梯度，全局优化器持续保存在协调器。

[PEFT 检查点格式](https://huggingface.co/docs/peft/v0.18.1/en/developer_guides/checkpoint) 解释了适配器键和冻结骨干的关系。本项目检查点还包含联邦状态，不是直接调用 PeftModel.from_pretrained 的目录。

## 头部角色

| head_policy | 本地训练 | 全局聚合 | 评估 |
| --- | --- | --- | --- |
| global_trainable | 更新头 | 加权平均头 | 全局模型 |
| private_trainable | 更新客户端自己的头 | 不聚合头 | 触发时选中客户端的参数快照各自评估，再等权平均指标 |
| frozen | 不更新头 | 保留初始头 | 全局模型 |

因果语言建模默认只训练 LoRA，不额外保存或训练 LM 头；head_policy 影响的是实际导出的额外头模块，可通过 head_modules 显式指定。

分类骨干不带分类头时，会按统一实验种子初始化新头。已有分类头类别数不匹配时，可显式设置 model.reset_head=true 重新初始化不匹配参数。默认自动保存 classifier、score、pre_classifier、classification_head 等现有顶层头模块，也可通过 model.head_modules 指定。私有头在客户端首次参与前使用公共初始头，随后按客户端 ID 保存。冻结头在适配器切换后仍保持冻结。

模型目录在实验和续跑期间应保持不变；base_reference 不对数 GB 权重逐字节求哈希。多节点必须准备同一骨干、tokenizer 和代码，检查点不包含骨干权重。

参数映射与布局元数据在创建或切换适配器时更新。相同、不可变的 CPU 快照可复用已加载参数；本地优化器开始训练前会标记模型状态需要重新恢复。不会在每个 microbatch 上遍历完整骨干、重复建立导出映射或再次迁移整个模型。
