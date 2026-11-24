# 安装与启动

需要 Linux、Python 3.11/3.12。依赖基线使用 PyTorch 2.7.1、Transformers 4.57.3、PEFT 0.18.1。CPU 可运行离线测试；GPT-2 XL 示例需要能容纳骨干和激活的 GPU。

## 独立环境

在仓库根目录执行。临时文件、下载缓存和运行结果保存在仓库内。

~~~bash
mkdir -p .cache/tmp
export TMPDIR="$PWD/.cache/tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
python -m pip check
~~~

没有 CUDA 的环境，将 Torch 安装源改为 https://download.pytorch.org/whl/cpu。Torch 平台安装命令见 [官方历史版本](https://pytorch.org/get-started/previous-versions/)。无需安装 torchvision、torchaudio 或 torchao。本项目不依赖系统 CUDA 编译工具链。

## 准备只读资源

模型目录需要配置、tokenizer 和完整的 safetensors 权重；支持 safetensors 索引分片。默认不加载 pickle 格式的模型文件。数据支持本地 Hugging Face 数据目录或 DatasetDict 的 save_to_disk 目录。分类数据须包含标签明确的训练和评估 split。

示例引用：

~~~text
/data/HF_MODELS/gpt2-xl
/data/HF_MODELS/data/ag_news
~~~

模型其他格式仍在下载时，只要本次使用的 safetensors、配置和 tokenizer 完整即可。不要把实验输出放到资源目录。程序将 Hugging Face、Torch、Triton、Matplotlib、临时文件缓存定向到仓库的 .cache/，并拒绝输出到 /data/HF_MODELS。

## 小规模启动

先检查配置，再进行两个客户端、各一步训练的双卡运行：

~~~bash
python main.py --config experiments/configs/agnews_gpt2_smoke.json --validate-only
python main.py --config experiments/configs/agnews_gpt2_smoke.json
~~~

单卡：

~~~bash
python main.py --config experiments/configs/agnews_gpt2_smoke.json \
  --set execution.backend=serial --set 'execution.devices=["cuda:0"]'
~~~

四卡联邦与标准中心式入口：

~~~bash
python main.py --config experiments/configs/agnews_gpt2_shared_smoke.json
python main.py --config experiments/configs/agnews_gpt2_centralized_smoke.json
~~~

这两个配置在评估时临时借用 cuda:2/3，其余 GPU 继续训练，评估完成后归还。单卡启动这两个配置时同时设置 evaluation.mode=sync、evaluation.devices=[] 和 execution.backend=serial、execution.devices=["cuda:0"]。

启动后会打印 EXPERIMENT_DIR。结果位于 experiments/runs/ 下，包括配置、rounds.csv、evaluations.csv、events.jsonl 和 checkpoint_last.pt。Smoke 配置限制训练/评估子集，指标用于检查流程。完整数据实验使用 agnews_gpt2.json，先根据显存调整 batch_size、seq_len、客户端秩和设备数量。

因果语言建模：

~~~bash
python main.py --config experiments/configs/agnews_gpt2_smoke.json \
  --set task.type=causal_lm
~~~

断点续跑使用原配置，只增加总轮数；会生成新的实验目录：

~~~bash
python main.py --config experiments/configs/agnews_gpt2_smoke.json \
  --resume experiments/runs/RUN/checkpoint_last.pt --set simulation.rounds=3
~~~

## 多卡和多机

单机原生分布式启动示例显式使用回环地址，避免依赖容器 hostname 的 DNS：

~~~bash
python -m torch.distributed.run --nnodes=1 --node-rank=0 \
  --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29519 \
  main.py --config experiments/configs/agnews_gpt2_smoke.json \
  --set execution.backend=distributed --set 'execution.devices=["cuda:0","cuda:1","cuda:2","cuda:3"]'
~~~

不同节点卡数不同时，在各节点分别执行 tools.launch；参见 [执行资源](docs/execution.md)。

## 测试与结果查看

测试构造本地小模型和数据，无需联网下载权重。临时产物收纳在 .cache/tests。

~~~bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest -q --basetemp=.cache/tests
python -m utils.draw experiments/runs/RUN --metric eval_loss
python -m tools.profile experiments/runs/RUN
python -m tools.run_suite --suite experiments/configs/compare.example.json --prepare-only
python -m tools.run_suite --suite experiments/configs/compare.example.json
~~~

训练失败时查看对应目录的 status.json 和终端异常。checkpoint_last.pt 保存最近完整轮次/步骤的续跑状态；best_adapter.pt 仅保存最佳评估模型。中心式步骤预算使用 training.budget.value；更多说明见 [中心式训练](docs/centralized.md)、[评估](docs/evaluation.md) 和 [对比矩阵](docs/experiments.md)。资源不足时降低 batch_size、seq_len、rank 或 worker 数量；每个 worker 都会驻留一个完整骨干。
