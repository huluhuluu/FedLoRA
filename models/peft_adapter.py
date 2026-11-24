"""PEFT LoRA backend for sequence classification and causal language modeling."""

from pathlib import Path
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import AutoTokenizer
from models.adapter_state import AdapterState, LoRABlockSpec
from algorithm.registry import build
from utils.helper import seed_training
from tasks import build_task, task_name


class PeftAdapter:
    def __init__(self, config, data_info, device, initializer=None, initialize=True):
        self.config, self.device = config, torch.device(device)
        self.precision = config["runtime"]["precision"]
        self.initializer = (
            (initializer or build("initialization", config["initialization"]))
            if initialize
            else None
        )
        if self.device.type == "cpu" and self.precision == "fp16":
            raise ValueError("fp16 training requires CUDA")
        seed_training(config["experiment"]["seed"])
        self.task = build_task(config, data_info)
        spec = config["model"]
        local = spec["source"] == "local"
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["path"], local_files_only=local, revision=spec.get("revision")
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs an existing pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            self.precision
        ]
        kwargs = {
            "local_files_only": local,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "revision": spec.get("revision"),
            "ignore_mismatched_sizes": spec.get("reset_head", False),
        }
        kwargs.update(self.task.model_kwargs())
        base = self.task.model_class.from_pretrained(spec["path"], **kwargs)
        self.head_modules = spec.get("head_modules") or self.task.head_modules(base)
        base.config.pad_token_id = self.tokenizer.pad_token_id
        base.config.use_cache = False
        self.base_reference = (
            str(Path(spec["path"]).resolve()) if local else spec["path"]
        )
        self.base_reference += (
            ":" + str(spec.get("revision", "local")) + ":" + task_name(config)
        )
        first = config["model"]["adapter"]
        initial = self._lora_config(first["rank"], first["alpha"])
        self.model = get_peft_model(base, initial).to(self.device)
        self.adapter_name = "default"
        self._apply_training_policy()
        self._refresh_layout()

    def _lora_config(self, rank, alpha, blocks=None):
        spec = self.config["model"]
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=spec["adapter"].get("dropout", 0.0),
            target_modules=spec["adapter"].get("target_modules"),
            modules_to_save=self.head_modules,
            bias="none",
            task_type=TaskType[self.task.peft_task_type],
            init_lora_weights=self.initializer.peft_value()
            if blocks is None and self.initializer is not None
            else False,
        )
        if blocks:
            peft_config.rank_pattern = {
                name.removeprefix("base_model.model."): block.rank
                for name, block in blocks.items()
            }
            peft_config.alpha_pattern = {
                name.removeprefix("base_model.model."): block.alpha
                for name, block in blocks.items()
            }
        return peft_config

    def _apply_training_policy(self):
        policy = self.config["model"]["head_policy"]
        if policy not in ("global_trainable", "private_trainable", "frozen"):
            raise ValueError("Unknown model.head_policy")
        # Replacement adapters and heads must preserve the initial optimization dtype.
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and self.adapter_name in module.lora_A:
                module.lora_A[self.adapter_name].float()
                module.lora_B[self.adapter_name].float()
            elif name.endswith(f".modules_to_save.{self.adapter_name}"):
                module.requires_grad_(policy != "frozen")
                if policy != "frozen":
                    module.float()

    def _refresh_layout(self):
        self._tensors = get_peft_model_state_dict(
            self.model, adapter_name=self.adapter_name
        )
        blocks = {}
        for name, module in self.model.named_modules():
            if not hasattr(module, "lora_A") or self.adapter_name not in module.lora_A:
                continue
            a_key, b_key = name + ".lora_A.weight", name + ".lora_B.weight"
            a, b = self._tensors[a_key], self._tensors[b_key]
            blocks[name] = LoRABlockSpec(
                a_key,
                b_key,
                a.shape[0],
                module.lora_alpha[self.adapter_name],
                module.scaling[self.adapter_name],
                a.shape[1],
                b.shape[0],
                type(module.get_base_layer()).__name__,
                "in_out" if getattr(module, "fan_in_fan_out", False) else "out_in",
            )
        if not blocks:
            raise ValueError("No LoRA blocks found; check target_modules")
        keys = {key for block in blocks.values() for key in (block.a_key, block.b_key)}
        role = {
            "global_trainable": "global",
            "private_trainable": "private",
            "frozen": "frozen",
        }[self.config["model"]["head_policy"]]
        self._template = AdapterState(
            {},
            blocks,
            {key: role for key in self._tensors.keys() - keys},
            self.base_reference,
        )
        self.layout = tuple(
            (name, block.rank, block.alpha) for name, block in blocks.items()
        )
        parameters = {
            p.data_ptr(): p for p in self.model.parameters() if p.requires_grad
        }
        self._parameters = {
            key: parameters[value.data_ptr()]
            for key, value in self._tensors.items()
            if value.data_ptr() in parameters
        }
        self.invalidate()

    def export(self):
        return self._template.with_tensors(
            {
                key: value.detach().to("cpu", copy=True)
                for key, value in self._tensors.items()
            }
        )

    def invalidate(self):
        self._prepared_state = None

    def prepare(self, state):
        if self._prepared_state is state:
            return
        layout = tuple(
            (name, block.rank, block.alpha) for name, block in state.blocks.items()
        )
        if layout != self.layout:
            block = next(iter(state.blocks.values()))
            old_name = self.adapter_name
            self.adapter_name = "client" if old_name == "default" else "default"
            # Keep one adapter active while PEFT creates replacement head wrappers.
            self.model.add_adapter(
                self.adapter_name,
                self._lora_config(block.rank, block.alpha, state.blocks),
            )
            self.model.set_adapter(self.adapter_name)
            self.model.delete_adapter(old_name)
            self._apply_training_policy()
            self._refresh_layout()
        if self._tensors.keys() != state.tensors.keys() or any(
            self._tensors[k].shape != v.shape for k, v in state.tensors.items()
        ):
            raise ValueError("Adapter keys/shapes do not match the model")
        # PEFT's canonical export tensors share storage with the live adapter and head.
        with torch.no_grad():
            for key, value in state.tensors.items():
                self._tensors[key].copy_(value)
        self._prepared_state = state

    def parameter_map(self):
        return self._parameters

    def autocast(self):
        from contextlib import nullcontext

        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(
            self.device.type,
            dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16,
        )
