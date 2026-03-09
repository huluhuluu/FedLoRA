#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, LoraModel, get_peft_model
from models.Fed import SVD, ACA


# Default LoRA target modules
DEFAULT_TARGET_MODULES = ['q_proj', 'v_proj']

# Model-specific LoRA target modules (only for models different from default)
MODEL_TARGET_MODULES = {
    # BERT family
    'bert': ['query', 'value'],
    'roberta': ['query', 'value'],
    'distilbert': ['q_lin', 'v_lin'],
    # GPT-2 (single merged projection)
    'gpt2': ['c_attn'],
}


def get_target_modules(model_path):
    """
    Auto-detect target modules based on model type.
    
    Args:
        model_path: path to the pretrained model
    
    Returns:
        list of target module names for LoRA
    """
    model_path_lower = model_path.lower()
    for model_type, targets in MODEL_TARGET_MODULES.items():
        if model_type in model_path_lower:
            return targets
    return DEFAULT_TARGET_MODULES


class LoraNet():
    def __init__(self, num_labels, rank=16, lora_alpha=16, args=None, contrast_flag=False):
        self.args = args
        self.name_lora_para = []
        self.target_modules = get_target_modules(args.model)
        
        # Load tokenizer and model
        self._load_tokenizer()
        self._load_model(num_labels)
        
        # Apply LoRA or freeze parameters
        if not contrast_flag:
            self._apply_lora(rank, lora_alpha)
            self._update_lora_names()
            self.model.print_trainable_parameters()
        else:
            self._freeze_parameters()
        
        self.model.to(self.args.device)

    def _load_tokenizer(self):
        """Load tokenizer from local path."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=True)
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, num_labels):
        """Load pretrained model from local path."""
        config = AutoConfig.from_pretrained(self.args.model, num_labels=num_labels)
        # Set pad_token_id in config
        if config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model, config=config
        )

    def _apply_lora(self, rank, lora_alpha):
        """Apply LoRA to model with specified configuration."""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=0.05,
            bias="none",
            init_lora_weights=self.args.init if self.args.init in ['gaussian', 'olora', 'eva'] else True
        )
        self.model = get_peft_model(self.model, lora_config)

    def _freeze_parameters(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _update_lora_names(self):
        """Update the list of LoRA parameter names."""
        self.name_lora_para = []
        name_lora = []
        for name in self.model.state_dict():
            if 'lora' in name and 'weight' in name:
                name_lora.append(name)
        # Reshape: [[[q_lora_a, q_lora_b], [v_lora_a, v_lora_b]], ...]
        for ind in range(0, len(name_lora) // 4):
            self.name_lora_para.append([name_lora[4*ind:4*ind+2], name_lora[4*ind+2:4*ind+4]])

    def get_lora_parameters(self):
        """
        Get LoRA parameters from model.
        
        Returns:
            lora_parameters: [[[q_lora_a, q_lora_b], [v_lora_a, v_lora_b]], ...]
        """
        lora_parameters = []
        model_dict = self.model.state_dict()
        
        for layer in self.name_lora_para:
            q_lora = [model_dict[layer[0][0]].detach().clone(), 
                      model_dict[layer[0][1]].detach().clone()]
            v_lora = [model_dict[layer[1][0]].detach().clone(), 
                      model_dict[layer[1][1]].detach().clone()]
            lora_parameters.append([q_lora, v_lora])
        
        return lora_parameters

    def put_lora_parameters(self, lora_parameters):
        """
        Apply LoRA parameters to model.
        
        Args:
            lora_parameters: [[[q_lora_a, q_lora_b], [v_lora_a, v_lora_b]], ...]
        """
        ind = 0
        for name, module in self.model.named_modules():
            layer_ind = ind // 4
            qv_ind = (ind % 4) // 2
            ab_index = (ind % 4) % 2
            
            if layer_ind >= len(self.name_lora_para):
                break
                
            if name + '.default.weight' == self.name_lora_para[layer_ind][qv_ind][ab_index]:
                new_weight = torch.nn.Parameter(
                    lora_parameters[layer_ind][qv_ind][ab_index].detach().clone()
                )
                module['default'].weight = new_weight
                ind += 1
            
            if ind >= len(self.name_lora_para) * 4:
                break

    def set_layer_ranks(self, layer_ranks):
        """
        Configure LoRA ranks for different layer groups.
        
        Args:
            layer_ranks: list of (num_layers, rank) tuples
                         rank=0 means no LoRA for that group
        
        Raises:
            ValueError: if total layers don't match model layers
        """
        # Get encoder layers (works for BERT, Qwen2, LLaMA, etc.)
        if hasattr(self.model, 'bert'):
            encoder_layers = self.model.bert.encoder.layer
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            encoder_layers = self.model.model.layers
        else:
            raise ValueError("Cannot find encoder layers in model")
        
        # Validate layer count
        total_layers = sum(num for num, _ in layer_ranks)
        model_layers = len(encoder_layers)
        
        if total_layers != model_layers:
            raise ValueError(
                f"Layer count mismatch: specified {total_layers} layers, "
                f"but model has {model_layers} layers"
            )
        
        begin_layer = 0
        
        for num_layers, rank in layer_ranks:
            end_layer = begin_layer + num_layers
            
            if rank == 0:
                begin_layer = end_layer
                continue
            
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                target_modules=self.target_modules,
                lora_dropout=0.05,
                bias="none"
            )
            
            for idx, layer in enumerate(encoder_layers):
                if begin_layer <= idx < end_layer:
                    # Find attention module
                    if hasattr(layer, 'attention'):
                        layer.attention = LoraModel(layer.attention, lora_config, "default")
                    elif hasattr(layer, 'self_attn'):
                        layer.self_attn = LoraModel(layer.self_attn, lora_config, "default")
            
            begin_layer = end_layer

        self._update_lora_names()
