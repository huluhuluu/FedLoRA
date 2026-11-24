"""Architecture adapters use actual named projections, including grouped-query attention."""

import pytest
from test_centralized import central_config, run


@pytest.mark.parametrize("task", ["classification", "causal_lm"])
def test_qwen3_task_mapping(config, assets, tmp_path, task):
    from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer

    path = tmp_path / "qwen3"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            pad_token_id=0,
            eos_token_id=2,
            bos_token_id=2,
        )
    ).save_pretrained(path)
    AutoTokenizer.from_pretrained(assets[0]).save_pretrained(path)
    cfg = central_config(
        config,
        task={"type": task},
        model={"path": str(path), "adapter": {"target_modules": ["q_proj", "v_proj"]}},
    )
    trainer = run(cfg)
    assert len(trainer.state.blocks) == 2
    assert {block.output_dim for block in trainer.state.blocks.values()} == {8, 16}


def test_suite_comparison_cases(config):
    from tools.run_suite import variants
    from utils.config import validate

    validate(config)
    spec = {
        "grid": {"experiment.seed": [1, 2]},
        "cases": [
            {
                "name": "central",
                "overrides": {
                    "experiment": {"mode": "centralized"},
                    "timing": {"mode": "off"},
                },
                "grid": {"initialization.method": ["kaiming", "gaussian"]},
            },
            {"name": "federated", "grid": {"aggregation.method": ["fedavg", "svd"]}},
        ],
    }
    planned = list(variants(config, spec))
    assert len(planned) == 8
    assert {cfg["evaluation"]["unit"] for _, _, cfg in planned} == {"round", "step"}


def test_custom_loss_is_used_for_training_and_evaluation(config):
    from algorithm.registry import load_plugins
    from utils.config import validate
    from utils.data import load_data
    from models.peft_adapter import PeftAdapter
    from models.test import evaluate
    from training.loop import forward_batch

    validate(config)
    load_plugins(config["plugins"])
    bundle = load_data(config)
    net = PeftAdapter(config, bundle.info, "cpu")
    net.model.eval()
    baseline = evaluate(net, bundle, net.export())["loss"]
    net.task.compute_loss = lambda output, batch: output.loss + 0.25
    measured = evaluate(net, bundle, net.export())["loss"]
    assert measured == pytest.approx(baseline + 0.25, abs=1e-6)
    _, output, loss = forward_batch(net, bundle, [0, 1])
    assert output.loss is loss
