"""Offline fixtures; no test downloads model weights or data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.helper import configure_runtime

configure_runtime(2)
import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def assets(tmp_path_factory):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("assets")
    model_path, data_path = root / "model", root / "data"
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        "news": 3,
        "world": 4,
        "sports": 5,
        "business": 6,
        "science": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    fast.model_input_names = ["input_ids", "attention_mask"]
    fast.padding_side = "left"
    fast.save_pretrained(model_path)
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_layer=1,
            n_head=2,
            n_embd=16,
            n_positions=32,
            pad_token_id=0,
            eos_token_id=2,
            bos_token_id=2,
        )
    )
    model.save_pretrained(model_path)
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["world", "sports", "business", "science"]),
        }
    )
    train = Dataset.from_dict(
        {
            "text": ["news world", "news sports", "news business", "news science"] * 4,
            "label": [0, 1, 2, 3] * 4,
        },
        features=features,
    )
    DatasetDict(train=train, test=train.select(range(4))).save_to_disk(data_path)
    return model_path, data_path


@pytest.fixture
def config(assets, tmp_path):
    from utils.config import DEFAULT, merge

    model, data = assets
    return merge(
        DEFAULT,
        {
            "experiment": {"name": "test", "seed": 13},
            "model": {"path": str(model), "adapter": {"target_modules": ["c_attn"]}},
            "data": {"path": str(data), "seq_len": 8},
            "clients": {
                "defaults": {
                    "adapter": {"rank": 2, "alpha": 4},
                    "train": {
                        "batch_size": 2,
                        "budget": {"unit": "steps", "value": 1},
                        "optimizer": {"name": "sgd", "lr": 0.03, "momentum": 0.5},
                    },
                }
            },
            "evaluation": {"batch_size": 2},
            "timing": {"mode": "analytic"},
            "output": {"root": str(tmp_path / "runs")},
        },
    )
