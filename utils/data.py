"""Read-only asset loading with cache files confined to the project."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datasets import ClassLabel, load_dataset, load_from_disk
from utils.helper import PROJECT_ROOT, stable_seed


@dataclass
class DataBundle:
    train: object
    evaluation: object
    info: dict

    def rows(self, indices, training=True):
        dataset = self.train if training else self.evaluation
        return [dataset[int(index)] for index in indices]


def load_data(config):
    spec = config["data"]
    cache = str(PROJECT_ROOT / ".cache/huggingface/datasets")
    path = Path(spec["path"])
    if spec["loader"] == "saved_dataset" or (path / "dataset_dict.json").is_file():
        raw = load_from_disk(str(path))
    else:
        raw = load_dataset(spec["path"], cache_dir=cache)
    train = raw[spec["train_split"]]
    evaluation = (
        raw[spec["eval_split"]] if config["evaluation"]["mode"] != "off" else None
    )
    label = spec.get("label_field", "label")
    if label != "label":
        train = train.rename_column(label, "label")
        if evaluation is not None:
            evaluation = evaluation.rename_column(label, "label")
    seed = config["experiment"]["seed"]
    for split, limit in (
        ("train", spec.get("max_train_samples")),
        ("evaluation", spec.get("max_eval_samples")),
    ):
        dataset = train if split == "train" else evaluation
        if dataset is not None and limit is not None and limit < len(dataset):
            if limit < 1:
                raise ValueError("Data sample limits must be positive")
            indices = np.random.default_rng(stable_seed(seed, "subset", split)).choice(
                len(dataset), limit, replace=False
            )
            dataset = dataset.select(sorted(indices.tolist()), keep_in_memory=True)
        if split == "train":
            train = dataset
        else:
            evaluation = dataset
    if not len(train) or evaluation is not None and not len(evaluation):
        raise ValueError("Training and evaluation datasets must be nonempty")
    if (
        config["experiment"].get("mode", "federated") == "federated"
        and config["partition"]["method"] in ("shard", "dirichlet")
        and "label" not in train.column_names
    ):
        raise ValueError(
            "Label-based partitioning needs a label column; use iid for unlabeled language data"
        )
    info = {
        "train_size": len(train),
        "eval_size": len(evaluation) if evaluation is not None else 0,
    }
    from tasks import task_class, task_name

    if task_class(task_name(config)).label_kind == "categorical":
        feature = train.features["label"]
        labels = (
            list(range(feature.num_classes))
            if isinstance(feature, ClassLabel)
            else sorted(set(train["label"]))
        )
        info.update(
            num_labels=len(labels),
            label_mapping={str(v): i for i, v in enumerate(labels)},
            label_names=feature.names
            if isinstance(feature, ClassLabel)
            else [str(v) for v in labels],
        )
        valid = set(info["label_mapping"])
        if evaluation is not None and any(
            str(value) not in valid for value in evaluation["label"]
        ):
            raise ValueError(
                "Evaluation contains unknown or unlabeled targets; select a labeled split"
            )
    result = DataBundle(train, evaluation, info)
    from tasks import build_task

    build_task(config, info).validate_data(result)
    return result
