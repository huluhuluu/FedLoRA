"""Task plugins share one training and evaluation protocol."""

import importlib

_TASKS = {}


def register_task(name):
    def decorate(cls):
        if name in _TASKS:
            raise ValueError(f"Duplicate task: {name}")
        _TASKS[name] = cls
        return cls

    return decorate


def task_class(name):
    importlib.import_module("tasks.classification")
    importlib.import_module("tasks.causal_lm")
    if name not in _TASKS:
        raise ValueError(f"Unknown task '{name}'; available: {sorted(_TASKS)}")
    return _TASKS[name]


def task_name(config):
    return config["task"]["type"]


def build_task(config, info):
    return task_class(task_name(config))(config["data"], info)


def merge_statistics(values):
    total = {}
    for value in values:
        for key, item in value.items():
            total[key] = total.get(key, 0) + item
    return total
