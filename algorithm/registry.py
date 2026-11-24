"""Explicit registries; user modules are imported only when configured."""

import importlib

_REGISTRY = {}


def register(kind, name):
    def decorator(implementation):
        key = (kind, name)
        if key in _REGISTRY:
            raise ValueError(f"Duplicate algorithm: {kind}/{name}")
        _REGISTRY[key] = implementation
        return implementation

    return decorator


def load_plugins(modules=()):
    importlib.import_module("algorithm.builtin")
    for module in modules:
        importlib.import_module(module)


def implementation(kind, spec):
    name = spec["method"]
    try:
        return _REGISTRY[(kind, name)]
    except KeyError:
        choices = sorted(n for k, n in _REGISTRY if k == kind)
        raise ValueError(f"Unknown {kind} '{name}'; available: {choices}") from None


def validate_spec(kind, spec, **extra):
    import inspect

    inspect.signature(implementation(kind, spec)).bind(
        **spec.get("params", {}), **extra
    )


def build(kind, spec, **extra):
    return implementation(kind, spec)(**spec.get("params", {}), **extra)
