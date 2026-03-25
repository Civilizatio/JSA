# src/utils/instantiate_utils.py
"""Small instantiate helper with an optional Hydra dependency.

The original project used ``hydra.utils.instantiate`` for sampler creation inside the
Lightning modules. Some lightweight environments used for code review may not ship Hydra even
though the training environment does. This helper preserves the original behavior when Hydra is
available and provides a minimal fallback that understands ``_target_`` as well as
``class_path``/``init_args`` style configs.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from typing import Any

try:  # pragma: no cover - trivial import branch
    from hydra.utils import instantiate as hydra_instantiate
except Exception:  # pragma: no cover - fallback path is tested instead
    hydra_instantiate = None


def _locate(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _to_plain_object(config: Any):
    """Recursively converts a config object to plain Python primitives (dicts, lists, tuples, and scalars).
    This is used to convert Hydra's DictConfig and ListConfig objects into standard Python types for the fallback instantiate implementation.
    The function handles:
    - Mappings (converted to dicts)
    - Sequences (converted to lists or tuples)
    - Objects with an `as_dict()` method (converted by calling that method)
    - Objects with an `items()` method (converted to dicts)
    - Objects with a `__dict__` attribute (converted to dicts of their attributes)
    - Other objects are returned as-is (e.g., scalars, strings).
    """
    if config is None:
        return None
    if isinstance(config, Mapping):
        return {k: _to_plain_object(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return type(config)(_to_plain_object(v) for v in config)
    if hasattr(config, "as_dict") and callable(config.as_dict):
        return _to_plain_object(config.as_dict())
    if hasattr(config, "items") and callable(config.items):
        try:
            return {k: _to_plain_object(v) for k, v in config.items()}
        except Exception:
            pass
    if hasattr(config, "__dict__") and not callable(config):
        data = {
            key: _to_plain_object(value)
            for key, value in vars(config).items()
            if not key.startswith("__")
        }
        if data:
            return data
    return config


def instantiate(config: Any, **kwargs):
    if hydra_instantiate is not None:
        return hydra_instantiate(config, **kwargs)

    cfg = _to_plain_object(config)
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        return cfg

    cfg = dict(cfg)
    if "_target_" in cfg:
        target = cfg.pop("_target_")
        params = {**cfg, **kwargs}
        return _locate(target)(**params)

    if "class_path" in cfg:
        target = cfg.pop("class_path")
        init_args = cfg.pop("init_args", {}) or {}
        params = _to_plain_object(init_args)
        if not isinstance(params, dict):
            raise TypeError("init_args must be a mapping when using class_path configs.")
        params = {**params, **cfg, **kwargs}
        return _locate(target)(**params)

    if kwargs:
        return {**cfg, **kwargs}
    return cfg


__all__ = ["instantiate"]

if __name__ == "__main__":
    
    # test for _to_plain_object
    class TestConfig:
        def __init__(self, a, b):
            self.a = a
            self.b = b
    cfg = TestConfig(a=1, b=[2, 3])
    assert _to_plain_object(cfg) == {"a": 1, "b": [2, 3]}
    assert _to_plain_object({"x": 1, "y": 2}) == {"x": 1, "y": 2}
    assert _to_plain_object([1, 2, 3]) == [1, 2, 3]
    assert _to_plain_object((1, 2)) == (1, 2)
    assert _to_plain_object(None) is None
    assert _to_plain_object(42) == 42
    assert _to_plain_object("hello") == "hello"
