# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from types import MappingProxyType

import yaml
import os


def deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def update(cfg):
    defaults = dict(DEFAULTS)
    return deep_update(defaults, cfg)


def load(path=None):
    if not path:
        defaults_path = os.path.join(os.path.dirname(__file__), "defaults.yaml")
        with open(defaults_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return update(cfg)


def populate(conf, name):
    section = conf[name]
    if "name" not in conf[name]:
        raise ValueError(f"missing 'name' field for section '{name}'")
    sel = conf[name]["name"]
    res = {"name": sel}
    if sel in section:
        res = res | section[sel]
    if "common" in section:
        res = res | section["common"]
    return res


DEFAULTS = MappingProxyType(load())
