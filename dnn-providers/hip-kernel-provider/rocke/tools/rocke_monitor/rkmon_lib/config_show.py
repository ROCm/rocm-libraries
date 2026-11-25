# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk config show [--source]`: the running configuration as runtime_config.load resolves it, one value a
line, `NAME = VALUE` with the value as JSON; with --source, the layer each value came from before it
(runtime_config.SOURCES: deploy, repository, package, state, derived, and config.json, one more source). It only
reads: it prints on the terminal and writes no file.

Each value is named as the file that holds it names it:
- the deploy layer's values by the deploy settings file's keys: every key of the file at its effective value, as the
  delivered layer holds them, a map one value (settings_file.map_keys: its entries are no keys of their own), each
  from the source runtime_config.deploy_source gives it: deploy when your file sets it, else package. While no layer
  is delivered, a deploy key by config.json's name;
- a state value by its key in the internal state file (state_store.file_key);
- a package value by its name in config.defaults.json, which the repository defaults share;
- a derived value, and a value config.json gives, by the configuration's name, which config.json uses.

It never shows a secret. A secret is a file, which it never opens, and no key's value is one. Each value shows as
config.safe_value shows it: an address without its user, password, query and fragment, and a text in the form of
a token (privacy.TOKENS), which no setting holds, as hidden (config.HIDDEN). The last probe's record (worker.probe)
shows without how it sent the key (key_sent, LEFT_OUT), a digest of the headers that carry it, which stays in the
internal state.

- `rows(resolved) -> list[tuple[str, object, str]]`: (name, value as shown, source) for each value of RESOLVED, a
  runtime_config.Resolved: the delivered layer's first, then the registry's keys in its order.
- `lines(resolved, source=False) -> list[str]`: the lines it prints.
- `main(source=False) -> int`: the running configuration's lines. Exit 0; 1 when the configuration cannot run, its
  problems after the lines. REFUSED, exit 4, when it cannot be loaded, there being none or one that cannot be read:
  why, then the fix the error names (applied.refuse).
"""

from __future__ import annotations

import json

from . import (
    applied,
    config,
    file_types,
    registry,
    runtime_config,
    settings_file,
    state_store,
)

# What a value shows without, by its key: the last probe's key_sent, a digest of the headers that carry the reviewer
# model's key, against which whoever saw it could test guesses of a weak key.
LEFT_OUT = {"worker.probe": ("key_sent",)}


def rows(resolved: runtime_config.Resolved) -> list:
    """(name, value as shown, source) for each value: see the module's docstring."""
    layer = resolved.layer
    settings = (layer or {}).get("settings") or {}
    maps = settings_file.map_keys(file_types.DEPLOY.load())
    out = [
        (name, registry.get(settings, name), runtime_config.deploy_source(layer, name))
        for name in registry.leaves(settings, maps)
    ]
    for key in registry.keys():
        source = resolved.source.get(key.name)
        if source is None or (layer is not None and key.owner == registry.DEPLOY):
            continue
        name = state_store.file_key(key) if source == runtime_config.STATE else key.name
        value = registry.get(resolved.values, key.name)
        if isinstance(value, dict) and key.name in LEFT_OUT:
            value = {k: v for k, v in value.items() if k not in LEFT_OUT[key.name]}
        out.append((name, value, source))
    return [(name, config.safe_value(value), source) for name, value, source in out]


def lines(resolved: runtime_config.Resolved, source: bool = False) -> list:
    """The lines config show prints: see the module's docstring."""
    found = rows(resolved)
    width = max((len(s) for _name, _value, s in found), default=0)
    return [
        (f"{s:<{width}}  " if source else "")
        + f"{name} = {json.dumps(value, ensure_ascii=False)}"
        for name, value, s in found
    ]


def main(source: bool = False) -> int:
    """`rk config show [--source]`: see the module's docstring."""
    try:
        resolved = runtime_config.load()
    except config.ConfigError as exc:
        return applied.refuse(exc)
    out = lines(resolved, source)
    if resolved.problems:
        out += ["", "It cannot run yet:"] + [f"  - {p}" for p in resolved.problems]
    print("\n".join(out))
    return 1 if resolved.problems else 0
