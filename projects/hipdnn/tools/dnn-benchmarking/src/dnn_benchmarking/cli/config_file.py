# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""TOML config file support for the dnn-benchmark CLI."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set


_CONFIG_ONLY_TOP_LEVEL_KEYS: Set[str] = {
    "version",
    "engines",
}

_ALLOWED_ENGINE_KEYS: Set[str] = {"id", "label", "plugin_path"}

_CONFIG_FIELD_RENAMES = {"graph": "graphs"}
_CONFIG_EXCLUDED_DESTS = {
    "help",
    "config",
    "engine",
    "internal_profiling_run",
    "internal_profiling_engine",
    "internal_profiling_graph",
}


@dataclass(frozen=True)
class _ConfigField:
    key: str
    dest: str
    kind: str
    typ: Optional[type] = None
    choices: Optional[FrozenSet[Any]] = None
    optional: bool = False


@lru_cache(maxsize=1)
def _parser_actions_by_dest() -> Dict[str, argparse.Action]:
    from .parser import create_parser

    actions: Dict[str, argparse.Action] = {}
    for action in create_parser()._actions:
        actions[action.dest] = action
    return actions


@lru_cache(maxsize=1)
def _option_dest_map() -> Dict[str, str]:
    option_dests: Dict[str, str] = {}
    for action in _parser_actions_by_dest().values():
        for option in action.option_strings:
            option_dests[option] = action.dest
    return option_dests


def _is_store_true_action(action: argparse.Action) -> bool:
    return (
        getattr(action, "nargs", None) == 0
        and getattr(action, "const", None) is True
        and action.default is False
    )


@lru_cache(maxsize=1)
def _config_fields_by_key() -> Dict[str, _ConfigField]:
    fields: Dict[str, _ConfigField] = {}
    for dest, action in _parser_actions_by_dest().items():
        if dest in _CONFIG_EXCLUDED_DESTS or action.help == argparse.SUPPRESS:
            continue

        key = _CONFIG_FIELD_RENAMES.get(dest, dest)
        field = _config_field_from_action(key, action)
        if field is not None:
            fields[key] = field
    return fields


def _config_field_from_action(
    key: str, action: argparse.Action
) -> Optional[_ConfigField]:
    dest = action.dest
    optional = action.default is None

    if dest == "graph":
        return _ConfigField(key=key, dest=dest, kind="path_list")
    if dest == "plugin_path":
        return _ConfigField(key=key, dest=dest, kind="plugin_path")
    if _is_store_true_action(action):
        return _ConfigField(key=key, dest=dest, kind="scalar", typ=bool)
    if action.type is Path:
        return _ConfigField(key=key, dest=dest, kind="path")
    if action.choices is not None:
        typ = action.type if action.type is not None else str
        return _ConfigField(
            key=key,
            dest=dest,
            kind="choice",
            typ=typ,
            choices=frozenset(action.choices),
            optional=optional,
        )
    if action.type in {int, float, str}:
        return _ConfigField(
            key=key,
            dest=dest,
            kind="scalar",
            typ=action.type,
            optional=optional,
        )
    return None


@lru_cache(maxsize=1)
def _allowed_top_level_keys() -> Set[str]:
    return set(_config_fields_by_key()) | _CONFIG_ONLY_TOP_LEVEL_KEYS


def collect_provided_options(argv: List[str]) -> Set[str]:
    """Return argparse destination names explicitly present in ``argv``."""
    option_dests = _option_dest_map()
    provided: Set[str] = set()
    for arg in argv:
        option = arg.split("=", 1)[0]
        dest = option_dests.get(option)
        if dest is not None and dest != "config":
            provided.add(dest)
    return provided


def apply_config_file(args: argparse.Namespace, provided: Set[str]) -> None:
    """Merge ``args.config`` into parsed CLI args without overriding CLI values."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        return

    path = Path(config_path)
    raw = _load_toml(path)
    overrides = _normalise_config(raw, path)

    engine_overridden = "engine" in provided or "plugin_path" in provided
    for key, value in overrides.items():
        if key in {"engine", "plugin_path", "_config_engine_names"}:
            if engine_overridden:
                continue
        elif key in provided:
            continue
        setattr(args, key, value)


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML config {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a TOML table: {path}")
    return data


def _normalise_config(raw: Dict[str, Any], path: Path) -> Dict[str, Any]:
    version = raw.get("version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported config version in {path}: {version!r}")

    _reject_unknown_keys(raw.keys(), _allowed_top_level_keys(), "config")

    base_dir = path.parent
    out: Dict[str, Any] = {}
    for field in _config_fields_by_key().values():
        if field.kind == "plugin_path":
            continue
        _copy_config_field(raw, out, field, base_dir)

    _normalise_engines(raw, out, base_dir=base_dir)
    plugin_path_field = _config_fields_by_key().get("plugin_path")
    if plugin_path_field is not None:
        _normalise_top_level_plugin_path(raw, out, base_dir=base_dir)

    return out


def _reject_unknown_keys(keys: Iterable[str], allowed: Set[str], context: str) -> None:
    unknown = sorted(set(keys) - allowed)
    if not unknown:
        return
    label = "fields" if len(unknown) > 1 else "field"
    raise ValueError(f"Unknown {context} {label}: {', '.join(unknown)}")


def _copy_config_field(
    raw: Dict[str, Any],
    out: Dict[str, Any],
    field: _ConfigField,
    base_dir: Path,
) -> None:
    if field.kind == "path_list":
        _copy_path_list(raw, out, src=field.key, dest=field.dest, base_dir=base_dir)
        return
    if field.kind == "path":
        _copy_path(raw, out, field.key, dest=field.dest, base_dir=base_dir)
        return
    if field.kind == "choice":
        if field.typ is None or field.choices is None:
            raise AssertionError(f"Invalid config field metadata for {field.key}")
        _copy_choice(
            raw,
            out,
            field.key,
            field.choices,
            typ=field.typ,
            dest=field.dest,
            optional=field.optional,
        )
        return
    if field.kind == "scalar":
        if field.typ is None:
            raise AssertionError(f"Invalid config field metadata for {field.key}")
        _copy_scalar(
            raw,
            out,
            field.key,
            field.typ,
            dest=field.dest,
            optional=field.optional,
        )
        return
    raise AssertionError(f"Unsupported config field kind: {field.kind}")


def _copy_scalar(
    raw: Dict[str, Any],
    out: Dict[str, Any],
    key: str,
    typ: type,
    *,
    dest: Optional[str] = None,
    optional: bool = False,
) -> None:
    if key not in raw:
        return
    value = raw[key]
    target = dest or key
    if value is None and optional:
        out[target] = None
        return
    if not _matches_type(value, typ):
        raise ValueError(f"Config field '{key}' must be {typ.__name__}")
    out[target] = value


def _copy_choice(
    raw: Dict[str, Any],
    out: Dict[str, Any],
    key: str,
    choices: FrozenSet[Any],
    *,
    typ: type,
    dest: Optional[str] = None,
    optional: bool = False,
) -> None:
    target = dest or key
    _copy_scalar(raw, out, key, typ, dest=target, optional=optional)
    if target not in out or (out[target] is None and optional):
        return
    if out[target] not in choices:
        valid = ", ".join(str(choice) for choice in sorted(choices))
        raise ValueError(f"Config field '{key}' must be one of: {valid}")


def _matches_type(value: Any, typ: type) -> bool:
    if typ is int:
        return type(value) is int
    if typ is float:
        return type(value) in {int, float}
    if typ is bool:
        return type(value) is bool
    return isinstance(value, typ)


def _path_from_config(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _copy_path(
    raw: Dict[str, Any],
    out: Dict[str, Any],
    key: str,
    *,
    dest: Optional[str] = None,
    base_dir: Path,
) -> None:
    if key not in raw:
        return
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config field '{key}' must be string path")
    out[dest or key] = _path_from_config(base_dir, value)


def _copy_path_list(
    raw: Dict[str, Any],
    out: Dict[str, Any],
    *,
    src: str,
    dest: str,
    base_dir: Path,
) -> None:
    if src not in raw:
        return
    value = raw[src]
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config field '{src}' must be a non-empty list of paths")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Config field '{src}' must contain only string paths")
    out[dest] = [str(_path_from_config(base_dir, item)) for item in value]


def _normalise_top_level_plugin_path(
    raw: Dict[str, Any], out: Dict[str, Any], *, base_dir: Path
) -> None:
    if "plugin_path" not in raw:
        return
    if "plugin_path" in out:
        raise ValueError(
            "Config cannot set both top-level plugin_path and engine plugin_path"
        )
    value = raw["plugin_path"]
    if isinstance(value, str) and value:
        out["plugin_path"] = [_path_from_config(base_dir, value)]
        return
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        out["plugin_path"] = [_path_from_config(base_dir, item) for item in value]
        return
    raise ValueError(
        "Config field 'plugin_path' must be a string or non-empty list of strings"
    )


def _normalise_engines(
    raw: Dict[str, Any], out: Dict[str, Any], *, base_dir: Path
) -> None:
    engines = raw.get("engines")
    if engines is None:
        return
    if not isinstance(engines, list) or not engines:
        raise ValueError("Config field 'engines' must be a non-empty array of tables")

    ids: List[int] = []
    labels: List[Optional[str]] = []
    plugin_paths: List[Path] = []
    any_plugin_path = False
    seen_labels: Set[str] = set()

    for index, engine in enumerate(engines):
        if not isinstance(engine, dict):
            raise ValueError("Each config engine entry must be a table")
        _reject_unknown_keys(
            engine.keys(), _ALLOWED_ENGINE_KEYS, f"config engine {index}"
        )
        engine_id = engine.get("id")
        if type(engine_id) is not int:
            raise ValueError(f"Config engine {index} must include integer id")
        ids.append(engine_id)

        label = engine.get("label")
        if label is not None:
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"Config engine {index} label must be a non-empty string"
                )
            if label in seen_labels:
                raise ValueError(f"Duplicate config engine label: {label}")
            seen_labels.add(label)
            labels.append(label)
        else:
            labels.append(None)

        plugin_path = engine.get("plugin_path")
        if plugin_path is not None:
            if not isinstance(plugin_path, str) or not plugin_path:
                raise ValueError(f"Config engine {index} plugin_path must be a string")
            any_plugin_path = True
            plugin_paths.append(_path_from_config(base_dir, plugin_path))
        else:
            plugin_paths.append(Path())

    if any_plugin_path:
        if any(not str(p) for p in plugin_paths):
            raise ValueError(
                "Every config engine must set plugin_path when any engine does"
            )
        out["plugin_path"] = plugin_paths
    out["engine"] = ids
    if any(label is not None for label in labels):
        out["_config_engine_names"] = labels
