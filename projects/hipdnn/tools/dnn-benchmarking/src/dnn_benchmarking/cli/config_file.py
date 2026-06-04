# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""TOML config file support for the dnn-benchmark CLI."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Set, cast

from .parser import CONFIG_OPTIONS, OPTION_DEFAULTS, CliOption, ConfigKind


_CONFIG_ONLY_TOP_LEVEL_KEYS: Set[str] = {
    "version",
    "engines",
}

_ALLOWED_ENGINE_KEYS: Set[str] = {"id", "plugin_path"}


def _allowed_top_level_keys() -> Set[str]:
    return {
        cast(str, option.config_key) for option in CONFIG_OPTIONS
    } | _CONFIG_ONLY_TOP_LEVEL_KEYS


def apply_config_file(args: argparse.Namespace) -> None:
    """Merge ``args.config`` into parsed CLI args without overriding CLI values.

    ``args`` must come from ``create_parser(suppress_defaults=True)`` for the
    CLI path. In that mode ``vars(args)`` contains only explicitly supplied
    public options, so argparse itself decides option spellings/abbreviations.
    """
    provided = set(vars(args))
    config_path = getattr(args, "config", None)
    overrides: Dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        raw = _load_toml(path)
        overrides = _normalise_config(raw, path)

    if "engine" in provided or "plugin_path" in provided:
        overrides.pop("engine", None)
        overrides.pop("plugin_path", None)

    merged = dict(OPTION_DEFAULTS)
    merged.update(overrides)
    merged.update(vars(args))

    vars(args).clear()
    vars(args).update(merged)


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
    for option in CONFIG_OPTIONS:
        _copy_config_field(raw, out, option, base_dir)

    _normalise_engines(raw, out, base_dir=base_dir)

    return out


def _reject_unknown_keys(keys: Iterable[str], allowed: Set[str], context: str) -> None:
    unknown = sorted(set(keys) - allowed)
    if not unknown:
        return
    label = "fields" if len(unknown) > 1 else "field"
    raise ValueError(f"Unknown {context} {label}: {', '.join(unknown)}")


ConfigFieldHandler = Callable[[Dict[str, Any], Dict[str, Any], CliOption, Path], None]


def _copy_config_field(
    raw: Dict[str, Any],
    out: Dict[str, Any],
    option: CliOption,
    base_dir: Path,
) -> None:
    kind = cast(ConfigKind, option.config_kind)
    _COPY_FIELD_HANDLERS[kind](raw, out, option, base_dir)


def _copy_scalar_field(
    raw: Dict[str, Any], out: Dict[str, Any], option: CliOption, _base_dir: Path
) -> None:
    _copy_scalar(
        raw,
        out,
        cast(str, option.config_key),
        cast(type, option.config_type),
        dest=option.dest,
        optional=option.config_optional,
    )


def _copy_choice_field(
    raw: Dict[str, Any], out: Dict[str, Any], option: CliOption, _base_dir: Path
) -> None:
    _copy_choice(
        raw,
        out,
        cast(str, option.config_key),
        cast(FrozenSet[Any], option.choices),
        typ=cast(type, option.config_type),
        dest=option.dest,
        optional=option.config_optional,
    )


def _copy_path_field(
    raw: Dict[str, Any], out: Dict[str, Any], option: CliOption, base_dir: Path
) -> None:
    _copy_path(
        raw,
        out,
        cast(str, option.config_key),
        dest=option.dest,
        base_dir=base_dir,
    )


def _copy_path_list_field(
    raw: Dict[str, Any], out: Dict[str, Any], option: CliOption, base_dir: Path
) -> None:
    _copy_path_list(
        raw,
        out,
        src=cast(str, option.config_key),
        dest=option.dest,
        base_dir=base_dir,
    )


def _copy_path_or_path_list_field(
    raw: Dict[str, Any], out: Dict[str, Any], option: CliOption, base_dir: Path
) -> None:
    _copy_path_or_path_list(
        raw,
        out,
        cast(str, option.config_key),
        dest=option.dest,
        base_dir=base_dir,
    )


_COPY_FIELD_HANDLERS: dict[ConfigKind, ConfigFieldHandler] = {
    ConfigKind.SCALAR: _copy_scalar_field,
    ConfigKind.CHOICE: _copy_choice_field,
    ConfigKind.PATH: _copy_path_field,
    ConfigKind.PATH_LIST: _copy_path_list_field,
    ConfigKind.PATH_OR_PATH_LIST: _copy_path_or_path_list_field,
}


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


def _copy_path_or_path_list(
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
    target = dest or key
    if isinstance(value, str) and value:
        out[target] = [_path_from_config(base_dir, value)]
        return
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        out[target] = [_path_from_config(base_dir, item) for item in value]
        return
    raise ValueError(
        f"Config field '{key}' must be a string or non-empty list of strings"
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
    plugin_paths: List[Optional[Path]] = []
    any_plugin_path = False

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

        plugin_path = engine.get("plugin_path")
        if plugin_path is not None:
            if not isinstance(plugin_path, str) or not plugin_path:
                raise ValueError(f"Config engine {index} plugin_path must be a string")
            any_plugin_path = True
            plugin_paths.append(_path_from_config(base_dir, plugin_path))
        else:
            plugin_paths.append(None)

    if any_plugin_path:
        if "plugin_path" in out:
            raise ValueError(
                "Config cannot set both top-level plugin_path and engine plugin_path"
            )
        if any(p is None for p in plugin_paths):
            raise ValueError(
                "Every config engine must set plugin_path when any engine does"
            )
        out["plugin_path"] = [p for p in plugin_paths if p is not None]
    out["engine"] = ids
