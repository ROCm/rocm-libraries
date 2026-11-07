# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Environment-variable configuration.

Parsing is pure Python with no torch, HIP or vLLM imports, so it is safe in
every vLLM process type, including the API server, which may have no GPU.
Invalid values raise :class:`ConfigError` rather than being silently replaced
by defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "ALL_OPS",
    "DEFAULT_OPS",
    "PLUGIN_MODES",
    "Config",
    "ConfigError",
    "load_config",
]

DEFAULT_OPS = (
    "attention",
    "encoder_attention",
    "rms_norm",
    "fused_add_rms_norm",
    "layer_norm",
)
# Opt-in: conv is the only op that needs the MIOpen provider, whose handle
# initialization is costly and paid once per process, so MIOpen is loaded only
# when conv is enabled.
ALL_OPS = DEFAULT_OPS + ("conv", "linear", "linear_all")

# `auto` and `absolute` load the bundled providers absolutely (the bundled
# backend never sees torch's stock providers); `additive` adds them to the
# bundled backend's default plugin directory. See README "Provider loading".
PLUGIN_MODES = ("auto", "additive", "absolute")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


class ConfigError(ValueError):
    """An environment variable holds a value the plugin cannot interpret."""


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean (use 0/1, true/false)")


def _int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = int(raw.strip(), 10)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not an integer") from None
    if val < minimum:
        raise ConfigError(f"{name}={val} must be >= {minimum}")
    return val


def _float(env: Mapping[str, str], name: str, default: float, minimum: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = float(raw.strip())
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a number") from None
    if val < minimum:
        raise ConfigError(f"{name}={val} must be >= {minimum}")
    return val


def _ops(env: Mapping[str, str]) -> frozenset[str]:
    raw = env.get("HIPDNN_VLLM_OPS")
    if raw is None:
        return frozenset(DEFAULT_OPS)
    ops = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    unknown = sorted(set(ops) - set(ALL_OPS))
    if unknown:
        raise ConfigError(
            f"HIPDNN_VLLM_OPS has unknown op(s) {unknown}; valid: {', '.join(ALL_OPS)}"
        )
    return frozenset(ops)


def _paths(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = env.get(name)
    if not raw:
        return ()
    parts = [p for chunk in raw.split(os.pathsep) for p in chunk.split(",")]
    return tuple(os.path.expanduser(p.strip()) for p in parts if p.strip())


def _norm_exact_rows(env: Mapping[str, str]) -> tuple[int, ...] | None:
    """HIPDNN_VLLM_NORM_EXACT_ROWS: ``capture`` (default: vLLM's cudagraph
    capture sizes, returned as None), ``none``, or comma-separated row counts."""
    raw = env.get("HIPDNN_VLLM_NORM_EXACT_ROWS", "capture").strip().lower()
    if raw in ("", "capture"):
        return None
    if raw == "none":
        return ()
    try:
        rows = tuple(sorted({int(tok, 10) for tok in raw.split(",") if tok.strip()}))
    except ValueError:
        raise ConfigError(
            f"HIPDNN_VLLM_NORM_EXACT_ROWS={raw!r}: use capture, none or comma-separated integers"
        ) from None
    if any(r < 1 for r in rows):
        raise ConfigError(f"HIPDNN_VLLM_NORM_EXACT_ROWS={raw!r}: row counts must be >= 1")
    return rows


@dataclass(frozen=True)
class Config:
    disable: bool = False
    ops: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_OPS))
    attn_debug_route: bool = False
    strict: bool = False
    report_dir: str | None = None
    report_interval: float = 60.0
    plan_cache_size: int = 4096
    plugin_mode: str = "auto"
    provider_dirs: tuple[str, ...] = ()
    norm_exact_rows: tuple[int, ...] | None = None  # None: vLLM capture sizes

    def op_enabled(self, op: str) -> bool:
        if op not in ALL_OPS:
            raise ConfigError(f"unknown op {op!r}")
        return not self.disable and op in self.ops


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Parse the plugin configuration from ``env`` (default ``os.environ``)."""
    env = os.environ if env is None else env
    mode = env.get("HIPDNN_VLLM_PLUGIN_MODE", "auto").strip().lower()
    if mode not in PLUGIN_MODES:
        raise ConfigError(
            f"HIPDNN_VLLM_PLUGIN_MODE={mode!r} is not valid; use one of {PLUGIN_MODES}"
        )
    report_dir = env.get("HIPDNN_VLLM_REPORT_DIR") or None
    return Config(
        disable=_bool(env, "HIPDNN_VLLM_DISABLE", False),
        ops=_ops(env),
        attn_debug_route=_bool(env, "HIPDNN_VLLM_ATTN_DEBUG_ROUTE", False),
        strict=_bool(env, "HIPDNN_VLLM_STRICT", False),
        report_dir=os.path.expanduser(report_dir) if report_dir else None,
        report_interval=_float(env, "HIPDNN_VLLM_REPORT_INTERVAL", 60.0, 0.0),
        plan_cache_size=_int(env, "HIPDNN_VLLM_PLAN_CACHE_SIZE", 4096, 1),
        plugin_mode=mode,
        provider_dirs=_paths(env, "HIPDNN_VLLM_PROVIDER_DIRS"),
        norm_exact_rows=_norm_exact_rows(env),
    )
