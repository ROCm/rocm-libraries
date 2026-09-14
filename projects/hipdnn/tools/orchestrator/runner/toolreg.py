# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The tool registry: where an executable lives on *this* machine, and nothing else.

Arguments, cwd, timeouts and stdin are deliberately rejected here. A tool called twice
in a flow needs different arguments each time, so registry-level arguments are either
useless or a second place to look when an argv comes out wrong.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError
from .refs import Resolver, platform_facts, render_value

TOOL_KEYS = {"exe", "env", "path_prepend"}
TOP_KEYS = {"version", "vars", "tools", "profiles"}
#: Keys that used to look plausible here. Naming one is a design mistake, not a typo,
#: so the message says where the key belongs instead of listing valid keys.
RELOCATED = {
    "args": "step",
    "cwd": "step",
    "timeout": "step",
    "stdin": "step",
    "prompt_file": "step",
}


@dataclass(frozen=True)
class Tool:
    name: str
    exe: str
    env: dict[str, str] = field(default_factory=dict)
    path_prepend: tuple[str, ...] = ()


@dataclass
class ToolRegistry:
    path: Path
    vars: dict[str, Any]
    tools: dict[str, Tool]

    @classmethod
    def load(cls, path: str | Path, profile: str | None = None) -> "ToolRegistry":
        path = Path(path).resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except OSError as error:
            raise ConfigError(f"cannot read tool registry {path}: {error}") from None
        except yaml.YAMLError as error:
            raise ConfigError(f"{path} is not valid YAML: {error}") from None
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{path}: top level must be a mapping")
        _reject_unknown(raw, TOP_KEYS, f"{path}: top level")

        overlay = _profile(raw, profile, path)
        machine_vars = _resolve_vars(
            {**(raw.get("vars") or {}), **(overlay.get("vars") or {})}, path
        )

        tool_specs = dict(raw.get("tools") or {})
        for name, spec in (overlay.get("tools") or {}).items():
            tool_specs[name] = {**(tool_specs.get(name) or {}), **(spec or {})}
        if not tool_specs:
            raise ConfigError(f"{path}: no tools declared")

        resolver = Resolver(vars=machine_vars)
        tools = {
            name: _tool(name, spec, resolver, path) for name, spec in tool_specs.items()
        }
        return cls(path=path, vars=machine_vars, tools=tools)

    def get(self, name: str) -> Tool:
        try:
            return self.tools[name]
        except KeyError:
            known = ", ".join(sorted(self.tools)) or "(none)"
            raise ConfigError(
                f"flow uses tool '{name}', which {self.path.name} does not declare; known: {known}"
            ) from None

    def resolve_exe(self, tool: Tool) -> Path:
        """Absolute path used as-is; a bare name is resolved through PATH.

        Called for every tool a flow uses *before the first step launches*, so a missing
        executable fails the run immediately instead of at step nine of ten.
        """
        candidate = Path(tool.exe)
        if candidate.is_absolute() or any(sep in tool.exe for sep in ("/", "\\")):
            if not candidate.exists():
                raise ConfigError(f"tool '{tool.name}': {candidate} does not exist")
            return candidate
        found = shutil.which(tool.exe)
        if not found:
            raise ConfigError(f"tool '{tool.name}': '{tool.exe}' is not on PATH")
        return Path(found)


def _profile(
    raw: Mapping[str, Any], profile: str | None, path: Path
) -> Mapping[str, Any]:
    profiles = raw.get("profiles") or {}
    if profile is None:
        return {}
    if profile not in profiles:
        known = ", ".join(sorted(profiles)) or "(none)"
        raise ConfigError(f"{path}: unknown profile '{profile}'; known: {known}")
    return profiles[profile] or {}


def _tool(name: str, spec: Any, resolver: Resolver, path: Path) -> Tool:
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{path}: tool '{name}' must be a mapping")
    for key in spec:
        if key in RELOCATED:
            raise ConfigError(
                f"{path}: tool '{name}' declares '{key}'. The registry only locates "
                f"executables; '{key}' belongs on the {RELOCATED[key]} that invokes it."
            )
    _reject_unknown(spec, TOOL_KEYS, f"{path}: tool '{name}'")

    exe_spec = spec.get("exe")
    if exe_spec is None:
        raise ConfigError(f"{path}: tool '{name}' has no 'exe'")
    if isinstance(exe_spec, Mapping):
        host = platform_facts()["os"]
        if host not in exe_spec:
            known = ", ".join(sorted(exe_spec))
            raise ConfigError(
                f"{path}: tool '{name}' has no 'exe' for this platform ({host}); declares: {known}"
            )
        exe_spec = exe_spec[host]
    if not isinstance(exe_spec, str) or not exe_spec.strip():
        raise ConfigError(f"{path}: tool '{name}' has an empty 'exe'")

    env = spec.get("env") or {}
    if not isinstance(env, Mapping):
        raise ConfigError(f"{path}: tool '{name}' env must be a mapping")
    path_prepend = spec.get("path_prepend") or []
    if isinstance(path_prepend, str):
        path_prepend = [path_prepend]

    return Tool(
        name=name,
        exe=str(render_value(exe_spec, resolver)),
        env={
            key: str(render_value(str(value), resolver)) for key, value in env.items()
        },
        path_prepend=tuple(
            str(render_value(str(item), resolver)) for item in path_prepend
        ),
    )


def _resolve_vars(raw: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Vars may reference other vars. Fixed-point iteration, with cycle detection."""
    resolved: dict[str, Any] = {}
    pending = dict(raw)
    while pending:
        progressed = False
        for name in list(pending):
            resolver = Resolver(vars=resolved)
            try:
                resolved[name] = render_value(str(pending[name]), resolver)
            except ConfigError:
                continue
            del pending[name]
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise ConfigError(
                f"{path}: vars cannot be resolved (cycle or unknown reference): {unresolved}"
            )
    return resolved


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )
