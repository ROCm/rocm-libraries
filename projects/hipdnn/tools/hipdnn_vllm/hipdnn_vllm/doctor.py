# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``python -m hipdnn_vllm doctor``: install self-check.

Prints the backend path and version, the version ``_C`` was built against, the
loaded providers and engines, the device arch and the pinned versions, then
exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

from . import _bootstrap
from .config import load_config

__all__ = ["main", "pins"]

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def pins() -> dict[str, Any]:
    """Pinned versions: from the wheel's ``_build_info.json``, else from the
    source tree's ``BUILD_PINS.toml``."""
    info = _bootstrap.build_info()
    if info.get("pins"):
        return info["pins"]
    path = os.path.join(os.path.dirname(_HERE), "BUILD_PINS.toml")
    if os.path.isfile(path):
        import tomllib

        with open(path, "rb") as f:
            return tomllib.load(f).get("pins", {})
    return {}


def _version(dist: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _check_pin(checks: list[Check], label: str, actual: str | None, pinned: str | None) -> None:
    if pinned is None:
        checks.append(Check(label, False, f"no pin recorded (installed {actual})"))
    else:
        checks.append(Check(label, actual == pinned, f"installed {actual}, pinned {pinned}"))


def _run(checks: list[Check], name: str, fn: Callable[[], str]) -> Any:
    try:
        detail = fn()
        checks.append(Check(name, True, detail))
        return True
    except Exception as ex:  # noqa: BLE001 - every failure is a reported check
        checks.append(Check(name, False, f"{type(ex).__name__}: {ex}"))
        return False


def collect(device: int = 0) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    report: dict[str, Any] = {}
    info = _bootstrap.build_info()
    p = pins()
    report["build_info"] = info
    report["pins"] = p

    _check_pin(checks, "vllm version", _version("vllm"), p.get("vllm"))
    _check_pin(checks, "torch version", _version("torch"), p.get("torch"))
    _check_pin(checks, "rocm version", _version("rocm"), p.get("rocm"))

    state_holder: dict[str, Any] = {}

    def boot() -> str:
        state = _bootstrap.load(load_config())
        state_holder["state"] = state
        return f"{state.backend_path} ({state.backend_version})"

    if not _run(checks, "bootstrap + version guard", boot):
        return checks, report
    state = state_holder["state"]
    from . import _C

    report["backend_path"] = state.backend_path
    report["backend_version"] = state.backend_version
    report["built_against_backend"] = _C.build_backend_version()
    report["frontend_build_version"] = _C.frontend_version()
    report["plugin_plan"] = {
        "mode": state.plugin_plan.mode,
        "paths": list(state.plugin_plan.paths),
        "why": state.plugin_plan.why,
    }

    torch_backend = _torch_backend()
    report["torch_backend"] = torch_backend
    origin = _C.backend_symbol_origin("hipdnnCreate")
    checks.append(
        Check(
            "_C resolves only the bundled backend",
            os.path.realpath(_C.backend_library_path()) == state.backend_path and origin == state.backend_path,
            f"hipdnnCreate from {origin}",
        )
    )
    mapped = _bootstrap.mapped_backends()
    report["mapped_backends"] = mapped
    checks.append(
        Check(
            "mapped libhipdnn_backend copies are the bundled and torch's",
            state.backend_path in mapped
            and all(m == state.backend_path or m == torch_backend.get("path") for m in mapped),
            ", ".join(mapped),
        )
    )

    import torch

    props = torch.cuda.get_device_properties(device)
    arch = props.gcnArchName.split(":")[0]
    report["device"] = {"index": device, "name": props.name, "arch": props.gcnArchName}
    targets = info.get("gfx_targets")
    if targets:
        checks.append(Check("device arch in wheel family", arch in targets, f"{arch} vs {targets}"))

    loaded_paths = [os.path.realpath(x) for x in _C.loaded_plugin_paths(device)]
    engines = _C.engines(device)
    report["loaded_plugins"] = loaded_paths
    report["engines"] = engines
    checks.append(Check("engines loaded", len(engines) > 0, f"{len(engines)} engines"))

    if state.bundled_dir:
        bundled = [os.path.realpath(x) for x in _bootstrap._plugin_files(state.bundled_dir)]
        wanted = [os.path.realpath(x) for x in state.plugin_plan.paths if os.path.realpath(x) in bundled]
        missing = [b for b in wanted if b not in loaded_paths]
        checks.append(Check("bundled providers loaded", not missing, f"missing {missing}" if missing else ", ".join(wanted)))
        foreign = [x for x in loaded_paths if x not in bundled]
        checks.append(
            Check(
                "only bundled providers loaded",
                not foreign or bool(load_config().provider_dirs),
                f"also loaded: {foreign}" if foreign else "every loaded provider is bundled",
            )
        )
    return checks, report


def _torch_backend() -> dict[str, Any]:
    """torch's own libhipdnn_backend (rocm-sdk-libraries): path and version.
    It stays loadable by torch; hipdnn_vllm never uses it."""
    import ctypes
    import glob

    import torch

    site = os.path.dirname(os.path.dirname(torch.__file__))
    hits = sorted(glob.glob(os.path.join(site, "_rocm_sdk_libraries*", "lib", "libhipdnn_backend.so*")))
    if not hits:
        return {}
    path = os.path.realpath(hits[0])
    try:
        fn = ctypes.CDLL(path, mode=os.RTLD_LAZY | os.RTLD_LOCAL).hipdnnVersionString_ext
        fn.restype = ctypes.c_char_p
        version = fn().decode()
    except (OSError, AttributeError) as e:
        version = f"unreadable: {e}"
    return {"path": path, "version": version}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m hipdnn_vllm doctor")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)

    checks, report = collect(args.device)
    if args.json:
        report["checks"] = [c.__dict__ for c in checks]
        print(json.dumps(report, indent=2, default=str))
    else:
        for key in (
            "backend_path",
            "backend_version",
            "built_against_backend",
            "frontend_build_version",
            "torch_backend",
            "mapped_backends",
            "device",
            "plugin_plan",
        ):
            if key in report:
                print(f"{key}: {report[key]}")
        if "loaded_plugins" in report:
            print("loaded plugins:")
            for path in report["loaded_plugins"]:
                print(f"  {path}")
        if "engines" in report:
            print("engines:")
            for e in report["engines"]:
                print(f"  {e['name']}  [{e['plugin']} {e['version']}]")
        print("checks:")
        for c in checks:
            print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"doctor: {len(failed)} check(s) failed", file=sys.stderr)
        return 1
    print("doctor: all checks passed")
    return 0
