# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small helpers shared by the attention backward tests.

Nothing here emits IR or needs a device at import time; the case and config
tables are imported when a helper is called, so a CPU-only collection never
pulls the device harness in.

* :func:`kernel_signature`: the ``(name, kind, text)`` kernargs of a lowered
  kernel, ``kind`` in the ``struct`` letters of the kernarg ABIs (``Q``
  pointer, ``f`` f32, ``q`` i64, ``i`` i32).
* :func:`device_arch`: the gfx id of device 0, ``None`` without a HIP device.
* :func:`find_bwd_case`: a supplementary kernel case (or a case of the extra
  lists) by id, else the case-table case.
* :func:`distinct_configs` / :func:`device_configs`: the kernel-config lists
  the IR and device tests parametrise over.
* :func:`fake_bench_ok`: a stand-in for the bench driver's ``main`` that
  writes one passing cell per id (every rocKE arm faster than the bar arm).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

__all__ = [
    "device_arch",
    "device_configs",
    "distinct_configs",
    "fake_bench_ok",
    "find_bwd_case",
    "kernel_signature",
]

_KERNEL_HEAD = re.compile(r"define amdgpu_kernel void @\S+\((.*?)\) #0")


def kernel_signature(ir: str) -> list[tuple[str, str, str]]:
    """``(name, kind, parameter text)`` of every kernarg of the lowered
    kernel in ``ir``: kind ``"Q"`` (pointer), ``"f"`` (f32), ``"q"`` (i64)
    or ``"i"`` (i32)."""
    head = _KERNEL_HEAD.search(ir).group(1)
    out = []
    for p in head.split(", "):
        name = p.split("%")[-1]
        if p.startswith("ptr"):
            kind = "Q"
        elif p.startswith("float"):
            kind = "f"
        else:
            kind = "q" if p.startswith("i64") else "i"
        out.append((name, kind, p))
    return out


def device_arch() -> Optional[str]:
    """The gfx id of HIP device 0, or ``None`` (no HIP runtime or no device)."""
    try:
        from rocke.runtime.hip_module import get_device_arch, get_device_count

        if get_device_count() < 1:
            return None
        return get_device_arch(0)
    except Exception:  # noqa: BLE001 - no HIP runtime on this host
        return None


def find_bwd_case(case_id: str, *extra: Sequence):
    """The supplementary kernel case ``case_id`` (then the cases of each
    ``extra`` list, in order), else the case-table case of that id."""
    from .sdpa.bwd_cases import get_bwd_case
    from .sdpa.bwd_kernel_cases import SUPPLEMENTARY_CASES

    for cases in (SUPPLEMENTARY_CASES, *extra):
        for c in cases:
            if c.id == case_id:
                return c
    return get_bwd_case(case_id)


def distinct_configs(arch: str) -> list:
    """The kernel configs whose knobs exist on ``arch``, one per distinct knob
    set (the first config of each set, in ``CONFIGS`` order)."""
    from .sdpa.bwd_kernel_cases import CONFIGS, config_knobs

    out, seen = [], []
    for cfg in CONFIGS:
        knobs = config_knobs(cfg, arch)
        if knobs is None or knobs in seen:
            continue
        seen.append(knobs)
        out.append(cfg)
    return out


def device_configs(arch: Optional[str], device_archs: Sequence[str]) -> list:
    """The selected kernel configs of a device target, ``["none"]`` (one
    skipped parameter) for a host that is not one."""
    from .sdpa.bwd_kernel_cases import selected_configs

    return selected_configs(arch) if arch in device_archs else ["none"]


def fake_bench_ok(bb) -> Callable[[list], int]:
    """A bench ``main(argv)`` for the sweep / split driver tests: per id in
    ``--ids`` one complete cell (the bar arm and one passing rocKE arm per
    config of ``--rocke-configs``) appended to ``<--out-dir>/cells.jsonl``
    through ``bb.append_jsonl``, plus one ``env.jsonl`` record."""

    def fake_bench(argv):
        out = Path(argv[argv.index("--out-dir") + 1])
        cfgs = json.loads(Path(argv[argv.index("--rocke-configs") + 1]).read_text())
        for cid in argv[argv.index("--ids") + 1].split(","):
            arms = {"default|native": {"status": "ok", "median_ms": 2.0, "spread": 0.0,
                                       "attempt_medians_ms": [2.0] * 5}}  # fmt: skip
            for c in cfgs["configs"]:
                arms[f"rocke|{c['label']}"] = {"status": "ok", "median_ms": 1.0,
                                               "spread": 0.0,
                                               "attempt_medians_ms": [1.0] * 5}  # fmt: skip
            bb.append_jsonl(out / "cells.jsonl", {
                "id": cid, "complete": True, "bar_arm": "default|native", "arms": arms,
                "verdicts": {f"rocke|{c['label']}": "pass" for c in cfgs["configs"]},
            })  # fmt: skip
        bb.append_jsonl(out / "env.jsonl", {"host": "n"})
        return 0

    return fake_bench
