# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parse MIOpenDriver ``bnorm*``/``layernorm*`` case lines for the norm shard runner.

Neither ``benchmark_batchnorm2d_aot.py`` nor ``benchmark_layernorm2d_aot.py``
takes a case file -- see ``run_norm_shards.py``'s module docstring for why --
so a MIOpenDriver case file has to be turned into the ``(dims, dtype, mode)``
tuples those two scripts' CLIs actually accept. This is deliberately not
``benchmark_implicit_gemm_conv.parse_miopen_cmd`` (or its grouped-conv/direct-conv
siblings): those return a conv-shaped ``(ConvGeometry, dtype, forw)``, the same
reason ``norm_bench_common`` already gives for not reusing
``case_results``/``early_stop`` for this family.

Not every MIOpenDriver line is runnable by the current kernel families -- see
``run_norm_shards.py``'s module docstring (and the plan this was built from)
for the exact scope:

* ``bnormfp16``/``bnormbfp16`` + ``--layout NCHW`` (or ``NCDHW`` with ``-D``, depth folded into the height) + ``--forw 1`` (training)
  or ``--forw 2`` (inference). Bare ``bnorm`` (fp32), ``--layout NHWC``,
  and ``--forw 0`` (backward) are out of scope.
* ``layernormbfp16`` + ``-L NW`` + ``-F 1`` (forward). ``-F 0`` (backward) is
  out of scope.

A skipped line is reported, not silently dropped: :func:`parse_file` returns a
``Counter`` of skip reasons alongside the cases it could parse.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

_INT_FLAG = r"-{{1,2}}{name}\s+(-?\d+)"


def _int_flag(line: str, name: str):
    m = re.search(_INT_FLAG.format(name=re.escape(name)), line)
    return int(m.group(1)) if m else None


def _str_flag(line: str, name: str):
    m = re.search(rf"-{{1,2}}{re.escape(name)}\s+(\S+)", line)
    return m.group(1) if m else None


@dataclass(frozen=True)
class ParsedCase:
    """One in-scope case, ready to become an AOT script's argv."""

    family: str  # "bnorm" | "lnorm"
    dims: Tuple[Tuple[str, int], ...]  # sorted dims.items(), e.g. (("c", 64), ("h", 28), ...)
    dtype: str  # "fp16" | "bf16"
    mode: str  # "training" | "inference" (bnorm) | "forward" (lnorm)
    layout: str = "NCHW"  # "NHWC" only from parse_bnorm_line(..., nhwc=True)

    @property
    def dims_dict(self) -> Dict[str, int]:
        return dict(self.dims)

    @property
    def dedup_key(self) -> tuple:
        return (self.family, self.dims, self.dtype, self.mode)


_BNORM_DTYPES = {"bnormfp16": "fp16", "bnormbfp16": "bf16", "bnorm": "fp32"}
_LNORM_DTYPES = {"layernormfp16": "fp16", "layernormbfp16": "bf16"}


def parse_bnorm_line(line: str, backward: bool = False, nhwc: bool = False):
    """One ``MIOpenDriver bnorm*`` line to a :class:`ParsedCase`, or a skip reason.

    ``backward=True`` additionally accepts ``--forw 0 -b 1`` rows (mode ``"backward"``);
    ``nhwc=True`` accepts ``--layout NHWC`` rows (``ParsedCase.layout == "NHWC"``)."""
    m = re.match(r"\s*(?:MIOpenDriver\s+)?(\S+)", line)
    if not m:
        return None, "unparseable"
    token = m.group(1)
    dtype = _BNORM_DTYPES.get(token)
    if dtype is None:
        return None, f"unknown-token({token})"

    layout = _str_flag(line, "layout") or "NCHW"
    depth = _int_flag(line, "D")
    if layout == "NCDHW":
        if depth is None:
            return None, "missing-dims"
    elif layout == "NHWC" and nhwc and depth is None:
        pass
    elif layout != "NCHW" or depth is not None:
        return None, f"unsupported-layout({layout})"

    forw = _int_flag(line, "forw")
    if forw == 2:
        mode = "inference"
    elif forw == 1:
        mode = "training"
    elif forw == 0:
        if not (backward and _int_flag(line, "b") == 1):
            return None, "backward-only(forw=0)"
        mode = "backward"
    else:
        return None, f"unknown-forw({forw})"

    n, c, h, w = (_int_flag(line, name) for name in ("n", "c", "H", "W"))
    if None in (n, c, h, w):
        return None, "missing-dims"
    if layout == "NCDHW":
        # NCDHW is contiguous N,C,(D,H,W): the same memory as NCHW with the depth folded into the height.
        h *= depth
    dims = tuple(sorted({"n": n, "c": c, "h": h, "w": w}.items()))
    return ParsedCase(family="bnorm", dims=dims, dtype=dtype, mode=mode, layout="NHWC" if layout == "NHWC" else "NCHW"), None


def parse_lnorm_line(line: str):
    """One ``MIOpenDriver layernorm*`` line to a :class:`ParsedCase`, or a skip reason."""
    m = re.match(r"\s*(?:MIOpenDriver\s+)?(\S+)", line)
    if not m:
        return None, "unparseable"
    token = m.group(1)
    dtype = _LNORM_DTYPES.get(token)
    if dtype is None:
        return None, "unsupported-dtype(fp32)" if token == "layernorm" else f"unknown-token({token})"

    layout = _str_flag(line, "L")
    if layout != "NW":
        return None, f"unsupported-layout({layout})"

    forw = _int_flag(line, "F")
    if forw == 0:
        return None, "backward-only(F=0)"
    if forw != 1:
        return None, f"unknown-forw({forw})"

    rows, cols = _int_flag(line, "N"), _int_flag(line, "W")
    if None in (rows, cols):
        return None, "missing-dims"
    dims = tuple(sorted({"rows": rows, "cols": cols}.items()))
    return ParsedCase(family="lnorm", dims=dims, dtype=dtype, mode="forward"), None


_PARSERS = {"bnorm": parse_bnorm_line, "lnorm": parse_lnorm_line}


def parse_file(path, family: str) -> Tuple[List[ParsedCase], Counter]:
    """Every in-scope, deduplicated :class:`ParsedCase` in *path*, plus skip counts.

    Lines are dealt in first-seen order; a line whose ``dedup_key`` repeats an
    earlier one is counted under ``"duplicate"`` rather than run again.
    """
    parser = _PARSERS[family]
    skips: Counter = Counter()
    cases: List[ParsedCase] = []
    seen = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        case, reason = parser(line)
        if case is None:
            skips[reason] += 1
            continue
        if case.dedup_key in seen:
            skips["duplicate"] += 1
            continue
        seen.add(case.dedup_key)
        cases.append(case)
    return cases, skips
