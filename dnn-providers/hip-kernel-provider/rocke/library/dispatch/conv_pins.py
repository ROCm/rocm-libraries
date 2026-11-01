# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pinned conv winners as a dispatch heuristic.

A *pins* file (``benchmarks/common/derive_conv_pins.py`` writes one from
benchmark runs; ``benchmark_implicit_gemm_conv --pins`` and
``conv_step_replay.py --pins`` read it) names, per measured case, the kernel
that won and its launch parameters::

    {"cases": {"fwd_s3": {"cmd": "MIOpenDriver convbfp16 ... -F 1 ...",
                          "kernel": "<identity label> @split_k=2/full1824",
                          "calls": 2, "ms": 1.75}, ...}}

This module lets dispatch use it the way a tuned table is used:

* :func:`load_conv_pins` reads the file and keys every case by its problem
  (direction, dtype and every dimension / stride / pad / dilation), parsed from
  the MIOpenDriver command -- no GPU, no benchmark import.
* :meth:`ConvPins.lookup` answers a :class:`ConvGroupedRequest` (direction
  ``fwd`` / ``dgrad`` / ``wgrad``) with its :class:`ConvPin`: the instance name
  (the identity label, :func:`kernels.conv.kernel_label.strip_launch_suffix`)
  and the launch parameters the label carries (``split_k``, a tail split's
  ``full`` tile count).
* :meth:`ConvPins.ranker` is a :data:`rocke.dispatch.core.Ranker` for
  :func:`dispatch.grouped_convolution.dispatch_conv_grouped`: the pinned
  instance first when it is among the admitted candidates, the registry's own
  order otherwise (a pin never makes dispatch refuse a request).

Nothing reads a pins file unless a caller passes one; registry order stays the
default policy.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_SPLIT_RE = re.compile(r"@split_k=(\d+)(?:/full(\d+))?")
_DIRECTIONS = {1: "fwd", 2: "dgrad", 4: "wgrad"}
_DTYPES = {"convbfp16": "bf16", "convfp16": "fp16", "conv": "fp32", "convfp32": "fp32"}

# MIOpenDriver flag -> problem field (long and short forms).
_FLAGS = {
    "-n": "N", "--batchsize": "N",
    "-c": "C", "--in_channels": "C",
    "-k": "K", "--out_channels": "K",
    "--in_d": "Di", "-H": "Hi", "--in_h": "Hi", "-W": "Wi", "--in_w": "Wi",
    "--fil_d": "Z", "-y": "Y", "--fil_h": "Y", "-x": "X", "--fil_w": "X",
    "--pad_d": "pad_d", "-p": "pad_h", "--pad_h": "pad_h", "-q": "pad_w", "--pad_w": "pad_w",
    "--conv_stride_d": "stride_d", "-u": "stride_h", "--conv_stride_h": "stride_h",
    "-v": "stride_w", "--conv_stride_w": "stride_w",
    "--dilation_d": "dilation_d", "-l": "dilation_h", "--dilation_h": "dilation_h",
    "-j": "dilation_w", "--dilation_w": "dilation_w",
    "-g": "G", "--group_count": "G",
    "-F": "forw", "--forw": "forw",
}
_DEFAULTS = {
    "N": 1, "G": 1, "stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
    "dilation_h": 1, "dilation_w": 1,
}
_KEY_FIELDS = (
    "N", "C", "K", "Hi", "Wi", "Y", "X", "G", "stride_h", "stride_w", "pad_h", "pad_w",
    "dilation_h", "dilation_w", "Di", "Z", "stride_d", "pad_d", "dilation_d",
)


def _norm_dtype(dtype: str) -> str:
    d = str(dtype).lower()
    return {"bfloat16": "bf16", "float16": "fp16", "half": "fp16", "float32": "fp32", "float": "fp32"}.get(d, d)


def parse_miopen_cmd(cmd: str) -> tuple[str, str, tuple]:
    """``(direction, dtype, dims key)`` of a MIOpenDriver conv command line."""
    toks = shlex.split(cmd)
    if toks and toks[0].endswith("MIOpenDriver"):
        toks = toks[1:]
    if not toks or toks[0] not in _DTYPES:
        raise ValueError(f"not a MIOpenDriver conv command: {cmd!r}")
    dtype = _DTYPES[toks[0]]
    vals: dict[str, int] = dict(_DEFAULTS)
    i = 1
    while i < len(toks):
        name = _FLAGS.get(toks[i])
        if name is not None and i + 1 < len(toks):
            vals[name] = int(toks[i + 1])
            i += 2
        else:
            i += 2 if toks[i].startswith("-") and i + 1 < len(toks) and not toks[i + 1].startswith("-") else 1
    forw = vals.pop("forw", 1)
    if forw not in _DIRECTIONS:
        raise ValueError(f"{cmd!r}: -F {forw} is not one direction (1 fwd, 2 dgrad, 4 wgrad)")
    if "Di" in vals:
        for f, d in (("Z", 1), ("stride_d", 1), ("pad_d", 0), ("dilation_d", 1)):
            vals.setdefault(f, d)
    return _DIRECTIONS[forw], dtype, tuple(vals.get(f) for f in _KEY_FIELDS)


def request_key(req) -> tuple[str, str, tuple]:
    """The pins key of a :class:`ConvGroupedRequest`-like object."""
    vals = {f: getattr(req, f, None) for f in _KEY_FIELDS}
    for f, d in _DEFAULTS.items():
        if vals[f] is None:
            vals[f] = d
    vals = {f: (int(v) if v is not None else None) for f, v in vals.items()}
    return str(req.direction).lower(), _norm_dtype(req.dtype), tuple(vals[f] for f in _KEY_FIELDS)


@dataclass(frozen=True)
class ConvPin:
    """One pinned winner: the instance and the launch it won with."""

    case: str
    instance: str
    label: str
    split_k: int | None = None
    tail_full: int | None = None
    ms: float | None = None
    calls: int = 1

    @property
    def launch(self) -> dict:
        d = {}
        if self.split_k is not None:
            d["split_k"] = self.split_k
        if self.tail_full is not None:
            d["tail_full"] = self.tail_full
        return d


@dataclass
class ConvPins:
    """A pins file keyed by problem."""

    source: str = ""
    pins: dict = field(default_factory=dict)  # key -> ConvPin

    def __len__(self) -> int:
        return len(self.pins)

    def lookup(self, req) -> ConvPin | None:
        return self.pins.get(request_key(req))

    def ranker(self):
        """A dispatch ranker: the pinned instance first, else registry order."""

        def rank(request, candidates: Sequence):
            pin = self.lookup(request)
            if pin is None:
                return tuple(candidates)
            first = [c for c in candidates if c.name == pin.instance]
            if not first:
                return tuple(candidates)
            return tuple(first) + tuple(c for c in candidates if c.name != pin.instance)

        return rank


def pins_from_mapping(data: Mapping, source: str = "") -> ConvPins:
    """:class:`ConvPins` from a decoded pins document."""
    out = ConvPins(source=source)
    for case, row in (data.get("cases") or {}).items():
        label = str(row["kernel"]).strip()
        instance = label.split()[0] if label else ""
        m = _SPLIT_RE.search(label)
        key = parse_miopen_cmd(row["cmd"])
        if key in out.pins and out.pins[key].instance != instance:
            raise ValueError(f"{source}: cases {out.pins[key].case} and {case} pin one problem to two kernels")
        out.pins[key] = ConvPin(
            case=case,
            instance=instance,
            label=label,
            split_k=int(m.group(1)) if m else None,
            tail_full=int(m.group(2)) if m and m.group(2) is not None else None,
            ms=float(row["ms"]) if row.get("ms") is not None else None,
            calls=int(row.get("calls", 1)),
        )
    return out


def load_conv_pins(path) -> ConvPins:
    """Read a pins JSON file."""
    p = Path(path)
    return pins_from_mapping(json.loads(p.read_text()), source=str(p))
