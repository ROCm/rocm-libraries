#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Speed and output A/B of the rocKE dense attention forward across rocKE trees and
storage layouts (stride-aware launches), every arm timed in ONE process on the SAME
device buffers.

Arms are rocKE trees that cannot be imported side by side (the kernel ABI differs), so a
run has two steps:

1. **Export** (``export`` subcommand; ``run`` starts one subprocess per arm tree): the
   tree's own ``rocke`` / ``kernels`` / ``dispatch`` on ``sys.path``, one compiler. For
   every case (a shape of the shape file on one layout) the tree's dispatch factory gives
   the spec and the tree's own ``run_attention_dense_torch`` launches it on views at
   sentinel addresses, with the HIP runtime replaced by a recorder: nothing runs. Per
   launch (the KV-segment combine included) it keeps the code object, kernel name, grid,
   block and packed kernarg bytes, and finds every pointer slot in them (each sentinel
   must appear exactly where the views put it). The export file is JSON-flow YAML. An
   export never asks a device (it is torch-free and may run on a host without one): its
   specs are built for ``--num-cus`` (default: the arch's ``ARCH_CUS`` entry), and it
   records that count. ``run`` passes its own count (the device's) to the exports it
   starts and refuses an export file built for another, so export by hand with the CU
   count of the part that will time it.
2. **Time** (``run``; this process, torch): per case one set of buffers (inputs from a
   fixed seed, decoy data around the views, NaN in O, the segment workspace); every arm's
   code objects are loaded into the process with the pointer slots patched to those
   buffers, and the A/A' control (``--control``) is the baseline's code objects loaded a
   second time. ``--rounds`` rounds in a balanced mirrored order, one HIP-graph replay per
   arm and round, N launches sized to ``--target-ms``. The numeric gate runs for every arm
   before and after the timing: the output against an fp64 reference on a row subset (max
   abs error within 2e-2 for fp16, 4e-2 for bf16, no NaN) and a hash of the output (the
   hashes of the arms are compared and reported). A case where an arm fails the gate is
   ``check_failed`` and never enters a geomean. A layout an arm's runner REFUSES
   (``ValueError`` / ``NotImplementedError``) is recorded as ``refused`` for that arm and
   the case leaves that arm's ratios; a case the BASELINE refuses has no ratios.

Layouts (logical tensors are ``[B, S, H, D]``, ``[B, Skv, Hkv, D]`` for K/V): ``bshd``
packed; ``bhsd`` storage ``[B, H, S, D]``; ``packed_qkv`` Q, K, V as views of one
``[B, S, Hq + 2 Hkv, D]`` buffer (needs sq == sk, O separate); ``batch_slice`` the odd
batches of ``[2B, S, H, D]`` buffers (O separate). The rest of every backing buffer holds
decoy data (random for inputs, NaN for O), so a kernel that reads or writes outside its
view gives a wrong output.

Modes: ``export`` (one tree -> ``--out``; ``--list``: specs only); ``--list`` (CPU, no
torch: every case's factory choice in every arm tree, against ``expect``, and spec
identity across arms); ``--dry-run`` (CPU, no torch, no compiler: a stand-in code object,
synthetic times, stubbed checks; export, kernarg patching, schedule, statistics and
output are real); a GPU run (default); ``bar RESULT`` (the speed bar of a finished result).

Dry-run hooks (``LABEL=VALUE``, addressed to one arm): ``--dry-run-effect B=0.05`` (times
+5%), ``--dry-run-refuse B=bhsd,g02/bshd`` (layouts / case ids), ``--dry-run-wrong B=bhsd``
(the arm's checks fail there).

Result: ``--out FILE.yaml`` is JSON-flow YAML (readable without PyYAML), one
``AB_ROW {json}`` line per finished case goes to stdout (``bar`` also reads a log of
them), ``FILE.stats.md`` holds the analysis; the arms' export files go beside it.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import datetime
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import re
import struct
import subprocess
import sys
import tempfile
import time
import types
import zlib
from collections.abc import Sequence
from typing import Any

SCRIPT_VERSION = "dense-strided-ab-1"
# CU count the factory request is built for when no device is queried (--dry-run,
# --list, a bare export); a measured run uses the device's own count (resolve_num_cus).
ARCH_CUS = {"gfx942": 304, "gfx950": 256}
MASK_TYPES = {"none": 0, "causal_tl": 1, "causal_br": 2}
LAYOUTS = ("bshd", "bhsd", "packed_qkv", "batch_slice")
TOL = {"fp16": 2e-2, "bf16": 4e-2}  # max-abs tolerances of the dense numeric tests
MIN_ROUNDS = 24
SHAPE_FIELDS = ("id", "batch", "hq", "hkv", "d", "sq", "sk", "mask", "dtype", "select")
LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.+-]*$")
_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Pure helpers (no tree imports, no torch)
# --------------------------------------------------------------------------- #


class LayoutNotApplicable(ValueError):
    """The layout cannot describe this shape (packed QKV needs sq == sk)."""


def load_stats():
    """The statistics module that ships next to this file (loaded by path, so the
    driver never depends on what ``benchmarks`` an arm tree provides)."""
    name = "attention_ab_stats"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_HERE, "attention_ab_stats.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def resolve_rocke_root(tree: str) -> str:
    """The rocKE root (holds ``platform/python`` and ``library``) of ``tree``: the
    directory itself, or the rocKE directory of a repository root."""
    top = os.path.realpath(os.path.expanduser(tree))
    for cand in (
        top,
        os.path.join(top, "dnn-providers", "hip-kernel-provider", "rocke"),
    ):
        if os.path.isdir(os.path.join(cand, "platform", "python")) and os.path.isdir(
            os.path.join(cand, "library")
        ):
            return cand
    raise SystemExit(
        f"{tree}: not a rocKE root (no platform/python and library below it or below dnn-providers/hip-kernel-provider/rocke)"
    )


def parse_arms(items: Sequence[str]) -> list[tuple[str, str]]:
    """``[(label, rocke root or export file)]`` from ``LABEL=TREE|EXPORT`` items;
    labels are unique."""
    out: list[tuple[str, str]] = []
    for it in items:
        label, sep, tree = it.partition("=")
        if not sep or not label or not tree:
            raise SystemExit(f"--arm {it!r}: expected LABEL=TREE")
        if not LABEL_RE.match(label):
            raise SystemExit(f"--arm {it!r}: label must match {LABEL_RE.pattern}")
        if label in [l for l, _ in out]:
            raise SystemExit(f"--arm: duplicate label {label!r}")
        out.append(
            (
                label,
                (
                    os.path.abspath(tree)
                    if os.path.isfile(tree)
                    else resolve_rocke_root(tree)
                ),
            )
        )
    return out


def load_shapes(path: str) -> list[dict[str, Any]]:
    data = load_stats().load_structured(path)
    shapes = data["shapes"] if isinstance(data, dict) else data
    if not isinstance(shapes, list) or not shapes:
        raise SystemExit(f"{path}: no shapes")
    ids = [s.get("id") for s in shapes]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate shape ids")
    for s in shapes:
        for f in SHAPE_FIELDS:
            if f not in s:
                raise SystemExit(f"{path}: shape {s.get('id')} lacks {f!r}")
        if s["mask"] not in MASK_TYPES:
            raise SystemExit(
                f"{path}: shape {s['id']}: mask {s['mask']!r} not in {sorted(MASK_TYPES)}"
            )
        if s["dtype"] not in TOL:
            raise SystemExit(
                f"{path}: shape {s['id']}: dtype {s['dtype']!r} not in {sorted(TOL)}"
            )
        bad = [l for l in s.get("layouts", ["bshd"]) if l not in LAYOUTS]
        if bad:
            raise SystemExit(
                f"{path}: shape {s['id']}: layouts {bad} not in {list(LAYOUTS)}"
            )
    return shapes


def expand_cases(
    shapes: Sequence[dict[str, Any]],
    layouts: Sequence[str] = (),
    only: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """One case per shape and layout (``--layouts`` filters, ``--only`` picks shape ids)."""
    unknown = [o for o in only if o not in {s["id"] for s in shapes}]
    if unknown:
        raise SystemExit(f"--only: unknown shape ids {unknown}")
    out = []
    for s in shapes:
        if only and s["id"] not in only:
            continue
        for lay in s.get("layouts", ["bshd"]):
            if layouts and lay not in layouts:
                continue
            c = dict(s)
            c["layout"] = lay
            c["case"] = f"{s['id']}/{lay}"
            out.append(c)
    if not out:
        raise SystemExit("no case left after --only / --layouts")
    return out


def layout_plan(layout: str, shape: dict[str, Any]) -> dict[str, Any]:
    """Where Q, K, V, O live: ``{"buffers": {key: numel}, "views": {name: {buffer,
    offset, size, stride}}, "strides": {name: [batch, head, token]}}`` (elements;
    the head-dim stride is 1). Logical tensors are ``[B, S, H, D]`` views of the
    buffers; see the module docstring for the layouts."""
    B, Sq, Sk, Hq, Hkv, D = (
        int(shape[k]) for k in ("batch", "sq", "sk", "hq", "hkv", "d")
    )
    dims = {
        "q": (B, Sq, Hq, D),
        "k": (B, Sk, Hkv, D),
        "v": (B, Sk, Hkv, D),
        "o": (B, Sq, Hq, D),
    }
    buffers: dict[str, int] = {}
    views: dict[str, dict[str, Any]] = {}

    def add(
        name: str, buf: str, offset: int, strides: tuple[int, int, int, int]
    ) -> None:
        views[name] = {
            "buffer": buf,
            "offset": offset,
            "size": list(dims[name]),
            "stride": list(strides),
        }

    def packed(name: str) -> tuple[int, int, int, int]:
        b, s, h, d = dims[name]
        return (s * h * d, h * d, d, 1)

    if layout == "bshd":
        for n, (b, s, h, d) in dims.items():
            buffers[n] = b * s * h * d
            add(n, n, 0, packed(n))
    elif layout == "bhsd":
        for n, (b, s, h, d) in dims.items():
            buffers[n] = b * s * h * d
            add(n, n, 0, (h * s * d, d, s * d, 1))
    elif layout == "packed_qkv":
        if Sq != Sk:
            raise LayoutNotApplicable(f"packed_qkv needs sq == sk (got {Sq}, {Sk})")
        width = Hq + 2 * Hkv
        buffers["qkv"] = B * Sq * width * D
        st = (Sq * width * D, width * D, D, 1)
        add("q", "qkv", 0, st)
        add("k", "qkv", Hq * D, st)
        add("v", "qkv", (Hq + Hkv) * D, st)
        buffers["o"] = B * Sq * Hq * D
        add("o", "o", 0, packed("o"))
    elif layout == "batch_slice":
        for n in ("q", "k", "v"):
            b, s, h, d = dims[n]
            buffers[n] = 2 * b * s * h * d
            add(
                n, n, s * h * d, (2 * s * h * d, h * d, d, 1)
            )  # batch index 1 of 2B, every second batch
        buffers["o"] = B * Sq * Hq * D
        add("o", "o", 0, packed("o"))
    else:
        raise ValueError(f"unknown layout {layout!r}")
    strides = {
        n: [v["stride"][0], v["stride"][2], v["stride"][1]] for n, v in views.items()
    }
    for n, v in views.items():  # every view lies inside its buffer
        last = v["offset"] + sum((s - 1) * st for s, st in zip(v["size"], v["stride"]))
        if last >= buffers[v["buffer"]]:
            raise AssertionError(
                f"layout {layout}: view {n} reaches {last} of {buffers[v['buffer']]}"
            )
    return {"buffers": buffers, "views": views, "strides": strides}


def order_schedule(arms: Sequence[str], rounds: int) -> list[list[str]]:
    """Mirrored, rotating order: round r uses the arms rotated by (r // 2) and
    reversed on odd rounds (ABBA-style pairs). Every 2k rounds each arm takes every
    position equally often (k arms)."""
    k = len(arms)
    out = []
    for r in range(rounds):
        c = (r // 2) % k
        rot = list(arms[c:]) + list(arms[:c])
        out.append(rot[::-1] if r % 2 else rot)
    return out


def round_up_rounds(rounds: int, n_arms: int) -> int:
    """``rounds`` rounded up to a multiple of 2 x arms (the balance period)."""
    cyc = 2 * max(1, n_arms)
    return int(math.ceil(rounds / cyc) * cyc)


def case_seed(seed: int, shape_id: str) -> int:
    """Input seed of a shape (the same data on every layout and in every arm)."""
    return zlib.crc32(f"{seed}:{shape_id}".encode("ascii", "replace")) & 0x7FFFFFFF


def ref_rows_plan(shape: dict[str, Any]) -> tuple[list[tuple[int, int]], list[int]]:
    """(batch, query head) pairs and query rows of the fp64 reference subset."""
    B, Hq, Sq = int(shape["batch"]), int(shape["hq"]), int(shape["sq"])
    pairs = sorted({(0, 0), (B - 1, Hq - 1), (B // 2, Hq // 2)})
    win = min(64, Sq)
    starts = sorted({0, max(0, Sq // 2 - win // 2), Sq - win})
    rows = sorted({r for st in starts for r in range(st, st + win)})
    return pairs, rows


def variant_of(spec, select: str, arch: str) -> tuple[str, str]:
    """``(body, variant)``: body is the stratum group (grid | persistent, and kvseg on gfx942)."""
    kvseg = int(getattr(spec, "kv_segments", 1))
    if getattr(spec, "persistent", False):
        return "persistent", (
            "persist_widedma" if getattr(spec, "wide_lds_dma", False) else "persist"
        )
    if kvseg > 1:
        return ("kvseg" if arch == "gfx942" else "grid"), "grid_kvseg"
    if select == f"{arch}_dense_qbminor":
        return "grid", "grid_qbminor"
    return "grid", "grid"


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def tree_head(root: str) -> str | None:
    """Git head of a tree (best effort; never fetches objects)."""
    try:
        done = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


# --------------------------------------------------------------------------- #
# Worker side: specs from the tree's own factory (CPU, no torch)
# --------------------------------------------------------------------------- #


def make_request(case: dict[str, Any], arch: str, num_cus: int):
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
        batch=int(case["batch"]),
        nhead_q=int(case["hq"]),
        nhead_k=int(case["hkv"]),
        seqlen_q=int(case["sq"]),
        seqlen_k=int(case["sk"]),
        hdim_q=int(case["d"]),
        hdim_v=int(case["d"]),
        arch=arch,
        dtype=str(case["dtype"]),
        mask_type=MASK_TYPES[case["mask"]],
        layout="dense",
        num_cus=num_cus,
    )


def kv_segments_rule(spec, num_cus: int) -> int:
    """The factory's own KV-split rule (``dense_kv_segments`` with the
    ``max_kv_segments_for_extent`` cap, as ``_base_spec`` applies it) on this spec's grid.
    """
    from dispatch.attention.dense_rules import dense_kv_segments
    from kernels.common.attention_dense_kv_split import max_kv_segments_for_extent

    work = (
        -(-int(spec.seqlen_q) // int(spec.block_m))
        * int(spec.batch)
        * int(spec.num_query_heads)
    )
    cap = max_kv_segments_for_extent(
        spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size, max_segments=8
    )
    return int(dense_kv_segments(work, num_cus, max_segments=cap))


def build_spec(case: dict[str, Any], arch: str, num_cus: int):
    """``(report, kernel_spec)`` from the production factory: ``select: auto`` = the
    dispatcher's own choice for a contiguous-layout request; otherwise a pinned spec_id
    (its default spec); then the shape's ``override`` fields via dataclasses.replace
    (``kv_segments: rule`` resolves through :func:`kv_segments_rule`). Never a
    hand-built spec."""
    import dataclasses

    from dispatch.attention import attention_tuning_spec, dispatch_attention

    req = make_request(case, arch, num_cus)
    if case["select"] == "auto":
        res = dispatch_attention(req)
        cand, tspec = res.candidate.name, res.spec
    else:
        tspec = attention_tuning_spec(req, case["select"])
        cand = getattr(tspec, "candidate_name", "") or f"attention_{case['select']}"
    kspec = getattr(tspec, "kernel_spec", tspec)
    want = f"Gfx{arch[3:]}AttentionDenseSpec"
    if type(kspec).__name__ != want:
        raise ValueError(
            f"case {case['case']}: the factory returned {type(kspec).__name__} via {cand}, not a {want}"
        )
    ov = dict(case.get("override") or {})
    if ov.get("kv_segments") == "rule":
        ov["kv_segments"] = kv_segments_rule(kspec, num_cus)
    if ov:
        kspec = dataclasses.replace(kspec, **ov)
    body, variant = variant_of(kspec, case["select"], arch)
    summary = {}
    for k in (
        "block_m",
        "block_n",
        "persistent",
        "wide_lds_dma",
        "kv_segments",
        "num_persistent",
        "causal",
        "causal_bottom_right",
        "ragged",
        "waves_per_eu",
        "nonpersist_decode",
        "persist_decode",
    ):
        if hasattr(kspec, k):
            summary[k] = getattr(kspec, k)
    exp = case.get("expect") or {}
    got = {
        "candidate": cand,
        "body": body,
        "variant": variant,
        "block_m": kspec.block_m,
        "block_n": kspec.block_n,
        "kv_segments": int(getattr(kspec, "kv_segments", 1)),
    }
    report = {
        "candidate": cand,
        "tuning_id": str(getattr(tspec, "tuning_id", "")),
        "kernel_name": kspec.kernel_name(),
        "body": body,
        "variant": variant,
        "summary": _jsonable(summary),
        "override": _jsonable(
            {k: getattr(kspec, k) for k in (case.get("override") or {})}
        )
        or None,
        "expect_mismatch": [
            f"{k}: expected {exp[k]!r}, factory gives {got[k]!r}"
            for k in exp
            if k in got and exp[k] != got[k]
        ],
    }
    return report, kspec


def spec_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {k: report[k] for k in ("candidate", "kernel_name", "tuning_id", "summary")}


def tuning_ids_equal(idents: dict[str, dict[str, Any]]) -> bool:
    """Whether every arm reports the same tuning id. Informational: the id hashes the
    kernel ABI version, so arms on either side of an ABI change differ here by design
    while building the same kernel."""
    return len({v["tuning_id"] for v in idents.values()}) <= 1


def identities_equal(idents: dict[str, dict[str, Any]]) -> bool:
    """Equal candidate and kernel name, and equal spec fields on the field names every
    arm's spec has (the tuning id is compared by :func:`tuning_ids_equal`, not here)."""
    vals = list(idents.values())
    if len(vals) < 2:
        return True
    keys = set(vals[0]["summary"])
    for v in vals[1:]:
        keys &= set(v["summary"])
    ref = vals[0]
    for v in vals[1:]:
        if any(v[k] != ref[k] for k in ("candidate", "kernel_name")):
            return False
        if any(v["summary"][k] != ref["summary"][k] for k in keys):
            return False
    return True


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Export: what one arm tree's own launch path does for each case
# --------------------------------------------------------------------------- #

ELEM_BYTES = 2  # fp16 / bf16
# Sentinel pointers: addresses no allocation has (above the 48-bit address space), one
# base per backing buffer, so every pointer slot of a recorded kernarg is recognizable.
SENTINEL_BASE = 0x7000_0000_0000_0000
SENTINEL_SPAN = 1 << 36


def sentinel_bases(buffers: Sequence[str]) -> dict[str, int]:
    return {k: SENTINEL_BASE + ((i + 1) << 40) for i, k in enumerate(sorted(buffers))}


class SentinelView:
    """What the runner reads of a tensor (``shape``, ``stride()``, ``data_ptr()``,
    ``device``), at a sentinel address: nothing is behind it."""

    is_cuda = True

    def __init__(self, base: int, offset: int, size, stride, device=None) -> None:
        self._ptr = base + ELEM_BYTES * offset
        self.shape = tuple(size)
        self._stride = tuple(stride)
        self.device = device

    def data_ptr(self) -> int:
        return self._ptr

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]

    def is_contiguous(self) -> bool:
        b, s, h, d = self.shape
        return self._stride == (s * h * d, h * d, d, 1)


def kernarg_layout(signature: Sequence[dict[str, str]]) -> list[tuple[str, str, int]]:
    """``(name, type, offset)`` of every kernel argument, at its natural alignment
    (the layout ``rocke.runtime.packing`` packs)."""
    from rocke.runtime.packing import SCALAR_ARG_FORMATS

    out, off = [], 0
    for arg in signature:
        ty = str(arg["type"])
        size, align = (8, 8) if ty.startswith("ptr<") else SCALAR_ARG_FORMATS[ty][1:]
        off += (-off) % align
        out.append((str(arg["name"]), ty, off))
        off += size
    return out


def locate_pointer_slots(
    signature: Sequence[dict[str, str]],
    values: dict[str, Any],
    kernarg: bytes,
    bases: dict[str, int],
    workspace: dict[int, int],
) -> list[dict[str, Any]]:
    """The pointer slots of one launch's kernarg bytes: ``{"offset", "buffer",
    "byte_offset"}`` for a sentinel (buffer ``key`` plus a byte offset) and
    ``{"offset", "workspace": group}`` for any other address (a buffer the runner
    allocated itself; equal addresses share a group, numbered through ``workspace``,
    which is shared by all launches of a case). Asserts that each slot holds the
    address the runner was given and that no sentinel sits anywhere else."""
    from rocke.runtime.packing import device_pointer

    slots, taken = [], set()
    for name, ty, off in kernarg_layout(signature):
        if not ty.startswith("ptr<"):
            continue
        ptr = device_pointer(values[name])
        if struct.unpack_from("<Q", kernarg, off)[0] != ptr:
            raise AssertionError(f"kernarg {name} at {off} is not the address given")
        taken.add(off)
        for key, base in bases.items():
            if base <= ptr < base + SENTINEL_SPAN:
                slots.append({"offset": off, "buffer": key, "byte_offset": ptr - base})
                break
        else:
            group = workspace.setdefault(ptr, len(workspace))
            slots.append({"offset": off, "workspace": group})
    for off in range(0, len(kernarg) - 7, 8):
        word = struct.unpack_from("<Q", kernarg, off)[0]
        if off not in taken and any(
            b <= word < b + SENTINEL_SPAN for b in bases.values()
        ):
            raise AssertionError(f"a sentinel address sits at kernarg offset {off}")
    return slots


def patch_kernarg(
    kernarg: bytes,
    slots: Sequence[dict[str, Any]],
    pointers: dict[str, int],
    workspace: Sequence[int],
) -> bytes:
    """``kernarg`` with every pointer slot set to a real address: ``pointers[buffer]``
    plus the byte offset, or the workspace buffer of its group."""
    out = bytearray(kernarg)
    for s in slots:
        if "buffer" in s:
            ptr = pointers[s["buffer"]] + s["byte_offset"]
        else:
            ptr = workspace[s["workspace"]]
        struct.pack_into("<Q", out, s["offset"], ptr)
    return bytes(out)


class _WorkspaceAddresses:
    """A ``workspace_allocator`` that hands out distinct made-up addresses."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, shape):
        self.count += 1
        return types.SimpleNamespace(
            data_ptr=lambda n=self.count: 0x6000_0000_0000_0000 + (n << 32)
        )


class _Capture:
    """Stands in for the HIP runtime while a tree's runner runs: ``load_module`` keeps
    the code object, every launch is recorded (kernel name, code object, grid, block,
    kernarg bytes, pointer slots) and nothing is launched."""

    def __init__(self, bases: dict[str, int]) -> None:
        self.bases = bases
        self.launches: list[dict[str, Any]] = []
        self.workspace: dict[int, int] = {}
        self._call: dict[str, Any] = {}

    def load_module(self, blob):
        class Module:
            def get_function(self, name):
                return (bytes(blob), name)

        return Module()

    def launch_blocking(self, fn, grid, block, args, *, shared_bytes=0, stream=0, **_):
        blob, name = fn
        kernarg = bytes(args)
        slots = locate_pointer_slots(
            self._call["signature"],
            self._call["values"],
            kernarg,
            self.bases,
            self.workspace,
        )
        self.launches.append(
            {
                "kernel": name,
                "hsaco": base64.b64encode(blob).decode("ascii"),
                "grid": list(grid),
                "block": list(block),
                "shared": int(shared_bytes),
                "kernarg": base64.b64encode(kernarg).decode("ascii"),
                "slots": slots,
            }
        )

    def launch(self, fn, grid, block, args, *, shared_bytes=0, stream=0, **_):
        self.launch_blocking(fn, grid, block, args, shared_bytes=shared_bytes)

    def retain_for_stream(self, *_):
        pass


@contextlib.contextmanager
def capturing(bases: dict[str, int], dry_run: bool):
    """Route every kernel launch of the runner into a :class:`_Capture` (and with
    ``dry_run`` replace the compiler by a stand-in that returns a placeholder code
    object), restoring everything on exit."""
    import rocke.helpers.compile as rcompile
    import rocke.runtime.launcher as rlauncher

    cap = _Capture(bases)
    real_call = rlauncher.KernelLauncher.__call__
    real_runtime = rlauncher._runtime
    real_compile = rcompile.compile_kernel

    def call(self, values, *, config):
        cap._call = {"signature": self._signature, "values": dict(values)}
        return real_call(self, values, config=config)

    def stand_in(kernel, **_):
        return types.SimpleNamespace(
            hsaco=b"dry-run code object " + kernel.name.encode(),
            kernel_name=kernel.name,
        )

    rlauncher.KernelLauncher.__call__ = call
    rlauncher._runtime = lambda: cap
    if dry_run:
        rcompile.compile_kernel = stand_in
    try:
        yield cap
    finally:
        rlauncher.KernelLauncher.__call__ = real_call
        rlauncher._runtime = real_runtime
        rcompile.compile_kernel = real_compile


def export_case(case: dict[str, Any], args, kmod) -> dict[str, Any]:
    """One case through this tree: its factory spec, then its own runner on views at
    sentinel addresses. Returns the export record (status ok with the recorded
    launches, or refused with the runner's reason)."""
    report, spec = build_spec(case, args.arch, args.num_cus or ARCH_CUS[args.arch])
    out: dict[str, Any] = {"case": case["case"], "report": report, "launches": []}
    ok, why = kmod.supports_attention_dense(spec, arch=args.arch)
    report["supports"] = [bool(ok), str(why)]
    if args.list:
        out["status"] = "listed"
        return out
    if not ok:
        return dict(out, status="refused", reason=f"supports_attention_dense: {why}")
    plan = layout_plan(case["layout"], case)
    bases = sentinel_bases(plan["buffers"])
    runner = kmod.run_attention_dense_torch
    extra: dict[str, Any] = {}
    device = None
    if int(getattr(spec, "kv_segments", 1)) > 1:
        if "workspace_allocator" in inspect.signature(runner).parameters:
            extra["workspace_allocator"] = (
                _WorkspaceAddresses()
            )  # no torch, nothing allocated
        elif "torch" in sys.modules:
            device = "cpu"  # an older runner allocates the workspace with torch
    views = {
        n: SentinelView(bases[v["buffer"]], v["offset"], v["size"], v["stride"], device)
        for n, v in plan["views"].items()
    }
    with capturing(bases, args.dry_run) as cap:
        try:
            runner(
                spec=spec,
                q=views["q"],
                k=views["k"],
                v=views["v"],
                out=views["o"],
                scale=1.0 / math.sqrt(float(case["d"])),
                stream=1,
                arch=args.arch,
                **extra,
            )
        except (ValueError, NotImplementedError) as exc:
            return dict(out, status="refused", reason=f"{type(exc).__name__}: {exc}")
    wanted = {(v["buffer"], ELEM_BYTES * v["offset"]) for v in plan["views"].values()}
    found = {
        (s["buffer"], s["byte_offset"])
        for launch in cap.launches
        for s in launch["slots"]
        if "buffer" in s
    }
    if found != wanted:
        raise AssertionError(
            f"{case['case']}: the launches address {sorted(found)}, the views are {sorted(wanted)}"
        )
    return dict(
        out,
        status="ok",
        launches=cap.launches,
        workspace_groups=len(cap.workspace),
        strides=plan["strides"],
    )


def export_arm(args) -> int:
    """``export``: every case of the shape file through THIS process's tree (its own
    ``rocke``, ``kernels`` and ``dispatch`` on ``sys.path``, one compiler), written to
    ``--out`` as JSON-flow YAML."""
    cases = expand_cases(
        load_shapes(args.shapes),
        [x for x in args.layouts.split(",") if x],
        [x for x in args.only.split(",") if x],
    )
    if not (args.dry_run or args.list):
        try:  # torch first: the compiler is then the one the torch wheel bundles
            from rocke.runtime.comgr import prefer_bundled_lib

            prefer_bundled_lib()
        except ImportError:
            pass
    kmod = importlib.import_module(f"kernels.{args.arch}.attention_dense")
    records = []
    for c in cases:
        try:
            records.append(export_case(c, args, kmod))
        except LayoutNotApplicable as exc:
            records.append(
                {"case": c["case"], "status": "not_applicable", "reason": str(exc)}
            )
    compiler = None
    if not (args.dry_run or args.list):
        from rocke.runtime.comgr import loaded_compiler_info

        info = loaded_compiler_info()
        compiler = _jsonable(dataclasses.asdict(info)) if info else None
    import rocke

    root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(rocke.__file__)))
        )
    )
    doc = {
        "meta": {
            "script_version": SCRIPT_VERSION,
            "tree": root,
            "head": tree_head(root),
            "arch": args.arch,
            "num_cus": args.num_cus or ARCH_CUS[args.arch],
            "python": sys.version.split()[0],
            "compiler": compiler,
            "torch_imported": "torch" in sys.modules,
            "env": {
                k: os.environ.get(k)
                for k in ("ROCKE_LLVM_FLAVOR", "ROCKE_BACKEND", "ROCKE_COMGR_LIB")
            },
        },
        "cases": records,
    }
    write_result(args.out, doc)
    return 0


def load_export(path: str) -> dict[str, Any]:
    doc = load_stats().load_structured(path)
    doc["by_case"] = {r["case"]: r for r in doc["cases"]}
    return doc


def run_export(
    label: str, root: str, args, out: str, extra: Sequence[str] = ()
) -> dict[str, Any]:
    """One ``export`` subprocess on the tree at ``root``; the exported document."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(root, "platform", "python"), os.path.join(root, "library")]
    )
    env.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "export",
        "--arch",
        args.arch,
        "--shapes",
        args.shapes,
        "--layouts",
        args.layouts,
        "--only",
        args.only,
        "--num-cus",
        str(args.num_cus),
        "--out",
        out,
        *extra,
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    done = subprocess.run(cmd, env=env, stdout=sys.stderr, check=False)
    if done.returncode:
        raise SystemExit(
            f"arm {label}: the export of {root} failed (exit {done.returncode})"
        )
    return load_export(out)


# --------------------------------------------------------------------------- #
# Time: every arm in THIS process, on the same device buffers
# --------------------------------------------------------------------------- #


class Unit:
    """One arm's launches of a case, loaded into this process: one module per launch
    (the control arm loads the baseline's code objects a second time, so its modules
    are distinct), pointer slots patched to this case's buffers."""

    def __init__(self, calls, modules) -> None:
        self.calls, self.modules, self.graph = calls, modules, None

    def run(self) -> None:
        for call in self.calls:
            call()


class Bench:
    """The timing process: one set of buffers per case, the fp64 reference subset, the
    arms' units, the HIP-graph replay. ``dry`` replaces the device by synthetic times
    and stubbed checks (the kernarg patching still runs, on made-up addresses)."""

    def __init__(self, args) -> None:
        self.args, self.dry = args, bool(args.dry_run)
        self.effects = {
            k: float(v[-1]) for k, v in _parse_label_map(args.dry_run_effect).items()
        }
        self.refuse = _parse_label_map(args.dry_run_refuse)
        self.wrong = _parse_label_map(args.dry_run_wrong)
        self.rng = random.Random(f"{args.seed}")
        self.torch = None
        if self.dry:
            return
        import torch

        try:  # torch first: HIP is then the one the torch wheel bundles
            from rocke.runtime.comgr import prefer_bundled_lib

            prefer_bundled_lib()
        except ImportError:
            pass
        from rocke.runtime.hip_module import Runtime

        if not torch.cuda.is_available():
            raise SystemExit("no HIP device visible to torch")
        self.torch, self.rt = torch, Runtime()
        self.device = torch.device("cuda", 0)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(
            self.stream
        )  # every torch operation and graph on ONE stream
        self.handle = int(self.stream.cuda_stream)

    # ---- device description ---------------------------------------------------- #

    def info(self) -> dict[str, Any]:
        if self.dry:
            return {"device_name": "dry-run (CPU)", "cu_count": None, "gcn_arch": None}
        props = self.torch.cuda.get_device_properties(0)
        return {
            "device_name": self.torch.cuda.get_device_name(0),
            "cu_count": int(props.multi_processor_count),
            "gcn_arch": getattr(props, "gcnArchName", None),
            "torch_version": self.torch.__version__,
            "hip_version": getattr(self.torch.version, "hip", None),
        }

    # ---- one case ------------------------------------------------------------- #

    def prepare(
        self, case: dict[str, Any], plan: dict[str, Any], groups: int, ws_bytes: int
    ):
        """Buffers (inputs from the fixed seed, decoy data around the views, NaN in O),
        the segment workspace and the fp64 reference subset of ``case``."""
        self.case, self.plan = case, plan
        self.scale = 1.0 / math.sqrt(float(case["d"]))
        if self.dry:
            self.pointers = {
                k: 0x1000_0000 * (i + 1) for i, k in enumerate(sorted(plan["buffers"]))
            }
            self.workspace = [0x9000_0000 + 0x100_0000 * g for g in range(groups)]
            return
        torch = self.torch
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[case["dtype"]]
        seed = case_seed(self.args.seed, case["id"])
        gen = torch.Generator(device="cpu").manual_seed(seed)
        gdev = torch.Generator(device=self.device).manual_seed(seed)
        bufs = {}
        for key, numel in plan["buffers"].items():
            buf = torch.empty(numel, dtype=dt, device=self.device)
            if key == "o":
                buf.fill_(float("nan"))
            else:
                buf.normal_(generator=gdev)
            bufs[key] = buf
        views = {
            n: bufs[v["buffer"]].as_strided(v["size"], v["stride"], v["offset"])
            for n, v in plan["views"].items()
        }
        for n in ("q", "k", "v"):
            data = torch.randn(
                *plan["views"][n]["size"], generator=gen, dtype=torch.float32
            )
            views[n].copy_(data.to(self.device, dt))
        self.bufs, self.views = bufs, views
        self.pointers = {k: int(b.data_ptr()) for k, b in bufs.items()}
        self.ws = [
            torch.empty(ws_bytes, dtype=torch.uint8, device=self.device)
            for _ in range(groups)
        ]
        self.workspace = [int(w.data_ptr()) for w in self.ws]
        self.ref = self._ref_subset()

    def load(self, launches: Sequence[dict[str, Any]]) -> Unit:
        """An arm's launches of the case as a :class:`Unit` (kernargs patched to this
        case's buffers; in a dry run nothing is loaded)."""
        calls, modules = [], []
        for ln in launches:
            kernarg = patch_kernarg(
                base64.b64decode(ln["kernarg"]),
                ln["slots"],
                self.pointers,
                self.workspace,
            )
            if self.dry:
                modules.append(kernarg)
                continue
            module = self.rt.load_module(base64.b64decode(ln["hsaco"]))
            fn = module.get_function(ln["kernel"])
            calls.append(
                self.rt.prepare_launch(
                    fn,
                    tuple(ln["grid"]),
                    tuple(ln["block"]),
                    kernarg,
                    shared_bytes=ln["shared"],
                    stream=self.handle,
                )
            )
            modules.append(module)
        return Unit(calls, modules)

    def release(self, units: Sequence[Unit]) -> None:
        if self.dry:
            return
        self.stream.synchronize()
        for u in units:  # the modules stay loaded: a graph may still name their kernels
            u.graph = None
        self.bufs = self.views = self.ws = self.ref = None
        self.torch.cuda.empty_cache()

    # ---- numeric gate ------------------------------------------------------------ #

    def _ref_subset(self) -> dict[tuple[int, int], Any]:
        """fp64 attention for the subset rows: {(b, h): [rows, D] float64}."""
        torch, case = self.torch, self.case
        pairs, rows = ref_rows_plan(case)
        Hq, Hkv, Sq, Sk = (int(case[k]) for k in ("hq", "hkv", "sq", "sk"))
        group = Hq // Hkv
        r = torch.tensor(rows, device=self.device)
        out = {}
        for b, h in pairs:
            qr = self.views["q"][b, r, h, :].double()
            kk = self.views["k"][b, :, h // group, :].double()
            vv = self.views["v"][b, :, h // group, :].double()
            s = (qr @ kk.T) * float(self.scale)
            if case["mask"] != "none":
                off = Sk - Sq if case["mask"] == "causal_br" else 0
                keys = torch.arange(Sk, device=self.device)
                s = s.masked_fill(keys[None, :] > (r[:, None] + off), float("-inf"))
            out[(b, h)] = torch.softmax(s, dim=-1) @ vv
        return out

    def check(self, label: str, unit: Unit) -> dict[str, Any]:
        """Run the arm once into a NaN-filled O and compare with the reference subset."""
        tol = TOL[self.case["dtype"]]
        if self.dry:
            bad = _matches(self.wrong.get(label, []), self.case)
            digest = hashlib.sha256(
                f"dry:{self.case['case']}:{label if bad else ''}".encode()
            ).hexdigest()
            return {
                "max_abs": 1.0 if bad else 0.0,
                "tol": tol,
                "rows": 0,
                "heads": 0,
                "nan_in_output": False,
                "pass": not bad,
                "o_hash": digest,
            }
        torch = self.torch
        o = self.views["o"]
        o.fill_(float("nan"))
        unit.run()
        self.stream.synchronize()
        _, rows = ref_rows_plan(self.case)
        r = torch.tensor(rows, device=self.device)
        worst = 0.0
        for (b, h), want in self.ref.items():
            err = (o[b, r, h, :].double() - want).abs().max().item()
            worst = max(worst, err if math.isfinite(err) else float("inf"))
        nan = bool(torch.isnan(o).any().item())
        flat = o.contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        return {
            "max_abs": worst,
            "tol": tol,
            "rows": len(rows),
            "heads": len(self.ref),
            "nan_in_output": nan,
            "pass": (not nan) and worst <= tol,
            "o_hash": hashlib.sha256(flat.tobytes()).hexdigest(),
        }

    # ---- timing ----------------------------------------------------------------- #

    def launches_for(self, unit: Unit) -> int:
        """How many repeats of the arm's launches fill ``--target-ms`` of one graph."""
        if self.dry:
            return 1
        torch = self.torch
        unit.run()  # warm
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        t0.record(self.stream)
        for _ in range(5):
            unit.run()
        t1.record(self.stream)
        t1.synchronize()
        ms = t0.elapsed_time(t1) / 5
        a = self.args
        return int(
            min(
                a.max_launches,
                max(a.min_launches, math.ceil(a.target_ms / max(ms, 1e-6))),
            )
        )

    def capture(self, unit: Unit, n: int) -> None:
        if self.dry:
            return
        torch = self.torch
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.stream):
            for _ in range(n):
                unit.run()
        unit.graph = g
        g.replay()
        self.stream.synchronize()

    def measure(self, label: str, unit: Unit, n: int) -> float:
        """Microseconds per launch of one graph replay (after an untimed one)."""
        if self.dry:
            return self._dry_time(label)
        torch = self.torch
        unit.graph.replay()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        t0.record(self.stream)
        unit.graph.replay()
        t1.record(self.stream)
        t1.synchronize()
        return t0.elapsed_time(t1) * 1000.0 / n

    def _dry_time(self, label: str) -> float:
        """Synthetic microseconds per launch: a per-shape base, the arm's injected
        effect and per-measurement log-normal noise."""
        c = self.case
        base = (
            50.0 + math.prod(int(c[k]) for k in ("sq", "sk", "hq", "d", "batch")) / 4e7
        )
        t = base * (1.0 + self.effects.get(label, 0.0))
        return t * math.exp(self.rng.gauss(0.0, 0.003))


def _parse_label_map(items: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for it in items or ():
        label, sep, val = it.partition("=")
        if not sep:
            raise SystemExit(f"{it!r}: expected LABEL=VALUE")
        out.setdefault(label, []).extend(x for x in val.split(",") if x)
    return out


def _matches(items: Sequence[str], case: dict[str, Any]) -> bool:
    return case["layout"] in items or case["case"] in items


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_list(exports: dict[str, dict[str, Any]], arms, cases) -> int:
    """Print every case's factory choice in every arm tree: the dispatcher choice
    against ``expect``, whether the arm's runner takes the spec, spec identity."""
    bad = 0
    for label, root in arms:
        print(f"arm {label}: {root}")
    for c in cases:
        reports = {
            l: exports[l]["by_case"][c["case"]]["report"]
            for l, _ in arms
            if "report" in exports[l]["by_case"][c["case"]]
        }
        head = (
            f"{c['case']:<18} {c.get('model', ''):<22} B{c['batch']} Hq{c['hq']}/Hkv{c['hkv']} D{c['d']} "
            f"Sq{c['sq']} Sk{c['sk']} {c['mask']} {c['dtype']} select={c['select']}"
        )
        print(head)
        if not reports:
            continue
        first = next(iter(reports.values()))
        s = first["summary"]
        print(
            f"     -> {first['candidate']} body={first['body']} variant={first['variant']} bm={s.get('block_m')} "
            f"bn={s.get('block_n')} kvseg={s.get('kv_segments')} tuning_id={first['tuning_id']}"
        )
        if first.get("override"):
            print(
                f"     override (dataclasses.replace after the factory): {first['override']}"
            )
        for label, rep in reports.items():
            for d in rep["expect_mismatch"]:
                print(f"     EXPECT MISMATCH ({label}) {d}")
                bad += 1
            if not rep["supports"][0]:
                print(f"     arm {label} REFUSES the spec: {rep['supports'][1]}")
        idents = {l: spec_identity(r) for l, r in reports.items()}
        equal = identities_equal(idents)
        print(
            "     kernel: "
            + "  ".join(f"{l}={r['kernel_name']}" for l, r in reports.items())
            + f"   spec identity across arms: {'equal' if equal else 'DIFFERS'}"
            + f"   tuning id: {'equal' if tuning_ids_equal(idents) else 'differs (informational)'}"
        )
        bad += 0 if equal else 1
    for label, _ in arms:  # --list must not have touched torch
        if exports[label]["meta"]["torch_imported"]:
            print(f"arm {label}: the export imported torch (it must not for --list)")
            bad += 1
    print(
        f"\n{len(cases)} cases on layouts {sorted({c['layout'] for c in cases})}; expectation mismatches/errors/identity differences: {bad}"
    )
    return 1 if bad else 0


def run_case(
    bench: Bench,
    exports: dict[str, dict[str, Any]],
    source: dict[str, str],
    labels: list[str],
    baseline: str,
    case: dict[str, Any],
    args,
) -> dict[str, Any]:
    """One case: every arm's recorded launches loaded into this process, the numeric
    gate, ``--rounds`` rounds in the mirrored order on one set of buffers, the gate
    again. ``source[label]`` names the export an arm's launches come from (the control
    reads the baseline's: its code objects are loaded a second time)."""
    stats = load_stats()
    rec: dict[str, Any] = {
        "case": case["case"],
        "shape_id": case["id"],
        "layout": case["layout"],
        "dtype": case["dtype"],
        "shape": {k: v for k, v in case.items() if k not in ("expect",)},
        "status": "ok",
        "arms": {},
        "measurements": [],
        "check_pre": {},
        "check_post": {},
    }
    t0 = time.time()
    try:
        plan = layout_plan(case["layout"], case)
    except LayoutNotApplicable as exc:
        rec.update(status="not_applicable", reason=str(exc))
        return rec
    rec["strides"] = plan["strides"]
    ex = {l: exports[source[l]]["by_case"][case["case"]] for l in labels}
    for l in labels:
        e, st, why = ex[l], ex[l]["status"], ex[l].get("reason")
        if st == "ok" and _matches(bench.refuse.get(l, []), case):
            st, why = "refused", "ValueError: dry-run refusal hook"
        rep = e.get("report", {})
        rec["arms"][l] = {
            "status": st,
            "reason": why,
            "candidate": rep.get("candidate"),
            "kernel_name": rep.get("kernel_name"),
            "tuning_id": rep.get("tuning_id"),
            "summary": rep.get("summary"),
            "compiler": exports[source[l]]["meta"]["compiler"],
            "strides": e.get("strides"),
        }
        if st == "ok":
            blobs = b"".join(base64.b64decode(x["hsaco"]) for x in e["launches"])
            rec["arms"][l]["code_sha256"] = hashlib.sha256(blobs).hexdigest()[:16]
    ref = ex[baseline].get("report")
    if ref is not None:
        rec.update(
            candidate=ref["candidate"],
            tuning_id=ref["tuning_id"],
            body=ref["body"],
            variant=ref["variant"],
            override=ref.get("override"),
            expect_mismatch=ref["expect_mismatch"],
            launch_unit=(
                "runner call (main + segment combine)"
                if int(ref["summary"].get("kv_segments", 1)) > 1
                else "runner call"
            ),
        )
    idents = {l: spec_identity(ex[l]["report"]) for l in labels if "report" in ex[l]}
    rec["spec_identity_equal"] = identities_equal(idents)
    rec["tuning_id_equal"] = tuning_ids_equal(idents)
    if rec["arms"][baseline]["status"] != "ok":
        rec["status"] = (
            "error" if rec["arms"][baseline]["status"] == "error" else "refused"
        )
        return rec
    live = [l for l in labels if rec["arms"][l]["status"] == "ok"]
    groups = max(ex[l].get("workspace_groups", 0) for l in live)
    kvs = int(ref["summary"].get("kv_segments", 1))
    ws_bytes = (
        int(case["batch"])
        * int(case["sq"])
        * int(case["hq"])
        * kvs
        * int(case["d"])
        * 4
    )
    bench.prepare(case, plan, groups, ws_bytes)
    units: dict[str, Unit] = {}
    try:
        for l in live:
            units[l] = bench.load(ex[l]["launches"])
            rec["check_pre"][l] = bench.check(l, units[l])
        n = bench.launches_for(units[baseline])
        for l in live:
            bench.capture(units[l], n)
        rounds = round_up_rounds(args.rounds, len(live))
        rec["rounds"] = rounds
        rec["n_launches_per_measurement"] = n
        schedule = order_schedule(live, rounds)
        rec["order"] = [",".join(o) for o in schedule]
        for r, orderr in enumerate(schedule):
            for pos, l in enumerate(orderr):
                rec["measurements"].append(
                    {
                        "round": r,
                        "pos": pos,
                        "arm": l,
                        "us": float(bench.measure(l, units[l], n)),
                    }
                )
        for l in live:
            rec["check_post"][l] = bench.check(l, units[l])
    finally:
        bench.release(list(units.values()))
        rec["wall_s"] = round(time.time() - t0, 2)
    for phase in ("check_pre", "check_post"):
        hashes = {l: v.get("o_hash") for l, v in rec[phase].items()}
        rec.setdefault("o_hash_equal", {})[phase[6:]] = len(set(hashes.values())) <= 1
    if stats.failed_check_arms(rec):
        rec["status"] = "check_failed"
    elif not rec["spec_identity_equal"]:
        rec["status"] = "spec_mismatch"
    return rec


def write_result(path: str, result: dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(load_stats().dump_flow_yaml(result))
    os.replace(tmp, path)


def resolve_num_cus(args) -> int:
    """The CU count every arm's factory requests are built for: ``--num-cus`` when
    given; otherwise the device's own count, so a run times the specs production
    would pick on that part; on ``--dry-run`` or ``--list`` (no device) the arch's
    usual count from ``ARCH_CUS``."""
    if args.num_cus:
        return int(args.num_cus)
    if args.dry_run or args.list:
        return ARCH_CUS[args.arch]
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no HIP device visible to torch (or pass --dry-run / --list)")
    return int(torch.cuda.get_device_properties(0).multi_processor_count)


def run_driver(args, arms: list[tuple[str, str]], baseline: str, control: str) -> int:
    args.num_cus = resolve_num_cus(args)
    stats = load_stats()
    cases = expand_cases(
        load_shapes(args.shapes),
        [x for x in args.layouts.split(",") if x],
        [x for x in args.only.split(",") if x],
    )
    workdir = (
        args.out
        and os.path.dirname(os.path.abspath(args.out))
        or tempfile.mkdtemp(prefix="ab_export_")
    )
    exports: dict[str, dict[str, Any]] = {}
    for label, path in arms:
        if os.path.isfile(path):
            exports[label] = load_export(path)
        else:
            exports[label] = run_export(
                label,
                path,
                args,
                os.path.join(workdir, f"{label}.export.yaml"),
                ["--list"] if args.list else [],
            )
    for label, doc in exports.items():
        built_for = (doc.get("meta") or {}).get("num_cus")
        if built_for is None:
            raise SystemExit(
                f"arm {label}: its export does not record the CU count its factory "
                "requests were built for (an export older than this check); export it "
                f"again with --num-cus {args.num_cus}"
            )
        if int(built_for) != args.num_cus:
            raise SystemExit(
                f"arm {label}: its export was built for {built_for} CUs, this run "
                f"for {args.num_cus}; export it again with --num-cus {args.num_cus} "
                "(an export never asks the device: without --num-cus it takes the "
                "arch's ARCH_CUS entry)"
            )
    if args.list:
        return run_list(exports, arms, cases)
    labels = [l for l, _ in arms] + [control]
    source = {l: l for l, _ in arms}
    source[control] = baseline
    bench = Bench(args)
    info = bench.info()
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    meta: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "argv": sys.argv,
        "label": args.label,
        "arch": args.arch,
        "started_utc": now(),
        "host": platform.node(),
        "baseline": baseline,
        "control": control,
        "arm_order": labels,
        "rounds_requested": args.rounds,
        "target_ms": args.target_ms,
        "seed": args.seed,
        "timing_method": (
            "dry-run (synthetic)" if args.dry_run else "graph, one process"
        ),
        "spec_num_cus": args.num_cus,
        "shapes_file": os.path.abspath(args.shapes),
        "layouts": sorted({c["layout"] for c in cases}),
        **{k: info[k] for k in ("device_name", "cu_count", "gcn_arch")},
        "arms": {
            l: dict(exports[source[l]]["meta"], again=(l == control)) for l in labels
        },
    }
    if not args.dry_run and meta["cu_count"] != meta["spec_num_cus"]:
        # Only an explicit --num-cus gets here (the default is the device's count):
        # every arm times the same spec, but not the one production picks on this part.
        print(
            f"WARNING: --num-cus {meta['spec_num_cus']} on a device of {meta['cu_count']} CUs: "
            "the factory requests are not the ones production builds here"
        )
    print("AB_META " + json.dumps(meta, default=str), flush=True)
    result: dict[str, Any] = {"meta": meta, "cases": []}
    rc = 0
    try:
        for idx, c in enumerate(cases):
            t0 = time.time()
            rec = run_case(bench, exports, source, labels, baseline, c, args)
            result["cases"].append(rec)
            rc |= rec["status"] in ("error", "check_failed", "spec_mismatch")
            print("AB_ROW " + json.dumps(rec, default=str), flush=True)
            arms_txt = " ".join(f"{l}:{a['status']}" for l, a in rec["arms"].items())
            print(
                f"[{idx + 1}/{len(cases)}] {c['case']} {rec.get('variant', '?')} status={rec['status']} "
                f"N={rec.get('n_launches_per_measurement')} {arms_txt} ({time.time() - t0:.1f}s)",
                flush=True,
            )
            write_result(args.out, result)
    finally:
        meta["finished_utc"] = now()
        write_result(args.out, result)
    print(
        "AB_DONE "
        + json.dumps({"finished_utc": meta["finished_utc"], "exit_code": int(rc)}),
        flush=True,
    )
    if not args.no_stats:
        an = stats.analyze(result, n_boot=args.n_boot, seed=args.seed)
        text = stats.to_markdown(an) + "\n" + "\n".join(stats.bar_check(an)["lines"])
        print("\n" + text, flush=True)
        if args.out:
            with open(
                os.path.splitext(args.out)[0] + ".stats.md", "w", encoding="utf-8"
            ) as fh:
                fh.write(text + "\n")
    return int(rc)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arch", choices=sorted(ARCH_CUS), default="gfx950")
    ap.add_argument(
        "--shapes",
        default="",
        help="shape file (YAML or JSON; schema in the shipped shape file)",
    )
    ap.add_argument(
        "--layouts",
        default="",
        help="comma-separated: filter the layouts of every shape",
    )
    ap.add_argument("--only", default="", help="comma-separated shape ids")
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="LABEL=TREE|EXPORT",
        help="a rocKE tree (exported in a subprocess) or an export file; repeat for more arms",
    )
    ap.add_argument(
        "--baseline",
        default="",
        help="arm every ratio is against (default: the first --arm)",
    )
    ap.add_argument(
        "--control",
        default="",
        help="label of the A/A' control: the baseline's code objects loaded a second time (required for a run)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=MIN_ROUNDS,
        help=f"rounds per case (>= {MIN_ROUNDS}; rounded up to a multiple of 2 x live arms)",
    )
    ap.add_argument(
        "--target-ms", type=float, default=25.0, help="device time of one graph replay"
    )
    ap.add_argument("--min-launches", type=int, default=3)
    ap.add_argument("--max-launches", type=int, default=2000)
    ap.add_argument(
        "--num-cus",
        type=int,
        default=0,
        help="CU count of the factory request (default: the device's own count; with "
        "--dry-run, --list or a bare export, the arch's entry of ARCH_CUS: "
        + ", ".join(f"{a} {n}" for a, n in ARCH_CUS.items())
        + ")",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--label", default="", help="free text recorded in the result")
    ap.add_argument(
        "--out",
        default="",
        help="result FILE.yaml (JSON-flow YAML); FILE.stats.yaml / .stats.md next to it; the arms' export files are written beside it",
    )
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument(
        "--list",
        action="store_true",
        help="CPU: specs through every arm tree's factory, no torch, nothing compiled",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU: stand-in compiler, synthetic times, stubbed checks (export and kernarg patching are real)",
    )
    ap.add_argument(
        "--dry-run-effect", action="append", default=[], metavar="LABEL=FRACTION"
    )
    ap.add_argument(
        "--dry-run-refuse",
        action="append",
        default=[],
        metavar="LABEL=LAYOUT|CASE[,..]",
    )
    ap.add_argument(
        "--dry-run-wrong", action="append", default=[], metavar="LABEL=LAYOUT|CASE[,..]"
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bar":
        return load_stats().main(argv[1:])
    mode = "run"
    if argv and argv[0] == "export":
        mode, argv = "export", argv[1:]
    args = build_parser().parse_args(argv)
    if args.num_cus < 0:
        raise SystemExit("--num-cus must be a positive CU count (0: the default)")
    if not args.shapes:
        raise SystemExit("--shapes is required")
    if mode == "export":
        if not args.out and not args.list:
            raise SystemExit("export needs --out")
        return export_arm(args)
    if not args.arm:
        raise SystemExit("need at least one --arm (the baseline)")
    if args.rounds < MIN_ROUNDS:
        raise SystemExit(f"--rounds must be >= {MIN_ROUNDS}")
    arms = parse_arms(args.arm)
    labels = [l for l, _ in arms]
    baseline = args.baseline or labels[0]
    if baseline not in labels:
        raise SystemExit(f"--baseline {baseline!r} is not an --arm label")
    control = args.control
    if not control and not args.list:
        raise SystemExit(
            "--control is required: every run carries an A/A' control (the baseline's code objects loaded a second time)"
        )
    if control and (control in labels or not LABEL_RE.match(control)):
        raise SystemExit(
            f"--control {control!r} must be a new label, not one of {labels}"
        )
    return run_driver(args, arms, baseline, control)


if __name__ == "__main__":
    raise SystemExit(main())
