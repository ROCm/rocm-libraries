#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sweep the gfx950 GDN/KDA decode configurations the dispatcher registry offers.

A *shape* is one decode problem: gate kind, mode (``plain``, or ``fused`` =
``fuse_conv`` + ``fuse_out_norm``), state dtype, head geometry ``Hk/Hv``, head
dim and batch. Per shape this walks ``dispatch_gdn_decode_all`` at
``--sweep-level``: every registered tile x knob configuration at the chosen
level (default ``production``: ``waves_per_eu`` 0 and 4 only; ``full``: a
random sample per tile, ``--tuning-sample``), or, at ``production`` only, just
each tile's base configuration with ``--bases-only`` (a ``full`` walk draws
random knob sets, so it would keep only the bases the sample happens to
contain, not each tile's: ``--bases-only`` is refused there). It checks every
configuration of that
walk once against the fp32 reference on disjoint read / write slots (output,
written state and conv-state slots, untouched slots bit-exact) and times the
ones that pass in cold memory: each timing is a HIP graph cycling through
buffer sets, one cycle touching at least ``--rotate-mb`` MiB
(``builders/gfx950/gdn/gdn_decode.py``, ``cold_copies``). A buffer set is
reused only after at least ``--rotate-mb`` MiB, less one set, of other
traffic; the default is meant to exceed the caches, which is not measured
here. A smaller value may leave the state cache-resident (those times are not
comparable with cold ones, for debugging). The sweep gates disjoint slots only
and times in place slots: the timed in-place launch is not itself gated.

* Stage 1 times every configuration on its own: median of ``--rounds`` replays.
* Stage 2 re-times the shape's ``--top-k`` fastest stage-1 configurations and
  the ``auto`` configuration together: one graph each, replays interleaved, the
  order reversed every round and the base order forward / reversed / rotated by
  half from run to run; per run the median of ``--s2-rounds`` replays, and the
  reported time is the median of ``--runs`` run medians.

The ``auto`` configuration (``dispatch_gdn_decode``) is always timed: it is
appended to the walk when the walk does not list it. Measurements never change
what ``auto`` selects; ``decode_defaults_report.py`` turns the rows into an
advisory proposal for ``AUTO_DEFAULTS``.

Exit codes: 0 every row timed and clean; 1 a row failed the gate, raised or
could not be timed, a shape was refused, ``auto`` was not timed (unless the run
is windowed), or the run timed nothing (see :func:`timed_nothing`); 2 the
command line was refused, or torch / a gfx950 device is missing.

    rocke-gdn-decode-table-sweep --gate-kind kda --mode fused --state-dtype f32 \\
        --heads 32/32 --batches 16 --output-json kda.json
    rocke-gdn-decode-table-sweep --shapes decode_shapes.json --list-only
    # one case on N GPUs: stage 1 split by configuration, then stage 2 by shape
    rocke-gdn-decode-table-sweep --shapes decode_shapes.json --gate-kind kda \\
        --mode plain --state-dtype bf16 --stage1-only --shard I/N --output-json s1_I.json
    rocke-gdn-decode-table-sweep --stage1-json 's1_*.json' --shard I/N --output-json s2_I.json
    rocke-gdn-decode-table-sweep --summarize 's1_*.json' 's2_*.json'
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import glob
import json
import math
import os
import statistics
import subprocess
import sys
from typing import Callable, Iterable, Sequence

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode, dispatch_gdn_decode_all
from dispatch.gdn.axes import AUTO_STATE_ROW
from dispatch.gdn.cli_args import (
    check_rotate_mb,
    fastest_first,
    fastest_timed,
    parse_batches,
    usage_error_text,
)

ARCH = "gfx950"
SCHEMA = "rocke.gdn_decode_table_sweep/1"
MODES = {"plain": {}, "fused": {"fuse_conv": True, "fuse_out_norm": True}}
STATE_DTYPES = ("bf16", "f16", "f32")
GATE_KINDS = ("gdn", "kda")
DEFAULT_HEADS = "16/32"
DEFAULT_BATCHES = (1, 8, 16, 32, 64, 128, 256)
DEFAULT_HEAD_DIM = 128
SHAPE_FIELDS = (
    "gate_kind",
    "mode",
    "state_dtype",
    "num_k_heads",
    "num_v_heads",
    "head_dim",
    "batch",
)
# The walk a stage-1 row was produced by; stage 2 from files records it, since
# the stage-2 rows' own arguments are forced to defaults. Every place that asks
# whether two rows come from the same walk, prints the flags of a walk, builds a
# row's run settings or refuses a walk flag reads this list.
WALK_FIELDS = ("sweep_level", "tuning_sample", "seed", "bases_only")
# What every reader of a saved row reads unconditionally. :func:`load_rows`
# refuses a row without them, so a malformed file is a usage error naming the
# row, never a ``KeyError`` deep in a summary.
ROW_KEYS = (
    SHAPE_FIELDS
    + (
        "stage",
        "index",
        "spec_id",
        "tile",
        "status",
        "auto",
        "us",
        "reason",
        "tuning_id",
        "knobs",
        "rotate_mb",
        "sha",
        "device",
    )
    + WALK_FIELDS
)
SWEEP_LEVELS = ("production", "full")
# A row's status: timed, failed the fp32 gate (never timed), raised while
# compiling or launching, or graph capture was unavailable.
OK, FAIL, ERROR, NO_GRAPH = "ok", "fail", "error", "no_graph"
STATUSES = (OK, FAIL, ERROR, NO_GRAPH)


def case_name(auto_key: tuple) -> str:
    """``"gdn_fused_bf16"``: the name of an ``AUTO_DEFAULTS`` key
    (:attr:`Shape.auto_key`)."""
    gate, fused, state = auto_key
    return f"{gate}_{'fused' if fused else 'plain'}_{state}"


@dc.dataclass(frozen=True, order=True)
class Shape:
    """One decode problem; ``case`` is its ``AUTO_DEFAULTS`` row."""

    gate_kind: str
    mode: str
    state_dtype: str
    num_k_heads: int
    num_v_heads: int
    head_dim: int
    batch: int

    def __post_init__(self):
        if self.gate_kind not in GATE_KINDS:
            raise ValueError(
                f"gate_kind must be one of {GATE_KINDS}: {self.gate_kind!r}"
            )
        if self.mode not in tuple(MODES):
            raise ValueError(f"mode must be one of {tuple(MODES)}: {self.mode!r}")
        if self.state_dtype not in STATE_DTYPES:
            raise ValueError(
                f"state_dtype must be one of {STATE_DTYPES}: {self.state_dtype!r}"
            )
        for name in ("num_k_heads", "num_v_heads", "head_dim", "batch"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive int: {value!r}")

    def request(self, **pin) -> GdnDecodeRequest:
        return GdnDecodeRequest(
            batch=self.batch,
            arch=ARCH,
            gate_kind=self.gate_kind,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            state_dtype=self.state_dtype,
            **MODES[self.mode],
            **pin,
        )

    @property
    def auto_key(self) -> tuple:
        """The ``AUTO_DEFAULTS`` key: (gate kind, fused, state row)."""
        return (self.gate_kind, self.mode == "fused", AUTO_STATE_ROW[self.state_dtype])

    @property
    def case(self) -> str:
        return case_name(self.auto_key)

    @property
    def label(self) -> str:
        return (
            f"{self.gate_kind} {self.mode} {self.state_dtype} "
            f"{self.num_k_heads}/{self.num_v_heads} D{self.head_dim} B{self.batch}"
        )

    def as_dict(self) -> dict:
        return dc.asdict(self)


def shape_of(row: dict) -> Shape:
    return Shape(**{name: row[name] for name in SHAPE_FIELDS})


def walk_of(source) -> tuple:
    """The :data:`WALK_FIELDS` values of a row (or of ``vars(args)``)."""
    return tuple(source[name] for name in WALK_FIELDS)


def walk_flags(walk: tuple) -> str:
    """The command-line flags that reproduce ``walk`` (:func:`walk_of`)."""
    parts = []
    for name, value in zip(WALK_FIELDS, walk):
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.append(f"{flag} {value}")
    return " ".join(parts)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value) -> bool:
    """A finite real number (not a bool). An int beyond the float range is not
    finite here: ``math.isfinite`` raises ``OverflowError`` on it."""
    try:
        return _is_number(value) and math.isfinite(value)
    except OverflowError:
        return False


def _is_int(value) -> bool:
    return type(value) is int


# A timed row's ``us`` is below this (1000 s): anything larger is not a kernel
# time the tool measured.
US_MAX = 1e9


# One validator per saved-row field: ``check(value, row)`` raises ``ValueError``
# naming the field. They run in this order, so a validator may read the fields
# validated before it (``stage``, ``status``, ``index``, ``tuning_id``).
def _check_shape_field(value, row) -> None:
    shape_of(row)  # the Shape constructor refuses every bad shape field


def _check_stage(value, row) -> None:
    if not _is_int(value) or value not in (1, 2):
        raise ValueError(f"stage must be 1 or 2: {value!r}")


def _check_status(value, row) -> None:
    if not isinstance(value, str) or value not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}: {value!r}")


def _check_index(value, row) -> None:
    if not _is_int(value) or value < -1:
        raise ValueError(f"index must be an int >= 0 (or -1): {value!r}")
    if value == -1 and row["stage"] == 1 and row["status"] != ERROR:
        raise ValueError("index -1 is only a stage-2 or an error row's")


def _check_tuning_id(value, row) -> None:
    if not isinstance(value, str):
        raise ValueError(f"tuning_id must be a string: {value!r}")
    if not value and not (row["index"] == -1 and row["status"] == ERROR):
        raise ValueError("tuning_id must be a non-empty string")


def _check_spec_id(value, row) -> None:
    if not isinstance(value, str):
        raise ValueError(f"spec_id must be a string: {value!r}")
    if bool(value) != bool(row["tuning_id"]):
        raise ValueError(f"spec_id must be empty exactly when tuning_id is: {value!r}")


def _check_tile(value, row) -> None:
    """``[num_warps, warp_threads_k, blocks_per_v_dim]`` of a configuration,
    or ``[]`` on the row of a shape with nothing to time (no ``tuning_id``)."""
    if not isinstance(value, list) or any(not _is_int(v) or v < 1 for v in value):
        raise ValueError(f"tile must be a list of positive ints: {value!r}")
    if len(value) != (3 if row["tuning_id"] else 0):
        raise ValueError(
            f"tile must hold 3 ints, or none on a row without a tuning_id: {value!r}"
        )


def _check_auto(value, row) -> None:
    if type(value) is not bool:
        raise ValueError(f"auto must be a bool: {value!r}")


def _check_us(value, row) -> None:
    status = row["status"]
    if status != OK:
        if value is not None:
            raise ValueError(f"us must be null on a {status} row: {value!r}")
        return
    if not _is_finite(value) or not 0 < value <= US_MAX:
        raise ValueError(
            f"us must be a finite number in (0, {US_MAX:g}] on an ok row: {value!r}"
        )


def _check_reason(value, row) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"reason must be a string or null: {value!r}")


def _check_knobs(value, row) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"knobs must be an object: {value!r}")


def _check_rotate_mb(value, row) -> None:
    if not _is_finite(value) or value < 0:
        raise ValueError(f"rotate_mb must be a finite number >= 0: {value!r}")


def _check_cold_target_mb(value, row) -> None:
    if not _is_finite(value) or value <= 0:
        raise ValueError(f"cold_target_mb must be a finite number > 0: {value!r}")


def _check_text(name: str):
    def check(value, row) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string: {value!r}")

    return check


def _check_sweep_level(value, row) -> None:
    if not isinstance(value, str) or value not in SWEEP_LEVELS:
        raise ValueError(f"sweep_level must be one of {SWEEP_LEVELS}: {value!r}")


def _check_tuning_sample(value, row) -> None:
    if not _is_int(value) or value < 0:
        raise ValueError(f"tuning_sample must be an int >= 0: {value!r}")


def _check_seed(value, row) -> None:
    if not _is_int(value):
        raise ValueError(f"seed must be an int: {value!r}")


def _check_bases_only(value, row) -> None:
    if type(value) is not bool:
        raise ValueError(f"bases_only must be a bool: {value!r}")


ROW_CHECKS = {
    **dict.fromkeys(SHAPE_FIELDS, _check_shape_field),
    "stage": _check_stage,
    "status": _check_status,
    "index": _check_index,
    "tuning_id": _check_tuning_id,
    "spec_id": _check_spec_id,
    "tile": _check_tile,
    "auto": _check_auto,
    "us": _check_us,
    "reason": _check_reason,
    "knobs": _check_knobs,
    "rotate_mb": _check_rotate_mb,
    "sha": _check_text("sha"),
    "device": _check_text("device"),
    "sweep_level": _check_sweep_level,
    "tuning_sample": _check_tuning_sample,
    "seed": _check_seed,
    "bases_only": _check_bases_only,
}
# Keys a row MAY carry (a file from before the key was recorded lacks it): the
# validator runs when the key is present.
OPTIONAL_ROW_CHECKS = {"cold_target_mb": _check_cold_target_mb}
assert set(ROW_CHECKS) == set(ROW_KEYS), "every ROW_KEYS key needs one validator"


def check_row(row: dict) -> None:
    """``ValueError`` naming what is wrong with the VALUES of a saved row
    (:func:`load_rows` has checked its keys): every key of :data:`ROW_KEYS`,
    and ``cold_target_mb`` when present, goes through its validator in
    :data:`ROW_CHECKS`. A value of the wrong type is a ``ValueError`` too,
    never a ``TypeError`` / ``OverflowError`` from a validator. Every reader of
    a saved row would otherwise fail on it deep inside a summary, or drop it
    without a word.

    ``us`` is a finite number in (0, :data:`US_MAX`] on a timed (``ok``) row
    and ``None`` on any other. ``index`` is ``>= 0``; ``-1`` marks a row with
    no position in the walk: the stage-2 ``auto`` row that stage 1 did not
    list, or the row of a shape with nothing to time (:func:`shape_error_row`:
    status ``error``, empty ``tuning_id`` and ``spec_id``, empty ``tile``; the
    only row that has them empty). ``rotate_mb`` is a finite number ``>= 0``
    and ``cold_target_mb`` one ``> 0``, so a row that records no real cold
    target cannot count as cold (:func:`is_cold`)."""
    checks = dict(ROW_CHECKS)
    checks.update((k, c) for k, c in OPTIONAL_ROW_CHECKS.items() if k in row)
    for name, check in checks.items():
        try:
            check(row[name], row)
        except (TypeError, OverflowError, KeyError, AttributeError) as exc:
            raise ValueError(f"{name}: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------- CLI parsing
# ``parse_batches`` is imported from ``dispatch.gdn.cli_args``: the one
# ``--batches`` parser of every GDN/KDA decode tool.


def parse_heads(text: str) -> tuple:
    """``"16/32,4/8"`` -> ``((16, 32), (4, 8))``: Hk/Hv pairs."""
    pairs = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        parts = item.split("/")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"--heads wants Hk/Hv pairs, got {item!r}")
        pairs.append((int(parts[0]), int(parts[1])))
    if not pairs:
        raise argparse.ArgumentTypeError(f"--heads wants Hk/Hv pairs, got {text!r}")
    return tuple(pairs)


def parse_shard(text: str) -> tuple:
    index, count = (int(x) for x in str(text).split("/"))
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError(
            f"--shard wants I/N with 0 <= I < N, got {text!r}"
        )
    return index, count


def parse_tuning_ids(text: str) -> tuple:
    """``"a,b"`` -> ``("a", "b")``. An empty list or an empty entry is refused:
    an empty tuple means "no filter" to :func:`shape_configs`, so it would run
    the whole walk."""
    ids = tuple(t.strip() for t in str(text).split(","))
    if not str(text).strip() or not all(ids):
        raise argparse.ArgumentTypeError(
            f"--tuning-ids needs a comma-separated list of non-empty ids, got {text!r}"
        )
    return ids


_ENTRY_KEYS = {
    "gate_kind",
    "mode",
    "state_dtype",
    "heads",
    "num_k_heads",
    "num_v_heads",
    "head_dim",
    "batches",
    "batch",
}


def expand_entry(entry: dict) -> list:
    """Shapes of one ``--shapes`` entry: ``gate_kind``, ``mode``,
    ``state_dtype``, ``heads`` (``["Hk/Hv", ...]``) or ``num_k_heads`` +
    ``num_v_heads``, ``batches`` (list) or ``batch``, optional ``head_dim``
    (default 128). An unknown key, both forms of the heads or the batches, or
    an empty list is an error, never resolved silently."""
    unknown = set(entry) - _ENTRY_KEYS
    if unknown:
        raise ValueError(f"unknown --shapes keys {sorted(unknown)} in {entry}")
    required = ["gate_kind", "mode", "state_dtype"]
    if "heads" not in entry:
        required += ["num_k_heads", "num_v_heads"]
    if "batches" not in entry:
        required.append("batch")
    for key in required:
        if key not in entry:
            raise ValueError(f"--shapes entry lacks {key}: {entry}")
    explicit = {"num_k_heads", "num_v_heads"} & set(entry)
    if "heads" in entry and explicit:
        raise ValueError(
            f"--shapes entry has both heads and {sorted(explicit)}: {entry}"
        )
    if "batches" in entry and "batch" in entry:
        raise ValueError(f"--shapes entry has both batches and batch: {entry}")
    for key in ("heads", "batches"):
        if key not in entry:
            continue
        if not isinstance(entry[key], list):
            raise ValueError(f"--shapes entry's {key} must be a list: {entry}")
        if not entry[key]:
            raise ValueError(f"--shapes entry has an empty {key} list: {entry}")
    if "heads" in entry:
        if not all(isinstance(h, str) and h.strip() for h in entry["heads"]):
            raise ValueError(
                f"--shapes entry's heads must be non-empty 'Hk/Hv' strings: {entry}"
            )
        try:
            heads = parse_heads(",".join(entry["heads"]))
        except (argparse.ArgumentTypeError, ValueError) as exc:
            raise ValueError(f"--shapes entry's heads: {exc}: {entry}") from exc
    else:
        heads = ((entry["num_k_heads"], entry["num_v_heads"]),)
    batches = entry["batches"] if "batches" in entry else [entry["batch"]]
    return [
        Shape(
            gate_kind=entry["gate_kind"],
            mode=entry["mode"],
            state_dtype=entry["state_dtype"],
            num_k_heads=hk,
            num_v_heads=hv,
            head_dim=entry.get("head_dim", DEFAULT_HEAD_DIM),
            batch=batch,
        )
        for hk, hv in heads
        for batch in batches
    ]


def load_shapes(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as fh:
            entries = json.load(fh)
    except ValueError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{path}: expected a JSON list of shape entries")
    shapes = []
    for n, entry in enumerate(entries):
        try:
            shapes.extend(expand_entry(entry))
        except ValueError as exc:
            raise ValueError(f"{path}: entry {n}: {exc}") from exc
    return shapes


def matches(shape: Shape, args) -> bool:
    """``shape`` passes every shape filter the command line set."""
    return (
        (args.gate_kind is None or shape.gate_kind == args.gate_kind)
        and (args.mode is None or shape.mode == args.mode)
        and (args.state_dtype is None or shape.state_dtype == args.state_dtype)
        and (
            args.heads is None
            or (shape.num_k_heads, shape.num_v_heads) in set(args.heads)
        )
        and (args.batches is None or shape.batch in set(args.batches))
        and (args.head_dim is None or shape.head_dim == args.head_dim)
    )


def select_shapes(args) -> list:
    """The shapes to sweep, deduplicated in first-seen order.

    With ``--shapes FILE`` the file's shapes, narrowed by any of
    ``--gate-kind`` / ``--mode`` / ``--state-dtype`` / ``--heads`` /
    ``--batches`` / ``--head-dim`` the command line gives. Without it, the
    grid those flags define (defaults: gdn, plain, bf16, 16/32, batches
    1..256, head dim 128).
    """
    if args.shapes:
        shapes = [s for s in load_shapes(args.shapes) if matches(s, args)]
    else:
        shapes = [
            Shape(
                gate_kind=args.gate_kind or "gdn",
                mode=args.mode or "plain",
                state_dtype=args.state_dtype or "bf16",
                num_k_heads=hk,
                num_v_heads=hv,
                head_dim=DEFAULT_HEAD_DIM if args.head_dim is None else args.head_dim,
                batch=batch,
            )
            for hk, hv in (args.heads or parse_heads(DEFAULT_HEADS))
            for batch in (args.batches or DEFAULT_BATCHES)
        ]
    return list(dict.fromkeys(shapes))


# ---------------------------------------------------------------- enumeration
def shape_configs(shape: Shape, args):
    """``(auto, configs)``: the ``auto`` result and the shape's configuration
    stream -- ``dispatch_gdn_decode_all`` at ``--sweep-level`` (base
    configurations only with ``--bases-only``; only ``--tuning-ids`` when
    given), with ``auto`` appended when the stream does not list it. The
    order is the registry's, so an index names the same configuration in
    every process that walks the same tree with the same arguments.

    A ``--tuning-ids`` entry this walk does not list is an error naming it and
    the walk, never dropped: a typo, or a shortlist from a walk at another
    ``--sweep-level`` / ``--tuning-sample`` / ``--seed``, would otherwise
    narrow the run to ``auto`` alone."""
    req = shape.request()
    auto = dispatch_gdn_decode(req)
    configs = list(
        dispatch_gdn_decode_all(
            req,
            sweep_level=args.sweep_level,
            tuning_sample=args.tuning_sample,
            seed=args.seed,
        )
    )
    if args.bases_only:
        configs = [r for r in configs if not r.spec.knobs]
    if args.tuning_ids:
        wanted = set(args.tuning_ids)
        configs = [r for r in configs if r.spec.tuning_id in wanted]
        found = {r.spec.tuning_id for r in configs} | {auto.spec.tuning_id}
        missing = [t for t in dict.fromkeys(args.tuning_ids) if t not in found]
        if missing:
            raise ValueError(
                f"--tuning-ids not in the walk of {shape.label} at "
                f"{walk_flags(walk_of(vars(args)))}: "
                f"{', '.join(missing)} (pass the walk flags the ids came from)"
            )
    if auto.spec.tuning_id not in {r.spec.tuning_id for r in configs}:
        configs.append(auto)
    return auto, configs


def window(count: int, shard=(0, 1), offset: int = 0, limit: int = 0) -> list:
    """Indices of a shape's ``count`` configurations this process runs:
    ``offset <= i < offset + limit`` (``limit`` 0 = to the end), and of those
    every ``N``-th from ``I`` (``shard = (I, N)``). Indices stay absolute, so
    a resumed or sharded run keeps the numbering, and the ``N`` shards of one
    window are disjoint and together cover it."""
    index, total = shard
    end = count if not limit else min(count, offset + limit)
    return [i for i in range(offset, end) if i % total == index]


def _unmeasured() -> dict:
    """A row's result fields before anything ran."""
    return dict(
        status="",
        us=None,
        runs=[],
        out_err=None,
        state_err=None,
        conv_err=None,
        spill=None,
        copies=None,
        reason="",
    )


def config_row(shape: Shape, result, index: int, auto_id: str, meta: dict) -> dict:
    """A row's identity: shape, configuration, run settings; no result yet."""
    kernel = result.spec.kernel_spec
    return dict(
        meta,
        **shape.as_dict(),
        case=shape.case,
        index=index,
        spec_id=result.candidate.spec_id,
        tile=[kernel.num_warps, kernel.warp_threads_k, kernel.blocks_per_v_dim],
        tuning_id=result.spec.tuning_id,
        knobs=dict(result.spec.knobs),
        auto=result.spec.tuning_id == auto_id,
        **_unmeasured(),
    )


def shape_error_row(shape: Shape, exc: Exception, meta: dict, stage: int = 1) -> dict:
    """The row of a shape with nothing to time: the dispatcher refuses it, or
    (stage 2 from files) it has no timed stage-1 row."""
    row = dict(
        meta,
        stage=stage,
        **shape.as_dict(),
        case=shape.case,
        index=-1,
        spec_id="",
        tile=[],
        tuning_id="",
        knobs={},
        auto=False,
        **_unmeasured(),
    )
    _error(row, exc)
    return row


# ---------------------------------------------------------------- stage 2
def run_order(count: int, run: int) -> list:
    """Run ``run``'s base order of ``count`` arms: forward, reversed, then
    rotated by half, repeating every three runs."""
    order = list(range(count))
    if run % 3 == 1:
        order.reverse()
    elif run % 3 == 2:
        order = order[count // 2 :] + order[: count // 2]
    return order


def interleaved_samples(
    samplers: Sequence[Callable[[], float]], rounds: int, run: int
) -> list:
    """``rounds`` samples per sampler, interleaved: each round takes one from
    every sampler, in :func:`run_order` on even rounds and reversed on odd
    ones, so slow drift and position in the round cancel across arms: exactly
    for linear drift when ``rounds`` is even, approximately when it is odd."""
    order = run_order(len(samplers), run)
    samples = [[] for _ in samplers]
    for r in range(rounds):
        for i in order if r % 2 == 0 else reversed(order):
            samples[i].append(samplers[i]())
    return samples


def stage2_picks(rows: Iterable[dict], top_k: int) -> list:
    """The stage-1 rows stage 2 re-times for one shape: the ``top_k`` fastest
    timed rows (ties by index), then the ``auto`` row if it is not among them
    (whatever its status: a failing ``auto`` is re-gated and reported, never
    dropped)."""
    rows = list(rows)
    timed = fastest_first(rows)
    picks = timed[:top_k]
    for row in rows:
        if row["auto"] and row not in picks:
            picks.append(row)
    return picks


# ---------------------------------------------------------------- JSON rows
def write_rows(path: str, rows: list) -> None:
    """Rewrite ``path`` atomically; a crash keeps the previous complete file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"schema": SCHEMA, "rows": rows}, fh, indent=1)
    os.replace(tmp, path)


def load_rows(patterns: Iterable[str]) -> list:
    """Every row of every file matching ``patterns`` (globs allowed). A file
    that is not valid JSON or lacks this tool's schema and row list is a
    ``ValueError`` naming the path; an unreadable one is an ``OSError``."""
    rows = []
    for pattern in patterns:
        paths = sorted(glob.glob(pattern)) or [pattern]
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                try:
                    data = json.load(fh)
                except ValueError as exc:
                    raise ValueError(f"{path}: not valid JSON: {exc}") from exc
            if (
                not isinstance(data, dict)
                or data.get("schema") != SCHEMA
                or not isinstance(data.get("rows"), list)
            ):
                raise ValueError(f"{path}: not a {SCHEMA} file")
            for n, row in enumerate(data["rows"]):
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: row {n} is not an object")
                for key in ROW_KEYS:
                    if key not in row:
                        raise ValueError(f"{path}: row {n} lacks {key}")
                try:
                    check_row(row)
                except ValueError as exc:
                    raise ValueError(f"{path}: row {n}: {exc}") from exc
            rows.extend(data["rows"])
    return rows


def rows_by_shape(rows: Iterable[dict]) -> dict:
    out: dict = {}
    for row in rows:
        out.setdefault(shape_of(row), []).append(row)
    return out


# ---------------------------------------------------------------- summary
def shape_summary(rows: list) -> dict:
    """One shape's answer from its rows. ``best`` / ``auto`` come from the
    stage-2 rows if it has any, else stage 1: ``best`` is the fastest timed
    row, ``auto`` the ``auto`` row (``None`` when not timed), ``ratio`` auto /
    best. ``bad`` is every row of EVERY stage that failed the gate, raised or
    could not be timed: a stage-1 failure is never re-timed, so stage 2 alone
    would hide it."""
    stage = 2 if any(r["stage"] == 2 for r in rows) else 1
    staged = [r for r in rows if r["stage"] == stage]
    timed = [r for r in staged if r["status"] == OK]
    best = fastest_timed(staged)
    auto = next((r for r in timed if r["auto"]), None)
    ratio = auto["us"] / best["us"] if auto and best else None
    return dict(
        stage=stage,
        best=best,
        auto=auto,
        ratio=ratio,
        timed=len(timed),
        bad=[r for r in rows if r["status"] != OK],
    )


def is_cold(row: dict) -> bool:
    """True when ``row`` was timed in cold memory: its rotation reached the
    cold target the run recorded. A row without a recorded target counts as
    not cold."""
    return row["rotate_mb"] >= row.get("cold_target_mb", float("inf"))


def summarize(rows: list, out=None, require_auto: bool = True) -> int:
    """Print the best configuration per shape against ``auto``. Returns 1 if
    there are no rows, any row failed the gate, raised or could not be timed,
    or (with ``require_auto``; a windowed stage-1 run times ``auto`` in one
    window only) a shape has no timed ``auto`` row; else 0."""
    out = out or sys.stdout
    if not rows:
        print("no rows to summarize", file=out)
        return 1
    exit_code = 0
    if not all(is_cold(r) for r in rows):
        print(
            "WARNING: some rows were not timed in cold memory (--rotate-mb below "
            "the cold target): debugging only",
            file=out,
        )
    print(
        f"{'shape':<36} {'stage':>5} {'timed':>5} {'best_us':>9} {'auto_us':>9} "
        f"{'auto/best':>9}  best tuning_id  knobs",
        file=out,
    )
    for shape, shape_rows in sorted(rows_by_shape(rows).items()):
        s = shape_summary(shape_rows)
        best, auto = s["best"], s["auto"]
        if best is None:
            print(
                f"{shape.label:<36} {s['stage']:>5} {0:>5}  no timed configuration",
                file=out,
            )
            exit_code = 1
        else:
            auto_us = f"{auto['us']:9.3f}" if auto else f"{'-':>9}"
            ratio = f"{s['ratio']:9.3f}" if s["ratio"] else f"{'-':>9}"
            print(
                f"{shape.label:<36} {s['stage']:>5} {s['timed']:>5} {best['us']:9.3f} "
                f"{auto_us} {ratio}  {best['tuning_id']}  "
                f"{json.dumps(best['knobs'], sort_keys=True)}",
                file=out,
            )
            if auto is None:
                print(f"{'':<36} auto configuration not timed", file=out)
                exit_code = 1 if require_auto else exit_code
        for row in s["bad"]:
            print(
                f"{'':<36} S{row['stage']} {row['status'].upper()} "
                f"{row['tuning_id']}: {row['reason'] or row['status']}",
                file=out,
            )
            exit_code = 1
    return exit_code


# ---------------------------------------------------------------- list-only
def list_configs(args, shapes: list) -> int:
    print(
        f"sweep_level={args.sweep_level} tuning_sample={args.tuning_sample} "
        f"seed={args.seed} bases_only={args.bases_only} shard={args.shard[0]}/"
        f"{args.shard[1]} offset={args.offset} limit={args.limit} (CPU list-only)"
    )
    refused = 0
    for shape in shapes:
        try:
            auto, configs = shape_configs(shape, args)
        except Exception as exc:  # the shape itself is refused
            print(f"\n{shape.label}  REFUSED {type(exc).__name__}: {exc}")
            refused += 1
            continue
        picked = window(len(configs), args.shard, args.offset, args.limit)
        print(
            f"\n{shape.label}  configurations={len(configs)} this process={len(picked)}"
        )
        for i in picked:
            result = configs[i]
            mark = "  <- auto" if result.spec.tuning_id == auto.spec.tuning_id else ""
            print(
                f"  {i:5d} {result.candidate.spec_id:<20} {result.spec.tuning_id}  "
                f"{json.dumps(dict(result.spec.knobs), sort_keys=True)}{mark}"
            )
    return 1 if refused else 0


# ---------------------------------------------------------------- GPU
def _driver():
    """The builder's host driver (imports torch): GPU paths only."""
    from builders.gfx950.gdn import gdn_decode

    return gdn_decode


class _ShapeInputs:
    """One shape's gate inputs and fp32 reference, timing inputs and cold value
    sets, shared by every configuration: they depend only on the problem, and
    every configuration of a problem takes the same kernel arguments.

    The gate runs on DISJOINT read and write slots (``make_inputs`` default):
    with ``read == write`` a kernel that reads its state from the write slot,
    or writes into the read slot, computes the same answer and passes, so the
    in-place set cannot see that bug. The pool keeps spare slots, which the
    gate checks stay untouched. Timing runs on the in-place set (read slots
    equal write slots, as in a serving engine's state update), which is also
    what :func:`touched_bytes` sizes the rotation from. So the sweep gates one
    launch on disjoint slots and times in place: the in-place launch it times
    is not itself gated (the gate can score in-place inputs by value, but
    cannot see a kernel that reads its write slot)."""

    def __init__(self, shape: Shape, auto, args):
        g = _driver()
        self.spec = auto.spec.kernel_spec
        self.batch = shape.batch
        self.rotate_mb = args.rotate_mb
        self.gate_inp = g.make_inputs(
            self.spec, shape.batch, seed=args.seed, disjoint_writes=True
        )
        self.ref = g.reference(self.spec, self.gate_inp)
        self.inp = g.make_inputs(
            self.spec, shape.batch, seed=args.seed, disjoint_writes=False
        )
        self.touched = g.touched_bytes(self.spec, self.inp, shape.batch)
        self._sets = None

    def gate(self, result, row: dict):
        """Gate ``result`` and fill ``row``'s errors; its launcher if it passed."""
        g = _driver()
        spec = result.spec.kernel_spec
        launcher = g.launcher_for(spec, arch=ARCH)
        values = g.run_values(spec, self.gate_inp, launcher, self.batch)
        verdict = g.gate(spec, self.gate_inp, values, self.ref)
        del values
        row.update(
            out_err=verdict.out_err,
            state_err=verdict.state_err,
            conv_err=verdict.conv_err,
            spill=verdict.spill,
        )
        if verdict.passed:
            return launcher
        row.update(
            status=FAIL,
            reason=f"fp32 gate: worst={verdict.worst:.3e} spill={verdict.spill}",
        )
        return None

    def capture(self, result, launcher):
        """A cold graph of ``result`` on this shape's value sets (made on first
        use, then shared); raises the driver's ``CaptureUnavailable`` if
        capture fails."""
        g = _driver()
        if self._sets is None:
            self._sets = g.cold_value_sets(
                self.spec, self.inp, self.batch, self.rotate_mb
            )
            k = len(self._sets)
            print(
                f"ROTATION {self.touched / 2**20:.3f} MiB touched per launch x "
                f"{k} buffer sets = {k * self.touched / 2**20:.1f} MiB per cycle "
                f"(target >= {self.rotate_mb}); a set is reused after "
                f"{(k - 1) * self.touched / 2**20:.1f} MiB",
                flush=True,
            )
        return g.capture_cold(result.spec.kernel_spec, self.batch, self._sets, launcher)

    def release(self) -> None:
        """Drop the value sets; the next :meth:`capture` makes fresh ones."""
        self._sets = None
        _driver().drain()


def _reason(text: str) -> str:
    """A row's ``reason``: one line, at most 400 characters."""
    return text.replace("\n", " ")[:400]


def _error(row: dict, exc: Exception) -> None:
    row.update(status=ERROR, reason=_reason(f"{type(exc).__name__}: {exc}"))


def _no_graph(row: dict, exc: Exception) -> None:
    row.update(status=NO_GRAPH, reason=_reason(f"graph capture unavailable: {exc}"))


def _print_row(shape: Shape, row: dict) -> None:
    us = f"{row['us']:9.3f}us" if row["us"] is not None else f"{'-':>11}"
    extra = "" if row["status"] == OK else f"  {row['reason']}"
    print(
        f"S{row['stage']} {shape.label} {row['index']:5d} {row['tuning_id']} {us} "
        f"{row['status'].upper()}{'  <- auto' if row['auto'] else ''}{extra}",
        flush=True,
    )


def stage1_shape(shape: Shape, auto, configs, indices, args, meta) -> list:
    """Gate and time ``configs[i]`` for every ``i`` in ``indices``, each on its
    own: median of ``--rounds`` replays of its cold graph."""
    g = _driver()
    data = _ShapeInputs(shape, auto, args)
    rows = []
    for i in indices:
        result = configs[i]
        row = config_row(shape, result, i, auto.spec.tuning_id, dict(meta, stage=1))
        try:
            launcher = data.gate(result, row)
            if launcher is not None:
                try:
                    cold = data.capture(result, launcher)
                except g.CaptureUnavailable as exc:
                    cold = None
                    _no_graph(row, exc)
                if cold is not None:
                    try:
                        samples = [cold.replay_us() for _ in range(args.rounds)]
                    finally:
                        cold.close()
                    us = statistics.median(samples)
                    row.update(status=OK, us=us, runs=[us], copies=cold.copies)
        except Exception as exc:  # compile / launch failure: recorded, not timed
            _error(row, exc)
        finally:
            g.drain()
        rows.append(row)
        _print_row(shape, row)
    data.release()
    return rows


def stage2_shape(shape: Shape, auto, picks: list, args, meta) -> list:
    """Re-time ``picks`` (``(index, result)``) together, interleaved
    (:func:`interleaved_samples`), for ``--runs`` runs of ``--s2-rounds``
    rounds; each run captures fresh graphs on fresh value sets."""
    data = _ShapeInputs(shape, auto, args)
    entries = []  # (row, result, launcher)
    rows = []
    for index, result in picks:
        row = config_row(shape, result, index, auto.spec.tuning_id, dict(meta, stage=2))
        rows.append(row)
        try:
            launcher = data.gate(result, row)
            if launcher is not None:
                entries.append((row, result, launcher))
        except Exception as exc:
            _error(row, exc)
        finally:
            _driver().drain()
    for run in range(args.runs):
        graphs = []
        for row, result, launcher in entries:
            if row["status"] in (ERROR, NO_GRAPH):
                continue
            try:
                cold = data.capture(result, launcher)
            except _driver().CaptureUnavailable as exc:
                _no_graph(row, exc)
                continue
            except Exception as exc:
                _error(row, exc)
                continue
            graphs.append((row, cold))
        samples = interleaved_samples(
            [cold.replay_us for _, cold in graphs], args.s2_rounds, run
        )
        for (row, cold), got in zip(graphs, samples):
            row["runs"].append(statistics.median(got))
            row["copies"] = cold.copies
            cold.close()
        del graphs
        data.release()
    for row, _, _ in entries:
        if row["status"] not in (ERROR, NO_GRAPH):
            row.update(status=OK, us=statistics.median(row["runs"]))
    for row in rows:
        _print_row(shape, row)
    return rows


def pinned_result(shape: Shape, row: dict):
    """Re-dispatch a recorded configuration from its ``spec_id``, ``tuning_id``
    and ``knobs``; the pin must resolve to exactly that ``tuning_id``."""
    result = dispatch_gdn_decode(
        shape.request(
            spec_id=row["spec_id"],
            tuning_id=row["tuning_id"],
            tuning_knobs=row["knobs"],
        )
    )
    if result.spec.tuning_id != row["tuning_id"]:
        raise ValueError(
            f"{row['tuning_id']} re-dispatched as {result.spec.tuning_id} on {shape.label}"
        )
    return result


def _git(*args: str) -> str | None:
    """Output of ``git -C <this tree> args``; ``None`` without git or a repo."""
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_sha() -> str:
    """``git HEAD`` of the source tree, ``-dirty`` appended when tracked files
    differ from it (untracked files, such as output files, do not count);
    ``unknown`` without git."""
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return "unknown"
    return sha + ("-dirty" if _git("status", "--porcelain", "-uno") else "")


def _toolchain() -> dict:
    """Compiler and framework versions, best effort (``unknown`` when a
    value cannot be read): the ROCm release and LLVM version of the loaded
    COMGR, torch's version, and the lowering backend and LLVM flavor a
    compile resolves now (what ``compile_kernel`` would use, not what a past
    run did)."""
    out = dict(
        rocm_version="unknown",
        llvm_version="unknown",
        torch_version="unknown",
        backend="unknown",
        llvm_flavor="unknown",
    )
    try:
        from rocke.runtime import comgr

        rocm = comgr.resolved_lib_rocm_version()
        if rocm:
            out["rocm_version"] = ".".join(map(str, rocm))
        info = comgr.loaded_compiler_info()
        if info is not None and info.llvm_version:
            out["llvm_version"] = ".".join(map(str, info.llvm_version))
    except Exception:  # a missing version is recorded as unknown, never fatal
        pass
    try:
        from rocke.core.backend import resolve_backend

        out["backend"] = str(resolve_backend())
    except Exception:
        pass
    try:
        from rocke.core.lower_llvm import _resolve_llvm_flavor

        out["llvm_flavor"] = str(_resolve_llvm_flavor())
    except Exception:
        pass
    try:
        import torch

        out["torch_version"] = str(torch.__version__)
    except Exception:
        pass
    return out


def run_meta(args, device: str, cold_target_mb: int) -> dict:
    """The run settings every row carries, so one row stands on its own.
    ``cold_target_mb`` is the driver's ``ROTATE_MB``, the rotation that counts
    as cold memory (:func:`is_cold`)."""
    return dict(
        arch=ARCH,
        device=device,
        sha=args.source_sha or _source_sha(),
        **_toolchain(),
        **{name: getattr(args, name) for name in WALK_FIELDS},
        rotate_mb=args.rotate_mb,
        cold_target_mb=cold_target_mb,
        rounds=args.rounds,
        s2_rounds=args.s2_rounds,
        top_k=args.top_k,
    )


def sweep(args, shapes: list, meta: dict) -> list:
    """Stage 1 per shape, then (unless ``--stage1-only``) stage 2 on it."""
    rows = []
    for shape in shapes:
        try:
            auto, configs = shape_configs(shape, args)
        except Exception as exc:  # the shape itself is refused
            print(f"ERROR {shape.label}: {type(exc).__name__}: {exc}", flush=True)
            rows.append(shape_error_row(shape, exc, meta))
            continue
        indices = window(len(configs), args.shard, args.offset, args.limit)
        print(
            f"\n=== {shape.label}: {len(configs)} configurations, "
            f"{len(indices)} in this process; auto {auto.spec.tuning_id}",
            flush=True,
        )
        if not indices:
            continue
        stage1 = stage1_shape(shape, auto, configs, indices, args, meta)
        rows.extend(stage1)
        if not args.stage1_only:
            picks = stage2_picks(stage1, args.top_k)
            rows.extend(
                stage2_shape(
                    shape,
                    auto,
                    [(r["index"], configs[r["index"]]) for r in picks],
                    args,
                    meta,
                )
            )
        if args.output_json:
            write_rows(args.output_json, rows)
    return rows


def timed_nothing(args, shapes: list) -> bool:
    """True when a sweep that produced no row at all is a failure.

    A run times nothing when every shape's window is empty. That is fine for
    one ``--shard I/N`` of a small shape (``i % N != I`` for every index) as
    long as another shard covers the window; it is a silent no-op, and must
    not exit 0, when ``--offset`` / ``--limit`` leave NO configuration of any
    shape for any shard. So: a failure unless some shape's window is
    non-empty ignoring the shard."""
    for shape in shapes:
        _, configs = shape_configs(shape, args)
        if window(len(configs), (0, 1), args.offset, args.limit):
            return False
    return True


def stage1_walk(rows: Iterable[dict]) -> dict:
    """The walk fields the stage-1 ``rows`` share. Rows from different walks
    are refused: stage 2 would be labelled with a walk that part of its input
    never ran. (:func:`load_rows` has already refused a row without the walk
    fields.)"""
    walks = {walk_of(row) for row in rows}
    if len(walks) != 1:
        raise ValueError(
            f"the stage-1 rows come from {len(walks)} walks "
            f"({', '.join(WALK_FIELDS)}): {sorted(map(str, walks))}"
        )
    return dict(zip(WALK_FIELDS, walks.pop()))


def stage1_shapes(args):
    """``(by_shape, shapes)``: the ``--stage1-json`` stage-1 rows by shape, and
    the recorded shapes the shape filters keep. No shape left is an error:
    stage 2 would time nothing and report success. So are stage-1 rows of
    different walks (:func:`stage1_walk`)."""
    by_shape = rows_by_shape(r for r in load_rows(args.stage1_json) if r["stage"] == 1)
    shapes = [s for s in sorted(by_shape) if matches(s, args)]
    if not shapes:
        raise ValueError("no recorded stage-1 shape matches the given filters")
    stage1_walk(r for rows in by_shape.values() for r in rows)
    return by_shape, shapes


def retime(args, meta: dict) -> list:
    """Stage 2 from saved stage-1 rows (``--stage1-json``): every matching
    shape (``--shard`` splits the shapes), top ``--top-k`` plus ``auto``. A
    shape with no timed stage-1 row gets an ``error`` row, so the run fails.
    Stage 2 takes its walk fields (``seed`` included: it seeds the gate and
    timing inputs) from the stage-1 rows it re-times, not from this command
    line, whose walk flags are refused: the rows record the walk their inputs
    were made with."""
    by_shape, shapes = stage1_shapes(args)
    walk = stage1_walk(r for rows in by_shape.values() for r in rows)
    meta = dict(meta, **walk)
    args = argparse.Namespace(**{**vars(args), **walk})
    mine = [s for i, s in enumerate(shapes) if i % args.shard[1] == args.shard[0]]
    rows = []
    for shape in mine:
        recorded = by_shape[shape]
        if not any(r["status"] == OK for r in recorded):
            print(f"ERROR {shape.label}: no timed stage-1 row", flush=True)
            rows.append(
                shape_error_row(
                    shape, ValueError("no timed stage-1 row"), meta, stage=2
                )
            )
            continue
        auto = dispatch_gdn_decode(shape.request())
        if not any(
            r["auto"] and r["tuning_id"] == auto.spec.tuning_id for r in recorded
        ):
            print(
                f"WARNING {shape.label}: auto is {auto.spec.tuning_id} here, not the "
                "stage-1 auto row; it is re-timed as auto",
                flush=True,
            )
        picks = []
        for row in stage2_picks(recorded, args.top_k):
            picks.append((row["index"], pinned_result(shape, row)))
        if auto.spec.tuning_id not in {res.spec.tuning_id for _, res in picks}:
            picks.append((-1, auto))
        print(
            f"\n=== {shape.label}: stage 2 on {len(picks)} configurations", flush=True
        )
        rows.extend(stage2_shape(shape, auto, picks, args, meta))
        if args.output_json:
            write_rows(args.output_json, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    shapes = ap.add_argument_group("shapes (filters when --shapes is given)")
    shapes.add_argument("--shapes", default="", help="JSON list of shape entries")
    shapes.add_argument("--gate-kind", choices=GATE_KINDS, default=None)
    shapes.add_argument("--mode", choices=tuple(MODES), default=None)
    shapes.add_argument("--state-dtype", choices=STATE_DTYPES, default=None)
    shapes.add_argument("--heads", type=parse_heads, default=None, help="Hk/Hv list")
    shapes.add_argument("--batches", type=parse_batches, default=None)
    shapes.add_argument("--head-dim", type=int, default=None)
    walk = ap.add_argument_group("configurations")
    walk.add_argument("--sweep-level", choices=SWEEP_LEVELS, default="production")
    walk.add_argument(
        "--tuning-sample",
        type=int,
        default=0,
        help="with --sweep-level full: random legal configurations per tile, "
        "seeded by --seed (required > 0 at full; refused at production)",
    )
    walk.add_argument("--seed", type=int, default=0, help="sampling and input seed")
    walk.add_argument(
        "--bases-only", action="store_true", help="each tile's base configuration only"
    )
    walk.add_argument(
        "--tuning-ids",
        type=parse_tuning_ids,
        default=(),
        help="only these tuning_ids (plus auto), e.g. a shortlist to re-time; "
        "an empty list or entry is refused",
    )
    walk.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="I/N")
    walk.add_argument("--offset", type=int, default=0, help="configs to skip per shape")
    walk.add_argument(
        "--limit", type=int, default=0, help="configs per shape (0 = all)"
    )
    timing = ap.add_argument_group("timing")
    timing.add_argument(
        "--rotate-mb",
        type=int,
        default=None,
        help="MiB one cold-memory rotation cycle touches at least (default: the "
        "driver's ROTATE_MB; see gdn_decode.cold_copies); 0 = warm, debugging "
        "only, and says so",
    )
    timing.add_argument("--rounds", type=int, default=5, help="stage-1 replays")
    timing.add_argument(
        "--top-k", type=int, default=8, help="stage-2 arms besides auto"
    )
    timing.add_argument(
        "--s2-rounds",
        type=int,
        default=30,
        help="stage-2 rounds per run; an even count cancels linear drift exactly "
        "(the order reverses every round)",
    )
    timing.add_argument("--runs", type=int, default=3, help="stage-2 runs")
    timing.add_argument("--stage1-only", action="store_true")
    timing.add_argument(
        "--stage1-json",
        nargs="+",
        default=None,
        metavar="JSON",
        help="run stage 2 only, from these stage-1 files; --shard splits the shapes",
    )
    out = ap.add_argument_group("output")
    out.add_argument("--output-json", default="")
    out.add_argument(
        "--source-sha", default="", help="recorded per row (default: git HEAD)"
    )
    out.add_argument("--list-only", action="store_true", help="CPU listing; no GPU")
    out.add_argument(
        "--summarize",
        nargs="+",
        default=None,
        metavar="JSON",
        help="best configuration per shape vs auto from saved files; no GPU",
    )
    return ap


def options(**overrides) -> argparse.Namespace:
    """The command line's defaults with ``overrides``: the settings object the
    sweep functions read, for callers that drive them from Python."""
    args = build_parser().parse_args([])
    unknown = set(overrides) - set(vars(args))
    if unknown:
        raise TypeError(f"unknown sweep options {sorted(unknown)}")
    vars(args).update(overrides)
    return args


# Flags that only stage 2 reads.
_STAGE2_FLAGS = ("top_k", "s2_rounds", "runs")


def check_args(ap: argparse.ArgumentParser, args) -> None:
    """Refuse every flag the chosen mode would accept and then ignore.

    Blind spot: a flag is detected as given only when its value differs from
    its default, so an explicit flag equal to its default (e.g. ``--rounds 5``)
    cannot be told from an absent one and is accepted."""
    defaults = vars(ap.parse_args([]))

    def changed(*names):
        return [
            "--" + name.replace("_", "-")
            for name in names
            if getattr(args, name) != defaults[name]
        ]

    if args.summarize:
        # --summarize reads saved rows and nothing else.
        ignored = changed(*(name for name in defaults if name != "summarize"))
        if ignored:
            ap.error(f"--summarize reads saved rows only: no {', '.join(ignored)}")
        return
    windowed = args.shard[1] > 1 or args.offset or args.limit
    if args.stage1_json:
        # Stage 2 from files re-times the recorded rows: nothing is walked, so
        # a walk flag would be accepted and ignored.
        ignored = changed(
            "rounds",
            "stage1_only",
            "list_only",
            "offset",
            "limit",
            "shapes",
            *WALK_FIELDS,
            "tuning_ids",
        )
        if ignored:
            ap.error(
                f"--stage1-json runs stage 2 on recorded rows: no {', '.join(ignored)}"
            )
    elif windowed and not args.stage1_only and not args.list_only:
        ap.error(
            "--shard/--offset/--limit see part of each shape's configurations, so "
            "stage 2 would rank a part: pass --stage1-only, then run stage 2 with "
            "--stage1-json"
        )
    if not args.stage1_json:
        if args.tuning_sample < 0:
            ap.error("--tuning-sample must be >= 0")
        if args.sweep_level == "production" and args.tuning_sample:
            ap.error(
                "--tuning-sample samples the full space: it needs --sweep-level "
                "full (production walks a fixed list)"
            )
        if args.sweep_level == "full" and args.tuning_sample == 0:
            ap.error(
                "--sweep-level full needs --tuning-sample N > 0: without it the "
                "walk is far more than production (tens of thousands per shape; "
                "see --list-only)"
            )
        if args.bases_only and args.sweep_level == "full":
            ap.error(
                "--bases-only keeps each tile's base configuration, which only a "
                "--sweep-level production walk lists: a full walk samples knob "
                "sets, so it keeps only the bases the sample happens to contain "
                "and drops the rest of the tiles"
            )
    if args.list_only:
        # A listing times nothing: every timing or output flag would be ignored.
        ignored = changed(
            "output_json",
            "rotate_mb",
            "rounds",
            "s2_rounds",
            "runs",
            "top_k",
            "stage1_only",
            "source_sha",
        )
        if ignored:
            ap.error(f"--list-only times nothing: no {', '.join(ignored)}")
    if args.stage1_only:
        ignored = changed(*_STAGE2_FLAGS)
        if ignored:
            ap.error(f"--stage1-only runs no stage 2: no {', '.join(ignored)}")
    for name in ("rounds", "s2_rounds", "runs", "top_k"):
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.offset < 0 or args.limit < 0:
        ap.error("--offset and --limit must be >= 0")
    if args.head_dim is not None and args.head_dim < 1:
        ap.error("--head-dim must be >= 1")
    check_rotate_mb(args.rotate_mb, ap)


def main(argv: Iterable[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    check_args(ap, args)
    if args.summarize:
        try:
            rows = load_rows(args.summarize)
        except (OSError, ValueError) as exc:
            ap.error(usage_error_text(exc))
        return summarize(rows)
    if args.stage1_json:
        try:
            _, recorded_shapes = stage1_shapes(args)
        except (OSError, ValueError) as exc:
            ap.error(usage_error_text(exc))
        shapes = []
    else:
        try:
            shapes = select_shapes(args)
        except (OSError, ValueError) as exc:
            ap.error(usage_error_text(exc))
        if not shapes:
            ap.error("no shape matches the given filters")
    if args.list_only:
        return list_configs(args, shapes)

    try:
        import torch  # noqa: F401  (the driver needs it; fail early and clearly)
    except ImportError:
        print("torch is required to time configurations", file=sys.stderr)
        return 2
    from rocke.runtime.hip_module import get_device_arch

    device = get_device_arch()
    if not device or not device.startswith(ARCH):
        print(f"requires a {ARCH} device; visible: {device!r}", file=sys.stderr)
        return 2
    from builders.gfx950.gdn.gdn_decode import ROTATE_MB, timing_label

    if args.rotate_mb is None:
        args.rotate_mb = ROTATE_MB
    print(timing_label(args.rotate_mb), flush=True)
    print(
        "slots: gate = disjoint read / write slots, one launch; "
        "timing = in-place slots (in-place correctness is not gated)",
        flush=True,
    )
    meta = run_meta(args, device, ROTATE_MB)
    rows = retime(args, meta) if args.stage1_json else sweep(args, shapes, meta)
    return finish(args, rows, shapes, len(recorded_shapes) if args.stage1_json else 0)


def finish(args, rows: list, shapes: list, recorded: int) -> int:
    """Write ``rows``, summarize them and give the exit code. ``recorded`` is
    the number of shapes in the ``--stage1-json`` files (0 otherwise).

    A run that timed no row is a failure unless its ``--shard I/N`` legitimately
    owns nothing: a stage-2 shard past the last recorded shape (I >= K for K
    shapes), or a stage-1 shard that owns no index of a small window while
    another shard does. Both print a line and succeed."""
    if args.output_json:
        write_rows(args.output_json, rows)
        print(f"\nwrote {len(rows)} rows to {args.output_json}")
    if not rows:
        if args.stage1_json:
            print(
                f"shard {args.shard[0]}/{args.shard[1]} owns no shape "
                f"({recorded} shapes recorded)"
            )
            return 0
        if timed_nothing(args, shapes):
            print(
                "ERROR: nothing was timed: --offset / --limit leave no "
                "configuration of any shape",
                file=sys.stderr,
            )
            return 1
        print(f"shard {args.shard[0]}/{args.shard[1]} owns no configuration")
        return 0
    print()
    windowed = args.shard[1] > 1 or args.offset or args.limit
    return summarize(rows, require_auto=bool(args.stage1_json) or not windowed)


if __name__ == "__main__":
    raise SystemExit(main())
