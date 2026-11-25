#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_sched_ir_emit.py -- Python reference emitter for
# the scheduling hints of the attention backward stage tables
# (kernels/common/_attention_bwd_sched.py: StageTable / DqStageTable
# emit_loop_head, emit_stage_begin, emit_stage_end). The module has no kernel
# of its own, so this builds a probe kernel that calls the hooks of one table
# the way the backward bodies do (the loop head, then begin / end around each
# stage); the C twin attention_bwd_sched_ir_emit.c builds the same kernel
# through helper_rocke.instances.common._attention_bwd_sched.h and the two .ll
# must match.
#
# A config is a table config of attention_bwd_sched_emit.py ("geom", "table",
# "dq" or "dqgeom") plus a "probe" key: "steps" (the default: every stage with
# prefetch and prev_atomics true), "last" (prefetch and prev_atomics false,
# the "ck" model's peeled step), "two" (two steps: the second ends its stages
# with prev_atomics false), or "bad" (an unknown stage name after the loop
# head: both engines refuse), or "badend" (an unknown stage name straight to
# emit_stage_end, past emit_stage_begin's own check). _CONFIGS is built by the loops below; keep the C
# table in sync with `attention_bwd_sched_ir_emit.py --print-c-configs`.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
import sys

from _emit_common import run_emit
from attention_bwd_sched_emit import build, parse, print_c_configs
from kernels.common._attention_bwd_sched import (
    DQ_SCHEDS,
    DQ_STAGES,
    SCHEDS,
    StageTable,
)
from rocke.core.ir import IRBuilder

_T64 = "head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4"
_T128 = "head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4"
_DQ = "head_size:64;k_m0:64;k_n0:64;waves:4"

# (name, arch, fields) of the main-kernel tables every sched value runs on
_BASES = (
    ("sync", "gfx942", _T64),
    ("sync wide", "gfx950", f"{_T128};atom_g02:32;atom_g4:32;dtype:bf16"),
    ("sync split", "gfx942", f"{_T64};dq_mode:split"),
    ("sync lds", "gfx950", f"{_T128};kv_residency:lds;kt_source:lds;acc_in_lds:True"),
    ("dma ring", "gfx950", f"{_T128};global_path:dma;pipeline:ring;ring_depth:2"),
    ("register ring", "gfx942", f"{_T64};ring_depth:3;pipeline:ring"),
    ("merged early", "gfx942", f"{_T64};barrier_merge:on;ring_depth:2;pipeline:merged"),
    (
        "merged late",
        "gfx950",
        f"{_T128};barrier_merge:on;ring_depth:2;pipeline:merged;ring_issue:late",
    ),
    ("counted early", "gfx950", f"{_T128};loop_sync:counted"),
    ("counted late", "gfx942", f"{_T64};loop_sync:counted;counted_issue:late"),
    ("ck", "gfx942", f"{_T64};pipeline:ck;transpose_source:xt_lds"),
    (
        "rdna",
        "gfx1201",
        "head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:2;transpose_source:lds_plain",
    ),
)
_DQ_BASES = (
    ("dq gfx950", "gfx950", _DQ),
    (
        "dq gfx942 prefetch",
        "gfx942",
        f"{_DQ};transpose_source:lds_plain;kv_prefetch:True;kv_pad:8",
    ),
)
# special configs: (comment, config)
_SPECIAL = (
    (
        "ck last step",
        f"table|gfx950|{_T128};pipeline:ck;sched:stage_table_atomics;probe:last",
    ),
    (
        "sync two steps deferred",
        f"table|gfx942|{_T64};sched:stage_table_atomics;probe:two",
    ),
    (
        "setprio two steps",
        f"table|gfx950|{_T128};sched:stage_table;setprio:True;probe:two",
    ),
    ("unknown stage", f"table|gfx942|{_T64};probe:bad"),
    (
        "unknown stage, split",
        f"table|gfx942|{_T64};dq_mode:split;sched:fences_only;probe:bad",
    ),
    ("unknown dQ stage", f"dq|gfx950|{_DQ};sched:stage_table_setprio;probe:bad"),
    (
        "unknown dQ stage at the end",
        f"dq|gfx950|{_DQ};sched:stage_table_setprio;probe:badend",
    ),
    ("table refused", f"table|gfx942|{_T64};sched:bogus"),
    (
        "dma batch refused in the groups",
        "table|gfx942|head_size:16;k_m0:16;k_n0:16;k_k4:16;waves:1;global_path:dma;pipeline:ring",
    ),
    (
        "geometry",
        "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;atoms:32/16/32;sched:iglp1;global_path:dma;ring_depth:2",
    ),
    (
        "dq geometry",
        "dqgeom|gfx950|head_size:128;k_m0:128;k_n0:32;waves:4;atom_g02:32;atom_g4:16;dtype:bf16;transpose_source:tr_read;sched:stage_table_setprio;kv_pad:16;kv_prefetch:True",
    ),
)


def _configs():
    out = []
    for name, arch, fields in _BASES:
        for s in SCHEDS:
            out.append((f"{name} {s}", f"table|{arch}|{fields};sched:{s}"))
        out.append(
            (f"{name} setprio", f"table|{arch}|{fields};sched:stage_table;setprio:True")
        )
    for name, arch, fields in _DQ_BASES:
        for s in DQ_SCHEDS:
            out.append((f"{name} {s}", f"dq|{arch}|{fields};sched:{s}"))
    out.extend(_SPECIAL)
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    kind, arch, fields = parse(_CONFIGS[idx][1])
    return (kind, fields), arch


def _step(b, t, probe: str, prev_atomics: bool) -> None:
    if isinstance(t, StageTable):
        names = t.stage_names
        flags = {"prefetch": probe != "last", "prev_atomics": prev_atomics}
    else:
        names = DQ_STAGES
        flags = {}
    for name in names:
        t.emit_stage_begin(b, name)
        t.emit_stage_end(b, name, **flags)


def build_attention_bwd_sched_kernel(spec, *, arch: str):
    """The probe kernel of one config."""
    kind, fields = spec
    fields = dict(fields)
    probe = fields.pop("probe", "steps")
    t = build(kind, arch, fields)
    b = IRBuilder("attn_bwd_sched_probe")
    t.emit_loop_head(b)
    if probe == "bad":
        t.emit_stage_begin(b, "X")
        t.emit_stage_end(b, "X")
    elif probe == "badend":
        t.emit_stage_end(b, "X")
    else:
        _step(b, t, probe, probe != "last")
        if probe == "two":
            t.emit_loop_head(b)
            _step(b, t, probe, False)
    b.ret()
    return b.kernel


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--print-c-configs":
        print_c_configs(_CONFIGS)
        return 0
    return run_emit(
        _spec,
        build_attention_bwd_sched_kernel,
        usage="usage: attention_bwd_sched_ir_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
