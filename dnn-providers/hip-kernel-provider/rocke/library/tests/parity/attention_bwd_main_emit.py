#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_main_emit.py -- Python reference emitter for the
# whole layout-general attention backward main kernel
# (kernels/common/attention_bwd.py: _emit_main with _cta_setup, _kv_tile, the
# prefetch ring of _attention_bwd_ring and the dK / dV epilogues). The C twin
# attention_bwd_main_emit.c builds the same kernel through
# rocke_build_attn_bwd_main_new (rocke/instance_attention_bwd.h) and the two
# .ll must match.
#
# Both engines build _emit_main(spec, arch, main_kernel_refusals=False): the
# recorded resource-gate refusals are Python-only.
#
# Configs:
#   * "@<case id>": the main kernel of a case of the phase-1 lowerer gate
#     (library/tests/test_attention_bwd_ir_cpp_parity.py, _cases / _plan):
#     every general shipped default and tuned-table entry on gfx942 and
#     gfx950 and the ws_bounds=buffer knob cases, planned here exactly as the
#     gate plans them;
#   * "@<case id>;<field>:<value>;...": that spec with knob overrides (the
#     spellings of attention_bwd_spec_emit.py);
#   * "<arch>|<field>:<value>;...": a spec from scratch.
# The C table holds each config resolved to "<arch>|<field>:<value>;..."
# (the planned spec's non-default fields, then the overrides); regenerate it
# with `attention_bwd_main_emit.py --print-c-configs` whenever the shipped
# tables or _CONFIGS change.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
import dataclasses
import importlib.util
import sys
from functools import cache
from pathlib import Path

from _emit_common import run_emit
from attention_bwd_spec_emit import _parse
from kernels.common.attention_bwd import AttnBwdSpec, _emit_main

_GATE = Path(__file__).resolve().parents[1] / "test_attention_bwd_ir_cpp_parity.py"
_ARCHES = ("gfx942", "gfx950")


@cache
def _gate():
    """The phase-1 gate module (loaded by path: its directory would shadow the
    library's ``dispatch`` package on sys.path)."""
    spec = importlib.util.spec_from_file_location("_attn_bwd_gate", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case_ids() -> list:
    """The general cases of the gate on gfx942 / gfx950, in its order."""
    return [
        cid
        for cid, family, arch, _recipe in _gate()._cases()
        if family == "general" and arch in _ARCHES
    ]


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, tuple):
        return f"{v[0]}x{v[1]}"
    return str(v)


def _spec_text(arch: str, spec: AttnBwdSpec) -> str:
    """``spec`` as an "<arch>|<field>:<value>;..." config (non-default fields)."""
    items = []
    for f in dataclasses.fields(spec):
        v = getattr(spec, f.name)
        if f.default is dataclasses.MISSING or v != f.default:
            items.append(f"{f.name}:{_fmt(v)}")
    return f"{arch}|" + ";".join(items)


@cache
def _case_text(cid: str) -> str:
    """The main spec the gate plans for case ``cid``, as a config."""
    gate = _gate()
    for case_id, family, arch, recipe in gate._cases():
        if case_id == cid:
            plan = gate._plan(family, arch, recipe)
            return _spec_text(arch, plan.specs["main"])
    raise SystemExit(f"unknown gate case {cid!r}")


def _resolve(config: str) -> str:
    """A config as "<arch>|<fields>" (gate cases planned, overrides last)."""
    if not config.startswith("@"):
        return config
    cid, _, over = config[1:].partition(";")
    text = _case_text(cid)
    return f"{text};{over}" if over else text


def _case(cid: str, over: str = "") -> str:
    return f"@{cid};{over}" if over else f"@{cid}"


# Knob variants on the shipped specs: every main-kernel branch the shipped
# tables leave out (see the comment of each group).
_T942 = "general/gfx942/tuned0"  # d128 bf16, head pack, ring 2 (vgpr)
_D942 = "general/gfx942/default0"  # d128 bf16, head pack, 8 waves
_T950 = "general/gfx950/tuned0"
_D950 = "general/gfx950/default0"
_BASE942 = "gfx942|head_size:64"
_BASE950 = "gfx950|head_size:128"
# 4 waves (256 threads), kN0 = 64, D = 32: the slices epilogue's half rows
# (128 chunks) leave a tail round
_TAIL = (
    "gfx942|head_size:32;waves:4;block_m:64;block_n:64;block_k4:32;"
    "warp_grid_g4:2x2;warp_grid_g02:1x4;warp_grid_g13:4x1"
)

# 4 waves, kM0 = kN0 = 32 on the small-tile grids: the LDS-staged direct
# epilogue's 32 x 32 / 8 = 128 chunks leave a tail round over 256 threads
_SMALL = (
    "gfx942|head_size:32;waves:4;block_m:32;block_n:32;block_k4:32;"
    "warp_grid_g4:2x2;warp_grid_g02:2x2;warp_grid_g13:2x2;pt_route:lds"
)


def _knobs() -> tuple:
    return (
        # the CTA decode: XCD remap (direct / atomic), interleave, mirror pairs
        ("xcd_remap direct gfx942 d64", f"{_BASE942};xcd_remap:True"),
        (
            "xcd_remap atomic gfx950 d128",
            f"{_BASE950};xcd_remap:True;dkv_mode:atomic",
        ),
        (
            "xcd_remap mirror interleave gfx942",
            f"{_BASE942};xcd_remap:True;lb_mirror:True;lb_interleave:True",
        ),
        ("lb_interleave gfx950", f"{_BASE950};lb_interleave:True"),
        ("lb_mirror gfx942 d64", f"{_BASE942};lb_mirror:True"),
        (
            "lb_mirror atomic thd gfx950",
            f"{_BASE950};lb_mirror:True;dkv_mode:atomic;seq_mode:thd",
        ),
        (
            "worklist thd gfx942",
            f"{_BASE942};seq_mode:thd;worklist:True;dkv_mode:atomic",
        ),
        (
            "rejected: worklist with mirror",
            f"{_BASE942};seq_mode:thd;worklist:True;lb_mirror:True",
        ),
        # atomic dK / dV: guards, sinks, epilogues, bounds, layouts
        ("atomic branch gfx942", f"{_BASE942};dkv_mode:atomic"),
        (
            "atomic select gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_atomic_guard:select",
        ),
        (
            "atomic token-major gfx942",
            f"{_BASE942};dkv_mode:atomic;ws_layout:token_major",
        ),
        (
            "atomic mul-free gfx942 runtime scale",
            f"{_BASE942};dkv_mode:atomic;static_scale:runtime",
        ),
        ("slices lds_staged gfx942", f"{_BASE942};dkv_mode:atomic;dkv_sink:slices"),
        (
            "slices lds_staged buffer gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;ws_bounds:buffer",
        ),
        (
            "slices pair gfx942",
            f"{_BASE942};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:lds_staged_pair",
        ),
        (
            "slices pair buffer gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:lds_staged_pair;ws_bounds:buffer",
        ),
        (
            "slices direct gfx942",
            f"{_BASE942};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct",
        ),
        (
            "slices direct buffer gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;ws_bounds:buffer",
        ),
        (
            "slices direct quad gfx942",
            f"{_BASE942};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;dkv_cell_vec:4",
        ),
        (
            "slices direct quad buffer gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;dkv_cell_vec:4;ws_bounds:buffer",
        ),
        (
            "slices acc_in_lds gfx942 d32",
            "gfx942|head_size:32;dkv_mode:atomic;dkv_sink:slices;acc_in_lds:True",
        ),
        (
            "slices acc_in_lds buffer gfx950 d32",
            "gfx950|head_size:32;dkv_mode:atomic;dkv_sink:slices;acc_in_lds:True;ws_bounds:buffer",
        ),
        (
            "slices vec 4 runtime scale gfx942",
            f"{_BASE942};dkv_mode:atomic;dkv_sink:slices;dkv_store_vec:4;static_scale:runtime",
        ),
        (
            "slices token-major gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;ws_layout:token_major",
        ),
        (
            "rejected: slices buffer token-major",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;ws_layout:token_major;ws_bounds:buffer",
        ),
        (
            "rejected: last_cta sink (not built)",
            f"{_BASE942};dkv_mode:atomic;dkv_sink:last_cta",
        ),
        # direct epilogues
        ("direct cells gfx942", f"{_BASE942};dkv_epilogue:direct"),
        ("direct quads gfx950", f"{_BASE950};dkv_epilogue:direct;dkv_cell_vec:4"),
        ("direct pair gfx942", f"{_BASE942};dkv_epilogue:lds_staged_pair"),
        ("direct vec 2 gfx950", f"{_BASE950};dkv_store_vec:2"),
        ("direct acc_in_lds gfx942 d32", "gfx942|head_size:32;acc_in_lds:True"),
        (
            "dkv_slab_hoist thd gfx950",
            f"{_BASE950};seq_mode:thd;dkv_slab_hoist:True",
        ),
        (
            "dkv_slab_hoist thd quads gfx942",
            f"{_BASE942};seq_mode:thd;dkv_slab_hoist:True;dkv_epilogue:direct;dkv_cell_vec:4",
        ),
        # step orders, edge tiles, rotation, carried slab offsets
        ("edge_tiles gfx942", f"{_BASE942};edge_tiles:True"),
        ("head_outer gfx942", f"{_BASE942};step_order:head_outer"),
        (
            "head_outer edge tiles (grouped) gfx950",
            f"{_BASE950};step_order:head_outer;edge_tiles:True",
        ),
        (
            "head_outer grouped head pack gfx942",
            f"{_BASE942};step_order:head_outer;edge_tiles:True;head_pack:True",
        ),
        (
            "head_outer grouped deferred dQ gfx950",
            f"{_BASE950};step_order:head_outer;edge_tiles:True;sched:stage_table_atomics",
        ),
        (
            "rejected: head_outer grouped ring 2",
            f"{_BASE942};step_order:head_outer;edge_tiles:True;ring_depth:2",
        ),
        (
            "head_outer grouped rotate deferred head pack gfx950",
            f"{_BASE950};step_order:head_outer;edge_tiles:True;q_rotate:True;sched:stage_table_atomics;head_pack:True",
        ),
        ("q_rotate gfx942", f"{_BASE942};q_rotate:True"),
        (
            "q_rotate head_outer gfx950",
            f"{_BASE950};q_rotate:True;step_order:head_outer",
        ),
        (
            "q_rotate grouped gfx942",
            f"{_BASE942};q_rotate:True;step_order:head_outer;edge_tiles:True",
        ),
        (
            "q_rotate ring 3 gfx950",
            f"{_BASE950};q_rotate:True;ring_depth:3",
        ),
        ("slab_carry gfx942", f"{_BASE942};slab_carry:True"),
        (
            "slab_carry edge tiles gfx950",
            f"{_BASE950};slab_carry:True;edge_tiles:True",
        ),
        (
            "slab_carry head_outer gfx942",
            f"{_BASE942};slab_carry:True;step_order:head_outer",
        ),
        (
            "slab_carry deferred dQ gfx950",
            f"{_BASE950};slab_carry:True;sched:stage_table_atomics",
        ),
        (
            "slab_carry grouped (not carried) gfx942",
            f"{_BASE942};slab_carry:True;step_order:head_outer;edge_tiles:True",
        ),
        (
            "rejected: slab_carry with q_rotate",
            f"{_BASE942};slab_carry:True;q_rotate:True",
        ),
        (
            "interior_clamp edge tiles gfx950",
            f"{_BASE950};interior_clamp:True;edge_tiles:True",
        ),
        # head packing, deferred dQ
        ("head_pack gfx942 d64", f"{_BASE942};head_pack:True"),
        (
            "head_pack edge tiles gfx950",
            f"{_BASE950};head_pack:True;edge_tiles:True",
        ),
        (
            "head_pack deferred dQ gfx942",
            f"{_BASE942};head_pack:True;sched:stage_table_atomics",
        ),
        ("stage_table_atomics gfx950", f"{_BASE950};sched:stage_table_atomics"),
        (
            "stage_table_atomics edge tiles gfx942",
            f"{_BASE942};sched:stage_table_atomics;edge_tiles:True",
        ),
        ("stage_table setprio gfx950", f"{_BASE950};sched:stage_table;setprio:True"),
        ("iglp1 gfx942", f"{_BASE942};sched:iglp1"),
        # counted waits
        ("counted gfx942", f"{_BASE942};loop_sync:counted"),
        (
            "counted late edge tiles gfx950",
            f"{_BASE950};loop_sync:counted;counted_issue:late;edge_tiles:True",
        ),
        (
            "counted deferred dQ gfx942",
            f"{_BASE942};loop_sync:counted;sched:stage_table_atomics",
        ),
        (
            "counted head pack gfx950",
            f"{_BASE950};loop_sync:counted;head_pack:True",
        ),
        (
            "counted head_outer gfx942",
            f"{_BASE942};loop_sync:counted;step_order:head_outer",
        ),
        (
            "counted rotate gfx950",
            f"{_BASE950};loop_sync:counted;q_rotate:True",
        ),
        ("loop_sync lds_only gfx942", f"{_BASE942};loop_sync:lds_only"),
        # the merged-barrier ring
        ("barrier merge gfx942", f"{_BASE942};barrier_merge:on;ring_depth:2"),
        (
            "barrier merge late edge tiles gfx950",
            f"{_BASE950};barrier_merge:on;ring_depth:2;ring_issue:late;edge_tiles:True",
        ),
        (
            "barrier merge head_outer gfx942",
            f"{_BASE942};barrier_merge:on;ring_depth:2;step_order:head_outer",
        ),
        (
            "barrier merge head pack gfx950",
            f"{_BASE950};barrier_merge:on;ring_depth:2;head_pack:True",
        ),
        (
            "barrier merge deferred gfx942",
            f"{_BASE942};barrier_merge:on;ring_depth:2;sched:stage_table_atomics",
        ),
        # the prefetch ring
        ("ring vgpr 2 gfx942", f"{_BASE942};ring_depth:2"),
        ("ring vgpr 3 gfx950", f"{_BASE950};ring_depth:3"),
        (
            "ring vgpr 2 edge tiles gfx942",
            f"{_BASE942};ring_depth:2;edge_tiles:True",
        ),
        (
            "ring vgpr 3 head_outer gfx950",
            f"{_BASE950};ring_depth:3;step_order:head_outer",
        ),
        (
            "ring vgpr 2 head pack deferred gfx942",
            f"{_BASE942};ring_depth:2;head_pack:True;sched:stage_table_atomics",
        ),
        ("ring dma 1 gfx950", f"{_BASE950};global_path:dma;ring_depth:1"),
        ("ring dma 2 gfx950", f"{_BASE950};global_path:dma;ring_depth:2"),
        (
            "ring dma 3 edge tiles gfx950",
            f"{_BASE950};global_path:dma;ring_depth:3;edge_tiles:True",
        ),
        (
            "ring dma 2 head_outer rotate gfx950",
            f"{_BASE950};global_path:dma;ring_depth:2;step_order:head_outer;q_rotate:True",
        ),
        (
            "ring dma 3 head_outer split dQ gfx950",
            f"{_BASE950};global_path:dma;ring_depth:3;step_order:head_outer;dq_mode:split",
        ),
        (
            "ring dma 2 mirror atomic gfx950",
            f"{_BASE950};global_path:dma;ring_depth:2;lb_mirror:True;dkv_mode:atomic",
        ),
        # kernel attributes, pointers, routes, masks, sequences
        ("ptr_noalias workspace gfx942", f"{_BASE942};ptr_noalias:workspace"),
        (
            "ptr_noalias all atomic gfx950",
            f"{_BASE950};ptr_noalias:all;dkv_mode:atomic",
        ),
        (
            "scheduler_strategy max-ilp waves_per_eu 2 gfx942",
            f"{_BASE942};scheduler_strategy:max-ilp;waves_per_eu:2",
        ),
        (
            "vgpr_form gfx942 d128",
            "gfx942|head_size:128;agpr_alloc:192x192;agpr_route:vgpr_form",
        ),
        (
            "agpr_fence gfx950 d128",
            f"{_BASE950};agpr_alloc:128x128;agpr_route:agpr_fence",
        ),
        ("agpr_alloc backend gfx942", f"{_BASE942};agpr_alloc:backend"),
        ("no mask gfx950", f"{_BASE950};mask_class:none;static_scale:runtime"),
        ("split dQ gfx942", f"{_BASE942};dq_mode:split"),
        ("thd gfx942", f"{_BASE942};seq_mode:thd"),
        ("stats_vec 4 gfx950", f"{_BASE950};stats_vec:4;stats_fma:True"),
        ("ws_bounds buffer gfx942", f"{_BASE942};ws_bounds:buffer"),
        ("rdna gfx1201 d64", "gfx1201|head_size:64"),
        (
            "rdna gfx1151 d64 bf16 atomic",
            "gfx1151|head_size:64;dtype:bf16;dkv_mode:atomic",
        ),
        ("rejected: unknown arch", "gfx1100|head_size:64"),
        # epilogue corners: multipliers on the slices paths, scalar buffer
        # stores, tail rounds (a tile the CTA's threads do not divide)
        (
            "slices direct runtime scale gfx942",
            f"{_BASE942};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;static_scale:runtime",
        ),
        (
            "slices direct runtime scale buffer gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;static_scale:runtime;ws_bounds:buffer",
        ),
        (
            "slices acc_in_lds runtime scale gfx942 d32",
            "gfx942|head_size:32;dkv_mode:atomic;dkv_sink:slices;acc_in_lds:True;static_scale:runtime",
        ),
        (
            "slices buffer vec 1 gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;ws_bounds:buffer;dkv_store_vec:1",
        ),
        ("tail rounds direct gfx942 d32", f"{_TAIL};dkv_store_vec:8"),
        (
            "tail rounds slices gfx942 d32",
            f"{_TAIL};dkv_mode:atomic;dkv_sink:slices",
        ),
        (
            "tail rounds slices buffer gfx942 d32",
            f"{_TAIL};dkv_mode:atomic;dkv_sink:slices;ws_bounds:buffer",
        ),
        # ring corners: short DMA rows, one wave, the joint register split
        ("ring dma 2 d32 gfx950", "gfx950|head_size:32;global_path:dma;ring_depth:2"),
        (
            "ring dma 2 one wave gfx950",
            "gfx950|head_size:32;global_path:dma;ring_depth:2;waves:1;block_m:16;block_n:16;warp_grid_g4:1x1;warp_grid_g02:1x1;warp_grid_g13:1x1;block_k4:16;atom_g4:16x16x16;atom_g02:16x16x16",
        ),
        ("ring vgpr 2 d32 joint gfx942", "gfx942|head_size:32;ring_depth:2"),
        ("ring vgpr 3 d32 joint gfx950", "gfx950|head_size:32;ring_depth:3"),
        (
            "ring vgpr 2 d32 head pack gfx942",
            "gfx942|head_size:32;ring_depth:2;head_pack:True",
        ),
        # schedules on the ring, DMA and counted bodies
        *(
            (f"sched {sc} {label}", f"{base};sched:{sc}")
            for sc in ("iglp0", "fences_only", "stage_table", "stage_table_atomics")
            for label, base in (
                ("dma 2 gfx950", f"{_BASE950};global_path:dma;ring_depth:2"),
                ("vgpr 2 gfx942", f"{_BASE942};ring_depth:2"),
                ("counted gfx942", f"{_BASE942};loop_sync:counted"),
            )
        ),
        (
            "stage_table setprio dma 3 gfx950",
            f"{_BASE950};global_path:dma;ring_depth:3;sched:stage_table;setprio:True",
        ),
        (
            "stage_table_atomics dma 3 edge tiles gfx950",
            f"{_BASE950};global_path:dma;ring_depth:3;edge_tiles:True;sched:stage_table_atomics",
        ),
        # knobs on shipped specs
        ("tuned gfx942 0 mirror", _case(_T942, "lb_mirror:True")),
        (
            "tuned gfx942 0 atomic slices",
            _case(_T942, "dkv_mode:atomic;dkv_sink:slices"),
        ),
        (
            "default gfx942 0 head_outer grouped",
            _case(_D942, "step_order:head_outer;edge_tiles:True"),
        ),
        ("tuned gfx950 0 edge tiles", _case(_T950, "edge_tiles:True")),
        ("default gfx950 0 atomic xcd", _case(_D950, "dkv_mode:atomic;xcd_remap:True")),
        # the R = 3 vgpr ring off the joint split (n_reg_chunks per tensor):
        # packed tiles, and a kM0 x D tile with as many chunks as threads
        ("ring vgpr 3 head pack gfx950", f"{_BASE950};ring_depth:3;head_pack:True"),
        (
            "ring vgpr 3 rotate head pack gfx950",
            f"{_BASE950};ring_depth:3;q_rotate:True;head_pack:True",
        ),
        ("ring vgpr 3 block_m 32 gfx950", f"{_BASE950};ring_depth:3;block_m:32"),
        (
            "ring vgpr 2 d128 block_m 32 gfx942",
            "gfx942|head_size:128;ring_depth:2;block_m:32",
        ),
        # LDS-staged direct epilogue with a tail round: the small-tile grids
        # (pt_route lds) allow kN0 = 32 under 4 waves, kN0 x D / vec < threads
        ("direct tail round gfx942 d32", _SMALL),
        ("direct pair tail round gfx942 d32", f"{_SMALL};dkv_epilogue:lds_staged_pair"),
        (
            "direct tail round gfx950 d64",
            "gfx950|head_size:64;waves:4;block_m:32;block_n:32;block_k4:32;warp_grid_g4:2x2;warp_grid_g02:2x2;warp_grid_g13:2x2;pt_route:lds",
        ),
        # slices store widths 2 and 1 in both bounds forms
        (
            "slices vec 2 clamp gfx942",
            f"{_BASE942};dkv_mode:atomic;dkv_sink:slices;dkv_store_vec:2",
        ),
        (
            "slices vec 2 buffer gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;dkv_store_vec:2;ws_bounds:buffer",
        ),
        (
            "slices pair buffer vec 1 gfx950",
            f"{_BASE950};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:lds_staged_pair;ws_bounds:buffer;dkv_store_vec:1",
        ),
        (
            "slices pair buffer vec 2 gfx942",
            f"{_BASE942};dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:lds_staged_pair;ws_bounds:buffer;dkv_store_vec:2",
        ),
        # stage_vec = 1: the narrow and scalar user-tensor accesses
        ("stage_vec 1 gfx942 d64", f"{_BASE942};stage_vec:1"),
        ("narrow 2 gfx942 d64", f"{_BASE942};stage_vec:1;narrow_vec:2"),
        ("narrow 4 bf16 gfx942 d64", f"{_BASE942};stage_vec:1;narrow_vec:4;dtype:bf16"),
        ("narrow 4 gfx950 d128", f"{_BASE950};stage_vec:1;narrow_vec:4"),
        (
            "narrow 2 bf16 gfx950 d128",
            f"{_BASE950};stage_vec:1;narrow_vec:2;dtype:bf16",
        ),
        (
            "narrow 4 atomic gfx950 d32",
            "gfx950|head_size:32;stage_vec:1;narrow_vec:4;dkv_mode:atomic",
        ),
        (
            "narrow 4 direct quads gfx942",
            f"{_BASE942};stage_vec:1;narrow_vec:4;dkv_epilogue:direct;dkv_cell_vec:4",
        ),
        (
            "narrow 4 direct cells gfx942",
            f"{_BASE942};stage_vec:1;narrow_vec:4;dkv_epilogue:direct",
        ),
        (
            "narrow 2 vec 1 gfx942",
            f"{_BASE942};stage_vec:1;narrow_vec:2;dkv_store_vec:1",
        ),
        (
            "narrow 4 thd ring 2 gfx950 d64",
            "gfx950|head_size:64;stage_vec:1;narrow_vec:4;seq_mode:thd;ring_depth:2",
        ),
        (
            "narrow 4 head pack gfx950 d64",
            "gfx950|head_size:64;stage_vec:1;narrow_vec:4;head_pack:True",
        ),
        # mirror over the THD decode in direct mode, AGPR reservations
        ("lb_mirror thd gfx942 d64", f"{_BASE942};lb_mirror:True;seq_mode:thd"),
        ("agpr_alloc 0x0 gfx942", f"{_BASE942};agpr_alloc:0x0"),
        ("agpr_alloc 128x128 gfx950", f"{_BASE950};agpr_alloc:128x128"),
        (
            "asm_g13 gfx950 d128",
            f"{_BASE950};agpr_alloc:128x128;agpr_route:asm_g13",
        ),
        (
            "rejected: split agpr_route on agpr_alloc backend",
            "gfx942|head_size:128;agpr_alloc:backend;agpr_route:vgpr_form",
        ),
        (
            "rejected: head_pack on a DMA ring",
            f"{_BASE950};global_path:dma;ring_depth:2;head_pack:True",
        ),
    )


def _configs() -> tuple:
    out = [(f"case {cid}", _case(cid)) for cid in _case_ids()]
    out += list(_knobs())
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    arch, fields = _parse(_resolve(_CONFIGS[idx][1]))
    return fields, arch


def build_attention_bwd_main_kernel(spec, *, arch: str):
    """The main kernel of one config (main_kernel_refusals=False)."""
    return _emit_main(AttnBwdSpec(**dict(spec)), arch, main_kernel_refusals=False)


def print_c_configs() -> None:
    for comment, config in _CONFIGS:
        c = comment.replace("*/", "* /")
        print(f'    /* {c} */\n    "{_resolve(config)}",')


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--print-c-configs":
        print_c_configs()
        return 0
    return run_emit(
        _spec,
        build_attention_bwd_main_kernel,
        usage="usage: attention_bwd_main_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
