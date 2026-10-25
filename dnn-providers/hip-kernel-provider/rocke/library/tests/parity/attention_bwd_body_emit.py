#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_body_emit.py -- Python reference emitter for the
# shared inner body of the attention backward main kernel
# (kernels/common/_attention_bwd_body.py: BwdTileGeometry, BwdLds, StatsView,
# BwdTileStep and its run_q_step, the K / V staging, the copy helpers and the
# dK / dV epilogue readers). The module has no kernel of its own, so this
# builds a probe kernel that drives the body the way the main kernel's
# _kv_tile does: the main kernel's kernargs, the sequence decode, the strided
# Q / dO / K / V addresses, the workspaces, the mask policy, the dQ sink, one
# flat LDS allocation at the planner's offsets, stage_kv, zero_acc, a few q
# steps (straight-line, no loop: the loop shell belongs to the main kernel),
# the deferred dQ flush, finish and one epilogue reader. The C twin
# attention_bwd_body_emit.c builds the same kernel through
# helper_rocke.instances.common._attention_bwd_body.h and the two .ll must
# match.
#
# A config is "<arch>|<AttnBwdSpec field>:<value>;..." (the spellings of
# attention_bwd_spec_emit.py) plus probe keys:
#   probe_steps  one letter per q step: "e" edge, "i" interior (default "e";
#                with head_pack the step is packed)
#   probe_epi    the epilogue reader after finish: "cells" (dkv_cells),
#                "rows" (dkv_rows, read once), "slots" (dkv_row_slots),
#                "quad" (dkv_quad_rows), "half" (dkv_f32_part, both halves),
#                "image" (dkv_f32_image read once), "none" (default: "cells",
#                "rows" with acc_in_lds)
#   probe_extra  "copy": after stage_kv, copy_packed_rows_to_lds of the Q
#                rows into the "q" buffer with 64 threads more than the CTA
#                (a predicated tail round); the body does not call it itself
# The spec is validated (validate_attn_bwd_spec, main_kernel_refusals=False)
# and the body runs its synchronous, merged-ring, counted or prefetch-ring
# step: ring_depth > 1 or global_path "dma" without the barrier merge stages
# through a QRing of _attention_bwd_ring as _cta_setup / _kv_tile build it (the
# prologue, one PrefetchStep per step with the loop counter a constant, the
# drain). The tuned-table geometries run twice: as tuned (their ring) and on
# the synchronous step (global_path "vgpr", ring_depth 1). Both engines refuse
# the same specs.
#
# _CONFIGS is built below; keep the C table in sync with
# `attention_bwd_body_emit.py --print-c-configs`.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
import sys

from _emit_common import run_emit
from attention_bwd_spec_emit import _CONFIGS as _SPEC_CONFIGS
from attention_bwd_spec_emit import _parse
from kernels.common._attention_bwd_addr import (
    PackedRows,
    StridedTensorAddr,
    WorkspaceAddr,
    decode_batched,
    decode_thd,
)
from kernels.common._attention_bwd_band import NoMaskKTail, RuntimeBand
from kernels.common._attention_bwd_body import (
    BwdLds,
    BwdTileGeometry,
    BwdTileStep,
    DkDvAcc,
    PackedStep,
    QStepInputs,
    RingStep,
    StatsView,
    copy_packed_rows_to_lds,
)
from kernels.common._attention_bwd_caps import BWD_BARRIER_FENCE, bwd_arch_caps
from kernels.common._attention_bwd_frag import ir_type
from kernels.common._attention_bwd_gemm import lane_and_wave
from kernels.common._attention_bwd_ring import PrefetchStep, PrefetchTile, QRing
from kernels.common._attention_bwd_sinks import (
    AtomicWorkspaceSink,
    NullSink,
    PackedAtomicWorkspaceSink,
)
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    _declare_main,
    lds_plan,
    static_unit_mults,
    validate_attn_bwd_spec,
)
from rocke.core.ir import F32, IRBuilder
from rocke.helpers.attention_band import AttnRuntimeBounds

_NAME = "attn_bwd_body_probe"

# The synchronous staging (the tuned prefetch ring replaced, see the header).
_SYNC = "global_path:vgpr;ring_depth:1"


def _spec_configs(prefix: str, *, sync: bool = True):
    """The shipped default and tuned-table specs of attention_bwd_spec_emit
    on gfx942 / gfx950, on the synchronous step (``sync``) or as given."""
    out = []
    for comment, config in _SPEC_CONFIGS:
        if not comment.startswith(prefix):
            continue
        if not config.startswith(("gfx942|", "gfx950|")):
            continue
        if sync:
            out.append((comment, f"{config};{_SYNC}"))
        else:
            out.append((f"{comment} (ring)", config))
    return out


_D942 = "gfx942|head_size:64"
_D950 = "gfx950|head_size:128"

# (comment, config) of the knob, edge and refusal configs
_KNOBS = (
    # two q steps: edge then interior, and the reverse
    ("steps ei gfx942 d64", f"{_D942};probe_steps:ei"),
    ("steps iee gfx950 d128", f"{_D950};probe_steps:iee"),
    ("steps ie gfx942 d64 edge tiles", f"{_D942};edge_tiles:True;probe_steps:ie"),
    # stage orders
    *(
        (f"stage_order {o} gfx942 d64", f"{_D942};stage_order:{o};probe_steps:ei")
        for o in ("g3g1g4", "g4g1g3", "g1g4g3", "g3g4g1", "g4g3g1")
    ),
    ("stage_order g4g3g1 gfx950 lds route", f"{_D950};stage_order:g4g3g1;pt_route:lds"),
    ("stage_order g3g1g4 split dQ", f"{_D942};stage_order:g3g1g4;dq_mode:split"),
    # copy orders (qdo_stage)
    ("qdo_stage loads_first gfx942", f"{_D942};qdo_stage:loads_first;probe_steps:ei"),
    ("qdo_stage joint gfx942", f"{_D942};qdo_stage:joint"),
    ("qdo_stage joint gfx950 d64", "gfx950|head_size:64;qdo_stage:joint"),
    (
        "qdo_stage loads_first gfx950 head pack",
        f"{_D950};qdo_stage:loads_first;head_pack:True",
    ),
    (
        "qdo_stage joint gfx942 block_m 32",
        "gfx942|head_size:64;qdo_stage:joint;block_m:32",
    ),
    # G4 slicing
    ("g4 slice gfx942 d64", f"{_D942};g4_slicing:slice;block_k4:32"),
    ("g4 slice_db gfx942 d64", f"{_D942};g4_slicing:slice_db;block_k4:32"),
    ("g4 slice_db gfx950 d128", f"{_D950};g4_slicing:slice_db;block_k4:64"),
    ("g4 slice_db_g3 gfx950 d128", f"{_D950};g4_slicing:slice_db_g3"),
    (
        "g4 slice_db_g3 g3g4g1 gfx942",
        f"{_D942};g4_slicing:slice_db_g3;stage_order:g3g4g1;block_k4:32",
    ),
    ("g4 slice gfx950 lds route", f"{_D950};g4_slicing:slice;pt_route:lds;block_k4:64"),
    # acc_in_lds and the dQ issue
    ("acc_in_lds gfx942 d32", "gfx942|head_size:32;acc_in_lds:True;probe_steps:ei"),
    (
        "acc_in_lds gfx950 d32 kt lds",
        "gfx950|head_size:32;acc_in_lds:True;kt_source:lds",
    ),
    (
        "acc_in_lds gfx950 d32 kv lds",
        "gfx950|head_size:32;acc_in_lds:True;kv_residency:lds;kt_source:lds",
    ),
    (
        "acc_in_lds gfx942 image epilogue",
        "gfx942|head_size:32;acc_in_lds:True;probe_epi:image",
    ),
    (
        "acc_in_lds atomic pair epilogue",
        "gfx942|head_size:32;acc_in_lds:True;dkv_mode:atomic;dkv_epilogue:lds_staged_pair;probe_epi:slots",
    ),
    ("dq_issue fragment gfx942", f"{_D942};dq_issue:fragment"),
    ("dq_issue fragment_regs gfx950", f"{_D950};dq_issue:fragment_regs"),
    # S in AGPRs (G0 / G2 asm clusters), asm knobs
    (
        "s_in_agpr gfx950 d32 asm gemm hoist",
        "gfx950|head_size:32;agpr_alloc:128x128;agpr_route:asm_g13;s_in_agpr:True;asm_tail_nop:8;asm_cluster:gemm;asm_load_hoist:True",
    ),
    (
        "s_in_agpr gfx942 d64 asm",
        f"{_D942};agpr_alloc:128x128;agpr_route:asm_g13;s_in_agpr:True",
    ),
    (
        "asm_g13 gfx942 tail nop 7",
        f"{_D942};agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:7",
    ),
    (
        "asm_g13 gfx942 d128 gemm cluster",
        "gfx942|head_size:128;agpr_alloc:192x192;agpr_route:asm_g13;asm_cluster:gemm",
    ),
    (
        "asm_g13 hoist g3g1g4",
        f"{_D942};agpr_alloc:128x128;agpr_route:asm_g13;asm_load_hoist:True;stage_order:g3g1g4",
    ),
    (
        "agpr_fence gfx950 d128",
        f"{_D950};agpr_alloc:128x128;agpr_route:agpr_fence;probe_steps:ei",
    ),
    (
        "vgpr_form gfx942 d128",
        "gfx942|head_size:128;agpr_alloc:192x192;agpr_route:vgpr_form",
    ),
    # stats loads
    ("stats_load step_top gfx942", f"{_D942};stats_load:step_top;probe_steps:ei"),
    (
        "stats_load step_top gfx950 head pack",
        f"{_D950};stats_load:step_top;head_pack:True",
    ),
    ("ws_bounds buffer gfx942", f"{_D942};ws_bounds:buffer;probe_steps:ei"),
    ("stats_vec 4 stats_fma gfx950", f"{_D950};stats_vec:4;stats_fma:True"),
    ("stats_vec 2 gfx942", f"{_D942};stats_vec:2"),
    ("token-major workspace gfx942", f"{_D942};ws_layout:token_major;probe_steps:ei"),
    # atomics placement: the deferred dQ atomics, the counted waits
    (
        "stage_table_atomics gfx942",
        f"{_D942};sched:stage_table_atomics;probe_steps:eie",
    ),
    (
        "stage_table_atomics gfx950 head pack",
        f"{_D950};sched:stage_table_atomics;head_pack:True;probe_steps:ee",
    ),
    (
        "stage_table gfx942 setprio",
        f"{_D942};sched:stage_table;setprio:True;probe_steps:ei",
    ),
    ("fences_only gfx950 kt lds", f"{_D950};sched:fences_only;kt_source:lds"),
    ("iglp0 gfx942", f"{_D942};sched:iglp0"),
    ("iglp1 gfx950", f"{_D950};sched:iglp1"),
    ("stage_table split dQ gfx942", f"{_D942};sched:stage_table;dq_mode:split"),
    ("counted early gfx942", f"{_D942};loop_sync:counted;probe_steps:ei"),
    (
        "counted late gfx950",
        f"{_D950};loop_sync:counted;counted_issue:late;probe_steps:ee",
    ),
    (
        "counted early deferred gfx950",
        f"{_D950};loop_sync:counted;sched:stage_table_atomics;probe_steps:ee",
    ),
    ("counted gfx942 head pack", f"{_D942};loop_sync:counted;head_pack:True"),
    ("loop_sync lds_only gfx942", f"{_D942};loop_sync:lds_only;probe_steps:ei"),
    # the merged-barrier ring
    (
        "barrier merge early gfx942",
        f"{_D942};barrier_merge:on;ring_depth:2;probe_steps:eie",
    ),
    (
        "barrier merge late gfx950",
        f"{_D950};barrier_merge:on;ring_depth:2;ring_issue:late;probe_steps:ee",
    ),
    (
        "barrier merge stage_table gfx942",
        f"{_D942};barrier_merge:on;ring_depth:2;sched:stage_table;probe_steps:ee",
    ),
    (
        "barrier merge deferred gfx950",
        f"{_D950};barrier_merge:on;ring_depth:2;sched:stage_table_atomics;probe_steps:ee",
    ),
    (
        "barrier merge head pack gfx942",
        f"{_D942};barrier_merge:on;ring_depth:2;head_pack:True",
    ),
    # the Q / dO prefetch ring (QRing, PrefetchStep)
    ("ring vgpr R2 gfx942", f"{_D942};ring_depth:2;probe_steps:eie"),
    ("ring vgpr R3 gfx950 d64", "gfx950|head_size:64;ring_depth:3;probe_steps:eiee"),
    ("ring vgpr R3 one step gfx950", f"{_D950};ring_depth:3"),
    (
        "ring vgpr R2 head pack gfx942",
        f"{_D942};ring_depth:2;head_pack:True;probe_steps:ee",
    ),
    (
        "ring vgpr R3 head pack gfx950",
        f"{_D950};ring_depth:3;head_pack:True;probe_steps:ee",
    ),
    (
        "ring vgpr R2 stage_table gfx942",
        f"{_D942};ring_depth:2;sched:stage_table;probe_steps:ee",
    ),
    (
        "ring vgpr R3 deferred gfx950",
        f"{_D950};ring_depth:3;sched:stage_table_atomics;probe_steps:eee",
    ),
    (
        "ring vgpr R2 lds route gfx950",
        "gfx950|head_size:64;ring_depth:2;pt_route:lds;probe_steps:ie",
    ),
    ("ring dma R1 gfx950", f"{_D950};global_path:dma;ring_depth:1;probe_steps:ei"),
    ("ring dma R2 gfx950", f"{_D950};global_path:dma;ring_depth:2;probe_steps:eie"),
    ("ring dma R3 gfx950", f"{_D950};global_path:dma;ring_depth:3;probe_steps:eiee"),
    (
        "ring dma R2 d64 bf16 gfx950",
        "gfx950|head_size:64;dtype:bf16;global_path:dma;ring_depth:2;probe_steps:ee",
    ),
    (
        "ring dma R2 stage_table gfx950",
        f"{_D950};global_path:dma;ring_depth:2;sched:stage_table;probe_steps:ee",
    ),
    (
        "ring dma R2 deferred gfx950",
        f"{_D950};global_path:dma;ring_depth:2;sched:stage_table_atomics;probe_steps:ee",
    ),
    (
        "ring dma R3 split dQ gfx950",
        f"{_D950};global_path:dma;ring_depth:3;dq_mode:split;probe_steps:ee",
    ),
    (
        "ring dma R2 d32 short rows gfx950",
        "gfx950|head_size:32;global_path:dma;ring_depth:2;probe_steps:ee",
    ),
    (
        "ring dma R2 one wave 16x16 gfx950",
        "gfx950|head_size:32;global_path:dma;ring_depth:2;waves:1;block_m:16;block_n:16;warp_grid_g4:1x1;warp_grid_g02:1x1;warp_grid_g13:1x1;block_k4:16;atom_g4:16x16x16;atom_g02:16x16x16",
    ),
    (
        "ring vgpr R2 d32 joint gfx942",
        "gfx942|head_size:32;ring_depth:2;probe_steps:ee",
    ),
    (
        "ring vgpr R3 d32 joint gfx950",
        "gfx950|head_size:32;ring_depth:3;probe_steps:eee",
    ),
    (
        "one wave 16x16 gfx942",
        "gfx942|head_size:64;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;warp_grid_g02:1x1;warp_grid_g13:1x1;probe_steps:ei",
    ),
    (
        "rejected: dma ring with head pack",
        f"{_D950};global_path:dma;ring_depth:2;head_pack:True",
    ),
    # transpose sources, routes, residency
    ("xt_lds gfx942 d64", f"{_D942};transpose_source:xt_lds;probe_steps:ei"),
    (
        "xt_lds gfx942 d128 kt lds",
        "gfx942|head_size:128;transpose_source:xt_lds;kt_source:lds",
    ),
    ("xt_lds gfx950 d128 split dQ", f"{_D950};transpose_source:xt_lds;dq_mode:split"),
    (
        "xt_lds gfx950 lds route",
        "gfx950|head_size:64;transpose_source:xt_lds;pt_route:lds",
    ),
    ("lds_plain gfx950 d64", "gfx950|head_size:64;transpose_source:lds_plain"),
    ("pt_route lds gfx942 d64", f"{_D942};pt_route:lds;probe_steps:ei"),
    ("kv_residency lds gfx942", f"{_D942};kv_residency:lds;kt_source:lds"),
    ("kt_source lds gfx950 d64", "gfx950|head_size:64;kt_source:lds"),
    (
        "replicated G0 grid gfx942",
        "gfx942|head_size:64;pt_route:lds;waves:4;block_n:64;warp_grid_g02:1x2;warp_grid_g4:1x4",
    ),
    # masks, dtypes, sequences
    (
        "no mask gfx942 runtime scale",
        f"{_D942};mask_class:none;static_scale:runtime;probe_steps:ei",
    ),
    (
        "no mask gfx950 dq before atomic",
        f"{_D950};mask_class:none;static_scale:dq_before_atomic",
    ),
    ("bf16 gfx942 rne_noguard", f"{_D942};dtype:bf16;bf16_cast:rne_noguard"),
    ("bf16 gfx950 fast exp2", f"{_D950};dtype:bf16;bwd_exp2:fast"),
    ("thd atomic gfx942", f"{_D942};seq_mode:thd;dkv_mode:atomic;probe_epi:slots"),
    (
        "thd gfx950 head pack interior clamp",
        f"{_D950};seq_mode:thd;head_pack:True;interior_clamp:True;probe_steps:ei",
    ),
    ("interior clamp gfx942", f"{_D942};interior_clamp:True;probe_steps:i"),
    ("stage_vec 1 gfx942", f"{_D942};stage_vec:1;narrow_vec:2"),
    ("head pack gfx942 d64", f"{_D942};head_pack:True;probe_steps:ee"),
    # epilogue readers
    ("epilogue rows gfx942", f"{_D942};probe_epi:rows"),
    ("epilogue quad gfx950", f"{_D950};probe_epi:quad"),
    ("epilogue half gfx942", f"{_D942};probe_epi:half"),
    (
        "epilogue half gfx950 g13 grid 2x2",
        "gfx950|head_size:64;pt_route:lds;waves:4;warp_grid_g13:2x2;warp_grid_g4:2x2;probe_epi:half",
    ),
    ("epilogue slots gfx950", f"{_D950};probe_epi:slots"),
    ("epilogue none gfx942", f"{_D942};probe_epi:none"),
    ("packed copy helper gfx942 d64", f"{_D942};probe_extra:copy"),
    ("packed copy helper gfx950 d128 bf16", f"{_D950};dtype:bf16;probe_extra:copy"),
    (
        "epilogue rows acc_in_lds runtime scale",
        "gfx942|head_size:32;acc_in_lds:True;static_scale:runtime;probe_epi:rows",
    ),
    (
        "epilogue rows vec 4 gfx950",
        f"{_D950};dkv_store_vec:4;static_scale:runtime;probe_epi:rows",
    ),
    (
        "epilogue half runtime scale gfx942",
        f"{_D942};static_scale:runtime;probe_epi:half",
    ),
    (
        "epilogue quad runtime scale gfx942",
        f"{_D942};static_scale:runtime;probe_epi:quad",
    ),
    (
        "barrier merge gfx950 d128 block_m 32",
        f"{_D950};barrier_merge:on;ring_depth:2;block_m:32;probe_steps:ei",
    ),
    # refusals
    ("rejected: unknown stage order", f"{_D942};stage_order:g1g1g4"),
    (
        "rejected: g4 slicing with acc_in_lds",
        "gfx942|head_size:32;g4_slicing:slice;acc_in_lds:True",
    ),
    (
        "rejected: quad epilogue with acc_in_lds",
        "gfx942|head_size:32;acc_in_lds:True;probe_epi:quad",
    ),
    ("rejected: image epilogue without acc_in_lds", f"{_D942};probe_epi:image"),
    ("rdna gfx1201 d64", "gfx1201|head_size:64;probe_steps:ei"),
    ("rdna gfx1151 d64 bf16", "gfx1151|head_size:64;dtype:bf16"),
)


# knob sweeps (each value on a base it is legal on)
_SWEEPS = (
    *(
        (
            f"sweep stage_order {o} gfx950 lds route",
            f"{_D950};pt_route:lds;stage_order:{o};probe_steps:ie",
        )
        for o in ("g1g3g4", "g3g1g4", "g4g1g3", "g1g4g3", "g3g4g1", "g4g3g1")
    ),
    *(
        (
            f"sweep qdo_stage {c} gfx942 head pack {hp}",
            f"{_D942};qdo_stage:{c};head_pack:{hp};probe_steps:ee",
        )
        for c in ("split", "loads_first", "joint")
        for hp in ("False", "True")
    ),
    *(
        (
            f"sweep g4_slicing {g4} gfx950 d64",
            f"gfx950|head_size:64;g4_slicing:{g4};block_k4:32;stage_order:g1g3g4",
        )
        for g4 in ("step", "slice", "slice_db", "slice_db_g3")
    ),
    *(
        (
            f"sweep sched {sc} gfx950 d128 bf16",
            f"{_D950};dtype:bf16;sched:{sc};probe_steps:ee",
        )
        for sc in (
            "none",
            "iglp0",
            "iglp1",
            "fences_only",
            "stage_table",
            "stage_table_atomics",
        )
    ),
    *(
        (
            f"sweep loop_sync {ls} gfx950 d64 kt lds",
            f"gfx950|head_size:64;kt_source:lds;loop_sync:{ls};probe_steps:ei",
        )
        for ls in ("full", "lds_only", "counted")
    ),
    *(
        (
            f"sweep dq_issue {dq} acc_in_lds {al} gfx942 d32",
            f"gfx942|head_size:32;dq_issue:{dq};acc_in_lds:{al}",
        )
        for dq in ("block", "fragment", "fragment_regs")
        for al in ("False", "True")
    ),
    *(
        (
            f"sweep mask {mk} edge_tiles {et} gfx942 d128",
            f"gfx942|head_size:128;mask_class:{mk};edge_tiles:{et};probe_steps:ei",
        )
        for mk in ("band", "none")
        for et in ("False", "True")
    ),
    *(
        (
            f"sweep dtype {dt} transpose {ts} gfx950 d64",
            f"gfx950|head_size:64;dtype:{dt};transpose_source:{ts}",
        )
        for dt in ("fp16", "bf16")
        for ts in ("tr_read", "lds_plain", "xt_lds")
    ),
)


# review follow-ups: the "loads_first" rounds with no tail (a tile of at
# least one chunk per thread), the merged ring on a non-default stage order,
# and refusals that pin the reject agreement (unknown arch, unknown knob
# value, the counted body with G4 slicing or loads_first, a merged ring of
# depth 3). run_diff enumerates at most MAX_CFG = 256 configs per family:
# this family is at the cap, so a new config must replace one.
_REVIEW = (
    (
        "qdo_stage loads_first full rounds gfx942 d128 block_m 32",
        "gfx942|head_size:128;qdo_stage:loads_first;block_m:32;probe_steps:ei",
    ),
    (
        "qdo_stage loads_first full rounds gfx942 d64 four waves",
        "gfx942|head_size:64;qdo_stage:loads_first;block_m:64;waves:4;warp_grid_g4:1x4;warp_grid_g02:1x4;warp_grid_g13:4x1",
    ),
    (
        "qdo_stage loads_first full rounds gfx950 d128 head pack",
        f"{_D950};qdo_stage:loads_first;block_m:32;head_pack:True;probe_steps:ee",
    ),
    (
        "barrier merge g3g1g4 gfx942",
        f"{_D942};barrier_merge:on;ring_depth:2;stage_order:g3g1g4;probe_steps:ei",
    ),
    ("rejected: unknown arch gfx1100", "gfx1100|head_size:64"),
    ("rejected: unknown transpose_source", f"{_D942};transpose_source:bogus"),
    (
        "rejected: counted with g4 slice",
        f"{_D942};loop_sync:counted;g4_slicing:slice;block_k4:32",
    ),
    (
        "rejected: counted with loads_first",
        f"{_D942};loop_sync:counted;qdo_stage:loads_first",
    ),
    ("rejected: barrier merge ring depth 3", f"{_D942};barrier_merge:on;ring_depth:3"),
)


def _configs():
    out = []
    out += _spec_configs("default ")
    out += _spec_configs("tuned ")
    out += _spec_configs("tuned ", sync=False)
    out += list(_KNOBS)
    out += list(_SWEEPS)
    out += list(_REVIEW)
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    arch, fields = _parse(_CONFIGS[idx][1])
    return fields, arch


def _side(b, s, a, bz, side):
    """The q-side or kv-side SeqInfo (attention_bwd._decode_side)."""
    smax, off, seq = {
        "q": ("S_q_max", "OFF_Q", "SEQ_Q"),
        "kv": ("S_kv_max", "OFF_KV", "SEQ_KV"),
    }[side]
    if s.seq_mode == "thd":
        return decode_thd(
            b, bz, s_max=a[smax], off_ptr=a[off], off64=a["off64"],
            mult=a[f"{side}_mult"], div=a[f"{side}_div"], seq_ptr=a[seq],
            has_len=a["has_len"], len_stride=a["len_stride"], ws_seg=a["ws_seg"],
        )  # fmt: skip
    return decode_batched(
        b, bz, s_max=a[smax], seq_ptr=a[seq], has_len=a["has_len"],
        len_stride=a["len_stride"],
    )  # fmt: skip


def build_attention_bwd_body_kernel(spec, *, arch: str):
    """The probe kernel of one config."""
    fields = dict(spec)
    steps = fields.pop("probe_steps", "e")
    extra = fields.pop("probe_extra", None)
    s = AttnBwdSpec(**{k: v for k, v in fields.items() if k != "probe_epi"})
    s = validate_attn_bwd_spec(s, arch, main_kernel_refusals=False)
    epi = fields.get("probe_epi", "rows" if s.acc_in_lds else "cells")
    geom = BwdTileGeometry.from_spec(s, arch)
    plan = lds_plan(s, arch)
    io = ir_type(s.dtype)
    D, km, kn = s.head_size, s.block_m, s.block_n

    b = IRBuilder(_NAME)
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare_main(b, io)
    a.update(dict.fromkeys(static_unit_mults(s.static_scale, s.dkv_mode)))
    tid, lane, wave = lane_and_wave(b, bwd_arch_caps(arch).wave_size)
    kt, hy, bz = b.block_id_x(), b.block_id_y(), b.block_id_z()
    seq_q = _side(b, s, a, bz, "q")
    seq_kv = _side(b, s, a, bz, "kv")
    hk = hy
    h0 = b.mul(hy, a["G"])
    n_h = a["G"]

    def tensor(name, prefix, seq):
        return StridedTensorAddr(
            b, a[name], dtype=io, seq=seq, stage_vec=s.stage_vec,
            strides=(a[f"{prefix}_b"], a[f"{prefix}_h"], a[f"{prefix}_t"]),
            narrow_vec=s.narrow_vec,
        )  # fmt: skip

    q_addr = tensor("Q", "q", seq_q)
    do_addr = tensor("dO", "do", seq_q)
    k_addr = tensor("K", "k", seq_kv)
    v_addr = tensor("V", "v", seq_kv)
    nh = a["h_q"] if s.ws_layout == "token_major" else None

    def ws(name, width):
        return WorkspaceAddr(
            b, a[name], ws_rows=a["ws_rows_q"], width=width, row0=seq_q.ws_row0,
            layout=s.ws_layout, n_heads=nh,
        )  # fmt: skip

    stats = StatsView(
        ws("WS_LSE2", 1), ws("WS_DSUM", 1),
        n_heads=a["h_q"] if s.stats_vec != 1 else None,
        bounds=s.ws_bounds, span=km,
    )  # fmt: skip
    if s.dq_mode == "split":
        sink = NullSink()
    else:
        sink_cls = PackedAtomicWorkspaceSink if s.head_pack else AtomicWorkspaceSink
        sink = sink_cls(b, ws("WS_DQ", D), seq=seq_q, scale=a["dq_mult"])
    zero = b.const_i32(0)
    if s.mask_class == "band":
        off = b.select(
            b.cmp_ne(a["bottom_right"], zero),
            b.sub(seq_kv.length, seq_q.length),
            zero,
        )
        bounds = AttnRuntimeBounds(
            seq_q.length, seq_kv.length, off, a["left"], a["right"]
        )
        mask = RuntimeBand(b, bounds, k_m0=km, k_n0=kn)
    else:
        mask = NoMaskKTail(b, seq_q.length, seq_kv.length, k_m0=km, k_n0=kn)
    lds = BwdLds(b, plan, s.dtype)
    step = BwdTileStep(
        b, geom, mask=mask, stats=stats, dq_sink=sink, lds=lds, tid=tid, lane=lane,
        wave=wave, scale_log2=a["scale_log2"], ds_mult=a["ds_mult"],
    )  # fmt: skip
    ring = None
    if (s.global_path, s.ring_depth) != ("vgpr", 1) and s.barrier_merge != "on":
        # _cta_setup: the Q / dO prefetch ring
        n_atomics = km * D // (geom.waves * geom.wave_size)
        ring = QRing(
            b, geom, lds, tid=tid, lane=lane, wave=wave, stats=stats,
            n_atomics=n_atomics if s.dq_mode == "atomic" else 0,
        )  # fmt: skip
        ring.check_step(step.sched)

    # _kv_tile: the staging, then the q steps
    one = b.const_i32(1)
    k0 = b.mul(kt, b.const_i32(kn))
    live = b.cmp_lt(k0, seq_kv.length)
    k_slab = k_addr.slab(hk, k0)
    v_slab = v_addr.slab(hk, k0)
    step.stage_kv(b, k_slab, v_slab, k0, live)
    if extra == "copy":
        rows = PackedRows(
            head0=h0, n_heads=n_h, rows_per_head=b.const_i32(km), row0=zero
        )
        copy_packed_rows_to_lds(
            b, q_addr, lds.view("q", D), rows, rows=km, cols=D, tid=tid,
            threads=geom.threads + 64, clamp=False,
        )  # fmt: skip
    elif extra is not None:
        raise ValueError(f"unknown probe_extra {extra!r}")
    if s.head_pack:
        pk = a["pack_heads"]
        pack_on = b.cmp_ne(pk, zero)
        per = b.select(pack_on, pk, one)
        rows_per_head = b.select(pack_on, b.smax(seq_q.length, one), b.const_i32(km))
    qt_start, _n_qt = mask.q_tile_range(b, k0)
    acc = step.zero_acc(b)
    pend = None
    if s.sched == "stage_table_atomics":
        if s.head_pack:
            pend = step.zero_pending_dq(b, h0, one, rows_per_head)
        else:
            pend = step.zero_pending_dq(b, h0)

    def inputs(i: int, *, edge: bool, slabs: bool = True) -> QStepInputs:
        qt = qt_start if i == 0 else b.add(qt_start, b.const_i32(i))
        q0 = b.mul(qt, b.const_i32(km))
        if s.head_pack:
            rows = PackedRows(
                head0=h0, n_heads=b.smin(per, n_h), rows_per_head=rows_per_head,
                row0=q0,
            )  # fmt: skip
            packed = PackedStep(rows=rows, q_addr=q_addr, do_addr=do_addr)
            return QStepInputs(q0=q0, h=h0, q_slab=None, do_slab=None, packed=packed)
        if not slabs:
            return QStepInputs(q0=q0, h=h0, q_slab=None, do_slab=None, packed=None)
        clamp = edge or s.interior_clamp
        return QStepInputs(
            q0=q0, h=h0, q_slab=q_addr.slab(h0, q0, clamp=clamp),
            do_slab=do_addr.slab(h0, q0, clamp=clamp), packed=None,
        )  # fmt: skip

    edges = [c == "e" for c in steps]
    n = len(edges)
    counted = s.loop_sync == "counted"
    merge = s.barrier_merge == "on"

    def step_loads(i: int):
        t = inputs(i, edge=edges[i])
        return step.counted_loads(
            b, h=t.h, q0=t.q0, q_slab=t.q_slab, do_slab=t.do_slab, packed=t.packed,
            edge=edges[i],
        )  # fmt: skip

    def ring_tile(i: int) -> PrefetchTile:
        t = inputs(i, edge=edges[i])
        return PrefetchTile(t.q0, t.h, t.q_slab, t.do_slab, t.packed)

    staged = None
    carried = []
    if ring is not None:
        tiles = [ring_tile(0)]
        if ring.depth == 3:
            tiles.append(ring_tile(1 if n > 1 else 0))
        if ring.path == "dma" and ring.depth == 1:
            tiles = []
        carried = ring.prologue(b, _n_qt, tiles, name="bwd_q")
    if merge:
        step.ring_prologue(b, inputs(0, edge=edges[0]))
        step.ring_publish(b)
    if counted:
        # (the main kernel's guard loop starts from these zeros)
        step.counted_zero(b)
        staged = step_loads(0)
        b.s_waitcnt(vmcnt=0)
    for i, edge in enumerate(edges):
        nxt_i = i + 1 if i + 1 < n else i
        if pend is None:
            step.sink.begin_head(b, h0)
        if counted:
            here = inputs(i, edge=edge, slabs=False)
            qt = qt_start if i == 0 else b.add(qt_start, b.const_i32(i))
            out, carry = step.run_q_step(
                b, acc, None, None, qt, here.h, k0, edge=edge, packed=here.packed,
                staged=staged, next_loads=lambda j=nxt_i: step_loads(j), pending=pend,
            )  # fmt: skip
            if pend is not None:
                staged, pend = carry
            else:
                staged = carry
            acc = out
            continue
        here = inputs(i, edge=edge)
        qt = qt_start if i == 0 else b.add(qt_start, b.const_i32(i))
        merged = None
        if merge:
            slot = b.mod(b.const_i32(i), b.const_i32(2))
            merged = RingStep(slot=slot, nxt=inputs(nxt_i, edge=edges[nxt_i]))
        pf = None
        if ring is not None:
            it = b.const_i32(i)
            tile = PrefetchTile(here.q0, here.h, here.q_slab, here.do_slab, here.packed)
            ahead = tile
            k = ring.schedule["ahead"]
            if k:
                ahead = ring_tile(min(i + k, n - 1))
            pf = PrefetchStep(
                ring, it=it, here=tile, ahead=ahead, carried=list(carried)
            )
        out = step.run_q_step(
            b, acc, here.q_slab, here.do_slab, qt, here.h, k0, edge=edge,
            packed=here.packed, ring=merged, prefetch=pf, pending=pend,
        )  # fmt: skip
        if pf is not None:
            carried = pf.yields
        if pend is not None:
            out, pend = out
        acc = out
        if acc.dk:  # the loop-carried form (no IR)
            g1 = geom.g1
            acc = DkDvAcc.from_flat(acc.flat(), g1.frags_m, g1.frags_n)
    if ring is not None:
        ring.drain(b)
    if pend is not None:
        step.flush_dq(b, pend)
    acc = step.finish(b, acc)
    mult = a["dk_mult"]
    if epi == "cells":
        step.dkv_cells(b, "dk", acc, mult)
        step.dkv_cells(b, "dv", acc, None)
    elif epi == "rows":
        read = step.dkv_rows(b, "dk", acc, mult, vec=s.dkv_store_vec)
        read(tid, b.const_i32(0))
    elif epi == "slots":
        step.dkv_row_slots(b, "dk", acc)
        step.dkv_row_slots(b, "dv", acc)
    elif epi == "quad":
        step.dkv_quad_rows(b, "dk", acc, mult)
    elif epi == "half":
        for half in (0, 1):
            step.dkv_f32_part(b, "dv", acc, mult, half=half)
            b.sync()
    elif epi == "image":
        view = step.dkv_f32_image("dk")
        view.load(b, tid, 0, dtype=F32, n=1)
    elif epi != "none":
        raise ValueError(f"unknown probe_epi {epi!r}")
    b.ret()
    return b.kernel


def print_c_configs(configs) -> None:
    for comment, config in configs:
        c = comment.replace("*/", "* /")
        print(f'    /* {c} */\n    "{config}",')


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--print-c-configs":
        print_c_configs(_CONFIGS)
        return 0
    return run_emit(
        _spec,
        build_attention_bwd_body_kernel,
        usage="usage: attention_bwd_body_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
