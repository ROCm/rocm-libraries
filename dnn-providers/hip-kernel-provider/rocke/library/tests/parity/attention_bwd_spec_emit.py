#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_spec_emit.py -- Python reference for the C port
# of the attention backward main kernel's spec and decisions
# (kernels/common/attention_bwd.py: AttnBwdSpec, resolved, kernel_name, the
# default geometry policy, validate_attn_bwd_spec, the LDS plan of a spec,
# the register estimates and the occupancy bound). These emit no IR, so
# this emitter prints a canonical text dump of every derived decision of one
# spec; the C twin attention_bwd_spec_emit.c prints the same dump from
# rocke/instance_attention_bwd.h and run_diff compares the two byte for
# byte (sha256), like a .ll.
#
# A config is "<arch>|<field>:<value>;..." (the AttnBwdSpec fields given; a
# value is an int, True / False, None, "RxC" for a warp grid / agpr_alloc
# pair, or a string). The dump prints the spec's __post_init__ verdict, its
# kernel names, the default geometry of its problem, the resolved spec field
# by field, the validate verdict (with main_kernel_refusals=False: the
# resource-gate table stays in Python), and, for the resolved spec, the LDS
# request and plan (every buffer, layout and ring stride), the register
# estimates, the AGPR copy budget, the counted-sync check, the occupancy,
# the AGPR policy, the static unit multipliers and the aux kernel names. An
# exception prints as "<type>: <str(exc)>", so both engines must refuse at
# the same rule with the same message.
#
# Configs: the shipped defaults on gfx942 / gfx950 (and the RDNA targets),
# every general tuned-table entry of dispatch/attention_bwd on gfx942 /
# gfx950 (the knobs as the plan passes them to the spec: runtime, aux and
# dQ knobs dropped, scale_placement as static_scale, lb_order and
# use_worklist as their decode switches; GQA entries also in atomic mode),
# then knob, edge and rejected specs. Keep _CONFIGS and the C table in sync:
# `attention_bwd_spec_emit.py --print-c-configs` prints the C table.
#
# argv[2] selects the output mode; only "ll" (the dump) is supported.
import dataclasses
import sys

from kernels.common._attention_bwd_tiles import attn_bwd_occupancy
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    _policy_agpr_alloc,
    agpr_copy_budget,
    attn_bwd_default_geometry,
    check_counted_sync,
    lds_plan,
    lds_request,
    register_estimates,
    static_unit_mults,
    validate_attn_bwd_spec,
)

_INT = {
    "head_size",
    "stage_vec",
    "dkv_store_vec",
    "stats_vec",
    "dkv_cell_vec",
    "narrow_vec",
}
_OPT_INT = {
    "waves",
    "block_m",
    "block_n",
    "block_k4",
    "ring_depth",
    "waves_per_eu",
    "asm_tail_nop",
}
_BOOL = {
    "xcd_remap", "asm_load_hoist", "q_rotate", "lb_interleave", "dkv_slab_hoist",
    "interior_clamp", "lb_mirror", "stats_fma", "worklist", "slab_carry",
    "setprio", "edge_tiles", "head_pack", "acc_in_lds", "s_in_agpr",
}  # fmt: skip
_PAIR = {"warp_grid_g4", "warp_grid_g02", "warp_grid_g13", "agpr_alloc"}

# (comment, config)
_CONFIGS = (
    ("default gfx942 d32 fp16 sv8", "gfx942|head_size:32;dtype:fp16;stage_vec:8"),
    ("default gfx942 d32 bf16 sv8", "gfx942|head_size:32;dtype:bf16;stage_vec:8"),
    ("default gfx942 d32 bf16 sv1", "gfx942|head_size:32;dtype:bf16;stage_vec:1"),
    (
        "default gfx942 d32 thd atomic",
        "gfx942|head_size:32;seq_mode:thd;dkv_mode:atomic",
    ),
    ("default gfx942 d64 fp16 sv8", "gfx942|head_size:64;dtype:fp16;stage_vec:8"),
    ("default gfx942 d64 bf16 sv8", "gfx942|head_size:64;dtype:bf16;stage_vec:8"),
    ("default gfx942 d64 bf16 sv1", "gfx942|head_size:64;dtype:bf16;stage_vec:1"),
    (
        "default gfx942 d64 thd atomic",
        "gfx942|head_size:64;seq_mode:thd;dkv_mode:atomic",
    ),
    (
        "default gfx942 d64 no mask, runtime scale",
        "gfx942|head_size:64;mask_class:none;static_scale:runtime",
    ),
    ("default gfx942 d64 split dQ", "gfx942|head_size:64;dq_mode:split"),
    ("default gfx942 d128 fp16 sv8", "gfx942|head_size:128;dtype:fp16;stage_vec:8"),
    ("default gfx942 d128 bf16 sv8", "gfx942|head_size:128;dtype:bf16;stage_vec:8"),
    ("default gfx942 d128 bf16 sv1", "gfx942|head_size:128;dtype:bf16;stage_vec:1"),
    (
        "default gfx942 d128 thd atomic",
        "gfx942|head_size:128;seq_mode:thd;dkv_mode:atomic",
    ),
    (
        "default gfx942 d128 no mask, runtime scale",
        "gfx942|head_size:128;mask_class:none;static_scale:runtime",
    ),
    ("default gfx942 d128 split dQ", "gfx942|head_size:128;dq_mode:split"),
    ("default gfx950 d32 fp16 sv8", "gfx950|head_size:32;dtype:fp16;stage_vec:8"),
    ("default gfx950 d32 bf16 sv8", "gfx950|head_size:32;dtype:bf16;stage_vec:8"),
    ("default gfx950 d32 bf16 sv1", "gfx950|head_size:32;dtype:bf16;stage_vec:1"),
    (
        "default gfx950 d32 thd atomic",
        "gfx950|head_size:32;seq_mode:thd;dkv_mode:atomic",
    ),
    ("default gfx950 d64 fp16 sv8", "gfx950|head_size:64;dtype:fp16;stage_vec:8"),
    ("default gfx950 d64 bf16 sv8", "gfx950|head_size:64;dtype:bf16;stage_vec:8"),
    ("default gfx950 d64 bf16 sv1", "gfx950|head_size:64;dtype:bf16;stage_vec:1"),
    (
        "default gfx950 d64 thd atomic",
        "gfx950|head_size:64;seq_mode:thd;dkv_mode:atomic",
    ),
    (
        "default gfx950 d64 no mask, runtime scale",
        "gfx950|head_size:64;mask_class:none;static_scale:runtime",
    ),
    ("default gfx950 d64 split dQ", "gfx950|head_size:64;dq_mode:split"),
    ("default gfx950 d128 fp16 sv8", "gfx950|head_size:128;dtype:fp16;stage_vec:8"),
    ("default gfx950 d128 bf16 sv8", "gfx950|head_size:128;dtype:bf16;stage_vec:8"),
    ("default gfx950 d128 bf16 sv1", "gfx950|head_size:128;dtype:bf16;stage_vec:1"),
    (
        "default gfx950 d128 thd atomic",
        "gfx950|head_size:128;seq_mode:thd;dkv_mode:atomic",
    ),
    (
        "default gfx950 d128 no mask, runtime scale",
        "gfx950|head_size:128;mask_class:none;static_scale:runtime",
    ),
    ("default gfx950 d128 split dQ", "gfx950|head_size:128;dq_mode:split"),
    ("default gfx1151 d32", "gfx1151|head_size:32"),
    ("default gfx1151 d64", "gfx1151|head_size:64"),
    ("default gfx1151 d128", "gfx1151|head_size:128"),
    (
        "default gfx1151 d64 bf16 thd atomic sv1",
        "gfx1151|head_size:64;dtype:bf16;seq_mode:thd;dkv_mode:atomic;stage_vec:1",
    ),
    ("default gfx1201 d32", "gfx1201|head_size:32"),
    ("default gfx1201 d64", "gfx1201|head_size:64"),
    ("default gfx1201 d128", "gfx1201|head_size:128"),
    (
        "default gfx1201 d64 bf16 thd atomic sv1",
        "gfx1201|head_size:64;dtype:bf16;seq_mode:thd;dkv_mode:atomic;stage_vec:1",
    ),
    (
        "tuned general/128/bf16/causal_tl/gqa/unequal/fixed/8/t16/full",
        "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:32;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/bf16/causal_tl/mha/square/fixed/8/t16/over",
        "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    ),
    (
        "tuned general/128/bf16/causal_tl/mha/square/fixed/8/t64/over",
        "gfx942|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    ),
    (
        "tuned general/128/bf16/none/mha/square/fixed/8/t16/over",
        "gfx942|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over",
        "gfx942|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:128;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    ),
    (
        "tuned general/64/bf16/causal_br/gqa/square/fixed/8/t64/full",
        "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    ),
    (
        "tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t16/full",
        "gfx942|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    ),
    (
        "tuned general/64/bf16/none/mha/square/thd/8/t16/over",
        "gfx942|head_size:64;dtype:bf16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/64/fp16/none/mha/square/thd/8/t64/over",
        "gfx942|head_size:64;dtype:fp16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;block_k4:32;block_m:16;block_n:64;edge_tiles:False;global_path:vgpr;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:lds_plain;warp_grid_g02:1x2;warp_grid_g13:2x1;warp_grid_g4:1x2;waves:2;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/bf16/causal_br/gqa/square/fixed/8/t16/over",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/bf16/causal_br/gqa/unequal/fixed/8/t16/over",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;step_order:head_outer;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/bf16/causal_tl/mha/square/fixed/8/t4/over",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/bf16/none/gqa/square/fixed/8/t64/over",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/bf16/none/gqa/unequal/fixed/8/t64/over",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/bf16/none/mha/square/fixed/8/t16/over",
        "gfx950|head_size:128;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:192;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/128/fp16/causal_br/gqa/unequal/fixed/8/t64/over",
        "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:128x128;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    ),
    (
        "tuned general/128/fp16/causal_tl/gqa/square/fixed/8/t4/full",
        "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_interleave:True",
    ),
    (
        "tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over",
        "gfx950|head_size:128;dtype:fp16;stage_vec:8;acc_in_lds:False;agpr_alloc:192x192;agpr_route:asm_g13;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:192;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:iterative-ilp;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/64/bf16/causal_br/gqa/unequal/fixed/8/t16/full",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;static_scale:convert;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;lds_order:phase_low;lds_swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_interleave:True",
    ),
    (
        "tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t4/full",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major;lb_mirror:True",
    ),
    (
        "tuned general/64/bf16/none/mha/square/fixed/8/t16/full",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_exp2:fast;dq_mode:split;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major",
    ),
    (
        "tuned general/64/bf16/none/mha/square/fixed/8/t16/over",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/64/bf16/none/mha/square/fixed/8/t4/over",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/64/bf16/none/mha/square/fixed/8/t64/full",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_exp2:fast;dq_mode:split;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:2x2;waves:4;waves_per_eu:2;ws_layout:head_major",
    ),
    (
        "tuned general/64/bf16/none/mha/square/fixed/8/t64/over",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:256;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/64/bf16/none/mha/square/thd/8/t4/over",
        "gfx950|head_size:64;dtype:bf16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:192;bwd_exp2:fast;edge_tiles:False;global_path:dma;kt_source:reg;kv_residency:reg;lds_swizzle:ds=pad32;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major;worklist:True",
    ),
    (
        "tuned general/64/fp16/none/mha/square/fixed/8/t16/full",
        "gfx950|head_size:64;dtype:fp16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;bwd_exp2:fast;dq_mode:split;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/64/fp16/none/mha/square/fixed/8/t4/full",
        "gfx950|head_size:64;dtype:fp16;stage_vec:8;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:32;block_n:128;edge_tiles:False;global_path:dma;kt_source:lds;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:2x4;waves:8;waves_per_eu:1;ws_layout:head_major",
    ),
    (
        "tuned general/64/fp16/none/mha/square/thd/8/t64/over",
        "gfx950|head_size:64;dtype:fp16;stage_vec:8;seq_mode:thd;mask_class:none;acc_in_lds:False;agpr_alloc:None;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;block_k4:32;block_m:16;block_n:192;bwd_exp2:fast;edge_tiles:True;global_path:dma;kt_source:reg;kv_residency:reg;pt_route:relabel;ring_depth:2;s_in_agpr:False;sched:none;scheduler_strategy:None;setprio:False;stats_fma:True;stats_vec:4;transpose_source:tr_read;warp_grid_g02:1x4;warp_grid_g13:4x1;warp_grid_g4:1x4;waves:4;waves_per_eu:2;ws_layout:head_major",
    ),
    (
        "xt_lds gfx942 d64 (predictor swizzle)",
        "gfx942|head_size:64;transpose_source:xt_lds",
    ),
    (
        "xt_lds gfx942 d128 K^T resident image",
        "gfx942|head_size:128;transpose_source:xt_lds;kt_source:lds",
    ),
    (
        "xt_lds gfx942 d32 explicit XT swizzles",
        "gfx942|head_size:32;transpose_source:xt_lds;lds_swizzle:qt=pad16,dot=none,kt_stage=pad8",
    ),
    ("xt_lds gfx950 d64 wide G4 atom", "gfx950|head_size:64;transpose_source:xt_lds"),
    (
        "xt_lds gfx950 d128 split dQ",
        "gfx950|head_size:128;transpose_source:xt_lds;dq_mode:split",
    ),
    (
        "xt_lds gfx950 d32 bf16 kt lds",
        "gfx950|head_size:32;dtype:bf16;transpose_source:xt_lds;kt_source:lds",
    ),
    (
        "xt_lds gfx1201 d64 (no wave64 predictor)",
        "gfx1201|head_size:64;transpose_source:xt_lds",
    ),
    (
        "row xor buffers gfx942 d64",
        "gfx942|head_size:64;lds_swizzle:q=xor,do=xor,ds=xor,k_stage=xor,v_stage=xor,dkv_out=xor",
    ),
    (
        "row pads gfx950 d64",
        "gfx950|head_size:64;lds_swizzle: dkv_out = pad16 , ds=pad8,q=pad32,unused=none ",
    ),
    (
        "rejected: ring-slotted row xor",
        "gfx942|head_size:64;ring_depth:2;lds_swizzle:q=xor",
    ),
    ("ring 3 gfx950 d64", "gfx950|head_size:64;ring_depth:3"),
    (
        "rejected: ring 2 with counted",
        "gfx942|head_size:64;ring_depth:2;loop_sync:counted",
    ),
    ("counted gfx942 d64", "gfx942|head_size:64;loop_sync:counted"),
    (
        "counted gfx950 d128 late issue",
        "gfx950|head_size:128;loop_sync:counted;counted_issue:late",
    ),
    (
        "rejected: counted late with stage_table_atomics",
        "gfx950|head_size:128;loop_sync:counted;counted_issue:late;sched:stage_table_atomics",
    ),
    (
        "counted early with stage_table_atomics",
        "gfx950|head_size:128;loop_sync:counted;sched:stage_table_atomics",
    ),
    (
        "rejected: counted with loads_first",
        "gfx942|head_size:64;loop_sync:counted;qdo_stage:loads_first",
    ),
    ("rejected: counted on gfx1151", "gfx1151|head_size:64;loop_sync:counted"),
    (
        "rejected: counted grouped head_outer",
        "gfx942|head_size:64;loop_sync:counted;step_order:head_outer;edge_tiles:True",
    ),
    (
        "rejected: counted with split dQ",
        "gfx950|head_size:64;loop_sync:counted;dq_mode:split",
    ),
    (
        "rejected: counted with barrier merge",
        "gfx950|head_size:64;loop_sync:counted;barrier_merge:on",
    ),
    (
        "rejected: counted atomics past vmcnt",
        "gfx942|head_size:128;block_m:64;block_n:32;waves:2;warp_grid_g4:1x2;block_k4:32;loop_sync:counted",
    ),
    ("stats_vec 2 gfx942 d64", "gfx942|head_size:64;stats_vec:2"),
    ("stats_vec 4 gfx950 d128 bf16", "gfx950|head_size:128;dtype:bf16;stats_vec:4"),
    ("rejected: stats_vec 4 on gfx1201", "gfx1201|head_size:64;stats_vec:4"),
    (
        "rejected: stats_vec token-major",
        "gfx942|head_size:64;stats_vec:2;ws_layout:token_major",
    ),
    (
        "dkv_cell_vec 4 gfx950 direct epilogue",
        "gfx950|head_size:64;dkv_epilogue:direct;dkv_cell_vec:4",
    ),
    ("rejected: dkv_cell_vec 4 on gfx1201", "gfx1201|head_size:64;dkv_cell_vec:4"),
    (
        "dkv_cell_vec 4 gfx942 atomic slices",
        "gfx942|head_size:32;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct;dkv_cell_vec:4",
    ),
    ("g4 slice_db gfx942 d64", "gfx942|head_size:64;g4_slicing:slice_db"),
    (
        "g4 slice -> step (kK4 = atom K)",
        "gfx942|head_size:64;g4_slicing:slice;block_k4:16",
    ),
    (
        "g4 slice_db -> slice (kK4 = kN0)",
        "gfx942|head_size:32;g4_slicing:slice_db;block_n:64;block_k4:64",
    ),
    (
        "g4 slice_db_g3 gfx950 d128 kt lds",
        "gfx950|head_size:128;g4_slicing:slice_db_g3",
    ),
    (
        "rejected: slice_db_g3 with g1g4g3",
        "gfx942|head_size:64;g4_slicing:slice_db_g3;stage_order:g1g4g3",
    ),
    (
        "rejected: g4 slicing on gfx1151 (resolve)",
        "gfx1151|head_size:64;g4_slicing:slice",
    ),
    (
        "rejected: g4 slicing with fragment issue",
        "gfx942|head_size:64;g4_slicing:slice;dq_issue:fragment",
    ),
    (
        "rejected: g4 slicing with acc_in_lds",
        "gfx942|head_size:64;g4_slicing:slice;acc_in_lds:True",
    ),
    (
        "rejected: g4 slicing with ring",
        "gfx950|head_size:64;g4_slicing:slice;ring_depth:2",
    ),
    (
        "rejected: g4 slicing xt kt lds",
        "gfx942|head_size:64;g4_slicing:slice;transpose_source:xt_lds;kt_source:lds",
    ),
    (
        "asm_g13 gfx942 tail nop 7",
        "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:7",
    ),
    (
        "rejected: asm tail nop below the ISA floor",
        "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:5",
    ),
    (
        "asm_g13 tail nop 15 -> None, gemm -> k_step",
        "gfx942|head_size:128;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:15;asm_cluster:gemm",
    ),
    (
        "asm_g13 s_in_agpr gfx950 tail 8 gemm d32",
        "gfx950|head_size:32;agpr_alloc:128x128;agpr_route:asm_g13;s_in_agpr:True;asm_tail_nop:8;asm_cluster:gemm;asm_load_hoist:True",
    ),
    (
        "rejected: asm tail 6 below the gfx950 floor",
        "gfx950|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;asm_tail_nop:6",
    ),
    (
        "rejected: vgpr_form with s_in_agpr",
        "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:vgpr_form;s_in_agpr:True",
    ),
    (
        "rejected: agpr_fence without reservation",
        "gfx942|head_size:64;agpr_route:agpr_fence",
    ),
    (
        "agpr_fence gfx950 d128",
        "gfx950|head_size:128;agpr_alloc:128x128;agpr_route:agpr_fence",
    ),
    (
        "vgpr_form gfx942 d128",
        "gfx942|head_size:128;agpr_alloc:192x192;agpr_route:vgpr_form",
    ),
    ("agpr backend gfx950 d128", "gfx950|head_size:128;agpr_alloc:backend"),
    ("backend with s_in_agpr", "gfx942|head_size:64;agpr_alloc:backend;s_in_agpr:True"),
    ("rejected: 256 AGPRs on eight waves", "gfx942|head_size:64;agpr_alloc:256x256"),
    (
        "rejected: reservation below the estimate",
        "gfx942|head_size:128;waves:4;block_n:128;warp_grid_g4:1x4;agpr_alloc:128x128",
    ),
    (
        "policy None: four-wave d128",
        "gfx942|head_size:128;waves:4;block_n:128;warp_grid_g4:1x4",
    ),
    ("rejected: agpr_alloc on WMMA", "gfx1151|head_size:64;agpr_alloc:0x0"),
    ("acc_in_lds gfx942 d32", "gfx942|head_size:32;acc_in_lds:True"),
    ("rejected: acc_in_lds d64 LDS plan", "gfx942|head_size:64;acc_in_lds:True"),
    (
        "rejected: acc_in_lds with direct epilogue",
        "gfx950|head_size:64;acc_in_lds:True;dkv_epilogue:direct",
    ),
    (
        "rejected: acc_in_lds with reservation",
        "gfx942|head_size:64;acc_in_lds:True;agpr_alloc:128x128",
    ),
    (
        "acc_in_lds atomic pair epilogue",
        "gfx942|head_size:32;acc_in_lds:True;dkv_mode:atomic;dkv_epilogue:lds_staged_pair",
    ),
    (
        "small tile lds route gfx942 d64 16x16x4",
        "gfx942|head_size:64;waves:4;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x4;warp_grid_g02:1x1;warp_grid_g13:1x4;pt_route:lds",
    ),
    (
        "small tile lds route gfx950 d32 32x32x2",
        "gfx950|head_size:32;waves:2;block_m:32;block_n:32;block_k4:32;warp_grid_g4:1x2;warp_grid_g02:2x1;warp_grid_g13:2x1;pt_route:lds",
    ),
    (
        "(2, W/2) grids gfx942 d64",
        "gfx942|head_size:64;warp_grid_g02:2x4;warp_grid_g13:2x4;pt_route:lds",
    ),
    ("rejected: non-natural grid on relabel", "gfx942|head_size:64;warp_grid_g02:2x4"),
    (
        "rejected: grid not a CTA grid",
        "gfx942|head_size:64;warp_grid_g13:4x4;pt_route:lds",
    ),
    (
        "rejected: grid does not divide",
        "gfx942|head_size:32;warp_grid_g13:1x8;pt_route:lds",
    ),
    (
        "rejected: natural next to small grid does not divide",
        "gfx942|head_size:64;waves:8;block_m:16;block_n:64;warp_grid_g02:1x8;warp_grid_g13:8x1;warp_grid_g4:1x8;pt_route:lds;block_k4:32",
    ),
    (
        "rejected: one wave on 32 x 32",
        "gfx942|head_size:64;waves:1;block_m:32;block_n:32;warp_grid_g4:1x1",
    ),
    (
        "one wave d128 debug tile gfx942 (scheduler)",
        "gfx942|head_size:128;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1",
    ),
    ("rejected: 8 waves on gfx1151", "gfx1151|head_size:64;waves:8"),
    ("rejected: wide atom on gfx942", "gfx942|head_size:64;atom_g02:16x16x32"),
    ("rejected: tr_read on gfx942", "gfx942|head_size:64;transpose_source:tr_read"),
    ("rejected: dma on gfx942", "gfx942|head_size:64;global_path:dma"),
    (
        "rejected: dma with head_pack",
        "gfx950|head_size:64;global_path:dma;head_pack:True",
    ),
    (
        "rejected: dma with lds_only",
        "gfx950|head_size:64;global_path:dma;loop_sync:lds_only",
    ),
    ("dma gfx950 d32 ring 3", "gfx950|head_size:32;global_path:dma;ring_depth:3"),
    (
        "dma gfx950 tr_read q xor",
        "gfx950|head_size:64;global_path:dma;lds_swizzle:q=xor,ds=pad16",
    ),
    (
        "rejected: dma tr_read q pad",
        "gfx950|head_size:64;global_path:dma;lds_swizzle:q=pad8",
    ),
    ("rejected: ring 3 on gfx942", "gfx942|head_size:64;ring_depth:3"),
    ("rejected: split dQ on gfx1151", "gfx1151|head_size:64;dq_mode:split"),
    (
        "rejected: kK4 above kN0",
        "gfx942|head_size:64;block_n:32;block_k4:64;waves:2;warp_grid_g4:1x2",
    ),
    ("rejected: warp_grid_g4 shape", "gfx942|head_size:64;warp_grid_g4:4x2"),
    (
        "rejected: warp_grid_g4 does not divide D",
        "gfx942|head_size:32;warp_grid_g4:1x8",
    ),
    ("rejected: block_n not a multiple of 16 W", "gfx942|head_size:64;block_n:64"),
    (
        "rejected: sched stage_table with replicated grid",
        "gfx942|head_size:64;waves:4;block_m:16;block_n:64;block_k4:16;warp_grid_g4:1x4;warp_grid_g02:1x2;warp_grid_g13:4x1;pt_route:lds;sched:stage_table",
    ),
    ("ptr_noalias workspace", "gfx942|head_size:64;ptr_noalias:workspace"),
    ("ptr_noalias all atomic", "gfx950|head_size:64;ptr_noalias:all;dkv_mode:atomic"),
    ("rejected: ptr_noalias all direct", "gfx950|head_size:64;ptr_noalias:all"),
    ("rejected: kv reg on WMMA", "gfx1201|head_size:64;kv_residency:reg;kt_source:lds"),
    ("rejected: relabel on WMMA", "gfx1201|head_size:64;pt_route:relabel"),
    ("lds_only gfx942 d64", "gfx942|head_size:64;loop_sync:lds_only"),
    ("qdo joint gfx942 d32", "gfx942|head_size:32;qdo_stage:joint"),
    (
        "rejected: qdo joint too many chunks",
        "gfx942|head_size:64;waves:4;block_n:128;warp_grid_g4:2x2;qdo_stage:joint",
    ),
    (
        "rejected: loads_first with ring",
        "gfx942|head_size:64;ring_depth:2;qdo_stage:loads_first",
    ),
    (
        "stats step_top gfx950",
        "gfx950|head_size:128;stats_load:step_top;qdo_stage:loads_first",
    ),
    (
        "barrier merge gfx950 d64",
        "gfx950|head_size:64;barrier_merge:on;ring_issue:late",
    ),
    (
        "barrier merge gfx942 d128 g3g1g4",
        "gfx942|head_size:128;barrier_merge:on;stage_order:g3g1g4",
    ),
    ("rejected: barrier merge on gfx1151", "gfx1151|head_size:64;barrier_merge:on"),
    (
        "rejected: barrier merge with dma",
        "gfx950|head_size:64;barrier_merge:on;global_path:dma",
    ),
    (
        "rejected: barrier merge G4 not last",
        "gfx942|head_size:64;barrier_merge:on;stage_order:g1g4g3",
    ),
    (
        "rejected: barrier merge split dQ",
        "gfx950|head_size:64;barrier_merge:on;dq_mode:split",
    ),
    (
        "rejected: barrier merge acc_in_lds",
        "gfx942|head_size:64;barrier_merge:on;acc_in_lds:True",
    ),
    ("rejected: ring_issue without merge", "gfx950|head_size:64;ring_issue:late"),
    (
        "rejected: grouped head_outer with ring",
        "gfx950|head_size:64;step_order:head_outer;edge_tiles:True;ring_depth:2",
    ),
    (
        "head_outer edge tiles gfx942",
        "gfx942|head_size:64;step_order:head_outer;edge_tiles:True",
    ),
    ("rejected: setprio without stage table", "gfx942|head_size:64;setprio:True"),
    (
        "stage_table setprio gfx942",
        "gfx942|head_size:64;sched:stage_table;setprio:True",
    ),
    (
        "rejected: stage order with stage table",
        "gfx942|head_size:64;sched:stage_table;stage_order:g3g1g4",
    ),
    (
        "rejected: stage_table_atomics split dQ",
        "gfx950|head_size:64;sched:stage_table_atomics;dq_mode:split",
    ),
    (
        "rejected: xt_lds with ring",
        "gfx942|head_size:64;transpose_source:xt_lds;ring_depth:2",
    ),
    (
        "rejected: xt_lds with fences",
        "gfx942|head_size:64;transpose_source:xt_lds;sched:fences_only",
    ),
    ("rejected: waves_per_eu 3 on WMMA", "gfx1151|head_size:64;waves_per_eu:3"),
    ("rejected: waves_per_eu 4 on d128", "gfx942|head_size:128;waves_per_eu:4"),
    ("rejected: waves_per_eu 2 RDNA d128", "gfx1201|head_size:128;waves_per_eu:2"),
    ("waves_per_eu 2 gfx1201 d32", "gfx1201|head_size:32;waves_per_eu:2"),
    (
        "rejected: spill combo gfx942 d128 edge row xor",
        "gfx942|head_size:128;edge_tiles:True;lds_swizzle:dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,v_stage=xor",
    ),
    (
        "rejected: spill combo gfx1151 xt_lds",
        "gfx1151|head_size:64;transpose_source:xt_lds",
    ),
    ("rejected: bf16_cast on fp16", "gfx942|head_size:64;bf16_cast:rne_noguard"),
    (
        "bf16_cast rne_noguard bf16, exp2 fast, stats_fma",
        "gfx950|head_size:64;dtype:bf16;bf16_cast:rne_noguard;bwd_exp2:fast;stats_fma:True",
    ),
    ("ws_bounds buffer", "gfx942|head_size:64;ws_bounds:buffer"),
    ("rejected: narrow_vec on sv8", "gfx942|head_size:64;narrow_vec:2"),
    ("narrow_vec 4 sv1 gfx950", "gfx950|head_size:128;stage_vec:1;narrow_vec:4"),
    (
        "rejected: dkv_store_vec on direct epilogue",
        "gfx942|head_size:64;dkv_epilogue:direct;dkv_store_vec:4",
    ),
    (
        "lds_staged_pair gfx950 d64 store vec 4",
        "gfx950|head_size:64;dkv_epilogue:lds_staged_pair;dkv_store_vec:4",
    ),
    (
        "atomic pair planned as staged",
        "gfx942|head_size:64;dkv_mode:atomic;dkv_epilogue:lds_staged_pair;dkv_atomic_guard:select",
    ),
    (
        "atomic slices direct epilogue",
        "gfx950|head_size:64;dkv_mode:atomic;dkv_sink:slices;dkv_epilogue:direct",
    ),
    (
        "last_cta sink (validated, not built)",
        "gfx942|head_size:64;dkv_mode:atomic;dkv_sink:last_cta",
    ),
    (
        "phase_low kv lds gfx942 d64",
        "gfx942|head_size:64;kv_residency:lds;kt_source:lds;lds_order:phase_low",
    ),
    (
        "phase_low gfx950 d128 (kt lds default)",
        "gfx950|head_size:128;lds_order:phase_low",
    ),
    (
        "kt lds kv reg gfx942 lds_plain (k_res copy)",
        "gfx942|head_size:64;kt_source:lds",
    ),
    ("kv lds tr_read gfx950 d64", "gfx950|head_size:64;kv_residency:lds;kt_source:lds"),
    (
        "head_pack + edge tiles gfx950",
        "gfx950|head_size:64;head_pack:True;edge_tiles:True",
    ),
    (
        "rejected: slab_carry with q_rotate",
        "gfx942|head_size:64;slab_carry:True;q_rotate:True",
    ),
    (
        "rejected: slab_carry with ring",
        "gfx950|head_size:64;slab_carry:True;ring_depth:2",
    ),
    (
        "rejected: slab_carry with counted",
        "gfx942|head_size:64;slab_carry:True;loop_sync:counted",
    ),
    (
        "slab_carry gfx942",
        "gfx942|head_size:64;slab_carry:True;interior_clamp:True;edge_tiles:True",
    ),
    ("rejected: worklist batched", "gfx950|head_size:64;worklist:True"),
    (
        "rejected: worklist with lb_mirror",
        "gfx950|head_size:64;seq_mode:thd;worklist:True;lb_mirror:True",
    ),
    (
        "worklist thd + flags",
        "gfx950|head_size:64;seq_mode:thd;dkv_mode:atomic;worklist:True;xcd_remap:True;q_rotate:True;lb_interleave:True;dkv_slab_hoist:True",
    ),
    (
        "rejected: counted_issue without counted",
        "gfx942|head_size:64;counted_issue:late",
    ),
    ("rejected: asm_tail_nop without asm", "gfx942|head_size:64;asm_tail_nop:8"),
    ("rejected: asm_tail_nop out of range", "gfx942|head_size:64;asm_tail_nop:16"),
    ("rejected: asm_cluster without asm", "gfx942|head_size:64;asm_cluster:gemm"),
    (
        "rejected: asm with iglp",
        "gfx942|head_size:64;agpr_alloc:128x128;agpr_route:asm_g13;sched:iglp1",
    ),
    ("rejected: lds_swizzle entry without value", "gfx942|head_size:64;lds_swizzle:q"),
    ("rejected: lds_swizzle bad value", "gfx942|head_size:64;lds_swizzle:q=pad4"),
    (
        "rejected: lds_swizzle names a buffer twice",
        "gfx942|head_size:64;lds_swizzle:q=xor,q=pad8",
    ),
    (
        "rejected: lds_swizzle name not an identifier",
        "gfx942|head_size:64;lds_swizzle:1q=xor",
    ),
    ("empty lds_swizzle", "gfx942|head_size:64;lds_swizzle:"),
    ("rejected: scheduler strategy", "gfx942|head_size:64;scheduler_strategy:fastest"),
    ("scheduler strategy max-ilp", "gfx950|head_size:64;scheduler_strategy:max-ilp"),
    ("custom name", "gfx942|head_size:64;name:my_bwd2"),
    ("rejected: empty name", "gfx942|head_size:64;name:"),
    ("rejected: agpr_alloc (64, 64)", "gfx942|head_size:64;agpr_alloc:64x64"),
    ("rejected: warp grid 0 x 4", "gfx942|head_size:64;warp_grid_g02:0x4"),
    ("rejected: head_size 48", "gfx942|head_size:48"),
    ("rejected: waves 0 (not None)", "gfx942|head_size:64;waves:0"),
    (
        "rejected: asm_tail_nop 0 without asm (not None)",
        "gfx942|head_size:64;asm_tail_nop:0",
    ),
    ("rejected: atom bad", "gfx942|head_size:64;atom_g4:32x32x8"),
    (
        "rejected: kt reg with kv lds",
        "gfx942|head_size:64;kt_source:reg;kv_residency:lds",
    ),
    (
        "rejected: atom_g13 wide on block_m 16",
        "gfx950|head_size:64;atom_g13:16x16x32;block_m:16",
    ),
    ("wide atom_g13 gfx950 block_m 32", "gfx950|head_size:64;atom_g13:16x16x32"),
    ("rejected: s_in_agpr with VGPR form policy", "gfx942|head_size:32;s_in_agpr:True"),
    (
        "rejected: LDS plan too big",
        "gfx942|head_size:128;kv_residency:lds;kt_source:lds;block_n:256;block_m:64;waves:8;warp_grid_g4:2x4",
    ),
    (
        "rejected: arch-VGPR estimate",
        "gfx942|head_size:128;waves:2;block_m:64;block_n:64;warp_grid_g4:1x2;agpr_alloc:0x0",
    ),
    (
        "rejected: workgroup share",
        "gfx942|head_size:128;waves:8;block_m:64;block_n:256;warp_grid_g4:2x4;agpr_alloc:backend",
    ),
    ("rejected: unknown arch", "gfx90a|head_size:64"),
    ("rejected: WMMA register accumulators", "gfx1151|head_size:64;acc_in_lds:False"),
    ("RDNA d128 one wave kv lds", "gfx1201|head_size:128;waves:1;kv_residency:lds"),
    (
        "gfx1201 d128 one wave defaults + phase_low",
        "gfx1201|head_size:128;lds_order:phase_low",
    ),
    (
        "dq_issue fragment gfx942 d32 runtime",
        "gfx942|head_size:32;dq_issue:fragment;static_scale:dq_before_atomic",
    ),
    (
        "static_scale convert atomic",
        "gfx950|head_size:64;static_scale:convert;dkv_mode:atomic",
    ),
    (
        "token-major thd gfx942",
        "gfx942|head_size:128;seq_mode:thd;ws_layout:token_major;mask_class:none",
    ),
    (
        "rejected: ws_bounds token-major",
        "gfx942|head_size:64;ws_bounds:buffer;ws_layout:token_major",
    ),
    ("rejected: ptr_noalias value", "gfx942|head_size:64;ptr_noalias:both"),
    (
        "rejected: barrier merge ring 3",
        "gfx950|head_size:64;barrier_merge:on;ring_depth:3",
    ),
    (
        "rejected: dma with stage_vec 1",
        "gfx950|head_size:64;global_path:dma;stage_vec:1",
    ),
    (
        "rejected: dma with xt_lds",
        "gfx950|head_size:64;global_path:dma;transpose_source:xt_lds",
    ),
    (
        "agpr_route attr spelled None",
        "gfx942|head_size:64;agpr_route:attr;agpr_alloc:128x128",
    ),
    ("rejected: asm_load_hoist without asm", "gfx942|head_size:64;asm_load_hoist:True"),
    ("rejected: kK4 below the wide atom", "gfx950|head_size:64;block_k4:16"),
    (
        "rejected: warp_grid_g4 does not divide kM0",
        "gfx942|head_size:128;warp_grid_g4:2x4",
    ),
    (
        "rejected: natural G1/G3 grid next to a small G0/G2 grid",
        "gfx942|head_size:64;waves:4;block_m:16;block_n:32;block_k4:16;warp_grid_g4:1x4;warp_grid_g02:1x2;pt_route:lds",
    ),
    (
        "rejected: VGPR form within the prune slack",
        "gfx942|head_size:64;block_m:64;block_n:128;waves:4;warp_grid_g4:1x4;block_k4:16;kv_residency:lds;kt_source:lds;atom_g02:16x16x16;atom_g4:16x16x16;agpr_alloc:0x0",
    ),
    (
        "rejected: AGPR estimate past the slack",
        "gfx942|head_size:64;block_m:16;block_n:256;waves:2;warp_grid_g4:1x2;block_k4:16",
    ),
    (
        "rejected: workgroup share of eight waves",
        "gfx942|head_size:128;block_m:16;block_n:256;waves:8;warp_grid_g4:1x8;block_k4:16",
    ),
    (
        "rejected: RDNA register estimate",
        "gfx1201|head_size:64;block_m:64;block_n:32;waves:2;warp_grid_g4:1x2;block_k4:16",
    ),
)


def _value(name: str, text: str):
    if text == "None":
        return None
    if name in _BOOL or (name == "ptr_noalias" and text == "False"):
        if text not in ("True", "False"):
            raise SystemExit(f"bad bool {text!r} for {name}")
        return text == "True"
    if name in _INT or name in _OPT_INT:
        return int(text)
    if name in _PAIR and "x" in text and text.replace("x", "").isdigit():
        r, c = text.split("x")
        return (int(r), int(c))
    return text


def _parse(config: str):
    arch, _, rest = config.partition("|")
    fields = {}
    for item in rest.split(";"):
        if item:
            name, _, text = item.partition(":")
            fields[name] = _value(name, text)
    return arch, fields


def _exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _plan_lines(plan) -> list:
    out = [
        (
            f"plan: total={plan.total_bytes} resident={plan.resident_bytes} "
            f"capacity={plan.capacity_bytes} fits={plan.fits} "
            f"phase_offset={plan.phase_offset} unaliased={plan.unaliased_bytes}"
        ),
        "plan phases: " + " ".join(f"{p}={n}" for p, n in plan.phase_bytes),
    ]
    for b in plan.buffers:
        out.append(
            f"plan buffer: {b.name}[{b.slot}] offset={b.offset} nbytes={b.nbytes} "
            f"phases={','.join(b.phases)} layout={b.layout} resident={b.resident}"
        )
    for name, lay in plan.layouts:
        out.append(f"plan layout: {name} = {lay!r}")
    for name in dict.fromkeys(b.name for b in plan.buffers):
        try:
            out.append(f"plan ring_stride: {name} = {plan.ring_stride(name)}")
        except (KeyError, ValueError) as e:
            out.append(f"plan ring_stride: {name} = {_exc(e)}")
    return out


def dump(arch: str, fields: dict) -> list:
    lines = [f"arch: {arch}"]
    try:
        spec = AttnBwdSpec(**fields)
    except ValueError as e:
        lines.append("spec: " + _exc(e))
        return lines
    lines.append("spec: ok")
    for stage in ("main", "dq"):
        lines.append(f"kernel_name {stage}: {spec.kernel_name(stage)}")
    try:
        lines.append(f"kernel_name bwd: {spec.kernel_name('bwd')}")
    except ValueError as e:
        lines.append("kernel_name bwd: " + _exc(e))
    try:
        lds_request(spec, arch)
        lines.append("unresolved lds_request: ok")
    except (KeyError, ValueError) as e:
        lines.append("unresolved lds_request: " + _exc(e))
    try:
        geo = attn_bwd_default_geometry(
            arch, spec.head_size, spec.dtype, spec.stage_vec, spec.seq_mode
        )
        lines.append(
            "default_geometry: "
            + ", ".join(f"{k}={v!r}" for k, v in sorted(geo.items()))
        )
    except ValueError as e:
        lines.append("default_geometry: " + _exc(e))
    try:
        r = spec.resolved(arch)
    except ValueError as e:
        lines.append("resolved: " + _exc(e))
        return lines
    for f in dataclasses.fields(r):
        lines.append(f"resolved {f.name}={getattr(r, f.name)!r}")
    for stage in ("main", "dq"):
        lines.append(f"resolved kernel_name {stage}: {r.kernel_name(stage)}")
    try:
        validate_attn_bwd_spec(spec, arch, main_kernel_refusals=False)
        lines.append("validate: ok")
    except ValueError as e:
        lines.append("validate: " + _exc(e))
    try:
        req = lds_request(r, arch)
        for f in dataclasses.fields(req):
            lines.append(f"lds_request {f.name}={getattr(req, f.name)!r}")
        lines += _plan_lines(lds_plan(r, arch))
    except (KeyError, ValueError) as e:
        lines.append("lds: " + _exc(e))
    try:
        est = register_estimates(r, arch)
        lines.append(
            f"register_estimates: arch_vgpr={est['arch_vgpr']} agpr={est['agpr']}"
        )
    except ValueError as e:
        lines.append("register_estimates: " + _exc(e))
    try:
        bud = agpr_copy_budget(r, arch)
        lines.append(
            f"agpr_copy_budget: reads={bud['reads']} writes={bud['writes']} "
            f"moves={bud['moves']}"
        )
    except ValueError as e:
        lines.append("agpr_copy_budget: " + _exc(e))
    try:
        check_counted_sync(r, arch)
        lines.append("check_counted_sync: ok")
    except ValueError as e:
        lines.append("check_counted_sync: " + _exc(e))
    try:
        occ = attn_bwd_occupancy(r, arch)
        lines.append(
            f"occupancy: by_registers={occ.by_registers} by_lds={occ.by_lds} "
            f"requested={occ.requested!r} waves={occ.waves} "
            f"hint_reachable={occ.hint_reachable}"
        )
    except ValueError as e:
        lines.append("occupancy: " + _exc(e))
    try:
        lines.append(f"policy_agpr_alloc: {_policy_agpr_alloc(r, arch)!r}")
    except ValueError as e:
        lines.append("policy_agpr_alloc: " + _exc(e))
    lines.append(
        "static_unit_mults: "
        + ",".join(sorted(static_unit_mults(r.static_scale, r.dkv_mode)))
    )
    for label, stage, cv in (
        ("prep", "prep", 8),
        ("convert", "convert", 16),
        ("prep cv16", "prep", 16),
    ):
        try:
            aux = r.aux_spec(stage, convert_vec=cv)
            lines.append(f"aux {label}: {aux.kernel_name(stage)}")
        except ValueError as e:
            lines.append(f"aux {label}: {type(e).__name__}")
    return lines


def _print_c_configs() -> None:
    for comment, config in _CONFIGS:
        c = comment.replace("*/", "* /")
        print(f'    /* {c} */\n    "{config}",')


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--print-c-configs":
        _print_c_configs()
        return 0
    if len(sys.argv) < 2:
        sys.stderr.write("usage: attention_bwd_spec_emit.py <config_index> [ll]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode != "ll":
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    arch, fields = _parse(_CONFIGS[idx][1])
    sys.stdout.write("".join(line + "\n" for line in dump(arch, fields)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
