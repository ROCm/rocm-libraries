#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_refusals_emit.py -- Python reference for the C
# port of the attention backward resource refusals
# (kernels/common/_attention_bwd_refusals.py: spec_digest and RESOURCE_REFUSED;
# kernels/common/attention_bwd_dq.py: DQ_RESOURCE_REFUSED). These emit no IR,
# so this emitter prints a text dump of one spec; the C twin
# attention_bwd_refusals_emit.c prints the same dump from
# rocke/instance_attention_bwd.h and rocke/instance_attention_bwd_dq.h, and
# run_diff compares the two byte for byte, like a .ll.
#
# A config is "main <config>" (an AttnBwdSpec in the spellings of
# attention_bwd_spec_emit.py) or "dq <config>" (an AttnBwdDqSpec, or the dQ
# spec of a main spec, in the spellings of attention_bwd_dq_emit.py). The
# dump prints the digest of the spec and of its resolved spec, the table's
# record of the resolved spec, and the validator's verdict (for a main spec
# with and without main_kernel_refusals). An exception prints as
# "<type>: <str(exc)>".
#
# Configs: shipped defaults, a knob spec and early rejections, then resolved
# specs of recorded rows of both tables (every target and failure kind, and
# every spec field among the recorded rows). The tables themselves are
# generated: `attention_bwd_refusals_emit.py --print-c-table` writes
# platform/cpp/instances/common/_attention_bwd_refusals_table.inc, and
# library/tests/test_attention_bwd_refusals_cpp_table.py fails when the
# committed file differs. Keep _CONFIGS and the C table in sync:
# `attention_bwd_refusals_emit.py --print-c-configs` prints the C table.
#
# argv[2] selects the output mode; only "ll" (the dump) is supported.
import sys

from attention_bwd_dq_emit import _parse as _parse_dq
from attention_bwd_spec_emit import _exc
from attention_bwd_spec_emit import _parse as _parse_main
from kernels.common._attention_bwd_refusals import (
    RESOURCE_REFUSED,
    resource_refusal,
    spec_digest,
)
from kernels.common.attention_bwd import AttnBwdSpec, validate_attn_bwd_spec
from kernels.common.attention_bwd_dq import (
    DQ_RESOURCE_REFUSED,
    AttnBwdDqSpec,
    validate_attn_bwd_dq_spec,
)

# (comment, config)
_CONFIGS = (
    ("default gfx942 d64", "main gfx942|head_size:64"),
    ("default gfx950 d128 bf16", "main gfx950|head_size:128;dtype:bf16"),
    ("default gfx1151 d64", "main gfx1151|head_size:64"),
    ("default gfx1201 d32", "main gfx1201|head_size:32"),
    (
        "knobs gfx950 d64: grids, agpr pair, swizzle, ptr_noalias",
        "main gfx950|head_size:64;waves:4;block_m:32;block_n:128;warp_grid_g4:1x4;agpr_alloc:128x128;lds_swizzle:ds=pad32;ptr_noalias:all;lb_mirror:True;dkv_mode:atomic",
    ),
    ("rejected spec: head_size 48", "main gfx942|head_size:48"),
    (
        "rejected by validate before the table: barrier_merge on gfx1201",
        "main gfx1201|head_size:64;barrier_merge:on",
    ),
    (
        "refused gfx942 sgpr_spill: ir row",
        "main gfx942|head_size:64;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_g13:1x1;step_order:head_outer",
    ),
    (
        "refused gfx942 scratch+vgpr_spill+sgpr_spill: ir row",
        "main gfx942|head_size:32;dkv_mode:atomic;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;loop_sync:lds_only;step_order:head_outer",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: set row",
        "main gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4",
    ),
    (
        "refused gfx942 compiler crash (comgr segfault): set row",
        "main gfx942|head_size:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;scheduler_strategy:iterative-ilp;warp_grid_g02:1x8;warp_grid_g13:8x1",
    ),
    (
        "refused gfx942 agpr_copy_traffic: set row",
        "main gfx942|head_size:32;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:backend;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    ),
    (
        "refused gfx950 sgpr_spill: ir row",
        "main gfx950|head_size:32;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;step_order:head_outer",
    ),
    (
        "refused gfx950 scratch+vgpr_spill+sgpr_spill: ir row",
        "main gfx950|head_size:32;dkv_mode:atomic;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;loop_sync:lds_only;step_order:head_outer",
    ),
    (
        "refused gfx950 agpr_copy_traffic: set row",
        "main gfx950|head_size:32;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:backend;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    ),
    (
        "refused gfx1151 sgpr_spill: ir row",
        "main gfx1151|head_size:32;dtype:bf16;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_g13:1x1;loop_sync:lds_only;step_order:head_outer",
    ),
    (
        "refused gfx1151 scratch+vgpr_spill+sgpr_spill: ir row",
        "main gfx1151|head_size:64;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_g13:1x1;loop_sync:lds_only;step_order:head_outer",
    ),
    (
        "refused gfx1151 scratch+vgpr_spill: ir row",
        "main gfx1151|head_size:128;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;scheduler_strategy:iterative-minreg;warp_grid_g02:1x1;warp_grid_g13:1x1;loop_sync:lds_only;step_order:head_outer",
    ),
    (
        "refused gfx1201 scratch+vgpr_spill: ir row",
        "main gfx1201|head_size:64;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_g13:1x1;loop_sync:lds_only;step_order:head_outer",
    ),
    (
        "refused gfx1201 sgpr_margin: ir row",
        "main gfx1201|head_size:32;dkv_mode:atomic;waves:2;block_m:16;block_n:32;block_k4:32;warp_grid_g4:1x2;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;edge_tiles:True;head_pack:True;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x2;warp_grid_g13:2x1;loop_sync:lds_only;step_order:head_outer",
    ),
    (
        "refused gfx950 scratch+vgpr_spill: mcell row",
        "main gfx950|head_size:128;dtype:bf16;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:stage_table_atomics;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;scheduler_strategy:iterative-minreg;warp_grid_g02:1x1;warp_grid_g13:1x1;lb_mirror:True",
    ),
    (
        "refused gfx950 scratch+vgpr_spill+agpr_copy_traffic: compose row",
        "main gfx950|head_size:64;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:True;agpr_route:asm_g13;warp_grid_g02:1x8;warp_grid_g13:8x1;loop_sync:counted",
    ),
    (
        "refused gfx1151 sgpr_margin: mirror row",
        "main gfx1151|head_size:32;seq_mode:thd;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;edge_tiles:False;head_pack:False;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_g13:1x1;dkv_slab_hoist:True;lb_mirror:True",
    ),
    (
        "refused gfx1201 sgpr_spill: mirror row",
        "main gfx1201|head_size:32;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;edge_tiles:True;head_pack:False;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_g13:1x1;step_order:head_outer;lb_mirror:True",
    ),
    (
        "refused gfx942 scratch+vgpr_spill+agpr_copy_traffic: spec row",
        "main gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:True;agpr_route:asm_g13;asm_cluster:gemm;warp_grid_g02:2x4;warp_grid_g13:8x1",
    ),
    (
        "refused gfx942 sgpr_spill+agpr_copy_traffic: spec row",
        "main gfx942|head_size:128;dtype:bf16;seq_mode:thd;waves:4;block_m:64;block_n:64;block_k4:32;warp_grid_g4:2x2;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    ),
    (
        "refused gfx950 compiler crash (comgr segfault): spec row",
        "main gfx950|head_size:128;dtype:bf16;seq_mode:thd;dkv_mode:atomic;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;scheduler_strategy:iterative-ilp;warp_grid_g02:1x8;warp_grid_g13:8x1;barrier_merge:on",
    ),
    (
        "refused gfx950 sgpr_spill+agpr_copy_traffic: spec row",
        "main gfx950|head_size:128;dtype:bf16;waves:4;block_m:64;block_n:64;block_k4:32;warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1",
    ),
    (
        "refused gfx950 vgpr_spill+agpr_copy_traffic: spec row",
        "main gfx950|head_size:64;mask_class:none;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:True;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:False;agpr_route:asm_g13;warp_grid_g02:1x8;warp_grid_g13:8x1",
    ),
    (
        "refused gfx942 vgpr_spill+agpr_copy_traffic: spec row",
        "main gfx942|head_size:64;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:192x192;s_in_agpr:True;agpr_route:asm_g13;asm_cluster:gemm;asm_load_hoist:True;warp_grid_g02:1x8;warp_grid_g13:2x4",
    ),
    (
        "refused gfx942 sgpr_spill: counted row",
        "main gfx942|head_size:32;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_g13:1x1;loop_sync:counted;counted_issue:late;lb_mirror:True",
    ),
    (
        "refused gfx942 sgpr_spill: mirror row",
        "main gfx942|head_size:128;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;xcd_remap:True;agpr_alloc:0x0;s_in_agpr:False;scheduler_strategy:iterative-minreg;warp_grid_g02:1x1;warp_grid_g13:1x1;lb_mirror:True",
    ),
    (
        "refused gfx942 sgpr_spill: mcell row",
        "main gfx942|head_size:64;dq_mode:split;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;lb_mirror:True",
    ),
    (
        "refused gfx942 sgpr_spill: mcell row",
        "main gfx942|head_size:64;dtype:bf16;dkv_mode:atomic;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;g4_slicing:slice_db;lb_mirror:True",
    ),
    (
        "refused gfx942 sgpr_spill: ccell row",
        "main gfx942|head_size:64;mask_class:none;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;loop_sync:counted;q_rotate:True",
    ),
    (
        "refused gfx942 sgpr_spill: spec row",
        "main gfx942|head_size:128;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;dkv_store_vec:1",
    ),
    (
        "refused gfx942 sgpr_spill: spec row",
        "main gfx942|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;stats_vec:4",
    ),
    (
        "refused gfx942 sgpr_spill: spec row",
        "main gfx942|head_size:64;dtype:bf16;mask_class:none;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;bf16_cast:rne_noguard;bwd_exp2:fast;stats_fma:True;stats_vec:4",
    ),
    (
        "refused gfx942 sgpr_spill: access row",
        "main gfx942|head_size:64;dkv_mode:atomic;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;dkv_sink:slices;ws_bounds:buffer",
    ),
    (
        "refused gfx942 sgpr_spill: access row",
        "main gfx942|head_size:64;dtype:bf16;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;slab_carry:True",
    ),
    (
        "refused gfx942 scratch+vgpr_spill+sgpr_spill: spec row",
        "main gfx942|head_size:128;dtype:bf16;seq_mode:thd;dkv_mode:atomic;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:2x4;worklist:True",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;seq_mode:thd;waves:8;block_m:16;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,v_stage=xor;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;seq_mode:thd;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:True;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;seq_mode:thd;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;ws_layout:token_major;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:2x4;dkv_epilogue:lds_staged_pair;dkv_store_vec:2",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;static_scale:runtime;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select;lb_interleave:True",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:2x4;stage_order:g4g1g3",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select;dq_issue:fragment_regs",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select;ptr_noalias:workspace",
    ),
    (
        "refused gfx942 scratch+vgpr_spill: spec row",
        "main gfx942|head_size:128;waves:8;block_m:32;block_n:128;block_k4:32;warp_grid_g4:1x8;atom_g02:16x16x16;atom_g13:16x16x16;atom_g4:16x16x16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:2x4;warp_grid_g13:2x4;dkv_atomic_guard:select;lds_order:phase_low",
    ),
    (
        "refused gfx950 sgpr_spill: ir row",
        "main gfx950|head_size:64;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;qdo_stage:joint",
    ),
    (
        "refused gfx950 sgpr_spill: spec row",
        "main gfx950|head_size:64;dkv_mode:atomic;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:True;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x8;warp_grid_g13:8x1;interior_clamp:True",
    ),
    (
        "refused gfx950 sgpr_spill: ringsched row",
        "main gfx950|head_size:32;waves:4;block_m:32;block_n:128;block_k4:64;warp_grid_g4:2x2;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:vgpr;sched:iglp1;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:0x0;s_in_agpr:False;warp_grid_g02:1x4;warp_grid_g13:4x1;barrier_merge:on;ring_issue:late",
    ),
    (
        "refused gfx950 agpr_copy_traffic: spec row",
        "main gfx950|head_size:64;waves:8;block_m:32;block_n:256;block_k4:32;warp_grid_g4:2x4;atom_g02:16x16x32;atom_g13:16x16x16;atom_g4:16x16x32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;sched:none;setprio:False;waves_per_eu:1;edge_tiles:False;head_pack:False;acc_in_lds:False;agpr_alloc:128x128;s_in_agpr:True;agpr_route:asm_g13;asm_tail_nop:12;asm_cluster:gemm;warp_grid_g02:1x8;warp_grid_g13:8x1",
    ),
    (
        "refused gfx1151 scratch+vgpr_spill: ir row",
        "main gfx1151|head_size:64;seq_mode:thd;dkv_mode:atomic;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:none;setprio:False;edge_tiles:False;head_pack:False;acc_in_lds:True;s_in_agpr:False;warp_grid_g02:1x1;warp_grid_g13:1x1;stats_load:step_top",
    ),
    (
        "refused gfx1151 scratch+vgpr_spill: sched row",
        "main gfx1151|head_size:128;stage_vec:1;waves:1;block_m:16;block_n:16;block_k4:16;warp_grid_g4:1x1;atom_g02:wmma16x16x16;atom_g13:wmma16x16x16;atom_g4:wmma16x16x16;pt_route:lds;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;ring_depth:1;global_path:vgpr;sched:iglp1;setprio:False;edge_tiles:False;head_pack:False;acc_in_lds:True;s_in_agpr:False;scheduler_strategy:iterative-minreg;warp_grid_g02:1x1;warp_grid_g13:1x1",
    ),
    ("default dq gfx942 d64", "dq gfx942|head_size:64"),
    ("default dq gfx950 d128 thd", "dq gfx950|head_size:128;seq_mode:thd"),
    (
        "dq of a main spec: gfx942 d128 bf16 split",
        "dq main=gfx942|head_size:128;dtype:bf16;dq_mode:split#kv_prefetch:True",
    ),
    ("rejected dq: bad name", "dq gfx942|head_size:64;name:bad-name"),
    (
        "refused dq gfx942 compiler crash: ('gfx942', 'default', 'bf16', 64, 'batched', 'band', 1, 'edge')",
        "dq gfx942|head_size:64;dtype:bf16;stage_vec:1;edge_tiles:True;scheduler_strategy:iterative-ilp",
    ),
    (
        "refused dq gfx950 compiler crash: ('gfx950', 'default', 'bf16', 64, 'batched', 'band', 1, 'prefetch')",
        "dq gfx950|head_size:64;dtype:bf16;stage_vec:1;kv_prefetch:True;scheduler_strategy:iterative-ilp",
    ),
    (
        "refused dq gfx950 compiler crash: ('gfx950', 's3', 'bf16', 128, 'batched', 'band', 1, 'edge')",
        "dq gfx950|head_size:128;dtype:bf16;stage_vec:1;waves:8;block_m:128;block_n:32;edge_tiles:True;kv_pad:8;scheduler_strategy:iterative-ilp",
    ),
    (
        "refused dq gfx942 scratch+vgpr_spill",
        "dq gfx942|head_size:64;dtype:bf16;waves:2;block_m:32;block_n:128;atom_g02:16x16x16;atom_g4:16x16x16;transpose_source:lds_plain;waves_per_eu:2;agpr_alloc:0x0;scheduler_strategy:iterative-maxocc",
    ),
    (
        "refused dq gfx942 scratch+vgpr_spill+sgpr_spill",
        "dq gfx942|head_size:128;dtype:bf16;mask_class:none;waves:2;block_m:32;block_n:128;atom_g02:16x16x16;atom_g4:16x16x16;transpose_source:lds_plain;waves_per_eu:2;agpr_alloc:0x0;scheduler_strategy:iterative-maxocc",
    ),
    (
        "refused dq gfx950 scratch+vgpr_spill",
        "dq gfx950|head_size:128;dtype:bf16;seq_mode:thd;waves:2;block_m:32;block_n:128;atom_g02:16x16x32;atom_g4:16x16x16;transpose_source:lds_plain;waves_per_eu:2;agpr_alloc:0x0;scheduler_strategy:iterative-maxocc",
    ),
    (
        "refused dq gfx950 compiler crash (comgr segfault)",
        "dq gfx950|head_size:128;waves:8;block_m:128;block_n:32;atom_g02:16x16x32;atom_g4:16x16x16;transpose_source:lds_plain;kv_pad:8;agpr_alloc:0x0;scheduler_strategy:iterative-ilp",
    ),
)


def _c_string(text: str) -> str:
    if not text.isascii() or not text.isprintable():
        raise SystemExit(f"not a printable ASCII record: {text!r}")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def c_table() -> str:
    """platform/cpp/instances/common/_attention_bwd_refusals_table.inc."""
    out = [
        "// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier: MIT",
        "/* Generated by library/tests/parity/attention_bwd_refusals_emit.py",
        " * --print-c-table from RESOURCE_REFUSED (_attention_bwd_refusals.py) and",
        " * DQ_RESOURCE_REFUSED (attention_bwd_dq.py); do not edit. Rows are sorted",
        " * by (arch, digest) for the binary search of _attention_bwd_refusals.cpp. */",
        "// clang-format off",
    ]
    for name, table in (
        ("RESOURCE_REFUSED", RESOURCE_REFUSED),
        ("DQ_RESOURCE_REFUSED", DQ_RESOURCE_REFUSED),
    ):
        out.append(f"const RefusalRow {name}[] = {{")
        for (arch, digest), reason in sorted(table.items()):
            out.append(f'    {{"{arch}", "{digest}", {_c_string(reason)}}},')
        out.append("};")
    out.append("// clang-format on")
    return "\n".join(out) + "\n"


def _dq_spec(config: str) -> tuple:
    """(arch, AttnBwdDqSpec) of a dQ config, as attention_bwd_dq_emit builds it."""
    if config.startswith("main="):
        main, _, knobs = config[len("main=") :].partition("#")
        arch, mfields = _parse_main(main)
        _a, kfields = _parse_dq(f"{arch}|{knobs}")
        return arch, AttnBwdSpec(**mfields).dq_spec(**kfields)
    arch, fields = _parse_dq(config)
    return arch, AttnBwdDqSpec(**fields)


def dump(config: str) -> list:
    kind, _, rest = config.partition(" ")
    lines = [f"kind: {kind}"]
    try:
        if kind == "main":
            arch, fields = _parse_main(rest)
            spec = AttnBwdSpec(**fields)
        else:
            arch, spec = _dq_spec(rest)
    except ValueError as e:
        lines.append("spec: " + _exc(e))
        return lines
    lines.append(f"arch: {arch}")
    lines.append(f"digest: {spec_digest(spec, arch)}")
    try:
        r = spec.resolved(arch)
    except (KeyError, ValueError) as e:
        lines.append("resolved: " + _exc(e))
        return lines
    lines.append(f"resolved digest: {spec_digest(r, arch)}")
    if kind == "main":
        lines.append(f"refusal: {resource_refusal(r, arch)}")
        for flag in (True, False):
            try:
                validate_attn_bwd_spec(spec, arch, main_kernel_refusals=flag)
                verdict = "ok"
            except (KeyError, ValueError) as e:
                verdict = _exc(e)
            lines.append(f"validate main_kernel_refusals={flag}: {verdict}")
    else:
        lines.append(
            f"refusal: {DQ_RESOURCE_REFUSED.get((arch, spec_digest(r, arch)))}"
        )
        try:
            validate_attn_bwd_dq_spec(spec, arch)
            verdict = "ok"
        except (KeyError, ValueError) as e:
            verdict = _exc(e)
        lines.append(f"validate: {verdict}")
    return lines


def _print_c_configs() -> None:
    for comment, config in _CONFIGS:
        c = comment.replace("*/", "* /")
        print(f'    /* {c} */\n    "{config}",')


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--print-c-configs":
        _print_c_configs()
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--print-c-table":
        sys.stdout.write(c_table())
        return 0
    if len(sys.argv) < 2:
        sys.stderr.write("usage: attention_bwd_refusals_emit.py <config_index> [ll]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode != "ll":
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    sys.stdout.write("".join(line + "\n" for line in dump(_CONFIGS[idx][1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
