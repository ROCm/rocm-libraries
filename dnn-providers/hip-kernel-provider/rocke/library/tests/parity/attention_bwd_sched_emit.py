#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_sched_emit.py -- Python reference for the C port
# of the attention backward stage tables (kernels/common/_attention_bwd_sched.py:
# StageTable, DqStageTable, interleave_groups, the atom K helpers, and the
# DmaBatch / dma_wait_counts facts of _attention_bwd_ring they read). The
# counts are numbers, so this emitter prints a canonical text dump of every
# field and method result of one config; the C twin attention_bwd_sched_emit.c
# prints the same dump from helper_rocke.instances.common._attention_bwd_sched.h
# and run_diff compares the two byte for byte (sha256), like a .ll. The hint
# emission is covered by the IR pair attention_bwd_sched_ir_emit.
#
# A config is "<kind>|<arch>|<key>:<value>;..." (a value is an int, True /
# False, None, "RxC" for a grid, "a/b/c" for the atoms table (each an int or
# None), or a string):
#   geom    StageTable.from_geometry over a BwdTileGeometry-like namespace (the
#           listed attributes only; kw_* keys are the keyword arguments), then
#           dataclasses.replace variants over every sched value, setprio and
#           the "ck" model;
#   table   StageTable(arch=..., **fields);
#   dq      DqStageTable(arch=..., **fields);
#   dqgeom  DqStageTable.from_geometry over a BwdDqGeometry-like namespace, then
#           replace variants over every dq sched value;
#   groups  interleave_groups(mfma, mem, lead=, spans=) ("m=c,m=c" lists);
#   counts  a StageCounts(stage, mfma, ...) and its memory / census / groups;
#   atoms   native_atom_ks and instruction_k over several atom K values;
#   dma     DmaBatch(block_m, head_size, waves, wave_size) and
#           dma_wait_counts(ring_depth, n_qd, n_a).
# An exception prints as "<type>: <str(exc)>", so both engines must refuse at
# the same rule with the same message.
#
# This family's configs are the geometries (BwdTileGeometry.from_spec fields of
# the resolved spec) of every shipped default on gfx942 / gfx950 (head size,
# dtype, stage_vec, dq_mode), every GENERAL_TUNED_TABLE and DENSE_TUNED entry
# of dispatch/attention_bwd on gfx942 / gfx950 (the knobs that are AttnBwdSpec
# fields), and every other spec config of attention_bwd_spec_emit /
# attention_bwd_spec_extra_emit on those arches that resolves, deduplicated.
# They are literal (generated once from the resolver) so a tuning change does
# not move them. attention_bwd_sched_params_emit holds the direct, edge and
# rejected configs. Keep _CONFIGS and the C table in sync:
# `attention_bwd_sched_emit.py --print-c-configs` prints the C table.
#
# argv[2] selects the output mode; only "ll" (the dump) is supported.
import dataclasses
import sys
from types import SimpleNamespace

from kernels.common._attention_bwd_ring import DmaBatch, dma_wait_counts
from kernels.common._attention_bwd_sched import (
    DQ_SCHEDS,
    DQ_STAGES,
    GEMMS,
    SCHEDS,
    STAGES,
    DqStageTable,
    StageCounts,
    StageTable,
    instruction_k,
    interleave_groups,
    native_atom_ks,
)

_GRIDS = {"grid_g4", "grid_g0", "grid_g1", "grid_g02", "grid_g13"}

# (comment, config)
_CONFIGS = (
    (
        "default gfx942 d32 fp16 sv8 batched atomic",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d32 fp16 sv8 batched split",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d32 fp16 sv1 batched atomic",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d32 fp16 sv1 batched split",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d32 bf16 sv8 batched atomic",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d32 bf16 sv8 batched split",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d32 bf16 sv1 batched atomic",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d32 bf16 sv1 batched split",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d64 fp16 sv8 batched atomic",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d64 fp16 sv8 batched split",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d64 fp16 sv1 batched atomic",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d64 fp16 sv1 batched split",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d64 bf16 sv8 batched atomic",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d64 bf16 sv8 batched split",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d64 bf16 sv1 batched atomic",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d64 bf16 sv1 batched split",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d128 fp16 sv8 batched atomic",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d128 fp16 sv8 batched split",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d128 fp16 sv1 batched atomic",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d128 fp16 sv1 batched split",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx942 d128 bf16 sv8 batched atomic",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d128 bf16 sv8 batched split",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d128 bf16 sv1 batched atomic",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx942 d128 bf16 sv1 batched split",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d32 fp16 sv8 batched atomic",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d32 fp16 sv8 batched split",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d32 fp16 sv1 batched atomic",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d32 fp16 sv1 batched split",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d32 bf16 sv8 batched atomic",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d32 bf16 sv8 batched split",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d32 bf16 sv1 batched atomic",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d32 bf16 sv1 batched split",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d64 fp16 sv8 batched atomic",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d64 fp16 sv8 batched split",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d64 fp16 sv1 batched atomic",
        "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d64 fp16 sv1 batched split",
        "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d64 bf16 sv8 batched atomic",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d64 bf16 sv8 batched split",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d64 bf16 sv1 batched atomic",
        "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d64 bf16 sv1 batched split",
        "geom|gfx950|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d128 fp16 sv8 batched atomic",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d128 fp16 sv8 batched split",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d128 fp16 sv1 batched atomic",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d128 fp16 sv1 batched split",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "default gfx950 d128 bf16 sv8 batched atomic",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d128 bf16 sv8 batched split",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d128 bf16 sv1 batched atomic",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "default gfx950 d128 bf16 sv1 batched split",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:64;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:1;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/128/bf16/causal_tl/gqa/unequal/fixed/8/t16/full (gfx942)",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:32;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/128/bf16/causal_tl/mha/square/fixed/8/t16/over (gfx942)",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over (gfx942)",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "tuned general/64/bf16/causal_br/gqa/square/fixed/8/t64/full (gfx942)",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/64/fp16/none/mha/square/thd/8/t64/over (gfx942)",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "tuned general/128/bf16/causal_br/gqa/square/fixed/8/t16/over (gfx950)",
        "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/128/bf16/causal_tl/mha/square/fixed/8/t4/over (gfx950)",
        "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/128/bf16/none/mha/square/fixed/8/t16/over (gfx950)",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:192;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/128/fp16/causal_br/gqa/unequal/fixed/8/t64/over (gfx950)",
        "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "tuned general/128/fp16/causal_tl/gqa/square/fixed/8/t4/full (gfx950)",
        "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "tuned general/128/fp16/causal_tl/mha/square/fixed/8/t16/over (gfx950)",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:192;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "tuned general/64/bf16/causal_br/gqa/unequal/fixed/8/t16/full (gfx950)",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:ds=pad32;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/64/bf16/causal_tl/gqa/square/fixed/8/t4/full (gfx950)",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/64/bf16/none/mha/square/fixed/8/t16/full (gfx950)",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/64/bf16/none/mha/square/fixed/8/t16/over (gfx950)",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/64/bf16/none/mha/square/fixed/8/t64/full (gfx950)",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/64/bf16/none/mha/square/thd/8/t4/over (gfx950)",
        "geom|gfx950|head_size:64;k_m0:16;k_n0:192;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:ds=pad32;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned general/64/fp16/none/mha/square/fixed/8/t16/full (gfx950)",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "tuned general/64/fp16/none/mha/square/fixed/8/t4/full (gfx950)",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "tuned general/64/fp16/none/mha/square/thd/8/t64/over (gfx950)",
        "geom|gfx950|head_size:64;k_m0:16;k_n0:192;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "tuned dense 1024/4096/32/8/128/bf16/bshd/causal_br (gfx950)",
        "geom|gfx950|head_size:128;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "tuned dense 2048/2048/12/12/64/bf16/bshd/none (gfx950)",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "spec: xt_lds gfx942 d64 (predictor swizzle)",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: xt_lds gfx942 d128 K^T resident image",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: xt_lds gfx942 d32 explicit XT swizzles",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;lds_swizzle:dot=none,kt_stage=pad8,qt=pad16;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: xt_lds gfx950 d64 wide G4 atom",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: xt_lds gfx950 d128 split dQ",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: xt_lds gfx950 d32 bf16 kt lds",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:bf16",
    ),
    (
        "spec: row xor buffers gfx942 d64",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:dkv_out=xor,do=xor,ds=xor,k_stage=xor,q=xor,v_stage=xor;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: row pads gfx950 d64",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:dkv_out=pad16,ds=pad8,q=pad32,unused=none;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: ring-slotted row xor",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:q=xor;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: ring 3 gfx950 d64",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:3;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: counted gfx942 d64",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: counted gfx950 d128 late issue",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:late;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: counted late with stage_table_atomics",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:stage_table_atomics;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:late;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: counted early with stage_table_atomics",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:stage_table_atomics;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: counted with split dQ",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: counted with barrier merge",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: counted atomics past vmcnt",
        "geom|gfx942|head_size:128;k_m0:64;k_n0:32;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:counted;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: g4 slice_db gfx942 d64",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:slice_db;kw_dtype:fp16",
    ),
    (
        "spec: g4 slice -> step (kK4 = atom K)",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:16;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: g4 slice_db -> slice (kK4 = kN0)",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:64;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:slice;kw_dtype:fp16",
    ),
    (
        "spec: g4 slice_db_g3 gfx950 d128 kt lds",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:slice_db_g3;kw_dtype:fp16",
    ),
    (
        "spec: rejected: g4 slicing with ring",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: g4 slicing xt kt lds",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:slice;kw_dtype:fp16",
    ),
    (
        "spec: rejected: reservation below the estimate",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: acc_in_lds gfx942 d32",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:True;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: acc_in_lds d64 LDS plan",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:True;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: acc_in_lds with direct epilogue",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:True;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: small tile lds route gfx942 d64 16x16x4",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:16;k_k4:16;waves:4;grid_g0:1x1;grid_g1:1x4;grid_g4:1x4;atoms:16/16/16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: small tile lds route gfx950 d32 32x32x2",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:32;k_k4:32;waves:2;grid_g0:2x1;grid_g1:2x1;grid_g4:1x2;atoms:32/16/32;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: (2, W/2) grids gfx942 d64",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:2x4;grid_g1:2x4;grid_g4:2x4;atoms:16/16/16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: non-natural grid on relabel",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:2x4;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: grid not a CTA grid",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:4x4;grid_g4:2x4;atoms:16/16/16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: grid does not divide",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:1x8;grid_g4:2x2;atoms:16/16/16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: natural next to small grid does not divide",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: one wave on 32 x 32",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:32;k_k4:32;waves:1;grid_g0:1x1;grid_g1:1x1;grid_g4:1x1;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: one wave d128 debug tile gfx942 (scheduler)",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:16;k_k4:16;waves:1;grid_g0:1x1;grid_g1:1x1;grid_g4:1x1;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: wide atom on gfx942",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: tr_read on gfx942",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: dma on gfx942",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: dma gfx950 d32 ring 3",
        "geom|gfx950|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:tr_read;ring_depth:3;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: dma gfx950 tr_read q xor",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:dma;lds_swizzle:ds=pad16,q=xor;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: dma tr_read q pad",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:dma;lds_swizzle:q=pad8;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: ring 3 on gfx942",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:3;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: kK4 above kN0",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:32;k_k4:64;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: warp_grid_g4 shape",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:4x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: warp_grid_g4 does not divide D",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: block_n not a multiple of 16 W",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:64;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: sched stage_table with replicated grid",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:64;k_k4:16;waves:4;grid_g0:1x2;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:stage_table;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: lds_only gfx942 d64",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:lds_only;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: qdo joint too many chunks",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: loads_first with ring",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: stats step_top gfx950",
        "geom|gfx950|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:step_top;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: barrier merge gfx950 d64",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_issue:late;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: barrier merge gfx942 d128 g3g1g4",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: barrier merge with dma",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:dma;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: barrier merge G4 not last",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: barrier merge split dQ",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: barrier merge acc_in_lds",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:True;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:on;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: stage_table setprio gfx942",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:stage_table;setprio:True;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: stage_table_atomics split dQ",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:stage_table_atomics;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:split;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: xt_lds with ring",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:2;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: xt_lds with fences",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:xt_lds;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:fences_only;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: spill combo gfx942 d128 edge row xor",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,v_stage=xor;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: phase_low kv lds gfx942 d64",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: kt lds kv reg gfx942 lds_plain (k_res copy)",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: kv lds tr_read gfx950 d64",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: head_pack + edge tiles gfx950",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:True;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: empty lds_swizzle",
        "geom|gfx942|head_size:64;k_m0:32;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: wide atom_g13 gfx950 block_m 32",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/32/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: LDS plan too big",
        "geom|gfx942|head_size:128;k_m0:64;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: arch-VGPR estimate",
        "geom|gfx942|head_size:128;k_m0:64;k_n0:64;k_k4:32;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: workgroup share",
        "geom|gfx942|head_size:128;k_m0:64;k_n0:256;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: dq_issue fragment gfx942 d32 runtime",
        "geom|gfx942|head_size:32;k_m0:32;k_n0:128;k_k4:64;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:2x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:fragment;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: kK4 below the wide atom",
        "geom|gfx950|head_size:64;k_m0:32;k_n0:256;k_k4:16;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:32/16/32;pt_route:relabel;kv_residency:reg;kt_source:lds;transpose_source:tr_read;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: warp_grid_g4 does not divide kM0",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:128;k_k4:32;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:2x4;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: natural G1/G3 grid next to a small G0/G2 grid",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:32;k_k4:16;waves:4;grid_g0:1x2;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:lds;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: VGPR form within the prune slack",
        "geom|gfx942|head_size:64;k_m0:64;k_n0:128;k_k4:16;waves:4;grid_g0:1x4;grid_g1:4x1;grid_g4:1x4;atoms:16/16/16;pt_route:relabel;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: AGPR estimate past the slack",
        "geom|gfx942|head_size:64;k_m0:16;k_n0:256;k_k4:16;waves:2;grid_g0:1x2;grid_g1:2x1;grid_g4:1x2;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
    (
        "spec: rejected: workgroup share of eight waves",
        "geom|gfx942|head_size:128;k_m0:16;k_n0:256;k_k4:16;waves:8;grid_g0:1x8;grid_g1:8x1;grid_g4:1x8;atoms:16/16/16;pt_route:relabel;kv_residency:reg;kt_source:reg;transpose_source:lds_plain;ring_depth:1;global_path:vgpr;lds_swizzle:None;sched:none;setprio:False;head_pack:False;acc_in_lds:False;dq_mode:atomic;stage_vec:8;loop_sync:full;stats_load:at_use;dq_issue:block;barrier_merge:off;ring_issue:early;counted_issue:early;g4_slicing:step;kw_dtype:fp16",
    ),
)

# replace() variants of a "geom" table: every sched value, setprio inside a
# stage table, and the reference ("ck") model with and without deferral
_VARIANTS = (
    *({"sched": s} for s in SCHEDS),
    {"sched": "stage_table", "setprio": True},
    {"pipeline": "ck", "sched": "stage_table_atomics"},
    {"pipeline": "ck"},
)
_DQ_VARIANTS = tuple({"sched": s} for s in DQ_SCHEDS)
_ERRORS = (ValueError, KeyError)


def _value(name: str, text: str):
    if text == "None":
        return None
    if text in ("True", "False"):
        return text == "True"
    if text.lstrip("-").isdigit():
        return int(text)
    if name in _GRIDS:
        r, c = text.split("x")
        return (int(r), int(c))
    return text


def parse(config: str):
    kind, arch, rest = config.split("|", 2)
    fields = {}
    for item in rest.split(";"):
        if item:
            name, _, text = item.partition(":")
            fields[name] = _value(name, text)
    return kind, arch, fields


def _exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _pairs(ps) -> str:
    return "[" + " ".join(f"{m}:{c}" for m, c in ps) + "]"


def _mlist(text):
    """``"m=c,m=c"`` -> ((m, c), ...); None stays None."""
    if text is None:
        return None
    out = []
    for item in str(text).split(","):
        if item:
            m, _, c = item.partition("=")
            out.append((int(m), int(c)))
    return tuple(out)


def _try(lines: list, label: str, fn) -> None:
    try:
        lines.append(f"{label}: {fn()}")
    except _ERRORS as e:
        lines.append(f"{label}: {_exc(e)}")


def _stage_text(s) -> str:
    return (
        f"{s.stage} mfma={s.mfma} vr={s.vmem_read} vw={s.vmem_write} "
        f"dr={s.ds_read} dw={s.ds_write} lead={_pairs(s.lead)} "
        f"spans={_pairs(s.spans)} fixed={_pairs(s.fixed)} "
        f"memory={_pairs(s.memory())} census={s.census()!r} "
        f"groups={_pairs(s.groups())}"
    )


def _stage_lines(lines: list, label: str, fn) -> None:
    try:
        st = fn()
    except _ERRORS as e:
        lines.append(f"{label}: {_exc(e)}")
        return
    for s in st:
        try:
            lines.append(f"{label}: {_stage_text(s)}")
        except _ERRORS as e:
            lines.append(f"{label}: {s.stage} {_exc(e)}")


def _fields(t) -> str:
    return " ".join(f"{f.name}={getattr(t, f.name)!r}" for f in dataclasses.fields(t))


def _region_lines(lines: list, fn) -> None:
    try:
        rs = fn()
    except _ERRORS as e:
        lines.append(f"regions: {_exc(e)}")
        return
    for name, ops, lead in rs:
        lines.append(f"region {name} {ops!r} lead={lead!r}")


def table_lines(t) -> list:
    lines = ["fields: " + _fields(t)]
    _try(lines, "wave_size", lambda: t.wave_size)
    _try(lines, "instr_k", lambda: t.instr_k)
    for g in (*GEMMS, "g9"):
        _try(lines, f"op {g}", lambda g=g: f"{t.mma_op_id(g)} frag_len={t.frag_len(g)}")
    _try(lines, "c_frag_len", lambda: t.c_frag_len)
    _try(lines, "mfma", lambda: f"{t.mfma.as_dict()!r} total={t.mfma.total}")
    lines.append(f"stage_names: {t.stage_names!r}")
    _try(lines, "dq_atomics_per_lane", lambda: t.dq_atomics_per_lane)
    _try(lines, "stats_loads", lambda: t.stats_loads())
    lines.append(f"counted: {t.counted!r}")
    _try(lines, "counted_terms", lambda: t.counted_terms())
    _try(lines, "counted_waits", lambda: t.counted_waits())
    _try(lines, "ring_prefetch", lambda: t.ring_prefetch())
    _try(lines, "prefetch_region", lambda: t.prefetch_region())
    _try(lines, "prefetch_fence", lambda: t.prefetch_fence)
    _try(lines, "ring_waits", lambda: t.ring_waits())
    _try(lines, "ring_atomics_per_lane", lambda: t.ring_atomics_per_lane)
    _try(lines, "step_waits", lambda: t.step_waits())
    for pa in (True, False):
        _try(
            lines,
            f"vm_after_prefetch {pa}",
            lambda pa=pa: t.vm_after_prefetch(prev_atomics=pa),
        )
    for pf, pa in ((True, True), (True, False), (False, True), (False, False)):
        _stage_lines(
            lines,
            f"stages {pf} {pa}",
            lambda pf=pf, pa=pa: t.stages(prefetch=pf, prev_atomics=pa),
        )
    for name in (*STAGES, "X"):
        _try(lines, f"stage {name}", lambda name=name: _stage_text(t.stage(name)))
        _try(lines, f"groups {name}", lambda name=name: _pairs(t.groups(name)))
        _try(
            lines,
            f"groups {name} False False",
            lambda name=name: _pairs(
                t.groups(name, prefetch=False, prev_atomics=False)
            ),
        )
    _try(lines, "hint_sequence", lambda: " ".join(repr(h) for h in t.hint_sequence()))
    _region_lines(lines, t.regions)
    for pf, pa in ((True, True), (False, False)):
        _try(
            lines,
            f"ir_count_model {pf} {pa}",
            lambda pf=pf, pa=pa: " ".join(
                f"{k}={v!r}"
                for k, v in t.ir_count_model(prefetch=pf, prev_atomics=pa).items()
            ),
        )
    return lines


def dq_lines(t) -> list:
    lines = ["fields: " + _fields(t)]
    _try(lines, "wave_size", lambda: t.wave_size)
    lines.append(
        f"setprio={t.setprio!r} table={t.table!r} fenced={t.fenced!r} "
        f"stage_names={t.stage_names!r}"
    )
    _try(lines, "mfma", lambda: t.mfma)
    _try(lines, "prefetch", lambda: t.prefetch())
    _stage_lines(lines, "stages", t.stages)
    for name in (*DQ_STAGES, "D"):
        _try(lines, f"stage {name}", lambda name=name: _stage_text(t.stage(name)))
        _try(lines, f"groups {name}", lambda name=name: _pairs(t.groups(name)))
    _try(lines, "hint_sequence", lambda: " ".join(repr(h) for h in t.hint_sequence()))
    _region_lines(lines, t.regions)
    _try(
        lines,
        "ir_count_model",
        lambda: " ".join(f"{k}={v!r}" for k, v in t.ir_count_model().items()),
    )
    return lines


def _namespace(fields: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in fields.items():
        if k.startswith("kw_"):
            continue
        if k == "atoms" and v is not None:
            a02, a13, a4 = (None if x == "None" else int(x) for x in str(v).split("/"))
            v = SimpleNamespace(g02=a02, g13=a13, g4=a4)
        setattr(ns, k, v)
    return ns


def build(kind: str, arch: str, fields: dict):
    """The StageTable ("geom", "table") or DqStageTable ("dq", "dqgeom") of a
    config."""
    if kind == "geom":
        kw = {k[3:]: v for k, v in fields.items() if k.startswith("kw_")}
        return StageTable.from_geometry(_namespace(fields), arch, **kw)
    if kind == "table":
        return StageTable(arch=arch, **fields)
    if kind == "dqgeom":
        return DqStageTable.from_geometry(_namespace(dict(fields, arch=arch)))
    if kind == "dq":
        return DqStageTable(arch=arch, **fields)
    raise SystemExit(f"no table for kind {kind}")


def _with_variants(lines, base, variants, body) -> None:
    lines += body(base)
    for v in variants:
        label = " ".join(f"{k}={x!r}" for k, x in v.items())
        lines.append(f"variant {label}")
        try:
            t = dataclasses.replace(base, **v)
        except _ERRORS as e:
            lines.append("init: " + _exc(e))
            continue
        lines += body(t)


def dump(kind: str, arch: str, fields: dict) -> list:
    lines = [f"kind: {kind} arch: {arch}"]
    if kind in ("geom", "table"):
        try:
            t = build(kind, arch, fields)
        except _ERRORS as e:
            lines.append("init: " + _exc(e))
            return lines
        lines.append("init: ok")
        _with_variants(lines, t, _VARIANTS if kind == "geom" else (), table_lines)
    elif kind in ("dq", "dqgeom"):
        try:
            t = build(kind, arch, fields)
        except _ERRORS as e:
            lines.append("init: " + _exc(e))
            return lines
        lines.append("init: ok")
        _with_variants(lines, t, _DQ_VARIANTS if kind == "dqgeom" else (), dq_lines)
    elif kind == "groups":
        _try(
            lines,
            "groups",
            lambda: _pairs(
                interleave_groups(
                    fields["mfma"],
                    _mlist(fields["mem"]),
                    lead=(
                        None
                        if "lead" not in fields
                        else dict(_mlist(fields["lead"]) or ())
                    ),
                    spans=(
                        None
                        if "spans" not in fields
                        else dict(_mlist(fields["spans"]) or ())
                    ),
                )
            ),
        )
    elif kind == "counts":
        s = StageCounts(
            fields["stage"],
            fields["mfma"],
            vmem_read=fields.get("vmem_read", 0),
            vmem_write=fields.get("vmem_write", 0),
            ds_read=fields.get("ds_read", 0),
            ds_write=fields.get("ds_write", 0),
            lead=_mlist(fields.get("lead")) or (),
            spans=_mlist(fields.get("spans")) or (),
            fixed=_mlist(fields.get("fixed")) or (),
        )
        lines.append(f"by_mask: {s.by_mask()!r}")
        _try(lines, "stage", lambda: _stage_text(s))
    elif kind == "atoms":
        dtype = fields.get("dtype", "fp16")
        _try(lines, "native_atom_ks", lambda: native_atom_ks(arch, dtype))
        for k in (16, 32, 64, 8, 48):
            _try(lines, f"instruction_k {k}", lambda k=k: instruction_k(arch, k, dtype))
    elif kind == "dma":

        def batch() -> str:
            d = DmaBatch(
                fields["block_m"],
                fields["head_size"],
                fields["waves"],
                fields["wave_size"],
            )
            waves = " ".join(repr(d.of_wave(w)) for w in range(d.waves))
            return (
                f"pieces={d.pieces} instructions={d.instructions!r} "
                f"of_wave={waves} min_per_wave={d.min_per_wave}"
            )

        _try(lines, "batch", batch)

        def waits() -> str:
            w = dma_wait_counts(fields["ring_depth"], fields["n_qd"], fields["n_a"])
            return f"prologue={w.prologue!r} step_top={w.step_top!r} step_end={w.step_end!r}"

        _try(lines, "waits", waits)
    else:
        raise SystemExit(f"unknown kind {kind}")
    return lines


def print_c_configs(configs) -> None:
    for comment, config in configs:
        c = comment.replace("*/", "* /")
        print(f'    /* {c} */\n    "{config}",')


def run(configs, argv, name: str) -> int:
    if len(argv) > 1 and argv[1] == "--print-c-configs":
        print_c_configs(configs)
        return 0
    if len(argv) < 2:
        sys.stderr.write(f"usage: {name} <config_index> [ll]\n")
        return 2
    idx = int(argv[1])
    mode = argv[2] if len(argv) > 2 else "ll"
    if mode != "ll":
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    if not 0 <= idx < len(configs):
        raise SystemExit(f"unknown config index {idx}")
    kind, arch, fields = parse(configs[idx][1])
    sys.stdout.write("".join(line + "\n" for line in dump(kind, arch, fields)))
    return 0


def main() -> int:
    return run(_CONFIGS, sys.argv, "attention_bwd_sched_emit.py")


if __name__ == "__main__":
    raise SystemExit(main())
