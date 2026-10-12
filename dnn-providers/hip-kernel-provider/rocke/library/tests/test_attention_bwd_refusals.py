# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The validator's resource refusals (``RESOURCE_REFUSED``) are exactly the
recorded resource-gate failures.

Every row below failed the backward resource gate (or crashed the code-object
compiler) on the production compiler when every gate row of the suites was
compiled: the IR suite's instances (``ir``), the knob value sets on each
target's default tile in both dK / dV modes (``set``) and on the device test's
cases (``setcase``), the schedule values over the gate variants
(``sched``), the counted-wait suite's rows (``counted``) with their
full-barrier twins (``countedfull``), the mirror suite's rows (``mirror``)
with their one-tile twins (``mirrortwin``), and the main kernels the device
tests of mirror pairing, counted waits and the composition sets launch
(``mcell``, ``ccell``, ``compose``, each with its twin, and ``compknobs``:
a device case with explicit knobs). CPU checks: each row's resolved spec is refused with its reason,
the table holds nothing else, the digest ignores the name and defaulted
fields, and no shipped default kernel is refused.
"""

from __future__ import annotations

import dataclasses

import pytest

from kernels.common._attention_bwd_refusals import (
    RESOURCE_REFUSED,
    resource_refusal,
    spec_digest,
)
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    attn_bwd_default_geometry,
    validate_attn_bwd_spec,
)

# (row, failing rule) per recorded gate failure.
REFUSED_ROWS: tuple = (
    (('ir', 'gfx942', 'debug', 64, 'fp16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx942', 'debug', 64, 'bf16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx942', 'default', 32, 'fp16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx942', 'default', 32, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx942', 'default', 32, 'bf16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx942', 'default', 32, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx942', 'default', 64, 'fp16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx942', 'default', 64, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx942', 'default', 64, 'bf16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx942', 'default', 64, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('set', 'gfx942', 'warp_grid_2xw', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx942', 'warp_grid_2xw', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx942', 'sched_max_ilp', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx942', 'sched_max_memory_clause', 64, 'direct'), "sgpr_spill"),
    (('set', 'gfx942', 'sched_max_memory_clause', 64, 'atomic'), "sgpr_spill"),
    (('set', 'gfx942', 'sched_iterative_ilp', 32, 'direct'), "sgpr_spill"),
    (('set', 'gfx942', 'sched_iterative_ilp', 32, 'atomic'), "sgpr_spill"),
    (('set', 'gfx942', 'sched_iterative_ilp', 64, 'direct'), "compiler crash (comgr segfault)"),
    (('set', 'gfx942', 'sched_iterative_ilp', 64, 'atomic'), "compiler crash (comgr segfault)"),
    (('set', 'gfx942', 'sched_iterative_maxocc', 64, 'direct'), "sgpr_spill"),
    (('set', 'gfx942', 'sched_iterative_maxocc', 64, 'atomic'), "sgpr_spill"),
    (('set', 'gfx942', 'agpr_alloc_backend', 32, 'direct'), "agpr_copy_traffic"),
    (('set', 'gfx942', 'agpr_alloc_backend', 32, 'atomic'), "agpr_copy_traffic"),
    (('set', 'gfx942', 'block_m_64', 64, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx942', 'block_m_64', 128, 'direct'), "agpr_copy_traffic"),
    (('set', 'gfx942', 'block_m_64', 128, 'atomic'), "agpr_copy_traffic"),
    (('sched', 'gfx942', 'default', 32, 'band_edge', 'iglp1', False), "sgpr_spill"),
    (('ir', 'gfx950', 'default', 32, 'fp16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx950', 'default', 32, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx950', 'default', 32, 'bf16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx950', 'default', 32, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx950', 'default', 64, 'fp16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx950', 'default', 64, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx950', 'default', 64, 'bf16', 'band_edge_hout'), "sgpr_spill"),
    (('ir', 'gfx950', 'default', 64, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx950', 'default', 64, 'fp16', 'band_edge+joint'), "sgpr_spill"),
    (('set', 'gfx950', 'sched_iterative_ilp', 32, 'direct'), "sgpr_spill"),
    (('set', 'gfx950', 'sched_iterative_ilp', 32, 'atomic'), "sgpr_spill"),
    (('set', 'gfx950', 'sched_iterative_ilp', 64, 'direct'), "sgpr_spill"),
    (('set', 'gfx950', 'sched_iterative_ilp', 64, 'atomic'), "sgpr_spill"),
    (('set', 'gfx950', 'agpr_alloc_backend', 32, 'direct'), "agpr_copy_traffic"),
    (('set', 'gfx950', 'agpr_alloc_backend', 32, 'atomic'), "agpr_copy_traffic"),
    (('set', 'gfx950', 'block_m_64', 32, 'atomic'), "sgpr_spill"),
    (('set', 'gfx950', 'block_m_64', 64, 'direct'), "scratch+vgpr_spill+sgpr_spill"),
    (('set', 'gfx950', 'block_m_64', 64, 'atomic'), "scratch+vgpr_spill+sgpr_spill"),
    (('set', 'gfx950', 'block_m_64', 128, 'direct'), "agpr_copy_traffic"),
    (('set', 'gfx950', 'block_m_64', 128, 'atomic'), "agpr_copy_traffic"),
    (('set', 'gfx950', 'asm_g13_s_in_agpr', 64, 'direct'), "agpr_copy_traffic"),
    (('set', 'gfx950', 'asm_g13_s_in_agpr', 64, 'atomic'), "agpr_copy_traffic"),
    (('ir', 'gfx1151', 'debug', 32, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "sgpr_spill"),
    (('ir', 'gfx1151', 'debug', 64, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx1151', 'debug', 64, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill+sgpr_spill"),
    (('ir', 'gfx1151', 'debug', 128, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1151', 'debug', 128, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1151', 'multi', 64, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "sgpr_spill"),
    (('ir', 'gfx1151', 'multi', 64, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "sgpr_spill"),
    (('ir', 'gfx1151', 'multi', 128, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1151', 'multi', 128, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1151', 'default', 64, 'fp16', 'thd_band_atomic+top'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_iglp1', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_iglp1', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_stage_table', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_stage_table', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_stage_table_setprio', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_stage_table_setprio', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_stage_table_atomics', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_stage_table_atomics', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_stage_table_atomics_setprio', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'sched_stage_table_atomics_setprio', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx1151', 'xt_lds', 64, 'atomic'), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_pack', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_edge', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'thd_band_atomic', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'sv1_band', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_pack', 'fences_only', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_edge', 'fences_only', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_pack', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_edge', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'thd_band_atomic', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'sv1_band', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_pack', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_edge', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'thd_band_atomic', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'sv1_band', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_pack', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_edge', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'thd_band_atomic', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'sv1_band', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_pack', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'band_edge', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'thd_band_atomic', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'debug', 128, 'sv1_band', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_pack', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_edge', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'thd_band_atomic', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'sv1_band', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_pack', 'fences_only', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_pack', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_edge', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'thd_band_atomic', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'sv1_band', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_pack', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_edge', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'thd_band_atomic', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'sv1_band', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_pack', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_edge', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'thd_band_atomic', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'sv1_band', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_pack', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'band_edge', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'thd_band_atomic', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1151', 'multi', 128, 'sv1_band', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('ir', 'gfx1201', 'debug', 64, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1201', 'debug', 64, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1201', 'debug', 128, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1201', 'debug', 128, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1201', 'multi', 32, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "sgpr_margin"),
    (('ir', 'gfx1201', 'multi', 32, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "sgpr_margin"),
    (('ir', 'gfx1201', 'multi', 128, 'fp16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('ir', 'gfx1201', 'multi', 128, 'bf16', 'band_edge_atomic_pack_hout_ldssync'), "scratch+vgpr_spill"),
    (('set', 'gfx1201', 'sched_iglp1', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx1201', 'sched_iglp1', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx1201', 'sched_stage_table_atomics', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx1201', 'sched_stage_table_atomics', 128, 'atomic'), "scratch+vgpr_spill"),
    (('set', 'gfx1201', 'sched_stage_table_atomics_setprio', 128, 'direct'), "scratch+vgpr_spill"),
    (('set', 'gfx1201', 'sched_stage_table_atomics_setprio', 128, 'atomic'), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_pack', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_edge', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'thd_band_atomic', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'sv1_band', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_pack', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_edge', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_pack', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_edge', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_pack', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_edge', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'thd_band_atomic', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'sv1_band', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_pack', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'band_edge', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'thd_band_atomic', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'debug', 128, 'sv1_band', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'band', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'band_pack', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'band_edge', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'thd_band_atomic', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'sv1_band', 'iglp1', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'band_pack', 'stage_table', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'band_pack', 'stage_table', True), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'band_pack', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'band_edge', 'stage_table_atomics', False), "scratch+vgpr_spill"),
    (('sched', 'gfx1201', 'multi', 128, 'band_pack', 'stage_table_atomics', True), "scratch+vgpr_spill"),
    (('setcase', 'gfx942', 'warp_grid_2xw', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('setcase', 'gfx942', 'sched_max_memory_clause', 'bwd_seqlen_33x72_br'), "sgpr_spill"),
    (('setcase', 'gfx942', 'sched_max_memory_clause', 'bwd_kern_hkv_842_tl_fp16_d64'), "sgpr_spill"),
    (('setcase', 'gfx942', 'sched_iterative_ilp', 'bwd_seqlen_33x72_br'), "compiler crash (comgr segfault)"),
    (('setcase', 'gfx942', 'sched_iterative_ilp', 'bwd_kern_hkv_842_tl_fp16_d64'), "compiler crash (comgr segfault)"),
    (('setcase', 'gfx942', 'sched_iterative_maxocc', 'bwd_seqlen_33x72_br'), "sgpr_spill"),
    (('setcase', 'gfx942', 'sched_iterative_maxocc', 'bwd_kern_hkv_842_tl_fp16_d64'), "sgpr_spill"),
    (('setcase', 'gfx942', 'block_m_64', 'bwd_gqa_gqa4_causal_tl'), "agpr_copy_traffic"),
    (('setcase', 'gfx942', 'block_m_64', 'bwd_kern_hkv_842_tl_fp16_d64'), "scratch+vgpr_spill"),
    (('setcase', 'gfx950', 'sched_iterative_ilp', 'bwd_seqlen_33x72_br'), "sgpr_spill"),
    (('setcase', 'gfx950', 'sched_iterative_ilp', 'bwd_kern_hkv_842_tl_fp16_d64'), "sgpr_spill"),
    (('setcase', 'gfx950', 'block_m_64', 'bwd_seqlen_33x72_br'), "scratch+vgpr_spill+sgpr_spill"),
    (('setcase', 'gfx950', 'block_m_64', 'bwd_gqa_gqa4_causal_tl'), "agpr_copy_traffic"),
    (('setcase', 'gfx950', 'block_m_64', 'bwd_kern_hkv_842_tl_fp16_d64'), "scratch+vgpr_spill+sgpr_spill"),
    (('setcase', 'gfx950', 'asm_g13_s_in_agpr', 'bwd_seqlen_33x72_br'), "agpr_copy_traffic"),
    (('setcase', 'gfx950', 'asm_g13_s_in_agpr', 'bwd_kern_hkv_842_tl_fp16_d64'), "agpr_copy_traffic"),
    (('setcase', 'gfx1151', 'sched_iglp1', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('setcase', 'gfx1151', 'sched_stage_table', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('setcase', 'gfx1151', 'sched_stage_table_setprio', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('setcase', 'gfx1151', 'sched_stage_table_atomics', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('setcase', 'gfx1151', 'sched_stage_table_atomics_setprio', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('setcase', 'gfx1151', 'xt_lds', 'bwd_kern_hkv_842_tl_fp16_d64'), "scratch+vgpr_spill"),
    (('setcase', 'gfx1201', 'sched_iglp1', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('setcase', 'gfx1201', 'sched_stage_table_atomics', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('setcase', 'gfx1201', 'sched_stage_table_atomics_setprio', 'bwd_gqa_gqa4_causal_tl'), "scratch+vgpr_spill"),
    (('counted', 'gfx942', 'debug', 32, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 32, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 32, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 32, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 64, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 64, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 64, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 64, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 128, 'fp16', 'band_mirror_edge', 'early'), "scratch+vgpr_spill+sgpr_spill"),
    (('countedfull', 'gfx942', 'debug', 128, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 128, 'fp16', 'band_mirror_edge', 'late'), "scratch+vgpr_spill+sgpr_spill"),
    (('countedfull', 'gfx942', 'debug', 128, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 128, 'bf16', 'band_mirror_edge', 'early'), "scratch+vgpr_spill+sgpr_spill"),
    (('countedfull', 'gfx942', 'debug', 128, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 128, 'bf16', 'band_mirror_edge', 'late'), "scratch+vgpr_spill+sgpr_spill"),
    (('countedfull', 'gfx942', 'debug', 128, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 32, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 32, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 32, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 32, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 64, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 64, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 64, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 64, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 128, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 128, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 128, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 128, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 32, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 32, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 32, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 32, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 64, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 64, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 64, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 64, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 128, 'fp16', 'band_mirror_edge', 'early'), "scratch+vgpr_spill+sgpr_spill"),
    (('countedfull', 'gfx950', 'debug', 128, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 128, 'fp16', 'band_mirror_edge', 'late'), "scratch+vgpr_spill+sgpr_spill"),
    (('countedfull', 'gfx950', 'debug', 128, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 128, 'bf16', 'band_mirror_edge', 'early'), "scratch+vgpr_spill+sgpr_spill"),
    (('countedfull', 'gfx950', 'debug', 128, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 128, 'bf16', 'band_mirror_edge', 'late'), "scratch+vgpr_spill+sgpr_spill"),
    (('countedfull', 'gfx950', 'debug', 128, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 32, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 32, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 32, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 32, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 64, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('countedfull', 'gfx950', 'default', 64, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 64, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('countedfull', 'gfx950', 'default', 64, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 64, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('countedfull', 'gfx950', 'default', 64, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 64, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('countedfull', 'gfx950', 'default', 64, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 128, 'fp16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 128, 'fp16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 128, 'bf16', 'band_mirror_edge', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 128, 'bf16', 'band_mirror_edge', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 32, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 32, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 32, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 32, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 64, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 64, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 64, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'debug', 64, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 32, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 32, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 32, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 32, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 64, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 64, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 64, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('countedfull', 'gfx942', 'default', 64, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx942', 'default', 64, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('countedfull', 'gfx942', 'default', 64, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 32, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 32, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 32, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 32, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 64, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 64, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 64, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'debug', 64, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 32, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 32, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 32, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 32, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 64, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('countedfull', 'gfx950', 'default', 64, 'fp16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 64, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('countedfull', 'gfx950', 'default', 64, 'fp16', 'band_mirror', 'late'), "sgpr_spill"),
    (('countedfull', 'gfx950', 'default', 64, 'bf16', 'band_mirror', 'early'), "sgpr_spill"),
    (('counted', 'gfx950', 'default', 64, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('countedfull', 'gfx950', 'default', 64, 'bf16', 'band_mirror', 'late'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 32, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 32, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 32, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 64, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 64, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 64, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 128, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 128, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 128, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx942', 'debug', 128, 'band_xcd'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 32, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 32, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 32, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 64, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 64, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 64, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 128, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 128, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 128, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx942', 'default', 128, 'band_xcd'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_ring2', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_split', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_sta', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_sta', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_xcd', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_merge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_ring2', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_sta', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_sta', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted_edge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted_edge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_merge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_merge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_ring2', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_split', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_split', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xt', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_sta', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_sta', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_g4slice', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_xcd', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_merge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_edge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_edge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted_edge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted_edge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_merge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_merge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_ring2', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_split', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_split', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xt', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xt', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'default'), "sgpr_spill"),
    (('mcell', 'gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_g4slice', 'default'), "sgpr_spill"),
    (('ccell', 'gfx942', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'edge', 'default'), "sgpr_spill"),
    (('ccell', 'gfx942', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'qrot_edge', 'default'), "sgpr_spill"),
    (('compknobs', 'gfx942', 'bwd_seqlen_33x72_br', (('agpr_alloc', (128, 128)), ('dq_mode', 'split'))), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_gqa_gqa4_causal_tl', (('agpr_alloc', (128, 128)), ('dq_mode', 'split'))), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_kern_hkv_842_tl_fp16_d64', (('agpr_alloc', (128, 128)), ('dq_mode', 'split'))), "agpr_copy_traffic"),
    (('compose', 'gfx942', 'merge_asm_sagpr', 'bwd_seqlen_33x72_br'), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_seqlen_33x72_br', (('agpr_alloc', (128, 128)), ('barrier_merge', 'on'))), "agpr_copy_traffic"),
    (('compose', 'gfx942', 'merge_asm_sagpr', 'bwd_gqa_gqa4_causal_tl'), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_gqa_gqa4_causal_tl', (('agpr_alloc', (128, 128)), ('barrier_merge', 'on'))), "agpr_copy_traffic"),
    (('compose', 'gfx942', 'merge_asm_sagpr', 'bwd_kern_hkv_842_tl_fp16_d64'), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_kern_hkv_842_tl_fp16_d64', (('agpr_alloc', (128, 128)), ('barrier_merge', 'on'))), "agpr_copy_traffic"),
    (('compose', 'gfx942', 'ring2_asm_sagpr', 'bwd_seqlen_33x72_br'), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_seqlen_33x72_br', (('agpr_alloc', (128, 128)), ('ring_depth', 2))), "agpr_copy_traffic"),
    (('compose', 'gfx942', 'ring2_asm_sagpr', 'bwd_gqa_gqa4_causal_tl'), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_gqa_gqa4_causal_tl', (('agpr_alloc', (128, 128)), ('ring_depth', 2))), "agpr_copy_traffic"),
    (('compose', 'gfx942', 'ring2_asm_sagpr', 'bwd_kern_hkv_842_tl_fp16_d64'), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_kern_hkv_842_tl_fp16_d64', (('agpr_alloc', (128, 128)), ('ring_depth', 2))), "agpr_copy_traffic"),
    (('compose', 'gfx942', 'counted_asm_sagpr', 'bwd_seqlen_33x72_br'), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_seqlen_33x72_br', (('agpr_alloc', (128, 128)), ('loop_sync', 'counted'))), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_gqa_gqa4_causal_tl', (('agpr_alloc', (128, 128)), ('loop_sync', 'counted'))), "agpr_copy_traffic"),
    (('compose', 'gfx942', 'counted_asm_sagpr', 'bwd_kern_hkv_842_tl_fp16_d64'), "agpr_copy_traffic"),
    (('compknobs', 'gfx942', 'bwd_kern_hkv_842_tl_fp16_d64', (('agpr_alloc', (128, 128)), ('loop_sync', 'counted'))), "agpr_copy_traffic"),
    (('mirror', 'gfx950', 'debug', 32, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 32, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 32, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 64, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 64, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 64, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 128, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 128, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 128, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx950', 'debug', 128, 'band_xcd'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 32, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 32, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 32, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 64, 'band'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 64, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 64, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 64, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 64, 'band_xcd'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 128, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 128, 'band_atomic_pack'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 128, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx950', 'default', 128, 'band_xcd'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_ring2', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_sta', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_sta', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_g4slice', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_xcd', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_merge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_ring2', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_sta', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_sta', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted_edge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted_edge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_merge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_merge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_ring2', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_split', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xt', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_sta', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_sta', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_g4slice', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_xcd', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_merge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_edge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_edge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted_edge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted_edge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_merge', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_merge', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_ring2', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_ring2', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_split', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_split', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xt', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xt', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'debug'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'default'), "sgpr_spill"),
    (('mcell', 'gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_g4slice', 'default'), "sgpr_spill"),
    (('ccell', 'gfx950', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'edge', 'default'), "sgpr_spill"),
    (('ccell', 'gfx950', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'qrot_edge', 'default'), "sgpr_spill"),
    (('ccelltwin', 'gfx950', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'qrot_edge', 'default'), "sgpr_spill"),
    (('compknobs', 'gfx950', 'bwd_seqlen_33x72_br', (('agpr_alloc', (128, 128)), ('dq_mode', 'split'))), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_gqa_gqa4_causal_tl', (('agpr_alloc', (128, 128)), ('dq_mode', 'split'))), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_kern_hkv_842_tl_fp16_d64', (('agpr_alloc', (128, 128)), ('dq_mode', 'split'))), "agpr_copy_traffic"),
    (('compose', 'gfx950', 'merge_asm_sagpr', 'bwd_seqlen_33x72_br'), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_seqlen_33x72_br', (('agpr_alloc', (128, 128)), ('barrier_merge', 'on'))), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_gqa_gqa4_causal_tl', (('agpr_alloc', (128, 128)), ('barrier_merge', 'on'))), "agpr_copy_traffic"),
    (('compose', 'gfx950', 'merge_asm_sagpr', 'bwd_kern_hkv_842_tl_fp16_d64'), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_kern_hkv_842_tl_fp16_d64', (('agpr_alloc', (128, 128)), ('barrier_merge', 'on'))), "agpr_copy_traffic"),
    (('compose', 'gfx950', 'ring2_asm_sagpr', 'bwd_seqlen_33x72_br'), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_seqlen_33x72_br', (('agpr_alloc', (128, 128)), ('ring_depth', 2))), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_gqa_gqa4_causal_tl', (('agpr_alloc', (128, 128)), ('ring_depth', 2))), "agpr_copy_traffic"),
    (('compose', 'gfx950', 'ring2_asm_sagpr', 'bwd_kern_hkv_842_tl_fp16_d64'), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_kern_hkv_842_tl_fp16_d64', (('agpr_alloc', (128, 128)), ('ring_depth', 2))), "agpr_copy_traffic"),
    (('compose', 'gfx950', 'counted_asm_sagpr', 'bwd_seqlen_33x72_br'), "scratch+vgpr_spill+agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_seqlen_33x72_br', (('agpr_alloc', (128, 128)), ('loop_sync', 'counted'))), "agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_gqa_gqa4_causal_tl', (('agpr_alloc', (128, 128)), ('loop_sync', 'counted'))), "agpr_copy_traffic"),
    (('compose', 'gfx950', 'counted_asm_sagpr', 'bwd_kern_hkv_842_tl_fp16_d64'), "scratch+vgpr_spill+agpr_copy_traffic"),
    (('compknobs', 'gfx950', 'bwd_kern_hkv_842_tl_fp16_d64', (('agpr_alloc', (128, 128)), ('loop_sync', 'counted'))), "agpr_copy_traffic"),
    (('mirror', 'gfx1151', 'debug', 32, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1151', 'debug', 32, 'thd_band_slab'), "sgpr_margin"),
    (('mirror', 'gfx1151', 'debug', 64, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1151', 'debug', 64, 'thd_band_slab'), "sgpr_margin"),
    (('mirror', 'gfx1151', 'debug', 128, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1151', 'debug', 128, 'band_atomic_pack'), "scratch+vgpr_spill"),
    (('mirror', 'gfx1151', 'multi', 32, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1151', 'multi', 32, 'thd_band_slab'), "sgpr_margin"),
    (('mirror', 'gfx1151', 'multi', 64, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1151', 'multi', 64, 'thd_band_slab'), "sgpr_spill"),
    (('mirror', 'gfx1151', 'multi', 128, 'band_edge_hout'), "sgpr_spill"),
    (('mcell', 'gfx1151', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcelltwin', 'gfx1151', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1151', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'multi'), "scratch+vgpr_spill"),
    (('mcelltwin', 'gfx1151', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'multi'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1151', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1151', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1151', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1151', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1151', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'multi'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1151', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'debug'), "sgpr_margin"),
    (('mcell', 'gfx1151', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'multi'), "sgpr_margin"),
    (('mirror', 'gfx1201', 'debug', 32, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1201', 'debug', 64, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1201', 'debug', 64, 'band_atomic_pack'), "scratch+vgpr_spill"),
    (('mirror', 'gfx1201', 'debug', 128, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1201', 'debug', 128, 'band_atomic_pack'), "scratch+vgpr_spill"),
    (('mirror', 'gfx1201', 'multi', 32, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1201', 'multi', 64, 'band_edge_hout'), "sgpr_spill"),
    (('mirror', 'gfx1201', 'multi', 128, 'band_edge'), "sgpr_margin"),
    (('mirror', 'gfx1201', 'multi', 128, 'band_edge_hout'), "sgpr_spill"),
    (('mcell', 'gfx1201', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_edge', 'multi'), "sgpr_margin"),
    (('mcell', 'gfx1201', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcelltwin', 'gfx1201', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1201', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_edge', 'multi'), "sgpr_margin"),
    (('mcell', 'gfx1201', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'debug'), "scratch+vgpr_spill"),
    (('mcell', 'gfx1201', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'multi'), "sgpr_margin"),
)  # fmt: skip


def unresolved_spec(row) -> AttnBwdSpec:
    """The spec a gate row compiles, before resolution."""
    from . import test_attention_bwd_ir as T
    from .sdpa.bwd_kernel_cases import KNOB_SMOKE_SETS

    kind, arch = row[0], row[1]
    if kind == "ir":
        return T.spec_of(*row[1:])
    if kind == "sched":
        from . import test_attention_bwd_sched_body as SB

        _, _, cfg, d, var, sched, prio = row
        kw = dict(SB.config_knobs(cfg, arch), **SB.VARIANTS[var])
        kw.pop("sched", None)
        kw.pop("setprio", None)
        return AttnBwdSpec(head_size=d, dtype="fp16", sched=sched, setprio=prio, **kw)
    if kind == "setcase":
        return _setcase_spec(row)
    if kind in ("counted", "countedfull"):
        from .test_attention_bwd_counted_sync import spec_of as counted_spec_of

        spec = counted_spec_of(*row[1:])
        if kind == "countedfull":  # the full-barrier twin of the counted row
            spec = dataclasses.replace(spec, loop_sync="full", counted_issue="early")
        return spec
    if kind in ("mirror", "mirrortwin"):
        from .test_attention_bwd_mirror import _spec as mirror_spec

        return mirror_spec(*row[1:], lb_mirror=(kind == "mirror"))
    if kind in _CELL_KINDS:
        return _cell_spec(row)
    _, _, name, d, mode = row
    kw = dict(T.config_knobs("default", arch))
    kw.update(mask_class="band", dkv_mode=mode)
    kw.update(KNOB_SMOKE_SETS[name][2])
    return AttnBwdSpec(head_size=d, dtype="fp16", **kw)


def _setcase_spec(row) -> AttnBwdSpec:
    """The main spec the device test of knob set ``name`` plans for its case
    (the plan's policy resolution, without the validator's refusal)."""
    from unittest import mock

    from kernels.common import attention_bwd as AB
    from kernels.common.attention_bwd_plan import attn_bwd_plan

    from .sdpa.bwd_kernel_cases import (
        KNOB_SMOKE_SETS,
        config_policy,
        request_from_case,
    )
    from .test_attention_bwd_kernel_cases import _smoke_case

    _, arch, name, cid = row
    pol = config_policy("default", arch, dict(KNOB_SMOKE_SETS[name][2]))
    with mock.patch.object(AB, "resource_refusal", lambda s, a: None):
        plan = attn_bwd_plan(request_from_case(_smoke_case(cid)), arch, policy=pol)
    return plan.specs["main"]


_CELL_KINDS = (
    "mcell", "mcelltwin", "ccell", "ccelltwin", "compose", "composetwin", "compknobs",
)  # fmt: skip


def _unrefused_plan(case, arch, config, overrides=None, request=None):
    """The device harness's plan with the validator's resource refusal
    bypassed (the spec a refused cell would launch)."""
    from unittest import mock

    from kernels.common import attention_bwd as AB

    from ._attention_bwd_harness import plan_case

    with mock.patch.object(AB, "resource_refusal", lambda s, a: None):
        return plan_case(
            case, arch=arch, config=config, overrides=overrides, request=request
        )


def _cell_spec(row) -> AttnBwdSpec:
    """The main spec of a device-test cell: mirror pairing (``mcell``),
    counted waits (``ccell``) and the composition twin sets (``compose``),
    each with its twin (the run the test compares against), or a device case
    on the default tile with explicit knobs (``compknobs``)."""
    kind, arch = row[0], row[1]
    if kind in ("mcell", "mcelltwin"):
        from . import test_attention_bwd_sched_numeric as S

        _, _, cid, var, cfg = row
        case = S._case(cid)
        extra, req = S.MIRROR_VARIANTS[var]
        if kind == "mcelltwin":
            twin = {k: v for k, v in extra.items() if k != "xcd_chunk"}
            return _unrefused_plan(case, arch, cfg, twin or None).specs["main"]
        overrides = {"lb_order": "mirror", **extra}
        request = {"num_cus": 1, **req}
        if overrides.get("xcd_chunk") == "gx":
            plan = _unrefused_plan(case, arch, cfg, overrides, request)
            overrides["xcd_chunk"] = plan.launches[1].grid[0]
        return _unrefused_plan(case, arch, cfg, overrides, request).specs["main"]
    if kind in ("ccell", "ccelltwin"):
        from . import test_attention_bwd_sched_numeric as S

        _, _, cid, var, cfg = row
        kw = S.COUNTED_VARIANTS[var]
        if kind == "ccelltwin":
            kw = {k: v for k, v in kw.items() if k != "counted_issue"}
        else:
            kw = {"loop_sync": "counted", **kw}
        return _unrefused_plan(S._case(cid), arch, cfg, kw or None).specs["main"]
    from .sdpa.bwd_kernel_cases import COMPOSE_TWIN_SETS
    from .test_attention_bwd_numeric import _case

    if kind == "compknobs":  # a device case on the default tile with these knobs
        _, _, cid, items = row
        return _unrefused_plan(_case(cid), arch, "default", dict(items)).specs["main"]
    _, _, name, cid = row
    _archs, _sizes, knobs, twin = COMPOSE_TWIN_SETS[name]
    kn = knobs if kind == "compose" else twin
    return _unrefused_plan(_case(cid), arch, "default", dict(kn)).specs["main"]


def _resolved(row):
    return unresolved_spec(row).resolved(row[1])


@pytest.mark.parametrize("row,rule", REFUSED_ROWS, ids=lambda x: str(x))
def test_every_recorded_row_is_refused(row, rule):
    arch = row[1]
    reason = resource_refusal(_resolved(row), arch)
    assert reason is not None and reason.startswith(rule), (row, reason)
    with pytest.raises(ValueError, match="RESOURCE_REFUSED"):
        validate_attn_bwd_spec(unresolved_spec(row), arch)


def test_the_table_holds_exactly_the_recorded_rows():
    keys = {(row[1], spec_digest(_resolved(row), row[1])) for row, _ in REFUSED_ROWS}
    assert keys == set(RESOURCE_REFUSED)


def test_the_digest_ignores_the_name_and_defaulted_fields():
    s = AttnBwdSpec(head_size=64).resolved("gfx942")
    d = spec_digest(s, "gfx942")
    assert spec_digest(dataclasses.replace(s, name="other"), "gfx942") == d
    assert spec_digest(s, "gfx950") != d
    # a field spelled at its default is the same kernel
    spelled = dataclasses.replace(s, dkv_atomic_guard="branch")
    assert spec_digest(spelled, "gfx942") == d
    assert spec_digest(dataclasses.replace(s, edge_tiles=True), "gfx942") != d


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_no_default_start_point_is_refused(arch):
    for d in (32, 64, 128):
        for dt in ("fp16", "bf16"):
            for sv in (8, 1):
                for seq in ("batched", "thd"):
                    g = attn_bwd_default_geometry(arch, d, dt, sv, seq)
                    assert g  # the policy resolves a start point
                    s = AttnBwdSpec(
                        head_size=d, dtype=dt, stage_vec=sv, seq_mode=seq
                    ).resolved(arch)
                    assert resource_refusal(s, arch) is None, (arch, d, dt, sv, seq)
