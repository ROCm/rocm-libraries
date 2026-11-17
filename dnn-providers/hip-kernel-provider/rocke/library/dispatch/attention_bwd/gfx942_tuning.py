# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuned tables of the attention backward on gfx942 (selection policy).

The gfx942 entries of the two tuned tables the backward routing reads
(:func:`~.routing.routed_policy`):

* :data:`GENERAL_TUNED_TABLE`: ``{tuning key: knobs}`` of the general family,
  keyed by :func:`kernels.common.attention_bwd_plan.attn_bwd_tuning_key` (the
  performance class, which names the arch, plus the kv-tile and grid-fill
  buckets; ``ANY_BUCKET`` matches any bucket). The plan takes the most
  specific match (``AttnBwdPolicy.tuned_table``).
* :data:`DENSE_TUNED`: ``{(arch, dense key): knobs}`` of the dense family,
  keyed by :func:`kernels.common.attention_bwd_plan._dense_key`; a request
  routes to the dense family only when its exact key is here (and its route
  is eligible). Merged into :data:`~.dense_tuning.DENSE_BWD_TUNED`.
* :data:`GENERAL_PERF_STATUS` / :data:`DENSE_PERF_STATUS`: ``{performance
  class: status}`` of the general family / of the dense keys
  (``AttnBwdPolicy.perf_status``; a class missing here is ``unverified``).
  Empty by policy: no per-class performance claim ships in the library, so
  every served class reads ``unverified``. The machinery stays (the plan
  records the status, the declaration publishes it, the opt-in
  ``perf_gate`` declines on it), and the status is information only: the
  routed policy serves every class the predicate accepts whatever its
  status.

An entry is a measured winner: the configuration passed the bench's confirm
stage on every measured cell of its key (a clean A/A control, the
pruning-set gate on; a dense entry also a paired win over the general
winner), one configuration on all of them (or within noise of each cell's
best), and its key passed :mod:`.tuned_tables`' check (its knobs plan on every
representative request of the key). The fill bucket of a key is the one of
the measured cells on the device (their request carried the device's
``num_cus``); a request without ``num_cus`` (fill bucket ``unknown``) keeps
the defaults. Keys whose cells disagree, whose winner needs a value the key
does not fix (a ``g_split`` that does not divide every head group of the
class, mirror pairing below an overfilled grid, an XCD remap without
``num_xcds``) or that have no confirmed cell are left out.

Regeneration (the selection rule above, no timing data in the repo):

1. run the sweep (``benchmarks/common/attention_bwd_sweep.py``) through its
   ``confirm`` stage for the arch's general and dense slices, results
   outside the source tree (``--out-dir``);
2. dense: ``attention_bwd_sweep.py --emit-dense-table PATH --out-dir DIR``
   writes ``{"entries": [{"arch", "key", "knobs", "cells"}]}`` (the keys whose
   confirmed dense arm has a paired win over the general winner and a clean
   A/A control); :func:`dense_entries` checks every entry;
3. general: the confirmed winner of every census cell, keyed by
   :func:`kernels.common.attention_bwd_plan.attn_bwd_tuning_key` of the
   cell's request (with the device's ``num_cus``), one configuration per key
   (the rule above), as ``{"entries": [{"key", "knobs"}]}``;
   :func:`general_entries` checks every entry;
4. the checked mappings replace the tables below.

Every entry is checked again when the package is imported
(:func:`.tuned_tables.check_arch_tables`, from :mod:`.routing`) and on every
representative request of its key in the tests.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import tuned_tables

__all__ = [
    "ARCH",
    "DENSE_PERF_STATUS",
    "DENSE_TUNED",
    "GENERAL_PERF_STATUS",
    "GENERAL_TUNED_TABLE",
    "check_dense_entry",
    "check_general_entry",
    "dense_entries",
    "general_entries",
]

ARCH = "gfx942"

# {attn_bwd_tuning_key(req, "gfx942"): knobs}
GENERAL_TUNED_TABLE: Mapping[tuple, Mapping[str, Any]] = {
    (
        "gfx942",
        "general",
        128,
        "bf16",
        "causal_tl",
        "gqa",
        "unequal",
        "fixed",
        8,
        "t16",
        "full",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": None,
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 32,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 2),
        "warp_grid_g13": (2, 1),
        "warp_grid_g4": (1, 2),
        "waves": 2,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
    (
        "gfx942",
        "general",
        128,
        "bf16",
        "causal_tl",
        "mha",
        "square",
        "fixed",
        8,
        "t16",
        "over",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": (192, 192),
        "agpr_route": "asm_g13",
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 128,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "lb_order": "mirror",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 4),
        "warp_grid_g13": (4, 1),
        "warp_grid_g4": (1, 4),
        "waves": 4,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
    (
        "gfx942",
        "general",
        128,
        "bf16",
        "causal_tl",
        "mha",
        "square",
        "fixed",
        8,
        "t64",
        "over",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": (192, 192),
        "agpr_route": "asm_g13",
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 128,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "lb_order": "mirror",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 4),
        "warp_grid_g13": (4, 1),
        "warp_grid_g4": (1, 4),
        "waves": 4,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
    (
        "gfx942",
        "general",
        128,
        "bf16",
        "none",
        "mha",
        "square",
        "fixed",
        8,
        "t16",
        "over",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": (192, 192),
        "agpr_route": "asm_g13",
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 128,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 4),
        "warp_grid_g13": (4, 1),
        "warp_grid_g4": (1, 4),
        "waves": 4,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
    (
        "gfx942",
        "general",
        128,
        "fp16",
        "causal_tl",
        "mha",
        "square",
        "fixed",
        8,
        "t16",
        "over",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": (192, 192),
        "agpr_route": "asm_g13",
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 128,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "lb_order": "mirror",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 4),
        "warp_grid_g13": (4, 1),
        "warp_grid_g4": (1, 4),
        "waves": 4,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
    (
        "gfx942",
        "general",
        64,
        "bf16",
        "causal_br",
        "gqa",
        "square",
        "fixed",
        8,
        "t64",
        "full",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": None,
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 64,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "lb_order": "mirror",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 2),
        "warp_grid_g13": (2, 1),
        "warp_grid_g4": (1, 2),
        "waves": 2,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
    (
        "gfx942",
        "general",
        64,
        "bf16",
        "causal_tl",
        "gqa",
        "square",
        "fixed",
        8,
        "t16",
        "full",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": None,
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 64,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "lb_order": "mirror",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 2),
        "warp_grid_g13": (2, 1),
        "warp_grid_g4": (1, 2),
        "waves": 2,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
    (
        "gfx942",
        "general",
        64,
        "bf16",
        "none",
        "mha",
        "square",
        "thd",
        8,
        "t16",
        "over",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": None,
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 64,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 2),
        "warp_grid_g13": (2, 1),
        "warp_grid_g4": (1, 2),
        "waves": 2,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
    (
        "gfx942",
        "general",
        64,
        "fp16",
        "none",
        "mha",
        "square",
        "thd",
        8,
        "t64",
        "over",
    ): {
        "acc_in_lds": False,
        "agpr_alloc": None,
        "atom_g02": "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x16",
        "block_k4": 32,
        "block_m": 16,
        "block_n": 64,
        "edge_tiles": False,
        "g_split": 1,
        "global_path": "vgpr",
        "kt_source": "reg",
        "kv_residency": "reg",
        "pt_route": "relabel",
        "ring_depth": 2,
        "s_in_agpr": False,
        "sched": "none",
        "scheduler_strategy": None,
        "setprio": False,
        "stats_fma": True,
        "stats_vec": 4,
        "transpose_source": "lds_plain",
        "warp_grid_g02": (1, 2),
        "warp_grid_g13": (2, 1),
        "warp_grid_g4": (1, 2),
        "waves": 2,
        "waves_per_eu": 1,
        "ws_layout": "head_major",
    },
}
# {("gfx942", dense key): dense knobs}
DENSE_TUNED: Mapping[tuple, Mapping[str, Any]] = {}
# {general performance class: status}; empty by policy (no per-class
# performance claim ships in the library: every class reads unverified)
GENERAL_PERF_STATUS: Mapping[tuple, str] = {}
# {dense performance class (arch, "dense", *dense key): status}; empty by
# policy
DENSE_PERF_STATUS: Mapping[tuple, str] = {}


def check_dense_entry(key: tuple, knobs: Mapping[str, Any], req=None) -> None:
    """Raise ``ValueError`` unless ``(key, knobs)`` is a gfx942 dense entry
    whose knobs plan on the key (:func:`.tuned_tables.check_dense_entry`;
    ``req``: a request of the key planned besides the representatives)."""
    tuned_tables.check_dense_entry(ARCH, key, knobs, req)


def check_general_entry(key: tuple, knobs: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``(key, knobs)`` is a gfx942 general entry
    whose knobs plan on the key (:func:`.tuned_tables.check_general_entry`)."""
    tuned_tables.check_general_entry(ARCH, key, knobs)


def dense_entries(doc: Mapping | Iterable, requests: Mapping | None = None) -> dict:
    """Checked gfx942 dense entries of a dense table document
    (:func:`.tuned_tables.dense_entries`)."""
    return tuned_tables.dense_entries(ARCH, doc, requests)


def general_entries(doc: Mapping | Iterable) -> dict:
    """Checked gfx942 general entries of a general table document
    (:func:`.tuned_tables.general_entries`)."""
    return tuned_tables.general_entries(ARCH, doc)
