# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device numerics of the scheduling levers: mirror pairing and counted waits.

Every run goes ``prep -> main -> convert`` through the host plan and is held
to the float64 oracle (``compare_bwd``) and the harness integrity facts; then:

* ``test_mirror_pairing_agrees``: ``lb_order = "mirror"`` (two kv tiles per
  CTA: ``t`` and ``n_kv_tiles - 1 - t``) on cases with odd and even kv-tile
  counts (MHA, GQA, ``h_k != h_v`` atomic, THD, a two-sided window, with the
  XCD remap, with edge tiles, with counted waits (with edge tiles as well it
  is refused on every tile); the cells the validator refuses,
  ``MIRROR_REFUSED``, are skipped with their reason): the plan pairs
  (``pair = 1``, half the grid), and in direct mode dK / dV are bit for bit
  those of the
  one-tile run (every kv tile is computed by the same code; dQ is summed by
  fp32 atomics in another order: the oracle decides);
* ``test_counted_waits_agree``: ``loop_sync = "counted"`` at both
  ``counted_issue`` values, alone and with edge tiles, the head-outer order,
  head packing, the LDS P / dS route, another stage order, per-fragment G4,
  q rotation and atomic dK / dV: dK / dV bit for bit equal to the full twin
  (the same overrides with ``loop_sync = "full"``) in direct mode, and dQ bit
  for bit equal where a dQ element has a single contributor (one kv tile, no
  head split).

``ROCKE_BWD_SHARD=i/n`` deals the cases over shards; ``ROCKE_BWD_CONFIG``
selects the configurations (default: every distinct configuration). Skipped
unless the device is gfx942 or gfx950 (counted waits are CDNA-only; mirror
pairing runs on every target and is also covered on the RDNA targets).
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from ._attention_bwd_harness import (
    ARCH,
    DEVICE_ARCHS,
    plan_case,
    run_case,
    skip_unless_planned,
)
from .sdpa.bwd_cases import _mkb, compare_bwd, validate_bwd_case
from .sdpa.bwd_kernel_cases import (
    CUSTOM_CONFIG,
    HEAD_PACK_CASES,
    selected_configs,
    shard_of,
)
from .sdpa.cases import _mask

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]
_MFMA = ("gfx942", "gfx950")


def _cases():
    """Multi-kv-tile cases: odd and even tile counts at kN0 128 and 256."""
    out = []
    for name, dt, d, h_q, h_k, h_v, sq, skv, mask in (
        ("mha_tl", "fp16", 64, 4, 4, 4, 520, 520, _mask("tl")),
        ("gqa4_br", "bf16", 128, 8, 2, 2, 300, 512, _mask("br")),
        ("mha_br_d32", "fp16", 32, 2, 2, 2, 260, 390, _mask("br")),
        ("hkv_842_tl", "bf16", 64, 8, 4, 2, 200, 400, _mask("tl")),
        (
            "gqa2_win2_tl", "fp16", 128, 4, 2, 2, 384, 384,
            dict(_mask("tl"), left_bound=150, right_bound=20),
        ),
    ):  # fmt: skip
        out.append(
            _mkb(
                f"sched_{name}_{dt}_d{d}",
                reqs=(3, 9),
                tier="smoke",
                tags=("optional",) if d == 32 else (),
                dtype=dt,
                d=d,
                b=2,
                h_q=h_q,
                h_k=h_k,
                h_v=h_v,
                s_q=sq,
                s_kv=skv,
                rdna=True,
                **mask,
            )
        )
    out.append(
        _mkb(
            "sched_thd_mha_br_bf16_d64",
            reqs=(3, 9),
            tier="smoke",
            dtype="bf16",
            d=64,
            b=3,
            h_q=4,
            h_k=4,
            h_v=4,
            q_lens=(300, 77, 512),
            kv_lens=(512, 300, 129),
            length_mode="ragged",
            rdna=True,
            **_mask("br"),
        )
    )
    for c in out:
        problems = validate_bwd_case(c)
        if problems:
            raise AssertionError(f"{c.id}: {problems}")
    return tuple(out)


CASES = _cases()
# the head-packed case of the counted variants (short query spans)
PACK_CASE = HEAD_PACK_CASES[0]


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


def _shard(ids):
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [c for c in ids if shard_of(c, n) == i]


_PLAIN: dict = {}


def _plain(case, config, overrides=None):
    key = (case.id, config, tuple(sorted((overrides or {}).items())))
    if key not in _PLAIN:
        run = run_case(case, arch=ARCH, config=config, overrides=overrides)
        _passes(run)
        _PLAIN[key] = run
    return _PLAIN[key]


# --------------------------------------------------------------------------- mirror
MIRROR_VARIANTS = {
    "mirror": ({}, {}),
    "mirror_xcd": ({"xcd_chunk": "gx"}, {"num_xcds": 2}),
    "mirror_edge": ({"edge_tiles": True}, {}),
    "mirror_counted": ({"loop_sync": "counted"}, {}),
    # refused on every device cell (pinned): counted waits with mirror pairing
    # and edge tiles fail the resource gate on every tile
    "mirror_counted_edge": ({"edge_tiles": True, "loop_sync": "counted"}, {}),
    # levers built apart from the mirror pairing (CDNA)
    "mirror_merge": ({"barrier_merge": "on"}, {}),
    "mirror_ring2": ({"ring_depth": 2}, {}),
    "mirror_split": ({"dq_mode": "split"}, {}),
    "mirror_xt": ({"transpose_source": "xt_lds"}, {}),
    "mirror_sta": ({"sched": "stage_table_atomics"}, {}),
    "mirror_g4slice": ({"g4_slicing": "slice_db", "block_k4": 32}, {}),
}
# Levers built apart from the mirror pairing (skipped with the validator's
# reason where the configuration's main kernel cannot take them).
_MIRROR_COMPOSE = (
    "mirror_merge",
    "mirror_ring2",
    "mirror_split",
    "mirror_xt",
    "mirror_sta",
    "mirror_g4slice",
)
# Variants whose levers are CDNA-only (counted waits, rings, split dQ, XT
# images on the device's default tile, G4 key slicing).
_MIRROR_MFMA_ONLY = {
    "mirror_counted",
    "mirror_counted_edge",
    "mirror_merge",
    "mirror_ring2",
    "mirror_split",
    "mirror_xt",
    "mirror_g4slice",
}


# Cells whose main kernel the validator refuses (RESOURCE_REFUSED: on the
# production compiler the paired kernel spills SGPRs on many tiles, and with
# stage-table atomics or one-wave d128 tiles scratch / VGPRs). The other cells
# of each variant still run; test_attention_bwd_mirror.py pins this table to
# the planner (every listed cell is refused, no other cell is).
MIRROR_REFUSED = {
    ('gfx1151', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'debug'): "scratch+vgpr_spill",
    ('gfx1151', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'multi'): "scratch+vgpr_spill",
    ('gfx1151', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'): "scratch+vgpr_spill",
    ('gfx1151', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'multi'): "scratch+vgpr_spill",
    ('gfx1151', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'debug'): "scratch+vgpr_spill",
    ('gfx1151', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'debug'): "scratch+vgpr_spill",
    ('gfx1151', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'debug'): "scratch+vgpr_spill",
    ('gfx1151', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'debug'): "sgpr_margin",
    ('gfx1151', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'multi'): "sgpr_margin",
    ('gfx1201', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_edge', 'multi'): "sgpr_margin",
    ('gfx1201', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'debug'): "scratch+vgpr_spill",
    ('gfx1201', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_edge', 'multi'): "sgpr_margin",
    ('gfx1201', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'): "scratch+vgpr_spill",
    ('gfx1201', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'multi'): "sgpr_margin",
    ('gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_counted_edge', 'debug'): "scratch+vgpr_spill+sgpr_spill",
    ('gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_edge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_merge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'debug'): "scratch+vgpr_spill",
    ('gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_xcd', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_counted_edge', 'debug'): "scratch+vgpr_spill+sgpr_spill",
    ('gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_edge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_merge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'): "scratch+vgpr_spill",
    ('gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_xcd', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted_edge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_g4slice', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_merge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_merge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_ring2', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_split', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_split', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_sta', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_sta', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xt', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted_edge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_ring2', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_sta', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_sta', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted_edge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_ring2', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_split', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_sta', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_mha_tl_fp16_d64', 'mirror_sta', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted_edge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_edge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_edge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_g4slice', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_merge', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_merge', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_ring2', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_split', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_split', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xt', 'debug'): "sgpr_spill",
    ('gfx942', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xt', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_counted_edge', 'debug'): "scratch+vgpr_spill+sgpr_spill",
    ('gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_edge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_merge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_sta', 'debug'): "scratch+vgpr_spill",
    ('gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_xcd', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa2_win2_tl_fp16_d128', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_counted_edge', 'debug'): "scratch+vgpr_spill+sgpr_spill",
    ('gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_edge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_merge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_sta', 'debug'): "scratch+vgpr_spill",
    ('gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_xcd', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_gqa4_br_bf16_d128', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted_edge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_g4slice', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_merge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_merge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_ring2', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_split', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_sta', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_sta', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_hkv_842_tl_bf16_d64', 'mirror_xt', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted_edge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_ring2', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_sta', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_br_d32_fp16_d32', 'mirror_sta', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted_edge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_g4slice', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_ring2', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_sta', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_sta', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_mha_tl_fp16_d64', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted_edge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_counted_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_edge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_edge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_g4slice', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_merge', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_merge', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_ring2', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_ring2', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_split', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_split', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_sta', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xcd', 'default'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xt', 'debug'): "sgpr_spill",
    ('gfx950', 'bwd_sched_thd_mha_br_bf16_d64', 'mirror_xt', 'default'): "sgpr_spill",
}  # fmt: skip


def _mirror_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    out = []
    for cid in _shard([c.id for c in CASES]):
        for var in MIRROR_VARIANTS:
            if var in _MIRROR_MFMA_ONLY and ARCH not in _MFMA:
                continue
            for cfg in selected_configs(ARCH):
                refused = MIRROR_REFUSED.get((ARCH, cid, var, cfg))
                marks = (
                    [
                        pytest.mark.skip(
                            reason=f"the validator refuses this cell ({refused})"
                        )
                    ]
                    if refused
                    else []
                )
                out.append(
                    pytest.param(cid, var, cfg, id=f"{cid}-{var}-{cfg}", marks=marks)
                )
    return out


def _case(cid):
    for c in CASES + (PACK_CASE,):
        if c.id == cid:
            return c
    raise KeyError(cid)


@pytest.mark.parametrize("case_id,variant,config", _mirror_params())
def test_mirror_pairing_agrees(case_id, variant, config):
    case = _case(case_id)
    extra, req = MIRROR_VARIANTS[variant]
    # the twin keeps the natural CTA order explicitly (a configuration may
    # pair itself: a custom one)
    twin = {"lb_order": "natural", **extra}
    if twin.get("xcd_chunk") == "gx":
        # the remap chunk is the paired grid x
        twin.pop("xcd_chunk")
    if config == CUSTOM_CONFIG:
        # MIRROR_REFUSED records the shipped configurations: under the custom
        # configuration a twin or a paired cell its validator refuses (a
        # coupling or a recorded resource refusal) skips with the reason
        kw = dict(arch=ARCH, config=config)
        skip_unless_planned(case, overrides=twin, **kw)
        skip_unless_planned(
            case, overrides={"lb_order": "mirror", **extra},
            request={"num_cus": 1, **req}, **kw,
        )  # fmt: skip
    elif variant in _MIRROR_COMPOSE:
        # a lever the configuration's main kernel cannot take is skipped with
        # the validator's reason
        try:
            plan_case(
                case, arch=ARCH, config=config,
                overrides={"lb_order": "mirror", **extra},
                request={"num_cus": 1, **req},
            )  # fmt: skip
        except ValueError as exc:
            if "RESOURCE_REFUSED" in str(exc):  # refusals are MIRROR_REFUSED only
                pytest.fail(f"refused but not in MIRROR_REFUSED: {exc}")
            pytest.skip(f"{extra} with mirror pairing not legal on {config}: {exc}")
    base = _plain(case, config, twin)
    overrides = {"lb_order": "mirror", **extra}
    request = {"num_cus": 1, **req}
    if overrides.get("xcd_chunk") == "gx":
        gx = (
            plan_case(
                case, arch=ARCH, config=config, overrides=overrides, request=request
            )
            .launches[1]
            .grid[0]
        )
        overrides["xcd_chunk"] = gx
    run = run_case(case, arch=ARCH, config=config, overrides=overrides, request=request)
    spec = run.plan.specs["main"]
    assert spec.lb_mirror is True
    main = run.plan.launches[1]
    assert main.scalars["pair"] == 1
    n = main.scalars["n_kv_tiles"]
    assert main.grid[0] == -(-n // 2)
    if "xcd_chunk" in overrides:
        assert spec.xcd_remap is True and run.plan.runtime["xcd_n"] == 2
    _passes(run)
    if spec.dkv_mode == "direct":
        assert np.array_equal(run.dk, base.dk), "dK differs under mirror pairing"
        assert np.array_equal(run.dv, base.dv), "dV differs under mirror pairing"


# --------------------------------------------------------------------------- counted
COUNTED_VARIANTS = {
    "early": {},
    "late": {"counted_issue": "late"},
    "edge": {"edge_tiles": True},
    "late_hout": {"counted_issue": "late", "step_order": "head_outer"},
    "ptlds": {"pt_route": "lds"},
    "g4g1g3": {"stage_order": "g4g1g3"},
    "dqfrag": {"dq_issue": "fragment_regs"},
    "qrot_edge": {"q_rotate": True, "edge_tiles": True},
    "gsplit2": {"g_split": 2},
}  # fmt: skip
# Cells whose counted kernel, or its full twin, the validator refuses
# (RESOURCE_REFUSED: SGPR spills on the production compiler); pinned to the
# planner by test_attention_bwd_mirror.py.
COUNTED_REFUSED = {
    ('gfx942', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'edge', 'default'): "counted: sgpr_spill",
    ('gfx942', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'qrot_edge', 'default'): "counted: sgpr_spill",
    ('gfx950', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'edge', 'default'): "counted: sgpr_spill",
    ('gfx950', 'bwd_kern_pack_sq1_g2_none_fp16_d64', 'qrot_edge', 'default'): "twin: sgpr_spill",
}  # fmt: skip


def _counted_params():
    if ARCH not in _MFMA:
        skip = pytest.mark.skip(reason=f"counted waits are CDNA-only; device {ARCH}")
        return [pytest.param(None, None, None, id="not-cdna", marks=skip)]
    out = []
    ids = [c.id for c in CASES] + [PACK_CASE.id]
    for cid in _shard(ids):
        case = _case(cid)
        for var, kw in COUNTED_VARIANTS.items():
            if "g_split" in kw and (case.h_q // case.h_k) % 2:
                continue
            for cfg in selected_configs(ARCH):
                refused = COUNTED_REFUSED.get((ARCH, cid, var, cfg))
                marks = (
                    [
                        pytest.mark.skip(
                            reason=f"the validator refuses this cell ({refused})"
                        )
                    ]
                    if refused
                    else []
                )
                out.append(
                    pytest.param(cid, var, cfg, id=f"{cid}-{var}-{cfg}", marks=marks)
                )
    return out


@pytest.mark.parametrize("case_id,variant,config", _counted_params())
def test_counted_waits_agree(case_id, variant, config):
    from kernels.common.attention_bwd import validate_attn_bwd_spec

    case = _case(case_id)
    # the issue point and the full twin's synchronisation set explicitly: a
    # configuration may count its waits itself (a custom one)
    kw = {"counted_issue": "early", **COUNTED_VARIANTS[variant]}
    full_kw = {k: v for k, v in kw.items() if k != "counted_issue"}
    full_kw["loop_sync"] = "full"
    if config == CUSTOM_CONFIG:
        # COUNTED_REFUSED records the shipped configurations: under the custom
        # configuration a full twin or a counted cell its validator refuses
        # skips with the reason
        skip_unless_planned(case, arch=ARCH, config=config, overrides=full_kw)
        skip_unless_planned(
            case, arch=ARCH, config=config, overrides={"loop_sync": "counted", **kw}
        )
    spec = plan_case(case, arch=ARCH, config=config, overrides=full_kw).specs["main"]
    spec_kw = {k: v for k, v in kw.items() if k != "g_split"}
    try:
        validate_attn_bwd_spec(
            dataclasses.replace(spec, loop_sync="counted", **spec_kw), ARCH
        )
    except ValueError as exc:
        if "RESOURCE_REFUSED" in str(exc) and config != CUSTOM_CONFIG:
            # in a shipped configuration the refusals are COUNTED_REFUSED only
            pytest.fail(f"refused but not in COUNTED_REFUSED: {exc}")
        pytest.skip(f"counted {kw} not legal on {spec.kernel_name('main')}: {exc}")
    base = _plain(case, config, full_kw)
    run = run_case(
        case, arch=ARCH, config=config, overrides={"loop_sync": "counted", **kw}
    )
    s = run.plan.specs["main"]
    assert s.loop_sync == "counted"
    assert s.counted_issue == kw["counted_issue"]
    _passes(run)
    if s.dkv_mode == "direct":
        assert np.array_equal(run.dk, base.dk), "dK differs under counted waits"
        assert np.array_equal(run.dv, base.dv), "dV differs under counted waits"
    main = run.plan.launches[1]
    if main.grid[0] == 1 and run.plan.runtime.get("g_split", 1) == 1:
        assert np.array_equal(run.dq, base.dq), "dQ differs under counted waits"
