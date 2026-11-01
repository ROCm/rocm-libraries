# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2a: unit plans of the E1 GEMM core (DESIGN-LB 4.3) for every dense projection of a large-batch step.

``plan(proj, tp, rows)`` returns the :class:`.e1.E1Knobs` build of a projection at a TP rank shape and a live row count
(1..128): the measured pick of WP-L2a's lookback-screened timing (``PLANS``, phase metric, both rocke nodes; see
``experiments/qwen38-true-mono/large-batch/WP-L2a.md``) for the row count's bucket, else the cost-model plan.

The cost model (``model_cost``) puts grid fill first (units against CTAs: a CU's second unit chain does not overlap the
first, P-L3 5.1), then per chunk the larger of the wave's MFMA work and its CU's share of bandwidth, the activation
staging, the split-K fold as a round-trip chain plus the partial bytes the folding CTA reads (all of them for
``fold=last``, 1 / S of them for ``fold=spread``), and the in-CTA (``kw``) reduction. It is a screening centre for the
tuner (``candidates``), never a verdict. Host only; nothing served imports it.
"""

from __future__ import annotations

from dataclasses import replace

from .e1 import E1Knobs

__all__ = ["SHAPES", "PLANS", "candidates", "model_cost", "plan", "regs_ok", "row_key"]

# per-rank (M, K); TP-invariant projections listed with tp 0. qsa_out has gdn_out's shape at both TPs.
SHAPES = {
    ("gdn_in", 1): (16480, 2560),
    ("gdn_in", 2): (8240, 2560),
    ("gdn_out", 1): (2560, 6144),
    ("gdn_out", 2): (2560, 3072),
    ("qsa_in", 1): (13952, 2560),
    ("qsa_in", 2): (7296, 2560),
    ("hc_down", 0): (324, 10240),
    ("hc_up", 0): (10240, 320),
    ("router", 0): (513, 2560),
    ("head", 1): (248320, 2560),
    ("head", 2): (124160, 2560),
}
ALIAS = {"qsa_out": "gdn_out"}
TP_FREE = ("hc_down", "hc_up", "router")
SOP = dict(wlay="tile", sop=True)

# Measured picks (WP-L2a): (projection, tp, rows) -> knob deviations from E1Knobs() (k and bucket from the shape).
# Filled from the plan-of-record runs by hackathon-prep/qwen38/truemono-lb/wp-l2a/make_plans.py; empty = model only.
PLANS: dict = {  # (projection, tp, rows) -> E1Knobs deviations (WP-L2a: screen picks + 16-round two-node lookback climb; confirmed only)
    ('gdn_in', 1, 5): {'waves': 4, 'mt': 2, 'depth': 4, 'xa': 2, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bpre': 1, 'bar': 'persist', 'xjunk': True},
    ('gdn_in', 1, 8): {'bar': 'persist', 'depth': 4, 'kw': 4, 'mt': 4, 'sb': False, 'sop': True, 'wcp': 'nt', 'wlay': 'tile', 'xa': 2, 'xjunk': True, 'xscale': 'glb', 'xslay': 'chunk'},
    ('gdn_in', 1, 16): {'mt': 4, 'depth': 4, 'xa': 2, 'xpad': 32, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'kw': 4, 'bar': 'persist', 'xjunk': True},
    ('gdn_in', 1, 32): {'waves': 4, 'mt': 2, 'depth': 4, 'xa': 2, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bpre': 2, 'bar': 'persist', 'xjunk': True},
    ('gdn_in', 1, 64): {'waves': 4, 'mt': 2, 'depth': 4, 'xa': 2, 'xpad': 32, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bar': 'persist'},
    ('gdn_in', 1, 128): {'waves': 4, 'mt': 4, 'tb': 4, 'depth': 4, 'xa': 2, 'xpad': 32, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bpre': 2, 'bar': 'persist'},
    ('gdn_in', 2, 5): {'bar': 'persist', 'depth': 4, 'kw': 4, 'map': 'xcd', 'mt': 4, 'sb': False, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('gdn_in', 2, 8): {'bar': 'persist', 'depth': 3, 'kw': 4, 'mt': 4, 'sb': False, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('gdn_in', 2, 16): {'bar': 'persist', 'kw': 4, 'mt': 4, 'sb': False, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('gdn_in', 2, 32): {'waves': 4, 'mt': 2, 'depth': 4, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bpre': 2, 'kw': 2, 'bar': 'persist', 'xjunk': True},
    ('gdn_in', 2, 64): {'waves': 4, 'mt': 2, 'tb': 2, 'depth': 4, 'xa': 2, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bar': 'persist', 'xjunk': True},
    ('gdn_in', 2, 128): {'waves': 4, 'mt': 2, 'tb': 4, 'depth': 4, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bar': 'persist', 'xjunk': True},
    ('gdn_out', 1, 5): {'waves': 4, 'depth': 3, 'xsrc': 'global', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'kw': 4, 'bar': 'persist'},
    ('gdn_out', 1, 8): {'waves': 4, 'depth': 3, 'xsrc': 'global', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'kw': 4, 'bar': 'persist'},
    ('gdn_out', 1, 16): {'waves': 4, 'xa': 2, 'xpad': 32, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'kw': 4, 'bar': 'persist', 'xjunk': True},
    ('gdn_out', 1, 32): {'waves': 4, 'mt': 2, 'tb': 1, 'depth': 4, 'xa': 2, 'xpad': 32, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bpre': 1, 'kw': 4, 'bar': 'persist', 'xjunk': True},
    ('gdn_out', 1, 64): {'waves': 4, 'splits': 6, 'depth': 4, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bar': 'persist'},
    ('gdn_out', 1, 128): {'mt': 2, 'tb': 2, 'depth': 4, 'xa': 2, 'xpad': 32, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bpre': 2, 'kw': 4, 'bar': 'persist', 'xjunk': True},
    ('gdn_out', 2, 5): {'waves': 4, 'xsrc': 'global', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bpre': 1, 'kw': 4, 'bar': 'persist'},
    ('gdn_out', 2, 8): {'bar': 'persist', 'kw': 4, 'sb': False, 'sop': True, 'waves': 4, 'wlay': 'tile', 'xsrc': 'global'},
    ('gdn_out', 2, 16): {'depth': 4, 'xpad': 32, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'xu': False, 'sb': False, 'kw': 8, 'bar': 'persist', 'xjunk': True},
    ('gdn_out', 2, 32): {'tb': 1, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'kw': 4, 'bar': 'persist', 'xjunk': True},
    ('gdn_out', 2, 64): {'waves': 4, 'tb': 2, 'depth': 3, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'xu': False, 'sb': False, 'bpre': 2, 'kw': 2, 'bar': 'persist', 'xjunk': True},
    ('gdn_out', 2, 128): {'waves': 4, 'mt': 2, 'tb': 2, 'depth': 6, 'xa': 2, 'xpad': 32, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'kw': 2, 'bar': 'persist', 'xjunk': True},
    ('hc_down', 0, 5): {'waves': 4, 'mt': 2, 'splits': 10, 'xsrc': 'global', 'wcp': 'nt', 'sop': True, 'wlay': 'tile', 'sb': False, 'bpre': 1, 'kw': 2, 'bar': 'persist'},
    ('hc_down', 0, 8): {'waves': 4, 'splits': 10, 'depth': 4, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bar': 'persist', 'xjunk': True},
    ('hc_down', 0, 16): {'waves': 4, 'splits': 10, 'depth': 4, 'xa': 2, 'xpad': 32, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'xu': False, 'sb': False, 'bar': 'persist', 'xjunk': True},
    ('hc_down', 0, 32): {'waves': 4, 'splits': 10, 'tb': 1, 'depth': 4, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bar': 'persist', 'xjunk': True},
    ('hc_down', 0, 64): {'waves': 4, 'mt': 2, 'splits': 5, 'tb': 1, 'xa': 2, 'xpad': 32, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bpre': 1, 'kw': 2, 'bar': 'persist', 'fg': 16},
    ('hc_down', 0, 128): {'waves': 4, 'splits': 5, 'tb': 2, 'depth': 3, 'xa': 2, 'xpad': 32, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'xu': False, 'sb': False, 'bpre': 2, 'bar': 'persist', 'xjunk': True},
    ('hc_up', 0, 5): {'bar': 'persist', 'bpre': 1, 'depth': 3, 'sop': True, 'waves': 4, 'wlay': 'tile', 'xsrc': 'global', 'xu': False},
    ('hc_up', 0, 8): {'bar': 'persist', 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global', 'xu': False},
    ('hc_up', 0, 16): {'bar': 'persist', 'bpre': 1, 'sop': True, 'waves': 4, 'wlay': 'tile', 'xsrc': 'global', 'xu': False},
    ('hc_up', 0, 32): {'waves': 4, 'depth': 6, 'xa': 2, 'xpad': 32, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'xu': False, 'sb': False, 'bar': 'persist', 'xjunk': True},
    ('hc_up', 0, 64): {'bar': 'persist', 'depth': 4, 'mt': 2, 'sop': True, 'tb': 2, 'waves': 4, 'wlay': 'tile', 'xsrc': 'global', 'xu': False},
    ('hc_up', 0, 128): {'tb': 4, 'depth': 4, 'xa': 2, 'xpad': 32, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'xu': False, 'sb': False, 'bpre': 2, 'bar': 'persist', 'xjunk': True},
    ('head', 1, 5): {'bar': 'persist', 'depth': 4, 'mt': 4, 'sop': True, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global', 'xu': False},
    ('head', 1, 8): {'bar': 'persist', 'depth': 4, 'mt': 4, 'sop': True, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('head', 1, 16): {'waves': 4, 'mt': 4, 'depth': 4, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bar': 'persist', 'xjunk': True},
    ('head', 1, 32): {'bar': 'persist', 'depth': 4, 'mt': 4, 'sop': True, 'wcp': 'nt', 'wlay': 'tile', 'xa': 2, 'xjunk': True, 'xpad': 32, 'xscale': 'glb', 'xslay': 'chunk'},
    ('head', 1, 64): {'bar': 'persist', 'depth': 4, 'map': 'xcd', 'mt': 4, 'sop': True, 'wcp': 'nt', 'wlay': 'tile', 'xjunk': True, 'xpad': 32, 'xscale': 'glb', 'xslay': 'chunk'},
    ('head', 1, 128): {'waves': 4, 'mt': 4, 'depth': 3, 'xpad': 32, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bpre': 2, 'bar': 'persist', 'xjunk': True},
    ('head', 2, 5): {'waves': 4, 'mt': 2, 'depth': 3, 'xsrc': 'global', 'wcp': 'nt', 'sop': True, 'wlay': 'tile', 'xu': False, 'bpre': 1, 'bar': 'persist'},
    ('head', 2, 8): {'bar': 'persist', 'bpre': 1, 'depth': 4, 'mt': 4, 'sb': False, 'sop': True, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('head', 2, 16): {'waves': 4, 'mt': 4, 'depth': 4, 'xa': 2, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'bar': 'persist', 'xjunk': True},
    ('head', 2, 32): {'bar': 'persist', 'depth': 3, 'mt': 4, 'sb': False, 'sop': True, 'wcp': 'nt', 'wlay': 'tile', 'xjunk': True, 'xscale': 'glb', 'xslay': 'chunk'},
    ('head', 2, 64): {'bar': 'persist', 'depth': 3, 'mt': 4, 'sop': True, 'wcp': 'nt', 'wlay': 'tile', 'xjunk': True, 'xpad': 32, 'xscale': 'glb', 'xslay': 'chunk'},
    ('head', 2, 128): {'waves': 4, 'mt': 4, 'xpad': 32, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'bpre': 2, 'bar': 'persist'},
    ('qsa_in', 1, 5): {'bar': 'persist', 'depth': 4, 'kw': 4, 'mt': 4, 'sb': False, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('qsa_in', 1, 8): {'bar': 'persist', 'depth': 3, 'kw': 4, 'mt': 4, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('qsa_in', 1, 16): {'bar': 'persist', 'depth': 3, 'kw': 4, 'mt': 4, 'sb': False, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('qsa_in', 1, 32): {'bar': 'persist', 'depth': 4, 'kw': 4, 'mt': 2, 'sb': False, 'sop': True, 'wlay': 'tile', 'xa': 2, 'xjunk': True, 'xscale': 'glb', 'xslay': 'chunk'},
    ('qsa_in', 1, 64): {'mt': 2, 'tb': 2, 'wcp': 'nt', 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'kw': 2, 'bar': 'persist'},
    ('qsa_in', 1, 128): {'waves': 4, 'mt': 2, 'tb': 4, 'depth': 4, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bar': 'persist'},
    ('qsa_in', 2, 5): {'bar': 'persist', 'kw': 4, 'mt': 2, 'sb': False, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('qsa_in', 2, 8): {'bar': 'persist', 'kw': 4, 'mt': 2, 'sb': False, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('qsa_in', 2, 16): {'bar': 'persist', 'kw': 4, 'mt': 2, 'sb': False, 'sop': True, 'waves': 4, 'wcp': 'nt', 'wlay': 'tile', 'xsrc': 'global'},
    ('qsa_in', 2, 32): {'waves': 4, 'mt': 2, 'tb': 1, 'depth': 6, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'kw': 2, 'bar': 'persist', 'xjunk': True},
    ('qsa_in', 2, 64): {'waves': 4, 'mt': 2, 'tb': 2, 'depth': 6, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'kw': 2, 'bar': 'persist', 'xjunk': True},
    ('qsa_in', 2, 128): {'waves': 4, 'mt': 2, 'tb': 2, 'depth': 4, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'map': 'xcd', 'sb': False, 'bar': 'persist', 'xjunk': True},
    ('router', 0, 5): {'bar': 'persist', 'kw': 4, 'sb': False, 'sop': True, 'waves': 4, 'wlay': 'tile', 'xsrc': 'global'},
    ('router', 0, 8): {'bar': 'persist', 'kw': 4, 'sb': False, 'sop': True, 'waves': 4, 'wlay': 'tile', 'xsrc': 'global'},
    ('router', 0, 16): {'bar': 'persist', 'depth': 4, 'kw': 4, 'sb': False, 'sop': True, 'waves': 4, 'wlay': 'tile', 'xjunk': True, 'xpad': 32, 'xscale': 'glb', 'xslay': 'chunk', 'xu': False},
    ('router', 0, 32): {'bar': 'persist', 'depth': 4, 'kw': 4, 'sb': False, 'sop': True, 'tb': 1, 'waves': 4, 'wlay': 'tile', 'xjunk': True, 'xscale': 'glb', 'xslay': 'chunk', 'xu': False},
    ('router', 0, 64): {'waves': 4, 'tb': 1, 'xa': 2, 'xslay': 'chunk', 'xscale': 'glb', 'sop': True, 'wlay': 'tile', 'sb': False, 'kw': 4, 'bar': 'persist', 'xjunk': True},
    ('router', 0, 128): {'bar': 'persist', 'depth': 4, 'kw': 4, 'sop': True, 'tb': 2, 'waves': 4, 'wlay': 'tile', 'xa': 2, 'xjunk': True, 'xscale': 'glb', 'xslay': 'chunk', 'xu': False},
}


def row_key(rows: int) -> int:
    """The measured row count whose pick serves ``rows`` live tokens (same bucket; 1-5 -> 5, 6-8 -> 8)."""
    if not 1 <= rows <= 128:
        raise ValueError("rows 1..128")
    for r in (5, 8, 16, 32, 64, 128):
        if rows <= r:
            return r
    return 128


def regs_ok(kn: E1Knobs) -> bool:
    """Rough VGPR budget (8 waves: 256 a lane; 4 waves: 512) for the screen; the compile gate decides."""
    acc = kn.mt * kn.tbt * 4 * (2 if kn.mfma == 32 else 1)
    wr = kn.depth * kn.mt * 5
    bo = 9 * max(1, kn.bpre or 2)
    if kn.xsrc == "global":
        bo += kn.depth * kn.tbt * (4 if kn.fp4x else 8)
    budget = 230 if kn.waves == 8 else 460
    return acc + wr + bo + 40 <= budget


def model_cost(M: int, K: int, rows: int, kn: E1Knobs, grid: int = 256) -> float:
    """Shader clocks per layer of the slowest CTA (screening centre)."""
    R = kn.rows_per_unit
    RG = -(-M // R)
    T = -(-rows // 16)
    ntb = -(-T // kn.tbt)
    U = RG * ntb * kn.splits
    rounds = -(-U // grid)
    busy = min(U, grid)
    bw = min(48.0, 2500.0 / max(busy, 1))
    tiles = min(kn.tbt, T)
    comp = kn.mt * (kn.rt // 16) * tiles * (100.0 if kn.fp4x else 180.0)
    chunk_bytes = kn.wr * kn.mt * kn.rt * 128 * 0.53125 * kn.kw
    xb = kn.kw * tiles * 16 * 128 * (0.5 if kn.fp4x else 1.0)
    chunk = max(comp, (chunk_bytes + xb / 4) / bw) + (150.0 if kn.xsrc == "global" else 300.0)
    fold_bytes = 0.0 if kn.splits == 1 else kn.splits * tiles * 16 * R * 4.0
    if kn.fold == "spread" and U <= grid:
        fold_bytes /= kn.splits
    unit = kn.kc * chunk + 1500.0 + (0.0 if kn.splits == 1 else 2500.0 + fold_bytes / 48.0) + (600.0 if kn.kw > 1 else 0.0)
    return rounds * unit


def _ok(kn: E1Knobs) -> bool:
    try:
        kn.check()
    except ValueError:
        return False
    return regs_ok(kn)


def candidates(proj: str, tp: int, rows: int, act: str = "fp8", top: int = 8) -> list:
    """Structural plans of a cell for the screen: the model's best ``top`` plus the best 2 (distinct splits) of every
    (kw, xsrc) and the best one-split plan (diversity). op_sel scales, tile weights, the persist barrier, branch-free
    staging; ``fold`` last and spread for K splits."""
    proj = ALIAS.get(proj, proj)
    M, K = SHAPES[(proj, 0 if proj in TP_FREE else tp)]
    bucket = max(16, -(-rows // 16) * 16)
    T = bucket // 16
    nq = -(-K // 128)
    cands = []
    for waves in (8, 4):
        for kw in (1, 2, 4, 8):
            if kw > waves:
                continue
            for mt in (1, 2, 4):
                for S in [d for d in range(1, nq + 1) if nq % d == 0]:
                    if nq % (S * kw) or nq // (S * kw) > 80:
                        continue
                    RG = -(-M // (16 * (waves // kw) * mt))
                    if RG * S > 4 * 256:
                        continue
                    for tb in sorted({T, max(1, T // 2), max(1, T // 4)}):
                        for xsrc in ("lds", "global"):
                            if xsrc == "global" and (waves // kw > 1 and kw > 1):
                                continue
                            kw_ = dict(k=K, bucket=bucket, waves=waves, kw=kw, mt=mt, splits=S, tb=0 if tb == T else tb,
                                       depth=4, act=act, bar="persist", xjunk=xsrc == "lds", **SOP)
                            if xsrc == "lds":
                                kw_.update(xslay="chunk", xscale="glb")
                            else:
                                kw_.update(xsrc="global")
                            for fold in (("last", "spread") if S > 1 else ("last",)):
                                kn = E1Knobs(**kw_, fold=fold)
                                if _ok(kn):
                                    cands.append((model_cost(M, K, rows, kn), kn))
    cands.sort(key=lambda x: x[0])
    out, seen = [], set()

    def add(kn):
        from .e1 import kernel_name

        n = kernel_name(kn)
        if n not in seen:
            seen.add(n)
            out.append(kn)

    for _, kn in cands[:top]:
        add(kn)
    for kwv in (1, 2, 4, 8):
        for xs in ("lds", "global"):
            seen_s = set()
            for _, kn in cands:
                if kn.kw == kwv and kn.xsrc == xs and kn.splits not in seen_s:
                    seen_s.add(kn.splits)
                    add(kn)
                    if len(seen_s) == 2:
                        break
    one = [kn for _, kn in cands if kn.splits == 1]
    if one:
        add(one[0])
    return out


def plan(proj: str, tp: int, rows: int) -> E1Knobs:
    """The build for ``rows`` live tokens: the measured pick of the row count's bucket, else the model's best plan."""
    key_p = ALIAS.get(proj, proj)
    tpk = 0 if key_p in TP_FREE else tp
    M, K = SHAPES[(key_p, tpk)]
    r = row_key(rows)
    bucket = max(16, -(-rows // 16) * 16)
    dev = PLANS.get((key_p, tpk, r))
    if dev is not None:
        kn = E1Knobs(k=K, bucket=bucket, **dev)
        if kn.tb > kn.tiles:
            kn = replace(kn, tb=0)
        kn.check()
        return kn
    return candidates(key_p, tpk, rows, top=1)[0]
