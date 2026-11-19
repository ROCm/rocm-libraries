# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Opt-in plan selector for the fused 3-D Winograd kernels of ``conv_winograd3d`` (pure Python).

Given a 3-D :class:`ConvProblem`, a direction (``"fwd"`` or ``"dgrad"``) and a call mode,
:func:`select_winograd3d` returns a :class:`Winograd3dPlan`: either a yx Winograd plan
(F(2x2,3x3) on y,x + direct 3-tap z, the ``WinogradYxSpec`` fields to build it with, the
weight-transform version and the prescale) or ``kind == "direct"`` -- use the implicit-GEMM
kernels -- with the reason. Nothing here touches a GPU; the module is not wired into any
default path.

Modes (what a call costs besides the main kernel):

* ``"dropin"`` -- NDHWC in / NDHWC out, no model change: one aux launch (LDS-transposed
  X / dY relayout to c8 [+ halo 2 for dgrad] next to the weight transform) + main kernel.
* ``"dropin"`` on the padded adapter forms -- pad (1,1,1) (input = output size) and pad
  (0,1,1) (a D-split run's D+2 input): the same aux launch adds the zero halo (FWD: the
  padding; BWD-data: 2 - padding around dY, whose dX is then the padded conv's full
  gradient) -- the main kernel is the halo form's. There the direct kernels lose their
  halo-form configurations (padding admission); the tuned tables ``TUNED_PAD1`` /
  ``TUNED_PAD011`` record per cell whether drop-in Winograd or the direct path is selected
  on gfx942. For FWD the direct path can instead zero-pad X
  (``conv_winograd3d.build_pad_copy``) and run the halo-form kernel; with
  ``direct_pad_copy=True`` (CLI ``--pad-copy``) the tables compare against that path instead.
* ``"chained"`` -- the input already arrives in the c8 layout (the previous conv's ``yc8h``
  epilogue wrote it): weight transform + main kernel. Requires the activations between the
  convs to stay in c8 (BN / activation kernels must read and write it). ``chained`` names
  the plan's input; its output is :attr:`Winograd3dPlan.out_layout` (``"c8"`` with the
  ``yc8h`` epilogue, ``"ndhwc"`` for a split-K plan, whose f32 slices ``build_ks_reduce``
  sums into NDHWC) and :attr:`Winograd3dPlan.out_map` (whether it writes the next conv's
  prescale map, ``bps_out``): a plan whose output is NDHWC ends the chain. With the
  prescale a chained plan reads the producer's map (``bps=2``), indexed by the producer's
  geometry: pass the producer's plan (``producer=``) so the map indexing follows it; without
  one, the producer is assumed to tile like this plan.

Numerics: the transformed operands are f16 (11 significant bits; the record's bf16 operands
carry 8): inside the f16 range the yx path is as accurate as the direct bf16 conv (mean
relative error 1.50e-3 vs 1.41e-3, the bf16 output floor). Outside it (gradients below
~2^-16 rms, |x| above 16k) the per-block power-of-two prescale (``bps``) keeps it so --
``prescale="auto"`` turns it on unless the caller vouches for the range (``range_safe``).
``prescale="none"`` without ``range_safe`` refuses Winograd (silent underflow / NaN otherwise).

``xpf=True`` (opt-in) plans the unrolled-prefetch main loop (``WinogradYxSpec.xpf``: no
loop-carried register copy, so the input prefetch is not waited for every iteration; bitwise
the same output) and, for the padded drop-in problems, decides on its own decision tables,
tuned against the best direct route of each cell (``TUNED_PAD1_XPF`` / ``TUNED_PAD011_XPF``).

Determinism: every yx plan is deterministic (no atomics; split-K partials go to f32 slices
that ``build_ks_reduce`` sums in a fixed order).

Selection: per-cell decision tables only (``TUNED*``: same-session comparisons on gfx942
against the deterministic direct kernels, reduced to the selected candidate and the direct
baselines it leads by at least ``TUNED_MARGIN``; near-ties go to the direct kernel: better accuracy,
no range caveat). A problem, mode or prescale without a table entry gets the direct plan
(``source="untuned"``): Winograd is never chosen without a comparison on the target.

F(4x2,3x3) (``my=4``) and F(4x4,3x3) are never selected: F(4x2) is correct (mean error
2.0e-3) but not selected on gfx942 (6 waves in one group leave the SIMDs unbalanced at
247 VGPRs; 12 waves spill), and F(4x4) does not fit -- 36 points x (z_tile + 2) staged planes
exceed 64 KiB of LDS without double buffering, and its f16 error exceeds the direct conv's.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Tuple

from kernels.common._conv_implicit_gemm_common import ConvProblem

__all__ = [
    "Winograd3dPlan",
    "select_winograd3d",
    "winograd3d_candidates",
    "winograd3d_ineligible_reason",
    "SPEC_B",
    "SPEC_Z",
    "TUNED",
    "TUNED_MARGIN",
    "TUNED_PAD1",
    "TUNED_PAD011",
    "TUNED_PAD011_XPF",
    "TUNED_PAD1_XPF",
    "UNET3D_SHAPES",
    "unet3d_plans",
]

# Main-kernel configurations (benchmark_winograd3d --spec bodies, "variant":"yx" implied)
SPEC_B = dict(
    stage="lds16",
    op_dtype="f16",
    fill_il=True,
    waves=8,
    nsplit=True,
    z_tile=2,
    n_tile=64,
    nfast=True,
    pkrtz=True,
    epi_vec=True,
    yf16=True,
)
SPEC_Z = dict(SPEC_B, nsplit=False, z_tile=4, n_tile=32, xcd=6)

# Decision tables (gfx942, bf16, deterministic): same-session A/B of each Winograd candidate
# against the deterministic direct kernels of the problem, kept as decisions only.
# (direction, Di, Hi, Wi, C, K) -> {"baselines": the direct alternatives compared for the cell,
# "<mode>[_nops]": (selected candidate name, the baselines it leads by TUNED_MARGIN)} with mode
# dropin / chained, "_nops" = without prescale (range-safe data). Baselines: "direct" = the
# selected direct kernel on the problem as given; "pad_copy" / "pad_copy_torch" (padded FWD only)
# = zero-pad copy of X (conv_winograd3d.build_pad_copy / a torch strided copy) + the
# halo-form direct kernel. Keys: the forward problem's (Di, Hi, Wi, C, K) for both
# directions (dgrad = the dgrad of that forward conv).
# TUNED: halo (valid) form; TUNED_PAD1: pad (1,1,1), the problem's own dims (input = output
# size); TUNED_PAD011: pad (0,1,1) with a D+2 input (the padded form of a D-split run).
# *_XPF: the unrolled-prefetch main loop (xpf) against the best direct route of the cell
# (direct kernel or pad copy + halo-form kernel).
TUNED_MARGIN = 0.02

TUNED: Dict[Tuple[str, int, int, int, int, int], dict] = {
    ("dgrad", 258, 130, 130, 64, 64): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ()),
        "dropin": ("B+lin+xcd6", ()),
        "chained_nops": ("B+lin+xcd6", ()),
        "chained": ("B+lin+xcd6", ()),
    },
    ("dgrad", 258, 130, 130, 128, 64): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ()),
        "dropin": ("B+lin+xcd6", ()),
        "chained_nops": ("B+lin+xcd6", ()),
        "chained": ("B+lin+xcd6", ()),
    },
    ("dgrad", 130, 66, 66, 256, 128): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ()),
        "dropin": ("B+lin+xcd6", ()),
        "chained_nops": ("B+lin+xcd6", ()),
        "chained": ("B+lin+xcd6", ()),
    },
    ("dgrad", 130, 66, 66, 128, 128): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ()),
        "dropin": ("B+lin+xcd6", ()),
        "chained_nops": ("B+lin+xcd6", ("direct",)),
        "chained": ("B+lin+xcd6", ("direct",)),
    },
    ("dgrad", 66, 34, 34, 512, 256): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ()),
        "dropin": ("B+lin+xcd6", ()),
        "chained_nops": ("B+lin+xcd6", ()),
        "chained": ("B+lin+xcd6", ()),
    },
    ("dgrad", 66, 34, 34, 256, 256): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ("direct",)),
        "dropin": ("B+lin+xcd6", ("direct",)),
        "chained_nops": ("B+lin+xcd6", ("direct",)),
        "chained": ("B+lin+xcd6", ("direct",)),
    },
    ("dgrad", 34, 18, 18, 512, 512): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ()),
        "dropin": ("B+lin+xcd6", ()),
        "chained_nops": ("B+lin+xcd6", ()),
        "chained": ("B+lin+xcd6", ()),
    },
    ("dgrad", 34, 18, 18, 1024, 512): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ()),
        "dropin": ("B+lin+xcd6", ()),
        "chained_nops": ("B+lin+xcd6", ()),
        "chained": ("B+lin+xcd6", ()),
    },
    ("dgrad", 130, 66, 66, 64, 128): {
        "baselines": ("direct",),
        "dropin_nops": ("B+lin+xcd6", ()),
        "dropin": ("B+lin+xcd6", ()),
        "chained_nops": ("B+lin+xcd6", ("direct",)),
        "chained": ("B+lin+xcd6", ("direct",)),
    },
    ("fwd", 258, 130, 130, 64, 64): {
        "baselines": ("direct",),
        "dropin_nops": ("Z", ()),
        "dropin": ("Z", ()),
        "chained_nops": ("Z", ()),
        "chained": ("Z", ()),
    },
    ("fwd", 258, 130, 130, 128, 64): {
        "baselines": ("direct",),
        "dropin_nops": ("Z", ()),
        "dropin": ("Z", ()),
        "chained_nops": ("Z", ("direct",)),
        "chained": ("Z", ("direct",)),
    },
    ("fwd", 130, 66, 66, 256, 128): {
        "baselines": ("direct",),
        "dropin_nops": ("Z", ()),
        "dropin": ("Z", ()),
        "chained_nops": ("Z", ("direct",)),
        "chained": ("Z", ("direct",)),
    },
    ("fwd", 130, 66, 66, 128, 128): {
        "baselines": ("direct",),
        "dropin_nops": ("Z", ()),
        "dropin": ("Z", ()),
        "chained_nops": ("Z", ("direct",)),
        "chained": ("Z", ("direct",)),
    },
    ("fwd", 66, 34, 34, 512, 256): {
        "baselines": ("direct",),
        "dropin_nops": ("Z", ("direct",)),
        "dropin": ("Z", ("direct",)),
        "chained_nops": ("Z", ("direct",)),
        "chained": ("Z", ("direct",)),
    },
    ("fwd", 66, 34, 34, 256, 256): {
        "baselines": ("direct",),
        "dropin_nops": ("Z", ("direct",)),
        "dropin": ("Z", ("direct",)),
        "chained_nops": ("Z", ("direct",)),
        "chained": ("Z", ()),
    },
    ("fwd", 34, 18, 18, 512, 512): {
        "baselines": ("direct",),
        "dropin_nops": ("Z+ks4", ()),
        "chained_nops": ("Z+ks4", ()),
        "dropin": ("Z+ks4", ()),
        "chained": ("Z+ks4", ()),
    },
    ("fwd", 34, 18, 18, 1024, 512): {
        "baselines": ("direct",),
        "dropin_nops": ("Z+ks4", ()),
        "chained_nops": ("Z+ks4", ("direct",)),
        "dropin": ("Z+ks4", ()),
        "chained": ("Z+ks4", ("direct",)),
    },
    ("fwd", 130, 66, 66, 64, 128): {
        "baselines": ("direct",),
        "dropin_nops": ("B+xcd6", ()),
        "dropin": ("B+xcd6", ()),
        "chained_nops": ("B+xcd6", ()),
        "chained": ("B+xcd6", ()),
    },
}
TUNED_PAD1: Dict[Tuple[str, int, int, int, int, int], dict] = {
    ("dgrad", 256, 128, 128, 64, 64): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ()),
    },
    ("dgrad", 256, 128, 128, 128, 64): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ()),
    },
    ("dgrad", 128, 64, 64, 256, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 128, 64, 64, 128, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 64, 32, 32, 512, 256): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 64, 32, 32, 256, 256): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 32, 16, 16, 512, 512): {
        "baselines": ("direct",),
        "dropin": ("Z+ks4", ()),
    },
    ("dgrad", 32, 16, 16, 1024, 512): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 128, 64, 64, 64, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("fwd", 256, 128, 128, 64, 64): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch")),
    },
    ("fwd", 256, 128, 128, 128, 64): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch")),
    },
    ("fwd", 128, 64, 64, 256, 128): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch", "pad_copy")),
    },
    ("fwd", 128, 64, 64, 128, 128): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch", "pad_copy")),
    },
    ("fwd", 64, 32, 32, 512, 256): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch", "pad_copy")),
    },
    ("fwd", 64, 32, 32, 256, 256): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch", "pad_copy")),
    },
    ("fwd", 32, 16, 16, 512, 512): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z+ks4", ("direct",)),
    },
    ("fwd", 32, 16, 16, 1024, 512): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z+ks4", ("direct",)),
    },
    ("fwd", 128, 64, 64, 64, 128): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch")),
    },
}
TUNED_PAD011: Dict[Tuple[str, int, int, int, int, int], dict] = {
    ("dgrad", 258, 128, 128, 64, 64): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ()),
    },
    ("dgrad", 258, 128, 128, 128, 64): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ()),
    },
    ("dgrad", 130, 64, 64, 256, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 130, 64, 64, 128, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 66, 32, 32, 512, 256): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 66, 32, 32, 256, 256): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 34, 16, 16, 512, 512): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6+ks4", ()),
    },
    ("dgrad", 34, 16, 16, 1024, 512): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 130, 64, 64, 64, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("fwd", 258, 128, 128, 64, 64): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch")),
    },
    ("fwd", 258, 128, 128, 128, 64): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch")),
    },
    ("fwd", 130, 64, 64, 256, 128): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch", "pad_copy")),
    },
    ("fwd", 130, 64, 64, 128, 128): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch", "pad_copy")),
    },
    ("fwd", 66, 32, 32, 512, 256): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch", "pad_copy")),
    },
    ("fwd", 66, 32, 32, 256, 256): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch", "pad_copy")),
    },
    ("fwd", 34, 16, 16, 512, 512): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z+ks4", ("direct",)),
    },
    ("fwd", 34, 16, 16, 1024, 512): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z+ks4", ("direct", "pad_copy_torch")),
    },
    ("fwd", 130, 64, 64, 64, 128): {
        "baselines": ("direct", "pad_copy_torch", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy_torch")),
    },
}

# xpf variants of the padded drop-in calls (see above).
TUNED_PAD1_XPF: Dict[Tuple[str, int, int, int, int, int], dict] = {
    ("dgrad", 256, 128, 128, 64, 64): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ()),
    },
    ("dgrad", 256, 128, 128, 128, 64): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ()),
    },
    ("dgrad", 128, 64, 64, 256, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 128, 64, 64, 128, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 64, 32, 32, 512, 256): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 64, 32, 32, 256, 256): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 32, 16, 16, 512, 512): {
        "baselines": ("direct",),
        "dropin": ("Z+ks4", ("direct",)),
    },
    ("dgrad", 32, 16, 16, 1024, 512): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 128, 64, 64, 64, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("fwd", 256, 128, 128, 64, 64): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("B+xcd6", ()),
    },
    ("fwd", 256, 128, 128, 128, 64): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("B+xcd6", ("direct", "pad_copy")),
    },
    ("fwd", 128, 64, 64, 256, 128): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy")),
    },
    ("fwd", 128, 64, 64, 128, 128): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy")),
    },
    ("fwd", 64, 32, 32, 512, 256): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy")),
    },
    ("fwd", 64, 32, 32, 256, 256): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy")),
    },
    ("fwd", 32, 16, 16, 512, 512): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z+ks4", ()),
    },
    ("fwd", 32, 16, 16, 1024, 512): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z+ks4", ("direct", "pad_copy")),
    },
    ("fwd", 128, 64, 64, 64, 128): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("B+xcd6", ("direct", "pad_copy")),
    },
}
TUNED_PAD011_XPF: Dict[Tuple[str, int, int, int, int, int], dict] = {
    ("dgrad", 34, 16, 16, 512, 512): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6+ks4", ()),
    },
    ("dgrad", 258, 128, 128, 64, 64): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ()),
    },
    ("dgrad", 258, 128, 128, 128, 64): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ()),
    },
    ("dgrad", 130, 64, 64, 256, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 130, 64, 64, 128, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 66, 32, 32, 512, 256): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 66, 32, 32, 256, 256): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 34, 16, 16, 1024, 512): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("dgrad", 130, 64, 64, 64, 128): {
        "baselines": ("direct",),
        "dropin": ("B+xcd6", ("direct",)),
    },
    ("fwd", 258, 128, 128, 64, 64): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ()),
    },
    ("fwd", 258, 128, 128, 128, 64): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("B+xcd6", ("direct", "pad_copy")),
    },
    ("fwd", 130, 64, 64, 256, 128): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy")),
    },
    ("fwd", 130, 64, 64, 128, 128): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy")),
    },
    ("fwd", 66, 32, 32, 512, 256): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy")),
    },
    ("fwd", 66, 32, 32, 256, 256): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z", ("direct", "pad_copy")),
    },
    ("fwd", 34, 16, 16, 512, 512): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z+ks4", ()),
    },
    ("fwd", 34, 16, 16, 1024, 512): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("Z+ks4", ("direct", "pad_copy")),
    },
    ("fwd", 130, 64, 64, 64, 128): {
        "baselines": ("direct", "pad_copy"),
        "dropin": ("B+xcd6", ("direct", "pad_copy")),
    },
}

_ERR_CLASS = {"yx_f2": 1.50e-3, "yx_f42": 2.0e-3}  # mean rel error, training-like data


@dataclass(frozen=True)
class Winograd3dPlan:
    kind: str  # "yx" | "direct"
    spec: Tuple[Tuple[str, object], ...] = ()
    wtv: int = 0
    mode: str = "dropin"
    direction: str = "fwd"
    prescale: str = "none"  # "bps" | "none"
    source: str = ""  # "tuned" | "untuned" | "route table"
    err_class: float = 0.0
    reason: str = ""
    alternatives: Tuple[Tuple[str, str], ...] = field(default=(), compare=False)

    @property
    def spec_dict(self) -> dict:
        return dict(self.spec)

    @property
    def out_layout(self) -> str:
        """Layout of the plan's output: "c8" (the ``yc8h`` epilogue, the next chained
        conv's input) or "ndhwc" (drop-in plans, split-K plans, direct)."""
        return "c8" if self.kind == "yx" and self.spec_dict.get("yc8h") else "ndhwc"

    @property
    def out_map(self) -> bool:
        """Whether the plan writes the next chained conv's prescale map (``bps_out``)."""
        return self.kind == "yx" and bool(self.spec_dict.get("bps_out"))

    @property
    def label(self) -> str:
        if self.kind == "direct":
            return "direct"
        s = self.spec_dict
        cfg = "B" if s.get("nsplit") else "Z"
        extra = [k for k in ("lin", "ksplit", "bps", "yc8h", "xpf") if s.get(k)]
        return f"yx:F(2x2,3x3)+z:{cfg}" + (
            "+"
            + "+".join(
                f"{k}{s[k] if not isinstance(s[k], bool) else ''}" for k in extra
            )
            if extra
            else ""
        )


_PADS = {(0, 0, 0): 0, (1, 1, 1): 1, (0, 1, 1): 2}


def _pad(p: ConvProblem) -> int:
    """0: halo / valid form; 1: adapter form, pad (1,1,1); 2: pad (0,1,1) (the adapter form of
    a D-split run, D+2 input); -1 for any other padding."""
    return _PADS.get((p.pD, p.pH, p.pW), -1)


def _relayout_halos(p: ConvProblem, direction: str) -> Tuple[int, int]:
    """(z, y/x) zero halo the drop-in relayout adds: FWD the conv's padding, BWD-data
    2 - padding."""
    if direction == "fwd":
        return p.pD, p.pH
    return 2 - p.pD, 2 - p.pH


def _relayout_halo(p: ConvProblem, direction: str) -> int:
    """y / x halo of the drop-in relayout (see _relayout_halos)."""
    return _relayout_halos(p, direction)[1]


def _winograd_problem(p: ConvProblem, direction: str) -> Tuple[int, int, int, int, int]:
    """(Di, Hi, Wi, C, K) of the forward conv the yx kernel runs."""
    hz, h = _relayout_halos(p, direction)
    if direction == "fwd":
        return p.Di + 2 * hz, p.Hi + 2 * h, p.Wi + 2 * h, p.C, p.K
    # dgrad: dX = conv(pad(dY, 2 - pad), flipped W), C <-> K
    return p.Do + 2 * hz, p.Ho + 2 * h, p.Wo + 2 * h, p.K, p.C


def winograd3d_ineligible_reason(
    p: ConvProblem, direction: str, dtype: str = "bf16"
) -> Optional[str]:
    """Why no yx plan applies at all (None if some may)."""
    if direction not in ("fwd", "dgrad"):
        return f"direction {direction!r} (fwd / dgrad only; wgrad has no Winograd plan)"
    if dtype != "bf16":
        return f"dtype {dtype} (bf16 I/O only)"
    if not p.is_3d:
        return "2-D conv (see conv_winograd_select)"
    if p.N != 1:
        return f"N={p.N} (the kernels take N=1)"
    if p.groups != 1:
        return "grouped conv"
    if (p.Z, p.Y, p.X) != (3, 3, 3):
        return f"filter {p.Z}x{p.Y}x{p.X} (3x3x3 only)"
    if (p.sD, p.sH, p.sW) != (1, 1, 1) or (p.dD, p.dH, p.dW) != (1, 1, 1):
        return "stride / dilation != 1"
    if _pad(p) < 0:
        return (
            f"padding {(p.pD, p.pH, p.pW)} (the kernels run the valid / halo form, or "
            "pad (1,1,1) / (0,1,1) through the drop-in relayout)"
        )
    if p.C % 8 or p.K % 32 or (direction == "dgrad" and (p.K % 8 or p.C % 32)):
        return "channels: C % 8 and K % 32 (of the conv the kernel runs)"
    # 32-bit buffers: the caller's input / output, the relayouted input and the output of
    # the conv the kernel runs (cw.is_valid_yx_spec also checks each spec's own buffers)
    from kernels.common import conv_winograd3d as cw

    Di, Hi, Wi, C, K = _winograd_problem(p, direction)
    why = cw._buffer_error(
        X=p.Di * p.Hi * p.Wi * p.C * 2,
        Y=p.Do * p.Ho * p.Wo * p.K * 2,
        X8=Di * Hi * Wi * C * 2,
        Yk=(Di - 2) * (Hi - 2) * (Wi - 2) * K * 2,
    )
    if why:
        return why
    return None


def _specs(direction: str, mode: str, pad: int = 0):
    """Candidate (name, spec extras, wtv) per direction."""
    if direction == "dgrad" and pad == 0:
        return [
            ("B+lin+xcd6", dict(SPEC_B, lin=True, xcd=6), 3),
            ("Z+lin", dict(SPEC_Z, lin=True), 3),
        ]
    return [
        ("Z", dict(SPEC_Z), 4),
        ("B", dict(SPEC_B), 4),
        ("B+xcd6", dict(SPEC_B, xcd=6), 3),
        ("Z+ks4", dict(SPEC_Z, ksplit=4), 4),
        ("B+ks4", dict(SPEC_B, ksplit=4), 4),
        ("B+xcd6+ks4", dict(SPEC_B, xcd=6, ksplit=4), 3),
    ]


def _producer_geometry(producer, C: int) -> dict:
    """bps=2 map geometry of a chained producer plan (its z_tile, n blocks, my, waves);
    only the fields that differ from the consumer's own are set by the caller."""
    from kernels.common import conv_winograd3d as cw

    s = producer.spec_dict
    dflt = {f.name: f.default for f in fields(cw.WinogradYxSpec)}
    return {
        "z_tile": s.get("z_tile", dflt["z_tile"]),
        "nb": C // s.get("n_tile", dflt["n_tile"]),
        "my": s.get("my", dflt["my"]),
        "waves": s.get("waves", dflt["waves"]),
    }


def _producer_error(producer, direction: str, prescale: str) -> Optional[str]:
    """Why ``producer`` cannot feed a chained plan (None if it can)."""
    if producer.kind != "yx" or producer.out_layout != "c8":
        return f"producer writes {producer.out_layout}, not the c8 layout chained reads"
    if producer.direction != direction:
        return f"producer is a {producer.direction} conv, this one {direction}"
    if prescale == "bps" and not producer.out_map:
        return "producer writes no prescale map (bps_out) for bps=2"
    return None


def winograd3d_candidates(
    p: ConvProblem,
    direction: str,
    mode: str = "dropin",
    prescale: str = "bps",
    xpf: bool = False,
    producer: Optional[Winograd3dPlan] = None,
):
    """Valid yx specs (WinogradYxSpec) for the problem, with their names and wtv
    (``xpf``: the unrolled-prefetch main loop where valid; ``producer``: the chained
    producer's plan, whose geometry indexes the bps=2 map)."""
    from kernels.common import conv_winograd3d as cw

    Di, Hi, Wi, C, K = _winograd_problem(p, direction)
    out = []
    for name, kw, wtv in _specs(direction, mode, _pad(p)):
        kw = dict(kw)
        if xpf:
            kw["xpf"] = 1
        if prescale == "bps":
            if mode == "chained":
                kw.update(bps=2, bps_halo=1)
                if producer is not None:
                    g = _producer_geometry(producer, C)
                    own = _producer_geometry(
                        Winograd3dPlan("yx", tuple(sorted(kw.items()))), C
                    )
                    for fld, key in (
                        ("bps_pz", "z_tile"),
                        ("bps_pn", "nb"),
                        ("bps_pmy", "my"),
                        ("bps_pw", "waves"),
                    ):
                        if g[key] != own[key]:
                            kw[fld] = g[key]
            else:
                hz, h = _relayout_halos(p, direction)
                kw.update(bps=1, bps_halo=h, **({"bps_halo_z": hz} if hz != h else {}))
        if mode == "chained" and not kw.get("ksplit"):
            kw["yc8h"] = True
            if direction == "dgrad":
                kw["yc8h_crop"] = True
            if prescale == "bps":
                kw["bps_out"] = True
        spec = cw.WinogradYxSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, **kw)
        ok, why = cw.is_valid_yx_spec(spec)
        if not ok and xpf:
            kw.pop("xpf")
            spec = cw.WinogradYxSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, **kw)
            ok, why = cw.is_valid_yx_spec(spec)
        if ok:
            out.append((name, spec, wtv))
    return out


def select_winograd3d(
    problem: ConvProblem,
    direction: str = "fwd",
    *,
    mode: str = "dropin",
    dtype: str = "bf16",
    prescale: str = "auto",
    range_safe: bool = False,
    use_measured: bool = True,
    direct_pad_copy: bool = False,
    xpf: bool = False,
    producer: Optional[Winograd3dPlan] = None,
) -> Winograd3dPlan:
    """The yx Winograd plan for ``problem`` or ``kind="direct"`` with the reason.

    Problems with a decision table entry (``use_measured``) take its decision (taken at
    ``TUNED_MARGIN``); every other problem, mode or prescale gets the direct plan
    (``source="untuned"``).

    ``xpf`` (opt-in): plan the unrolled-prefetch main loop (``WinogradYxSpec.xpf``) and, for
    padded drop-in problems, decide on its tables (``TUNED_PAD1_XPF`` / ``TUNED_PAD011_XPF``,
    tuned against the best direct route incl. the pad-copy path) when present.

    ``direct_pad_copy``: for a padded (adapter-form) FWD problem, Winograd must also beat
    "zero-pad copy + halo-form direct kernel" (``pad_copy_torch``: torch strided copy;
    ``pad_copy``: conv_winograd3d.build_pad_copy -- the realistic direct path for
    adapter-form FWD on gfx942).

    ``producer`` (chained mode): the plan of the conv whose ``yc8h`` output this one reads;
    the bps=2 map is indexed by its geometry, and a producer that writes no c8 output (or,
    with the prescale, no map) gets the direct plan. Without it the producer is assumed
    to tile like this plan (see the module doc)."""
    if mode not in ("dropin", "chained"):
        raise ValueError("mode: dropin | chained")
    why = winograd3d_ineligible_reason(problem, direction, dtype)
    if why:
        return Winograd3dPlan("direct", mode=mode, direction=direction, reason=why)
    if _pad(problem) > 0 and mode != "dropin":
        return Winograd3dPlan(
            "direct",
            mode=mode,
            direction=direction,
            reason="padded: drop-in only (chained = the halo form)",
        )
    if prescale == "auto":
        prescale = "none" if range_safe else "bps"
    if prescale == "none" and not range_safe:
        return Winograd3dPlan(
            "direct",
            mode=mode,
            direction=direction,
            reason="f16 operands without prescale need range_safe=True",
        )
    if producer is not None:
        if mode != "chained":
            raise ValueError("producer: chained mode only")
        why = _producer_error(producer, direction, prescale)
        if why:
            return Winograd3dPlan("direct", mode=mode, direction=direction, reason=why)
    cands = winograd3d_candidates(
        problem, direction, mode, prescale, xpf=xpf, producer=producer
    )
    if not cands:
        return Winograd3dPlan(
            "direct",
            mode=mode,
            direction=direction,
            reason="no valid yx tiling (e.g. < 32 2-D tiles per plane)",
        )
    key = (direction,) + ((problem.Di, problem.Hi, problem.Wi, problem.C, problem.K))
    table = {0: TUNED, 1: TUNED_PAD1, 2: TUNED_PAD011}.get(_pad(problem), {})
    if xpf:
        tx = {1: TUNED_PAD1_XPF, 2: TUNED_PAD011_XPF}.get(_pad(problem), {})
        if key in tx:
            table = tx
            direct_pad_copy = (
                True  # its "direct" baseline is already the best direct route
            )
    tuned = table.get(key) if use_measured else None
    mkey = mode + ("" if prescale == "bps" else "_nops")
    if tuned is None or mkey not in tuned:
        return Winograd3dPlan(
            "direct",
            mode=mode,
            direction=direction,
            source="untuned",
            reason=f"no decision table entry for {key} {mkey}",
        )
    name, beats = tuned[mkey]
    alts = tuple((k, v[0]) for k, v in tuned.items() if k != "baselines")
    # padded FWD: the direct alternative may copy X into a zero-haloed buffer and run the
    # halo-form kernel (torch strided pad copy, or the rocKE pad copy kernel
    # build_pad_copy)
    need = [
        b
        for b in tuned["baselines"]
        if b == "direct" or (direct_pad_copy and b.startswith("pad_copy"))
    ]
    pick = [c for c in cands if c[0] == name]
    if not pick:
        # the decision was taken for that candidate only: never substitute another one
        return Winograd3dPlan(
            "direct",
            mode=mode,
            direction=direction,
            source="tuned",
            reason=f"tuned {name} is not a valid candidate here",
            alternatives=alts,
        )
    name, spec, wtv = pick[0]
    lost = [b for b in need if b not in beats]
    if not lost:
        return Winograd3dPlan(
            "yx",
            tuple(sorted(_spec_fields(spec).items())),
            wtv,
            mode,
            direction,
            prescale,
            source="tuned",
            err_class=_ERR_CLASS["yx_f2"],
            reason=f"{name}: tuned, beats {', '.join(need)} " f"by {TUNED_MARGIN}",
            alternatives=alts,
        )
    return Winograd3dPlan(
        "direct",
        mode=mode,
        direction=direction,
        source="tuned",
        reason=f"tuned {name} does not beat {', '.join(lost)} " f"by {TUNED_MARGIN}",
        alternatives=alts,
    )


def _spec_fields(spec) -> dict:
    """Non-default WinogradYxSpec fields (the benchmark's --spec JSON, minus the shape)."""
    out = {}
    for f in fields(spec):
        if f.name in ("Di", "Hi", "Wi", "C", "K", "name"):
            continue
        v = getattr(spec, f.name)
        if v != f.default:
            out[f.name] = v
    return out


# The ten 3x3x3 convs of a 3-D UNet (halo form, N=1): cell -> (Di, Hi, Wi, C, K).
UNET3D_SHAPES = {
    1: (258, 130, 130, 64, 64),
    2: (258, 130, 130, 128, 64),
    3: (130, 66, 66, 256, 128),
    4: (130, 66, 66, 128, 128),
    5: (66, 34, 34, 512, 256),
    6: (66, 34, 34, 256, 256),
    7: (34, 18, 18, 512, 512),
    8: (34, 18, 18, 1024, 512),
    9: (130, 66, 66, 64, 128),
    10: (18, 10, 10, 1024, 1024),
}


def unet3d_plans(mode: str = "dropin", form: str = "halo", **kw):
    """{(direction, cell): Winograd3dPlan} for the UNET3D_SHAPES convs (FWD and BWD-data);
    ``form="adapter"``: the pad-(1,1,1) problems (input 2 smaller per dim).
    """
    out = {}
    pd, ph = {"halo": (0, 0), "adapter": (1, 1), "adapter011": (0, 1)}[form]
    for sid, (D, H, W, C, K) in UNET3D_SHAPES.items():
        p = ConvProblem(
            N=1,
            Hi=H - 2 * ph,
            Wi=W - 2 * ph,
            C=C,
            K=K,
            Y=3,
            X=3,
            Di=D - 2 * pd,
            Z=3,
            sD=1,
            pD=pd,
            dD=1,
            pH=ph,
            pW=ph,
        )
        for d in ("fwd", "dgrad"):
            out[(d, sid)] = select_winograd3d(p, d, mode=mode, **kw)
    return out


if (
    __name__ == "__main__"
):  # python -m kernels.common.conv_winograd3d_select [dropin|chained] [--range-safe]
    # [--adapter | --adapter011] [--pad-copy] [--xpf]
    import json
    import sys

    md = (
        sys.argv[1]
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
        else "dropin"
    )
    rs = "--range-safe" in sys.argv
    fm = (
        "adapter011"
        if "--adapter011" in sys.argv
        else "adapter" if "--adapter" in sys.argv else "halo"
    )
    pcp = "--pad-copy" in sys.argv
    xkw = {"xpf": True} if "--xpf" in sys.argv else {}
    for (d, sid), pl in unet3d_plans(
        md, form=fm, range_safe=rs, direct_pad_copy=pcp, **xkw
    ).items():
        print(
            json.dumps(
                {
                    "dir": d,
                    "cell": sid,
                    "mode": md,
                    "form": fm,
                    "kind": pl.kind,
                    "label": pl.label,
                    "source": pl.source,
                    "wtv": pl.wtv,
                    "spec": pl.spec_dict,
                    "reason": pl.reason,
                }
            )
        )
