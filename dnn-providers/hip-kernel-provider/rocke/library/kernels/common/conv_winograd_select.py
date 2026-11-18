# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Winograd forward-conv plan selector (pure Python, no GPU).

Given a :class:`ConvProblem`, an I/O dtype and a :class:`WinogradCostModel`,
:func:`select_winograd` returns the *safe* Winograd plan with the lowest
modelled cost.
There are three plan kinds, one per implementation:

* ``unfused2d`` - the 4-kernel pipeline (K1 filter, K2 input, K3 TDM+WMMA
  batched GEMM, K4 output transform) for 3x3 convs, output tile
  ``(mh, mw)`` (``conv_winograd_fwd`` + ``conv_winograd_gemm_wmma``), with
  the merged K2+K1 launch, a recommended K3 configuration and the batch
  split (``chunks``) that keeps every buffer under the 32-bit offsets.
* ``fused1d`` - the fused 1-D segmented kernel
  (``conv_winograd_fused_1d_wmma``) for ``1 x r`` convs on ``H = 1``.
* ``unfused_gen`` - the generalized 4-kernel pipeline
  (``conv_winograd_gen``) for non-3x3 filters (1-D, 7x1, 5x5), including
  segmented long filters (``6s2`` = 2 x F(6,6) for r = 11).

``kind == "none"`` means no safe Winograd plan exists (see ``reason``).

Cost model
----------
An analytic roofline sum in byte equivalents (relative units; the model
ranks plans and predicts no time)::

    cost = gemm_flops / occupancy / flops_per_byte      (matrix work)
         + xform_flops / (flops_per_byte / valu_ratio)  (fused 1-D only)
         + bytes moved
         + launches * launch_bytes

* ``unfused2d`` / ``unfused_gen``: the K3 flops (T and K/g padded to the
  recommended K3 tile) over the occupancy of its first round of blocks on
  ``num_cus`` CUs; the bytes the transforms move (x, V twice, M twice, y,
  w, U twice), the extra x pass of the bf16 prescale and the shifted V
  re-reads of a segmented (``taps > 1``) K3 (over the occupancy); K1 plus
  K2, K3 and K4 per batch-split chunk as launches (K1 merged into the
  first K2).
* ``fused1d``: the WMMA flops (C padded to 32, K to the K block); the
  input transform, repeated for every K block; x and y; one launch.

:class:`WinogradCostModel` holds the parameters. ``num_cus`` has no default
(it is the target's CU count); the other fields are nominal ratios, not a
fit to measurements, and a caller can override every one of them.

Eligibility
-----------
* 2-D convs only, fp16 / bf16 I/O, stride 1, ``C/g >= 32`` and
  ``K/g >= 32`` (below that a direct kernel always wins).
* Dilation >= 36 is refused (not validated).
* Numerics (rocke metric max|err|/max(|ref|,1), budget ~1.5e-2):
  F(6,3) in 2-D (6x6 tiles) is never auto-selected (1.4-2.1e-2);
  F(4,11), F(6,11), F(2,11), F(6,7) single-segment and F(4x4,5x5) are
  refused; the 4x6 tile is refused for bf16 I/O (its fused-prescale K2
  spills).  6x4 / 4x6 tiles are allowed (1.4-1.9e-2) but lose a near-tie
  to a lower-error tile (``accuracy_margin``).
* Kernel limits: fused 1-D only
  for fp16 I/O, ``H = Y = 1``, groups 1, ``C`` in {32, 64, 128} and ``K % 64 == 0``
  (32 for C = 32), r in {3, 7, 11}.
* Buffers: x, y, U, V and M must each stay below 2 GiB (32-bit buffer
  offsets); otherwise the batch is split into the fewest equal chunks (a
  divisor of N) that fit, and the plan is dropped if none does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fwd import COST_FLOPS_PER_BYTE

__all__ = [
    "WinogradCostModel",
    "WinogradPlan",
    "select_winograd",
    "winograd_plans",
    "winograd_ineligible_reason",
    "plan_is_unsafe",
    "plan_cost",
    "UNSAFE",
]

_IO_DTYPES = ("fp16", "bf16")
_MAX_BUFFER_BYTES = 0x7FFF0000  # conv_winograd_fwd._OOB: 32-bit buffer offsets
_MIN_CPG = 32
_MAX_DILATION = 35  # dilation >= 36 is not validated
_FUSED1D_C = (32, 64, 128)
# Plans above this error class lose a near-tie (``accuracy_margin``).
_ERR_TIEBREAK = 1.5e-2

# Output tiles of the 3x3 pipeline the selector may pick ((6, 6) is refused).
TILES_2D: Tuple[Tuple[int, int], ...] = (
    (2, 2),
    (4, 4),
    (4, 2),
    (2, 4),
    (6, 2),
    (2, 6),
    (6, 4),
    (4, 6),
)

# Expected rocke-metric error class of each plan (GPU validation).
_ERR_CLASS = {
    ("unfused2d", (2, 2)): 1.0e-3,
    ("unfused2d", (4, 2)): 5.5e-3,
    ("unfused2d", (2, 4)): 5.5e-3,
    ("unfused2d", (6, 2)): 7.2e-3,
    ("unfused2d", (2, 6)): 7.2e-3,
    ("unfused2d", (4, 4)): 1.4e-2,
    ("unfused2d", (6, 4)): 1.7e-2,
    ("unfused2d", (4, 6)): 1.7e-2,
}

# ``(kind, form)`` pairs marked unsafe (numerics / validation); never emitted
# (``WinogradPlan.form``; 1-D forms are per axis, "3xF(4,4)" = 3 segments).
UNSAFE = frozenset(
    {
        ("unfused2d", "F(6x6,3x3)"),  # F(6,3) in 2-D: 1.4-2.1e-2, over budget
        ("unfused_gen", "F(4,11)"),  # >= 5e-2 for every point set
        ("unfused_gen", "F(6,11)"),  # 0.7
        ("unfused_gen", "F(2,11)"),  # 1.36e-2, borderline
        ("unfused_gen", "F(6,7)"),  # 1.47e-2 CPU, ~1.75e-2 GPU
        ("unfused_gen", "F(4x4,5x5)"),  # 1.67e-2
        ("fused1d", "F(4,11)"),
        ("fused1d", "F(6,11)"),
        ("fused1d", "F(2,11)"),
        ("fused1d", "F(6,7)"),
    }
)

# 1-D variants per filter length (driver ``--variant`` strings).
_GEN_VARIANTS_1D = {3: ("6", "4"), 5: ("4",), 7: ("4", "4s2"), 11: ("6s2", "4s3")}
_GEN_VARIANTS_5X5 = ("3x3", "2x2")
_FUSED1D_VARIANT = {11: "4s3", 7: "4s2", 3: "6"}

# ---------------------------------------------------------------------------
# Cost model parameters (see the module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WinogradCostModel:
    """Parameters of the analytic cost model (byte-equivalent units).

    Attributes:
        num_cus: compute units of the target (block occupancy of the K3
            GEMM, and the small-GEMM rule of the K3 configuration).
        flops_per_byte: nominal machine balance: matrix (WMMA) flops that
            cost as much as one byte of memory traffic.
        valu_ratio: how many times slower vector-ALU flops (the fused 1-D
            input transform) are than matrix flops.
        launch_bytes: byte equivalent of one kernel launch (its fixed
            latency and tail).
    """

    num_cus: int
    flops_per_byte: float = COST_FLOPS_PER_BYTE
    valu_ratio: float = 8.0
    launch_bytes: float = float(16 << 20)

    def __post_init__(self) -> None:
        if self.num_cus < 1:
            raise ValueError(f"num_cus must be >= 1, got {self.num_cus}")
        if min(self.flops_per_byte, self.valu_ratio) <= 0 or self.launch_bytes < 0:
            raise ValueError("cost model ratios must be positive")


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WinogradPlan:
    """A selected Winograd implementation for one conv.

    ``tile`` is the output tile ``(mh, mw)`` (1-D plans: ``(1, m)`` along
    W, or ``(m, 1)`` for ``r x 1`` filters); ``variant`` is the driver
    variant string of the 1-D / generalized pipelines (``4s3`` = 3
    segments of F(4,4)); ``k3`` the recommended K3 GEMM configuration
    label; ``fused_cfg`` the fused-kernel knobs (as sorted items);
    ``chunks`` the batch split.
    """

    kind: str
    tile: Tuple[int, int] = (0, 0)
    variant: str = ""
    k3: str = ""
    fused_cfg: Tuple[Tuple[str, object], ...] = ()
    chunks: int = 1
    merge: str = ""
    cost: float = float("inf")
    err_class: float = 0.0
    reason: str = ""
    form: str = ""
    features: Tuple[Tuple[str, float], ...] = field(default=(), compare=False)

    @property
    def label(self) -> str:
        """``kind:F(mhxmw,3x3)`` (3x3 plans) or ``kind:<variant>`` (1-D /
        generalized plans)."""
        if self.kind == "none":
            return "none"
        if self.kind in ("unfused2d", "unfused2d_rt"):
            return f"{self.kind}:F({self.tile[0]}x{self.tile[1]},3x3)"
        return f"{self.kind}:{self.variant}"


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def winograd_ineligible_reason(problem: ConvProblem, dtype: str) -> Optional[str]:
    """Why no Winograd plan applies at all (None if some may)."""
    p = problem
    if dtype not in _IO_DTYPES:
        return f"dtype {dtype} (fp16/bf16 only)"
    if p.is_3d:
        return "3-D conv"
    if (p.sH, p.sW) != (1, 1):
        return "stride != 1"
    if (p.Y, p.X) == (1, 1):
        return "1x1 filter (a GEMM)"
    if p.Ho <= 0 or p.Wo <= 0:
        return "empty output"
    if p.cpg < _MIN_CPG or p.kpg < _MIN_CPG:
        return f"C/g={p.cpg}, K/g={p.kpg} < {_MIN_CPG}"
    if max(p.dH, p.dW) > _MAX_DILATION:
        return f"dilation {max(p.dH, p.dW)} >= 36 (not validated)"
    if (p.Y, p.X) == (3, 3):
        return None
    if _gen_variants(p):
        return None
    return f"filter {p.Y}x{p.X} has no safe Winograd variant"


# -- geometry ---------------------------------------------------------------


@dataclass(frozen=True)
class _Axis:
    m: int  # outputs per tile
    r: int  # filter taps
    seg: int = 1  # segments (segmented long filter)

    @property
    def rs(self) -> int:
        return _cdiv(self.r, self.seg)

    @property
    def n(self) -> int:
        return self.m + self.rs - 1 if self.r > 1 else 1


def _gen_axes(p: ConvProblem, variant: str) -> Tuple[_Axis, _Axis]:
    """``(ay, ax)`` of a driver variant string (mirrors benchmark_winograd_gen)."""
    my = 0
    text = variant
    if "x" in text:
        a, text = text.split("x", 1)
        my = int(a)
    seg = 1
    if "s" in text:
        text, s = text.split("s", 1)
        seg = int(s)
    mx = int(text)
    if p.Y == 1:
        return _Axis(1, 1), _Axis(mx, p.X, seg)
    if p.X == 1:
        return _Axis(mx, p.Y, seg), _Axis(1, 1)
    return _Axis(my or mx, p.Y), _Axis(mx, p.X, seg)


def _gen_variants(p: ConvProblem) -> Tuple[str, ...]:
    if p.Y == 1 and p.X in _GEN_VARIANTS_1D:
        return _GEN_VARIANTS_1D[p.X]
    if p.X == 1 and p.Y in _GEN_VARIANTS_1D:
        return _GEN_VARIANTS_1D[p.Y]
    if (p.Y, p.X) == (5, 5):
        return _GEN_VARIANTS_5X5
    return ()


def _variant_label(p: ConvProblem, variant: str) -> str:
    ay, ax = _gen_axes(p, variant)
    parts = []
    for a in (ay, ax):
        if a.r == 1:
            continue
        parts.append(
            (a, f"F({a.m},{a.r})" if a.seg == 1 else f"{a.seg}xF({a.m},{a.rs})")
        )
    if len(parts) == 2 and parts[0][0].seg == parts[1][0].seg == 1:
        return f"F({ay.m}x{ax.m},{ay.r}x{ax.r})"
    return "/".join(s for _, s in parts)


def _unsafe_gen_variant(p: ConvProblem, variant: str) -> bool:
    ay, ax = _gen_axes(p, variant)
    if ("unfused_gen", _variant_label(p, variant)) in UNSAFE:
        return True
    for a in (ay, ax):
        if a.r == 1:
            continue
        # single-segment long filters (n > 10) are ill-conditioned in fp16
        if a.seg == 1 and a.r >= 7 and a.n > 10:
            return True
        if ("unfused_gen", f"F({a.m},{a.r})") in UNSAFE and a.seg == 1:
            return True
    return False


def _tiles(p: ConvProblem, ay: _Axis, ax: _Axis, N: int) -> int:
    th = _cdiv(_cdiv(p.Ho, p.dH), ay.m) if ay.r > 1 else _cdiv(p.Ho, p.dH)
    tw = _cdiv(_cdiv(p.Wo, p.dW), ax.m) if ax.r > 1 else _cdiv(p.Wo, p.dW)
    return N * p.dH * p.dW * th * tw


def _buffers(p: ConvProblem, ay: _Axis, ax: _Axis, N: int) -> Dict[str, int]:
    b = 2
    T = _tiles(p, ay, ax, N)
    bins = ay.n * ax.n
    taps = ay.seg * ax.seg
    return {
        "x": N * p.Hi * p.Wi * p.C * b,
        "y": N * p.Ho * p.Wo * p.K * b,
        "w": p.K * p.Y * p.X * p.cpg * b,
        "U": bins * taps * p.K * p.cpg * b,
        "V": (T + taps - 1) * bins * p.C * b,
        "M": T * bins * p.K * b,
    }


def _batch_chunks(p: ConvProblem, ay: _Axis, ax: _Axis) -> int:
    """Fewest N-chunks (a divisor of N) whose buffers fit 32-bit offsets; 0 if none."""
    for c in range(1, p.N + 1):
        if p.N % c:
            continue
        if all(v < _MAX_BUFFER_BYTES for v in _buffers(p, ay, ax, p.N // c).values()):
            return c
    return 0


def _k3_tile(label: str) -> Tuple[int, int]:
    bm, bn = label.split("_")[0].split("x")[:2]
    return int(bm), int(bn)


def _unfused_cost(
    p: ConvProblem,
    ay: _Axis,
    ax: _Axis,
    dtype: str,
    chunks: int,
    model: WinogradCostModel,
    merged: bool,
) -> Dict[str, float]:
    """Cost terms (byte equivalents) of the 4-kernel pipeline."""
    N = p.N
    T = _tiles(p, ay, ax, N)
    bins = ay.n * ax.n
    taps = ay.seg * ax.seg
    g, cg, kg = p.groups, p.cpg, p.kpg
    bm, bn = _k3_tile(recommended_k3(p, ay, ax, model, chunks))
    # K3 runs once per chunk: bm x bn blocks per (bin, group).
    tc = _cdiv(T, chunks)
    tb, nb = _cdiv(tc, bm), _cdiv(kg, bn)
    blocks = tb * nb * bins * g
    occ = min(1.0, blocks * (bm * bn) / (256 * 256) / model.num_cus)
    flops = 2.0 * chunks * tb * bm * bins * cg * nb * bn * g * taps
    b = _buffers(p, ay, ax, N)
    moved = b["x"] + 2 * b["V"] + 2 * b["M"] + b["y"] + b["w"] + 2 * b["U"]
    prescale = (
        b["x"]
        if (dtype == "bf16" and max(ay.m, ax.m) >= 4 and ay.r == ax.r == 3)
        else 0
    )
    launches = 3 * chunks + (0 if merged else 1)
    return {
        "gemm": flops / occ / model.flops_per_byte,
        "moved": float(moved),
        "prescale": float(prescale),
        # segmented K3 (taps > 1): shifted V re-reads per tap
        "seg_v": (b["V"] * nb * (taps - 1) / occ) if taps > 1 else 0.0,
        "launch": launches * model.launch_bytes,
    }


def _fused_cost(
    p: ConvProblem, ay: _Axis, ax: _Axis, kb: int, model: WinogradCostModel
) -> Dict[str, float]:
    """Cost terms (byte equivalents) of the fused 1-D kernel."""
    T = _tiles(p, ay, ax, p.N)
    bins = ay.n * ax.n
    taps = ay.seg * ax.seg
    nkb = _cdiv(p.kpg, kb)
    cpad = _cdiv(p.cpg, 32) * 32
    flops = 2.0 * T * bins * cpad * nkb * kb * p.groups * taps
    # input transform: bins x n multiply-adds per tile and channel, redone
    # for every K block
    xform = 2.0 * T * bins * ax.n * cpad * p.groups * nkb
    io = (p.N * p.Hi * p.Wi * p.C + p.N * p.Ho * p.Wo * p.K) * 2
    return {
        "gemm": flops / model.flops_per_byte,
        "xform": xform * model.valu_ratio / model.flops_per_byte,
        "moved": float(io),
        "launch": model.launch_bytes,
    }


def plan_cost(plan: WinogradPlan) -> float:
    """Model cost of ``plan`` from its stored terms."""
    return sum(v for _, v in plan.features)


def recommended_k3(
    p: ConvProblem, ay: _Axis, ax: _Axis, model: WinogradCostModel, chunks: int = 1
) -> str:
    """K3 configuration the measured rows chose (the driver still sweeps):
    short-K (C/g <= 384) rows use the 64-deep 3-stage ring, small GEMMs
    (K/g or C/g <= 128, or under half a round of 256x256 blocks) the
    128x128 tile, everything else the 256x256x128 2-stage default."""
    T = _cdiv(_tiles(p, ay, ax, p.N), chunks)
    bins = ay.n * ax.n
    blocks = _cdiv(T, 256) * _cdiv(p.kpg, 256) * bins * p.groups
    if p.kpg <= 128 or p.cpg <= 128 or blocks < model.num_cus // 2:
        return "128x128x128_w2x2_f4x4_s2_tdm"
    if p.cpg <= 384:
        return "256x256x64_w2x4_f8x4_s3_tdm"
    return "256x256x128_w2x4_f8x4_s2_tdm"


def _fused1d_cfg(p: ConvProblem) -> Dict[str, object]:
    if p.C == 32:
        return dict(
            q=4, fk=2, xf="wmma", xf_batch=8, prefetch=True, m_f16=True, merge=True
        )
    return dict(
        q=2,
        fk=4,
        waves=4,
        xf="wmma",
        xf_batch=16,
        prefetch=True,
        m_f16=True,
        merge=True,
    )


def _items(d: Dict[str, object]) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(d.items()))


def winograd_plans(
    problem: ConvProblem, dtype: str = "fp16", *, model: WinogradCostModel
) -> List[WinogradPlan]:
    """Every safe Winograd plan for ``problem``, sorted by model cost."""
    p = problem
    if winograd_ineligible_reason(p, dtype) is not None:
        return []
    out: List[WinogradPlan] = []
    if (p.Y, p.X) == (3, 3):
        for mh, mw in TILES_2D:
            if dtype == "bf16" and (mh, mw) == (4, 6):
                continue
            ay, ax = _Axis(mh, 3), _Axis(mw, 3)
            ch = _batch_chunks(p, ay, ax)
            if not ch:
                continue
            f = _unfused_cost(p, ay, ax, dtype, ch, model, merged=True)
            out.append(
                WinogradPlan(
                    kind="unfused2d",
                    tile=(mh, mw),
                    variant=f"{mh}x{mw}" if mh != mw else str(mh),
                    k3=recommended_k3(p, ay, ax, model, ch),
                    chunks=ch,
                    merge="K2,K1",
                    cost=sum(f.values()),
                    err_class=_ERR_CLASS[("unfused2d", (mh, mw))],
                    form=f"F({mh}x{mw},3x3)",
                    features=tuple(f.items()),
                )
            )
    else:
        # fused 1-D segmented kernel (fp16 I/O only)
        if (
            dtype == "fp16"
            and p.Y == 1
            and p.Hi == 1
            and p.pH == 0
            and p.groups == 1
            and p.X in _FUSED1D_VARIANT
            and p.C in _FUSED1D_C
        ):
            kb = 32 if p.C == 32 else 64
            if p.K % kb == 0:
                v = _FUSED1D_VARIANT[p.X]
                ay, ax = _gen_axes(p, v)
                f = _fused_cost(p, ay, ax, kb, model)
                out.append(
                    WinogradPlan(
                        kind="fused1d",
                        tile=(1, ax.m),
                        variant=v,
                        fused_cfg=_items(_fused1d_cfg(p)),
                        cost=sum(f.values()),
                        err_class=2.5e-3,
                        form=_variant_label(p, v),
                        features=tuple(f.items()),
                    )
                )
        for v in _gen_variants(p):
            if _unsafe_gen_variant(p, v):
                continue
            ay, ax = _gen_axes(p, v)
            ch = _batch_chunks(p, ay, ax)
            if not ch:
                continue
            f = _unfused_cost(p, ay, ax, dtype, ch, model, merged=False)
            out.append(
                WinogradPlan(
                    kind="unfused_gen",
                    tile=(ay.m, ax.m),
                    variant=v,
                    k3=recommended_k3(p, ay, ax, model, ch),
                    chunks=ch,
                    cost=sum(f.values()),
                    err_class=1.0e-2 if (ay.n * ax.n) >= 11 else 6.0e-3,
                    form=_variant_label(p, v),
                    features=tuple(f.items()),
                )
            )
    out = [pl for pl in out if not plan_is_unsafe(pl, p)]
    out.sort(key=lambda pl: pl.cost)
    return out


def plan_is_unsafe(plan: WinogradPlan, problem: Optional[ConvProblem] = None) -> bool:
    """True for a plan marked unsafe (numerics, validation)."""
    if plan.kind == "none":
        return False
    if (plan.kind, plan.form) in UNSAFE:
        return True
    if plan.kind == "unfused2d" and max(plan.tile) >= 6 and min(plan.tile) >= 6:
        return True
    if problem is not None:
        if max(problem.dH, problem.dW) > _MAX_DILATION:
            return True
        if plan.kind in ("unfused_gen", "fused1d") and _unsafe_gen_variant(
            problem, plan.variant
        ):
            return True
        if plan.kind == "fused1d" and plan.variant != _FUSED1D_VARIANT.get(problem.X):
            return True
    return False


def select_winograd(
    problem: ConvProblem,
    dtype: str = "fp16",
    *,
    model: WinogradCostModel,
    accuracy_margin: float = 0.03,
    kinds: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
) -> WinogradPlan:
    """The safe Winograd plan of lowest model cost (``kind == "none"`` if
    there is none).

    ``accuracy_margin``: a plan whose error class is above 1.5e-2 (6x4 /
    4x6, 1.4-1.9e-2 on the GPU rows) loses to a lower-error plan whose
    cost is within this fraction of its own.  ``kinds`` / ``variants`` restrict the
    candidates (labels as in :attr:`WinogradPlan.label`).
    """
    why = winograd_ineligible_reason(problem, dtype)
    if why is not None:
        return WinogradPlan(kind="none", reason=why)
    plans = winograd_plans(problem, dtype, model=model)
    if kinds is not None:
        plans = [pl for pl in plans if pl.kind in kinds]
    if variants is not None:
        plans = [pl for pl in plans if pl.label in variants]
    if not plans:
        return WinogradPlan(
            kind="none", reason="no plan fits (buffers / kernel limits)"
        )
    best = plans[0]
    if best.err_class > _ERR_TIEBREAK and accuracy_margin > 0:
        for pl in plans[1:]:
            if pl.err_class < best.err_class and pl.cost <= best.cost * (
                1 + accuracy_margin
            ):
                best = pl
                break
    return best
