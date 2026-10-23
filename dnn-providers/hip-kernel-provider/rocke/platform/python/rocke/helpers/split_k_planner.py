# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split-K planner: choose a split degree and a reduction by predicted cost.

Split-K cuts a GEMM's reduction axis into ``split_k`` slices, one CTA per
(output tile, slice), and folds the partials together afterwards. Whether that
pays, and how far to split, is a balance of four effects that the usual
"fill the device" formulas (``floor(capacity / tiles)``) collapse into one:

* **Concurrency.** A short grid leaves CUs idle; splitting adds CTAs.
* **Latency and throughput.** Every K step of a CTA waits for its loads
  (latency) and then shares the CU's MFMA and load throughput with the other
  resident CTAs -- a big dense tile saturates the MFMA pipes with one
  workgroup per CU, while a small, padded tile is dominated by latency and
  keeps gaining from shorter K slices.
* **CTA count.** Every CTA has a fixed cost (dispatch, prologue, epilogue),
  so many tiny CTAs -- a deep split of a grid that is already wide -- lose.
* **Dispatch rounds.** CTAs beyond the resident slots run in later rounds; a
  partly filled last round costs a whole round of K-loop.
* **Reduction.** Every slice adds its partial tile into the output (or a
  workspace) with atomics, so the reduction traffic grows linearly with the
  degree -- and atomic throughput differs widely between accumulation
  dtypes.

:func:`select_split_k_plan` evaluates a small roofline cost model for every
candidate (degree, reduction) pair and returns the cheapest. Costs are in units
of ``L``, the load latency of one K iteration. Per CTA round with ``a``
resident workgroups per CU, one K iteration costs::

    tau(a) = 1 + max(a * flop_iter / F, a * byte_iter / B)

(the latency, plus the CU's MFMA or load throughput shared by ``a``
workgroups, ``F`` and ``B`` per latency unit), so the whole launch costs::

    t(s) = rounds * iters_per_cta * tau + C * c
           + C * out_elems * 4 / A_reduction + setup_reduction

with ``C = tiles * s`` CTAs and ``c`` a fixed cost per CTA. ``setup`` is what
a reduction costs once per launch: zeroing the output before atomics
accumulate into it, or clearing a workspace and folding it in a second pass --
the bytes those passes stream. Their launches queue behind the main kernel, so
launch latency itself does not show; the measured cost is the traffic.

``flop_iter`` counts the full tile (the MFMA runs padded lanes),
``byte_iter`` only the unpadded extent (out-of-bounds lanes issue no loads) --
which is what lets a 24 x 27 output in a 128 x 32 tile come out latency-bound
rather than compute-bound.

Candidate degrees are not limited to powers of two (:func:`_degrees`): the
divisors and multiples of the CU count, the smallest degree for each K
iteration count per CTA, and the degrees that end a dispatch round are scored
too. Two hardware rules shape the set. A grouped launch keeps the flat-id
stride between groups a multiple of the XCD count, so the groups' shared NHWC
cache lines are fetched into one die's L2. And a degree off the power-of-two
ladder must beat the best ladder degree by
:attr:`SplitKCostModel.ladder_margin`: between ladder points the model resolves
round filling more finely than its occupancy bound supports.

The model needs no compiled binary. Resident workgroups per CU are bounded by
the wave limit (``max_waves_per_cu // waves_per_block``): register-heavy
kernels fall short of it, but those are the big tiles that are throughput-bound
long before their occupancy matters, so on the power-of-two ladder the bound
picks the same degree as the exact occupancy on the measured corpus. One
deterministic answer is what lets the dispatcher, the kernel builders and the
benchmark agree.

The constants are per arch and fitted on that arch's measurements (see
:class:`SplitKCostModel`); only gfx950 has been fitted. An arch without a model
has no plan (:func:`split_k_cost_model` returns ``None``) and callers keep their
previous heuristic there -- gfx942 included, until it is fitted on its own
measurements and the device geometries it ships in are validated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

__all__ = [
    "MAX_GRID_DIM_Z",
    "SplitKCostModel",
    "SplitKKernel",
    "SplitKPlan",
    "SplitKProblem",
    "SplitKReduction",
    "predict_split_k_cost",
    "select_split_k_plan",
    "split_k_cost_model",
]

# gridDim.z is a 16-bit field in the HSA dispatch packet on every arch rocke
# targets. Split-K rides the slice index (times any batch) on z.
MAX_GRID_DIM_Z = 65535


@dataclass(frozen=True)
class SplitKCostModel:
    """Per-arch constants of the split-K cost model.

    ``num_cus`` and ``max_waves_per_cu`` are hardware facts. The rest are fitted
    on measured split-K degree curves (every power-of-two degree of every cached
    kernel over a conv wgrad shape corpus). The planner only compares candidates,
    so the model is invariant to the time unit; it is fixed to ``L``, the load
    latency of one K iteration, which makes every constant below a ratio to it.
    The fit covers 16-bit operands; other operand widths reuse the constants
    (their load bytes are counted, their MFMA rate is not re-derived).

    Attributes:
        num_cus: compute units.
        max_waves_per_cu: hardware wave slots per CU (SIMDs x waves per SIMD).
        flops_per_latency: MFMA throughput per CU, per latency unit.
        load_bytes_per_latency: operand-load throughput per CU, per latency
            unit (it includes L2 reuse of the implicit-GEMM operands).
        cta_cost: fixed cost per CTA, device-wide, in latency units --
            workgroup dispatch, prologue and epilogue. Matters once a split makes
            many short CTAs.
        atomic_bytes_per_latency: device atomic-add throughput by accumulation
            dtype (``"fp32"`` for the workspace path, ``"bf16"``/``"fp16"`` for
            the packed 16-bit atomics into the output).
        fill_bytes_per_latency: device throughput of a reduction's extra passes
            (zeroing the output or a workspace, reading a workspace back).
        max_split_k: the largest degree the model was fitted and validated on;
            the planner never chooses beyond it.
        ladder_margin: how much cheaper (as a fraction) the model must find a
            degree off the ladder -- 1 and the powers of two, where it was
            fitted -- than the best ladder degree before choosing it. Between
            ladder points the model resolves round filling more finely than
            its occupancy estimate supports, so a small predicted gain there is
            noise; fitted on measured off-ladder degree curves.
        num_xcds: dies the workgroups are dealt to, round-robin by flat
            workgroup id. A grouped launch keeps the id stride between groups
            a multiple of it, so the groups' matching K-slices share a die and
            its L2 (see :func:`_degrees`).
    """

    num_cus: int
    max_waves_per_cu: int
    flops_per_latency: float
    load_bytes_per_latency: float
    cta_cost: float
    atomic_bytes_per_latency: Mapping[str, float]
    fill_bytes_per_latency: float
    max_split_k: int = 1024
    ladder_margin: float = 0.0
    num_xcds: int = 1


# Fitted by library/benchmarks/common/split_k_cost_fit.py on gfx950, in units of
# one K iteration's load latency (see SplitKCostModel).
_COST_MODELS: dict[str, SplitKCostModel] = {
    "gfx950": SplitKCostModel(
        num_cus=256,
        max_waves_per_cu=32,
        flops_per_latency=843448.3,
        load_bytes_per_latency=19716.8,
        cta_cost=0.012296,
        atomic_bytes_per_latency={
            "bf16": 328579.7,
            "fp16": 13196.0,
            "fp32": 1344520.3,
        },
        fill_bytes_per_latency=43105.9,
        ladder_margin=0.05,
        num_xcds=8,
    ),
}


def split_k_cost_model(
    arch: str, num_cus: int | None = None, num_xcds: int | None = None
) -> SplitKCostModel | None:
    """The cost model for ``arch``, or ``None`` if it has none.

    ``num_cus`` and ``num_xcds`` replace the arch's defaults, for a part of the
    arch built differently. The per-CU constants carry over exactly; the
    device-wide ones (atomic, fill and per-CTA costs) are kept, an
    approximation on a part with fewer XCDs.
    """
    cost = _COST_MODELS.get(arch.split(":", 1)[0])
    if cost is None or (num_cus is None and num_xcds is None):
        return cost
    for name, v in (("num_cus", num_cus), ("num_xcds", num_xcds)):
        if v is not None and v < 1:
            raise ValueError(f"{name} must be >= 1, got {v}")
    return replace(
        cost,
        num_cus=cost.num_cus if num_cus is None else num_cus,
        num_xcds=cost.num_xcds if num_xcds is None else num_xcds,
    )


@dataclass(frozen=True)
class SplitKProblem:
    """The GEMM a split degree is chosen for.

    Attributes:
        m, n: output extents of one GEMM (unpadded).
        k: reduction extent.
        batch: independent GEMMs launched on the same grid (e.g. conv groups);
            multiplies the CTA count.
        operand_bytes: bytes per A/B element.
        output_bytes: bytes per output element.
        z_batch: how many of the ``batch`` GEMMs share gridDim.z with the
            slice index (the z budget per GEMM is ``MAX_GRID_DIM_Z // z_batch``);
            defaults to ``batch``. A kernel that puts the batch on another grid
            axis passes 1.
    """

    m: int
    n: int
    k: int
    batch: int = 1
    operand_bytes: int = 2
    output_bytes: int = 2
    z_batch: int | None = None

    def __post_init__(self) -> None:
        for name in ("m", "n", "k", "batch", "operand_bytes", "output_bytes"):
            if getattr(self, name) < 1:
                raise ValueError(f"SplitKProblem.{name} must be >= 1, got {self}")
        if self.z_batch is not None and self.z_batch < 1:
            raise ValueError(f"SplitKProblem.z_batch must be >= 1, got {self}")


@dataclass(frozen=True)
class SplitKKernel:
    """The kernel geometry the cost depends on: block tile and block size."""

    tile_m: int
    tile_n: int
    tile_k: int
    waves_per_block: int

    def __post_init__(self) -> None:
        for name in ("tile_m", "tile_n", "tile_k", "waves_per_block"):
            if getattr(self, name) < 1:
                raise ValueError(f"SplitKKernel.{name} must be >= 1, got {self}")


@dataclass(frozen=True)
class SplitKReduction:
    """One way to fold the split partials.

    Attributes:
        name: caller's label for the reduction (e.g. ``"atomic"``,
            ``"workspace"``); returned in the plan unchanged.
        accum_dtype: dtype of the atomic adds -- keys
            :attr:`SplitKCostModel.atomic_bytes_per_latency`.
        max_split_k: highest degree this reduction stays accurate at (16-bit
            atomics round every partial into the output, so the error grows
            with the degree), or ``None`` for no cap.
        workspace: the partials go to an f32 workspace that a second pass
            folds into the output; otherwise they accumulate into the output,
            which is zeroed first.
        workspace_copies: f32 copies of the output the workspace holds (more
            copies spread the atomic contention, and cost more to clear).
    """

    name: str
    accum_dtype: str
    max_split_k: int | None = None
    workspace: bool = False
    workspace_copies: int = 1


@dataclass(frozen=True)
class SplitKPlan:
    """The chosen degree and reduction.

    Attributes:
        split_k: the degree (1 = no split).
        reduction: the chosen reduction, ``None`` at degree 1.
        predicted_cost: model cost of the choice, in latency units (see
            :func:`predict_split_k_cost`).
        reason: one-line explanation for logs.
        candidates: every evaluated ``(degree, reduction name, predicted_cost)``,
            for diagnostics.
    """

    split_k: int
    reduction: SplitKReduction | None
    predicted_cost: float
    reason: str
    candidates: tuple[tuple[int, str, float], ...] = field(default=(), repr=False)


def _waves_bound_blocks_per_cu(kernel: SplitKKernel, cost: SplitKCostModel) -> int:
    return max(1, cost.max_waves_per_cu // kernel.waves_per_block)


def predict_split_k_cost(
    problem: SplitKProblem,
    kernel: SplitKKernel,
    split_k: int,
    reduction: SplitKReduction | None,
    cost: SplitKCostModel,
) -> float:
    """Model cost of one launch at ``split_k``, in latency units (see the
    module docstring).

    Comparable across degrees and reductions of the same kernel; not a
    prediction of absolute kernel time (kernel-specific efficiency is not
    modelled, and cancels when only the degree changes).
    """
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if split_k > 1 and reduction is None:
        raise ValueError("split_k > 1 needs a reduction")
    tm, tn, tk = kernel.tile_m, kernel.tile_n, kernel.tile_k
    tiles = math.ceil(problem.m / tm) * math.ceil(problem.n / tn) * problem.batch
    k_iters = math.ceil(problem.k / tk)
    ctas = tiles * split_k
    iters = math.ceil(k_iters / split_k)
    slots = cost.num_cus * _waves_bound_blocks_per_cu(kernel, cost)

    flop_iter = 2.0 * tm * tn * tk
    m_eff, n_eff = min(tm, problem.m), min(tn, problem.n)
    byte_iter = (m_eff + n_eff) * tk * problem.operand_bytes

    def tau(blocks_per_cu: float) -> float:
        return 1.0 + max(
            blocks_per_cu * flop_iter / cost.flops_per_latency,
            blocks_per_cu * byte_iter / cost.load_bytes_per_latency,
        )

    full_rounds, rest = divmod(ctas, slots)
    t = full_rounds * iters * tau(slots / cost.num_cus)
    if rest:
        t += iters * tau(math.ceil(rest / cost.num_cus))
    t += ctas * cost.cta_cost
    if split_k > 1:
        bw = cost.atomic_bytes_per_latency[reduction.accum_dtype]
        t += ctas * m_eff * n_eff * 4.0 / bw
        out_bytes = problem.batch * problem.m * problem.n * problem.output_bytes
        if reduction.workspace:
            ws_bytes = problem.batch * problem.m * problem.n * 4.0
            ws_bytes *= reduction.workspace_copies
            # Clear, then a fold that reads the workspace and writes the output.
            t += (2 * ws_bytes + out_bytes) / cost.fill_bytes_per_latency
        else:
            t += out_bytes / cost.fill_bytes_per_latency
    return t


def _degrees(
    problem: SplitKProblem, kernel: SplitKKernel, cost: SplitKCostModel, cap: int
) -> Iterable[int]:
    """Split degrees in ``[2, cap]`` worth scoring, ascending.

    A degree is a candidate when any of these rules gives it a reason to be:

    * a power of two -- the ladder the model was first fitted on;
    * a divisor or a multiple of the CU count -- one CTA tile then spreads
      evenly over the CUs, which matters on parts whose CU count is not a power
      of two (gfx942's 304, should it be fitted);
    * the smallest degree for its K iterations per CTA -- any larger degree
      with the same ``ceil(k_iters / s)`` only adds CTAs, so it is never
      cheaper, and the model's optimum always lies on this set;
    * the largest degree before the CTA count starts another round of the
      device, counted per CU and per wave-bound slot -- one more slice would
      begin a round that is mostly idle.

    A grouped launch (``z_batch > 1``: ``z = group * split_k + slice``) scores
    instead the powers of two and every degree that makes the flat-id stride
    between groups, ``tiles_per_group * split_k``, a multiple of the XCD count
    (at most ``cap / step`` of them, so the rules above are not needed). The
    groups of a grouped conv read the same NHWC cache lines, and an aligned
    stride puts the groups' matching K-slices on one die, so each line is
    fetched into one L2 rather than once per die (misaligned degrees measured
    up to nearly twice as slow on depthwise wgrad). The model does not see that
    cost, so off the ladder only aligned degrees are offered; the ladder itself
    stays whole, since a small misaligned degree can still beat the nearest
    aligned one.
    """
    k_iters = math.ceil(problem.k / kernel.tile_k)
    group_tiles = math.ceil(problem.m / kernel.tile_m) * math.ceil(
        problem.n / kernel.tile_n
    )
    tiles = group_tiles * problem.batch
    cus = cost.num_cus
    rounds = (cus, cus * _waves_bound_blocks_per_cu(kernel, cost))
    z_batch = problem.batch if problem.z_batch is None else problem.z_batch
    step = 1
    if z_batch > 1:
        step = cost.num_xcds // math.gcd(group_tiles, cost.num_xcds)
    for s in range(2, cap + 1):
        if s & (s - 1) == 0:
            yield s
        elif step > 1:
            if s % step == 0:
                yield s
        elif (
            cus % s == 0
            or s % cus == 0
            or math.ceil(k_iters / s) < math.ceil(k_iters / (s - 1))
            or any(
                math.ceil(tiles * (s + 1) / u) > math.ceil(tiles * s / u)
                for u in rounds
            )
        ):
            yield s


def select_split_k_plan(
    problem: SplitKProblem,
    kernel: SplitKKernel,
    reductions: Iterable[SplitKReduction],
    cost: SplitKCostModel,
    *,
    allow_unsplit: bool = True,
    max_split_k: int | None = None,
) -> SplitKPlan:
    """Cheapest (degree, reduction) under the cost model.

    Candidates are degree 1 (when ``allow_unsplit``) and the degrees of
    :func:`_degrees` from 2 up to the tightest of: ``max_split_k``, the model's
    validated range (:attr:`SplitKCostModel.max_split_k`), the K-iteration count
    (a larger degree only adds empty slices), the gridDim.z budget
    (``MAX_GRID_DIM_Z // z_batch``) and each reduction's own cap. Ties go to the
    smaller degree: it is the cheaper reduction for the same predicted cost. A
    degree off the ladder (1 and the powers of two) is chosen only when it beats
    the best ladder degree by more than :attr:`SplitKCostModel.ladder_margin`.

    Raises:
        ValueError: no candidate is admissible (no unsplit and no reduction
            admits a degree >= 2).
    """
    reductions = tuple(reductions)
    k_iters = math.ceil(problem.k / kernel.tile_k)
    z_batch = problem.batch if problem.z_batch is None else problem.z_batch
    z_cap = MAX_GRID_DIM_Z // z_batch
    if z_cap < 1:
        raise ValueError(
            f"no admissible split-K candidate: z_batch={z_batch} exceeds the "
            f"gridDim.z limit {MAX_GRID_DIM_Z} even unsplit"
        )
    cap = min(
        k_iters,
        z_cap,
        cost.max_split_k,
        *(() if max_split_k is None else (max_split_k,)),
    )

    cands: list[tuple[float, int, SplitKReduction | None]] = []
    if allow_unsplit:
        cands.append((predict_split_k_cost(problem, kernel, 1, None, cost), 1, None))
    for red in reductions:
        red_cap = cap if red.max_split_k is None else min(cap, red.max_split_k)
        for s in _degrees(problem, kernel, cost, red_cap):
            cands.append((predict_split_k_cost(problem, kernel, s, red, cost), s, red))
    if not cands:
        raise ValueError(
            f"no admissible split-K candidate: allow_unsplit=False and no "
            f"reduction admits a degree >= 2 (cap={cap}, "
            f"reductions={[r.name for r in reductions]})"
        )
    t, s, red = min(cands, key=lambda c: (c[0], c[1]))
    ladder = [c for c in cands if c[1] & (c[1] - 1) == 0]
    if ladder and s & (s - 1):
        lt, ls, lred = min(ladder, key=lambda c: (c[0], c[1]))
        if lt <= t * (1.0 + cost.ladder_margin):
            t, s, red = lt, ls, lred
    tiles = (
        math.ceil(problem.m / kernel.tile_m)
        * math.ceil(problem.n / kernel.tile_n)
        * problem.batch
    )
    reason = (
        f"cost model: {tiles} tiles x {math.ceil(k_iters / s)} K-iters/CTA at "
        f"split_k={s}"
        + (f" ({red.name})" if red else "")
        + f", predicted cost {t:.4g} over {len(cands)} candidates (cap {cap})"
    )
    table = tuple((c[1], c[2].name if c[2] else "none", c[0]) for c in cands)
    return SplitKPlan(s, red, t, reason, table)
