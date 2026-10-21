# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split-K resolution for the implicit-GEMM convolutions (fwd, wgrad, dgrad).

The one place that turns a conv problem and a kernel tile into a split-K
degree and a reduction path. The kernel builders and the benchmark (JIT
sweep and ``--run-from-cache``) all call :func:`resolve_conv_split_k`, so a
degree requested as ``split_k=-1`` (auto) comes out the same wherever it is
resolved -- the benchmark times the degree a builder would build, and a
builder never re-derives it from different inputs. (The grouped-convolution
dispatcher serves registered instance lists and registers no wgrad instances,
so it does not resolve a wgrad degree.)

An auto degree comes from the split-K cost model
(:mod:`rocke.helpers.split_k_planner`) on arches that have one. This module
contributes what is conv-specific:

* the GEMM each direction launches: ``kpg x Z*Y*X*cpg x N*[Do*]Ho*Wo`` for
  wgrad, widened by ``group_merge`` (a merged tile covers that many groups'
  channels); ``N*Hi*Wi x cpg x Y*X*kpg`` for dgrad;
* the reductions the kernels can build, with their limits: the packed 16-bit
  atomic needs an even dW row and an even store vector
  (:func:`wgrad_atomic_epilogue_available`) and accumulates rounding error with
  the degree (:data:`ATOMIC_16BIT_MAX_SPLIT_K`); a merged tile reduces through
  the workspace only; the workspace must fit the i32 ``ws_bytes`` kernarg;
* the gridDim.z budget (``z = groups * split_k``).

An explicit degree, an override, and an auto degree on an arch without a cost
model (gfx942 included: it has not been fitted) all go through the
long-standing dispatcher rule instead: the CK formula
(:func:`rocke.helpers.split_k.select_split_k_wgrad`) for auto, then the packed
atomic whenever it can be built, else the workspace, else no split. The kernel
builders and the benchmark, which used to feed the CK formula unmerged
per-group dims, get it on the launched geometry (merged dims, launched group
count).

``override`` forces the degree of an auto request: ``1`` forces no split, an
integer ``n >= 2`` forces that degree, held to what the kernel can build --
including :data:`ATOMIC_16BIT_MAX_SPLIT_K` for the packed 16-bit atomic (a
wgrad request free to pick its reduction moves to the workspace above it). An
explicit ``split_k`` is never overridden. The resolver reads no environment, so
a spec resolves the same way in every process; the benchmark maps its
``ROCKE_CONV_SPLIT_K`` variable onto ``override`` through
:func:`parse_split_k_override`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from rocke.core.arch.target import ArchTarget
from rocke.helpers.split_k import select_split_k_wgrad
from rocke.helpers.split_k_planner import (
    MAX_GRID_DIM_Z,
    SplitKKernel,
    SplitKProblem,
    SplitKReduction,
    select_split_k_plan,
    split_k_cost_model,
)

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm_dgrad import (
    _dg_K,
    _dg_M,
    _dg_N,
    dgrad_split_k_available,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    _DEFAULT_WS_REPLICAS,
    _wg_K,
    _wg_M,
    _wg_N,
    wgrad_atomic_epilogue_available,
)
from kernels.common.conv_implicit_gemm_wgrad_two_stage import wgrad_workspace_nbytes

__all__ = [
    "ATOMIC_16BIT_MAX_SPLIT_K",
    "MAX_WGRAD_WS_BYTES",
    "ConvSplitK",
    "parse_split_k_override",
    "resolve_conv_split_k",
]

# Highest degree the planner gives the packed bf16/fp16 atomic reduction. Each
# slice rounds its partial into the 16-bit output, so the error grows with the
# degree: on the gfx950 wgrad corpus the verify gate fails broadly from 256 on,
# while the f32 workspace reduction passes at every degree. An override is held
# to it too; only an explicit degree is the caller's own.
ATOMIC_16BIT_MAX_SPLIT_K = 128

# ws_bytes is an i32 kernarg, so a two-stage workspace above this overflows it.
MAX_WGRAD_WS_BYTES = (1 << 31) - 4

_DIRECTIONS = ("fwd", "wgrad", "dgrad")
_REDUCTIONS = ("auto", "atomic", "workspace")


@dataclass(frozen=True)
class ConvSplitK:
    """A resolved split-K configuration.

    Attributes:
        split_k: the degree to build and launch (1 = no split).
        two_stage: reduce through the f32 workspace and Stage 2 (wgrad only).
        requested: the degree before the launch limits applied -- the explicit
            value, the override, or the auto choice.
        reason: why, for logs and diagnostics.
    """

    split_k: int
    two_stage: bool
    requested: int
    reason: str


def parse_split_k_override(text: str | None) -> int | None:
    """Parse a split-K override setting into the resolver's ``override``.

    ``None``, ``""`` or ``"auto"``: no override (``None``). ``"off"`` (or
    ``"0"``/``"1"``): no split (``1``). An integer ``n >= 2``: that degree.
    Anything else is an error.
    """
    raw = (text or "").strip().lower()
    if raw in ("", "auto"):
        return None
    if raw in ("off", "none", "false"):
        return 1
    try:
        n = int(raw)
    except ValueError:
        n = -1
    if n < 0:
        raise ValueError(
            f"split-K override {raw!r}: expected 'auto', 'off' or an integer >= 0"
        )
    return max(1, n)


def _is_16bit(dtype: str) -> bool:
    return dtype in ("bf16", "fp16")


def _operand_bytes(dtype: str) -> int:
    # The conv kernels read A and B, and write the output, in one dtype.
    return 4 if dtype == "fp32" else 2


def _atomic_reduction(dtype: str) -> SplitKReduction:
    if _is_16bit(dtype):
        return SplitKReduction("atomic", dtype, max_split_k=ATOMIC_16BIT_MAX_SPLIT_K)
    return SplitKReduction("atomic", "fp32")


def _resolve_wgrad(
    p: ConvProblem,
    *,
    arch: str,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    group_merge: int,
    ws_replicas: int,
    split_k: int,
    reduction: str,
    epilogue: str | None,
    allow_unsplit: bool,
    override: int | None,
    num_cus: int | None,
    num_xcds: int | None,
) -> ConvSplitK:
    gm = max(1, group_merge)
    ws_replicas = ws_replicas if ws_replicas > 0 else _DEFAULT_WS_REPLICAS
    grid_groups = max(1, p.groups // gm)
    z_cap = MAX_GRID_DIM_Z // grid_groups
    # Merged tiles cannot use the packed atomic: its epilogue has no way to
    # drop an off-diagonal group pair, so only the workspace carries the mask.
    # The packed 16-bit atomic is emitted by the cshuffle epilogue only (the
    # direct-store one scatters the MFMA layout and has no pairs to add).
    layout_ok, atomic_why = wgrad_atomic_epilogue_available(p, dtype, None)
    if gm > 1:
        atomic_why = "a merged tile reduces through the workspace only"
    elif layout_ok and epilogue == "default" and _is_16bit(dtype):
        atomic_why = "the default epilogue emits no packed 16-bit atomic"
    atomic_ok = atomic_why == "ok"
    # Deliberately unmerged: the workspace is R copies of the true dW.
    ws_ok = wgrad_workspace_nbytes(p, ws_replicas) <= MAX_WGRAD_WS_BYTES
    if reduction == "workspace" and not ws_ok:
        # Fixing the reduction does not lift its limit: the caller's two-stage
        # binary cannot encode this workspace at any degree.
        raise ValueError(
            "wgrad split-K cannot be built: the workspace exceeds the i32 "
            f"ws_bytes limit (ws_replicas={ws_replicas})"
        )

    def _rule(requested: int, source: str, *, atomic_cap: bool) -> ConvSplitK:
        """The long-standing path choice for a given degree.

        ``atomic_cap`` holds the packed 16-bit atomic to its accuracy limit: a
        degree above it moves to the workspace when the reduction is free,
        else comes down to the limit.
        """
        degree = max(1, min(requested, z_cap))
        if degree == 1:
            return ConvSplitK(1, False, requested, f"{source}: no split")
        if reduction == "atomic" and not atomic_ok:
            # Fixing the reduction is the caller saying its binary is the packed
            # atomic one; it cannot be, so fail here rather than hand back a
            # degree the builder then rejects (as for the workspace above).
            raise ValueError(
                f"wgrad split-K cannot be built with the packed atomic: {atomic_why}"
            )
        over_cap = atomic_cap and _is_16bit(dtype) and degree > ATOMIC_16BIT_MAX_SPLIT_K
        # A free reduction takes the packed atomic when it can, unless the
        # degree is past the atomic's limit and the workspace can take it.
        use_atomic = reduction == "atomic" or (
            reduction == "auto" and atomic_ok and not (over_cap and ws_ok)
        )
        if use_atomic:
            if over_cap:
                return ConvSplitK(
                    ATOMIC_16BIT_MAX_SPLIT_K,
                    False,
                    requested,
                    f"{source}: atomic, held to the 16-bit atomic limit "
                    f"{ATOMIC_16BIT_MAX_SPLIT_K}",
                )
            return ConvSplitK(degree, False, requested, f"{source}: atomic")
        if ws_ok:
            return ConvSplitK(degree, True, requested, f"{source}: workspace")
        return ConvSplitK(
            1, False, requested, f"{source}: workspace too large, no split"
        )

    if split_k != -1:
        return _rule(split_k, f"explicit split_k={split_k}", atomic_cap=False)
    if override is not None:
        return _rule(override, f"override split_k={override}", atomic_cap=True)

    # The launched GEMM: a merged tile covers gm groups' channels at once.
    m, n, k = _wg_M(p) * gm, _wg_N(p) * gm, _wg_K(p)
    cost = split_k_cost_model(arch, num_cus, num_xcds)
    if cost is None:
        legacy = select_split_k_wgrad(
            wg_M=m,
            wg_N=n,
            wg_K=k,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            arch=arch,
            groups=grid_groups,
            block_size=block_size,
        )
        return _rule(legacy.split_k, f"CK formula ({legacy.reason})", atomic_cap=False)

    reductions = []
    if reduction in ("auto", "atomic") and atomic_ok:
        reductions.append(_atomic_reduction(dtype))
    if reduction in ("auto", "workspace") and ws_ok:
        reductions.append(
            SplitKReduction(
                "workspace", "fp32", workspace=True, workspace_copies=ws_replicas
            )
        )
    can_split = bool(reductions) and -(-k // tile_k) >= 2 and z_cap >= 2
    if not can_split:
        if not allow_unsplit:
            raise ValueError(
                f"wgrad split-K cannot be built for this problem: "
                f"{'no room to split' if reductions else 'no reduction available'} "
                f"(reduction={reduction!r}, group_merge={gm}, k={k}, tile_k={tile_k})"
            )
        return ConvSplitK(
            1, False, 1, "no split possible: one K tile, no z budget or no reduction"
        )
    waves = max(1, block_size // ArchTarget.from_gfx(arch).wave_size)
    # Merged tiles exist only in split-K form in the AOT cache (merging is what
    # makes a depthwise tile wide enough for split-K to pay), so a merged auto
    # request splits whenever it can.
    plan = select_split_k_plan(
        SplitKProblem(
            m=m,
            n=n,
            k=k,
            batch=grid_groups,
            operand_bytes=_operand_bytes(dtype),
            output_bytes=_operand_bytes(dtype),
        ),
        SplitKKernel(tile_m, tile_n, tile_k, waves),
        reductions,
        cost,
        allow_unsplit=allow_unsplit and gm == 1,
    )
    two_stage = plan.reduction is not None and plan.reduction.workspace
    return ConvSplitK(plan.split_k, two_stage, plan.split_k, plan.reason)


def _resolve_dgrad(
    p: ConvProblem,
    *,
    arch: str,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    split_k: int,
    allow_unsplit: bool,
    override: int | None,
    num_cus: int | None,
    num_xcds: int | None,
) -> ConvSplitK:
    if split_k != -1:
        return ConvSplitK(split_k, False, split_k, f"explicit split_k={split_k}")
    wave_size = ArchTarget.from_gfx(arch).wave_size
    if override is not None:
        # Held to what the kernel can build: an unbuildable split becomes 1
        # rather than a spec the builder then rejects, and the 16-bit atomic
        # stays within its accuracy limit (dgrad has no workspace to move to).
        ok, why = dgrad_split_k_available(p, dtype, wave_size=wave_size)
        cap = ATOMIC_16BIT_MAX_SPLIT_K if _is_16bit(dtype) else MAX_GRID_DIM_Z
        degree = min(override, cap) if ok else 1
        if not ok and override != 1:
            note = f" (no split: {why})"
        elif degree < override:
            note = f" (held to {degree})"
        else:
            note = ""
        return ConvSplitK(degree, False, override, f"override split_k={override}{note}")
    m, n, k = _dg_M(p), _dg_N(p), _dg_K(p)
    cost = split_k_cost_model(arch, num_cus, num_xcds)
    if cost is not None:
        ok, why = dgrad_split_k_available(p, dtype, wave_size=wave_size)
        if not ok:
            return ConvSplitK(1, False, 1, f"no split: {why}")
    if cost is None:
        legacy = select_split_k_wgrad(
            wg_M=m,
            wg_N=n,
            wg_K=k,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            arch=arch,
        )
        return ConvSplitK(
            legacy.split_k, False, legacy.split_k, f"CK formula ({legacy.reason})"
        )
    waves = max(1, block_size // wave_size)
    # The group rides grid y, the slice grid z: groups multiply the CTAs but do
    # not share the z budget.
    plan = select_split_k_plan(
        SplitKProblem(
            m=m,
            n=n,
            k=k,
            batch=max(1, p.groups),
            operand_bytes=_operand_bytes(dtype),
            output_bytes=_operand_bytes(dtype),
            z_batch=1,
        ),
        SplitKKernel(tile_m, tile_n, tile_k, waves),
        [_atomic_reduction(dtype)],
        cost,
        allow_unsplit=allow_unsplit,
    )
    return ConvSplitK(plan.split_k, False, plan.split_k, plan.reason)


def resolve_conv_split_k(
    direction: str,
    problem: ConvProblem,
    *,
    arch: str,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    group_merge: int = 1,
    ws_replicas: int = _DEFAULT_WS_REPLICAS,
    split_k: int = -1,
    reduction: str = "auto",
    epilogue: str | None = None,
    allow_unsplit: bool = True,
    override: int | None = None,
    num_cus: int | None = None,
    num_xcds: int | None = None,
) -> ConvSplitK:
    """Resolve the split-K degree and, for wgrad, the reduction path.

    Args:
        direction: ``"fwd"``, ``"wgrad"`` or ``"dgrad"``.
        problem: the convolution.
        arch: target arch.
        dtype: the output (dW / dX) dtype -- it decides which atomics exist.
        tile_m, tile_n, tile_k: the kernel's block tile.
        block_size: threads per workgroup.
        group_merge: wgrad groups folded into one tile (1 = unmerged).
        ws_replicas: wgrad only. Workspace copies of the two-stage kernel the
            caller builds or holds; sizes the workspace and its setup cost
            (<= 0: the builder default).
        split_k: ``-1`` resolves automatically; any other value is kept
            (wgrad clamps it to the gridDim.z budget).
        reduction: wgrad only. ``"auto"`` chooses between the packed atomic
            and the workspace; ``"atomic"`` / ``"workspace"`` fix the reduction
            the caller's kernel is built with, so only the degree is chosen.
        epilogue: wgrad only. The kernel's epilogue, when known: under
            ``"default"`` (direct store) the packed 16-bit atomic cannot be
            emitted, so ``"auto"`` does not offer it. ``None``: not checked.
        allow_unsplit: whether the auto resolution may return degree 1. False
            for a caller holding a split-K binary (degree 1 is a different
            kernel) or building one that requires a split (two-stage).
        override: forces the degree of an auto request (``split_k=-1``):
            ``None`` resolves automatically, ``1`` forces no split, ``n >= 2``
            forces ``n`` held to what the kernel can build (the gridDim.z
            budget, the reductions available, and
            :data:`ATOMIC_16BIT_MAX_SPLIT_K` for the packed 16-bit atomic).
            Ignored for an explicit ``split_k``. Not part of the spec, so a
            caller that builds or caches a kernel puts the resolved degree on
            the spec. See :func:`parse_split_k_override`.
        num_cus, num_xcds: CU and XCD counts of the device the kernel runs
            on, for a part built differently from its arch's default;
            ``None`` keeps the arch default.

    Forward convolution has no split-K path and always resolves to 1.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    if direction != "wgrad" and reduction != "auto":
        raise ValueError(f"reduction={reduction!r} applies to wgrad only")
    if direction != "wgrad":
        epilogue = None
    if override is not None and override < 1:
        raise ValueError(f"override must be None or >= 1, got {override}")
    if direction == "fwd":
        return ConvSplitK(1, False, 1, "fwd implicit GEMM has no split-K path")
    return _resolve_cached(
        direction,
        problem,
        arch=arch,
        dtype=dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_size=block_size,
        group_merge=group_merge,
        ws_replicas=ws_replicas,
        split_k=split_k,
        reduction=reduction,
        epilogue=epilogue,
        allow_unsplit=allow_unsplit,
        override=override,
        num_cus=num_cus,
        num_xcds=num_xcds,
    )


# A benchmark resolves the same (problem, kernel) once per binary and degree
# list, and planning scores up to a few hundred degrees; the answer is a pure
# function of the arguments.
@functools.lru_cache(maxsize=4096)
def _resolve_cached(
    direction: str,
    problem: ConvProblem,
    *,
    arch: str,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    group_merge: int,
    ws_replicas: int,
    split_k: int,
    reduction: str,
    epilogue: str | None,
    allow_unsplit: bool,
    override: int | None,
    num_cus: int | None,
    num_xcds: int | None,
) -> ConvSplitK:
    common = {
        "arch": arch,
        "dtype": dtype,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "block_size": block_size,
        "split_k": split_k,
        "allow_unsplit": allow_unsplit,
        "override": override,
        "num_cus": num_cus,
        "num_xcds": num_xcds,
    }
    if direction == "dgrad":
        return _resolve_dgrad(problem, **common)
    return _resolve_wgrad(
        problem,
        group_merge=group_merge,
        ws_replicas=ws_replicas,
        reduction=reduction,
        epilogue=epilogue,
        **common,
    )
