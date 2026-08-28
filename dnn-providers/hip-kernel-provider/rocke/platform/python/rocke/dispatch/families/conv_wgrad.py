# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Backward-weight convolution (wgrad) dispatcher family.

Implements the deterministic two-stage wgrad path (Parts 1 & 2).

The two-stage path avoids nondeterministic atomic adds by:
  1. Stage 1: writing f32 partial sums per split-K partition to a workspace.
  2. Stage 2: summing workspace slices in a fixed sequential order into dW.

Workspace sizing
----------------
Required bytes = ``split_k * wg_M * wg_N * 4`` (always f32).
A hard cap of 2 GiB is applied; when the preferred workspace exceeds the cap
the dispatcher transparently falls back to ``split_k = 1`` (no workspace, no
atomics, always correct).

Query::

    from rocke.dispatch.families.conv_wgrad import query_wgrad_support
    info = query_wgrad_support(req, split_k=8)
    print(info["workspace"].preferred_bytes)

Fast atomic mode (split_k > 1 without two-stage) is deferred to Part 3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence, Tuple

from ...helpers.manifest import conv_args_signature
from ...helpers.split_k import select_split_k_wgrad
from ...instances.common._conv_implicit_gemm_common import ConvProblem
from ...instances.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    _wg_K,
    _wg_M,
    _wg_N,
    build_implicit_gemm_conv_wgrad,
    is_valid_wgrad_spec,
)
from ...instances.common.conv_implicit_gemm_wgrad_two_stage import (
    _wgrad_stage1_signature,
)
from ..core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)


def _wgrad_dtype(dtype: str) -> str:
    """Normalise dtype string to the canonical short form used by Capability."""
    d = dtype.lower()
    _map = {"fp16": "f16", "fp32": "f32", "float16": "f16", "float32": "f32"}
    return _map.get(d, d)


_FAMILY = "conv_bwd_weight"
_ALGORITHM = "implicit_gemm_wgrad_two_stage"
CONV_WGRAD_ABI_VERSION = "hipkg-conv-wgrad-two-stage/v1"

# Hard cap on workspace allocation.  Requests that exceed this fall back to
# split_k=1 (single pass, no workspace, always correct and deterministic).
# Kept one f32 element below 2 GiB so ws_bytes (passed as i32) never overflows
# signed int32 (max 2,147,483,647).
CONV_WGRAD_WORKSPACE_HARD_CAP: int = 2 * 1024 * 1024 * 1024 - 4  # 2 GiB − 1 f32


# ---------------------------------------------------------------------------
# Request type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvWgradRequest(OperatorRequest):
    """Normalised backward-weight (wgrad) convolution request.

    Covers 2-D NHWC wgrad for now; 3-D is an extension point.
    """

    N: int
    C: int
    K: int
    Hi: int
    Wi: int
    Y: int
    X: int
    arch: str
    G: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    dtype: str = "fp16"
    layout: str = "NHWC"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _wgrad_dtype(self.dtype)
        d["layout"] = self.layout.upper()
        return d

    def dims(self) -> dict[str, int]:
        """Stored dims plus derived GEMM dims the tiling is checked against."""
        d = {
            name: int(getattr(self, name))
            for name in (
                "N",
                "C",
                "K",
                "Hi",
                "Wi",
                "Y",
                "X",
                "G",
                "stride_h",
                "stride_w",
                "pad_h",
                "pad_w",
                "dilation_h",
                "dilation_w",
            )
        }
        try:
            p = _problem(self)
            d.update(
                Ho=int(p.Ho),
                Wo=int(p.Wo),
                wg_M=int(_wg_M(p)),
                wg_N=int(_wg_N(p)),
                wg_K=int(_wg_K(p)),
            )
        except Exception:
            pass
        return d


CONV_WGRAD_DIM_VOCABULARY = (
    "N",
    "C",
    "K",
    "Hi",
    "Wi",
    "Y",
    "X",
    "G",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
    "dilation_h",
    "dilation_w",
    "Ho",
    "Wo",
    "wg_M",
    "wg_N",
    "wg_K",
)


# ---------------------------------------------------------------------------
# Workspace spec and query
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvWgradWorkspaceSpec:
    """Workspace requirement for the two-stage wgrad path.

    Attributes:
        preferred_bytes:    Full workspace for the preferred split_k.
        minimum_bytes:      Always 0 — the fallback (split_k=1) needs none.
        preferred_split_k:  The split_k that generated ``preferred_bytes``.
        fallback_split_k:   Always 1.
        workspace_fits:     True when ``preferred_bytes <= hard_cap``.
        hard_cap:           The cap that was applied.
        fallback_reason:    Empty string when fits; human-readable otherwise.
    """

    preferred_bytes: int
    minimum_bytes: int
    preferred_split_k: int
    fallback_split_k: int
    workspace_fits: bool
    hard_cap: int
    fallback_reason: str


def compute_wgrad_workspace_spec(
    req: ConvWgradRequest,
    split_k: int,
    hard_cap: int = CONV_WGRAD_WORKSPACE_HARD_CAP,
) -> ConvWgradWorkspaceSpec:
    """Compute workspace requirements without allocating anything.

    This is a pure function: no HIP calls, no side effects.

    Special values of ``split_k``:
      * ``split_k == 1``  — no two-stage path; workspace is 0 bytes and
        ``workspace_fits`` is ``True`` (no allocation needed).
      * ``split_k == -1`` — auto-select: resolved via
        :func:`~rocke.helpers.split_k.select_split_k_wgrad` using the
        problem geometry and ``req.arch``.
    """
    if split_k == 1:
        # split_k=1 runs a single-pass wgrad with no workspace.
        return ConvWgradWorkspaceSpec(
            preferred_bytes=0,
            minimum_bytes=0,
            preferred_split_k=1,
            fallback_split_k=1,
            workspace_fits=True,
            hard_cap=hard_cap,
            fallback_reason="",
        )
    if split_k == -1:
        # Auto: resolve to the heuristic split_k for this problem, then recurse.
        p = _problem(req)
        decision = select_split_k_wgrad(
            wg_M=_wg_M(p),
            wg_N=_wg_N(p),
            wg_K=_wg_K(p),
            tile_m=64,
            tile_n=64,  # smallest registered tile gives largest base_grid
            tile_k=64,  # and therefore the most conservative (lowest) split_k estimate
            arch=req.arch,
        )
        return compute_wgrad_workspace_spec(req, decision.split_k, hard_cap)
    p = _problem(req)
    wg_m = _wg_M(p)
    wg_n = _wg_N(p)
    preferred = split_k * wg_m * wg_n * 4  # f32 = 4 bytes
    fits = preferred <= hard_cap
    return ConvWgradWorkspaceSpec(
        preferred_bytes=preferred,
        minimum_bytes=0,
        preferred_split_k=split_k,
        fallback_split_k=1,
        workspace_fits=fits,
        hard_cap=hard_cap,
        fallback_reason=(
            ""
            if fits
            else (
                f"preferred workspace {preferred // (1024*1024)} MiB exceeds "
                f"hard cap {hard_cap // (1024*1024)} MiB; "
                f"falling back to split_k=1"
            )
        ),
    )


def query_wgrad_support(
    req: ConvWgradRequest,
    split_k: int,
    hard_cap: int = CONV_WGRAD_WORKSPACE_HARD_CAP,
) -> dict:
    """Return a human-readable support dict for the two-stage wgrad path.

    Keys:
        workspace:              :class:`ConvWgradWorkspaceSpec`
        supports_deterministic: True when the two-stage pipeline can run —
            i.e. split_k > 1 and the workspace fits within the hard cap.
            False for split_k=1 (no two-stage path; no workspace allocated).
        supports_fast_atomic:   Always False (Part 3).
        abi_version:            The ABI version string for this family.
    """
    ws = compute_wgrad_workspace_spec(req, split_k, hard_cap)
    return {
        "workspace": ws,
        "supports_deterministic": ws.workspace_fits and ws.preferred_split_k > 1,
        "supports_fast_atomic": False,
        "abi_version": CONV_WGRAD_ABI_VERSION,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _problem(req: ConvWgradRequest) -> ConvProblem:
    return ConvProblem(
        N=int(req.N),
        Hi=int(req.Hi),
        Wi=int(req.Wi),
        C=int(req.C),
        K=int(req.K),
        Y=int(req.Y),
        X=int(req.X),
        sH=int(req.stride_h),
        sW=int(req.stride_w),
        pH=int(req.pad_h),
        pW=int(req.pad_w),
        dH=int(req.dilation_h),
        dW=int(req.dilation_w),
        groups=int(req.G),
    )


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, ConvWgradRequest):
        return [f"expected ConvWgradRequest, got {type(req).__name__}"]
    errors = []
    for dim in ("N", "C", "K", "Hi", "Wi", "Y", "X"):
        if int(getattr(req, dim)) <= 0:
            errors.append(f"{dim}={getattr(req, dim)} must be > 0")
    if req.G <= 0:
        errors.append(f"G={req.G} must be > 0")
    try:
        p = _problem(req)
        if p.Ho <= 0 or p.Wo <= 0:
            errors.append(f"degenerate output: Ho={p.Ho} Wo={p.Wo}")
    except Exception as exc:
        errors.append(f"ConvProblem construction failed: {exc}")
    return errors


def _selector_matches(
    req: ConvWgradRequest, candidate: KernelCandidate
) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


def _default_wgrad_spec(
    req: ConvWgradRequest,
    tile_m: int = 64,
    tile_n: int = 64,
    tile_k: int = 64,
    warp_m: int = 2,
    warp_n: int = 2,
    warp_tile_m: int = 16,
    warp_tile_n: int = 16,
    warp_tile_k: int = 16,
    pipeline: str = "mem",
) -> WgradConvSpec:
    """Build a default WgradConvSpec for the given request.

    The request dtype is threaded through to ConvDataSpec so that fp16, bf16,
    and fp32 requests produce correctly-typed kernels.  For fp16/bf16 inputs the
    dtype_a/dtype_b are set to the request dtype; the internal accumulator and
    the workspace are always f32.
    """
    from ...instances.common._conv_implicit_gemm_common import ConvDataSpec

    p = _problem(req)
    # Map dispatcher dtype string to the kernel's dtype_a/b/d convention.
    # Dispatcher uses "f16"/"bf16" (normalized); kernel uses "fp16"/"bf16"/"fp32".
    _dtype_map = {
        "f16": "fp16",
        "fp16": "fp16",
        "bf16": "bf16",
        "fp32": "fp32",
        "f32": "fp32",
    }
    dtype_kernel = _dtype_map.get(req.dtype.lower(), "fp16")
    data = ConvDataSpec(
        dtype_a=dtype_kernel, dtype_b=dtype_kernel, dtype_d=dtype_kernel
    )

    decision = select_split_k_wgrad(
        wg_M=_wg_M(p),
        wg_N=_wg_N(p),
        wg_K=_wg_K(p),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        arch=req.arch,
    )
    ws_spec = compute_wgrad_workspace_spec(req, decision.split_k)
    effective_k = decision.split_k if ws_spec.workspace_fits else 1
    return WgradConvSpec(
        problem=p,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_tile_m=warp_tile_m,
        warp_tile_n=warp_tile_n,
        warp_tile_k=warp_tile_k,
        pipeline=pipeline,
        split_k=effective_k,
        two_stage=(effective_k > 1),
        data=data,
    )


# ---------------------------------------------------------------------------
# Candidate factory
# ---------------------------------------------------------------------------


def _make_candidate(
    *,
    name: str,
    spec_id: str,
    priority: int,
    spec_fn: Callable[[ConvWgradRequest], WgradConvSpec],
    arches: Tuple[str, ...],
    dtypes: Tuple[str, ...] = ("f16", "bf16"),
) -> KernelCandidate:
    """Factory mirroring conv.py's _make_candidate."""

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvWgradRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = spec_fn(req)
        ok, why = is_valid_wgrad_spec(spec, arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> WgradConvSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvWgradRequest)
        return spec_fn(req)

    def grid(spec: WgradConvSpec, req: OperatorRequest) -> Tuple[int, int, int]:
        p = spec.problem
        gm = (_wg_M(p) + spec.tile_m - 1) // spec.tile_m
        gn = (_wg_N(p) + spec.tile_n - 1) // spec.tile_n
        # Z dimension = split_k (each partition is one Z-slice)
        return (gn, gm, max(1, spec.split_k))

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=CONV_WGRAD_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=arches, dtypes=dtypes, layouts=("NHWC",)),
        _supports=support,
        select_spec=select,
        signature=lambda spec: (
            _wgrad_stage1_signature(spec)
            if spec.two_stage
            else tuple(conv_args_signature(spec.data.dtype_a))
        ),
        grid=grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        # When two_stage=True (the common split_k>1 case), build() returns only
        # Stage 1.  Callers that need both stages must use
        # build_implicit_gemm_conv_wgrad_two_stage(spec, arch) directly.
        # DispatchResult.build() is intentionally Stage 1 only so the KernelDef
        # can be lowered and inspected independently; the full pipeline is
        # assembled by the launcher (see conv_implicit_gemm_wgrad_two_stage.py).
        build=build_implicit_gemm_conv_wgrad,
    )
    return candidate


# ---------------------------------------------------------------------------
# Spec factories (one per tuned tile configuration)
# ---------------------------------------------------------------------------

_CDNA_ARCHES = ("gfx942", "gfx950")


def _spec_cdna_mem_64x64(req: ConvWgradRequest) -> WgradConvSpec:
    # gfx942 only supports 16x16x16 f16 atom; gfx950 supports 32x32x16 too.
    wt_m, wt_n, wt_k = (16, 16, 16)
    return _default_wgrad_spec(
        req,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=4,
        warp_n=4,
        warp_tile_m=wt_m,
        warp_tile_n=wt_n,
        warp_tile_k=wt_k,
        pipeline="mem",
    )


def _spec_cdna_mem_128x64(req: ConvWgradRequest) -> WgradConvSpec:
    wt_m, wt_n, wt_k = (16, 16, 16)
    return _default_wgrad_spec(
        req,
        tile_m=128,
        tile_n=64,
        tile_k=64,
        warp_m=8,
        warp_n=4,
        warp_tile_m=wt_m,
        warp_tile_n=wt_n,
        warp_tile_k=wt_k,
        pipeline="mem",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONV_WGRAD_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=CONV_WGRAD_DIM_VOCABULARY,
    require_build=True,
)
CONV_WGRAD_REGISTRY.extend(
    (
        _make_candidate(
            name="conv_wgrad_cdna_mem_64x64",
            spec_id="cdna_mem_64x64",
            priority=10,
            spec_fn=_spec_cdna_mem_64x64,
            arches=_CDNA_ARCHES,
        ),
        _make_candidate(
            name="conv_wgrad_cdna_mem_128x64",
            spec_id="cdna_mem_128x64",
            priority=20,
            spec_fn=_spec_cdna_mem_128x64,
            arches=_CDNA_ARCHES,
        ),
    )
)


# ---------------------------------------------------------------------------
# Kernel ID helper
# ---------------------------------------------------------------------------


def _kernel_id(
    req: ConvWgradRequest,
    candidate: KernelCandidate,
    spec: WgradConvSpec,
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash({"kernel_name": spec.kernel_name()}, n=16)
    return KernelId(
        op="conv_bwd_weight",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def wgrad_stage2_grid(spec: "WgradConvSpec") -> Tuple[int, int, int]:
    """Return the Stage 2 (workspace-reduce) launch grid for a dispatched spec.

    Callers that need the Stage 2 grid without doing a full compile can call
    this on ``result.spec`` after ``dispatch_conv_wgrad``::

        result = dispatch_conv_wgrad(req)
        s1_grid = result.grid                  # Stage 1: (N, M, split_k)
        s2_grid = wgrad_stage2_grid(result.spec)  # Stage 2: (N, M, 1)
    """
    from rocke.instances.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        wgrad_reduce_grid,
    )

    rs = WgradReduceSpec(problem=spec.problem, dtype_d=spec.data.dtype_d)
    return wgrad_reduce_grid(rs)


def dispatch_conv_wgrad(
    req: ConvWgradRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered wgrad candidate for ``req``.

    **Two-stage launch note:** ``result.grid`` and ``result.block`` describe
    Stage 1 only (the GEMM kernel, Z-dim = split_k).  Stage 2 (the workspace
    reduce kernel) uses a different grid with no Z dimension.  Obtain the full
    pipeline via::

        from rocke.instances.common.conv_implicit_gemm_wgrad_two_stage import (
            build_implicit_gemm_conv_wgrad_two_stage)
        pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(
            result.spec, req.arch)
    """
    candidate = CONV_WGRAD_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    s1_grid = candidate.grid(spec, req)
    if spec.two_stage:
        s2_grid = wgrad_stage2_grid(spec)
        grid_explanation = f"stage1_grid={s1_grid} stage2_grid={s2_grid}"
    else:
        grid_explanation = f"grid={s1_grid}"
    # Compute the workspace spec against the *preferred* (pre-fallback) split_k
    # so the fallback reason is visible when the cap was exceeded.  spec.split_k
    # is already the effective (post-fallback) value; re-running select_split_k_wgrad
    # gives us the original heuristic decision before the cap check.
    _p = _problem(req)
    _preferred_decision = select_split_k_wgrad(
        wg_M=_wg_M(_p),
        wg_N=_wg_N(_p),
        wg_K=_wg_K(_p),
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        arch=req.arch,
    )
    ws_spec = compute_wgrad_workspace_spec(req, _preferred_decision.split_k)
    explanation: tuple = (
        f"selected {candidate.name} for wgrad on {req.arch}",
        f"algorithm={candidate.algorithm}",
        f"spec_id={candidate.spec_id}",
        f"two_stage={spec.two_stage}",
        f"split_k={spec.split_k}",
        grid_explanation,
        f"spec_hash={kid.spec_hash}",
        f"request_hash={kid.request_hash}",
    )
    if ws_spec.fallback_reason:
        explanation = explanation + (f"workspace_fallback: {ws_spec.fallback_reason}",)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=s1_grid,
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=explanation,
    )
