# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped convolution dispatcher (forward + backward-weight).

Covers ``implicit_gemm_conv`` (forward NHWC × KYXC → NHWK) and
``implicit_gemm_conv_wgrad`` (dY × X → dW weight gradient), sharing a single
``ConvGroupedRequest`` so callers can dispatch both directions from the same
shape description.

SCOPE -- what this dispatcher decides
-------------------------------------
Each candidate commits to a fixed set of tile / warp / pipeline / epilogue
parameters.  All values are hard-coded for now (sweep results TBD).

The epilogue is derived from ``vec_size_c``, which is computed by the same
static heuristic the kernel builder uses
(``ImplicitGemmConvSpec.default_vector_sizes``):

    vec_size_c = largest power-of-two factor of K (fp16/bf16) or K (fp32)
    epilogue   = "cshuffle"  if vec_size_c > 1
    epilogue   = "default"   otherwise

Two arch families are supported, each with its own hard-coded tile config:

* **CDNA** (gfx942, gfx950) — wave64, MFMA 32×32×16 atom:
      tile 64×64×64, warp 2×2, atom 32×32×16, pipeline ``mem``
* **gfx1250** (wave32, WMMA) — WMMA 16×16×32 atom:
      tile 32×32×32, warp 2×2, atom 16×16×32, pipeline ``mem``
  (gfx1250 restricts WMMA conv to pipeline=``mem``, groups=1, no cshuffle
  override, and wgrad is not yet supported)

DEFERRED -- per-problem divisibility and MMA atom selection
------------------------------------------------------------
``is_valid_spec`` / ``is_valid_wgrad_spec`` perform the arch-aware MMA-atom
and LDS-budget checks at the instance level.  Those checks are delegated to
the existing validators rather than duplicated here.


GEOMETRY SELECTION -- approaches for replacing hard-coded tiles
---------------------------------------------------------------
The candidates below use a single fixed tile configuration per arch family.
This section documents the planned approaches for replacing those constants
with data-driven or learned selection.

A key constraint: conv has more independent dimensions than GEMM.  Two
problems with the same implicit-GEMM shape ``(M, N_gemm, K_gemm)`` can have
very different performance characteristics if they differ in how those dims
are composed — e.g. a large ``Y*X`` vs a large ``C`` in ``K_gemm`` changes
the address-computation pattern and L1/L2 reuse completely.  Any selection
strategy must either expose these raw dims as features or bucket them
conservatively enough that the composition differences are irrelevant.

**Approach A — offline sweep + lookup table (simplest)**

Run ``benchmark_implicit_gemm_conv.py`` on a representative set of shapes
(e.g. ``bench_cases_conv.json``) for each target arch and dtype.  Record the
best ``(tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline,
epilogue)`` per shape.  At dispatch time, find the nearest entry in the table
by bucketing the implicit-GEMM dims to the nearest power-of-two:

    M_bucket  = prev_power_of_2(N * Ho * Wo)
    N_bucket  = prev_power_of_2(K)
    K_bucket  = prev_power_of_2(Y * X * C)
    key       = (arch, dtype, M_bucket, N_bucket, K_bucket)

Limitations: bucketing loses the Y*X vs C composition information; a new
arch or dtype requires a fresh sweep; the table must be re-measured whenever
kernel codegen changes.

**Approach B — analytic heuristics**

Choose tiles based on arithmetic intensity and expected occupancy without
measuring.  Example rules:

* If ``M * N_gemm`` is small (< 4096) the 2D grid is too thin to saturate
  the device — prefer a larger tile in the larger dim to reduce wave count.
* If ``K_gemm`` is small (< 64) there is little K-loop work; use a smaller
  ``tile_k`` and the simpler ``mem`` pipeline.
* If ``K % 8 == 0`` use cshuffle with ``vec_size_c = 8`` for wide stores.
* For wgrad, if ``K_wg = N * Ho * Wo`` is large relative to the ``M * N_wg``
  tile area, increase ``split_k`` to saturate the device.

Limitations: heuristics require careful calibration; they are brittle on
edge-cases and need re-tuning when new micro-arch behaviours are found.

**Approach C — online sweep + persistent cache (recommended next step)**

``sweep_space(req)`` already returns all valid ``ConvGroupedSpec`` instances
for a request.  A thin caching layer around ``dispatch_conv_grouped`` can:

1. Hash the request with ``stable_json_hash(req.normalized())`` (already
   computed as ``KernelId.request_hash``).
2. Look up the hash in a JSON/SQLite cache on disk.
3. On a cache miss: compile and time all specs from ``sweep_space``, write
   the winner to the cache, return it.
4. On a cache hit: return the cached spec directly (pure CPU, no GPU needed).

This is equivalent to MIOpen's find-mode and CK's profiler cache, adapted to
the rocke dispatch contract.  The cache file path can be controlled by an env
var (e.g. ``ROCKE_CONV_CACHE``).  The ``Ranker`` hook on
``dispatch_conv_grouped`` provides the injection point: a ``CachingRanker``
can implement steps 2–4 without touching the candidate code.

**Approach D — ML heuristic (long-term)**

Train a small model (gradient-boosted tree or a 2–3 layer MLP) to predict
the best tile config directly from problem features.  The training data comes
from approach A/C sweep results.

Feature vector (all scalar, normalised to log scale):

    [M, N_gemm, K_gemm,       -- implicit-GEMM dims
     N, Ho, Wo,               -- spatial decomposition of M
     K,                       -- = N_gemm
     Y, X, C,                 -- decomposition of K_gemm
     G,                       -- groups
     sH, sW, pH, pW, dH, dW,  -- conv geometry
     dtype_id,                -- 0=fp16, 1=bf16, 2=fp32
     arch_id]                 -- 0=gfx942, 1=gfx950, 2=gfx1250

Target: one-hot over the discrete config space
``{tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline}``.
Epilogue is always derived from ``vec_size_c`` (not predicted).

The model inference is a few hundred microseconds on CPU — negligible vs
kernel compile time, and can be cached by ``request_hash`` like approach C.

Integration: ship the trained model weights as a small binary blob alongside
the dispatcher; load lazily on first call.  A ``MLRanker`` implementing the
``Ranker`` protocol re-orders candidates by predicted TFLOPS before
``CandidateRegistry.select`` picks the first supported one.  Fallback to the
hard-coded defaults if the model is absent or predicts an invalid config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from rocke.core.arch import ArchTarget
from rocke.instances.common.conv_implicit_gemm import (
    ConvDataSpec,
    ConvProblem,
    ImplicitGemmConvSpec,
    is_valid_spec as _fwd_is_valid_spec,
)
from rocke.instances.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    is_valid_wgrad_spec as _wgrad_is_valid_spec,
)
from rocke.dispatch.core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)

# ---------------------------------------------------------------------------
# Family / ABI constants
# ---------------------------------------------------------------------------

_FAMILY_FWD = "conv_implicit_gemm"
_FAMILY_WGRAD = "conv_implicit_gemm_wgrad"

CONV_GROUPED_ABI_VERSION = "hipkg-conv-grouped/v1"

# ---------------------------------------------------------------------------
# Hard-coded tile parameters (to be replaced by sweep-derived tuning tables)
# ---------------------------------------------------------------------------

# CDNA (gfx942, gfx950) — wave64, MFMA 32x32x16
_CDNA_TILE_M = 64
_CDNA_TILE_N = 64
_CDNA_TILE_K = 64
_CDNA_WARP_M = 2
_CDNA_WARP_N = 2
_CDNA_WARP_TILE_MN = 32
_CDNA_WARP_TILE_K = 16

# gfx1250 — wave32, WMMA 16x16x32; pipeline must be "mem", groups=1 only
_GFX1250_TILE_M = 32
_GFX1250_TILE_N = 32
_GFX1250_TILE_K = 32
_GFX1250_WARP_M = 2
_GFX1250_WARP_N = 2
_GFX1250_WARP_TILE_MN = 16
_GFX1250_WARP_TILE_K = 32

_PIPELINE = "mem"

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvGroupedRequest(OperatorRequest):
    """Normalized grouped convolution request (fwd or wgrad, NHWC)."""

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
    # "fwd" | "wgrad"
    direction: str = "fwd"
    # split_k applies to wgrad only; 1 = disabled, -1 = auto, >1 = fixed
    split_k: int = 1
    # optional vec_size_c override; None = let the candidate decide
    vec_size_c: Optional[int] = None
    op: str = "conv_grouped"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = d["dtype"].lower()
        d["layout"] = d["layout"].upper()
        d["direction"] = d["direction"].lower()
        return d


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _problem(req: ConvGroupedRequest) -> ConvProblem:
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
    if not isinstance(req, ConvGroupedRequest):
        return [f"expected ConvGroupedRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "conv_grouped":
        errors.append(f"unsupported op {req.op!r}")
    if req.direction not in ("fwd", "wgrad"):
        errors.append(f"direction must be 'fwd' or 'wgrad', got {req.direction!r}")
    for field_name in ("N", "C", "K", "Hi", "Wi", "Y", "X"):
        if int(getattr(req, field_name)) <= 0:
            errors.append(f"{field_name} must be positive")
    if int(req.G) <= 0:
        errors.append("G (groups) must be positive")
    if req.dtype.lower() not in ("fp16", "bf16"):
        errors.append(f"unsupported dtype {req.dtype!r}; fp16 or bf16 only")
    if req.layout.upper() != "NHWC":
        errors.append(f"unsupported layout {req.layout!r}; NHWC only")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    if errors:
        return errors
    p = _problem(req)
    if p.Ho <= 0 or p.Wo <= 0:
        errors.append(
            f"degenerate output spatial dims Ho={p.Ho} Wo={p.Wo} "
            "(filter larger than padded input)"
        )
    return errors


def _selector_matches(req: ConvGroupedRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


def _vec_size_c(req: ConvGroupedRequest) -> int:
    """Compute vec_size_c the same way the kernel builder does.

    Uses the explicit override from the request when set, otherwise falls back
    to ``ImplicitGemmConvSpec.default_vector_sizes`` which picks the largest
    power-of-two factor of K (matching what ``CShuffleEpilogue.from_grid``
    would auto-select).
    """
    if req.vec_size_c is not None:
        return req.vec_size_c
    _va, _vb, vc = ImplicitGemmConvSpec.default_vector_sizes(
        req.C, req.K, req.dtype.lower()
    )
    return vc


def _epilogue_for(req: ConvGroupedRequest) -> str:
    """Derive epilogue from vec_size_c: cshuffle when >1, default otherwise."""
    return "cshuffle" if _vec_size_c(req) > 1 else "default"


def _data_spec(req: ConvGroupedRequest) -> ConvDataSpec:
    return ConvDataSpec(
        dtype_a=req.dtype.lower(),
        dtype_b=req.dtype.lower(),
        dtype_d=req.dtype.lower(),
    )


def _is_gfx1250(req: ConvGroupedRequest) -> bool:
    return req.arch == "gfx1250"


# ---------------------------------------------------------------------------
# Spec type returned by select_spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvGroupedSpec:
    """Selected spec for a grouped conv candidate (fwd or wgrad)."""

    direction: str          # "fwd" | "wgrad"
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_tile_mn: int
    warp_tile_k: int
    pipeline: str
    epilogue: str
    dtype: str
    arch: str
    split_k: int = 1        # wgrad only
    name: str = "rocke_conv_grouped"

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join
        parts = [
            self.direction,
            self.dtype,
            f"tile{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"warp{self.warp_m}x{self.warp_n}",
            f"atom{self.warp_tile_mn}x{self.warp_tile_mn}x{self.warp_tile_k}",
            self.pipeline,
            self.epilogue,
        ]
        if self.direction == "wgrad" and self.split_k != 1:
            parts.append(f"spk{self.split_k}")
        return kernel_name_join(self.name, *parts)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _fwd_grid(spec: ConvGroupedSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, ConvGroupedRequest)
    p = _problem(req)
    gm = (p.M + spec.tile_m - 1) // spec.tile_m
    gn = (p.N_gemm + spec.tile_n - 1) // spec.tile_n
    # grid_order "NM": x=n-tiles, y=m-tiles — mirrors the fwd conv manifest
    return (gn, gm, p.groups)


def _wgrad_grid(spec: ConvGroupedSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, ConvGroupedRequest)
    p = _problem(req)
    wg_M = p.K              # output channels
    wg_N = p.Y * p.X * p.C  # filter spatial × input channel
    gx = (wg_N + spec.tile_n - 1) // spec.tile_n
    gy = (wg_M + spec.tile_m - 1) // spec.tile_m
    return (gx, gy, spec.split_k)


def _block(spec: ConvGroupedSpec) -> Tuple[int, int, int]:
    # wave_size is baked into block_size via warp_m * warp_n * wave_size.
    # For CDNA wave64: 2*2*64=256; for gfx1250 wave32: 2*2*32=128.
    target = ArchTarget.from_gfx(spec.arch)
    block_size = spec.warp_m * spec.warp_n * target.wave_size
    return (block_size, 1, 1)


# ---------------------------------------------------------------------------
# CDNA forward candidate (gfx942, gfx950)
# ---------------------------------------------------------------------------


def _make_cdna_fwd_candidate() -> KernelCandidate:
    """Forward conv for CDNA (gfx942/gfx950): 64×64×64, 2×2, 32×32×16 MFMA."""
    name = "implicit_gemm_conv"
    spec_id = "igemm_conv_fwd_64x64"
    algorithm = "implicit_gemm_fwd"

    def _tile(req: ConvGroupedRequest):
        return (
            _CDNA_TILE_M, _CDNA_TILE_N, _CDNA_TILE_K,
            _CDNA_WARP_M, _CDNA_WARP_N,
            _CDNA_WARP_TILE_MN, _CDNA_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ImplicitGemmConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm, tile_n=tn, tile_k=tk,
            warp_m=wm, warp_n=wn,
            warp_tile_m=wtmn, warp_tile_n=wtmn, warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            groups=int(req.G),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if req.direction != "fwd":
            return False, f"candidate handles 'fwd', got direction={req.direction!r}"
        if _is_gfx1250(req):
            return False, "gfx1250 is handled by implicit_gemm_conv_gfx1250"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _fwd_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="fwd",
            tile_m=tm, tile_n=tn, tile_k=tk,
            warp_m=wm, warp_n=wn,
            warp_tile_mn=wtmn, warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            dtype=req.dtype.lower(),
            arch=req.arch,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_FWD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_fwd_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx1250 forward candidate (wave32, WMMA 16×16×32)
# ---------------------------------------------------------------------------


def _make_gfx1250_fwd_candidate() -> KernelCandidate:
    """Forward conv for gfx1250: 32×32×32, 2×2, 16×16×32 WMMA, groups=1 only.

    gfx1250 WMMA conv is restricted to pipeline=``mem``, no cshuffle epilogue
    override (``is_valid_spec`` gates cshuffle on WMMA; epilogue is derived
    from vec_size_c as usual), and groups=1.
    """
    name = "implicit_gemm_conv_gfx1250"
    spec_id = "igemm_conv_fwd_gfx1250_32x32"
    algorithm = "implicit_gemm_fwd_gfx1250"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX1250_TILE_M, _GFX1250_TILE_N, _GFX1250_TILE_K,
            _GFX1250_WARP_M, _GFX1250_WARP_N,
            _GFX1250_WARP_TILE_MN, _GFX1250_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ImplicitGemmConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm, tile_n=tn, tile_k=tk,
            warp_m=wm, warp_n=wn,
            warp_tile_m=wtmn, warp_tile_n=wtmn, warp_tile_k=wtk,
            wave_size=32,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            groups=int(req.G),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx1250(req):
            return False, f"gfx1250 candidate requires arch=gfx1250 (got {req.arch!r})"
        if req.direction != "fwd":
            return False, f"candidate handles 'fwd', got direction={req.direction!r}"
        if int(req.G) != 1:
            return False, "WMMA conv on gfx1250 supports only groups=1"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _fwd_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="fwd",
            tile_m=tm, tile_n=tn, tile_k=tk,
            warp_m=wm, warp_n=wn,
            warp_tile_mn=wtmn, warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            dtype=req.dtype.lower(),
            arch=req.arch,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_FWD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_fwd_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# CDNA wgrad candidate (gfx942, gfx950)
# ---------------------------------------------------------------------------


def _make_cdna_wgrad_candidate() -> KernelCandidate:
    """Backward-weight conv for CDNA: 64×64×64, 2×2, 32×32×16 MFMA.

    Epilogue derived from vec_size_c (cshuffle when >1, default otherwise).
    Split-K forwarded from request (1=disabled, -1=auto CK formula, >1=fixed).
    gfx1250 wgrad is not yet supported; use CDNA only.
    """
    name = "implicit_gemm_conv_wgrad"
    spec_id = "igemm_conv_wgrad_64x64"
    algorithm = "implicit_gemm_wgrad"

    def _tile(req: ConvGroupedRequest):
        return (
            _CDNA_TILE_M, _CDNA_TILE_N, _CDNA_TILE_K,
            _CDNA_WARP_M, _CDNA_WARP_N,
            _CDNA_WARP_TILE_MN, _CDNA_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> WgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return WgradConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm, tile_n=tn, tile_k=tk,
            warp_m=wm, warp_n=wn,
            warp_tile_m=wtmn, warp_tile_n=wtmn, warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            split_k=int(req.split_k),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if req.direction != "wgrad":
            return False, f"candidate handles 'wgrad', got direction={req.direction!r}"
        if _is_gfx1250(req):
            return False, "wgrad is not yet supported on gfx1250"
        if req.split_k > 1 and _epilogue_for(req) == "cshuffle":
            return False, "split_k > 1 is incompatible with cshuffle epilogue"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _wgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="wgrad",
            tile_m=tm, tile_n=tn, tile_k=tk,
            warp_m=wm, warp_n=wn,
            warp_tile_mn=wtmn, warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=int(req.split_k),
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_WGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_wgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# Registries — one per direction (different family strings)
# ---------------------------------------------------------------------------

CONV_FWD_REGISTRY = CandidateRegistry(_FAMILY_FWD)
CONV_FWD_REGISTRY.register(_make_cdna_fwd_candidate())
CONV_FWD_REGISTRY.register(_make_gfx1250_fwd_candidate())

CONV_WGRAD_REGISTRY = CandidateRegistry(_FAMILY_WGRAD)
CONV_WGRAD_REGISTRY.register(_make_cdna_wgrad_candidate())


def _registry_for(req: ConvGroupedRequest) -> CandidateRegistry:
    if req.direction == "wgrad":
        return CONV_WGRAD_REGISTRY
    return CONV_FWD_REGISTRY


# ---------------------------------------------------------------------------
# KernelId
# ---------------------------------------------------------------------------


def _kernel_id(
    req: ConvGroupedRequest, candidate: KernelCandidate, spec: ConvGroupedSpec
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op=f"conv_{req.direction}",
        family=candidate.family,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def conv_grouped_candidates(direction: str = "fwd") -> Tuple[KernelCandidate, ...]:
    if direction == "wgrad":
        return CONV_WGRAD_REGISTRY.candidates()
    return CONV_FWD_REGISTRY.candidates()


def conv_grouped_sweep_space(req: OperatorRequest) -> Sequence[ConvGroupedSpec]:
    if _request_errors(req):
        return ()
    assert isinstance(req, ConvGroupedRequest)
    registry = _registry_for(req)
    specs = []
    seen: set[str] = set()
    for candidate in registry.supported(req):
        spec = candidate.select_spec(req)
        h = spec.kernel_name()
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def dispatch_conv_grouped(
    req: ConvGroupedRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the appropriate grouped conv candidate (fwd or wgrad) for ``req``."""
    registry = _registry_for(req)
    candidate = registry.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"selected {candidate.name} ({req.direction}) on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"epilogue={spec.epilogue} (vec_size_c={_vec_size_c(req)})",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
