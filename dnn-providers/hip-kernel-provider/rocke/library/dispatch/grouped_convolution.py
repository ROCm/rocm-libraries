# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped convolution dispatcher (forward, backward-weight, backward-data).

Covers ``implicit_gemm_conv`` (forward NHWC × KYXC → NHWK),
``implicit_gemm_conv_wgrad`` (dY × X → dW weight gradient) and
``implicit_gemm_conv_dgrad`` (dY × W → dX input gradient), sharing a single
``ConvGroupedRequest`` so callers can dispatch all three directions from the
same shape description.

Coverage is not uniform across the three. Forward and wgrad each have gfx942,
gfx950 and gfx1250 candidates; dgrad has gfx950 only, and pins split_k=1.
Grouped stride-1 dgrad with small channel groups, filters up to 7x7 and
enough work to fill the device is served by a second gfx950 candidate, the
direct-MFMA pipeline (``direct_mfma_conv_dgrad``), which outranks the igemm one
on the problems it admits; its selected spec is a
``ConvGroupedDirectDgradSpec`` whose ``launch_plan`` lists the weight pre-pass
and main-kernel launches and the workspace they need.

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

Three arch families are supported, each with its own hard-coded tile config:

* **gfx942** — wave64, MFMA 16×16×16 atom (largest fp16/bf16 atom available):
      tile 64×64×64, warp 2×2, atom 16×16×16, pipeline ``mem``
* **gfx950** — wave64, MFMA 32×32×16 atom:
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

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Optional, Sequence, Tuple

from rocke.core.arch import ArchTarget
from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    DirectConvSpec,
    DirectDepthwiseDgradWindowedSpec,
    DGRAD_4C_DEFAULT_BLOCK_GROUPS,
    DGRAD_4C_DEFAULT_BLOCK_Q,
    DirectMfmaDgradPlan,
    build_direct_depthwise_dgrad_windowed,
    direct_mfma_dgrad_main_grid,
    is_valid_depthwise_dgrad_win_spec,
    is_valid_spec as _direct_is_valid_spec,
    is_valid_spec_4c as _direct_is_valid_spec_4c,
    make_dgrad_4c_spec,
    make_dgrad_fprop_spec,
    plan_direct_mfma_dgrad,
    preload_weight_vgprs,
)
from kernels.common.conv_implicit_gemm import (
    ConvDataSpec,
    ConvProblem,
    ImplicitGemmConvSpec,
    is_valid_spec as _fwd_is_valid_spec,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    _DEFAULT_WS_REPLICAS as _WGRAD_WS_REPLICAS,
    is_valid_wgrad_spec as _wgrad_is_valid_spec,
    wgrad_atomic_epilogue_available as _wgrad_atomic_epilogue_available,
)
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    is_valid_dgrad_spec as _dgrad_is_valid_spec,
)
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
    selector_matches,
)

# ---------------------------------------------------------------------------
# Family / ABI constants
# ---------------------------------------------------------------------------

_FAMILY_FWD = "conv_implicit_gemm"
_FAMILY_WGRAD = "conv_implicit_gemm_wgrad"
_FAMILY_DGRAD = "conv_implicit_gemm_dgrad"

# Every candidate's kernel takes the argument prefix
# (A, B, D, A_bytes, B_bytes, D_bytes); some append family-specific trailing
# params (the implicit-GEMM dgrad kernel adds sub_gemm_buf and num_sub_gemms,
# the windowed depthwise dgrad kernel adds none). A launcher must size the
# argument list from the built kernel's params, not from this version string.
CONV_GROUPED_ABI_VERSION = "hipkg-conv-grouped/v1"

# ---------------------------------------------------------------------------
# Hard-coded tile parameters (to be replaced by sweep-derived tuning tables)
# ---------------------------------------------------------------------------

# gfx950 — wave64, MFMA 32x32x16 (gfx942 lacks this fp16 atom)
_GFX950_TILE_M = 64
_GFX950_TILE_N = 64
_GFX950_TILE_K = 64
_GFX950_WARP_M = 2
_GFX950_WARP_N = 2
_GFX950_WARP_TILE_MN = 32
_GFX950_WARP_TILE_K = 16

# gfx942 — wave64, MFMA 16x16x16 (largest fp16/bf16 atom available)
_GFX942_TILE_M = 64
_GFX942_TILE_N = 64
_GFX942_TILE_K = 64
_GFX942_WARP_M = 2
_GFX942_WARP_N = 2
_GFX942_WARP_TILE_MN = 16
_GFX942_WARP_TILE_K = 16

# gfx950 wgrad, scalar-B variant — wave64, MFMA 16x16x32, 8 waves per CTA.
#
# Taken when the X (B-operand) free-axis load cannot vectorise at all, i.e.
# ``cpg`` is odd so ``vec_b == 1`` -- the small-channel "stem" convs (C=1, C=3).
# There every B element costs its own full coordinate decode (unmerge k_wg into
# (n, ho, wo) with two magic divides, embed to (h, w), bounds-check), so the
# wgrad hot loop is address arithmetic, not memory or MFMA.  The cost per thread
# scales with the B elements it owns, ``tile_n * tile_k / block_size``, so the
# lever is to halve the B tile (the padded ``tile_n=64`` is mostly waste when
# ``wg_N = Y*X*cpg`` is small anyway) and double the threads.  The 16-wide atom
# is what makes 8 waves fit a 64x32 tile; the 32-wide one caps it at 2x2.
_GFX950_WGRAD_SCALARB_TILE_M = 64
_GFX950_WGRAD_SCALARB_TILE_N = 32
_GFX950_WGRAD_SCALARB_TILE_K = 64
_GFX950_WGRAD_SCALARB_WARP_M = 4
_GFX950_WGRAD_SCALARB_WARP_N = 2
_GFX950_WGRAD_SCALARB_WARP_TILE_MN = 16
_GFX950_WGRAD_SCALARB_WARP_TILE_K = 32
# The MFMA-clustering schedule. It helps this address-bound tile and does
# nothing on the vectorised one, which is why it rides with the variant rather
# than replacing _PIPELINE.
_GFX950_WGRAD_SCALARB_PIPELINE = "compv3"

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
    # 3D depth fields — all must be set together or all left as None (2D)
    Di: Optional[int] = None
    Z: Optional[int] = None
    stride_d: Optional[int] = None
    pad_d: Optional[int] = None
    dilation_d: Optional[int] = None
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

    def dims(self) -> dict[str, int]:
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
        if self.Di is not None:
            for name in ("Di", "Z", "stride_d", "pad_d", "dilation_d"):
                v = getattr(self, name)
                if v is not None:
                    d[name] = int(v)
        try:
            p = _problem(self)
            d.update(Ho=int(p.Ho), Wo=int(p.Wo))
            if p.is_3d:
                d["Do"] = int(p.Do)
        except Exception:
            pass
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
        Di=int(req.Di) if req.Di is not None else None,
        Z=int(req.Z) if req.Z is not None else None,
        sD=int(req.stride_d) if req.stride_d is not None else None,
        pD=int(req.pad_d) if req.pad_d is not None else None,
        dD=int(req.dilation_d) if req.dilation_d is not None else None,
    )


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, ConvGroupedRequest):
        return [f"expected ConvGroupedRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "conv_grouped":
        errors.append(f"unsupported op {req.op!r}")
    if req.direction not in ("fwd", "wgrad", "dgrad"):
        errors.append(
            f"direction must be 'fwd', 'wgrad' or 'dgrad', got {req.direction!r}"
        )
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
    _3d_fields = {
        "Di": req.Di,
        "Z": req.Z,
        "stride_d": req.stride_d,
        "pad_d": req.pad_d,
        "dilation_d": req.dilation_d,
    }
    _set_3d = {k for k, v in _3d_fields.items() if v is not None}
    if _set_3d and _set_3d != set(_3d_fields):
        missing = sorted(set(_3d_fields) - _set_3d)
        errors.append(f"3D fields must all be set or all be None; missing: {missing}")
    if errors:
        return errors
    p = _problem(req)
    if p.Ho <= 0 or p.Wo <= 0:
        errors.append(
            f"degenerate output spatial dims Ho={p.Ho} Wo={p.Wo} "
            "(filter larger than padded input)"
        )
    if p.is_3d and p.Do <= 0:
        errors.append(
            f"degenerate output depth Do={p.Do} "
            "(filter larger than padded input depth)"
        )
    return errors


def _vec_size_c(req: ConvGroupedRequest) -> int:
    """Compute vec_size_c the same way the kernel builder does.

    Forward pass (D=NHWK, last dim K):  uses ImplicitGemmConvSpec, returns _vec(K).
    Wgrad       (D=KYXC, last dim C):   uses WgradConvSpec,          returns _vec(C).
    Dgrad       (D=NHWC, last dim C):   uses DgradConvSpec,          returns _vec(C).

    Each direction has its own ``default_vector_sizes``; they are not
    interchangeable. Dgrad's takes the *per-group* runs ``(cpg, kpg)`` rather
    than ``(C, K)``, so falling through to the forward formula here would size
    the store vector off the wrong extent on a grouped problem.
    """
    if req.vec_size_c is not None:
        return req.vec_size_c
    if req.direction == "wgrad":
        _va, _vb, vc = WgradConvSpec.default_vector_sizes(
            req.C, req.K, req.dtype.lower(), split_k=1
        )
        return vc
    if req.direction == "dgrad":
        p = _problem(req)
        _va, _vb, vc = DgradConvSpec.default_vector_sizes(
            p.cpg, p.kpg, req.dtype.lower()
        )
        return vc
    _va, _vb, vc = ImplicitGemmConvSpec.default_vector_sizes(
        req.C, req.K, req.dtype.lower()
    )
    return vc


def _epilogue_for(req: ConvGroupedRequest) -> str:
    """Derive epilogue from vec_size_c: cshuffle when >1, default otherwise."""
    return "cshuffle" if _vec_size_c(req) > 1 else "default"


def _wgrad_grouped_overrides(req: ConvGroupedRequest) -> Tuple[str, int]:
    """(epilogue, split_k) for a wgrad spec.

    Grouping is orthogonal to the epilogue and split-K: every epilogue (direct,
    cshuffle, and the split-K atomic path) threads the per-group fold
    ``k_out += group*kpg`` into the dW store, and when split_k>1 the group rides
    block_id_z alongside the K-slice (z = groups*split_k).  So grouped uses the
    same vec-derived epilogue and auto split-K formula (split_k=-1) as ungrouped.
    (WMMA grouped is handled by its own candidate, which forces
    default@split_k=1 -- WMMA has neither a cshuffle nor a split-K path.)
    """
    return _epilogue_for(req), -1


def _data_spec(req: ConvGroupedRequest) -> ConvDataSpec:
    return ConvDataSpec(
        dtype_a=req.dtype.lower(),
        dtype_b=req.dtype.lower(),
        dtype_d=req.dtype.lower(),
    )


def _is_gfx942(req: ConvGroupedRequest) -> bool:
    return req.arch == "gfx942"


def _is_gfx950(req: ConvGroupedRequest) -> bool:
    return req.arch == "gfx950"


def _is_gfx1250(req: ConvGroupedRequest) -> bool:
    return req.arch == "gfx1250"


# ---------------------------------------------------------------------------
# Spec type returned by select_spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvGroupedSpec:
    """Selected spec for a grouped conv candidate (fwd or wgrad)."""

    direction: str  # "fwd" | "wgrad" | "dgrad"
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
    split_k: int = 1  # wgrad only
    lds_k_outer: bool = False  # wgrad only
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
        if self.direction in ("wgrad", "dgrad") and self.split_k != 1:
            parts.append(f"spk{self.split_k}")
        # These two change the emitted body, so they have to reach the name --
        # this is the layer whose names key the host-side compile cache, and the
        # instance-level WgradConvSpec.kernel_name() already tags both.
        #   lds_k_outer: different LDS tile shape and a transpose-read operand
        #     fetch rather than a transpose-on-store.
        if self.direction in ("wgrad", "dgrad") and self.lds_k_outer:
            parts.append("kouter")
        return kernel_name_join(self.name, *parts)

    def to_fwd_spec(self, problem: "ConvProblem") -> "ImplicitGemmConvSpec":
        """Build an ImplicitGemmConvSpec from this dispatcher spec and a ConvProblem."""
        assert self.direction == "fwd", "to_fwd_spec is only valid for fwd specs"
        target = ArchTarget.from_gfx(self.arch)
        return ImplicitGemmConvSpec(
            problem=problem,
            name=self.name,
            data=ConvDataSpec(
                dtype_a=self.dtype,
                dtype_b=self.dtype,
                dtype_d=self.dtype,
            ),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_mn,
            warp_tile_n=self.warp_tile_mn,
            warp_tile_k=self.warp_tile_k,
            wave_size=target.wave_size,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            groups=problem.groups,
        )

    def to_wgrad_spec(self, problem: "ConvProblem") -> "WgradConvSpec":
        """Build a WgradConvSpec with a resolved split_k from this dispatcher spec.

        split_k=-1 (auto) is resolved here via the CK formula so the returned
        spec always has a concrete split_k >= 1, ready for grid calculation and
        launch without re-running the formula in the caller.
        """
        assert self.direction == "wgrad", "to_wgrad_spec is only valid for wgrad specs"
        target = ArchTarget.from_gfx(self.arch)
        resolved_split_k, two_stage, _ = _resolve_wgrad_split_k(self, problem)
        return WgradConvSpec(
            problem=problem,
            name=self.name,
            lds_k_outer=self.lds_k_outer,
            data=ConvDataSpec(
                dtype_a=self.dtype,
                dtype_b=self.dtype,
                dtype_d=self.dtype,
            ),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_mn,
            warp_tile_n=self.warp_tile_mn,
            warp_tile_k=self.warp_tile_k,
            wave_size=target.wave_size,
            pipeline=self.pipeline,
            # two_stage writes one f32 per element via workspace store; cshuffle
            # is not used and would produce an invalid spec (validator rejects it).
            epilogue="default" if two_stage else self.epilogue,
            split_k=resolved_split_k,
            two_stage=two_stage,
            # Pin the replica count explicitly rather than inheriting the
            # dataclass default: _resolve_wgrad_split_k caps the scratch against
            # the i32 ws_bytes ABI using this same constant, and a cap computed
            # over a different R than the spec carries bounds nothing.
            ws_replicas=_WGRAD_WS_REPLICAS,
        )

    def to_dgrad_spec(self, problem: "ConvProblem") -> "DgradConvSpec":
        """Build a DgradConvSpec from this dispatcher spec and a ConvProblem.

        No split-K auto-resolution counterpart to :meth:`to_wgrad_spec`: dgrad's
        reduction is ``Y*X*K``, which is not the lopsided axis wgrad's ``N*Ho*Wo``
        is, so the CK formula that rescues wgrad's grid does not apply and the
        candidate pins a concrete degree instead.
        """
        assert self.direction == "dgrad", "to_dgrad_spec is only valid for dgrad specs"
        target = ArchTarget.from_gfx(self.arch)
        return DgradConvSpec(
            problem=problem,
            name=self.name,
            lds_k_outer=self.lds_k_outer,
            data=ConvDataSpec(
                dtype_a=self.dtype,
                dtype_b=self.dtype,
                dtype_d=self.dtype,
            ),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_mn,
            warp_tile_n=self.warp_tile_mn,
            warp_tile_k=self.warp_tile_k,
            wave_size=target.wave_size,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            split_k=self.split_k,
        )


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


# ws_bytes is i32 in the kernel ABI, so a two-stage workspace above this would
# overflow the argument.
_MAX_WGRAD_WS_BYTES = (1 << 31) - 4  # 2 GiB - 4

# hipDeviceAttributeMaxGridDimZ. Unlike x, the z extent is a 16-bit field in the
# dispatch packet on every arch rocke targets, so this is an architectural limit
# rather than a per-device one and is safe as a constant. Grouped wgrad puts
# groups*split_k on z, which is the only rocke launch that can reach it.
_MAX_GRID_DIM_Z = 65535


def _resolve_wgrad_split_k(
    spec: ConvGroupedSpec, p: "ConvProblem"
) -> Tuple[int, bool, int]:
    """Resolve wgrad split_k and two-stage eligibility; returns
    ``(split_k, two_stage, requested_split_k)``.

    Honors a concrete split-K fixed by the candidate (WMMA forces 1 -- it has
    no split-K path); otherwise auto-resolves via the CK formula on the
    per-group GEMM dims (== full dims when groups==1). Grouped rides the group
    on block_id_z alongside the K-slice (z = groups*split_k), so grouped
    split-K is valid on MFMA.

    The result is then held under the scratch cap: when it bites, split_k
    falls back to 1 (a plain store, no scratch needed). Shared by
    ``to_wgrad_spec`` and ``_wgrad_grid`` so the spec and the launch grid can
    never disagree on split_k.

    Two further constraints, both of which only bite once ``groups`` is large
    (they are no-ops at groups == 1):

    * **gridDim.z.** The group and the K-slice share the z axis, so the launch
      needs ``groups * split_k`` blocks there. The CK formula sizes split_k
      from the *per-group* GEMM and never sees the groups factor, so on a
      depthwise problem (groups == C, per-group GEMM 1 x Y*X) it happily asks
      for a degree that overflows the z limit and the launch fails with
      ``hipErrorInvalidValue``. Clamp here, in the one resolver both the spec
      and the grid go through, so they cannot desynchronise.
    * **Atomic availability.** The packed ``<2 x dtype>`` atomic the split-K
      epilogue emits for a 16-bit dW needs an even dW row length
      ``wg_N = Y*X*cpg``. When that fails there is no 16-bit atomic to fall
      back on (gfx9 has no scalar 16-bit atomic add), so the choice is
      split_k=1 or the two-stage f32-workspace path. Promote to two-stage and
      keep the parallelism -- collapsing the reduction axis of a wgrad whose
      K_wg is N*Ho*Wo is the expensive way out.
    """
    spatial = (p.Z if p.is_3d else 1) * p.Y * p.X
    wg_M = p.K // p.groups
    wg_N = spatial * (p.C // p.groups)
    if spec.split_k != -1:
        requested = spec.split_k
    else:
        from rocke.helpers.split_k import select_split_k_wgrad

        requested = select_split_k_wgrad(
            wg_M=wg_M,
            wg_N=wg_N,
            wg_K=p.N * p.Ho * p.Wo * (p.Do if p.is_3d else 1),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            arch=spec.arch,
            groups=p.groups,
            block_size=_block(spec)[0],
        ).split_k
    # The helper already keeps groups*split_k inside the z limit on the auto
    # path; this clamp still has to run for an explicitly requested split_k.
    split_k = max(1, min(requested, _MAX_GRID_DIM_Z // max(1, p.groups)))
    # When the packed 16-bit atomic cannot represent this problem, the f32
    # scratch path is the only way to keep split_k > 1. Delegate rather than
    # re-deriving the rule: a local copy is exactly how the dispatcher came to
    # hand the builder a spec the builder then rejected.
    # vector_size_c is None here by construction: to_wgrad_spec never sets it,
    # so the epilogue derives the store width from the channel dims.
    _atomic_ok, _ = _wgrad_atomic_epilogue_available(p, spec.dtype.lower(), None)
    two_stage = (not _atomic_ok) and split_k > 1
    # The scratch carries no split_k factor, but it does carry the replica
    # factor -- R copies of dW per group. Use the same R this module hands the
    # spec in to_wgrad_spec, or the cap bounds an allocation nobody makes.
    ws_bytes = p.groups * _WGRAD_WS_REPLICAS * wg_M * wg_N * 4
    if two_stage and ws_bytes > _MAX_WGRAD_WS_BYTES:
        two_stage = False
        split_k = 1
    return split_k, two_stage, requested


def _wgrad_grid(spec: ConvGroupedSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, ConvGroupedRequest)
    p = _problem(req)
    spatial = (p.Z if p.is_3d else 1) * p.Y * p.X
    # Per-group GEMM dims. For the ungrouped path G==1 these reduce to wg_M=K,
    # wg_N=spatial*C.
    kpg = p.K // p.groups
    cpg = p.C // p.groups
    wg_M = kpg  # per-group output channels
    wg_N = spatial * cpg  # per-group filter spatial × input channel
    gx = (wg_N + spec.tile_n - 1) // spec.tile_n
    gy = (wg_M + spec.tile_m - 1) // spec.tile_m
    split_k, _two_stage, _requested = _resolve_wgrad_split_k(spec, p)
    # The group index rides on block_id_z alongside the K-slice: z = groups*
    # split_k, decoded in-kernel as group = z // split_k, slice = z % split_k.
    # For G==1 this reduces to (gx, gy, split_k); for split_k==1 to (gx, gy, groups).
    return (gx, gy, p.groups * split_k)


def _dgrad_grid(spec: ConvGroupedSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    """``(flat_tiles, groups, split_k)`` for the unified tilde-decomposed kernel.

    Unlike fwd/wgrad the M-tile count is not a closed form over the problem
    dims: stride > 1 splits the convolution into ``y_tilde * x_tilde``
    independent sub-GEMMs of differing sizes, and the kernel binary-searches a
    packed record buffer to find its own. The flat tile count is therefore the
    cumulative ``block_end`` of the last sub-GEMM, which the instance already
    computes -- derive it there rather than re-deriving the decomposition here.
    The sub-GEMM geometry is channel-independent, so the count is per-group and
    the group rides ``blockIdx.y``.
    """
    assert isinstance(req, ConvGroupedRequest)
    p = _problem(req)
    sub_gemms = spec.to_dgrad_spec(p).compute_sub_gemms()
    flat_tiles = sub_gemms[-1].block_end
    return (flat_tiles, max(int(p.groups), 1), max(int(spec.split_k), 1))


def _block(spec: ConvGroupedSpec) -> Tuple[int, int, int]:
    # wave_size is baked into block_size via warp_m * warp_n * wave_size.
    # For gfx942/gfx950 wave64: 2*2*64=256; for gfx1250 wave32: 2*2*32=128.
    target = ArchTarget.from_gfx(spec.arch)
    block_size = spec.warp_m * spec.warp_n * target.wave_size
    return (block_size, 1, 1)


# ---------------------------------------------------------------------------
# gfx950 forward candidate (wave64, MFMA 32×32×16)
# ---------------------------------------------------------------------------


def _make_gfx950_fwd_candidate() -> KernelCandidate:
    """Forward conv for gfx950: 64×64×64, 2×2, 32×32×16 MFMA (gfx942 lacks this atom)."""
    name = "implicit_gemm_conv"
    spec_id = "igemm_conv_fwd_64x64"
    algorithm = "implicit_gemm_fwd"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX950_TILE_M,
            _GFX950_TILE_N,
            _GFX950_TILE_K,
            _GFX950_WARP_M,
            _GFX950_WARP_N,
            _GFX950_WARP_TILE_MN,
            _GFX950_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ImplicitGemmConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
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
        if not _is_gfx950(req):
            return False, f"gfx950 candidate requires arch=gfx950 (got {req.arch!r})"
        if req.direction != "fwd":
            return False, f"candidate handles 'fwd', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _fwd_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="fwd",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
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
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_fwd_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
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
            _GFX1250_TILE_M,
            _GFX1250_TILE_N,
            _GFX1250_TILE_K,
            _GFX1250_WARP_M,
            _GFX1250_WARP_N,
            _GFX1250_WARP_TILE_MN,
            _GFX1250_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ImplicitGemmConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
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
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _fwd_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="fwd",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
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
        capability=Capability(
            arches=("gfx1250",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_fwd_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx942 forward candidate (wave64, MFMA 16×16×16)
# ---------------------------------------------------------------------------


def _make_gfx942_fwd_candidate() -> KernelCandidate:
    """Forward conv for gfx942: 64×64×64, 2×2, 16×16×16 MFMA."""
    name = "implicit_gemm_conv_gfx942"
    spec_id = "igemm_conv_fwd_gfx942_64x64"
    algorithm = "implicit_gemm_fwd_gfx942"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX942_TILE_M,
            _GFX942_TILE_N,
            _GFX942_TILE_K,
            _GFX942_WARP_M,
            _GFX942_WARP_N,
            _GFX942_WARP_TILE_MN,
            _GFX942_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> ImplicitGemmConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ImplicitGemmConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
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
        if not _is_gfx942(req):
            return False, f"gfx942 candidate requires arch=gfx942 (got {req.arch!r})"
        if req.direction != "fwd":
            return False, f"candidate handles 'fwd', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _fwd_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="fwd",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
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
        capability=Capability(
            arches=("gfx942",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_fwd_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx942 wgrad candidate (wave64, MFMA 16×16×16)
# ---------------------------------------------------------------------------


def _make_gfx942_wgrad_candidate() -> KernelCandidate:
    """Backward-weight conv for gfx942: 64×64×64, 2×2, 16×16×16 MFMA.

    Epilogue derived from vec_size_c (cshuffle when >1, default otherwise).
    Split-K forwarded from request (1=disabled, -1=auto CK formula, >1=fixed).
    """
    name = "implicit_gemm_conv_wgrad_gfx942"
    spec_id = "igemm_conv_wgrad_gfx942_64x64"
    algorithm = "implicit_gemm_wgrad_gfx942"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX942_TILE_M,
            _GFX942_TILE_N,
            _GFX942_TILE_K,
            _GFX942_WARP_M,
            _GFX942_WARP_N,
            _GFX942_WARP_TILE_MN,
            _GFX942_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> WgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return WgradConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_PIPELINE,
            epilogue=_ep,
            split_k=_sk,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx942(req):
            return False, f"gfx942 candidate requires arch=gfx942 (got {req.arch!r})"
        if req.direction != "wgrad":
            return False, f"candidate handles 'wgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _wgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return ConvGroupedSpec(
            direction="wgrad",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_ep,
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=_sk,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_WGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_wgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx950 wgrad candidate (wave64, MFMA 32×32×16)
# ---------------------------------------------------------------------------


def _wgrad_lds_k_outer(req: "ConvGroupedRequest", warp_tile_mn: int) -> bool:
    """Whether a wgrad candidate should use the K-outer LDS layout.

    Used by both the gfx950 (wave64 MFMA) and gfx1250 (wave32 WMMA) candidates;
    the wave size is resolved from ``req.arch`` below rather than assumed, so
    the one gate covers both regimes.

    wgrad's stride-1 global axis is the GEMM *free* axis, so an M-outer LDS tile
    forces a transpose on store: one ``ds_write_b16`` per element, all of them
    bank-conflicting because the row stride is a multiple of the LDS bank period.
    The K-outer layout stores the tile in global order (one wide store) and lets
    ``ds_read_b64_tr_b16`` transpose on the read side instead.

    K-outer lowers the LDS instruction count and removes the row-stride bank
    conflicts the M-outer transpose-on-store incurs. See the ``lds_k_outer``
    field docs on ``WgradConvSpec``.

    Gated to exactly what the transpose-read lane mapping is validated for:
    16-bit A/B operands, and either wave64 MFMA on gfx950 (``ds_read_b64_tr_b16``
    does not exist on gfx942) with a 16- or 32-wide atom edge, or wave32 WMMA on
    gfx1250 (``ds_load_tr16_b128``) with the 16-wide edge.

    Delegates so dispatch and the sweep driver cannot drift: this used to be a
    second copy of the gate that compared the module constant against a tuple
    (constant-true regardless of the request) and never checked wave_size.
    """
    return WgradConvSpec.default_lds_k_outer(
        arch=req.arch,
        dtype_a=req.dtype.lower(),
        dtype_b=req.dtype.lower(),
        warp_tile_m=warp_tile_mn,
        warp_tile_n=warp_tile_mn,
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
    )


def _wgrad_b_is_scalar(req: "ConvGroupedRequest") -> bool:
    """Whether the X (B-operand) free-axis load degenerates to one element.

    ``vec_b`` is the widest power-of-two width that divides ``cpg`` (X is NHWC,
    so the free-axis run is the channel dim), which is 1 exactly when ``cpg``
    is odd. Asks :meth:`WgradConvSpec.default_vector_sizes` rather than testing
    parity directly so the two cannot drift apart if the width ladder changes.

    Depthwise (``cpg == kpg == 1`` with real groups) is excluded even though it
    is the extreme of the scalar-B case. There the CTA count comes from the
    group axis, not from the tile, so the "fewer B elements per thread" argument
    the variant rests on does not drive it -- and measurement agrees: the
    narrow tile is a win on some depthwise shapes and a loss on others, with no
    predictor separating them. Depthwise keeps the tile it was tuned on.
    """
    p = _problem(req)
    cpg = p.C // max(int(p.groups), 1)
    if cpg == 1 and p.kpg == 1 and p.groups > 1:
        return False
    _va, vec_b, _vc = WgradConvSpec.default_vector_sizes(cpg, p.kpg, req.dtype.lower())
    return vec_b == 1


def _make_gfx950_wgrad_candidate() -> KernelCandidate:
    """Backward-weight conv for gfx950: 64×64×64, 2×2, 32×32×16 MFMA.

    Epilogue derived from vec_size_c (cshuffle when >1, default otherwise).
    Split-K forwarded from request (1=disabled, -1=auto CK formula, >1=fixed).
    gfx1250 wgrad is handled by ``_make_gfx1250_wgrad_candidate`` (WMMA 16x16x32).

    Two geometries, selected per request by :func:`_wgrad_b_is_scalar`: the
    64×64 default, and the 64×32 / 8-wave variant for the odd-``cpg`` shapes
    whose B load cannot vectorise (see the ``_GFX950_WGRAD_SCALARB_*``
    constants). One candidate rather than two because the choice is a pure
    function of the request with no overlap -- a second candidate would need
    the same predicate in its ``support`` anyway, and priorities to break a tie
    that cannot occur.
    """
    name = "implicit_gemm_conv_wgrad"
    spec_id = "igemm_conv_wgrad_64x64"
    algorithm = "implicit_gemm_wgrad"

    def _tile(req: ConvGroupedRequest):
        if _wgrad_b_is_scalar(req):
            return (
                _GFX950_WGRAD_SCALARB_TILE_M,
                _GFX950_WGRAD_SCALARB_TILE_N,
                _GFX950_WGRAD_SCALARB_TILE_K,
                _GFX950_WGRAD_SCALARB_WARP_M,
                _GFX950_WGRAD_SCALARB_WARP_N,
                _GFX950_WGRAD_SCALARB_WARP_TILE_MN,
                _GFX950_WGRAD_SCALARB_WARP_TILE_K,
            )
        return (
            _GFX950_TILE_M,
            _GFX950_TILE_N,
            _GFX950_TILE_K,
            _GFX950_WARP_M,
            _GFX950_WARP_N,
            _GFX950_WARP_TILE_MN,
            _GFX950_WARP_TILE_K,
        )

    def _pipeline(req: ConvGroupedRequest) -> str:
        return _GFX950_WGRAD_SCALARB_PIPELINE if _wgrad_b_is_scalar(req) else _PIPELINE

    def _build_instance_spec(req: ConvGroupedRequest) -> WgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return WgradConvSpec(
            problem=_problem(req),
            name=name,
            lds_k_outer=_wgrad_lds_k_outer(req, wtmn),
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_pipeline(req),
            epilogue=_ep,
            split_k=_sk,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx950(req):
            return False, f"gfx950 candidate requires arch=gfx950 (got {req.arch!r})"
        if req.direction != "wgrad":
            return False, f"candidate handles 'wgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _wgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return ConvGroupedSpec(
            direction="wgrad",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_pipeline(req),
            epilogue=_ep,
            lds_k_outer=_wgrad_lds_k_outer(req, wtmn),
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=_sk,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_WGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_wgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _dgrad_lds_k_outer(req: "ConvGroupedRequest", warp_tile_mn: int) -> bool:
    """Whether the gfx950 dgrad candidate should use the K-outer LDS layout.

    Delegates to the instance predicate for the same reason the wgrad helper
    does -- a second copy of the gate is how the two drift apart.

    Note the asymmetry with :func:`_wgrad_lds_k_outer`: dgrad flips only the B
    tile, so the predicate takes only the B-side dtype and warp tile, and it
    additionally keys on ``cpg``. The saving is proportional to the B load
    width, which collapses to 1 on an odd channel run -- there ``axis_b`` is
    already ``"col"``, there is no transpose-on-store to remove, and K-outer
    would only add the read-side cost.
    """
    p = _problem(req)
    return DgradConvSpec.default_lds_k_outer(
        arch=req.arch,
        dtype_b=req.dtype.lower(),
        warp_tile_n=warp_tile_mn,
        cpg=p.C // max(int(p.groups), 1),
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        pipeline=_PIPELINE,
    )


# Shape-keyed gfx950 dgrad tiles: (tile_m, tile_n, tile_k, warp_m, warp_n,
# warp_tile_mn, warp_tile_k). Derived from a same-session sweep of the
# implicit-GEMM dgrad (folded record + tap-outer K loop) over a cohort of dense
# 3x3 / 5x5 stride-1, 1x1, stride-2 and grouped problems; the measurements live
# outside the tree. The default 64x64 tile keeps the grid large, which is what
# small-M and few-channel problems need; the 128x128 tile halves the operand
# traffic per MFMA but only pays once the grid still fills the device.
_GFX950_DGRAD_TILE_DEFAULT = (64, 64, 64, 2, 2, 32, 16)
_GFX950_DGRAD_TILE_LARGE = (128, 128, 64, 2, 2, 16, 32)
# Minimum workgroup count for the 128x128 tile: 1.25 per CU on gfx950 (256
# CUs). At one workgroup per CU (and up to about 1.1) the 64x64 tile, with
# four times the workgroups, measured faster on every problem tried; from
# 1.25 up the 128x128 tile wins on most.
_GFX950_DGRAD_LARGE_MIN_CTAS = 320
# Minimum K-loop length, in tile_k steps over the per-group output channels,
# for the 128x128 tile on an ungrouped pointwise (1x1) problem. With one or
# two steps the loop is too short to amortize the larger tile, and the 64x64
# tile measured as fast or faster on some large grids too; from three steps
# up the 128x128 tile won on every large-grid problem measured.
_GFX950_DGRAD_POINTWISE_LARGE_MIN_K_STEPS = 3
# Wide-C guard for the 128x128 tile on an ungrouped pointwise problem. With
# cpg above _GFX950_DGRAD_POINTWISE_WIDE_CPG and a K loop of at most
# _GFX950_DGRAD_POINTWISE_WIDE_MAX_K_STEPS tile_k steps, the 128x128 tile
# measured slower than the 64x64 default on several grids under 8 workgroups
# per CU (2048 on gfx950's 256 CUs) -- for example the ResNet-50 stage-3 1x1
# at N=64 -- with no grid-size or tail rule that separates its wins from its
# losses there. From 8 workgroups per CU up, at cpg <= 768, or with a longer
# K loop it was at parity or faster. Fitted on a refit cohort over C in
# 128..3072, K in 192..2048, N in 8..256 and 7x7..56x56 images, then checked
# on a hold-out cohort, same-session against the unmodified tree; the
# measurements live outside the tree.
_GFX950_DGRAD_POINTWISE_WIDE_CPG = 768
_GFX950_DGRAD_POINTWISE_WIDE_MAX_K_STEPS = 8
_GFX950_DGRAD_POINTWISE_WIDE_MIN_CTAS = 2048
# No tile_k 32 entry. With kpg % 64 == 32, tile_k 32 would keep the
# tap-outer K loop (tile_k 64 straddles filter taps and takes the flat folded
# loop) at twice the K-loop trip count. In back-to-back launches of the same
# problem it was often faster than the 64x64x64 default, but that advantage
# relied on the operands staying cache-resident between launches: with the
# caches flushed before every launch it lost on the geomean in every problem
# class tried (kpg 32..288, 1x1..7x7, 1..300 groups, short and long
# reductions), and ran below the unmodified tree on many problems where the
# default tile never did. Measured same-session against the unmodified tree;
# the measurements live outside the tree.


def _gfx950_dgrad_tile(req: ConvGroupedRequest) -> tuple[int, ...]:
    """Tile table for the gfx950 dgrad candidate (see the constants above).

    Strided problems keep the default tile: they run the runtime tilde record
    path, which the table was not tuned on. Stride-1 problems pick
      - for ungrouped pointwise (1x1) problems whose K loop is at most two
        tile_k steps (kpg <= 128), the 64x64 default: the loop is too short
        for the 128x128 tile to pay off reliably, whatever the grid;
      - the 128x128 tile when each group has >= 128 input channels and the
        grid it yields still has at least 1.25 workgroups per CU -- except an
        ungrouped pointwise problem with cpg > 768 and a K loop of at most
        eight tile_k steps (kpg <= 512), which needs at least 8 workgroups
        per CU (``_GFX950_DGRAD_POINTWISE_WIDE_MIN_CTAS``);
      - for the other ungrouped pointwise problems, the 64x64 default
        unchanged: that path keeps the runtime record (see
        DgradConvSpec.folds_sub_gemm_record), and every wider tile measured
        slower on small grids, which is what most of these problems have;
      - otherwise the 64x64 default, including when the per-group output
        channels are a multiple of 32 but not of 64 (see the note above
        ``_gfx950_dgrad_tile`` on why there is no tile_k 32 entry).
    """
    p = _problem(req)
    groups = max(int(p.groups), 1)
    strided = p.sH != 1 or p.sW != 1 or p.dH != 1 or p.dW != 1
    if strided:
        return _GFX950_DGRAD_TILE_DEFAULT
    tm, tn, tk = _GFX950_DGRAD_TILE_LARGE[:3]
    pointwise = p.is_pointwise and groups == 1
    if pointwise and -(-p.kpg // tk) < _GFX950_DGRAD_POINTWISE_LARGE_MIN_K_STEPS:
        return _GFX950_DGRAD_TILE_DEFAULT
    m = p.N * p.Hi * p.Wi
    ctas = -(-m // tm) * -(-p.cpg // tn) * groups
    wide_pointwise_small_grid = (
        pointwise
        and p.cpg > _GFX950_DGRAD_POINTWISE_WIDE_CPG
        and -(-p.kpg // tk) <= _GFX950_DGRAD_POINTWISE_WIDE_MAX_K_STEPS
        and ctas < _GFX950_DGRAD_POINTWISE_WIDE_MIN_CTAS
    )
    if (
        p.cpg >= tn
        and ctas >= _GFX950_DGRAD_LARGE_MIN_CTAS
        and not wide_pointwise_small_grid
    ):
        return _GFX950_DGRAD_TILE_LARGE
    return _GFX950_DGRAD_TILE_DEFAULT


def _make_gfx950_dgrad_candidate() -> KernelCandidate:
    """Backward-data conv for gfx950, tile chosen by :func:`_gfx950_dgrad_tile`.

    Pins ``epilogue="default"`` and ``split_k=1``. dgrad already dispatches its
    epilogue internally on ``needs_atomic`` (stride > 1 gives more than one
    sub-GEMM, which forces the atomic store regardless of split-K), so the
    epilogue knob does not carry the same meaning it does for wgrad. Split-K is
    left at 1 rather than auto-resolved: the CK formula wgrad uses keys on its
    lopsided ``N*Ho*Wo`` reduction, and dgrad's ``Y*X*K`` is not that shape.

    Output contract (pre-existing, tracked as an open issue): when the stride
    exceeds the filter extent (for example 1x1, stride 2, pad 0) some dX
    pixels are reached by no filter tap. The kernel does not write them and
    ``needs_atomic`` is False there, so the caller must pass a zeroed dX.

    No gfx942 or gfx1250 counterpart yet. gfx942 lacks the 32x32x16 atom, and
    the gfx1250 wave32 path is exercised through the sweep driver but has no
    dispatch-level dual-engine test of its own.
    """
    name = "implicit_gemm_conv_dgrad"
    spec_id = "igemm_conv_dgrad_tile_table"
    algorithm = "implicit_gemm_dgrad"

    def _tile(req: ConvGroupedRequest):
        return _gfx950_dgrad_tile(req)

    def _build_instance_spec(req: ConvGroupedRequest) -> DgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return DgradConvSpec(
            problem=_problem(req),
            name=name,
            lds_k_outer=_dgrad_lds_k_outer(req, wtmn),
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            split_k=1,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx950(req):
            return False, f"gfx950 candidate requires arch=gfx950 (got {req.arch!r})"
        if req.direction != "dgrad":
            return False, f"candidate handles 'dgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _dgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="dgrad",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue=_epilogue_for(req),
            lds_k_outer=_dgrad_lds_k_outer(req, wtmn),
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=1,
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_DGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_dgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx950 depthwise dgrad candidate (windowed direct kernel, cpg = kpg = 1)
# ---------------------------------------------------------------------------

# Widest block_w kept whole; wider rows are tiled at about this many columns.
_DW_DGRAD_FULL_ROW_W = 16
_DW_DGRAD_TILE_W = 8
# H is halved while the grid holds fewer waves than this (and the chunk keeps
# at least two filter heights of rows, so the dY halo re-read stays bounded).
_DW_DGRAD_TARGET_WAVES = 1000
# Channels one block spans at most (block_waves * 64 * ch_per_lane).
_DW_DGRAD_MAX_BLOCK_CH = 256
# Statically unrolled FMAs per lane the heuristic allows (compile time).
_DW_DGRAD_UNROLL_BUDGET = 1 << 14


@dataclass(frozen=True)
class ConvDepthwiseDgradSpec:
    """Selected spec for the windowed depthwise dgrad candidate.

    Wraps the instance spec, so the selection is directly buildable and the
    grid comes from the same object the kernel was built from.
    """

    instance: DirectDepthwiseDgradWindowedSpec
    arch: str
    direction: str = "dgrad"
    # dX is written by direct buffer stores; read by dispatch explanations.
    epilogue: str = "direct"

    def kernel_name(self) -> str:
        return self.instance.kernel_name()


def _is_depthwise_dgrad(req: ConvGroupedRequest) -> tuple[bool, str]:
    """Shape gate of the windowed depthwise dgrad kernel."""
    if req.Di is not None:
        return False, "depthwise dgrad candidate is 2D only"
    if int(req.G) != int(req.C) or int(req.G) != int(req.K):
        return (
            False,
            f"requires depthwise G == C == K (got G={req.G}, C={req.C}, K={req.K})",
        )
    if int(req.stride_h) != 1 or int(req.stride_w) != 1:
        return False, f"requires stride 1 (got {req.stride_h}x{req.stride_w})"
    if int(req.dilation_h) != 1 or int(req.dilation_w) != 1:
        return False, "requires dilation 1"
    if int(req.pad_h) != int(req.pad_w):
        return False, f"requires pad_h == pad_w (got {req.pad_h}, {req.pad_w})"
    return True, "ok"


def _dw_dgrad_win_spec(req: ConvGroupedRequest) -> DirectDepthwiseDgradWindowedSpec:
    """Pick the windowed depthwise dgrad knobs for ``req``.

    - ``dot2`` pairs filter taps on the packed dot unit; it pays for KW >= 5,
      where the kernel is bound on the FMA issue rate.
    - ``ch_per_lane = 2`` (dword channel pairs) for the memory-bound small
      filters when there are enough channels to fill the wider block.
    - ``block_w`` keeps a row of up to 16 columns whole, else tiles it at
      about 8 columns, which is a divisor of W whenever one is near.
    - ``block_waves`` covers the channels with at most 256 per block (so 4
      waves of single channels or 2 waves of channel pairs).
    - H is halved until the grid holds enough waves, bounded by the halo and
      by the unroll budget.
    """
    C, H, W, KH, KW = int(req.C), int(req.Hi), int(req.Wi), int(req.Y), int(req.X)
    dot2 = KW >= 5
    cpl = 2 if (not dot2 and C % 2 == 0 and C >= 128) else 1
    waves = min(_DW_DGRAD_MAX_BLOCK_CH // (64 * cpl), -(-C // (64 * cpl)))
    if W <= _DW_DGRAD_FULL_ROW_W:
        block_w = W
    else:
        block_w = -(-W // -(-W // _DW_DGRAD_TILE_W))
    blocks_per_tile = -(-W // block_w) * -(-C // (64 * waves * cpl)) * int(req.N)

    rows = H
    while (
        blocks_per_tile * -(-H // rows) * waves < _DW_DGRAD_TARGET_WAVES
        and -(-rows // 2) >= 2 * KH
    ):
        rows = -(-rows // 2)

    def over_budget() -> bool:
        return rows * KH * KW * block_w * cpl > _DW_DGRAD_UNROLL_BUDGET

    # Shrink rows down to about one filter height first; past that (large
    # filters) shrink the larger of rows and block_w, which keeps the block's
    # dY footprint (rows + KH - 1) x (block_w + KW - 1) smallest per output.
    while over_budget() and -(-rows // 2) >= KH:
        rows = -(-rows // 2)
    while over_budget() and (rows > 1 or block_w > 1):
        if block_w > rows:
            block_w = -(-block_w // 2)
        else:
            rows = -(-rows // 2)
    problem = DirectConvProblem(
        N=int(req.N),
        H=H,
        W=W,
        groups=C,
        cpg=1,
        kpg=1,
        KH=KH,
        KW=KW,
        PAD=int(req.pad_h),
        stride=1,
        dtype=req.dtype.lower(),
    )
    return DirectDepthwiseDgradWindowedSpec(
        problem=problem,
        name="direct_depthwise_dgrad_win",
        block_w=block_w,
        block_waves=waves,
        ch_per_lane=cpl,
        block_h=0 if rows >= H else rows,
        dot2=dot2,
    )


def _make_gfx950_depthwise_dgrad_candidate() -> KernelCandidate:
    """Depthwise (cpg = kpg = 1) stride-1 dgrad on gfx950: windowed direct kernel.

    The implicit-GEMM dgrad candidate rejects every cpg = 1 request, so the
    depthwise requests this one declines (strides, dilation, asymmetric
    padding, 3D, and tensors past the kernel's 1 GiB sentinel range) have no
    gfx950 dispatch path, as before this candidate existed.

    The kernel takes only the six-argument ABI prefix (A = dY, B = W, D = dX
    and their byte sizes); see ``CONV_GROUPED_ABI_VERSION``.
    """
    name = "direct_depthwise_dgrad_win"
    spec_id = "direct_dw_dgrad_windowed"
    algorithm = "direct_depthwise_dgrad"

    def support(req: OperatorRequest) -> tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx950(req):
            return False, f"gfx950 candidate requires arch=gfx950 (got {req.arch!r})"
        if req.direction != "dgrad":
            return False, f"candidate handles 'dgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _is_depthwise_dgrad(req)
        if not ok:
            return False, why
        spec = _dw_dgrad_win_spec(req)
        ok, why = is_valid_depthwise_dgrad_win_spec(spec, arch=req.arch)
        if not ok:
            return False, why
        _gx, gy, gz = spec.grid()
        if max(gy, gz) > _MAX_GRID_DIM_Z:
            return False, f"grid y/z ({gy}, {gz}) exceeds {_MAX_GRID_DIM_Z}"
        return True, "ok"

    def select(req: OperatorRequest) -> ConvDepthwiseDgradSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        return ConvDepthwiseDgradSpec(instance=_dw_dgrad_win_spec(req), arch=req.arch)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_DGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=5,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, _req: spec.instance.grid(),
        block=lambda spec: (spec.instance.threads_per_block, 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, arch: build_direct_depthwise_dgrad_windowed(
            spec.instance, arch=arch
        ),
    )
    return candidate


# ---------------------------------------------------------------------------
# gfx950 grouped dgrad: direct-MFMA pipeline (weight pre-pass + streaming fprop)
# ---------------------------------------------------------------------------
#
# dX = transposed fprop of dY with a flipped, channel-swapped copy of W.  The
# pipeline is a weight pre-pass kernel (W -> W_T workspace) followed by the
# direct streaming MFMA kernel run on (dY, W_T); see
# ``kernels.common.conv_direct_grouped.plan_direct_mfma_dgrad``, which owns the
# stage list, grids and workspace sizes.  The selected spec exposes that plan
# through :meth:`ConvGroupedDirectDgradSpec.launch_plan`, the same way the wgrad
# two-stage path exposes its scratch through ``to_wgrad_spec().two_stage``: the
# candidate's ``grid``/``block`` describe the main kernel, the plan describes
# everything a launcher needs around it.
#
# Eligibility is the measured win region over the igemm candidate: grouped,
# stride 1, cpg and kpg multiples of 4 up to 32, square 'same'-padded filters
# up to 7x7 (the structural region, :func:`_direct_dgrad_shape_errors`), minus
# the corners where the selected main kernel measured slower than the igemm
# candidate (:func:`_direct_dgrad_policy_errors`). Everything else keeps the
# igemm candidate, which stays registered as the fallback.
#
# The policy was fitted on a same-session sweep of randomly drawn shapes over
# the structural region (channel pairs, filter sizes, image sizes from 3 to
# 64, 2 to 64 groups, batch sizes spanning ~150 to ~25000 igemm workgroups)
# and checked on a disjoint hold-out draw; the measurements live outside the
# tree. Its terms, in the order they are tested:
#
# * the 4c row (cpg == kpg == 4, 1x1/3x3, groups % 16 == 0) runs one wave per
#   16 groups, so it needs a grid of at least _DIRECT_DGRAD_4C_MIN_GRID
#   workgroups; below that the generic kernel takes the shape;
# * a main kernel whose fused weight fragments exceed their register budget
#   falls back to the weight pre-pass pipeline. That pipeline only pays for
#   its extra launch and workspace traffic with full 16-wide K atoms on the
#   wide-output side (cpg > 20 needs kpg % 16 == 0, filled column strips and
#   a minimum size), or, with narrower outputs, above a size floor;
# * the fused kernel maps output columns onto 16-wide MFMA M strips: images
#   W <= 3 or H*W <= 16 leave most of the strip empty, and so does W <= 5 with
#   cpg > 20; wide-output narrow reductions (cpg > 20, kpg <= 10) also need
#   the strips at least _DIRECT_DGRAD_MIN_STRIP_FILL full;
# * 1x1 filters with a wide reduction (kpg >= 16) are igemm's best case
#   (a plain GEMM with a full K): direct only wins below 32 channels, on
#   strips wider than 6 columns, above a size floor;
# * small problems (fewer than _DIRECT_DGRAD_SMALL_TILES igemm workgroups)
#   with a narrow reduction or only 4 channels need a per-class size floor.

_DIRECT_DGRAD_MAX_CPG = 32
_DIRECT_DGRAD_MAX_KPG = 32
# Largest filter of the measured region. Larger 'same'-padded filters compute
# correctly but the main kernel's per-row accumulator ring spills registers.
_DIRECT_DGRAD_MAX_FILTER = 7
# Narrow reduction: cpg > kpg with kpg <= 8 (the reduction fills at most half
# of a 16-wide MFMA K atom while each wave carries ceil(cpg/16) output tiles).
_DIRECT_DGRAD_NARROW_KPG = 8
# "Wide output": more than 20 channels per group on the dX side, i.e. two
# 16-wide MFMA output tiles per wave.
_DIRECT_DGRAD_WIDE_CPG = 20
_DIRECT_DGRAD_4C_MIN_GRID = 300
_DIRECT_DGRAD_PREPASS_MIN_TILES = 300
_DIRECT_DGRAD_PREPASS_WIDE_MIN_TILES = 280
_DIRECT_DGRAD_PREPASS_MIN_STRIP_FILL = 0.6
_DIRECT_DGRAD_PREPASS_LOW_FILL = 0.5
_DIRECT_DGRAD_MIN_STRIP_FILL = 0.6
_DIRECT_DGRAD_TINY_W = 3
_DIRECT_DGRAD_TINY_HW = 16
_DIRECT_DGRAD_WIDE_TINY_W = 5
_DIRECT_DGRAD_PW_MAX_CHANNELS = 32
_DIRECT_DGRAD_PW_MIN_TILES = 300
_DIRECT_DGRAD_PW_MIN_W = 7
_DIRECT_DGRAD_SMALL_TILES = 768
_DIRECT_DGRAD_SMALL_NARROW_WIDE_MIN_TILES = 500
_DIRECT_DGRAD_SMALL_NARROW_MIN_TILES = 400
_DIRECT_DGRAD_SMALL_4CH_MIN_TILES = 275
# Pre-pass pipeline cost model (see _direct_dgrad_prepass_cost_ratio). The
# costs are unitless: each is relative to the igemm candidate's fixed
# per-call cost, which is 1. Fitted by least squares (log-linear in the waste
# terms) on a same-session random sweep of problems that take the pre-pass
# form and pass every other gate (G 2..300, cpg/kpg 4..32, 3x3..7x7, H, W
# 3..64, N 1..256, fp16/bf16, plus the G 64..300 / cpg 24..32 / H, W 8..16
# review neighbourhood) and checked on a disjoint hold-out draw of both; the
# measurements live outside the tree.
# Weight transpose: fixed + per 2**20 weight elements.
_DIRECT_DGRAD_PREPASS_TRANSPOSE_COST = (0.566, 0.0722)
# Main kernel: fixed + GFLOP * exp(coef . (1, log strip padding, log 16-wide
# output-tile padding, partial second K atom, log N*H*W, log K-atom padding)).
_DIRECT_DGRAD_PREPASS_MAIN_FIXED_COST = 0.75
_DIRECT_DGRAD_PREPASS_MAIN_COEF = (-0.385, 0.776, 0.817, 0.576, -0.100, 1.059)
# igemm candidate: 1 + GFLOP * exp(coef . (1, log 64-wide N-tile padding)).
_DIRECT_DGRAD_PREPASS_IGEMM_COEF = (-1.177, 1.091)
# The pipeline is kept only while its predicted cost is at most this fraction
# of igemm's: the largest margin that kept every fitted problem at or above
# the unmodified tree, less a step for the main-kernel model's error band.
_DIRECT_DGRAD_PREPASS_MAX_COST_RATIO = 0.78
# Spatial policy (see _direct_dgrad_block_h). Tall images (H > 16): H tiles
# are tried from the largest (whole image) down; the first that gives the
# grid this many waves wins, else the smallest tile. Short images (H <= 16)
# take the tile with the smallest modelled cost (see _direct_dgrad_block_h).
_DIRECT_DGRAD_TILE_H = 8
_DIRECT_DGRAD_SMALL_TILE_H = 4
_DIRECT_DGRAD_MAX_UNTILED_H = 64
_DIRECT_DGRAD_SHORT_H = 16
_DIRECT_DGRAD_TARGET_WAVES = 3072
# Concurrent-wave budget of the short-image cost model for the 32-channel
# rule row (one wave per SIMD on a 256-CU gfx950). It is scaled by the rule's
# block_groups (2 for 16..31 channels, 4 below): less channel work per wave
# leaves room for more resident waves.
_DIRECT_DGRAD_SHORT_WAVE_SLOTS = 1024
# block_q: 16 by default; 32 halves the per-row halo re-load (KW - 1 columns
# per strip) when the strip is wide enough to pay for it, only when H is
# tiled (an untiled grid is already short of waves) and only while the
# grid keeps this many waves.
_DIRECT_DGRAD_BLOCK_Q = 16
_DIRECT_DGRAD_WIDE_BLOCK_Q = 32
_DIRECT_DGRAD_WIDE_MIN_WAVES = 768
# Main-kernel forms. cpg == kpg == 4 (1x1/3x3) takes the batched 4x4x4 kernel
# (variant "4c"); every other admitted shape takes the generic kernel. Both
# read W directly (fused weight transform, no pre-pass) where the spec
# validates; the generic pre-pass pipeline is the fallback past the fused
# form's register budget.
_DIRECT_DGRAD_USE_4C = True
_DIRECT_DGRAD_USE_FUSED = True
_DIRECT_DGRAD_POLICY_PREFIX = "outside the measured direct win region:"
_DIRECT_DGRAD_RULE_4C = "cpg_kpg_4"
_DIRECT_DGRAD_VARIANT_4C = "4c"
_DIRECT_DGRAD_FUSED_WPE = 4
_DIRECT_DGRAD_FUSED_WPE_MAX_VGPRS = 36


@dataclass(frozen=True)
class DirectDgradRule:
    """One row of the direct-MFMA dgrad selection table.

    Rows are tried in order; the first whose ``applies(cpg, kpg)`` holds sets
    the per-channel knobs.  ``variant`` names the main-kernel family the row
    selects; today every row is the generic ``DirectConvSpec`` kernel, and a
    channel-specialised kernel (for example a cpg == 4 batched-MFMA variant)
    slots in as a new row ahead of the generic ones without touching the
    spatial policy in :func:`_direct_dgrad_block_h`.
    """

    rule_id: str
    applies: Callable[[int, int], bool]  # (cpg, kpg) -> bool
    block_groups: int
    variant: str = "generic"


# Measured on gfx950 bf16/fp16 grouped 3x3/5x5/7x7 stride-1 shapes (see
# platform/python/rocke/examples/gfx950/conv_dgrad/
# grouped_direct_dgrad_dispatch_case_study.md for the levers swept and the
# replay commands).  The knob that matters per row is block_groups: it sets
# how many groups one workgroup streams, and the best value keeps the wider
# of the two channel counts times block_groups near a 32..64-channel slice.
# A 5x5/7x7 filter halves it (see _select_direct_dgrad_spec): the per-row
# accumulator ring is KH deep, so fewer groups per workgroup keep the
# register footprint and the grid size in balance. fold_k32 is taken
# whenever kpg allows it.
GFX950_DIRECT_DGRAD_RULES: tuple[DirectDgradRule, ...] = (
    DirectDgradRule("chan_ge_32", lambda cpg, kpg: max(cpg, kpg) >= 32, block_groups=1),
    DirectDgradRule("chan_16_31", lambda cpg, kpg: max(cpg, kpg) >= 16, block_groups=2),
    DirectDgradRule("chan_le_15", lambda cpg, kpg: True, block_groups=4),
)


@dataclass(frozen=True)
class ConvGroupedDirectDgradSpec:
    """Selected spec for the gfx950 direct-MFMA grouped dgrad candidate."""

    direction: str  # always "dgrad"
    block_q: int
    block_groups: int
    block_h: int
    waves_q: int
    waves_k: int
    runtime_k_loop: bool
    fold_k32: bool
    dtype: str
    arch: str
    rule_id: str
    variant: str = "generic"
    # Single-kernel form: the main kernel reads the original W with flipped,
    # k<->c transposed addressing (no weight pre-pass, no workspace);
    # ``weights_lds`` stages the slice through LDS and transpose reads.
    fused_weights: bool = False
    weights_lds: bool = False
    waves_per_eu: int = 0
    name: str = "rocke_conv_grouped_direct_dgrad"

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            self.direction,
            self.dtype,
            self.variant,
            f"bq{self.block_q}",
            f"bg{self.block_groups}",
            f"bh{self.block_h}",
            f"wq{self.waves_q}",
            f"wk{self.waves_k}",
            "rk" if self.runtime_k_loop else "",
            "k32" if self.fold_k32 else "",
            ("fwl" if self.weights_lds else "fw") if self.fused_weights else "",
            f"we{self.waves_per_eu}" if self.waves_per_eu else "",
        )

    def to_direct_problem(self, req: ConvGroupedRequest) -> DirectConvProblem:
        return _direct_dgrad_problem(req)

    def to_fprop_spec(
        self, problem: DirectConvProblem
    ) -> DirectConvSpec | DirectConv4cSpec:
        """The main (transposed-fprop) kernel spec for ``problem``."""
        if self.variant == _DIRECT_DGRAD_VARIANT_4C:
            return make_dgrad_4c_spec(
                problem,
                block_q=self.block_q,
                block_groups=self.block_groups,
                dgrad_fused_weights=self.fused_weights,
                dgrad_weights_lds=self.weights_lds,
            )
        return make_dgrad_fprop_spec(
            problem,
            block_q=self.block_q,
            block_groups=self.block_groups,
            block_h=self.block_h,
            waves_q=self.waves_q,
            waves_k=self.waves_k,
            runtime_k_loop=self.runtime_k_loop,
            fold_k32=self.fold_k32,
            dgrad_fused_weights=self.fused_weights,
            dgrad_weights_lds=self.weights_lds,
            waves_per_eu=self.waves_per_eu,
        )

    def launch_plan(self, req: ConvGroupedRequest) -> DirectMfmaDgradPlan:
        """Every kernel launch, grid and workspace buffer of the pipeline."""
        problem = self.to_direct_problem(req)
        return plan_direct_mfma_dgrad(problem, self.to_fprop_spec(problem))


def _direct_dgrad_problem(req: ConvGroupedRequest) -> DirectConvProblem:
    p = _problem(req)
    return DirectConvProblem(
        N=int(p.N),
        H=int(p.Hi),
        W=int(p.Wi),
        groups=int(p.groups),
        cpg=int(p.cpg),
        kpg=int(p.kpg),
        KH=int(p.Y),
        KW=int(p.X),
        PAD=int(p.pH),
        stride=1,
        dtype=req.dtype.lower(),
    )


def _direct_dgrad_shape_errors(req: ConvGroupedRequest) -> list[str]:
    """Request-level reasons the direct-MFMA dgrad pipeline cannot run ``req``.

    The transposed fprop takes one square ``PAD`` and pads by ``KH - 1 - PAD``
    on both axes, so it needs a square filter with equal pads; it streams
    stride-1 rows only; and its loads/stores are 4-channel vectors on both the
    dY (kpg) and dX (cpg) side. The channel and filter-size caps keep the
    candidate inside the measured region; :func:`_direct_dgrad_policy_errors`
    then declines the corners of it where igemm measured faster.
    """
    p = _problem(req)
    errors: list[str] = []
    if p.is_3d:
        errors.append("3D convolution is not supported")
    if int(req.G) < 2:
        errors.append("grouped problems only (G >= 2); dense dgrad stays on igemm")
    if (p.sH, p.sW) != (1, 1):
        errors.append(f"stride must be 1 (got {p.sH}x{p.sW})")
    if (p.dH, p.dW) != (1, 1):
        errors.append(f"dilation must be 1 (got {p.dH}x{p.dW})")
    if p.Y != p.X:
        errors.append(f"square filters only (got {p.Y}x{p.X})")
    if p.pH != p.pW:
        errors.append(f"equal H/W padding only (got {p.pH}/{p.pW})")
    if 2 * p.pH != p.Y - 1:
        # The streaming kernel emits output rows 0..H-1 of its input height,
        # so the transposed problem must be 'same'-padded (dY and dX of equal
        # spatial size), which holds exactly when the forward pad is (Y-1)/2.
        errors.append(f"'same' padding only (2*pad == Y-1; got pad={p.pH}, Y={p.Y})")
    if p.cpg % 4 or p.kpg % 4:
        errors.append(f"cpg and kpg must be multiples of 4 (got {p.cpg}/{p.kpg})")
    if p.cpg > _DIRECT_DGRAD_MAX_CPG or p.kpg > _DIRECT_DGRAD_MAX_KPG:
        errors.append(
            f"cpg/kpg above {_DIRECT_DGRAD_MAX_CPG}/{_DIRECT_DGRAD_MAX_KPG} "
            f"(got {p.cpg}/{p.kpg}); igemm is the measured winner there"
        )
    if p.Y > _DIRECT_DGRAD_MAX_FILTER:
        errors.append(
            f"filter above {_DIRECT_DGRAD_MAX_FILTER}x{_DIRECT_DGRAD_MAX_FILTER} "
            f"(got {p.Y}x{p.X}); outside the measured region, main kernel spills"
        )
    return errors


def _direct_dgrad_policy_errors(
    req: ConvGroupedRequest, spec: ConvGroupedDirectDgradSpec
) -> list[str]:
    """Measured-loss corners of the selected direct spec (see the notes above).

    Every reason starts with ``_DIRECT_DGRAD_POLICY_PREFIX`` so a caller can
    tell a measured-performance decline from a structural one.

    ``fill`` is always measured against ``_DIRECT_DGRAD_BLOCK_Q``-wide column
    strips, whatever ``block_q`` the spec selected: the thresholds were fitted
    on that measure, so the reasons quote it as a fixed-width strip fill.
    """
    if spec.variant == _DIRECT_DGRAD_VARIANT_4C:
        return []  # the 4c row is only selected above its grid floor
    p = _direct_dgrad_problem(req)
    cpg, kpg, Y, H, W = p.cpg, p.kpg, p.KH, p.H, p.W
    tiles = _direct_dgrad_igemm_tiles(req)
    fill = W / (-(-W // _DIRECT_DGRAD_BLOCK_Q) * _DIRECT_DGRAD_BLOCK_Q)
    narrow = cpg > kpg and kpg <= _DIRECT_DGRAD_NARROW_KPG
    wide = cpg > _DIRECT_DGRAD_WIDE_CPG
    why = ""
    if not spec.fused_weights:
        if wide:
            if kpg % 16 or fill < _DIRECT_DGRAD_PREPASS_MIN_STRIP_FILL:
                why = (
                    f"pre-pass pipeline (fused weights over budget) with cpg {cpg} "
                    f"needs kpg % 16 == 0 and column strips >= "
                    f"{_DIRECT_DGRAD_PREPASS_MIN_STRIP_FILL} full (kpg {kpg}, "
                    f"W={W} fills {fill:.2f})"
                )
            elif tiles < _DIRECT_DGRAD_PREPASS_WIDE_MIN_TILES:
                why = (
                    f"pre-pass pipeline needs >= {_DIRECT_DGRAD_PREPASS_WIDE_MIN_TILES} "
                    f"igemm workgroups (got {tiles})"
                )
        elif tiles < _DIRECT_DGRAD_PREPASS_MIN_TILES:
            why = (
                f"pre-pass pipeline needs >= {_DIRECT_DGRAD_PREPASS_MIN_TILES} "
                f"igemm workgroups (got {tiles})"
            )
        elif cpg >= 16 and kpg % 16 and fill < _DIRECT_DGRAD_PREPASS_LOW_FILL:
            why = (
                f"pre-pass pipeline with cpg {cpg}, kpg {kpg} on W={W} "
                f"(strip fill {fill:.2f} < {_DIRECT_DGRAD_PREPASS_LOW_FILL})"
            )
        if not why:
            ratio = _direct_dgrad_prepass_cost_ratio(p, spec)
            if ratio > _DIRECT_DGRAD_PREPASS_MAX_COST_RATIO:
                why = (
                    f"pre-pass pipeline not amortized: predicted transpose + main "
                    f"cost is {ratio:.2f} of igemm's (limit "
                    f"{_DIRECT_DGRAD_PREPASS_MAX_COST_RATIO})"
                )
    elif W <= _DIRECT_DGRAD_TINY_W or H * W <= _DIRECT_DGRAD_TINY_HW:
        why = f"tiny image {H}x{W}: the 16-wide output strips stay mostly empty"
    elif wide and W <= _DIRECT_DGRAD_WIDE_TINY_W:
        why = f"cpg {cpg} on W={W}: two output tiles per wave on a mostly empty strip"
    elif wide and min(cpg, kpg) <= 10 and fill < _DIRECT_DGRAD_MIN_STRIP_FILL:
        why = (
            f"cpg {cpg} with kpg {kpg}: W={W} fills only {fill:.2f} of "
            f"{_DIRECT_DGRAD_BLOCK_Q}-wide column strips (fitted fill measure)"
        )
    elif (
        Y == 1
        and kpg >= 16
        and (
            max(cpg, kpg) >= _DIRECT_DGRAD_PW_MAX_CHANNELS
            or tiles < _DIRECT_DGRAD_PW_MIN_TILES
            or W < _DIRECT_DGRAD_PW_MIN_W
        )
    ):
        why = (
            f"1x1 with a wide reduction (kpg {kpg}, cpg {cpg}, W={W}, "
            f"{tiles} igemm workgroups) is igemm's best case"
        )
    elif Y > 1 and tiles < _DIRECT_DGRAD_SMALL_TILES:
        if narrow and wide and tiles < _DIRECT_DGRAD_SMALL_NARROW_WIDE_MIN_TILES:
            floor = _DIRECT_DGRAD_SMALL_NARROW_WIDE_MIN_TILES
        elif narrow and not wide and max(cpg, kpg) <= 10:
            floor = _DIRECT_DGRAD_SMALL_NARROW_MIN_TILES
        elif max(cpg, kpg) <= 6:
            floor = _DIRECT_DGRAD_SMALL_4CH_MIN_TILES
        else:
            floor = 0
        if tiles < floor:
            why = f"small problem: {tiles} igemm workgroups < {floor} for this class"
    return [f"{_DIRECT_DGRAD_POLICY_PREFIX} {why}"] if why else []


def _direct_dgrad_prepass_cost_ratio(
    p: DirectConvProblem, spec: ConvGroupedDirectDgradSpec
) -> float:
    """Predicted (transpose + main) / igemm cost of the pre-pass pipeline.

    The weight transpose pre-pass runs on every call and its cost follows the
    weight tensor (``K * Y * X * cpg`` elements), not the activations, so it
    is only worth paying where the main kernel beats igemm by more than it.
    Each stage is a fitted unitless model (``_DIRECT_DGRAD_PREPASS_*``, costs
    relative to igemm's fixed per-call cost):

    * transpose: linear in the weight elements;
    * main kernel: GFLOPs scaled by its waste terms -- 16-wide column strips
      over ``W`` (``block_q`` wide), 16-wide output-channel tiles over
      ``cpg``, the K-atom padding of ``kpg`` (32-deep atoms under
      ``fold_k32``) and a partial second 16-deep K atom (zero-masked lane by
      lane) -- and by the problem size ``N * H * W``;
    * igemm: GFLOPs scaled by its 64-wide N-tile padding over ``cpg``.
    """
    gflop = 2.0 * p.N * p.H * p.W * p.total_c * p.kpg * p.KH * p.KW / 1e9
    weights = p.total_k * p.KH * p.KW * p.cpg / float(1 << 20)
    strips = -(-p.W // spec.block_q) * spec.block_q / p.W
    partial_atom = 1.0 if p.kpg % 16 and p.kpg > 16 else 0.0
    k_pad = (p.kpg if p.kpg % 32 == 0 else -(-p.kpg // 16) * 16) / p.kpg
    main_x = (
        1.0,
        math.log(strips),
        math.log(-(-p.cpg // 16) * 16 / p.cpg),
        partial_atom,
        math.log(p.N * p.H * p.W),
        math.log(k_pad),
    )
    igemm_x = (1.0, math.log(-(-p.cpg // 64) * 64 / p.cpg))

    def scaled(coef: tuple[float, ...], x: tuple[float, ...]) -> float:
        return gflop * math.exp(sum(c * v for c, v in zip(coef, x)))

    tr_fixed, tr_per_weight = _DIRECT_DGRAD_PREPASS_TRANSPOSE_COST
    direct = (
        tr_fixed
        + tr_per_weight * weights
        + _DIRECT_DGRAD_PREPASS_MAIN_FIXED_COST
        + scaled(_DIRECT_DGRAD_PREPASS_MAIN_COEF, main_x)
    )
    igemm = 1.0 + scaled(_DIRECT_DGRAD_PREPASS_IGEMM_COEF, igemm_x)
    return direct / igemm


def _direct_dgrad_igemm_tiles(req: ConvGroupedRequest) -> int:
    """Workgroups the igemm dgrad candidate would launch for ``req`` (stride 1).

    ``ceil(N * Hi * Wi / tile_m) * groups``: the M tiles of each per-group GEMM
    times the groups (``cpg <= 64`` fits one N tile). Used as the work measure
    of the direct candidate's size floors.
    """
    p = _problem(req)
    m_tiles = -(-int(p.N) * int(p.Hi) * int(p.Wi) // _GFX950_TILE_M)
    return m_tiles * max(int(p.groups), 1)


def _direct_dgrad_rule(cpg: int, kpg: int) -> DirectDgradRule:
    for rule in GFX950_DIRECT_DGRAD_RULES:
        if rule.applies(cpg, kpg):
            return rule
    raise AssertionError("GFX950_DIRECT_DGRAD_RULES has no catch-all row")


def _direct_dgrad_block_h(p: DirectConvProblem, block_groups: int) -> int:
    """Spatial policy: stream the whole image height per workgroup, or tile it.

    ``block_h = 0`` streams every row of the image through one workgroup, so
    each dY row is loaded once.  Tiling re-loads ``KH - 1`` halo rows per tile
    and pays when the untiled grid is too small to fill the device.  A 5x5/7x7
    filter always takes 4-row tiles: its per-row work is long enough that the
    serial row chain, not the halo, dominates.

    Short images (``H <= 16``) take the tile (whole image, 8 or 4 rows) with
    the smallest modelled cost ``(tile rows + KH - 1) * max(slots, waves)``:
    each wave's serial row chain including its halo, times how many rounds of
    waves the grid needs once it has more than ``slots`` waves (see
    :data:`_DIRECT_DGRAD_SHORT_WAVE_SLOTS`; ``block_groups`` is the rule
    table's value).  This keeps a short image whole when its grid already
    fills the device: a 9- or 10-row image cut into 8- or 4-row tiles leaves a
    nearly empty last tile and re-loads the halo for every tile.

    Taller images: the largest tile (whole image, then 8 rows) whose grid
    reaches :data:`_DIRECT_DGRAD_TARGET_WAVES` wins; otherwise 4-row tiles.
    Very tall images are never streamed whole (the row loop is unrolled when
    ``block_h == 0``).
    """
    if p.H <= _DIRECT_DGRAD_TILE_H:
        return 0
    if p.KH >= 5:
        return _DIRECT_DGRAD_SMALL_TILE_H
    wave_columns = -(-p.Wo // _DIRECT_DGRAD_BLOCK_Q) * p.groups * p.N
    if p.H <= _DIRECT_DGRAD_SHORT_H:
        slots = _DIRECT_DGRAD_SHORT_WAVE_SLOTS * block_groups
        best_bh, best_cost = 0, None
        for bh in (0, _DIRECT_DGRAD_TILE_H, _DIRECT_DGRAD_SMALL_TILE_H):
            h_tiles = -(-p.H // bh) if bh else 1
            rows = (bh or p.H) + p.KH - 1
            cost = rows * max(slots, wave_columns * h_tiles)
            if best_cost is None or cost < best_cost:
                best_bh, best_cost = bh, cost
        return best_bh
    tiles = (
        (_DIRECT_DGRAD_TILE_H,)
        if p.H > _DIRECT_DGRAD_MAX_UNTILED_H
        else (0, _DIRECT_DGRAD_TILE_H)
    )
    for bh in tiles:
        h_tiles = -(-p.H // bh) if bh else 1
        if wave_columns * h_tiles >= _DIRECT_DGRAD_TARGET_WAVES:
            return bh
    return _DIRECT_DGRAD_SMALL_TILE_H


def _direct_dgrad_block_q(p: DirectConvProblem, block_h: int) -> int:
    """Output columns per workgroup: 16, or 32 where the wider strip pays.

    A 32-wide strip halves the ``KW - 1`` halo columns re-loaded per strip
    and the per-strip weight traffic; it needs H tiling (an untiled grid is
    already short of waves), no extra padding over 16-wide strips, enough
    channels per wave that the strip's work is not latency-bound (the wider
    channel count >= 16, or >= 8 with a 5x5/7x7 filter), and a grid that
    keeps :data:`_DIRECT_DGRAD_WIDE_MIN_WAVES` waves after halving.
    """
    wide, narrow = _DIRECT_DGRAD_WIDE_BLOCK_Q, _DIRECT_DGRAD_BLOCK_Q
    if block_h == 0:
        return narrow
    if -(-p.Wo // wide) * wide > -(-p.Wo // narrow) * narrow:
        return narrow
    chans = max(p.cpg, p.kpg)
    if not (chans >= 16 or (p.KH >= 5 and chans >= 8)):
        return narrow
    waves = -(-p.Wo // wide) * p.groups * p.N * -(-p.H // block_h)
    return wide if waves >= _DIRECT_DGRAD_WIDE_MIN_WAVES else narrow


def _direct_dgrad_has_transpose_reads(arch: str) -> bool:
    try:
        return bool(ArchTarget.from_gfx(arch).memory.has_ds_read_tr)
    except KeyError:
        return False


def _direct_dgrad_main_is_valid(
    spec: ConvGroupedDirectDgradSpec, problem: DirectConvProblem
) -> tuple[bool, str]:
    """Whether the main kernel of ``spec`` validates and fits on its arch."""
    try:
        main = spec.to_fprop_spec(problem)
        main.validate()
    except ValueError as e:
        return False, str(e)
    if isinstance(main, DirectConv4cSpec):
        return _direct_is_valid_spec_4c(main, arch=spec.arch)
    return _direct_is_valid_spec(main, arch=spec.arch)


def _select_direct_dgrad_4c_spec(
    req: ConvGroupedRequest, p: DirectConvProblem
) -> ConvGroupedDirectDgradSpec | None:
    """The 4c row: cpg == kpg == 4 on the batched 4x4x4 MFMA kernel, or None.

    One wave carries 16 groups at full MFMA width (the generic kernel's 16-wide
    atom is a quarter full in both dimensions there), and the weights are read
    in the kernel's prologue (LDS-staged transpose reads where the target has
    them), so there is no pre-pass. Knobs are the swept 4c defaults. One wave
    is one workgroup, so the row needs a grid of at least
    :data:`_DIRECT_DGRAD_4C_MIN_GRID` workgroups; smaller problems take the
    generic kernel, whose H tiling keeps the device busy.
    """
    if not (p.cpg == 4 and p.kpg == 4 and p.KH == p.KW and p.KH in (1, 3)):
        return None
    if p.groups % DGRAD_4C_DEFAULT_BLOCK_GROUPS:
        return None
    grid = (
        -(-p.W // DGRAD_4C_DEFAULT_BLOCK_Q)
        * (p.groups // DGRAD_4C_DEFAULT_BLOCK_GROUPS)
        * p.N
    )
    if grid < _DIRECT_DGRAD_4C_MIN_GRID:
        return None
    spec = ConvGroupedDirectDgradSpec(
        direction="dgrad",
        block_q=DGRAD_4C_DEFAULT_BLOCK_Q,
        block_groups=DGRAD_4C_DEFAULT_BLOCK_GROUPS,
        block_h=0,
        waves_q=1,
        waves_k=1,
        runtime_k_loop=False,
        fold_k32=False,
        dtype=req.dtype.lower(),
        arch=req.arch,
        rule_id=_DIRECT_DGRAD_RULE_4C,
        variant=_DIRECT_DGRAD_VARIANT_4C,
        fused_weights=True,
        weights_lds=_direct_dgrad_has_transpose_reads(req.arch),
    )
    return spec if _direct_dgrad_main_is_valid(spec, p)[0] else None


def _select_direct_dgrad_spec(req: ConvGroupedRequest) -> ConvGroupedDirectDgradSpec:
    p = _direct_dgrad_problem(req)
    if _DIRECT_DGRAD_USE_4C:
        spec4c = _select_direct_dgrad_4c_spec(req, p)
        if spec4c is not None:
            return spec4c
    rule = _direct_dgrad_rule(p.cpg, p.kpg)
    block_groups = rule.block_groups
    if p.KH >= 5:
        block_groups = max(1, block_groups // 2)
    while p.groups % block_groups:
        block_groups //= 2
    block_h = _direct_dgrad_block_h(p, rule.block_groups)
    base = ConvGroupedDirectDgradSpec(
        direction="dgrad",
        block_q=_direct_dgrad_block_q(p, block_h),
        block_groups=block_groups,
        block_h=block_h,
        waves_q=1,
        waves_k=1,
        runtime_k_loop=False,
        fold_k32=p.kpg % 32 == 0,
        dtype=req.dtype.lower(),
        arch=req.arch,
        rule_id=rule.rule_id,
        variant=rule.variant,
    )
    if not _DIRECT_DGRAD_USE_FUSED:
        return base
    # Single-kernel form first: the same knobs with the weight transform fused
    # into the main kernel's prologue (LDS-staged transpose reads where the
    # target has them and the staged slice fits, register gathers otherwise).
    # Its preloaded weight fragments have a register budget; past it the
    # pre-pass pipeline above remains the fallback.
    lds_opts = (
        (True, False) if _direct_dgrad_has_transpose_reads(req.arch) else (False,)
    )
    for use_lds in lds_opts:
        fused = replace(base, fused_weights=True, weights_lds=use_lds)
        if _direct_dgrad_main_is_valid(fused, p)[0]:
            wpe = _direct_dgrad_fused_waves_per_eu(fused, p)
            return replace(fused, waves_per_eu=wpe) if wpe else fused
    return base


def _direct_dgrad_fused_waves_per_eu(
    spec: ConvGroupedDirectDgradSpec, p: DirectConvProblem
) -> int:
    """``waves_per_eu`` hint for a fused-weight generic spec (0 = none).

    A small preloaded weight footprint (16-wide atoms only) lets the hint keep
    the scheduler from serialising the next-row input loads behind one reused
    register pair; see ``direct_dgrad_spec_for_problem``.
    """
    if spec.fold_k32:
        return 0
    main = spec.to_fprop_spec(p)
    if preload_weight_vgprs(main) <= _DIRECT_DGRAD_FUSED_WPE_MAX_VGPRS:
        return _DIRECT_DGRAD_FUSED_WPE
    return 0


_I32_MAX = (1 << 31) - 1
_MAX_GRID_DIM = 65535  # y and z


def _direct_dgrad_grid(
    spec: ConvGroupedDirectDgradSpec, req: OperatorRequest
) -> tuple[int, int, int]:
    """Main-kernel grid ``(q_tiles, groups / block_groups, N * h_tiles)``."""
    assert isinstance(req, ConvGroupedRequest)
    problem = _direct_dgrad_problem(req)
    return direct_mfma_dgrad_main_grid(spec.to_fprop_spec(problem))


def _direct_dgrad_block(spec: ConvGroupedDirectDgradSpec) -> tuple[int, int, int]:
    wave = ArchTarget.from_gfx(spec.arch).wave_size
    if spec.variant == _DIRECT_DGRAD_VARIANT_4C:
        # 16 groups per wave on the batched 4x4x4 atom.
        return ((spec.block_groups // 16) * wave, 1, 1)
    waves = spec.block_groups * spec.waves_q * spec.waves_k
    return (waves * wave, 1, 1)


def _make_gfx950_direct_dgrad_candidate() -> KernelCandidate:
    """Grouped backward-data conv for gfx950 on the direct-MFMA pipeline.

    Outranks the igemm dgrad candidate (priority 5 < 10) on the grouped,
    stride-1 problems it admits; :func:`_direct_dgrad_shape_errors` is the
    admitted region. ``ConvGroupedRequest.vec_size_c`` is an igemm epilogue
    hint and has no meaning here: the direct kernels store 4-channel vectors
    regardless, so the field is ignored (the explanation says so when set).
    Knobs come from :data:`GFX950_DIRECT_DGRAD_RULES` (the per-channel table),
    :func:`_direct_dgrad_block_h` and :func:`_direct_dgrad_block_q` (the
    spatial policy).
    """
    name = "direct_mfma_conv_dgrad"
    spec_id = "direct_mfma_dgrad_grouped"
    algorithm = "direct_mfma_dgrad"

    def support(req: OperatorRequest) -> tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if req.direction != "dgrad":
            return False, f"candidate handles 'dgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        errors = _direct_dgrad_shape_errors(req)
        if errors:
            return False, "; ".join(errors)
        spec = _select_direct_dgrad_spec(req)
        plan = spec.launch_plan(req)
        ok, why = _direct_dgrad_main_is_valid(spec, plan.problem)
        if not ok:
            return False, why
        errors = _direct_dgrad_policy_errors(req, spec)
        if errors:
            return False, "; ".join(errors)
        too_big = [role for role, nb in plan.buffer_bytes if nb > _I32_MAX]
        if too_big:
            return False, f"buffers {too_big} exceed the i32 byte-size kernel ABI"
        for stage in plan.stages:
            if max(stage.grid[1:]) > _MAX_GRID_DIM:
                return False, f"{stage.role} grid {stage.grid} exceeds the y/z limit"
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedDirectDgradSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        return _select_direct_dgrad_spec(req)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_DGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=5,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_direct_dgrad_grid,
        block=_direct_dgrad_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _make_gfx1250_wgrad_candidate() -> KernelCandidate:
    """Backward-weight conv for gfx1250: 32x32x32, 2x2, 16x16x32 WMMA (wave32).

    gfx1250's only fp16/bf16 atom is 16x16x32 (there is no 16x16x16). WMMA wgrad
    requires split_k=1 and the direct-store ('default') epilogue, so both are
    forced here regardless of the request. Grouped convolution is supported and
    runs grid-per-group (the kernel is validated dual-engine by
    test_gfx1250_grouped_wgrad_dual_engine). Group merging (``group_merge > 1``)
    is MFMA-only and additionally needs split_k > 1 on the two-stage path, so it
    is unreachable here on both counts: is_valid_wgrad_spec rejects it for
    wave32 and for the split_k=1 this candidate forces.
    """
    name = "implicit_gemm_conv_wgrad_gfx1250"
    spec_id = "igemm_conv_wgrad_gfx1250_32x32"
    algorithm = "implicit_gemm_wgrad_gfx1250"

    def _tile(req: ConvGroupedRequest):
        return (
            _GFX1250_TILE_M,
            _GFX1250_TILE_N,
            _GFX1250_TILE_K,
            _GFX1250_WARP_M,
            _GFX1250_WARP_N,
            _GFX1250_WARP_TILE_MN,
            _GFX1250_WARP_TILE_K,
        )

    def _build_instance_spec(req: ConvGroupedRequest) -> WgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return WgradConvSpec(
            problem=_problem(req),
            name=name,
            data=_data_spec(req),
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_m=wtmn,
            warp_tile_n=wtmn,
            warp_tile_k=wtk,
            wave_size=32,
            pipeline=_PIPELINE,
            epilogue="default",
            split_k=1,
            lds_k_outer=_wgrad_lds_k_outer(req, wtmn),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvGroupedRequest)
        if not _is_gfx1250(req):
            return False, f"gfx1250 candidate requires arch=gfx1250 (got {req.arch!r})"
        if req.direction != "wgrad":
            return False, f"candidate handles 'wgrad', got direction={req.direction!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _wgrad_is_valid_spec(_build_instance_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        return ConvGroupedSpec(
            direction="wgrad",
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            warp_m=wm,
            warp_n=wn,
            warp_tile_mn=wtmn,
            warp_tile_k=wtk,
            pipeline=_PIPELINE,
            epilogue="default",
            dtype=req.dtype.lower(),
            arch=req.arch,
            split_k=1,
            lds_k_outer=_wgrad_lds_k_outer(req, wtmn),
            name=name,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_WGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx1250",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_wgrad_grid,
        block=_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


# ---------------------------------------------------------------------------
# Registries — one per direction (different family strings)
# ---------------------------------------------------------------------------

_CONV_DIM_VOCABULARY = (
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
    # 3D depth dims (None when 2D)
    "Di",
    "Z",
    "stride_d",
    "pad_d",
    "dilation_d",
    "Do",
)

CONV_DGRAD_REGISTRY = CandidateRegistry(
    _FAMILY_DGRAD, dim_vocabulary=_CONV_DIM_VOCABULARY
)
CONV_DGRAD_REGISTRY.register(_make_gfx950_direct_dgrad_candidate())
CONV_DGRAD_REGISTRY.register(_make_gfx950_dgrad_candidate())
CONV_DGRAD_REGISTRY.register(_make_gfx950_depthwise_dgrad_candidate())

CONV_FWD_REGISTRY = CandidateRegistry(_FAMILY_FWD, dim_vocabulary=_CONV_DIM_VOCABULARY)
CONV_FWD_REGISTRY.register(_make_gfx942_fwd_candidate())
CONV_FWD_REGISTRY.register(_make_gfx950_fwd_candidate())
CONV_FWD_REGISTRY.register(_make_gfx1250_fwd_candidate())

CONV_WGRAD_REGISTRY = CandidateRegistry(
    _FAMILY_WGRAD, dim_vocabulary=_CONV_DIM_VOCABULARY
)
CONV_WGRAD_REGISTRY.register(_make_gfx942_wgrad_candidate())
CONV_WGRAD_REGISTRY.register(_make_gfx950_wgrad_candidate())
CONV_WGRAD_REGISTRY.register(_make_gfx1250_wgrad_candidate())


def _registry_for(req: ConvGroupedRequest) -> CandidateRegistry:
    if req.direction == "wgrad":
        return CONV_WGRAD_REGISTRY
    if req.direction == "dgrad":
        return CONV_DGRAD_REGISTRY
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
    if direction == "dgrad":
        return CONV_DGRAD_REGISTRY.candidates()
    return CONV_FWD_REGISTRY.candidates()


def registered_conv_grouped_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, ConvGroupedSpec], ...]:
    """Every registered grouped-conv candidate that can launch ``req``.

    Probes opt-in variants and expands each candidate's ``sweep_space``.
    Production :func:`dispatch_conv_grouped` is unchanged.
    """
    if _request_errors(req):
        return ()
    assert isinstance(req, ConvGroupedRequest)
    return _registry_for(req).combos(req)


def conv_grouped_sweep_space(req: OperatorRequest) -> Sequence[ConvGroupedSpec]:
    if _request_errors(req):
        return ()
    assert isinstance(req, ConvGroupedRequest)
    return _registry_for(req).sweep_space(req, spec_key=lambda spec: spec.kernel_name())


def dispatch_conv_grouped_all(
    req: ConvGroupedRequest,
) -> Tuple[DispatchResult, ...]:
    """Every eligible grouped-conv kernel for ``req``, including opt-in variants."""
    if _request_errors(req):
        return ()
    return _registry_for(req).dispatch_all(req, kernel_id=_kernel_id)


def dispatch_conv_grouped(
    req: ConvGroupedRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the appropriate grouped conv candidate (fwd or wgrad) for ``req``."""
    registry = _registry_for(req)
    candidate = registry.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    explanation = [
        f"selected {candidate.name} ({req.direction}) on {req.arch}",
        f"algorithm={candidate.algorithm}",
        f"spec_id={candidate.spec_id}",
    ]
    if isinstance(spec, ConvGroupedDirectDgradSpec):
        plan = spec.launch_plan(req)
        explanation += [
            (
                f"rule={spec.rule_id} variant={spec.variant} "
                f"block_q={spec.block_q} block_groups={spec.block_groups} "
                f"block_h={spec.block_h} fold_k32={spec.fold_k32}"
            ),
            (
                "pipeline="
                + "+".join(stage.role for stage in plan.stages)
                + f" workspace_bytes={plan.workspace_bytes}"
            ),
        ]
        if req.vec_size_c is not None:
            explanation.append(
                f"vec_size_c={req.vec_size_c} ignored (igemm epilogue hint)"
            )
    else:
        explanation.append(f"epilogue={spec.epilogue} (vec_size_c={_vec_size_c(req)})")
    explanation += [
        f"spec_hash={kid.spec_hash}",
        f"request_hash={kid.request_hash}",
    ]
    if req.direction == "wgrad":
        split_k, _two_stage, requested = _resolve_wgrad_split_k(spec, _problem(req))
        if split_k != requested:
            explanation.append(
                f"split_k {requested} -> {split_k}: the two-stage scratch would "
                f"exceed the {_MAX_WGRAD_WS_BYTES}-byte i32 limit, so the "
                f"plain store is used instead"
            )
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=tuple(explanation),
    )
