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
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from rocke.core.arch import ArchTarget
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
    # fwd only. Folds Gm depthwise groups into one GEMM tile so the group index
    # becomes the fastest-varying factor of GemmN and GemmK, restoring Gm-wide
    # vector access on a shape whose per-group channel count is 1. Default 1 is
    # the unmerged path, byte-for-byte.
    group_merge: int = 1
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
        #   group_merge: changes the launch grid, the B load's addressing and
        #     mask, and every operand's vector width. Same cache-collision
        #     argument as the one above -- untagged, two Gm degrees resolve to
        #     one symbol and the second silently runs the first's binary.
        if self.direction == "fwd" and self.group_merge != 1:
            parts.append(f"gm{self.group_merge}")
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
            group_merge=self.group_merge,
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
    # Launch over the MERGED dims, which is what the emitted kernel indexes.
    # Under group_merge=Gm a workgroup owns Gm conv groups at once: the GEMM-N
    # extent grows by Gm (kpg -> kpg*Gm) and the number of group-slices shrinks
    # by Gm. Taking either off the true problem is host/device divergence --
    # the true N_gemm is 1 on depthwise, so gn would stay 1 while the kernel
    # writes Gm columns, and z would launch Gm x redundant CTAs racing on the
    # same outputs. Both expressions are the identity at Gm == 1.
    gmerge = spec.group_merge if spec.direction == "fwd" else 1
    gm = (p.M + spec.tile_m - 1) // spec.tile_m
    gn = (p.N_gemm * gmerge + spec.tile_n - 1) // spec.tile_n
    # grid_order "NM": x=n-tiles, y=m-tiles — mirrors the fwd conv manifest
    return (gn, gm, p.groups // gmerge)


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
# gfx950 depthwise forward candidate with merged groups
# ---------------------------------------------------------------------------

# Degrees the emitter's gate accepts, ascending. The policy below evaluates a
# cost at each and takes the argmin, and ties resolve to the first entry, so the
# order *is* the tie-break: prefer the smaller degree.
_FWD_MERGE_DEGREES = (1, 2, 4, 8, 16, 32, 64)

# Fitted constants of the merge-degree cost model (see _fwd_merge_cost for the
# mechanism each one weighs). Reviewers reasonably ask where eight magic numbers
# came from, so, briefly:
#
# WHERE THEY COME FROM. The functional form is hand-derived from the four things
# that move when Gm changes -- K padding, A-load vector width, cache-line
# utilisation, and the CTA's A working set. Only the *weights* are fitted: a
# random search over a measured corpus of several hundred depthwise shapes, each
# swept across every admissible degree at the tile this file pins. The target is
# the geometric mean of the realised fraction -- what share of a shape's own
# measured best the single modelled pick keeps -- so every quantity in the fit is
# a ratio inside one shape.
#
# WHY A SEARCH AND NOT A SOLVE. The model's output is an argmin over seven
# degrees, so the objective is piecewise constant in these constants: nudging one
# changes nothing at all until it flips some shape's winner. There is no gradient
# to descend, anywhere. Brute force is the method, not a shortcut.
#
# WHY EIGHT. Each term was admitted by cross-validation against a *nested* model
# that pins the constant at the value making its term vanish, so a win is
# attributable to the term added rather than to a reparametrisation. All eight
# earn their place; the A-load vector term is the largest single contributor and
# removing it is clearly visible out of sample.
#
# WHY TWO OF THEM LOOK NEGLIGIBLE. B_ALOAD_ISSUE and W_MEMORY are ~1e-2 and
# ~3e-4, which reads as inert until you notice they are weighed against
# D_CTA_FIXED at ~2e6. The cost has no physical unit -- only ratios between
# degrees are ever compared -- so the scale is arbitrary and absolute magnitude
# says nothing about influence. Do not "simplify" a small-looking constant away;
# check it with a nested fit instead.
#
# SCOPE. gfx950, fp16/bf16, forward, at the tile this file pins. Re-fit before
# pointing this at another arch, element size, or tile. The corpus, the fitting
# driver and the scoring harness are in
# ``platform/python/rocke/examples/gfx950/conv_fwd/merge_degree_model/``;
# ``score_shipped.py`` there re-derives every pick from this very function and
# fails if it has drifted from the reference model it was fitted as.

# Cost of issuing the A load, relative to one K step of MFMA: charged per step
# and divided by the achieved vector width, so narrow loads cost more issue.
_FWD_MERGE_B_ALOAD_ISSUE = 0.014175
# Weight of the DRAM bucket against the compute bucket -- how many cost units one
# byte of A traffic is worth.
_FWD_MERGE_W_MEMORY = 0.00030657
# How much of the available W-direction halo reuse the cache actually delivers.
# 0 would be none, 1 perfect; fitted below 1/2, i.e. real but partial.
_FWD_MERGE_R_REUSE = 0.4249
# Per-CTA overhead that no degree can avoid (launch, prologue, epilogue). Large
# relative to the degree-dependent terms, which is exactly why halving the CTA
# count is such a strong pull toward merging.
_FWD_MERGE_D_CTA_FIXED = 1.9548e6
# Effective CTAs resident machine-wide. Divides the CTA count into waves, so it
# is what makes "this shape has too little parallelism left to merge further"
# expressible at all.
_FWD_MERGE_CUPAR = 558.42
# Footprint scale of the brake, in elements: the CTA A working set at which the
# penalty has doubled before exponentiation. The only term that argues against
# merging.
_FWD_MERGE_C_FOOTPRINT = 130.64
# Exponent on cache-line utilisation. Above 1, so a half-used line costs rather
# more than twice a full one -- partial lines also burn request slots, not just
# bandwidth.
_FWD_MERGE_A_UTIL = 2.7279
# Exponent on the brake. Above 1, so the working-set penalty is superlinear and
# can eventually overturn the halved CTA count; at 1 it never does.
_FWD_MERGE_Q_BRAKE = 1.6812

# Machine constants the model is written against, not fitted: a 128-byte cache
# line and a 16-byte (dwordx4) widest load.
_FWD_MERGE_LINE_BYTES = 128
_FWD_MERGE_VEC_BYTES = 16


def _pick_group_merge(req: ConvGroupedRequest, tile_m: int, tile_n: int) -> int:
    """Merge degree worth paying for on this request, or 1.

    Unwraps the request and defers to :func:`fwd_group_merge_for_geometry`,
    which is where the policy actually lives.
    """
    groups = int(req.G)
    y, x, stride = int(req.Y), int(req.X), int(req.stride_w)
    try:
        m = int(_problem(req).M)
    except Exception:
        # A degenerate shape is _request_errors' business, not this function's;
        # m <= 0 lets the model fall back to the geometry-free admissibility set.
        m = 0
    # support() already rejects everything but fp16/bf16 (see _request_errors),
    # so the element size is 2 by construction. Passed rather than assumed so
    # the model stays re-fittable against a wider dtype corpus.
    return fwd_group_merge_for_geometry(
        groups, m, tile_m, tile_n, y=y, x=x, stride=stride, esize=2
    )


def _fwd_merge_admissible(groups: int, tile_n: int, y: int, x: int) -> tuple:
    """Degrees the emitter's gate will actually build for this geometry.

    Mirrors ``fwd_group_merge_available``: a power of two that divides ``groups``
    (a workgroup must not own a partial group) and fits the N tile (the GEMM-N
    extent under merge *is* ``Gm``). Pointwise is excluded by the gate outright,
    so there is nothing to choose and the caller gets ``(1,)``.
    """
    if y * x <= 1:
        return (1,)
    return tuple(
        gm for gm in _FWD_MERGE_DEGREES if gm <= tile_n and groups % gm == 0
    )


def _fwd_merge_cost(
    gm: int,
    *,
    groups: int,
    m: int,
    y: int,
    x: int,
    stride: int,
    tile_m: int,
    tile_k: int,
    esize: int,
) -> float:
    """Modelled time for one degree, in arbitrary units -- only ratios are used.

    ``Gm`` consecutive conv groups fold into one GEMM with ``Gm`` on GemmN *and*
    GemmK, so ``M`` is untouched and the launch becomes

        CTAs(Gm) = ceil(M/tile_m) * G/Gm          -- halves per doubling
        Kpad(Gm) = tile_k * ceil(Y*X*Gm/tile_k)   -- grows ~linearly

    Four things move with the degree, three of them in its favour:

    ``pad``  K-padding waste, already folded into ``Kpad``. At ``Y*X = 9`` the
        per-group K extent is 9 of a 64-wide tile, so ``Gm = 1`` wastes 7.1x and
        by ``Gm = 8`` it is 1.1x. At ``Y*X = 961`` there is nothing to recover,
        which is the first reason huge filters never want to merge.
    ``vw``   A-load vector width, ``min(Gm, 16/esize)``. k's innermost field is
        the channel, so merging makes ``Gm`` channels contiguous in NHWC.
        Saturates at dwordx4 -- but the gain does *not* stop there, which is why
        an earlier revision capping at ``dwordx4/itemsize = 8`` was wrong.
    ``u``    A-load cache-line utilisation, ``min(1, Gm*esize/128)``. This is
        what keeps 8 -> 16 -> 32 paying on large shapes after ``vw`` saturates.
    ``fp``   The brake: the CTA's A working set is ``tile_m * Y*X*Gm`` elements,
        linear in the degree *and* in the filter area.

    Occupancy is the obvious candidate for that brake and it is the wrong one.
    Shapes with tens of thousands of CTAs, where the machine cannot run dry,
    still turn over at 8..16, and they turn over sooner the larger ``Y*X`` is --
    across the corpus whether the top degree still pays is a clean monotone
    function of ``Y*X`` alone, essentially independent of how many CTAs the
    shape has. So the penalty multiplies the *whole* per-CTA cost rather than
    the DRAM bucket only: an oversized K working set costs issue slots and
    occupancy, not just traffic. Charged against traffic alone it never has the
    leverage to overturn a halved CTA count, and the model merges to 64 almost
    everywhere.

    Occupancy is still in here, as ``ceil(CTAs/CUPAR)``, and still does the work
    it was doing before: an ``N = 1``, ``7x7`` shape has M in a single tile, so
    ``grid.z`` *is* its parallelism and merging hands the machine back to itself
    empty.
    """
    ctas = -(-m // tile_m) * (groups // gm)
    kpad = tile_k * -(-(y * x * gm) // tile_k)

    vw = min(gm, _FWD_MERGE_VEC_BYTES // esize)
    u = min(1.0, gm * esize / _FWD_MERGE_LINE_BYTES)
    footprint = tile_m * y * x * gm * esize
    # Available W-direction reuse: adjacent output columns read input columns
    # `stride` apart, so X-1 of the X taps land on data a neighbour already
    # pulled at stride 1, and about half that at stride 2.
    reuse = (1.0 + (x - 1) / stride) ** _FWD_MERGE_R_REUSE

    compute = kpad * (1.0 + _FWD_MERGE_B_ALOAD_ISSUE / vw)
    memory = (
        kpad
        * tile_m
        * esize
        * _FWD_MERGE_W_MEMORY
        / max(u**_FWD_MERGE_A_UTIL * reuse, 1e-30)
    )

    brake = (1.0 + footprint / _FWD_MERGE_C_FOOTPRINT) ** _FWD_MERGE_Q_BRAKE
    per_cta = (compute + memory) * brake + _FWD_MERGE_D_CTA_FIXED
    return math.ceil(ctas / _FWD_MERGE_CUPAR) * per_cta


def fwd_group_merge_for_geometry(
    groups: int,
    m: int,
    tile_m: int,
    tile_n: int,
    *,
    y: int,
    x: int,
    stride: int,
    esize: int = 2,
    tile_k: int = _GFX950_TILE_K,
) -> int:
    """:func:`_pick_group_merge` with the request unwrapped to plain integers.

    Evaluate :func:`_fwd_merge_cost` at every admissible degree and return the
    argmin; ties go to the smaller degree, which is the safer side on shapes the
    model cannot see (odd ``Ho*Wo`` tails, say).

    Split out so the sweep benchmark can centre its merge-degree window on the
    *same* policy dispatch will apply, without constructing a
    ``ConvGroupedRequest`` it does not otherwise need. Keeping one implementation
    matters more than the indirection costs: the benchmark prunes the degree axis
    around this pick, so a second copy drifting from this one would silently
    sweep a window that no longer brackets what ships.

    ``m <= 0`` means "geometry unknown": every cost then shares the same CTA
    count, which leaves the degree-dependent part of the model intact, so the
    pick degrades rather than failing.

    This replaced a three-cap rule (tile fit, a flat ``Gm <= 32`` ceiling, and an
    occupancy floor) that was fitted by hand to a corpus of a few dozen shapes.
    Held out from the fit, the model keeps a materially larger share of each
    shape's own measured best than that rule did, and leaves materially fewer
    shapes far short of it. The measured comparison, the per-shape curves and the
    reasoning behind the functional form are in
    ``platform/python/rocke/examples/gfx950/conv_fwd/fwd_merged_groups_case_study.md``;
    the corpus, the fitting driver and the scoring harness are beside it in
    ``platform/python/rocke/examples/gfx950/conv_fwd/merge_degree_model/``.
    """
    cands = _fwd_merge_admissible(groups, tile_n, y, x)
    best, best_cost = 1, None
    for gm in cands:
        c = _fwd_merge_cost(
            gm,
            groups=groups,
            m=max(int(m), 1),
            y=y,
            x=x,
            stride=max(int(stride), 1),
            tile_m=max(int(tile_m), 1),
            tile_k=tile_k,
            esize=esize,
        )
        if best_cost is None or c < best_cost * (1.0 - 1e-9):
            best, best_cost = gm, c
    return best


def _make_gfx950_fwd_dw_merged_candidate() -> KernelCandidate:
    """Depthwise forward conv for gfx950 with ``group_merge`` folded in.

    Same tile as :func:`_make_gfx950_fwd_candidate`; the only difference is the
    merge degree. It sits at a higher priority so it wins whenever it admits,
    and falls through to the plain fwd candidate when it does not -- which is
    the whole gating story, since ``support()`` defers to the same
    ``is_valid_spec`` the builder calls. A shape with no admissible degree
    (``G = 3`` in the corpus: no power of two divides it) picks ``Gm = 1``, and
    is rejected here rather than shipping a merged spec that is merged in name
    only.
    """
    # Distinct from the plain fwd candidate: candidate names key the registry
    # (duplicates are a hard error) and also prefix the emitted symbol.
    name = "implicit_gemm_conv_dw_merged"
    spec_id = "igemm_conv_fwd_dw_merged_64x64"
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
            # Forced, not derived. Merging makes the store vector Gm-wide, and
            # is_valid_spec rejects the direct epilogue once that exceeds 1.
            # _epilogue_for reads the *true* C/K, which on depthwise happens to
            # land on cshuffle anyway -- relying on that coincidence is how the
            # dispatcher would come to hand the builder a spec it then rejects.
            epilogue="cshuffle",
            groups=int(req.G),
            group_merge=_pick_group_merge(req, tm, tn),
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
        tm, tn, *_rest = _tile(req)
        gm = _pick_group_merge(req, tm, tn)
        if gm == 1:
            return False, f"no admissible merge degree for groups={int(req.G)}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        # Depthwise, power-of-two degree, wave64, non-pointwise, no async_dma:
        # all of it lives in fwd_group_merge_available, reached through here.
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
            epilogue="cshuffle",
            dtype=req.dtype.lower(),
            arch=req.arch,
            name=name,
            group_merge=_pick_group_merge(req, tm, tn),
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_FWD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        # Ahead of the plain fwd candidate's 10. Lower number wins:
        # CandidateRegistry.candidates() sorts ascending and select() takes
        # ranked[0], so "higher priority" here is a smaller integer.
        priority=5,
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


def _make_gfx950_dgrad_candidate() -> KernelCandidate:
    """Backward-data conv for gfx950: 64x64x64, 2x2, 32x32x16 MFMA.

    Pins ``epilogue="default"`` and ``split_k=1``. dgrad already dispatches its
    epilogue internally on ``needs_atomic`` (stride > 1 gives more than one
    sub-GEMM, which forces the atomic store regardless of split-K), so the
    epilogue knob does not carry the same meaning it does for wgrad. Split-K is
    left at 1 rather than auto-resolved: the CK formula wgrad uses keys on its
    lopsided ``N*Ho*Wo`` reduction, and dgrad's ``Y*X*K`` is not that shape.

    No gfx942 or gfx1250 counterpart yet. gfx942 lacks the 32x32x16 atom, and
    the gfx1250 wave32 path is exercised through the sweep driver but has no
    dispatch-level dual-engine test of its own.
    """
    name = "implicit_gemm_conv_dgrad"
    spec_id = "igemm_conv_dgrad_64x64"
    algorithm = "implicit_gemm_dgrad"

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
CONV_DGRAD_REGISTRY.register(_make_gfx950_dgrad_candidate())

CONV_FWD_REGISTRY = CandidateRegistry(_FAMILY_FWD, dim_vocabulary=_CONV_DIM_VOCABULARY)
CONV_FWD_REGISTRY.register(_make_gfx942_fwd_candidate())
CONV_FWD_REGISTRY.register(_make_gfx950_fwd_candidate())
CONV_FWD_REGISTRY.register(_make_gfx950_fwd_dw_merged_candidate())
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
        f"epilogue={spec.epilogue} (vec_size_c={_vec_size_c(req)})",
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
