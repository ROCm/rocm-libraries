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
gfx950 dgrad picks among the binaries of the AOT caches (every selection
names one cached kernel): the direct-MFMA candidate (4c and fused generic
kernels, ``ConvDirectDgradSpec``) takes stride-1 problems with small channel
groups, the depthwise candidates (windowed kernel at stride 1, row-streaming
kernel otherwise) take cpg = kpg = 1, and the implicit-GEMM candidate (tile
table and dY halo pick, ``ConvGroupedSpec``) the rest. The direct policy is
:mod:`dispatch.grouped_conv_dgrad_direct`.

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

import functools
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from rocke.core.arch import ArchTarget
from dispatch import grouped_conv_dgrad_direct as _ddp
from kernels.common.conv_direct_grouped import (
    DIRECT_DGRAD_SINGLE_STEP_ABI,
    DirectConvProblem,
    direct_launch_geometry,
    make_direct_dgrad_spec,
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
    _GROUP_MERGE_DEGREES,
    is_valid_wgrad_spec as _wgrad_is_valid_spec,
    wgrad_atomic_epilogue_available as _wgrad_atomic_epilogue_available,
)
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    dgrad_lds_bytes as _dgrad_lds_bytes,
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

# There is deliberately no tuned constant for depthwise wgrad group merging:
# :func:`_wgrad_merge_degree` is the kernel's own admissibility rule and nothing
# more. See its docstring for why no knob survived the fit.

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

# ConvGroupedSpec fields that only dgrad sets; each is left out of the spec
# hash at its default (see _kernel_id).
_DGRAD_SPEC_FIELDS = (
    "vector_size_a",
    "vector_size_b",
    "vector_size_c",
    "uniform_k_decode",
    "tap_outer_k",
    "dy_halo",
    "dy_halo_setprio",
    "dy_halo_kouter_pad",
    "halo_max_wo",
)


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
    group_merge: int = 1  # wgrad only, depthwise only
    name: str = "rocke_conv_grouped"
    # dgrad only: the fields that name one AOT dgrad binary beyond the tile
    # (KernelIdentity's), forwarded verbatim by to_dgrad_spec. Explicit load
    # / store widths (0 = the builder's default), the K loop
    # (uniform_k_decode on the tilde binary, tap_outer_k on the stride-1
    # one) and the dY halo knobs with the widest Wo the binary serves. All
    # default off and are left out of the spec hash at their defaults
    # (_DGRAD_SPEC_FIELDS), so every spec without them keeps its hash.
    vector_size_a: int = 0
    vector_size_b: int = 0
    vector_size_c: int = 0
    uniform_k_decode: bool = False
    tap_outer_k: bool = False
    dy_halo: int = 0
    dy_halo_setprio: int = 0
    dy_halo_kouter_pad: int = 0
    halo_max_wo: int = 0

    def __post_init__(self) -> None:
        if self.direction != "dgrad":
            set_fields = [f for f in _DGRAD_SPEC_FIELDS if getattr(self, f)]
            if set_fields:
                raise ValueError(
                    f"{', '.join(set_fields)} only apply to direction='dgrad' "
                    f"(got direction={self.direction!r})"
                )

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
        #   group_merge: a different GEMM entirely -- Gm groups folded into one
        #     tile, with a diagonal mask in the epilogue.
        if self.direction in ("wgrad", "dgrad") and self.lds_k_outer:
            parts.append("kouter")
        if self.direction == "wgrad" and self.group_merge > 1:
            parts.append(f"gm{self.group_merge}")
        # The dgrad binary fields each change the body (or, for halo_max_wo,
        # the LDS shape), with the tags of DgradConvSpec.kernel_name().
        if self.direction == "dgrad":
            vecs = (self.vector_size_a, self.vector_size_b, self.vector_size_c)
            if any(vecs):
                parts.append("v" + "x".join(str(v) for v in vecs))
            for flag, tag in (
                (self.uniform_k_decode, "ukd"),
                (self.tap_outer_k, "tapk"),
                (self.dy_halo, f"halo{self.dy_halo}"),
                (self.dy_halo_setprio, f"hprio{self.dy_halo_setprio}"),
                (self.dy_halo_kouter_pad, f"hkp{self.dy_halo_kouter_pad}"),
                (self.halo_max_wo, f"hwo{self.halo_max_wo}"),
            ):
                if flag:
                    parts.append(tag)
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
            group_merge=self.group_merge,
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
            vector_size_a=self.vector_size_a or None,
            vector_size_b=self.vector_size_b or None,
            vector_size_c=self.vector_size_c or None,
            uniform_k_decode=self.uniform_k_decode,
            tap_outer_k=self.tap_outer_k,
            dy_halo=self.dy_halo,
            dy_halo_setprio=self.dy_halo_setprio,
            dy_halo_kouter_pad=self.dy_halo_kouter_pad,
            halo_max_wo=self.halo_max_wo,
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
    # Group merging folds Gm groups into one GEMM: the tile covers Gm times the
    # per-group dims, and there are Gm times fewer of them on the grid. Both
    # the CK formula and the gridDim.z clamp below are sized off the *launched*
    # geometry, so feed them the merged numbers -- sizing a degree off the
    # unmerged grid is how the pre-merge formula came to ask for a z extent the
    # launch could not express.
    gm = max(1, int(spec.group_merge))
    grid_groups = max(1, p.groups // gm)
    if spec.split_k != -1:
        requested = spec.split_k
    else:
        from rocke.helpers.split_k import select_split_k_wgrad

        requested = select_split_k_wgrad(
            wg_M=wg_M * gm,
            wg_N=wg_N * gm,
            wg_K=p.N * p.Ho * p.Wo * (p.Do if p.is_3d else 1),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            arch=spec.arch,
            groups=grid_groups,
            block_size=_block(spec)[0],
        ).split_k
    # The helper already keeps groups*split_k inside the z limit on the auto
    # path; this clamp still has to run for an explicitly requested split_k.
    split_k = max(1, min(requested, _MAX_GRID_DIM_Z // grid_groups))
    # When the packed 16-bit atomic cannot represent this problem, the f32
    # scratch path is the only way to keep split_k > 1. Delegate rather than
    # re-deriving the rule: a local copy is exactly how the dispatcher came to
    # hand the builder a spec the builder then rejected.
    # vector_size_c is None here by construction: to_wgrad_spec never sets it,
    # so the epilogue derives the store width from the channel dims.
    _atomic_ok, _ = _wgrad_atomic_epilogue_available(p, spec.dtype.lower(), None)
    # A merged tile cannot use the packed-atomic split-K epilogue at all: that
    # epilogue has no way to drop an off-diagonal group pair, so it would
    # accumulate garbage into a live dW element. Route any merged split_k > 1
    # through two-stage, whose scratch atomic carries the diagonal mask. This
    # mirrors the kernel-side gate in ``wgrad_group_merge_available``; keeping
    # the dispatcher's answer inside it is what stops select() from handing the
    # builder a spec the builder then rejects.
    #
    # As written today the ``gm > 1`` term is subsumed and never flips the
    # result: dispatch admits only fp16/bf16, merging is depthwise-only, and a
    # depthwise cpg == 1 forces a store-vector width of 1, so _atomic_ok is
    # already False on every shape that can merge. It is kept because it is the
    # term that stays true if any of those three premises moves -- an fp32 dW,
    # or merging extended past depthwise -- and in that world it is the only
    # thing between a merged tile and a silently wrong gradient. Deriving the
    # invariant from the merge degree directly is also simply what the rule
    # means. ``test_merged_split_k_never_takes_the_packed_atomic`` asserts the
    # subsumption explicitly, so it reports when it lifts.
    two_stage = ((not _atomic_ok) or gm > 1) and split_k > 1
    # The scratch carries no split_k factor, but it does carry the replica
    # factor -- R copies of dW per group. Use the same R this module hands the
    # spec in to_wgrad_spec, or the cap bounds an allocation nobody makes.
    #
    # Deliberately *unmerged*: this mirrors wgrad_two_stage_workspace_nbytes,
    # which sizes off WgradConvSpec.wg_M/wg_N -- the true per-group dims, kept
    # separate from grid_M/grid_N for exactly this reason. The scratch is R
    # copies of dW, and merging does not change how much dW there is; it only
    # changes how many CTAs write it. Scaling these by gm would overstate the
    # allocation by gm**2 and start refusing two-stage on problems that fit.
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
    # Group merging folds Gm groups into one GEMM, so the tile covers Gm times
    # the per-group dims and the grid carries Gm times fewer groups. Both
    # factors have to land, and together: scaling the dims without shrinking z
    # would launch Gm times the work, and shrinking z without scaling the dims
    # would leave (Gm-1)/Gm of the merged GEMM uncomputed.
    gm = max(1, int(spec.group_merge))
    wg_M = kpg * gm  # merged output channels
    wg_N = spatial * cpg * gm  # merged filter spatial × input channel
    gx = (wg_N + spec.tile_n - 1) // spec.tile_n
    gy = (wg_M + spec.tile_m - 1) // spec.tile_m
    split_k, _two_stage, _requested = _resolve_wgrad_split_k(spec, p)
    # The merged-group index rides on block_id_z alongside the K-slice: z =
    # (groups/Gm)*split_k, decoded in-kernel as group = z // split_k, slice =
    # z % split_k. For G==1 this reduces to (gx, gy, split_k); for split_k==1
    # to (gx, gy, groups/Gm); for Gm==1 to the unmerged groups*split_k.
    return (gx, gy, (p.groups // gm) * split_k)


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
    the variant rests on does not drive it -- and measurement agreed: the
    narrow tile was a win on some depthwise shapes and a loss on others, with
    no predictor separating them.

    The predictor turned out not to be the tile at all but the merge degree:
    see :func:`_wgrad_merge_degree`, which fixes the depthwise scalar run by
    packing several groups into one tile at the shipped width, rather than by
    narrowing the tile to cut per-thread B work. So depthwise keeps the shipped
    geometry and changes only how many groups share it.
    """
    p = _problem(req)
    cpg = p.C // max(int(p.groups), 1)
    if cpg == 1 and p.kpg == 1 and p.groups > 1:
        return False
    _va, vec_b, _vc = WgradConvSpec.default_vector_sizes(cpg, p.kpg, req.dtype.lower())
    return vec_b == 1


def _wgrad_merge_degree(req: "ConvGroupedRequest") -> int:
    """``group_merge`` for a gfx950 wgrad request: as much as the gate allows.

    Depthwise wgrad is the degenerate case of the wgrad GEMM. ``wg_M = kpg = 1``
    and ``wg_N = Y*X*cpg = Y*X``, so the tile covers a handful of filter taps
    and the GEMM free axis is one element per group -- every dY/X load is
    scalar. Merging ``Gm`` groups into one tile makes that run ``Gm`` elements
    long, which is the whole reason merging helps: ``Gm`` *is* the load vector
    width.

    The policy is the kernel's own admissibility rule and nothing else, which
    is worth being explicit about because it is the result of a fit rather than
    an assumption. Measured across a depthwise wgrad corpus, merging to the
    maximum admissible degree at the shipped tile was the best degree available
    at that tile on four shapes in five, and within a couple of percent of the
    best on all but one of the rest. Three candidate knobs were fitted on top
    and all three were rejected:

    * **A tile_n ladder.** Widening the tile is the only way a large filter
      merges at all -- at the shipped 64 a 7x7 stage cannot merge, since
      ``49 * 2 > 64``. It does raise the typical speedup, but it buys that with
      a tail of shapes that regress by up to half, concentrated on the widest
      rung. The exchange is bad for a dispatch default, which has to serve
      shapes nobody measured: the upside is bounded and the downside is not.
    * **A vectorisation cap** (stop merging once the run reaches the widest
      buffer load). It only ever binds together with the ladder, since at a
      fixed tile the ``spatial * Gm <= tile_n`` bound bites first on nearly
      every shape.
    * **A grid-occupancy floor** (refuse to merge past some number of surviving
      CTAs). Merging divides the group axis of the grid by ``Gm``, but split-K
      is resolved *after* merging and simply refills it, so the floor mostly
      declines free speedup.

    What is left needs no tuned constant, and that is the point: the rule is
    the gate, so it cannot drift away from the gate.

    Mirrors the three request-dependent clauses of
    :func:`kernels.common.conv_implicit_gemm_wgrad.wgrad_group_merge_available`
    -- the degree ladder, group divisibility, and ``grid_N <= tile_n``. The
    remaining clauses of the gate (depthwise, wave64, split-K/two-stage) are
    properties of the candidate rather than of the request and are checked by
    the caller. ``grid_M <= tile_m`` is not re-derived: depthwise is
    ``kpg == 1`` so ``grid_M == Gm``, and the ladder already stops at
    64 == ``tile_m``.

    This re-derives rather than probing the gate, because the gate takes a
    built :class:`WgradConvSpec` and building one per degree to answer a
    closed-form question is a lot of work to learn ``spatial * Gm <= tile_n``.
    The cost is that the two can drift, which
    ``TestWgradMergeDegree.test_every_choice_is_admissible`` is there to catch:
    it runs the choice back through the real gate across a grid of shapes, so a
    clause that moves in the kernel and not here fails there.
    """
    p = _problem(req)
    groups = max(int(p.groups), 1)
    # Merging is depthwise-only in the kernel, and pointless with one group.
    if p.cpg != 1 or p.kpg != 1 or groups <= 1:
        return 1
    spatial = p.Y * p.X * ((p.Z or 1) if p.is_3d else 1)
    best = 1
    for gm in _GROUP_MERGE_DEGREES:
        if gm <= groups and groups % gm == 0 and spatial * gm <= _GFX950_TILE_N:
            best = gm
    return best


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

    def _group_merge(req: ConvGroupedRequest) -> int:
        # The scalar-B variant pins its own narrower tile, which the degree
        # rule's tile bound was not derived against; rather than re-derive it
        # for a second geometry, that path keeps the unmerged behaviour it was
        # tuned with. The two do not overlap in practice -- _wgrad_b_is_scalar
        # excludes depthwise, and merging is depthwise-only -- so this is
        # belt-and-braces, not a policy choice.
        if _wgrad_b_is_scalar(req):
            return 1
        return _wgrad_merge_degree(req)

    def _build_instance_spec(req: ConvGroupedRequest) -> WgradConvSpec:
        tm, tn, tk, wm, wn, wtmn, wtk = _tile(req)
        _ep, _sk = _wgrad_grouped_overrides(req)
        return WgradConvSpec(
            group_merge=_group_merge(req),
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
            group_merge=_group_merge(req),
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


# ---------------------------------------------------------------------------
# gfx950 implicit-GEMM dgrad: tile table and dY halo pick
# ---------------------------------------------------------------------------
#
# Every pick names one binary of the AOT dgrad grid
# (benchmarks.common.kernel_sweep): a cached tile with explicit vector widths
# (A along kpg, B and D along cpg; D scalar under the default epilogue), the
# K-outer B layout the grid builds for that B width, and one of the grid's
# capabilities:
#
# * stride 1 / dilation 1 with kpg % tile_k == 0: the stride-1 binary (its
#   sub-GEMM record rebuilt from the kernargs) with the tap-outer K loop;
# * stride 1, square 3x3 / 5x5 / 7x7 filters, Ho == Hi, Wo == Wi up to the
#   widest cached halo row, kpg % 64 == 0: the dY halo mode of that loop on
#   one of the three halo tiles, sized for the smallest cached halo_max_wo
#   covering Wo (:func:`_gfx950_dgrad_halo_pick`);
# * everything else: the tilde binary (stride and dilation in the record
#   buffer) with the per-tile reduction decode.
#
# tests/dispatch/test_grouped_conv_dgrad_dispatch_aot.py checks that every
# pick is a cached identity the cache offers for the problem.

# Shape-keyed tiles: (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn,
# warp_tile_k). The default 64x64 tile keeps the grid large, which is what
# small-M and few-channel problems need; the 128x128 tile halves the operand
# traffic per MFMA but only pays once the grid still fills the device, from
# _GFX950_DGRAD_LARGE_MIN_CTAS workgroups (1.25 per CU on gfx950). Strided
# problems keep the default (the tilde path the table was not tuned on).
_GFX950_DGRAD_TILE_DEFAULT = (64, 64, 64, 2, 2, 32, 16)
_GFX950_DGRAD_TILE_LARGE = (128, 128, 64, 2, 2, 16, 32)
_GFX950_DGRAD_LARGE_MIN_CTAS = 320
# Small channel groups: a 64-wide N tile over cpg <= 32 output channels is
# mostly empty, so narrow N tiles (16 or 32 wide) with a taller M tile keep
# the MFMAs full, on 256-row tiles with 32-wide K tiles where the reduction
# is short (kpg <= 32). Only on grids that still fill the device with the taller M tile
# (else the default, whose 64-row M tiles give more workgroups). Stride-1
# problems; the direct candidate takes the small-channel problems in its
# region first.
_GFX950_DGRAD_TILE_CPG16 = (128, 16, 64, 4, 1, 16, 32)
_GFX950_DGRAD_TILE_CPG16_SHORT_K = (256, 16, 32, 2, 1, 16, 32)
_GFX950_DGRAD_TILE_CPG32 = (128, 32, 64, 4, 2, 16, 32)
_GFX950_DGRAD_TILE_CPG32_SHORT_K = (256, 32, 32, 4, 1, 16, 32)
# No tile_k 32 entry beside the default: with kpg % 64 == 32 the default
# tile takes the tilde binary (its tile_k straddles taps), and tile_k 32
# only won with cache-resident operands.


def _gfx950_dgrad_tile(req: ConvGroupedRequest) -> tuple[int, ...]:
    """Tile table of the gfx950 dgrad candidate (see the constants above)."""
    p = _problem(req)
    groups = max(int(p.groups), 1)
    if p.sH != 1 or p.sW != 1 or p.dH != 1 or p.dW != 1:
        return _GFX950_DGRAD_TILE_DEFAULT
    m = p.N * p.Hi * p.Wi
    if p.cpg <= 32:
        short_k = p.kpg <= 32
        if p.cpg <= 16:
            tile = (
                _GFX950_DGRAD_TILE_CPG16_SHORT_K
                if short_k
                else _GFX950_DGRAD_TILE_CPG16
            )
        else:
            tile = (
                _GFX950_DGRAD_TILE_CPG32_SHORT_K
                if short_k
                else _GFX950_DGRAD_TILE_CPG32
            )
        if -(-m // tile[0]) * groups >= _GFX950_DGRAD_LARGE_MIN_CTAS:
            return tile
        return _GFX950_DGRAD_TILE_DEFAULT
    tm, tn = _GFX950_DGRAD_TILE_LARGE[:2]
    ctas = -(-m // tm) * -(-p.cpg // tn) * groups
    if p.cpg >= tn and ctas >= _GFX950_DGRAD_LARGE_MIN_CTAS:
        return _GFX950_DGRAD_TILE_LARGE
    return _GFX950_DGRAD_TILE_DEFAULT


# dY halo reuse picks (DgradConvSpec.dy_halo) for gfx950 stride-1 dgrad whose
# output has the input's size, a square 3x3..7x7 filter and kpg % 64 == 0.
# The staged halo with the double-buffered B tile (dy_halo=2) was faster
# than the tile table once the tile matched the grid, the LDS occupancy and
# the filter's dY reuse, except where the rules below keep the table. Three
# tiles, all 32x32x16 MFMA with a K-outer B tile:
#   - 64x64, 2x2 waves, for small grids;
#   - 128x64, 4x1 waves (each wave reads every B column), for mid-size grids;
#   - 256x64, 4x1 waves (half the B traffic per output pixel), for large
#     grids, and on mid-size grids when it keeps the 128x64 tile's LDS
#     occupancy, still has _GFX950_DGRAD_HALO_LARGE_MIN_WGS workgroups and
#     either fills more than one workgroup per CU or has a per-tap halo of at
#     least _GFX950_DGRAD_HALO_LARGE_MIN_HALO_TILES 128x64 tiles.
# The 256x64 tile is limited to 3x3 filters: with more taps its per-tap row
# masks lift the register count to one wave per SIMD.
#
# A cached halo binary is sized for halo_max_wo (the smallest
# _GFX950_DGRAD_HALO_MAX_WO bucket covering Wo), not for the problem's Wo,
# so every occupancy below is the LDS charge of that binary.
_GFX950_DGRAD_HALO_TILE_SMALL = (64, 64, 64, 2, 2, 32, 16)
_GFX950_DGRAD_HALO_TILE_MID = (128, 64, 64, 4, 1, 32, 16)
_GFX950_DGRAD_HALO_TILE_LARGE = (256, 64, 64, 4, 1, 32, 16)
# The cached halo filters and halo_max_wo buckets (CACHE_DGRAD_HALO_* of the
# AOT grid).
_GFX950_DGRAD_HALO_FILTERS = (3, 5, 7)
_GFX950_DGRAD_HALO_MAX_WO = (8, 16, 32, 48, 64, 96, 128)
# 128x64 workgroup counts that bound the mid band: below the first the 64x64
# tile won, from the second on the 256x64 tile won.
_GFX950_DGRAD_HALO_MID_MIN_WGS = 192
_GFX950_DGRAD_HALO_LARGE_GRID_WGS = 768
_GFX950_DGRAD_HALO_LARGE_MIN_WGS = 128
_GFX950_DGRAD_HALO_LARGE_MAX_TAPS = 9
_GFX950_DGRAD_HALO_LARGE_MIN_HALO_TILES = 2
# s_setprio around each tap's MFMA block on the 4x1-wave tiles (not on the
# 64x64 tile).
_GFX950_DGRAD_HALO_SETPRIO = 1
# K-outer B row pad for the 4x1-wave tiles, only where it keeps the
# LDS-limited workgroups per CU (a padded tile that drops one was slower) --
# except on the 256x64 tile with a dY tensor of at least
# _GFX950_DGRAD_HALO_PAD_BIG_DY_BYTES: there the tile ran faster at one
# workgroup per CU than at two, so the pad is taken even where it costs the
# second one. (The bucketed halo_max_wo puts many images on that occupancy
# step; smaller dY tensors kept the second workgroup's win.)
_GFX950_DGRAD_HALO_KOUTER_PAD = 32
_GFX950_DGRAD_HALO_PAD_BIG_DY_BYTES = 128 << 20
# A halo tile that fits one workgroup per CU in LDS exposes the halo load of
# every channel chunk; once the grid needs more than one workgroup per CU the
# 256x64 tile falls back to 128x64 if that keeps two per CU, and every other
# pick to the tile table.
_GFX950_NUM_CUS = 256
# The tile table's 128x128 tile is kept over the 128x64 halo tile at fewer
# than three LDS-limited workgroups per CU where the 128-wide N tiles cover a
# group's input channels exactly in one or two tiles.
_GFX950_DGRAD_HALO_MID_OVER_LARGE_MIN_OCC = 3
_GFX950_DGRAD_HALO_KEEP_LARGE_MAX_N_TILES = 2
# 4x1-wave tiles take filters of at most 7x7 taps.
_GFX950_DGRAD_HALO_4X1_MAX_TAPS = 49
# dY reuse floor of a halo tile: filter taps x tile rows over the staged rows
# (the tile plus its (Y-1)*Wo + (X-1) halo pixels). Below it the tile table
# is kept: the 4x1-wave tiles below _GFX950_DGRAD_HALO_MIN_REUSE_4X1, the
# 64x64 tile below _GFX950_DGRAD_HALO_MIN_REUSE_2X2 once the grid needs more
# than one workgroup per CU, below ..._SMALL_GRID otherwise.
_GFX950_DGRAD_HALO_MIN_REUSE_4X1 = 1.3
_GFX950_DGRAD_HALO_MIN_REUSE_2X2 = 1.0
_GFX950_DGRAD_HALO_MIN_REUSE_2X2_SMALL_GRID = 0.7


def _dgrad_vector_sizes(req: ConvGroupedRequest) -> tuple[int, int, int]:
    """``(a, b, c)`` load / store widths of a dgrad pick: the widest of
    (8, 4, 2, 1) dividing kpg (A, dY) and cpg (B, W; D, dX), D scalar under
    the default epilogue. A ``vec_size_c`` override caps the cpg widths."""
    p = _problem(req)

    def widest(n: int, cap: int = 8) -> int:
        return next(v for v in (8, 4, 2, 1) if v <= cap and n % v == 0)

    vec_c = _vec_size_c(req)
    vec_b = widest(p.cpg, vec_c if req.vec_size_c is not None and vec_c > 1 else 8)
    return widest(p.kpg), vec_b, (vec_b if vec_c > 1 else 1)


def _dgrad_halo_max_wo(wo: int) -> int:
    """The smallest cached halo_max_wo bucket covering ``wo`` (0: none)."""
    return next((b for b in _GFX950_DGRAD_HALO_MAX_WO if b >= wo), 0)


def _dgrad_instance_spec(
    req: ConvGroupedRequest, tile: tuple[int, ...], **halo
) -> DgradConvSpec:
    """The gfx950 dgrad instance spec of ``tile`` (and halo knobs ``halo``)
    for ``req``: the binary the pick names, built on the real problem."""
    return _dgrad_dispatch_spec(req, tile, halo).to_dgrad_spec(_problem(req))


def _dgrad_dispatch_spec(
    req: ConvGroupedRequest, tile: tuple[int, ...], halo: dict
) -> ConvGroupedSpec:
    """The dispatcher spec of a gfx950 dgrad pick (see the notes above)."""
    p = _problem(req)
    tm, tn, tk, wm, wn, wtmn, wtk = tile
    vec_a, vec_b, vec_c = _dgrad_vector_sizes(req)
    target = ArchTarget.from_gfx(req.arch)
    stride1 = (p.sH, p.sW, p.dH, p.dW) == (1, 1, 1, 1) and not p.is_3d
    tap_outer = stride1 and target.wave_size == 64 and p.kpg % tk == 0
    epilogue = "cshuffle" if vec_c > 1 else "default"
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
        epilogue=epilogue,
        lds_k_outer=DgradConvSpec.default_lds_k_outer(
            arch=req.arch,
            dtype_b=req.dtype.lower(),
            warp_tile_n=wtmn,
            cpg=vec_b,
            wave_size=target.wave_size,
            pipeline=_PIPELINE,
        ),
        dtype=req.dtype.lower(),
        arch=req.arch,
        split_k=1,
        name="implicit_gemm_conv_dgrad",
        vector_size_a=vec_a,
        vector_size_b=vec_b,
        vector_size_c=vec_c,
        uniform_k_decode=(
            not tap_outer
            and DgradConvSpec.default_uniform_k_decode(
                arch=req.arch, wave_size=target.wave_size, pipeline=_PIPELINE
            )
        ),
        tap_outer_k=tap_outer,
        **halo,
    )


def _gfx950_dgrad_halo_pick(
    req: ConvGroupedRequest,
) -> tuple[tuple[int, ...], dict[str, int]] | None:
    """``(tile, halo knobs)`` of the dY halo pick, or None (see above).

    None where no cached halo binary serves the request (filter, Wo bucket,
    kpg, layout), where no halo tile is valid for it (the instance validator
    decides eligibility and the LDS fit of the bucketed binary), where a
    4x1-wave tile would get a filter of more than
    ``_GFX950_DGRAD_HALO_4X1_MAX_TAPS`` taps, where the tile table's 128x128
    tile is kept (``_GFX950_DGRAD_HALO_MID_OVER_LARGE_MIN_OCC``) or where
    the picked tile's dY reuse is below its floor. Occupancy is the
    LDS-limited workgroup count per CU of the bucketed binary.
    """
    p = _problem(req)
    if p.is_3d or (p.sH, p.sW, p.dH, p.dW) != (1, 1, 1, 1):
        return None
    if p.Y != p.X or p.Y not in _GFX950_DGRAD_HALO_FILTERS:
        return None
    if (p.Ho, p.Wo) != (p.Hi, p.Wi) or p.kpg % 64:
        return None
    max_wo = _dgrad_halo_max_wo(p.Wo)
    if not max_wo:
        return None
    vec_a, vec_b, _vec_c = _dgrad_vector_sizes(req)
    if vec_a != 8 or vec_b < 2:
        return None  # the halo binaries are K-outer with 8-wide dY loads
    lds_cap = ArchTarget.from_gfx(req.arch).lds_capacity_bytes

    def _valid(spec: DgradConvSpec) -> bool:
        return _dgrad_is_valid_spec(spec, arch=req.arch)[0]

    def _wgs(tile: tuple[int, ...]) -> int:
        tm, tn = tile[0], tile[1]
        return -(-(p.N * p.Hi * p.Wi) // tm) * -(-p.cpg // tn) * max(int(p.groups), 1)

    small, mid, large = (
        _GFX950_DGRAD_HALO_TILE_SMALL,
        _GFX950_DGRAD_HALO_TILE_MID,
        _GFX950_DGRAD_HALO_TILE_LARGE,
    )
    base = {"dy_halo": 2, "halo_max_wo": max_wo}
    specs = {t: _dgrad_instance_spec(req, t, **base) for t in (small, mid, large)}
    ok = {t: _valid(s) for t, s in specs.items()}
    if not ok[small] and not ok[mid]:
        return None
    occ = {t: lds_cap // _dgrad_lds_bytes(s) for t, s in specs.items()}
    # dY pixels each tap's halo adds to the staged tile.
    halo_rows = (p.Y - 1) * p.Wo + (p.X - 1)
    if (_wgs(mid) < _GFX950_DGRAD_HALO_MID_MIN_WGS or not ok[mid]) and ok[small]:
        tile = small
    elif (
        ok[large]
        and p.Y * p.X <= _GFX950_DGRAD_HALO_LARGE_MAX_TAPS
        and (
            _wgs(mid) >= _GFX950_DGRAD_HALO_LARGE_GRID_WGS
            or (
                occ[large] >= occ[mid]
                and _wgs(large) >= _GFX950_DGRAD_HALO_LARGE_MIN_WGS
                and (
                    _wgs(large) > _GFX950_NUM_CUS
                    or halo_rows >= _GFX950_DGRAD_HALO_LARGE_MIN_HALO_TILES * mid[0]
                )
            )
        )
    ):
        tile = large
    else:
        tile = mid
    if occ[tile] < 2 and _wgs(tile) > _GFX950_NUM_CUS:
        if tile == large and ok[mid] and occ[mid] >= 2:
            tile = mid
        else:
            return None
    if tile[4] == 1 and p.Y * p.X > _GFX950_DGRAD_HALO_4X1_MAX_TAPS:
        return None
    large_tn = _GFX950_DGRAD_TILE_LARGE[1]
    if (
        tile == mid
        and occ[mid] < _GFX950_DGRAD_HALO_MID_OVER_LARGE_MIN_OCC
        and p.cpg % large_tn == 0
        and p.cpg // large_tn <= _GFX950_DGRAD_HALO_KEEP_LARGE_MAX_N_TILES
        and _gfx950_dgrad_tile(req) == _GFX950_DGRAD_TILE_LARGE
    ):
        return None
    if tile[4] == 1:
        min_reuse = _GFX950_DGRAD_HALO_MIN_REUSE_4X1
    elif _wgs(tile) > _GFX950_NUM_CUS:
        min_reuse = _GFX950_DGRAD_HALO_MIN_REUSE_2X2
    else:
        min_reuse = _GFX950_DGRAD_HALO_MIN_REUSE_2X2_SMALL_GRID
    if p.Y * p.X * tile[0] < min_reuse * (tile[0] + halo_rows):
        return None
    knobs = dict(base)
    if tile[4] == 1:
        knobs["dy_halo_setprio"] = _GFX950_DGRAD_HALO_SETPRIO
        padded = _dgrad_instance_spec(
            req, tile, dy_halo_kouter_pad=_GFX950_DGRAD_HALO_KOUTER_PAD, **knobs
        )
        dy_bytes = p.N * p.Ho * p.Wo * p.K * 2
        big_dy = tile == large and dy_bytes >= _GFX950_DGRAD_HALO_PAD_BIG_DY_BYTES
        if _valid(padded) and (
            big_dy or lds_cap // _dgrad_lds_bytes(padded) == occ[tile]
        ):
            knobs["dy_halo_kouter_pad"] = _GFX950_DGRAD_HALO_KOUTER_PAD
    return tile, knobs


@functools.lru_cache(maxsize=256)
def _gfx950_dgrad_pick_cached(
    req: ConvGroupedRequest,
) -> tuple[tuple[int, ...], tuple[tuple[str, int], ...]]:
    halo = _gfx950_dgrad_halo_pick(req)
    if halo is not None:
        tile, knobs = halo
        return tile, tuple(knobs.items())
    return _gfx950_dgrad_tile(req), ()


def _gfx950_dgrad_pick(
    req: ConvGroupedRequest,
) -> tuple[tuple[int, ...], dict[str, int]]:
    """``(tile, halo knobs)`` of the gfx950 dgrad candidate.

    The dY halo pick where it applies (:func:`_gfx950_dgrad_halo_pick`),
    otherwise the tile table (:func:`_gfx950_dgrad_tile`) with no halo knobs.
    Memoized per request: support() and select() both need it, and the halo
    pick builds and validates several instance specs.
    """
    tile, knobs = _gfx950_dgrad_pick_cached(req)
    return tile, dict(knobs)


def _make_gfx950_dgrad_candidate() -> KernelCandidate:
    """Backward-data conv for gfx950, configured by :func:`_gfx950_dgrad_pick`.

    Pins ``split_k=1``; the epilogue follows the D store width (cshuffle when
    dX stores more than one element). dgrad already dispatches its store
    internally on ``needs_atomic`` (stride > 1 gives more than one sub-GEMM,
    which forces the atomic store regardless of split-K). Split-K is left at
    1 rather than auto-resolved: the CK formula wgrad uses keys on its
    lopsided ``N*Ho*Wo`` reduction, and dgrad's ``Y*X*K`` is not that shape.

    Every selected spec names one AOT dgrad binary (see the notes above
    :func:`_gfx950_dgrad_tile`).

    No gfx942 or gfx1250 counterpart yet. gfx942 lacks the 32x32x16 atom, and
    the gfx1250 wave32 path is exercised through the sweep driver but has no
    dispatch-level dual-engine test of its own.
    """
    name = "implicit_gemm_conv_dgrad"
    spec_id = "igemm_conv_dgrad_tile_table"
    algorithm = "implicit_gemm_dgrad"

    def _select(req: ConvGroupedRequest) -> ConvGroupedSpec:
        tile, halo = _gfx950_dgrad_pick(req)
        return _dgrad_dispatch_spec(req, tile, halo)

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
        spec = _select(req)
        ok, why = _dgrad_is_valid_spec(spec.to_dgrad_spec(_problem(req)), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> ConvGroupedSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        return _select(req)

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
# gfx950 direct dgrad candidates (single-kernel AOT variants)
# ---------------------------------------------------------------------------
#
# The policy is dispatch.grouped_conv_dgrad_direct; these candidates wrap it.
# A selection is a ConvDirectDgradSpec naming one cached direct binary: the
# cache variant, the capabilities (filter, padding, stride, per-group channel
# counts) and the identity's knob dict.


@dataclass(frozen=True)
class ConvDirectDgradSpec:
    """Selected spec of a gfx950 direct dgrad candidate.

    ``variant`` is the direct cache variant (``KernelIdentity.algorithm``);
    ``KH``/``PAD``/``stride``/``cpg``/``kpg`` its baked capabilities and
    ``knobs`` the identity's knob dict as sorted ``(name, value)`` pairs.
    The kernel is one launch (no workspace): ``A = dY``, ``B = W``,
    ``D = dX``, with the ABI :attr:`abi_direction` names over
    :meth:`launch_problem`.
    """

    variant: str
    KH: int
    PAD: int
    stride: int
    cpg: int
    kpg: int
    knobs: tuple
    dtype: str
    arch: str
    threads_per_block: int
    direction: str = "dgrad"
    # dX is written by direct stores; read by dispatch explanations.
    epilogue: str = "direct"

    def knob_dict(self) -> dict:
        return dict(self.knobs)

    @property
    def abi_direction(self) -> str:
        """``"fwd"`` (the 4c / fused kernels run the forward kernel over the
        transposed problem) or ``"dgrad"`` (the depthwise kernels)."""
        return DIRECT_DGRAD_SINGLE_STEP_ABI[self.variant]

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            "rocke_conv_direct_dgrad",
            self.variant,
            self.dtype,
            f"r{self.KH}p{self.PAD}s{self.stride}c{self.cpg}k{self.kpg}",
            *(f"{k}{int(v) if isinstance(v, bool) else v}" for k, v in self.knobs),
        )

    def instance(self, problem: DirectConvProblem):
        """The kernel spec on ``problem`` (the spec the cache compiled)."""
        return make_direct_dgrad_spec(self.variant, problem, self.knob_dict())

    def launch_problem(self, problem: DirectConvProblem) -> DirectConvProblem:
        """The problem the kernargs describe: the transposed one for the
        forward-ABI kernels, ``problem`` itself for the dgrad-ABI ones."""
        if self.abi_direction == "fwd":
            return self.instance(problem).problem
        return problem


def _direct_problem(req: ConvGroupedRequest) -> DirectConvProblem:
    """The direct-conv problem of a 2D dgrad request (square pad and stride
    are read from the H axis; callers check the W axis matches)."""
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
        stride=int(p.sH),
        dtype=req.dtype.lower(),
        dil_h=int(p.dH),
        dil_w=int(p.dW),
    )


def _direct_request_errors(req: ConvGroupedRequest) -> list[str]:
    """Request shapes no direct dgrad kernel takes: 3D, and a pad or stride
    that differs between the axes (the direct problem has one of each)."""
    errors = []
    if req.Di is not None:
        errors.append("direct dgrad is 2D only")
    if int(req.pad_h) != int(req.pad_w):
        errors.append(f"requires pad_h == pad_w (got {req.pad_h}, {req.pad_w})")
    if int(req.stride_h) != int(req.stride_w):
        errors.append(
            f"requires stride_h == stride_w (got {req.stride_h}, {req.stride_w})"
        )
    return errors


def _direct_dgrad_spec(
    req: ConvGroupedRequest, variant: str, knobs: dict
) -> ConvDirectDgradSpec:
    p = _direct_problem(req)
    spec = make_direct_dgrad_spec(variant, p, knobs)
    return ConvDirectDgradSpec(
        variant=variant,
        KH=p.KH,
        PAD=p.PAD,
        stride=p.stride,
        cpg=p.cpg,
        kpg=p.kpg,
        knobs=tuple(sorted(knobs.items())),
        dtype=req.dtype.lower(),
        arch=req.arch,
        threads_per_block=int(spec.threads_per_block),
    )


def _direct_grid(spec: ConvDirectDgradSpec, req: OperatorRequest) -> tuple:
    assert isinstance(req, ConvGroupedRequest)
    grid, _block = direct_launch_geometry(spec.instance(_direct_problem(req)))
    return grid


def _direct_signature(spec: ConvDirectDgradSpec) -> tuple:
    """The kernarg signature of the one launch (the step signature
    ``direct_kernel_sweep.plan_for`` gives the cached binary)."""
    from kernels.common.conv_abi import conv_direct_args_signature

    return tuple(conv_direct_args_signature(spec.dtype, direction=spec.abi_direction))


@functools.lru_cache(maxsize=256)
def _direct_generic_pick(req: ConvGroupedRequest) -> tuple:
    """``(variant, knobs items, reasons)`` of the grouped direct dgrad policy."""
    errors = _direct_request_errors(req)
    if errors:
        return None, (), tuple(errors)
    variant, knobs, errors = _ddp.select_generic(_direct_problem(req), req.arch)
    return variant, tuple(sorted(knobs.items())), tuple(errors)


@functools.lru_cache(maxsize=256)
def _direct_depthwise_pick(req: ConvGroupedRequest, streaming: bool) -> tuple:
    """``(knobs items, reasons)`` of a depthwise dgrad kernel for ``req``."""
    errors = _direct_request_errors(req)
    if int(req.G) != int(req.C) or int(req.G) != int(req.K):
        errors.append(
            f"requires depthwise G == C == K (got G={req.G}, C={req.C}, K={req.K})"
        )
    if errors:
        return (), tuple(errors)
    select = _ddp.select_depthwise_stream if streaming else _ddp.select_depthwise_win
    knobs, errors = select(_direct_problem(req), req.arch)
    return tuple(sorted((knobs or {}).items())), tuple(errors)


def _make_gfx950_direct_candidate(
    name: str,
    spec_id: str,
    algorithm: str,
    priority: int,
    pick: Callable[[ConvGroupedRequest], tuple[str | None, dict, list]],
) -> KernelCandidate:
    """A gfx950 direct dgrad candidate around ``pick``, which returns
    ``(variant, knobs, reasons)`` for a request (``variant`` None: declined
    with ``reasons``)."""

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
        variant, _knobs, reasons = pick(req)
        if variant is None:
            return False, "; ".join(reasons)
        return True, "ok"

    def select(req: OperatorRequest) -> ConvDirectDgradSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvGroupedRequest)
        variant, knobs, _ = pick(req)
        return _direct_dgrad_spec(req, variant, knobs)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY_DGRAD,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=CONV_GROUPED_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("fp16", "bf16"),
            layouts=("NHWC",),
        ),
        _supports=support,
        select_spec=select,
        signature=_direct_signature,
        grid=_direct_grid,
        block=lambda spec: (spec.threads_per_block, 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _generic_pick(req: ConvGroupedRequest) -> tuple:
    variant, knobs, reasons = _direct_generic_pick(req)
    return variant, dict(knobs), list(reasons)


def _depthwise_win_pick(req: ConvGroupedRequest) -> tuple:
    knobs, reasons = _direct_depthwise_pick(req, False)
    return (_ddp.VARIANT_DW_WIN if knobs else None), dict(knobs), list(reasons)


def _depthwise_stream_pick(req: ConvGroupedRequest) -> tuple:
    knobs, reasons = _direct_depthwise_pick(req, True)
    return (_ddp.VARIANT_DW_STREAM if knobs else None), dict(knobs), list(reasons)


def _make_gfx950_direct_dgrad_candidate() -> KernelCandidate:
    """Grouped and small-channel stride-1 dgrad: the 4c and fused generic
    direct-MFMA kernels. Outranks the implicit-GEMM candidate on the problems
    it admits (:func:`dispatch.grouped_conv_dgrad_direct.select_generic`),
    groups == 1 included."""
    return _make_gfx950_direct_candidate(
        name="direct_mfma_conv_dgrad",
        spec_id="direct_mfma_dgrad_grouped",
        algorithm="direct_mfma_dgrad",
        priority=5,
        pick=_generic_pick,
    )


def _make_gfx950_depthwise_dgrad_win_candidate() -> KernelCandidate:
    """Depthwise (cpg = kpg = 1) stride-1 dgrad: the windowed kernel (VALU or
    Toeplitz MFMA form). The implicit-GEMM candidate rejects every cpg = 1
    request; what this one declines goes to the streaming kernel."""
    return _make_gfx950_direct_candidate(
        name="direct_depthwise_dgrad_win",
        spec_id="direct_dw_dgrad_windowed",
        algorithm="direct_depthwise_dgrad_win",
        priority=5,
        pick=_depthwise_win_pick,
    )


def _make_gfx950_depthwise_dgrad_stream_candidate() -> KernelCandidate:
    """Depthwise dgrad at stride 1 or 2: the row-streaming kernel. Serves the
    strided problems, and the stride-1 ones the windowed kernel declines."""
    return _make_gfx950_direct_candidate(
        name="direct_depthwise_dgrad_stream",
        spec_id="direct_dw_dgrad_stream",
        algorithm="direct_depthwise_dgrad",
        priority=6,
        pick=_depthwise_stream_pick,
    )


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
CONV_DGRAD_REGISTRY.register(_make_gfx950_depthwise_dgrad_win_candidate())
CONV_DGRAD_REGISTRY.register(_make_gfx950_depthwise_dgrad_stream_candidate())

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
    spec_dict = asdict(spec)
    # The dgrad binary fields are hashed only when set, so every spec without
    # them keeps the hash it had before the fields existed.
    if isinstance(spec, ConvGroupedSpec):
        for f in _DGRAD_SPEC_FIELDS:
            if not spec_dict[f]:
                del spec_dict[f]
    spec_hash = stable_json_hash(spec_dict, n=16)
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


def _signature_for(
    req: ConvGroupedRequest, spec: ConvGroupedSpec | ConvDirectDgradSpec
) -> tuple:
    """Launch signature for a dispatched kernel, including its variant extras.

    The trailing args are not cosmetic: implicit-GEMM dgrad always carries the
    tilde record buffer, wgrad always carries ``ks``/``ks_count``, and a
    two-stage wgrad also carries the workspace pair. Kernargs pack
    positionally, so omitting one shifts every argument after it. A direct
    dgrad pick takes the direct ABI of its kernel (forward over the
    transposed problem, or dgrad), with none of those extras.
    """
    from kernels.common.conv_abi import conv_args_signature

    if isinstance(spec, ConvDirectDgradSpec):
        return _direct_signature(spec)
    p = _problem(req)
    dtype = req.dtype
    if req.direction == "wgrad":
        # Two-stage is decided by the resolver, not carried on the dispatcher
        # spec: it is what to_wgrad_spec builds the kernel from, so asking
        # anything else would drop ws_ptr/ws_bytes from a two-stage ABI.
        _split_k, two_stage, _requested = _resolve_wgrad_split_k(spec, p)
        return tuple(
            conv_args_signature(
                dtype,
                direction="wgrad",
                is_3d=p.is_3d,
                two_stage=two_stage,
            )
        )
    if req.direction == "dgrad":
        return tuple(conv_args_signature(dtype, direction="dgrad", is_3d=p.is_3d))
    return tuple(conv_args_signature(dtype, is_3d=p.is_3d))


def launch_values_for(
    req: ConvGroupedRequest,
    spec: ConvGroupedSpec | ConvDirectDgradSpec,
    *,
    A_ptr: int,
    B_ptr: int,
    D_ptr: int,
    A_bytes: int,
    B_bytes: int,
    D_bytes: int,
    ws_ptr: Optional[int] = None,
    ws_bytes: Optional[int] = None,
    sub_gemm_buf: Optional[int] = None,
    num_sub_gemms: Optional[int] = None,
) -> dict:
    """Compute the full AOT ``values`` dict for a dispatch request.

    Operand roles by direction:
      fwd   -- A = activation, B = filter,  D = output
      wgrad -- A = dY,         B = X,       D = dW
      dgrad -- A = dY,         B = W,       D = dX

    The tile size comes from ``spec`` because the chiplet-swizzle path decodes
    the workgroup id against ``p_num_pid_m/n``, which are tile-count values.

    A direct dgrad pick (:class:`ConvDirectDgradSpec`) takes the direct ABI
    over its launch problem and no extras: ``ws_*`` and ``sub_gemm_*`` must
    be left unset (only the implicit-GEMM dgrad kernel carries the record
    buffer).
    """
    from kernels.common.conv_args import ConvArgs

    if isinstance(spec, ConvDirectDgradSpec):
        extras = (ws_ptr, ws_bytes, sub_gemm_buf, num_sub_gemms)
        if any(v is not None for v in extras):
            raise ValueError(
                "a direct dgrad kernel takes no workspace and no sub-GEMM "
                "record buffer"
            )
        problem = spec.launch_problem(_direct_problem(req))
        return ConvArgs.from_problem(
            problem, direction=spec.abi_direction
        ).to_launch_values(A_ptr, B_ptr, D_ptr, A_bytes, B_bytes, D_bytes)
    p = _problem(req)
    tile_m, tile_n = spec.tile_m, spec.tile_n
    if req.direction == "wgrad":
        # The resolved degree, not spec.split_k: the latter may be -1 (auto)
        # or exceed the clamp, while the kernel and _wgrad_grid both use the
        # resolver's value -- ks_count has to match the z extent it decodes.
        split_k, two_stage, _requested = _resolve_wgrad_split_k(spec, p)
        # Both workspace fields or neither: a two-stage launch missing only
        # ws_bytes would otherwise encode a zero-byte workspace, and Stage 1's
        # bounded stores would be silently dropped.
        has_ws_ptr = ws_ptr is not None
        has_ws_bytes = ws_bytes is not None
        if has_ws_ptr != has_ws_bytes or two_stage != has_ws_ptr:
            raise ValueError(
                "wgrad resolved to the "
                + ("two-stage" if two_stage else "single-stage")
                + " kernel; ws_ptr/ws_bytes must be passed exactly when it is "
                "two-stage"
            )
        # Deliberately the TRUE problem, not spec.group_merge's merged one.
        # Two separate reasons, and both have to hold:
        #   - The problem kernargs (p_C/p_K/p_cpg/p_kpg/p_wg_M/p_wg_N/...)
        #     describe the unmerged problem by contract: the kernel scales the
        #     ones the merged load side needs by Gm itself, since Gm is a
        #     build-time constant there. Handing it pre-merged dims would apply
        #     the factor twice.
        #   - p_num_pid_m/n are launch geometry, so they would have to be
        #     merged -- but wgrad_group_merge_available admits a merged spec
        #     only when grid_M <= tile_m and grid_N <= tile_n, i.e. the merged
        #     GEMM is exactly one tile. Both counts are therefore 1, and the
        #     unmerged dims (which are smaller still) also ceil to 1. The two
        #     agree by that gate, not by luck.
        # test_merged_tile_counts_are_single_tile pins the second one, so if
        # the gate ever admits a multi-tile merge this stops being true loudly.
        return ConvArgs.from_problem(
            p, direction="wgrad", tile_m=tile_m, tile_n=tile_n, tile_k=spec.tile_k
        ).to_launch_values(
            A_ptr,
            B_ptr,
            D_ptr,
            A_bytes,
            B_bytes,
            D_bytes,
            split_k=split_k,
            ws_ptr=ws_ptr,
            ws_bytes=ws_bytes,
        )
    if req.direction == "dgrad":
        if sub_gemm_buf is None or num_sub_gemms is None:
            raise ValueError(
                "dgrad needs sub_gemm_buf/num_sub_gemms: the tilde record "
                "buffer is always part of the dgrad kernarg ABI"
            )
        return ConvArgs.from_problem(
            p, direction="dgrad", tile_m=tile_m, tile_n=tile_n
        ).to_launch_values(
            A_ptr,
            B_ptr,
            D_ptr,
            A_bytes,
            B_bytes,
            D_bytes,
            sub_gemm_buf=sub_gemm_buf,
            num_sub_gemms=num_sub_gemms,
        )
    return ConvArgs.from_problem(p, tile_m=tile_m, tile_n=tile_n).to_launch_values(
        A_ptr, B_ptr, D_ptr, A_bytes, B_bytes, D_bytes
    )


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
    if isinstance(spec, ConvDirectDgradSpec):
        explanation.append(
            f"variant={spec.variant} knobs={spec.knob_dict()} "
            f"abi={spec.abi_direction}"
        )
        if req.vec_size_c is not None:
            explanation.append(
                f"vec_size_c={req.vec_size_c} ignored (implicit-GEMM epilogue hint)"
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
    # AOT: signature includes all runtime problem-dim args.
    _sig = _signature_for(req, spec)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=_sig,
        explanation=tuple(explanation),
    )
