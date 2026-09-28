# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pieces of the attention family: request, spec, and the gates every
candidate re-uses.

Everything here is arch-neutral by construction. The arch modules import from
this one; nothing here imports them, which is what keeps the registry assembly
in ``__init__`` free of import-order dependence.

SCOPE -- what this dispatcher decides
-------------------------------------
The load-bearing dispatch decision for unified attention is the *kernel path*:
the 2D-tiled (per-(kv_head, q_block) CTA) kernel vs the 3D split-KV kernel. That
decision is a PURE function of the problem
(:func:`rocke.helpers.attention.use_2d_kernel`, surfaced as
``UnifiedAttentionProblem.select_path``), so it can be mirrored byte-faithfully
on the C++ side. Backend coverage is gated by
:func:`attention_unified.supports_native_unified_attention` (head_size /
block_size / dtype / feature gate -- also pure).

The structural identity used for selection parity is therefore::

    (path, head_size, block_size)

where ``path`` is ``"2d"`` or ``"3d"``.

DEFERRED -- arch-tuned block geometry
-------------------------------------
The 2D-tiled kernel's exact CTA geometry (``num_warps`` / ``block_m_per_warp`` /
``tile_size``) is chosen by heuristics in ``attention_unified`` that query the
running device arch (``_resolve_attention_arch``) and encode many measured,
shape-specific thresholds (see ``_select_2d_num_warps`` et al.). Those are
downstream PERFORMANCE-TUNING knobs, not a "which kernel family" decision, and
they are not reproducible CPU-only without a device. They are intentionally OUT
of the parity identity here; modelling them faithfully across C++/Python is a
separate, larger effort. This dispatcher selects the path + backend; geometry is
left to the instance builder at launch time.

That deferral is also why the unified candidates report ``grid=(0, 0, 0)`` and
an empty ``signature``, and so cannot carry a ``bind``: there is no geometry to
bind to until phase 6 moves the routing policy up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import IntEnum
from operator import index
from typing import Any, Tuple

from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    # Re-exported, never redeclared. The kernel owns what it covers; dispatch's
    # job is to state that coverage as a Capability so it can be filtered and
    # queried without probing. Copying the numbers would drift in the one
    # direction that fails silently -- the prefilter rejecting a shape the
    # kernel had since learned to run -- so they come from the backend itself.
    UNIFIED_BLOCK_SIZES,
    UNIFIED_DTYPES,
    UNIFIED_HEAD_SIZES,
)
from rocke.core.arch import ArchTarget
from rocke.dispatch.core import KernelCandidate, OperatorRequest, selector_matches

FAMILY = "attention_unified"
ATTENTION_ABI_VERSION = "hipkg-attention-unified/v1"


class AttentionMaskType(IntEnum):
    """Attention-mask ordinals matching ``plan_utils::MaskType``.

    These are mask kinds, not hipDNN ``DiagonalAlignment`` ordinals.
    """

    NO_MASK = 0
    TOP_LEFT_CAUSAL = 1
    BOTTOM_RIGHT_CAUSAL = 2
    SLIDING_WINDOW = 3


_ATTENTION_MASK_ORDINALS = tuple(mask_type.value for mask_type in AttentionMaskType)


def _parse_attention_mask_type(value: object) -> AttentionMaskType:
    """Return a validated mask enum without truncating or parsing strings."""
    try:
        ordinal = index(value)
        return AttentionMaskType(ordinal)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "mask_type must be an exact integer ordinal in "
            f"{_ATTENTION_MASK_ORDINALS}, got {value!r}"
        ) from exc


@dataclass(frozen=True)
class AttentionRequest(OperatorRequest):
    """Normalized scaled-dot-product-attention request."""

    batch: int
    nhead_q: int
    nhead_k: int
    seqlen_q: int
    seqlen_k: int
    hdim_q: int
    hdim_v: int
    arch: str
    mask_type: AttentionMaskType | int = AttentionMaskType.NO_MASK
    use_sinks: bool = False
    sliding_window: int = 0
    kv_block_size: int = 16  # paged KV block_size (modulus); {16,32,64}
    num_cus: int = (
        0  # 0 => auto-resolve to the device CU count at dispatch (_resolve_num_cus)
    )
    target_ctas: int = (
        0  # 0 => auto: num_cus*4. >0 pins the routing/segmentation target directly.
    )
    op: str = "attention"
    dtype: str = "fp16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    # Concrete micro-configuration returned by a unified_tuning candidate.
    # "auto" selects that geometry candidate's baseline; sweep APIs enumerate
    # every valid value and persist this id with the result.
    attention_tuning_id: str = "auto"
    use_fp8: bool = False
    fp8_fnuz: bool = False
    # --- standalone attention_dense knobs (only consumed by the opt-in
    #     ``attention_dense`` candidate; ignored by the unified 2D/3D paths).
    #     Defaults deliver the best qualified persistent prefill path for large Sq:
    #     ``dense_persistent="auto"`` turns on the grid-stride variant once there
    #     is enough work to fill the persistent grid, and ``persist_decode="auto"``
    #     resolves through the selected architecture's concrete dense spec.
    #     gfx950 may select one-/two-phase GQA pairing and wide DMA; gfx942
    #     supports only qb-major/hkv-major ordering. ---
    dense_persistent: str = "auto"  # "auto" | "on" | "off"
    dense_num_persistent: int = 256
    # 0 keeps the architecture's shipped policy; 1..8 pins the emitted
    # amdgpu-waves-per-eu attribute. Dense sweeps expand an unpinned request.
    dense_waves_per_eu: int = 0
    # Common: auto/qb_major/hkv_major; gfx950 also supports gqa_pair variants.
    dense_persist_decode: str = "auto"
    # gfx950 dense variant pins. ``auto`` does not filter that axis; the
    # ranker / ``dense_spec_for_request`` still apply the historical policy.
    # ``dense_tile`` names a DENSE_TILE_GEOMETRIES key (``default`` / ``bm128``).
    dense_tile: str = "auto"  # "auto" | "default" | "bm128"
    dense_wide_lds_dma: str = "auto"  # "auto" | "on" | "off"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
        # IntEnum and raw-int callers describe the same request/cache identity.
        d["mask_type"] = _parse_attention_mask_type(self.mask_type).value
        return d

    def dims(self) -> dict[str, int]:
        return {
            "batch": int(self.batch),
            "nhead_q": int(self.nhead_q),
            "nhead_k": int(self.nhead_k),
            "seqlen_q": int(self.seqlen_q),
            "seqlen_k": int(self.seqlen_k),
            "hdim_q": int(self.hdim_q),
            "hdim_v": int(self.hdim_v),
            "kv_block_size": int(self.kv_block_size),
        }

    def features(self) -> frozenset[str]:
        active = set()
        try:
            mask_type = _parse_attention_mask_type(self.mask_type)
        except ValueError:
            # Let _request_errors report the invalid ordinal. In particular, do
            # not silently classify an arbitrary nonzero value as causal.
            mask_type = None
        if mask_type is not None and mask_type != AttentionMaskType.NO_MASK:
            active.add("causal")
        if mask_type == AttentionMaskType.BOTTOM_RIGHT_CAUSAL and int(
            self.seqlen_q
        ) != int(self.seqlen_k):
            active.add("causal_bottom_right")
        if int(self.sliding_window) > 0:
            active.add("sliding_window")
        if bool(self.use_sinks):
            active.add("sinks")
        if bool(self.use_fp8):
            active.add("fp8")
        return frozenset(active)


def _resolve_dense_waves_per_eu(req: AttentionRequest, default: int) -> int:
    """Resolve the dense WPE request without changing the shipped default policy."""
    value = int(req.dense_waves_per_eu)
    if value == 0:
        value = int(default)
    if not 1 <= value <= 8:
        raise ValueError(
            "dense_waves_per_eu must be 0 (auto) or in [1, 8], "
            f"got {req.dense_waves_per_eu}"
        )
    return value


ATTENTION_DIM_VOCABULARY = (
    "batch",
    "nhead_q",
    "nhead_k",
    "seqlen_q",
    "seqlen_k",
    "hdim_q",
    "hdim_v",
    "kv_block_size",
)

ATTENTION_FEATURES = frozenset(
    {"causal", "causal_bottom_right", "sliding_window", "sinks", "fp8"}
)


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, AttentionRequest):
        return [f"expected AttentionRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "attention":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("batch", "nhead_q", "nhead_k", "seqlen_q", "seqlen_k", "hdim_q"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if req.hdim_q != req.hdim_v:
        errors.append("only hdim_q == hdim_v is supported")
    if int(req.nhead_q) % int(req.nhead_k):
        errors.append("nhead_q must be divisible by nhead_k (GQA grouping)")
    try:
        _parse_attention_mask_type(req.mask_type)
    except ValueError as exc:
        errors.append(str(exc))
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _device_num_cus() -> "int | None":
    """Live device multiprocessor (CU) count, or None if unqueryable.

    Torch-free: delegates to the ctypes ``libamdhip64`` wrapper
    (``rocke.runtime.hip_module``) so the library layer stays off torch. NOTE:
    this resolver -- and the ``target_ctas`` routing/segmentation override -- cover
    the Python dispatch path only. The C++ C-ABI engine keeps its own num_cus
    default (attention_unified_entry.cpp) with no target_ctas field, so both need
    the mirror resolver + target_ctas there for production (companion change).
    """
    try:
        from rocke.runtime.hip_module import get_device_num_cus

        return get_device_num_cus()
    except Exception:
        return None


# Arches whose CU count is auto-resolved from the live device when the request
# runs on-box; every other arch keeps the legacy 120. This set governs
# device-resolution ONLY. The split-KV over-split guard is a separate per-arch
# branch in kernels/common/attention_unified.py::_num_segments -- adding an arch
# here does NOT give it a segment clamp; add a matching branch there too (both
# read the resolved value through the shared _effective_target_ctas seam).
# Membership also does NOT apply the partitioned-device floor in
# _resolve_num_cus: that floor is gfx950-scoped, because only gfx950's clamp is
# defined against the 120-CU baseline. An arch added here gets the raw live
# count; decide per arch whether it also needs the floor.
_AUTO_RESOLVE_ARCHS = frozenset({"gfx942", "gfx950"})


def _resolve_num_cus(req: AttentionRequest) -> int:
    """Resolve the split-KV device-subscription target (``num_cus``).

    ``num_cus`` is the dispatcher's "how many CUs does this device have" knob; it
    drives 2D<->3D routing (``select_path``) and the 3D segment count. Resolution:
      1. an explicit caller value (benchmarks pass a real count),
      2. **verified on-box only** -- the live device CU count, used ONLY when the
         request targets an arch in ``_AUTO_RESOLVE_ARCHS`` (gfx942, gfx950) AND
         the build box IS that same arch, so a cross-compile never bakes the wrong
         device's count into the kernel,
      3. otherwise the legacy ``120`` (matches develop): every non-auto-resolve
         arch, and an auto-resolve arch built off-box / with no visible matching
         device.
    The on-box count is floored at ``120`` for **gfx950 only** (see below); every
    other arch passes the live count through exactly as develop does.
    An explicit ``target_ctas`` on the spec supersedes all of the above. Because
    the resolved value feeds the 3D ``num_segments`` (a compiled-kernel constant),
    the on-box value is device-dependent within an arch (varies across parts); for
    a reproducible or cross-compile target pass an explicit ``num_cus`` or the
    ``target_ctas`` spec override rather than relying on the live query.
    """
    n = int(req.num_cus)
    if n > 0:
        return n
    arch = req.arch.lower()
    if arch in _AUTO_RESOLVE_ARCHS:
        try:
            from rocke.runtime.hip_module import get_device_arch

            if get_device_arch() == arch:
                cus = _device_num_cus()
                if cus and cus > 0:
                    # gfx950 ONLY: floor at the legacy 120 baseline. On a
                    # partitioned device (CPX / NPS4) the query returns the
                    # PARTITION's CU count (~32), which would drop the routing
                    # target below the 480 pre-bump baseline -- flipping
                    # already-3D shapes back to 2D (the opposite of this
                    # resolver's intent) and breaking the byte-identical no-op
                    # guarantee the gfx950 _num_segments clamp is built on (that
                    # clamp is defined against _PRE_BUMP_CUS = 120).
                    #
                    # gfx942 is deliberately NOT floored: it shipped unfloored,
                    # its clamp uses measured per-shape carve-outs rather than a
                    # blanket pre-bump bound, and flooring would change 2D/3D
                    # routing and segment counts on partitioned gfx942 parts --
                    # an unmeasured change this resolver does not need.
                    return max(cus, 120) if arch == "gfx950" else cus
        except Exception:
            pass
    return 120


def _problem(req: AttentionRequest) -> UnifiedAttentionProblem:
    # total_q = batch * seqlen_q (the flattened query rows). num_seqs = batch.
    return UnifiedAttentionProblem(
        total_q=int(req.batch) * int(req.seqlen_q),
        num_seqs=int(req.batch),
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=int(req.hdim_q),
        block_size=int(req.kv_block_size),
        max_seqlen_q=int(req.seqlen_q),
        max_seqlen_k=int(req.seqlen_k),
        dtype=req.dtype.lower(),
        sliding_window=int(req.sliding_window),
        use_sinks=bool(req.use_sinks),
        use_fp8=bool(req.use_fp8),
        fp8_fnuz=bool(req.fp8_fnuz),
        num_cus=_resolve_num_cus(req),
        target_ctas=int(req.target_ctas),
        clamp_arch=req.arch.lower(),
    )


# Shared pin-selector: identical across families, so it lives in the dispatch
# core and each family re-exports it under its own name.
_selector_matches = selector_matches


@dataclass(frozen=True)
class AttentionSpec:
    """The selected attention path + the dims that determine the kernel family."""

    path: str  # "2d" | "3d"
    head_size: int
    block_size: int
    dtype: str
    num_query_heads: int
    num_kv_heads: int
    # fp8 KV-cache decode is a distinct kernel from the same-shape bf16 decode
    # (per-element dequant in the inner loop), so it must not collapse onto the
    # bf16 spec_hash/kernel_name -- consumers key their compile cache on
    # kernel_name(), and the sweep space dedupes on asdict(spec).
    use_fp8: bool = False
    fp8_fnuz: bool = False
    name: str = "rocke_attention_unified"
    # When set, this verbatim kernel_name is returned by :meth:`kernel_name`
    # instead of the composed unified name. Used by the dense candidate to
    # surface the concrete persistent/ragged/decode kernel it will launch.
    kernel_name_override: str = ""
    # Optional pinned codegen knobs a specialized candidate wants the builder to
    # apply (sorted (key, value) pairs; empty for generic candidates). See
    # ``attention_unified._d256_gfx950_spec_overrides``; the builder consumes them
    # via ``_tiled_spec_from_problem(problem, overrides=...)``.
    tiled_overrides: Tuple[Tuple[str, object], ...] = ()

    def kernel_name(self) -> str:
        if self.kernel_name_override:
            return self.kernel_name_override
        from rocke.helpers.spec import kernel_name_join

        parts = [
            self.name,
            self.path,
            self.dtype,
            f"hd{self.head_size}",
            f"bs{self.block_size}",
            f"gqa{self.num_query_heads}x{self.num_kv_heads}",
        ]
        if self.use_fp8:
            parts.append("fp8fnuz" if self.fp8_fnuz else "fp8")
        return kernel_name_join(*parts)


@dataclass(frozen=True)
class AttentionTuningSpec:
    """Concrete dispatcher-owned tiled spec used by exhaustive sweeps.

    Generic production candidates intentionally return :class:`AttentionSpec`
    and defer geometry.  A tuning candidate returns this wrapper instead: the
    exact arch spec, builder choice, compile backend, and optional 3D reduce
    spec are serializable and therefore participate in dispatch/cache identity.

    It is also the whole launch contract the runtime needs:
    ``run_unified_attention_torch(tuning_spec=...)`` compiles :meth:`build`
    under :meth:`cache_key` and launches with :meth:`launch_grid` /
    :meth:`launch_block`, so the runtime never decodes ``builder_kind``.
    ``allow_unsupported`` skips the runtime's problem-shape support check.
    """

    path: str
    arch: str
    builder_kind: str
    compile_backend: str
    candidate_name: str
    tuning_id: str
    kernel_spec: Any
    fp8_fnuz: bool = False
    num_kv_blocks: int = 0
    reduce_spec: Any = None
    tuning_id_prefix: str = ""
    allow_unsupported: bool = False

    def kernel_name(self) -> str:
        return self.kernel_spec.kernel_name()

    def cache_key(self) -> Tuple:
        """Field-complete launcher-cache identity of the kernel(s) built."""

        def items(spec):
            if spec is None:
                return None
            if not is_dataclass(spec):
                raise TypeError(f"tuning kernel spec must be a dataclass, got {spec!r}")
            return tuple((f.name, repr(getattr(spec, f.name))) for f in fields(spec))

        return (
            "explicit",
            self.arch,
            self.builder_kind,
            self.compile_backend,
            items(self.kernel_spec),
            items(self.reduce_spec),
        )

    def build(self, arch: str | None = None):
        """IR for this spec: one kernel for 2D, ``(segment, reduce)`` for 3D."""
        from .tuning_specs import (
            build_explicit_attention_2d,
            build_explicit_attention_3d,
        )

        arch = arch or self.arch
        if self.path == "3d":
            return build_explicit_attention_3d(
                self.kernel_spec, self.reduce_spec, arch=arch
            )
        if self.builder_kind == "gfx942_4warp_gqa":
            from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

            return build_gfx942_4warp_gqa(self.kernel_spec, arch=arch)
        return build_explicit_attention_2d(self.kernel_spec, arch=arch)

    def launch_grid(self, problem: UnifiedAttentionProblem) -> Tuple[int, int, int]:
        ks = self.kernel_spec
        if self.path == "3d":
            block_q = max(1, 16 // problem.num_queries_per_kv)
            qblocks = problem.total_q // block_q + problem.num_seqs
            return (int(qblocks), int(problem.num_kv_heads), int(ks.num_segments))
        if self.builder_kind == "gfx942_4warp_gqa":
            from kernels.common.attention_unified import gfx942_4warp_launch_grid

            return gfx942_4warp_launch_grid(problem)
        block_m = int(ks.block_m)
        block_q = (
            block_m // problem.num_queries_per_kv
            if problem.num_queries_per_kv <= block_m
            else 1
        )
        qblocks = int(problem.total_q // block_q + problem.num_seqs)
        if bool(getattr(ks, "use_q_major_grid", False)):
            return (qblocks, int(problem.num_kv_heads), 1)
        return (int(problem.num_kv_heads), qblocks, 1)

    def launch_block(self) -> Tuple[int, int, int]:
        if self.path == "3d":
            return (64, 1, 1)
        if self.builder_kind == "gfx942_4warp_gqa":
            return (256, 1, 1)
        return (64 * int(self.kernel_spec.num_warps), 1, 1)

    def with_num_kv_blocks(self, num_kv_blocks: int) -> "AttentionTuningSpec":
        """Refresh runtime-addressing state after the paged cache is known."""
        from .tuning_common import retarget_tuning_spec

        return retarget_tuning_spec(self, num_kv_blocks=int(num_kv_blocks))
