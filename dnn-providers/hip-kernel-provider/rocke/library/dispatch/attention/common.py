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

from dataclasses import asdict, dataclass
from typing import Tuple

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
from rocke.dispatch.core import KernelCandidate, OperatorRequest

FAMILY = "attention_unified"
ATTENTION_ABI_VERSION = "hipkg-attention-unified/v1"


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
    mask_type: int = 0  # 0=none, 1=causal/top-left, ...
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
    use_fp8: bool = False
    fp8_fnuz: bool = False
    # --- gfx950 attention_dense knobs (only consumed by the opt-in
    #     ``attention_dense`` candidate; ignored by the unified 2D/3D paths).
    #     Defaults deliver the persistent ~970-TFLOPS prefill path for large Sq:
    #     ``dense_persistent="auto"`` turns on the grid-stride variant once there
    #     is enough work to fill the persistent grid, and ``persist_decode="auto"``
    #     picks the L2-locality hkv-major decode where it is balance-safe. ---
    dense_persistent: str = "auto"  # "auto" | "on" | "off"
    dense_num_persistent: int = 256
    dense_persist_decode: str = "auto"  # "auto" | "qb_major" | "hkv_major"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
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
        if int(self.mask_type) != 0:
            active.add("causal")
        if int(self.sliding_window) > 0:
            active.add("sliding_window")
        if bool(self.use_sinks):
            active.add("sinks")
        if bool(self.use_fp8):
            active.add("fp8")
        return frozenset(active)


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

ATTENTION_FEATURES = frozenset({"causal", "sliding_window", "sinks", "fp8"})


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


def _resolve_num_cus(req: AttentionRequest) -> int:
    """Resolve the split-KV device-subscription target (``num_cus``).

    ``num_cus`` is the dispatcher's "how many CUs does this device have" knob; it
    drives 2D<->3D routing (``select_path``) and the 3D segment count. Resolution:
      1. an explicit caller value (benchmarks pass a real count),
      2. **verified on-box gfx942 only** -- the live device CU count, used ONLY
         when the request targets gfx942 AND the build box IS gfx942, so a
         cross-compile never bakes the wrong device's count into the kernel,
      3. otherwise the legacy ``120`` (matches develop): every non-gfx942 arch,
         and gfx942 built off-box / with no visible gfx942 device.
    An explicit ``target_ctas`` on the spec supersedes all of the above. Because
    the resolved value feeds the 3D ``num_segments`` (a compiled-kernel constant),
    the on-box value is device-dependent within gfx942 (varies across parts);
    for a reproducible or cross-compile target pass an explicit ``num_cus`` or the
    ``target_ctas`` spec override rather than relying on the live query.
    """
    n = int(req.num_cus)
    if n > 0:
        return n
    if req.arch.lower() == "gfx942":
        try:
            from rocke.runtime.hip_module import get_device_arch

            if get_device_arch() == "gfx942":
                cus = _device_num_cus()
                if cus and cus > 0:
                    return cus
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
    )


def _selector_matches(
    req: AttentionRequest, candidate: KernelCandidate
) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


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
