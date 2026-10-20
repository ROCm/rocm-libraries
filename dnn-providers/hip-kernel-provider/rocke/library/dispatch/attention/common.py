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
``tile_size``) is chosen by heuristics in ``attention_unified`` that encode many
measured, shape-specific thresholds (see ``_select_2d_num_warps`` et al.). Those
are downstream PERFORMANCE-TUNING knobs, not a "which kernel family" decision.
They are intentionally OUT of the parity identity here; modelling them faithfully
across C++/Python is a separate, larger effort. This dispatcher selects the path
+ backend; geometry is left to the instance builder at launch time.

Since AICK-2146 those heuristics take ``arch`` as an explicit argument rather
than probing the running device, so they ARE reproducible CPU-only: geometry for
``gfx950`` can be computed from a ``gfx942`` host. That makes the deferral a
scoping choice about the C++ parity identity, not a capability limit. The one
remaining host read on the selection path is the CU count when ``num_cus`` is
left at 0 -- see :func:`_resolve_num_cus` for the exact scope of the guarantee.

That deferral is also why the unified candidates report ``grid=(0, 0, 0)`` and
an empty ``signature``, and so cannot carry a ``bind``: there is no geometry to
bind to until phase 6 moves the routing policy up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
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
from rocke.core.arch import ArchTarget, base_arch_from_target_id
from rocke.dispatch.core import KernelCandidate, OperatorRequest, selector_matches
from rocke.dispatch.tuning.identity import jsonable, normalize_knobs

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


def canonical_arch(arch: str) -> str:
    """The single arch authority for the attention family.

    Every arch comparison in this family is an exact string compare against a
    lowercase base name (``"gfx942"``, ``"gfx950"``). That only holds if the
    strings agree on case and carry no target-ID decoration, and previously each
    site defended itself -- some with ``.lower()``, some with ``.strip()``, some
    not at all. A feature suffix therefore silently failed a gate: a request for
    ``gfx950:sramecc+`` is a request for gfx950, but ``== "gfx950"`` said no.

    :func:`base_arch_from_target_id` is the platform's existing canonicalizer and
    is already what :func:`get_device_arch` applies, so the host-resolved arch is
    canonical by construction and ``request.arch`` was the only uncanonical
    source. Reusing it here -- rather than writing a second normalizer -- is what
    makes the two sources comparable.

    A missing arch stays missing. ``None`` must not become the *string*
    ``"none"`` -- that would turn "no arch was supplied", which callers detect by
    emptiness, into a plausible-looking arch name that fails much later with a
    worse message.
    """
    if not isinstance(arch, str) or not arch.strip():
        return ""
    return base_arch_from_target_id(arch.strip().lower())


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
    window_left: int | None = None
    window_right: int | None = None
    window_bottom_right: bool = False
    tensor_layout: str = "auto"  # dense tensors only: "auto" | "bshd" | "bhsd"
    # Right bound and diagonal alignment of the attention band, in the cuDNN / hipDNN
    # vocabulary (``sliding_window`` is the left bound, 0 = unbounded). Consulted
    # only for ``NO_MASK`` / ``SLIDING_WINDOW``: ``TOP_LEFT_CAUSAL`` /
    # ``BOTTOM_RIGHT_CAUSAL`` fix the right bound at 0 and name their own alignment.
    # ``right_bound``: -1 = unbounded (default), 0 = causal, R > 0 = R keys of
    # lookahead. ``diagonal_alignment``: None = unset (TOP_LEFT for NO_MASK /
    # SLIDING_WINDOW, as hipDNN; the kind's own alignment for the causal kinds),
    # 0 = TOP_LEFT, 1 = BOTTOM_RIGHT. Setting it against a causal kind's alignment
    # is an error rather than being ignored. The paged unified kernels apply it per
    # sequence (``seqlen_q`` / ``seqlen_k`` are the batch maxima there, so
    # ``seqlen_q == 1`` means every sequence has one query row), so it matters even
    # when they are equal; dense takes uniform lengths and ignores it at
    # ``seqlen_q == seqlen_k``. The same holds for ``TOP_LEFT_CAUSAL`` vs
    # ``BOTTOM_RIGHT_CAUSAL`` on gfx950.
    right_bound: int = -1
    diagonal_alignment: int | None = None
    kv_block_size: int = 16  # paged KV block_size (modulus); {16,32,64}
    # 0 => auto-resolve at dispatch (_resolve_num_cus). NOTE: auto-resolution is
    # the one part of selection that is NOT host-independent -- on-box it reads
    # the live CU count. Pass a real count (or `target_ctas`) for a reproducible,
    # host-independent decision. See _resolve_num_cus for the full contract.
    num_cus: int = 0
    target_ctas: int = (
        0  # 0 => auto: num_cus*4. >0 pins the routing/segmentation target directly.
    )
    op: str = "attention"
    dtype: str = "fp16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    # Concrete configuration of the candidate ``spec_id`` names (unified tuning
    # or dense). "auto" selects that candidate's default spec; sweep APIs
    # enumerate every feasible value and persist this id with the result.
    tuning_id: str = "auto"
    # The knob overrides that id stands for, as recorded next to it. When set,
    # the candidate rebuilds the spec from them directly (and checks it against
    # tuning_id unless that is "auto") instead of searching its space.
    tuning_knobs: Tuple[Tuple[str, object], ...] = ()
    use_fp8: bool = False
    fp8_fnuz: bool = False
    # Explicit layouts and score features require candidate capability support.
    # "auto" retains each existing candidate's layout convention.
    layout: str = "auto"  # "auto" | "dense" | "ragged" | "paged"
    use_softcap: bool = False
    use_alibi: bool = False
    use_qq_bias: bool = False
    # Dense additive attention bias ``[B|1, H|1, Sq|1, Sk]`` (unit-stride keys)
    # added to the scaled scores; ``attn_bias_dtype`` is "f32" or "q" (the Q
    # dtype). A candidate must declare the "attn_bias" feature.
    use_attn_bias: bool = False
    attn_bias_dtype: str = "f32"
    kv_layout: str = "paged"  # "paged" | "strided" (non-paged decode)
    # Also write the natural-log softmax normalizer (LSE) into a caller-owned
    # FP32 ``tensors["lse"]``. Reported as the ``lse`` feature, so only
    # candidates whose kernels implement it (dense, RDNA WMMA) admit it.
    emit_lse: bool = False
    # Serve runtime head widths from one compiled bucket: ``hdim_q``/``hdim_v``
    # are the bucket maxima, and any widths the kernel takes up to them (RDNA
    # WMMA: multiples of 8) run at launch, as does the ``v`` tensor's head
    # count. Reported as the ``runtime_head_dims`` feature, so only candidates
    # whose kernels implement it (RDNA WMMA) admit it.
    runtime_head_dims: bool = False

    def __post_init__(self) -> None:
        # ONE ``__post_init__``, normalizing both fields. Two of them is not two
        # hooks -- Python keeps the last binding and the earlier one never runs.
        #
        # Callers rebuild pins from stored JSON; a bad value fails here, with
        # its name, rather than as an unhashable key inside a cache.
        object.__setattr__(self, "tuning_knobs", normalize_knobs(self.tuning_knobs))
        # The one place ``arch`` is canonicalized. The class calls itself
        # "Normalized" -- this is what makes that true of the arch, so every
        # downstream comparison can be an exact ``==`` against a base name
        # without each site re-normalizing (and disagreeing about how).
        # ``object.__setattr__`` because the dataclass is frozen.
        object.__setattr__(self, "arch", canonical_arch(self.arch))

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
        # IntEnum and raw-int callers describe the same request/cache identity.
        d["mask_type"] = _parse_attention_mask_type(self.mask_type).value
        # An unset alignment and the value it resolves to are the same request.
        if self.diagonal_alignment is None:
            d["diagonal_alignment"] = int(_implied_alignment(self.mask_type))
        d["layout"] = (
            self.layout.strip().lower() if isinstance(self.layout, str) else ""
        )
        d["tensor_layout"] = (
            self.tensor_layout.strip().lower()
            if isinstance(self.tensor_layout, str)
            else ""
        )
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
        if self.kv_layout == "strided":
            active.add("strided_kv")
        try:
            mask_type = _parse_attention_mask_type(self.mask_type)
        except ValueError:
            # Let _request_errors report the invalid ordinal. In particular, do
            # not silently classify an arbitrary nonzero value as causal.
            mask_type = None
        explicit_window = self.window_left is not None or (
            self.window_right is not None and int(self.window_right) >= 0
        )
        band = None
        if mask_type is not None and _uses_band_dispatch(self, explicit_window):
            try:
                band = normalize_band(self)
            except ValueError:
                pass  # _request_errors reports it
        if band is not None:
            if band.is_causal:
                active.add("causal")
                if band.alignment is Alignment.BOTTOM_RIGHT and int(
                    self.seqlen_q
                ) != int(self.seqlen_k):
                    active.add("causal_bottom_right")
            elif band.is_full:
                active.add("full_attention")
            else:
                active.add("band")
        else:
            if mask_type in (
                AttentionMaskType.TOP_LEFT_CAUSAL,
                AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
            ):
                active.add("causal")
            if mask_type == AttentionMaskType.BOTTOM_RIGHT_CAUSAL and int(
                self.seqlen_q
            ) != int(self.seqlen_k):
                active.add("causal_bottom_right")
        windowed = (
            mask_type == AttentionMaskType.SLIDING_WINDOW
            or int(self.sliding_window) > 0
            or explicit_window
        )
        if windowed:
            active.add("sliding_window")
            if explicit_window:
                active.add("window_bounds")
                right = -1 if self.window_right is None else int(self.window_right)
                if right != 0:
                    active.add("noncausal_window")
        if self.window_right is not None and int(self.window_right) >= 0:
            active.add("window_right")
        if self.emit_lse:
            active.add("lse")
        if self.runtime_head_dims:
            active.add("runtime_head_dims")
        if bool(self.use_sinks):
            active.add("sinks")
        if bool(self.use_fp8):
            active.add("fp8")
        if bool(self.use_softcap):
            active.add("softcap")
        if bool(self.use_alibi):
            active.add("alibi")
        if bool(self.use_qq_bias):
            active.add("qq_bias")
        if bool(self.use_attn_bias):
            active.add("attn_bias")
        layout = self.layout.strip().lower() if isinstance(self.layout, str) else ""
        if layout not in ("auto", ""):
            active.add(f"layout_{layout}")
        tensor_layout = (
            self.tensor_layout.strip().lower()
            if isinstance(self.tensor_layout, str)
            else ""
        )
        if tensor_layout == "bhsd":
            active.add("layout_bhsd")
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

# Feature set of the existing unified backend, not the union of standalone capabilities.
# The attention band. hipDNN / cuDNN describe the mask as (left_bound, right_bound,
# diagonal_alignment): unset = unbounded, ``right_bound = 0`` is causal, and any
# bounded trio that is neither full nor causal is what hipDNN classifies as
# SLIDING_WINDOW. An AttentionRequest carries the same trio as (``sliding_window``,
# ``right_bound``, ``diagonal_alignment``) plus a mask kind:
#   * TOP_LEFT_CAUSAL / BOTTOM_RIGHT_CAUSAL -- right bound 0, alignment from the kind;
#   * NO_MASK / SLIDING_WINDOW -- the trio is read as given, so NO_MASK with a window
#     is a (non-causal) left-only band, exactly as in cuDNN. NO_MASK without bounds
#     is full attention, the SDPA default. SLIDING_WINDOW names a band, so it needs
#     at least one bound.
# The request is normalised once, by ``normalize_band`` below, into an
# ``AttentionBand``; every dispatcher reads that instead of re-deriving the aliases.
# A candidate that cannot compute the band is excluded by its declared features, so
# the request fails with a reason instead of being served a different mask.
# gfx950 implements the whole band in the unified kernels. KNOWN GAP: on the other
# arches the tiled kernels are bottom-right causal only, so ``normalize_band`` keeps
# serving ``NO_MASK`` (the request default) by that causal body and rejects every
# other non-causal band. That is the only place the logical mask depends on the arch;
# the gfx942 tiled / dense kernels implement the full band in the stacked follow-up
# (#13211), which removes it.
_BAND_NEEDS_BOUND_ERROR = (
    "mask_type SLIDING_WINDOW names a band: set sliding_window > 0 and/or "
    "right_bound > 0 (use TOP_LEFT_CAUSAL / BOTTOM_RIGHT_CAUSAL for a causal mask "
    "and NO_MASK for full attention)"
)
_BAND_GFX950_ONLY_ERROR = (
    "a non-causal attention band (right_bound != 0 with NO_MASK / SLIDING_WINDOW) "
    "is implemented on gfx950 only"
)

# ``full_attention``: no left and no right bound (NO_MASK without a window, more than
# one query row). ``band``: any other non-causal band -- a lookahead or a non-causal
# window. A causal window is ``causal`` + ``sliding_window``.
ATTENTION_FEATURES = frozenset(
    {
        "causal",
        "causal_bottom_right",
        "sliding_window",
        "full_attention",
        "band",
        "sinks",
        "fp8",
    }
)


def _request_errors(
    req: OperatorRequest, *, allow_unequal_head_dims: bool = False
) -> list[str]:
    """Structural request errors.

    ``allow_unequal_head_dims`` lets a candidate that supports a Q/K head size
    different from the V/output head size (hdim_q != hdim_v) opt in; every
    other candidate keeps the strict equality requirement.
    """
    if not isinstance(req, AttentionRequest):
        return [f"expected AttentionRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.kv_layout not in ("paged", "strided"):
        errors.append("kv_layout must be paged or strided")
    if req.op != "attention":
        errors.append(f"unsupported op {req.op!r}")
    if type(req.emit_lse) is not bool:
        errors.append(f"emit_lse must be a bool, got {type(req.emit_lse).__name__}")
    if type(req.runtime_head_dims) is not bool:
        errors.append(
            "runtime_head_dims must be a bool, got "
            f"{type(req.runtime_head_dims).__name__}"
        )
    for field in ("batch", "nhead_q", "nhead_k", "seqlen_q", "seqlen_k", "hdim_q"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if allow_unequal_head_dims:
        if int(req.hdim_v) <= 0:
            errors.append("hdim_v must be positive")
    elif req.hdim_q != req.hdim_v:
        errors.append("only hdim_q == hdim_v is supported")
    if int(req.nhead_k) > 0 and int(req.nhead_q) % int(req.nhead_k):
        errors.append("nhead_q must be divisible by nhead_k (GQA grouping)")
    layout = req.layout.strip().lower() if isinstance(req.layout, str) else ""
    if layout not in ("auto", "dense", "ragged", "paged"):
        errors.append(f"unsupported attention layout {req.layout!r}")
    tensor_layout = (
        req.tensor_layout.strip().lower() if isinstance(req.tensor_layout, str) else ""
    )
    if tensor_layout not in ("auto", "bshd", "bhsd"):
        errors.append(f"unsupported dense tensor layout {req.tensor_layout!r}")
    if layout in ("ragged", "paged") and tensor_layout != "auto":
        errors.append("tensor_layout is only valid for dense attention")
    try:
        mask_type = _parse_attention_mask_type(req.mask_type)
    except ValueError as exc:
        errors.append(str(exc))
        mask_type = None
    explicit_left = req.window_left is not None
    explicit_right = req.window_right is not None and int(req.window_right) >= 0
    if int(req.sliding_window) < 0:
        errors.append("sliding_window must be nonnegative")
    for field in ("window_left", "window_right"):
        value = getattr(req, field)
        if value is not None and (int(value) < -1 or int(value) > 0x7FFF_FFFF):
            errors.append(f"{field} must be -1 (unbounded) or a nonnegative I32")
    if explicit_left and int(req.sliding_window) != 0:
        errors.append("sliding_window cannot be combined with explicit window_left")
    # A causal mask plus a left bound is the cuDNN spelling of a causal sliding
    # window: the bounds become (window_left, 0) on the causal diagonal.
    if explicit_left and mask_type not in (
        AttentionMaskType.SLIDING_WINDOW,
        AttentionMaskType.TOP_LEFT_CAUSAL,
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    ):
        errors.append(
            "explicit window_left requires mask_type SLIDING_WINDOW, "
            "TOP_LEFT_CAUSAL, or BOTTOM_RIGHT_CAUSAL"
        )
    if explicit_right and mask_type not in (
        AttentionMaskType.NO_MASK,
        AttentionMaskType.SLIDING_WINDOW,
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    ):
        errors.append(
            "window_right requires mask_type NO_MASK, SLIDING_WINDOW, "
            "or BOTTOM_RIGHT_CAUSAL"
        )
    if bool(req.window_bottom_right) and mask_type != AttentionMaskType.SLIDING_WINDOW:
        errors.append("window_bottom_right requires mask_type=SLIDING_WINDOW")
    if (explicit_left or explicit_right or bool(req.window_bottom_right)) and (
        int(req.right_bound) != -1 or req.diagonal_alignment is not None
    ):
        errors.append(
            "window_left / window_right / window_bottom_right carry their own bounds "
            "and alignment; right_bound and diagonal_alignment must stay unset"
        )
    if mask_type is not None and _uses_band_dispatch(
        req, explicit_left or explicit_right
    ):
        try:
            normalize_band(req)
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

    **``120`` is a develop-inherited placeholder, not a per-arch CU count.**
    Branch 3 returns it for *every* arch outside ``_AUTO_RESOLVE_ARCHS`` --
    gfx1250, gfx90a and anything else alike -- so those arches route against a
    number that describes none of them. That is develop's behaviour and this
    change deliberately preserves it: replacing it with real per-arch counts
    moves 2D/3D routing and ``num_segments`` for every caller leaving
    ``num_cus`` at 0, which wants measurement rather than a refactor (the same
    reasoning as the scope limit below). Until then, a caller dispatching for an
    arch outside ``_AUTO_RESOLVE_ARCHS`` and relying on CU-sensitive routing
    should pass an explicit ``num_cus``; adding an arch to
    ``_AUTO_RESOLVE_ARCHS`` only helps a host that IS that arch.

    **Scope limit of AICK-2146's host-independence guarantee.** Branch 2 is the
    one surviving host read on the selection path, and it is deliberate: it is
    how a same-arch on-box request keeps picking the kernel tuned for the part it
    is actually running on. So the guarantee this dispatcher makes is
    *arch*-independence, not full host-independence:

    * ``num_cus > 0`` (or ``target_ctas > 0``) -- selection is a pure function of
      ``(problem, arch)``. Same inputs, same kernel, on any box or no box.
    * ``num_cus == 0`` and ``arch in _AUTO_RESOLVE_ARCHS`` -- the CU count, and
      therefore 2D/3D routing and ``num_segments``, is still host-derived when the
      host happens to BE that arch. A gfx942 request can route differently on a
      gfx942 box than on a gfx950 box or a CPU box (both of which take branch 3's
      ``120``). ``tests/dispatch/attention/test_host_independent_dispatch.py::
      TestNumCusHostDependence`` pins both halves of this.

    Collapsing branch 2 into a per-arch constant would make selection fully
    host-independent, but it would also change the shipped on-box routing and
    segment counts for every caller that leaves ``num_cus`` at 0 -- a measured
    behaviour change, not a refactor. It is deliberately out of scope here.
    """
    n = int(req.num_cus)
    if n > 0:
        return n
    arch = req.arch
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


def _has_no_mask_kernel(arch: str) -> bool:
    """Arches whose unified kernels have a top-left body. Elsewhere the tiled kernels
    are bottom-right causal only, so an equal-length top-left request is served by the
    bottom-right body (see ``normalize_band``)."""
    return arch.strip().lower() == "gfx950"


# Arches served by the unified / dense candidates, i.e. the ones ``normalize_band``
# governs. Other arches (gfx1151) have their own candidates and, with the explicit
# ``window_left`` / ``window_right`` / ``window_bottom_right`` fields, their own window
# vocabulary: those requests are never band-normalised here.
_BAND_ARCHES = ("gfx942", "gfx950", "gfx1250")


def _uses_band_dispatch(req: "AttentionRequest", explicit_window: bool) -> bool:
    return req.arch.strip().lower() in _BAND_ARCHES and not explicit_window


class Alignment(IntEnum):
    """Which corner the diagonal is anchored to (the cuDNN ``diagonal_alignment``)."""

    TOP_LEFT = 0
    BOTTOM_RIGHT = 1


@dataclass(frozen=True)
class AttentionBand:
    """The normalised attention mask: key ``k`` is visible to the query at diagonal
    position ``d`` iff ``d - k < left_bound`` and ``k <= d + right_bound``.

    ``None`` is unbounded. ``left_bound = W`` therefore sees W keys, counting the
    diagonal. ``alignment`` places ``d``: top-left ``d = q``, bottom-right
    ``d = q + kv_len - q_len`` (per sequence).
    """

    left_bound: "int | None"
    right_bound: "int | None"
    alignment: Alignment

    @property
    def is_full(self) -> bool:
        """No bound on either side (alignment is moot)."""
        return self.left_bound is None and self.right_bound is None

    @property
    def is_causal(self) -> bool:
        """Right bound 0, with or without a window."""
        return self.right_bound == 0

    @property
    def is_real_band(self) -> bool:
        """Neither causal nor full: a lookahead or a non-causal window."""
        return not self.is_causal and not self.is_full

    @property
    def top_left(self) -> bool:
        return self.alignment is Alignment.TOP_LEFT

    @property
    def kernel_right_bound(self) -> int:
        """``UnifiedAttentionProblem.right_bound`` (-1 = unbounded)."""
        return -1 if self.right_bound is None else self.right_bound


def _implied_alignment(mask_type) -> Alignment:
    """The alignment an unset ``diagonal_alignment`` stands for: the causal kinds
    name their own, NO_MASK / SLIDING_WINDOW default to top-left (hipDNN)."""
    try:
        kind = _parse_attention_mask_type(mask_type)
    except ValueError:
        return Alignment.TOP_LEFT
    return (
        Alignment.BOTTOM_RIGHT
        if kind == AttentionMaskType.BOTTOM_RIGHT_CAUSAL
        else Alignment.TOP_LEFT
    )


def normalize_band(
    req: AttentionRequest, *, uniform_lengths: bool = False
) -> AttentionBand:
    """The one place a request's mask aliases are resolved. Raises ``ValueError``
    (with the reason) for a request that names no valid band.

    * Causal kinds fix the right bound at 0; the alignment comes from the kind.
    * ``NO_MASK`` / ``SLIDING_WINDOW`` read the ``(sliding_window, right_bound,
      diagonal_alignment)`` trio as given, so ``NO_MASK`` + window is a non-causal
      left-only band (cuDNN semantics). ``SLIDING_WINDOW`` needs at least one bound.
    * Arches without the band kernels serve ``NO_MASK`` by the legacy causal body
      (``_has_no_mask_kernel``); see the known gap above.
    * Full attention ignores the alignment, and with one query row per sequence
      (``seqlen_q == 1``, single-token decode) it is the same computation as
      bottom-right causal, so it is normalised to that.
    * Top-left and bottom-right coincide when every sequence has ``q_len == kv_len``.
      ``uniform_lengths`` (the dense kernel, which takes uniform lengths) folds them at
      ``seqlen_q == seqlen_k``. The paged unified kernels do not: ``seqlen_q`` /
      ``seqlen_k`` are batch maxima and the kernel derives ``context_len`` per
      sequence at runtime. Arches without a top-left kernel
      (``_has_no_mask_kernel``) fold regardless, which is only right for uniform
      lengths: a known limitation.
    """
    mask_type = _parse_attention_mask_type(req.mask_type)
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    right = int(req.right_bound)
    alignment = req.diagonal_alignment
    window = int(req.sliding_window)
    if right < -1:
        raise ValueError(f"right_bound must be >= -1 (-1 = unbounded), got {right}")
    if alignment is not None and int(alignment) not in (0, 1):
        raise ValueError(
            f"diagonal_alignment must be 0 (TOP_LEFT) or 1 (BOTTOM_RIGHT), got {alignment}"
        )
    left_bound = window if window > 0 else None
    if mask_type in (
        AttentionMaskType.TOP_LEFT_CAUSAL,
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    ):
        if right not in (-1, 0):
            raise ValueError(
                "right_bound applies only to NO_MASK / SLIDING_WINDOW; the causal "
                "mask kinds fix it at 0"
            )
        right_bound = 0
        align = _implied_alignment(mask_type)
        if alignment is not None and Alignment(int(alignment)) is not align:
            raise ValueError(
                f"diagonal_alignment={int(alignment)} contradicts mask_type "
                f"{mask_type.name}, which is {align.name}; leave it unset"
            )
    else:
        if mask_type == AttentionMaskType.SLIDING_WINDOW and window <= 0 and right <= 0:
            raise ValueError(_BAND_NEEDS_BOUND_ERROR)
        if not _has_no_mask_kernel(req.arch):
            if mask_type == AttentionMaskType.NO_MASK and right == -1:
                # KNOWN GAP (see above): the legacy causal body, window included.
                return AttentionBand(left_bound, 0, Alignment.BOTTOM_RIGHT)
            if right != 0:
                raise ValueError(_BAND_GFX950_ONLY_ERROR)
        right_bound = None if right < 0 else right
        align = Alignment.TOP_LEFT if alignment is None else Alignment(int(alignment))
    if left_bound is None and right_bound is None:
        # Full attention: the alignment is moot; one query row sees every key.
        return AttentionBand(
            None, 0 if sq == 1 and not uniform_lengths else None, Alignment.BOTTOM_RIGHT
        )
    if sq == sk and (uniform_lengths or not _has_no_mask_kernel(req.arch)):
        align = Alignment.BOTTOM_RIGHT
    return AttentionBand(left_bound, right_bound, align)


def _attention_band(req: AttentionRequest) -> Tuple[bool, int]:
    """``(causal_top_left, right_bound)`` the unified kernels compute for ``req``.

    Requests the unified family does not serve (other arches, and the explicit
    ``window_*`` fields, which carry their own bounds) get the default mask: the
    problem is then only a geometry carrier, e.g. for the gfx1151 launch grid.
    """
    explicit = req.window_left is not None or (
        req.window_right is not None and int(req.window_right) >= 0
    )
    if not _uses_band_dispatch(req, explicit):
        return False, 0
    band = normalize_band(req)
    return band.top_left, band.kernel_right_bound


def _problem_with_num_cus(
    req: AttentionRequest, num_cus: int
) -> UnifiedAttentionProblem:
    top_left, right_bound = _attention_band(req)
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
        num_cus=int(num_cus),
        target_ctas=int(req.target_ctas),
        causal_top_left=top_left,
        right_bound=right_bound,
    )


def _problem(req: AttentionRequest) -> UnifiedAttentionProblem:
    """The production-routing problem, including the live-device CU policy."""
    return _problem_with_num_cus(req, _resolve_num_cus(req))


def _tuning_problem(req: AttentionRequest) -> UnifiedAttentionProblem:
    """Problem input for an explicit tuning candidate.

    A tuning candidate already fixes path and geometry, so it does not consume
    the live-device CU heuristic used by production routing. Keep its base a
    pure function of the request: honor an explicit ``num_cus`` and otherwise
    take the arch's full-part CU count (the static table the split-K heuristics
    use). Only the split-rule segment count and the dispatcher's K/V layout
    knobs read it, and a partitioned or smaller part passes its own ``num_cus``.
    """
    from rocke.helpers.split_k import _ARCH_NUM_CUS

    num_cus = int(req.num_cus)
    if num_cus <= 0:
        if req.arch not in _ARCH_NUM_CUS:
            raise ValueError(f"no CU count known for {req.arch}; pass num_cus")
        num_cus = _ARCH_NUM_CUS[req.arch]
    return _problem_with_num_cus(req, num_cus)


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
    # Band the kernel computes (see ``UnifiedAttentionProblem``): diagonal alignment
    # and right bound. Top-left and a non-causal right bound are distinct kernel
    # bodies, so they must not collapse onto the bottom-right kernel_name / cache key.
    causal_top_left: bool = False
    right_bound: int = 0
    # Left bound, part of kernel_name() for every mask.
    sliding_window: int = 0

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
        if self.causal_top_left:
            parts.append("tl")
        if self.right_bound != 0:
            parts.append("rbu" if self.right_bound < 0 else f"rb{self.right_bound}")
        if self.sliding_window > 0:
            parts.append(f"sw{self.sliding_window}")
        return kernel_name_join(*parts)


def _dense_kernel_module(arch: str):
    if arch == "gfx950":
        from kernels.gfx950 import attention_dense
    elif arch == "gfx942":
        from kernels.gfx942 import attention_dense
    else:
        raise ValueError(f"no dense attention kernel for arch {arch!r}")
    return attention_dense


@dataclass(frozen=True)
class AttentionTuningSpec:
    """Concrete dispatcher-owned spec for every executable tuned candidate.

    Generic production candidates intentionally return :class:`AttentionSpec`
    and defer geometry. A unified tuning candidate, a dense candidate or an RDNA
    WMMA tuned candidate returns this wrapper instead: the exact arch spec,
    builder choice, compile backend, and optional 3D reduce spec are
    serializable and therefore participate in dispatch/cache identity. ``path``
    is ``"2d"`` / ``"3d"`` for the unified tiled kernels, ``"dense"`` for the
    standalone dense kernel, ``"wmma"`` for the RDNA WMMA prefill and
    ``"wmma_decode"`` for the RDNA split-KV decode (decode + merge kernels).

    For the unified and dense paths it is also the whole launch contract the
    runtime needs: ``run_unified_attention_torch(tuning_spec=...)`` compiles
    :meth:`build` under :meth:`cache_key` and launches with :meth:`launch_grid`
    / :meth:`launch_block`, so the runtime never decodes ``builder_kind``. The
    RDNA ``"wmma"`` / ``"wmma_decode"`` specs launch through their candidate's
    grid (``gfx1151.py``), which depends on the live device, so
    :meth:`launch_grid` refuses them. The dense runner reads ``kernel_spec``
    directly. ``allow_unsupported`` skips the unified runtime's problem-shape
    support check.

    Two identities, for two consumers. ``tuning_id`` / ``config_key`` name the
    configuration -- ``variant_id`` plus the canonical ``knobs`` -- and are the
    same on every problem (see :mod:`.identity`). :meth:`identity` names the
    compiled kernel(s), problem and runtime addressing included, and is what
    ``KernelId.spec_hash`` hashes. Neither covers ``candidate_name`` or
    ``allow_unsupported``.
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
    variant_id: str = ""
    config_key: str = ""
    knobs: Tuple[Tuple[str, object], ...] = ()
    allow_unsupported: bool = False

    def identity(self) -> dict:
        """Explicit payload of the compiled kernel(s): :meth:`cache_key`, with
        classes named by qualified name so it serializes stably."""
        return {"v": 1, "path": self.path, "kernel": jsonable(self.cache_key())}

    def kernel_name(self) -> str:
        return self.kernel_spec.kernel_name()

    def cache_key(self) -> Tuple:
        """Field-complete launcher-cache identity of the kernel(s) built."""
        if self.path == "dense":
            from kernels.common.attention_dense_spec import attention_dense_cache_key

            return attention_dense_cache_key(self.kernel_spec, arch=self.arch)

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
        """IR for this spec: one kernel for 2D, dense and RDNA prefill;
        ``(segment, reduce)`` for 3D; ``(decode, merge)`` for RDNA decode."""
        from .tuning_specs import (
            build_explicit_attention_2d,
            build_explicit_attention_3d,
        )

        arch = arch or self.arch
        if self.path == "dense":
            return _dense_kernel_module(self.arch).build_attention_dense(
                self.kernel_spec, arch=arch
            )
        if self.path == "wmma":
            from kernels.gfx1151.wmma_fmha_fwd_v2 import build_wmma_fmha_fwd_v2

            return build_wmma_fmha_fwd_v2(self.kernel_spec, arch=arch)
        if self.path == "wmma_decode":
            from .gfx1151 import build_decode_kernels

            return build_decode_kernels(self.kernel_spec, arch)
        if self.path == "3d":
            return build_explicit_attention_3d(
                self.kernel_spec, self.reduce_spec, arch=arch
            )
        if self.builder_kind == "gfx942_4warp_gqa":
            from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

            return build_gfx942_4warp_gqa(self.kernel_spec, arch=arch)
        return build_explicit_attention_2d(self.kernel_spec, arch=arch)

    def launch_grid(
        self, problem: UnifiedAttentionProblem | None = None
    ) -> Tuple[int, int, int]:
        ks = self.kernel_spec
        if self.path == "dense":
            # The dense spec bakes (or declares) its whole problem shape.
            return tuple(_dense_kernel_module(self.arch).attention_dense_grid(ks))
        if self.path in ("wmma", "wmma_decode"):
            # The RDNA candidates own their launch grid (gfx1151.py).
            raise ValueError(
                f"{self.path!r} specs launch through their candidate's grid"
            )
        if self.path == "3d":
            from kernels.common.attention_unified import _num_q_blocks

            block_q = max(1, 16 // problem.num_queries_per_kv)
            qblocks = _num_q_blocks(problem, block_q, decode_grid=ks.use_decode_grid)
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
        if self.path == "dense":
            return tuple(
                _dense_kernel_module(self.arch).attention_dense_block(self.kernel_spec)
            )
        if self.path == "wmma":
            return (int(self.kernel_spec.threads), 1, 1)
        if self.path == "wmma_decode":
            return (int(self.kernel_spec.block_size), 1, 1)
        if self.path == "3d":
            return (64, 1, 1)
        if self.builder_kind == "gfx942_4warp_gqa":
            return (256, 1, 1)
        return (64 * int(self.kernel_spec.num_warps), 1, 1)

    def with_num_kv_blocks(self, num_kv_blocks: int) -> "AttentionTuningSpec":
        """Specialize i32/i64 paged addressing once the physical cache is known.

        Addressing width is a runtime specialization, not a configuration: it
        changes :meth:`identity` (a different binary) but never ``tuning_id``,
        so an id recorded after binding still replays from the request.
        """
        count = int(num_kv_blocks)
        if count < 0:
            raise ValueError("num_kv_blocks must be non-negative")
        if self.path == "dense":
            return self  # contiguous K/V: nothing to retarget
        kernel_spec = self.kernel_spec
        if not hasattr(kernel_spec, "use_i64_kv_addr"):
            return replace(self, num_kv_blocks=count)
        elem_bytes = 1 if kernel_spec.kv_storage_dtype == "fp8e4m3" else 2
        block_stride = (
            int(kernel_spec.block_size)
            * int(kernel_spec.num_kv_heads)
            * int(kernel_spec.head_size)
            * elem_bytes
        )
        use_i64 = count > 0 and count * block_stride > 0x8000_0000
        return replace(
            self,
            kernel_spec=replace(kernel_spec, use_i64_kv_addr=use_i64),
            num_kv_blocks=count,
        )
