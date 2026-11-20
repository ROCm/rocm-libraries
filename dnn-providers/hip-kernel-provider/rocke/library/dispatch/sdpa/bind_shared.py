# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pieces the SDPA unified and dense bindings share.

The ``SdpaRequest`` -> ``AttentionRequest`` translation, the band comparison
both gates use to refuse a band their kernels would not compute, the value
checks on carried hints, and the per-operand views the bindings copy through.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from rocke.runtime.torch_interop import resolve_stream, torch_dtype

from dispatch.attention import AttentionMaskType, AttentionRequest
from dispatch.attention.common import Alignment, AttentionBand

if TYPE_CHECKING:
    from .common import SdpaRequest, SdpaTensor

__all__ = [
    "TORCH_DTYPES",
    "Operand",
    "asked_band",
    "band_text",
    "check_dtype",
    "declined_table",
    "host_ints",
    "keeps_asked_keys",
    "kept_keys",
    "launch_stream",
    "nonfinite_scale_reject",
    "to_attention_request",
    "usable_in_place",
    "value_rejects",
]

# ``SdpaRequest`` dispatch selectors; only ``"auto"`` is served.
_DISPATCH_PINS = ("algorithm", "spec_id", "tuning_id")

# Torch dtype names, resolved through ``rocke.runtime.torch_interop.torch_dtype``:
# the I/O dtypes, FP32 stats, and the integer lengths and ragged offsets.
TORCH_DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "int32": "int32",
    "int64": "int64",
}

# Base-pointer alignment, in bytes, the unified and dense kernels declare for
# the operands a binding hands them in place.
_KERNEL_ALIGNMENT = 16


def to_attention_request(req: SdpaRequest) -> AttentionRequest:
    """The ``AttentionRequest`` the attention candidates admit for ``req``.

    ``SdpaRequest.left_bound`` / ``right_bound`` / ``diagonal`` are in the
    same cuDNN/hipDNN band vocabulary as ``AttentionRequest.right_bound`` /
    ``diagonal_alignment``, and hipDNN ``left_bound = L`` keeps the L keys
    ending at the diagonal, which is what ``sliding_window = L`` sees, so the
    left bound passes through unchanged.
    """
    features = req.features()
    if "causal_tl" in features:
        mask_type = AttentionMaskType.TOP_LEFT_CAUSAL
    elif "causal_br" in features:
        mask_type = AttentionMaskType.BOTTOM_RIGHT_CAUSAL
    elif "window" in features:
        mask_type = AttentionMaskType.SLIDING_WINDOW
    else:
        mask_type = AttentionMaskType.NO_MASK

    is_window = mask_type == AttentionMaskType.SLIDING_WINDOW
    return AttentionRequest(
        batch=req.batch,
        nhead_q=req.h_q,
        nhead_k=req.h_k,
        seqlen_q=req.s_q,
        seqlen_k=req.s_kv,
        hdim_q=req.d_qk,
        hdim_v=req.d_v,
        arch=req.arch,
        dtype=req.dtype,
        mask_type=mask_type,
        sliding_window=req.left_bound if is_window and req.left_bound > 0 else 0,
        right_bound=req.right_bound if is_window else -1,
        diagonal_alignment=(
            (1 if req.diagonal == "bottom_right" else 0) if is_window else None
        ),
        use_sinks=req.sink_token is not None,
        use_alibi=req.alibi,
        emit_lse=req.stats is not None,
    )


# --- band ----------------------------------------------------------------------


def kept_keys(s_q: int, s_kv: int, top_left: bool, left: int, right: int):
    """Per query row, the first and last key a band keeps (``-1`` unbounded;
    ``left = W`` is FlashAttention's window and keeps W + 1 keys). A row that keeps no key is
    ``(0, -1)``, so two bands keep the same keys iff both arrays are equal.

    A band keeps one contiguous run of keys per row, so this is O(s_q) where
    the dense s_q x s_kv mask would be quadratic.
    """
    d = np.arange(s_q, dtype=np.int64) + (0 if top_left else s_kv - s_q)
    first = np.maximum(d - left, 0) if left >= 0 else np.zeros_like(d)
    last = np.minimum(d + right, s_kv - 1) if right >= 0 else np.full_like(d, s_kv - 1)
    empty = first > last
    return np.where(empty, 0, first), np.where(empty, -1, last)


def asked_band(req: SdpaRequest) -> AttentionBand:
    """The band ``req`` names, in ``AttentionBand`` terms (``left_bound = L``
    keeps L keys there, as in hipDNN)."""
    return AttentionBand(
        left_bound=None if req.left_bound < 0 else req.left_bound,
        right_bound=None if req.right_bound < 0 else req.right_bound,
        alignment=(
            Alignment.TOP_LEFT if req.diagonal == "top_left" else Alignment.BOTTOM_RIGHT
        ),
    )


def keeps_asked_keys(req: SdpaRequest, band: AttentionBand) -> bool:
    """Whether a kernel computing ``band`` keeps the keys ``req`` asks for.

    Bands that differ in form but keep the same keys are equal: full vs
    bottom-right causal at ``s_q == 1`` always, and any other pair (e.g.
    top-left vs bottom-right at ``s_q == s_kv``) only for fixed lengths, since
    with per-sequence lengths the diagonal moves per batch.
    """
    asked = asked_band(req)
    if (band.left_bound, band.right_bound) == (
        asked.left_bound,
        asked.right_bound,
    ) and (band.alignment == asked.alignment or asked.is_full):
        return True
    # One query row per sequence: bottom-right causal keeps every key of each
    # sequence, whatever its length, so it is full attention.
    if (
        asked.is_full
        and req.s_q == 1
        and band.left_bound is None
        and band.right_bound == 0
        and band.alignment == Alignment.BOTTOM_RIGHT
    ):
        return True

    features = req.features()
    if "padded" in features or "ragged" in features:
        return False

    def kept(b: AttentionBand):
        return kept_keys(
            req.s_q,
            req.s_kv,
            b.alignment == Alignment.TOP_LEFT,
            -1 if b.left_bound is None else b.left_bound - 1,
            -1 if b.right_bound is None else b.right_bound,
        )

    (first_a, last_a), (first_b, last_b) = kept(band), kept(asked)
    return np.array_equal(first_a, first_b) and np.array_equal(last_a, last_b)


def band_text(b: AttentionBand) -> str:
    if b.is_full:
        return "full attention"
    return f"left={b.left_bound} right={b.right_bound} " f"{b.alignment.name.lower()}"


# --- declines --------------------------------------------------------------


# Reason text for every feature a routed family (unified, dense) might
# decline, with ``{family}`` filled in by :func:`declined_table`. A reason
# tagged ``NOT_YET_IMPLEMENTED`` names an implementation shortcut (a missing
# reference oracle, an unmapped schema) that nothing about the hardware or
# ABI rules out, as opposed to a permanent kernel/ABI limit; keep a feature's
# wording identical across families unless the kernels genuinely differ on
# it (see each gate's own ``overrides``, e.g. dense's ``d_v_differs``).
_SHARED_DECLINE_REASONS = {
    "kv_heads_differ": "the {family} request has one nhead_k for K and V",
    "dtype_mixed": "the {family} kernels read and write one dtype",
    "compute_dtype": "the {family} kernels accumulate in fp32 only",
    "scale_tensor": "NOT_YET_IMPLEMENTED: the {family} kernels take only a scalar scale",
    "bias": "NOT_YET_IMPLEMENTED: no {family} candidate declares additive-bias support",
    "bias_broadcast": "NOT_YET_IMPLEMENTED: no {family} candidate declares additive-bias support",
    "alibi": "NOT_YET_IMPLEMENTED: no {family} candidate declares alibi support",
    "sinks": "NOT_YET_IMPLEMENTED: no reference oracle for sink tokens",
    "dropout": "NOT_YET_IMPLEMENTED: the {family} kernels have no dropout",
    "paged": "NOT_YET_IMPLEMENTED: the page-table format is not mapped (see SdpaPaged)",
    "block_mask": "NOT_YET_IMPLEMENTED: the {family} kernels take no block mask",
    "fp8": "NOT_YET_IMPLEMENTED: the {family} kernels take no descale/scale tensors",
    "fp8_s_o": "NOT_YET_IMPLEMENTED: the {family} kernels take no softmax/output quantization",
    "max_sum_exp": "the {family} kernels write only LSE, not max and sum-exp",
}


def declined_table(
    family: str, order: Sequence[str], overrides: Mapping[str, str] = {}
) -> dict:
    """``{feature: "feature: reason"}`` for every feature in ``order``, in
    that order (a gate checks its declines in this order, so the reason it
    reports for a request that fails more than one is deterministic).

    A feature not in ``overrides`` gets the shared reason text with
    ``family`` filled in; one in ``overrides`` gets that text verbatim,
    already carrying its own feature prefix, for a reason specific to this
    family (e.g. dense's ``d_v_differs``, which unified does not decline at
    all -- it carries that feature instead).
    """
    return {
        feature: (
            overrides[feature]
            if feature in overrides
            else f"{feature}: {_SHARED_DECLINE_REASONS[feature].format(family=family)}"
        )
        for feature in order
    }


# --- carried hints -------------------------------------------------------------


def nonfinite_scale_reject(req: SdpaRequest, family: str) -> tuple[bool, str] | None:
    """Decline a NaN or infinite scale; each gate adds its kernels' own range."""
    scale = float(req.scale)
    if not math.isfinite(scale):
        return False, f"scale: the {family} kernels need a finite scale, got {scale}"
    return None


def value_rejects(req: SdpaRequest, family: str) -> tuple[bool, str] | None:
    """Declines for carried features whose value the ``family`` launch cannot
    honor: an MMA mode other than the I/O dtype, a composite implementation
    hint, a dispatch pin, and a padding mask with no lengths."""
    if req.mma_core_mode is not None and req.mma_core_mode != req.dtype:
        return False, (
            f"mma_core_mode: the {family} kernels run MMA in {req.dtype}, "
            f"not {req.mma_core_mode}"
        )
    if req.implementation == "composite":
        return False, "implementation_hint: composite was requested"
    # The launch picks its own kernel (no tuning spec), so a pin would be
    # admitted and then ignored.
    for pin in _DISPATCH_PINS:
        if getattr(req, pin) != "auto":
            return False, (
                f"dispatch_pin: {pin}={getattr(req, pin)!r} is pinned; the "
                f"{family} launch takes only the automatic route"
            )
    if req.tuning_knobs:
        return False, (
            f"dispatch_pin: tuning_knobs are set; the {family} launch takes only "
            "the automatic route"
        )
    if req.padding_mask and req.seq_len_q is None and req.seq_len_kv is None:
        return False, "padded: padding_mask is set but no seq_len tensor is given"
    return None


# --- launch --------------------------------------------------------------------


def launch_stream(stream: Any, device: Any, family: str) -> int:
    """The stream a binding's ``launch(stream=...)`` runs on: Torch's current
    stream, whether the caller leaves ``stream`` unset (``None`` or ``0``) or
    names it by handle, as graph capture does.

    Any other stream is refused. The binding's ``pack()`` / ``unpack()`` are
    Torch copies on the current stream, and its workspace buffers were
    allocated against the stream current at bind time (see
    ``resolve_stream``), so launching only the kernel on another stream would
    race on those buffers.
    """
    current = resolve_stream(0, device=device)
    if stream is None or int(stream) in (0, current):
        return current
    raise NotImplementedError(
        f"the {family} SDPA binding cannot honor stream={stream!r}: it packs "
        "and unpacks on the current stream, and its workspace buffers belong "
        "to it; make the stream current (`with torch.cuda.stream(s):`) around "
        "launch() instead"
    )


# --- operands ------------------------------------------------------------------


def host_ints(tensor) -> list[int]:
    return [int(x) for x in tensor.detach().reshape(-1).cpu().tolist()]


def check_dtype(name: str, tensor, desc: SdpaTensor) -> None:
    """Refuse a tensor whose dtype is not the one its descriptor names.

    The one dtype rule for every tensor a binding reads: operands, stats,
    ``seq_len_*`` and ragged offsets. A lenient read would bind the wrong
    data silently: ``int()`` truncates a float offset (0.5 -> 0), and a copy
    into an FP32 workspace casts FP16 stats.
    """
    want = TORCH_DTYPES.get(desc.dtype)
    if want is None:
        raise ValueError(f"{name}: no torch dtype for the request's {desc.dtype}")
    want = torch_dtype(want)
    if tensor.dtype != want:
        raise ValueError(
            f"{name} must be {want} as the request describes, got {tensor.dtype}"
        )


def usable_in_place(view) -> bool:
    """Whether a binding may hand ``view`` to a kernel as is: contiguous, with
    the 16-byte-aligned base pointer the kernels declare (a contiguous view
    one FP16 element into its allocation is not). Anything else is copied
    through a workspace buffer."""
    return view.is_contiguous() and view.data_ptr() % _KERNEL_ALIGNMENT == 0


class Operand:
    """Per-batch ``(rows, heads, d)`` views of one operand.

    A dense operand is its logical BHSD tensor, with the request's strides. A
    ragged operand is its packed buffer plus ``B + 1`` offsets: batch ``b``
    starts at element ``offsets[b] * ragged_offset_multiplier``, as hipDNN
    reads it, and the batch stride is unused.
    """

    def __init__(self, name: str, desc: SdpaTensor, tensor, offsets=None) -> None:
        batch, heads, extent, width = desc.dims
        check_dtype(name, tensor, desc)
        self.name, self.desc, self.tensor = name, desc, tensor
        self.heads, self.extent, self.width = heads, extent, width
        self.starts: list[int] | None = None
        self.lengths: list[int] | None = None
        if not desc.ragged:
            if tuple(tensor.shape) != tuple(desc.dims):
                raise ValueError(
                    f"{name}: tensor shape {tuple(tensor.shape)} is not the "
                    f"request's dims {desc.dims}"
                )
            for n, got, want in zip(desc.dims, tensor.stride(), desc.strides):
                if n > 1 and got != want:
                    raise ValueError(
                        f"{name}: tensor strides {tuple(tensor.stride())} are not "
                        f"the request's strides {desc.strides}"
                    )
            return
        if offsets is None:
            raise ValueError(f"ragged operand {name!r} needs its ragged offsets")
        if not tensor.is_contiguous():
            raise ValueError(f"{name}: a ragged buffer must be contiguous")
        _, s_h, s_s, s_d = desc.strides
        mult = int(desc.ragged_offset_multiplier)
        check_dtype(f"{name}: ragged offsets", offsets, desc.ragged_offset)
        self.flat = tensor.reshape(-1)
        self.starts = [o * mult for o in host_ints(offsets)]
        self.row_strides = (s_s, s_h, s_d)
        if len(self.starts) != batch + 1 or any(x % s_s for x in self.starts):
            raise ValueError(
                f"{name}: ragged offsets must be B + 1 = {batch + 1} whole sequence "
                f"rows, got element starts {self.starts}"
            )
        self.lengths = [(e - s) // s_s for s, e in pairwise(self.starts)]
        if any(n < 0 or n > extent for n in self.lengths):
            raise ValueError(
                f"{name}: ragged lengths {self.lengths} must be in [0, {extent}]"
            )

    def rows(self, b: int, n: int):
        """Batch ``b``'s first ``n`` rows as ``(n, heads, d)``."""
        if self.starts is None:
            return self.tensor[b, :, :n, :].transpose(0, 1)
        return self.flat.as_strided(
            (n, self.heads, self.width),
            self.row_strides,
            self.flat.storage_offset() + self.starts[b],
        )

    def packed_view(self, cu: list[int]):
        """A ``(total, heads, d)`` view when the rows already sit packed in
        token order (packed BSHD at full length, or contiguous ragged) and
        :func:`usable_in_place` allows it, else ``None`` (the caller copies
        through its workspace).
        """
        total, h, d = cu[-1], self.heads, self.width
        if self.starts is None:
            if any(b - a != self.extent for a, b in pairwise(cu)):
                return None
            bshd = self.tensor.transpose(1, 2)
            return bshd.reshape(total, h, d) if usable_in_place(bshd) else None
        if self.row_strides != (h * d, d, 1) or any(
            s != c * h * d for s, c in zip(self.starts, cu)
        ):
            return None
        view = self.flat[: total * h * d].view(total, h, d)
        return view if usable_in_place(view) else None

    def paged_view(self, lengths: list[int], block: int):
        """A ``(B * S / block, block, heads, d)`` paged-cache view, for the
        identity block table, when the rows already sit there: every batch at
        full length (``lengths`` all equal the extent, a nonzero multiple of
        ``block``), packed in token order, and :func:`usable_in_place`. Else
        ``None`` and the caller copies into a zero-filled cache.

        Full length only: a kernel may read a KV tile's masked trailing rows
        (see the gfx942 4-warp body), and a masked row still reaches P @ V, so
        a NaN the caller left past a sequence's length would leak into O. The
        copy path's zero fill is what makes those rows safe.
        """
        h, d, extent = self.heads, self.width, self.extent
        if extent == 0 or extent % block or any(n != extent for n in lengths):
            return None
        if self.starts is None:
            rows = self.tensor.transpose(1, 2)
        else:
            if self.row_strides != (h * d, d, 1) or any(
                s != b * extent * h * d for b, s in enumerate(self.starts)
            ):
                return None
            rows = self.flat[: len(lengths) * extent * h * d]
        if not usable_in_place(rows):
            return None
        return rows.reshape(-1, block, h, d)
