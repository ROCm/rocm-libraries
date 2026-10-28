# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SDPA admission and binding for the unified attention candidates.

Translates a :class:`~dispatch.sdpa.common.SdpaRequest` into the
``dispatch/attention/`` request the unified candidates already consume and
asks :data:`dispatch.attention.ATTENTION_ROUTE_REGISTRY` whether a unified
route candidate serves it. The existing ``dispatch/attention/`` candidates
are not edited.

This module adds no shape, dtype or head-size rule of its own: "can it run"
is the candidates' own ``admits()``, so new kernel coverage reaches SDPA
routing without touching this file. What it does add is the guarantee that
nothing is admitted and then dropped: every feature in ``SDPA_FEATURES`` is
either carried (:data:`SDPA_UNIFIED_FEATURES`) or declined with a reason, a
request is declined when the band the unified kernels would compute is not
the band it asked for, and a request that pins a dispatch choice is declined.

:func:`sdpa_unified_bind` binds an admitted request's tensors to a launch on
the production route (``run_unified_attention_torch`` with no tuning spec).
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from rocke.core.arch import known_arches
from rocke.dispatch.core import TorchBinding

from dispatch.attention import (
    ATTENTION_ROUTE_REGISTRY,
    UNIFIED_HEAD_SIZES,
    AttentionMaskType,
    AttentionRequest,
)
from dispatch.attention.common import (
    Alignment,
    AttentionBand,
    _attention_band,
    _problem,
)

if TYPE_CHECKING:
    from rocke.dispatch.core import KernelCandidate

    from .common import SdpaRequest, SdpaTensor

__all__ = [
    "SDPA_UNIFIED_FEATURES",
    "sdpa_unified_admits",
    "sdpa_unified_bind",
    "sdpa_unified_candidates",
]

# What the unified launch path routes among (``run_unified_attention_torch``
# with no tuning spec). Tuning geometries, dense, RDNA WMMA and strided-KV
# decode candidates are other launch paths.
_UNIFIED_ALGORITHMS = frozenset(
    {"unified_2d", "unified_3d", "d256_decode", "d256_gfx950", "dense_pipe"}
)
# Only the split-KV (3D) path writes LSE.
_LSE_ALGORITHMS = frozenset({"unified_3d", "d256_decode"})
# ``SdpaRequest`` dispatch selectors; only ``"auto"`` is served.
_DISPATCH_PINS = ("algorithm", "spec_id", "tuning_id")

# The binding's ``supports_features``: what the unified launch path carries.
# Layouts other than packed BSHD are carried by copying Q/O to and from a
# packed buffer; K/V are always copied into the paged cache.
SDPA_UNIFIED_FEATURES = frozenset(
    {
        "gqa",
        "d_v_differs",
        "layout_bshd",
        "layout_bhsd",
        "layout_strided",
        "kv_layout_differs",
        "o_layout_differs",
        "batch_stride",
        "d_stride",
        "unaligned",
        "causal_tl",
        "causal_br",
        "window",
        "window_br",
        "window_right",
        "padded",
        "ragged",
        "lse",
        "mma_core_mode",
        "implementation_hint",
    }
)

# Every other feature, with the reason it is declined. Checked in this order,
# so the reported reason is deterministic.
_DECLINED = {
    "kv_heads_differ": "kv_heads_differ: the unified request has one nhead_k for K and V",
    "dtype_mixed": "dtype_mixed: the unified kernels read and write one dtype",
    "compute_dtype": "compute_dtype: the unified kernels accumulate in fp32 only",
    "scale_tensor": "scale_tensor: the unified kernels take only a scalar scale",
    "bias": "bias: no unified candidate declares additive-bias support yet",
    "bias_broadcast": "bias_broadcast: no unified candidate declares additive-bias support yet",
    "alibi": "alibi: no unified candidate declares alibi support yet",
    "sinks": "sinks: no reference oracle for sink tokens yet",
    "dropout": "dropout: the unified kernels have no dropout",
    "paged": "paged: the page-table format is not mapped yet (see SdpaPaged)",
    "block_mask": "block_mask: the unified kernels take no block mask",
    "fp8": "fp8: the unified kernels take no descale/scale tensors",
    "fp8_s_o": "fp8_s_o: the unified kernels take no softmax/output quantization",
    "max_sum_exp": "max_sum_exp: the unified kernels write only LSE, not max and sum-exp",
}


def _to_attention_request(req: SdpaRequest) -> AttentionRequest:
    """The ``AttentionRequest`` the unified candidates admit for ``req``.

    ``SdpaRequest.left_bound`` / ``right_bound`` / ``diagonal`` are in the
    same cuDNN/hipDNN band vocabulary as ``AttentionRequest.right_bound`` /
    ``diagonal_alignment``. The one adjustment is ``sliding_window``: it sees
    W keys, while hipDNN ``left_bound = W`` keeps W + 1, so lower with
    ``left_bound + 1``.
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
        sliding_window=(req.left_bound + 1) if is_window and req.left_bound >= 0 else 0,
        right_bound=req.right_bound if is_window else -1,
        diagonal_alignment=(
            (1 if req.diagonal == "bottom_right" else 0) if is_window else None
        ),
        use_sinks=req.sink_token is not None,
        use_alibi=req.alibi,
        emit_lse=req.stats is not None,
    )


def _kept_keys(s_q: int, s_kv: int, top_left: bool, left: int, right: int):
    """Per query row, the first and last key a hipDNN band keeps (``-1``
    unbounded, ``left = W`` keeps W + 1 keys). A row that keeps no key is
    ``(0, -1)``, so two bands keep the same keys iff both arrays are equal.

    A band keeps one contiguous run of keys per row, so this is O(s_q) where
    the dense s_q x s_kv mask would be quadratic.
    """
    d = np.arange(s_q, dtype=np.int64) + (0 if top_left else s_kv - s_q)
    first = np.maximum(d - left, 0) if left >= 0 else np.zeros_like(d)
    last = np.minimum(d + right, s_kv - 1) if right >= 0 else np.full_like(d, s_kv - 1)
    empty = first > last
    return np.where(empty, 0, first), np.where(empty, -1, last)


def _computed_band(req: SdpaRequest) -> AttentionBand:
    """The band the unified kernels compute for ``req``.

    Read from ``_attention_band``, which is what ``_problem`` hands every
    unified route candidate, not from ``normalize_band``: outside the band
    arches the problem is always built bottom-right causal, whatever band the
    request names. ``normalize_band`` raises for a band the band arches cannot
    serve.
    """
    attn = _to_attention_request(req)
    top_left, right_bound = _attention_band(attn)
    return AttentionBand(
        left_bound=attn.sliding_window if attn.sliding_window > 0 else None,
        right_bound=None if right_bound < 0 else right_bound,
        alignment=Alignment.TOP_LEFT if top_left else Alignment.BOTTOM_RIGHT,
    )


def _band_reject(req: SdpaRequest) -> tuple[bool, str] | None:
    """Decline when the unified kernels would compute a different band.

    Bands that differ in form but keep the same keys are equal: full vs
    bottom-right causal at ``s_q == 1`` always, and any other pair (e.g.
    top-left vs bottom-right at ``s_q == s_kv``) only for fixed lengths, since
    with per-sequence lengths the diagonal moves per batch.
    """
    try:
        band = _computed_band(req)
    except ValueError as exc:
        return False, f"band: {exc}"

    asked = AttentionBand(
        left_bound=None if req.left_bound < 0 else req.left_bound + 1,
        right_bound=None if req.right_bound < 0 else req.right_bound,
        alignment=(
            Alignment.TOP_LEFT if req.diagonal == "top_left" else Alignment.BOTTOM_RIGHT
        ),
    )
    if (band.left_bound, band.right_bound) == (
        asked.left_bound,
        asked.right_bound,
    ) and (band.alignment == asked.alignment or asked.is_full):
        return None
    # One query row per sequence: bottom-right causal keeps every key of each
    # sequence, whatever its length, so it is full attention.
    if (
        asked.is_full
        and req.s_q == 1
        and band.left_bound is None
        and band.right_bound == 0
        and band.alignment == Alignment.BOTTOM_RIGHT
    ):
        return None

    features = req.features()
    if "padded" not in features and "ragged" not in features:

        def kept(b: AttentionBand):
            return _kept_keys(
                req.s_q,
                req.s_kv,
                b.alignment == Alignment.TOP_LEFT,
                -1 if b.left_bound is None else b.left_bound - 1,
                -1 if b.right_bound is None else b.right_bound,
            )

        (first_a, last_a), (first_b, last_b) = kept(band), kept(asked)
        if np.array_equal(first_a, first_b) and np.array_equal(last_a, last_b):
            return None

    return False, (
        f"band: {req.arch} unified kernels compute {_band_text(band)} for "
        f"requested {_band_text(asked)}"
    )


def _band_text(b: AttentionBand) -> str:
    if b.is_full:
        return "full attention"
    return f"left={b.left_bound} right={b.right_bound} " f"{b.alignment.name.lower()}"


def _sdpa_unified_rejects(req: SdpaRequest) -> tuple[bool, str] | None:
    """Return ``(False, reason)`` if this request cannot be served, else ``None``.

    The single gate shared by :func:`sdpa_unified_admits` and
    :func:`sdpa_unified_candidates`, so the two cannot drift. The request is
    validated (head_dim, arch) first, so a malformed request reports its own
    cause rather than a feature it also happens to use.
    """
    if req.d_qk not in UNIFIED_HEAD_SIZES or req.d_v not in UNIFIED_HEAD_SIZES:
        return False, (
            f"head_dim (d_qk={req.d_qk}, d_v={req.d_v}) is not a unified head "
            f"size; supported: {sorted(UNIFIED_HEAD_SIZES)}"
        )
    if req.arch not in known_arches():
        return False, f"unknown arch {req.arch!r}"

    features = req.features()
    for feature, reason in _DECLINED.items():
        if feature in features:
            return False, reason

    if req.mma_core_mode is not None and req.mma_core_mode != req.dtype:
        return False, (
            f"mma_core_mode: the unified kernels run MMA in {req.dtype}, "
            f"not {req.mma_core_mode}"
        )
    if req.implementation == "composite":
        return False, "implementation_hint: composite was requested"
    # The unified launch routes on its own (no tuning spec), so a pin would be
    # admitted and then ignored.
    for pin in _DISPATCH_PINS:
        if getattr(req, pin) != "auto":
            return False, (
                f"dispatch_pin: {pin}={getattr(req, pin)!r} is pinned; the "
                f"unified launch takes only the automatic route"
            )
    if req.tuning_knobs:
        return False, (
            "dispatch_pin: tuning_knobs are set; the unified launch takes only "
            "the automatic route"
        )
    if req.padding_mask and req.seq_len_q is None and req.seq_len_kv is None:
        return False, "padded: padding_mask is set but no seq_len tensor is given"

    return _band_reject(req)


def _unified_supported(req: SdpaRequest) -> tuple[KernelCandidate, ...]:
    """Unified route candidates that admit the translated request."""
    supported = tuple(
        c
        for c in ATTENTION_ROUTE_REGISTRY.supported(_to_attention_request(req))
        if c.algorithm in _UNIFIED_ALGORITHMS
    )
    if req.stats is not None:
        supported = tuple(c for c in supported if c.algorithm in _LSE_ALGORITHMS)
    return supported


def sdpa_unified_admits(req: SdpaRequest) -> tuple[bool, str]:
    """Top-level SDPA admission check for the unified family.

    Returns ``(True, "ok")`` only when nothing in
    :func:`_sdpa_unified_rejects` applies and at least one unified route
    candidate admits the translated request.
    """
    rejected = _sdpa_unified_rejects(req)
    if rejected is not None:
        return rejected
    if _unified_supported(req):
        return True, "ok"
    path = "split-KV (LSE) " if req.stats is not None else ""
    return False, (
        f"no unified {path}candidate covers arch={req.arch!r} "
        f"dtype={req.dtype!r} d_qk={req.d_qk} d_v={req.d_v} "
        f"s_q={req.s_q} s_kv={req.s_kv}"
    )


def sdpa_unified_candidates(req: SdpaRequest) -> tuple[str, ...]:
    """Names of the unified candidates that serve ``req``, for tests and triage.

    In registry priority order; empty whenever :func:`sdpa_unified_admits`
    would return ``False``.
    """
    if _sdpa_unified_rejects(req) is not None:
        return ()
    return tuple(c.name for c in _unified_supported(req))


# --- binding -------------------------------------------------------------------

_TORCH_DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


def _host_ints(tensor) -> list[int]:
    return [int(x) for x in tensor.detach().reshape(-1).cpu().tolist()]


class _Operand:
    """Per-batch ``(rows, heads, d)`` views of one operand.

    A dense operand is its logical BHSD tensor, with the request's strides. A
    ragged operand is its packed buffer plus ``B + 1`` offsets: batch ``b``
    starts at element ``offsets[b] * ragged_offset_multiplier``, as hipDNN
    reads it, and the batch stride is unused.
    """

    def __init__(self, name: str, desc: SdpaTensor, tensor, offsets=None) -> None:
        batch, heads, extent, width = desc.dims
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
        self.flat = tensor.reshape(-1)
        self.starts = [o * mult for o in _host_ints(offsets)]
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
        import torch

        return torch.as_strided(
            self.flat,
            (n, self.heads, self.width),
            self.row_strides,
            self.flat.storage_offset() + self.starts[b],
        )

    def packed_view(self, cu: list[int]):
        """A ``(total, heads, d)`` view when the rows already sit packed in
        token order (packed BSHD at full length, or contiguous ragged), else
        ``None``."""
        total, h, d = cu[-1], self.heads, self.width
        if self.starts is None:
            if any(b - a != self.extent for a, b in pairwise(cu)):
                return None
            bshd = self.tensor.transpose(1, 2)
            return bshd.reshape(total, h, d) if bshd.is_contiguous() else None
        if self.row_strides != (h * d, d, 1) or any(
            s != c * h * d for s, c in zip(self.starts, cu)
        ):
            return None
        return self.flat[: total * h * d].view(total, h, d)


def _lengths(req_tensor, tensors, key: str, operand: _Operand, batch: int, extent: int):
    """Per-batch lengths: ``seq_len_*`` when set, else ragged, else the extent."""
    if req_tensor is None:
        return list(operand.lengths) if operand.lengths else [extent] * batch
    values = _host_ints(tensors[key])
    if len(values) != batch or any(n < 0 or n > extent for n in values):
        raise ValueError(f"{key}: need {batch} lengths in [0, {extent}], got {values}")
    if operand.lengths and any(n > r for n, r in zip(values, operand.lengths)):
        raise ValueError(f"{key} {values} exceeds the ragged lengths {operand.lengths}")
    return values


class _LaunchPlan:
    """The unified paged ABI for one bound request.

    Lengths are snapshotted and buffers allocated once, at bind time, so the
    kernel sees the same tensors on every launch (the 2D path replays graphs
    keyed on them). Each launch copies the caller's current data in and the
    outputs back.
    """

    def __init__(self, req: SdpaRequest, tensors: Mapping[str, Any]) -> None:
        import dataclasses

        import torch

        dtype = getattr(torch, _TORCH_DTYPES[req.dtype])
        device = tensors["q"].device
        for name in ("q", "k", "v", "o"):
            t = tensors[name]
            if t.dtype != dtype or t.device != device:
                raise ValueError(
                    f"{name}: expected {dtype} on {device}, got {t.dtype} on {t.device}"
                )

        def offsets(name: str, fallback: str | None = None):
            key = f"ragged_offset_{name}"
            if key in tensors:
                return tensors[key]
            return tensors.get(f"ragged_offset_{fallback}") if fallback else None

        self.q = _Operand("q", req.q, tensors["q"], offsets("q"))
        self.k = _Operand("k", req.k, tensors["k"], offsets("k"))
        self.v = _Operand("v", req.v, tensors["v"], offsets("v", "k"))
        self.o = _Operand("o", req.o, tensors["o"], offsets("o", "q"))
        self.stats = (
            None
            if req.stats is None
            else _Operand("stats", req.stats, tensors["stats"], offsets("stats", "q"))
        )
        if self.k.lengths and self.v.lengths and self.k.lengths != self.v.lengths:
            raise ValueError(
                f"ragged K lengths {self.k.lengths} and V lengths {self.v.lengths} differ"
            )

        batch = req.batch
        self.q_len = _lengths(
            req.seq_len_q, tensors, "seq_len_q", self.q, batch, req.s_q
        )
        self.kv_len = _lengths(
            req.seq_len_kv, tensors, "seq_len_kv", self.k, batch, req.s_kv
        )
        cu = [0]
        for n in self.q_len:
            cu.append(cu[-1] + n)
        self.cu = cu

        attn = _to_attention_request(req)
        self.block = int(attn.kv_block_size)
        self.pages = max(1, -(-req.s_kv // self.block))
        self.problem = dataclasses.replace(_problem(attn), total_q=cu[-1])

        self.q_packed = self.q.packed_view(cu)
        self.q_inplace = self.q_packed is not None
        if not self.q_inplace:
            self.q_packed = torch.empty(
                (cu[-1], req.h_q, req.d_qk), dtype=dtype, device=device
            )
        self.o_packed = self.o.packed_view(cu)
        self.o_inplace = self.o_packed is not None
        if not self.o_inplace:
            self.o_packed = torch.empty(
                (cu[-1], req.h_q, req.d_v), dtype=dtype, device=device
            )

        # Identity page table, exactly ceil(S_kv / block) wide: the scalar
        # fallback indexes it with that row stride.
        slots = batch * self.pages
        self.k_cache = torch.zeros(
            (slots, self.block, req.h_k, req.d_qk), dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            (slots, self.block, req.h_v, req.d_v), dtype=dtype, device=device
        )
        self.block_table = (
            torch.arange(slots, dtype=torch.int32, device=device)
            .view(batch, self.pages)
            .contiguous()
        )
        self.cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
        self.seqused_k = torch.tensor(self.kv_len, dtype=torch.int32, device=device)
        self.lse = (
            None
            if self.stats is None
            else torch.empty((cu[-1], req.h_q, 1), dtype=torch.float32, device=device)
        )

    def kernel_tensors(self) -> dict[str, Any]:
        return {
            "q": self.q_packed,
            "k": self.k_cache,
            "v": self.v_cache,
            "out": self.o_packed,
            "cu_seqlens_q": self.cu_seqlens_q,
            "seqused_k": self.seqused_k,
            "block_table": self.block_table,
        }

    def pack(self) -> None:
        cu = self.cu
        if not self.q_inplace:
            for b, n in enumerate(self.q_len):
                if n:
                    self.q_packed[cu[b] : cu[b] + n].copy_(self.q.rows(b, n))
        span = self.pages * self.block
        k_rows = self.k_cache.view(-1, span, *self.k_cache.shape[2:])
        v_rows = self.v_cache.view(-1, span, *self.v_cache.shape[2:])
        for b, n in enumerate(self.kv_len):
            if n:
                k_rows[b, :n].copy_(self.k.rows(b, n))
                v_rows[b, :n].copy_(self.v.rows(b, n))

    def unpack(self) -> None:
        """Scatter O and LSE back. Rows past a batch's query length are O = 0
        and LSE = -inf, as hipDNN's reference writes them."""
        cu = self.cu
        for b, n in enumerate(self.q_len):
            if n and not self.o_inplace:
                self.o.rows(b, n).copy_(self.o_packed[cu[b] : cu[b] + n])
            if n < self.o.extent and not self.o.desc.ragged:
                self.o.tensor[b, :, n:, :].zero_()
            if self.stats is None:
                continue
            if n:
                self.stats.rows(b, n).copy_(self.lse[cu[b] : cu[b] + n])
            if n < self.stats.extent and not self.stats.desc.ragged:
                self.stats.tensor[b, :, n:, :].fill_(float("-inf"))


def sdpa_unified_bind(req: SdpaRequest, tensors: Mapping[str, Any]) -> TorchBinding:
    """Bind an admitted request and its device tensors to a unified launch.

    ``tensors`` holds ``q``, ``k``, ``v``, ``o`` and, when the request sets
    them, ``stats``, ``seq_len_q`` / ``seq_len_kv`` and
    ``ragged_offset_<operand>`` (V, O and stats fall back to K's, Q's and Q's
    offsets). A dense operand is its logical BHSD tensor with the request's
    strides; a ragged one is its contiguous packed buffer. The launch runs on
    Torch's current stream through the production route
    (``run_unified_attention_torch`` with no tuning spec; ``backend="3d"``
    when LSE is requested, the only path that writes it).

    Lengths and offsets are read once, here; rebind after changing them.
    Raises ``ValueError`` with the admission reason for a declined request.
    """
    ok, reason = sdpa_unified_admits(req)
    if not ok:
        raise ValueError(reason)

    import torch

    from dispatch.attention.bindings import validate_tuning_attention_tensors
    from kernels.common.attention_unified import run_unified_attention_torch

    plan = _LaunchPlan(req, tensors)
    if plan.problem.total_q:
        validate_tuning_attention_tensors(plan.problem, plan.kernel_tensors())
    backend = "3d" if req.stats is not None else "auto"
    scale = float(req.scale)

    def launch(**_kw: Any) -> None:
        plan.pack()
        if plan.problem.total_q:
            device = plan.q_packed.device
            run_unified_attention_torch(
                problem=plan.problem,
                **plan.kernel_tensors(),
                softmax_scale=scale,
                softcap=0.0,
                backend=backend,
                stream=int(torch.cuda.current_stream(device).cuda_stream),
                lse=plan.lse,
            )
        plan.unpack()

    return TorchBinding(launch=launch, grid=(0, 0, 0), block=(0, 0, 0))
