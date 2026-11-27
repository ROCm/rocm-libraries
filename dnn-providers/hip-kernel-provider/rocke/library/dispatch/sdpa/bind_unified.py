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
the production route (``run_unified_attention_torch`` with no tuning spec),
on torch tensors or :class:`rocke.runtime.device_array.DeviceArray` s alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from kernels.common._sdpa_seqlen import validate_lengths
from rocke.core.arch import known_arches
from rocke.dispatch.core import TorchBinding
from rocke.runtime.device_array import (
    ArrayInfo,
    array_info,
    copy_elements,
    dense_strides,
    device_values,
    device_workspace,
    host_values,
    zero_elements,
)

from dispatch.attention import (
    ATTENTION_ROUTE_REGISTRY,
    UNIFIED_ALGORITHMS,
    UNIFIED_HEAD_SIZES,
    attention_band,
    attention_problem,
)
from dispatch.attention.common import Alignment, AttentionBand

from .bind_shared import (
    ELEMENT_TYPES,
    Operand,
    asked_band,
    band_text,
    check_dtype,
    declined_table,
    dropout_launch_inputs,
    keeps_asked_keys,
    launch_stream,
    nonfinite_scale_reject,
    require_one_device,
    to_attention_request,
    value_rejects,
)

if TYPE_CHECKING:
    from rocke.dispatch.core import KernelCandidate

    from .common import SdpaRequest

__all__ = [
    "SDPA_UNIFIED_FEATURES",
    "sdpa_unified_admits",
    "sdpa_unified_bind",
    "sdpa_unified_candidates",
]

# What the unified launch path routes among (``run_unified_attention_torch``
# with no tuning spec), named by the modules that register them rather than
# duplicated here as literals, so a renamed algorithm breaks this import
# instead of silently leaving an orphaned string nothing matches. Tuning
# geometries, dense, RDNA WMMA and strided-KV decode candidates are other
# launch paths (see ``dispatch.attention.UNIFIED_ALGORITHMS`` and
# ``TestRouteAlgorithmCoverage`` in
# tests/dispatch/attention/test_review_followups.py for the full partition of
# ``ATTENTION_ROUTE_REGISTRY`` by launch path).
_UNIFIED_ALGORITHMS = UNIFIED_ALGORITHMS

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
        "noncausal_window",
        "padded",
        "ragged",
        "lse",
        "dropout",
        "mma_core_mode",
        "implementation_hint",
    }
)

# Every other feature, with the reason it is declined (shared template in
# bind_shared.py, so the unified and dense wording cannot drift apart).
# Checked in this order, so the reported reason is deterministic.
_DECLINED = declined_table(
    "unified",
    (
        "kv_heads_differ",
        "dtype_mixed",
        "compute_dtype",
        "scale_tensor",
        "bias",
        "bias_broadcast",
        "alibi",
        "sinks",
        "dropout_mask",
        "dropout_scale",
        "rng_dump",
        "paged",
        "block_mask",
        "fp8",
        "fp8_s_o",
        "max_sum_exp",
    ),
)


def _computed_band(req: SdpaRequest) -> AttentionBand:
    """The band the unified kernels compute for ``req``.

    Read from ``attention_band``, which is what ``attention_problem`` hands
    every unified route candidate, not from ``normalize_band``: outside the
    band arches the problem is always built bottom-right causal, whatever band
    the request names. ``normalize_band`` raises for a band the band arches
    cannot serve.
    """
    attn = to_attention_request(req)
    band = attention_band(attn)
    return AttentionBand(
        left_bound=attn.sliding_window if attn.sliding_window > 0 else None,
        right_bound=None if band.kernel_right_bound < 0 else band.kernel_right_bound,
        alignment=Alignment.TOP_LEFT if band.top_left else Alignment.BOTTOM_RIGHT,
    )


def _band_reject(req: SdpaRequest) -> tuple[bool, str] | None:
    """Decline when the unified kernels would compute a different band."""
    try:
        band = _computed_band(req)
    except ValueError as exc:
        return False, f"band: {exc}"
    if keeps_asked_keys(req, band):
        return None
    return False, (
        f"band: {req.arch} unified kernels compute {band_text(band)} for "
        f"requested {band_text(asked_band(req))}"
    )


def _scale_reject(req: SdpaRequest) -> tuple[bool, str] | None:
    """Decline a scale the unified kernels cannot honor: non-finite, or not
    positive.

    The gfx942 4-warp bodies (``_online_softmax_rescale`` in
    ``kernels/gfx942/attention_tiled_2d.py``) mask declined score positions
    with ``-1e30`` *before* multiplying by the scale: a zero scale brings the
    masked positions back to logit 0, and a negative scale makes them the
    largest logits, so either attends to keys outside the band. The other
    unified kernels mask after scaling, but the gate cannot tell which body
    the launch will pick, so it declines for every arch.
    """
    rejected = nonfinite_scale_reject(req, "unified")
    if rejected is not None:
        return rejected
    if float(req.scale) <= 0:
        return (
            False,
            f"scale: the unified kernels need a positive scale, got {req.scale}",
        )
    return None


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

    rejected = value_rejects(req, "unified")
    if rejected is not None:
        return rejected
    rejected = _scale_reject(req)
    if rejected is not None:
        return rejected
    return _band_reject(req)


def _unified_supported(req: SdpaRequest) -> tuple[KernelCandidate, ...]:
    """Unified route candidates that admit the translated request.

    No separate LSE filter: ``to_attention_request`` already carries
    ``emit_lse`` into the translated request, and each unified candidate's own
    ``Capability`` and ``support()`` (see ``dispatch/attention/generic.py``)
    already decide which arches/paths can write it -- both the tiled 2D and
    split-KV (3D) paths do, on gfx942/gfx950. Filtering again here by a static
    algorithm set only ever drops candidates the registry already admitted.
    """
    return tuple(
        c
        for c in ATTENTION_ROUTE_REGISTRY.supported(to_attention_request(req))
        if c.algorithm in _UNIFIED_ALGORITHMS
    )


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
    path = "".join(
        f"{name} "
        for name, on in (("LSE", req.stats is not None), ("dropout", req.dropout))
        if on
    )
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


def _host_lengths(req_tensor, tensors, key: str):
    """The caller's ``seq_len_*`` as a host array, once its dtype is the
    request's (int32)."""
    if req_tensor is None:
        return None
    check_dtype(key, tensors[key], req_tensor)
    return host_values(tensors[key]).reshape(-1)


def _lengths(given, key: str, operand: Operand, batch: int, extent: int):
    """Per-batch lengths: the host ``seq_len_*`` values ``given`` (``None`` when
    the request has none), else ragged, else the extent.

    The range/dtype/count check is ``kernels.common._sdpa_seqlen``'s host-side
    adapter (:func:`validate_lengths`), shared with ``sdpa_seqlens`` /
    ``sdpa_cu_seqlens`` so the rule lives in one place and is covered torch-free
    in ``tests/test_sdpa_seqlen.py``.
    """
    if given is None:
        return list(operand.lengths) if operand.lengths else [extent] * batch
    values = validate_lengths(given, batch=batch, s_max=extent, name=key).tolist()
    if operand.lengths and any(n > r for n, r in zip(values, operand.lengths)):
        raise ValueError(f"{key} {values} exceeds the ragged lengths {operand.lengths}")
    return values


class _LaunchPlan:
    """The unified paged ABI for one bound request.

    Lengths are snapshotted and buffers allocated once, at bind time, so the
    kernel sees the same arrays on every launch (the 2D path replays graphs
    keyed on them). Each launch copies the caller's current data in and the
    outputs back, except for an operand the kernel reads or writes in place
    (``q_inplace``, ``o_inplace``, ``k_inplace``, ``v_inplace``). Buffers are
    of the caller's kind (``device_workspace``); copies are queued on the
    launch stream.
    """

    def __init__(self, req: SdpaRequest, tensors: Mapping[str, Any]) -> None:
        import dataclasses

        dtype = ELEMENT_TYPES[req.dtype]
        like = tensors["q"]
        require_one_device(tensors)

        def offsets(name: str, fallback: str | None = None):
            key = f"ragged_offset_{name}"
            if key in tensors:
                return tensors[key]
            return tensors.get(f"ragged_offset_{fallback}") if fallback else None

        self.q = Operand("q", req.q, tensors["q"], offsets("q"))
        self.k = Operand("k", req.k, tensors["k"], offsets("k"))
        self.v = Operand("v", req.v, tensors["v"], offsets("v", "k"))
        self.o = Operand("o", req.o, tensors["o"], offsets("o", "q"))
        self.stats = (
            None
            if req.stats is None
            else Operand("stats", req.stats, tensors["stats"], offsets("stats", "q"))
        )
        if self.k.lengths and self.v.lengths and self.k.lengths != self.v.lengths:
            raise ValueError(
                f"ragged K lengths {self.k.lengths} and V lengths {self.v.lengths} differ"
            )

        batch = req.batch
        self.q_len = _lengths(
            _host_lengths(req.seq_len_q, tensors, "seq_len_q"),
            "seq_len_q",
            self.q,
            batch,
            req.s_q,
        )
        self.kv_len = _lengths(
            _host_lengths(req.seq_len_kv, tensors, "seq_len_kv"),
            "seq_len_kv",
            self.k,
            batch,
            req.s_kv,
        )
        cu = [0]
        for n in self.q_len:
            cu.append(cu[-1] + n)
        self.cu = cu

        attn = to_attention_request(req)
        self.block = int(attn.kv_block_size)
        self.pages = max(1, -(-req.s_kv // self.block))
        # max_seqlen_q/k stay the declared s_q/s_kv, not the longest sequence:
        # a dropout launch takes them as the mask's extents (SEM-015), so the
        # padded, ragged and dense forms of one problem draw the same mask.
        self.problem = dataclasses.replace(attention_problem(attn), total_q=cu[-1])

        self.q_packed = self.q.packed_view(cu)
        self.q_inplace = self.q_packed is not None
        if not self.q_inplace:
            self.q_packed = device_workspace(
                (cu[-1], req.h_q, req.d_qk), dtype, like=like
            )
        self.o_packed = self.o.packed_view(cu)
        self.o_inplace = self.o_packed is not None
        if not self.o_inplace:
            self.o_packed = device_workspace(
                (cu[-1], req.h_q, req.d_v), dtype, like=like
            )

        # Identity page table, exactly ceil(S_kv / block) wide: the scalar
        # fallback indexes it with that row stride. A full-length packed K/V
        # already is that paged cache (Operand.paged_view). Otherwise the
        # cache is zeroed, not just allocated: some kernels (e.g. the gfx942
        # 4-warp body) read a KV tile's trailing rows past a sequence's own
        # length, masked but still dereferenced, so they must be finite.
        slots = batch * self.pages
        bind_stream = launch_stream(None, like.device, "unified")
        self.k_cache = self.k.paged_view(self.kv_len, self.block)
        self.k_inplace = self.k_cache is not None
        if not self.k_inplace:
            self.k_cache = device_workspace(
                (slots, self.block, req.h_k, req.d_qk), dtype, like=like
            )
            zero_elements(array_info(self.k_cache), stream=bind_stream)
        self.v_cache = self.v.paged_view(self.kv_len, self.block)
        self.v_inplace = self.v_cache is not None
        if not self.v_inplace:
            self.v_cache = device_workspace(
                (slots, self.block, req.h_v, req.d_v), dtype, like=like
            )
            zero_elements(array_info(self.v_cache), stream=bind_stream)

        self.block_table = device_values(
            np.arange(slots, dtype=np.int32).reshape(batch, self.pages), like=like
        )
        self.cu_seqlens_q = device_values(np.asarray(cu, np.int32), like=like)
        self.seqused_k = device_values(np.asarray(self.kv_len, np.int32), like=like)
        self.lse = None
        self.neg_inf = None
        if self.stats is not None:
            self.lse = device_workspace((cu[-1], req.h_q, 1), "fp32", like=like)
            # The -inf a padded row's LSE gets, copied from here (a memset
            # writes bytes, not an FP32 value).
            self.neg_inf = device_values(
                np.full(req.h_q * req.s_q, -np.inf, np.float32), like=like
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

    def pack(self, stream: int = 0) -> None:
        """Copy the caller's Q, K and V rows into the kernel's buffers, queued
        on ``stream``."""
        cu = self.cu
        if not self.q_inplace:
            packed = array_info(self.q_packed)
            for b, n in enumerate(self.q_len):
                if n:
                    copy_elements(
                        _token_rows(packed, cu[b], n), self.q.rows(b, n), stream=stream
                    )
        span = self.pages * self.block
        for operand, cache, inplace in (
            (self.k, self.k_cache, self.k_inplace),
            (self.v, self.v_cache, self.v_inplace),
        ):
            if inplace:
                continue
            # The cache as one run of token rows: batch b's start at b * span.
            slots, block, *row = array_info(cache).shape
            shape = (slots * block, *row)
            tokens = array_info(cache).view(shape, dense_strides(shape))
            for b, n in enumerate(self.kv_len):
                if n:
                    copy_elements(
                        _token_rows(tokens, b * span, n),
                        operand.rows(b, n),
                        stream=stream,
                    )

    def unpack(self, stream: int = 0) -> None:
        """Scatter O and LSE back, queued on ``stream``. Rows past a batch's
        query length are O = 0 and LSE = -inf, as hipDNN's reference writes
        them."""
        cu = self.cu
        packed = array_info(self.o_packed)
        lse = None if self.lse is None else array_info(self.lse)
        for b, n in enumerate(self.q_len):
            if n and not self.o_inplace:
                copy_elements(
                    self.o.rows(b, n), _token_rows(packed, cu[b], n), stream=stream
                )
            if n < self.o.extent and not self.o.desc.ragged:
                zero_elements(self.o.tail(b, n), stream=stream)
            if self.stats is None:
                continue
            if n:
                copy_elements(
                    self.stats.rows(b, n), _token_rows(lse, cu[b], n), stream=stream
                )
            if n < self.stats.extent and not self.stats.desc.ragged:
                tail = self.stats.tail(b, n)
                neg_inf = array_info(self.neg_inf).view(
                    tail.shape, dense_strides(tail.shape)
                )
                copy_elements(tail, neg_inf, stream=stream)


def _token_rows(info: ArrayInfo, start: int, n: int) -> ArrayInfo:
    """Rows ``[start, start + n)`` of a dense ``[tokens, ...]`` buffer."""
    row = info.strides[0]
    return info.view((n, *info.shape[1:]), info.strides, offset=start * row)


def sdpa_unified_bind(req: SdpaRequest, tensors: Mapping[str, Any]) -> TorchBinding:
    """Bind an admitted request and its device arrays to a unified launch.

    ``tensors`` holds ``q``, ``k``, ``v``, ``o`` and, when the request sets
    them, ``stats``, ``seq_len_q`` / ``seq_len_kv`` and
    ``ragged_offset_<operand>`` (V, O and stats fall back to K's, Q's and Q's
    offsets): torch tensors or :class:`rocke.runtime.device_array.DeviceArray`
    s. A dense operand is its logical BHSD array with the request's strides; a
    ragged one is its contiguous packed buffer. Every array's element type
    must be the one its request descriptor names. The launch runs on the
    current stream (Torch's, or HIP's null stream without torch) through the
    production route (``run_unified_attention_torch`` with no tuning spec,
    ``backend="auto"``): tiled 2D and split-KV (3D) both write LSE on
    gfx942/gfx950, so the launch defers to the kernel's own path selection
    rather than re-deciding it here.

    A dropout request also takes ``dropout_seed`` and ``dropout_offset``, read
    at every launch (see
    :func:`~dispatch.sdpa.bind_shared.dropout_launch_inputs`). Lengths and
    offsets are read once, here; rebind after changing them.
    Raises ``ValueError`` with the admission reason for a declined request.
    ``launch(stream=...)`` accepts only the current stream; see
    :func:`~dispatch.sdpa.bind_shared.launch_stream`.
    """
    ok, reason = sdpa_unified_admits(req)
    if not ok:
        raise ValueError(reason)

    from dispatch.attention.bindings import validate_tuning_attention_tensors
    from kernels.common.attention_unified import run_unified_attention_torch

    # Refuses missing or malformed seed/offset before anything is allocated.
    dropout_launch_inputs(req, tensors)
    plan = _LaunchPlan(req, tensors)
    if plan.problem.total_q:
        validate_tuning_attention_tensors(plan.problem, plan.kernel_tensors())
    scale = float(req.scale)
    device = tensors["q"].device

    def launch(*, stream: Any = None) -> None:
        """Pack, launch and unpack, all on the current stream."""
        launch_on = launch_stream(stream, device, "unified")
        dropout = dropout_launch_inputs(req, tensors)
        if dropout:
            dropout["dropout_p"] = req.dropout.probability
        plan.pack(launch_on)
        if plan.problem.total_q:
            run_unified_attention_torch(
                problem=plan.problem,
                **plan.kernel_tensors(),
                softmax_scale=scale,
                softcap=0.0,
                backend="auto",
                stream=launch_on,
                lse=plan.lse,
                **dropout,
            )
        plan.unpack(launch_on)

    return TorchBinding(launch=launch, grid=(0, 0, 0), block=(0, 0, 0))
