# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.sdpa -- route ``F.scaled_dot_product_attention`` onto hipDNN's fused
attention, forward and backward.

**Pure passthrough, routed by grad-eligibility.** The override builds the SDPA graph
from Q/K/V's *actual* dims + strides -- whatever layout they arrive in (Q/K/V are
almost always permuted views of a packed QKV buffer, never contiguous BHSD) -- and
submits it. It imposes no layout (no ``.contiguous()``), pre-judges nothing about
which shapes an engine can serve, and makes no structural correctness check (rank,
GQA head counts, D agreement): hipDNN's ``check_support`` / ``execute`` decides, and
a shape no loaded engine claims falls back to native and is counted.

A call is **grad-eligible** -- ``dropout_p > 0``, or grad mode is on and any of
Q/K/V requires grad -- runs forward *and* backward on hipDNN, or both on native,
decided once at forward time: a ``torch.autograd.Function`` wraps a forward graph
with ``set_generate_stats(True)`` and a paired backward graph
(``SdpaBackwardAttributes``), probed together before either runs; either declining
sends the whole call native, so no half-native step is ever counted served (the one
exception is a failed backward *execute* at ``dropout_p == 0``, which recomputes
natively and is counted so). dO is copied into a fresh, aligned buffer in O's layout
first when its strides, overlap, or alignment would otherwise disagree with what the
backward graph declares. A plain (non grad-eligible) call keeps today's single
forward graph and census row unchanged.

Dropout (``dropout_p != 0``) is always grad-eligible, whatever the grad mode, so a
reentrant checkpoint's no-grad first pass and its later recompute always decide
alike (the no-grad pass runs the stats forward graph directly, without the
``Function``, and discards the stats). Forward and backward share one Philox mask:
the seed/offset come from torch's default CUDA generator for Q's device, advanced by
the same increment native's own chosen backend (``torch._fused_sdp_choice``) would
use, so every other RNG consumer sees the same stream whether the call runs on
hipDNN or natively. ``set_dropout(p, seed, offset)`` carries it in both graphs; a
backend this override cannot mirror (native's math backend, CK-preferred flash,
stream capture, or a torch without ``_fused_sdp_choice``) sends the call native
instead. A grad-eligible call under ``torch.compile`` tracing also goes native; a
plain call is unaffected.

Every torch SDPA parameter is translated into the graph:

  * **scale** (or ``1/sqrt(Dqk)`` from the *actual* Dqk) -> ``set_attn_scale``.
  * **is_causal** -> ``set_causal_mask``.
  * **attn_mask** -> ``set_bias`` (an additive score bias). A float mask passes
    through as its own additive tensor at its actual dims/strides; a bool mask is
    materialized to a 0/-inf additive tensor (``True`` keeps, ``False`` masks) --
    an intrinsic, documented materialization, the only way an additive-bias node
    can carry a boolean mask. A float mask requiring grad under grad mode is
    native-only (no dBias support here).

GQA (``H_kv < H_q``) needs no special handling -- it is simply K/V operands with a
smaller head count, described directly; dQ/dK/dV keep their own source tensor's head
count and memory order.
"""

import logging
import math
import threading

from .base import BWD_PREFIX, GRAD_PREFIX, NotApplicable, OpOverride

_Q_UID, _K_UID, _V_UID, _O_UID, _BIAS_UID = 1, 2, 3, 4, 5
_STATS_UID, _SEED_UID, _OFFSET_UID = 6, 7, 8
_DO_UID, _DQ_UID, _DK_UID, _DV_UID = 9, 10, 11, 12

# Backend names ``torch._fused_sdp_choice``'s int result maps to through
# ``torch.nn.attention.SDPBackend`` (verified on torch 2.15.0: MATH=0,
# FLASH_ATTENTION=1, EFFICIENT_ATTENTION=2, CUDNN_ATTENTION=3, OVERRIDEABLE=4,
# ERROR=-1), shortened to the tokens ``_dropout_advance`` compares against.
_SDP_BACKEND_TOKENS = {
    "MATH": "MATH",
    "FLASH_ATTENTION": "FLASH",
    "EFFICIENT_ATTENTION": "EFFICIENT",
    "CUDNN_ATTENTION": "CUDNN",
    "OVERRIDEABLE": "OVERRIDEABLE",
    "ERROR": "ERROR",
}

# Guards the CUDA generator's fallback seed/offset draw, used when
# ``philox_state`` is missing: the read-modify-write of a torch ``Generator``'s
# offset is not atomic on its own, unlike ``philox_state``.
_dropout_lock = threading.Lock()


def _output_layout(q_shape, q_stride, dv):
    """Dims to allocate contiguously, and the permutation that turns that
    allocation into a ``[*q_shape[:-1], dv]`` output ordered like Q in memory.

    Dims are sorted outermost-first by Q's stride; ties keep index order, so a
    contiguous Q maps to a contiguous output. Only the order is taken from Q: the
    output is always a fresh dense buffer, so gaps in Q (a slice of a packed QKV
    buffer) are fine. A Q whose elements overlap (a zero stride on a non-unit dim,
    as from ``expand``) has no memory order to copy, so the output is contiguous.
    Pure shape/stride math, so the gates tests can check it without torch.

    Also used, unchanged, to lay out dQ/dK/dV: called on a source tensor's own
    shape/stride with its own last dim, it reproduces that tensor's exact shape in
    its own memory order as a fresh dense allocation.
    """
    shape = (*q_shape[:-1], dv)
    rank = len(shape)
    order = sorted(range(rank), key=lambda i: (-q_stride[i], i))
    span = 1  # elements covered by the dims inner to the current one
    for i in reversed(order):
        if q_shape[i] == 1:
            continue
        if q_stride[i] < span:
            order = list(range(rank))  # overlapping: no order to follow
            break
        span = q_stride[i] * q_shape[i]
    alloc_dims = [shape[i] for i in order]
    inverse = [order.index(i) for i in range(rank)]
    return alloc_dims, inverse


def _view_strides(alloc_dims, inverse):
    """Strides of ``torch.empty(alloc_dims).permute(inverse)``, computed purely
    from shape + permutation (no allocation): the row-major strides of the dense
    allocation, read back through ``inverse`` (as :func:`_output_layout` returns
    it). Used for graph descriptors and cache keys, where only the numbers --
    never the buffer -- are needed before the tensor is actually allocated."""
    rank = len(alloc_dims)
    alloc_stride = [1] * rank
    for i in range(rank - 2, -1, -1):
        alloc_stride[i] = alloc_stride[i + 1] * alloc_dims[i + 1]
    return tuple(alloc_stride[i] for i in inverse)


def _needs_do_copy(do_shape, do_stride, o_stride, do_ptr) -> bool:
    """dO must be copied into a fresh, aligned buffer in O's layout first
    when it would otherwise disagree with what the backward graph declares (O's
    strides): True if an axis with extent > 1 differs in stride from O's, if dO
    self-overlaps (stride 0 on an axis with extent > 1, as from ``expand``), or if
    its pointer is not 16-byte aligned (every tensor is declared with the default
    alignment). A stride difference on a size-1 axis is ignored: the declared dO
    strides are O's, and only the row's D elements are ever read on that axis."""
    if do_ptr % 16 != 0:
        return True
    for extent, d_stride, o_s in zip(do_shape, do_stride, o_stride):
        if extent <= 1:
            continue
        if d_stride == 0:
            return True
        if d_stride != o_s:
            return True
    return False


def _dropout_advance(sdp_choice, b, hq, sq, skv):
    """The Philox counter increment that mirrors native's own advance for the
    backend ``torch._fused_sdp_choice`` would pick (``sdp_choice``, one of the
    tokens in :data:`_SDP_BACKEND_TOKENS`), so hipDNN's dropout and native's draw
    from the same generator identically: ``"FLASH"`` -> ``b*hq*32`` (the query head
    count, Hq -- also under GQA, where Hkv is unrelated); ``"EFFICIENT"`` /
    ``"CUDNN"`` -> ``b*hq*sq*skv``; anything else (including ``None``, meaning no
    backend this override can mirror) -> ``None``, the whole call must go native.
    Rounded up to a multiple of 4, as torch's own ``philox_cuda_state`` does."""
    if sdp_choice == "FLASH":
        n = b * hq * 32
    elif sdp_choice in ("EFFICIENT", "CUDNN"):
        n = b * hq * sq * skv
    else:
        return None
    return (n + 3) // 4 * 4


def _unpack_philox(state):
    """Unpack ``Generator.philox_state``'s 3-tuple (seed, offset, intragraph
    offset -- zero outside CUDA-graph capture) into the ``(seed, offset)`` pair
    ``set_dropout`` takes. Host-only (``.item()`` on a CPU int64 tensor, no device
    sync). ``seed`` keeps its signed int64 bit pattern as-is: nothing here widens
    or reinterprets it."""
    seed_t, offset_t, intragraph_t = state
    return int(seed_t.item()), int(offset_t.item()) + int(intragraph_t.item())


def _grad_route(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    *,
    grad_enabled,
    deterministic,
    warn_only,
    capturing,
    sdp_choice,
    ck_preferred,
    compiling,
):
    """Every native-routing decline that needs no graph probe, as one pure
    ``(route, reason)`` decision, so ``_call`` only drives it and the two probes
    it can't replace. ``route`` is ``"plain"`` (today's non grad-eligible
    graph/census), ``"grad"`` (probe both grad-eligible graphs), or
    ``"native"`` (``reason`` is one of the census reason strings; ``""`` otherwise).

    A float ``attn_mask`` requiring grad forces native even when Q/K/V do not
    (whenever grad mode is on, or whenever ``dropout_p > 0`` regardless of grad
    mode), so it is checked ahead of -- not gated by -- plain ``q/k/v``
    grad-eligibility. Tensors are duck-typed (``.shape``, ``.stride()``,
    ``.data_ptr()``, ``.requires_grad``, ``.dtype``, ``.is_floating_point()``), so
    tests drive this with fakes, no torch needed.
    """
    mask_requires_grad = (
        attn_mask is not None
        and attn_mask.is_floating_point()
        and attn_mask.requires_grad
        and (grad_enabled or dropout_p > 0)
    )
    grad_eligible = dropout_p > 0 or (
        grad_enabled and (q.requires_grad or k.requires_grad or v.requires_grad)
    )
    if not grad_eligible and not mask_requires_grad:
        return "plain", ""

    if compiling:
        return "native", "compile: tracing"
    if mask_requires_grad:
        return "native", "grad: attn_mask requires grad"
    if grad_enabled and deterministic and not warn_only:
        return "native", "grad: deterministic algorithms"

    for t in (q, k, v):
        if t.stride()[-1] != 1 or t.stride()[-2] < t.shape[-1]:
            return "native", "operand: stride"

    for t in (q, k, v, attn_mask):
        if t is None:
            continue
        if t is attn_mask and not t.is_floating_point():
            continue  # a bool mask is materialized fresh before use, always aligned
        if t.data_ptr() % 16 != 0:
            return "native", "operand: pointer not 16-byte aligned"

    if dropout_p > 0:
        if capturing:
            return "native", "dropout: stream capturing"
        if sdp_choice is None:
            return "native", "dropout: cannot mirror native RNG advance"
        if sdp_choice == "FLASH" and ck_preferred:
            return "native", "dropout: CK flash preferred"
        if sdp_choice not in ("FLASH", "EFFICIENT", "CUDNN"):
            return "native", "dropout: native math backend"

    return "grad", ""


def _reraise_checkpoint_stop(torch, ex):
    """Let a non-reentrant ``torch.utils.checkpoint`` recompute's early-stop
    signal through any broad exception catch on this path. The recompute
    forward raises ``_StopRecomputationError`` from inside the saved-tensor pack
    hook that fires on ``ctx.save_for_backward`` (or any autograd save) once
    every tensor the checkpoint still needs has been produced
    (``torch/utils/checkpoint.py``); it is meant to propagate all the way out to
    checkpoint's own ``except _StopRecomputationError: pass`` around the whole
    recompute call. Catching and suppressing it here instead would silently run
    extra ops the recompute no longer needs and defeat the early stop."""
    checkpoint_mod = getattr(torch.utils, "checkpoint", None)
    stop_cls = getattr(checkpoint_mod, "_StopRecomputationError", None)
    if stop_cls is not None and isinstance(ex, stop_cls):
        raise ex


def _recompute_native(
    torch, real, query, key, value, bias, is_causal, scale, enable_gqa, do
):
    """A backward execute failure at ``dropout_p == 0`` recomputes dQ/dK/dV
    with the **saved native function** (``real``, never the patched
    ``torch.nn.functional`` attribute, which would re-enter the override) on
    detached, grad-enabled copies of Q/K/V, fed the already-materialized ``bias``
    (equivalent to the user's original mask, bool or float) and ``do`` (the
    logical-shape gradient already computed for the hipDNN attempt)."""
    with torch.enable_grad():
        q2 = query.detach().requires_grad_(True)
        k2 = key.detach().requires_grad_(True)
        v2 = value.detach().requires_grad_(True)
        out = real(
            q2,
            k2,
            v2,
            attn_mask=bias,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        dq, dk, dv = torch.autograd.grad(out, (q2, k2, v2), grad_outputs=do)
    return dq, dk, dv


_autograd_fn_cache = {}


def _sdpa_autograd_fn(torch):
    """Build (once per ``torch`` module) the ``autograd.Function`` that runs
    the grad-eligible forward graph and, on ``.backward()``, the paired backward
    graph. Built lazily because subclassing ``torch.autograd.Function`` needs the
    real torch module, which the rest of this file never imports at module level
    (see the module docstring / ``self.state.torch``)."""
    cached = _autograd_fn_cache.get(id(torch))
    if cached is not None:
        return cached

    once_differentiable = torch.autograd.function.once_differentiable

    class _SdpaBackwardFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, bias, seed_t, offset_t, payload):
            ov = payload["ov"]
            device = query.device
            o_dense = torch.empty(
                payload["alloc_dims"], dtype=payload["o_dtype"], device=device
            )
            stats = torch.empty(
                payload["stats_dims"], dtype=torch.float32, device=device
            )

            variant = {
                _Q_UID: query.data_ptr(),
                _K_UID: key.data_ptr(),
                _V_UID: value.data_ptr(),
                _O_UID: o_dense.data_ptr(),
                _STATS_UID: stats.data_ptr(),
            }
            if bias is not None:
                variant[_BIAS_UID] = bias.data_ptr()
            if seed_t is not None:
                variant[_SEED_UID] = seed_t.data_ptr()
                variant[_OFFSET_UID] = offset_t.data_ptr()

            ov._execute(payload["entry_fwd"], variant, device)

            # Counted immediately after a successful execute, before
            # save_for_backward: a non-reentrant torch.utils.checkpoint
            # recompute can raise _StopRecomputationError from inside the
            # saved-tensor pack hook that fires on save_for_backward, once
            # every tensor the checkpoint still needs has been produced. The
            # hipDNN forward has already run by then, so it must be counted
            # here -- waiting until after save_for_backward would lose it to
            # that signal propagating out uncounted.
            recompute_extra = {}
            if torch.is_grad_enabled() and torch._C._current_graph_task_id() != -1:
                recompute_extra["recompute"] = 1
            dropout_extra = {"dropout": 1} if payload["dropout_p"] else {}
            ov.note_aot(
                payload["grad_key"],
                payload["entry_fwd"]["engine"],
                **recompute_extra,
                **dropout_extra,
                **payload["mask_extras"],
            )

            ctx.save_for_backward(
                query, key, value, o_dense, stats, bias, seed_t, offset_t
            )
            ctx.payload = payload
            return o_dense

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            query, key, value, o_dense, stats, bias, seed_t, offset_t = (
                ctx.saved_tensors
            )
            payload = ctx.payload
            ov = payload["ov"]
            device = query.device
            inverse = payload["inverse"]
            o_view_stride = payload["o_view_stride"]
            dropout_p = payload["dropout_p"]
            bwd_key = payload["bwd_key"]

            do = grad_output.permute(inverse)
            if _needs_do_copy(
                tuple(do.shape), tuple(do.stride()), o_view_stride, do.data_ptr()
            ):
                fixed = torch.empty(
                    payload["alloc_dims"], dtype=do.dtype, device=device
                ).permute(inverse)
                fixed.copy_(do)
                do = fixed

            dq_alloc, dq_inv = payload["dq_layout"]
            dk_alloc, dk_inv = payload["dk_layout"]
            dv_alloc, dv_inv = payload["dv_layout"]
            dq = torch.empty(dq_alloc, dtype=query.dtype, device=device).permute(dq_inv)
            dk = torch.empty(dk_alloc, dtype=key.dtype, device=device).permute(dk_inv)
            dv = torch.empty(dv_alloc, dtype=value.dtype, device=device).permute(dv_inv)

            variant = {
                _Q_UID: query.data_ptr(),
                _K_UID: key.data_ptr(),
                _V_UID: value.data_ptr(),
                _O_UID: o_dense.data_ptr(),
                _DO_UID: do.data_ptr(),
                _STATS_UID: stats.data_ptr(),
                _DQ_UID: dq.data_ptr(),
                _DK_UID: dk.data_ptr(),
                _DV_UID: dv.data_ptr(),
            }
            if bias is not None:
                variant[_BIAS_UID] = bias.data_ptr()
            if seed_t is not None:
                variant[_SEED_UID] = seed_t.data_ptr()
                variant[_OFFSET_UID] = offset_t.data_ptr()

            extras = {"dropout": 1} if dropout_p else {}
            try:
                ov._execute(payload["entry_bwd"], variant, device)
            except NotApplicable as na:
                if dropout_p:
                    raise
                ov.note_native(
                    bwd_key,
                    f"bwd execute failed, recomputed natively: {na}",
                    level=logging.WARNING,
                )

                def _mark_recompute():
                    row = ov._row(bwd_key)
                    row["bwd_recompute"] = row.get("bwd_recompute", 0) + 1

                ov._locked(_mark_recompute)
                dq, dk, dv = _recompute_native(
                    torch,
                    payload["real"],
                    query,
                    key,
                    value,
                    bias,
                    payload["is_causal"],
                    payload["scale"],
                    payload["enable_gqa"],
                    do,
                )
                return dq, dk, dv, None, None, None, None

            ov.note_aot(bwd_key, payload["entry_bwd"]["engine"], **extras)
            return dq, dk, dv, None, None, None, None

    _autograd_fn_cache[id(torch)] = _SdpaBackwardFunction
    return _SdpaBackwardFunction


class SdpaOverride(OpOverride):
    op_name = "scaled_dot_product_attention"

    def _gate(self, query):
        # The only two legitimate stops: we need a device pointer to execute
        # against, and a dtype the graph builder can express. Everything about
        # size -- B, H, D, S, GQA, causal, mask, dropout -- is built into the graph
        # and left to hipDNN.
        if not query.is_cuda:
            return False, "query not on cuda"  # execute() needs a device pointer
        if query.dtype not in self.state.dtype_map:
            return False, f"dtype {self._tok(query.dtype)} not graph-mappable"
        return True, ""

    @staticmethod
    def _safe_bool(fn):
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001 -- a capability probe must never raise
            return False

    def _sdp_choice(
        self,
        torch,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
    ):
        """The backend name native's own dispatch (``torch._fused_sdp_choice``,
        private) would pick for this call, mapped through ``SDPBackend`` to the
        tokens :func:`_dropout_advance` and :func:`_grad_route` compare against.
        ``None`` only when the private API is missing or raises -- not when it
        resolves to MATH/OVERRIDEABLE/ERROR, which are real (non-``None``) tokens
        so ``_grad_route`` can tell "no mirrorable backend" (``None``) apart from
        "a real backend this override does not serve" (``"MATH"`` etc.)."""
        fn = getattr(torch, "_fused_sdp_choice", None)
        if fn is None:
            return None
        try:
            backend = fn(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
            name = torch.nn.attention.SDPBackend(int(backend)).name
        except Exception:  # noqa: BLE001 -- unresolvable choice -> native, not a crash
            return None
        return _SDP_BACKEND_TOKENS.get(name)

    def _ck_preferred(self, torch):
        fn = getattr(torch.backends.cuda, "preferred_rocm_fa_library", None)
        if fn is None:
            return False
        try:
            return fn().name == "Ck"
        except Exception:  # noqa: BLE001
            return False

    def _draw_dropout_state(self, gen, sdp_choice, b, hq, sq, skv):
        """Draw the seed/offset this call's dropout mask uses, advancing ``gen``
        exactly as native's chosen backend would, and return
        ``(seed, offset, offset_before)`` as plain ints -- no device allocation,
        so a caller can capture ``offset_before`` before attempting the
        (separate, allocation-based) :meth:`_dropout_tensors` step, which can
        fail on its own (OOM) after the generator has already advanced."""
        inc = _dropout_advance(sdp_choice, b, hq, sq, skv)
        philox_state = getattr(gen, "philox_state", None)
        if philox_state is not None:
            seed, offset = _unpack_philox(philox_state(inc))
            offset_before = offset
        else:
            with _dropout_lock:
                seed = gen.initial_seed()
                offset_before = gen.get_offset()
                gen.set_offset(offset_before + inc)
            offset = offset_before
            if seed >= (1 << 63):
                # initial_seed() can return the seed as an unsigned 64-bit
                # value; set_dropout's seed tensor is signed int64.
                seed -= 1 << 64
        return seed, offset, offset_before

    def _dropout_tensors(self, torch, seed, offset, device):
        """The device INT64 ``[1]`` tensors ``set_dropout`` takes for the
        ``seed``/``offset`` :meth:`_draw_dropout_state` already drew. A separate
        step from the draw so a caller already holds ``offset_before`` before
        this (the only allocating, OOM-risking part) runs."""
        seed_t = torch.full((1,), seed, dtype=torch.int64, device=device)
        offset_t = torch.full((1,), offset, dtype=torch.int64, device=device)
        return seed_t, offset_t

    def _materialize_bias(self, torch, query, attn_mask):
        """Translate ``attn_mask`` into the additive bias node both the plain and
        grad-eligible graphs set via ``set_bias``: a bool mask (``True`` keeps,
        ``False`` masks) is materialized into a fresh 0/-inf tensor; a float mask
        is used as-is. Returns ``(bias_tensor_or_None, bias_spec_or_None,
        extras)``."""
        if attn_mask is None:
            return None, None, {}
        if attn_mask.dtype == torch.bool:
            bias = torch.zeros(attn_mask.shape, dtype=query.dtype, device=query.device)
            bias.masked_fill_(~attn_mask, float("-inf"))
            extras = {"mask_bool": 1}
        else:
            bias = attn_mask
            extras = {"mask_additive": 1}
        bias_spec = (tuple(bias.shape), tuple(bias.stride()), bias.dtype)
        return bias, bias_spec, extras

    def _graph(
        self,
        q_dims,
        q_str,
        q_dtype,
        k_dims,
        k_str,
        k_dtype,
        v_dims,
        v_str,
        v_dtype,
        o_dims,
        o_str,
        o_dtype,
        scale,
        causal,
        bias_spec,
    ):
        st = self.state
        hipdnn = st.hipdnn
        hq = st.dtype_map[q_dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hq)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        def _t(name, dims, strides, dtype, uid):
            return g.tensor(
                hipdnn.Tensor()
                .set_name(name)
                .set_dim(list(dims))
                .set_stride(list(strides))
                .set_data_type(st.dtype_map[dtype])
                .set_uid(uid)
            )

        q_t = _t("Q", q_dims, q_str, q_dtype, _Q_UID)
        k_t = _t("K", k_dims, k_str, k_dtype, _K_UID)
        v_t = _t("V", v_dims, v_str, v_dtype, _V_UID)

        attrs = hipdnn.SdpaAttributes()
        # Scalar softmax scale. The binding is named ``set_attn_scale`` in newer
        # frontends and ``set_attn_scale_value`` in older ones (same float signature);
        # accept whichever this build exposes -- version skew, not a capability gap.
        _set_scale = (
            getattr(attrs, "set_attn_scale", None) or attrs.set_attn_scale_value
        )
        _set_scale(float(scale))
        attrs.set_causal_mask(bool(causal))
        if bias_spec is not None:
            b_dims, b_str, b_dtype = bias_spec
            attrs.set_bias(_t("bias", b_dims, b_str, b_dtype, _BIAS_UID))

        o_t = g.sdpa(q_t, k_t, v_t, attrs)[0]  # [o, stats]; stats is None
        o_t.set_dim(list(o_dims))
        o_t.set_stride(list(o_str))
        o_t.set_data_type(st.dtype_map[o_dtype])
        o_t.set_uid(_O_UID)
        o_t.set_output(True)  # terminal output must be non-virtual (MIOpen requires it)
        return g

    def _fwd_stats_graph(
        self,
        q_dims,
        q_str,
        q_dtype,
        k_dims,
        k_str,
        k_dtype,
        v_dims,
        v_str,
        v_dtype,
        o_dims,
        o_str,
        o_dtype,
        stats_dims,
        stats_str,
        scale,
        causal,
        dropout_p,
        bias_spec,
    ):
        """The grad-eligible forward graph. Like :meth:`_graph`, but also
        requests stats (natural-log LSE, FLOAT, packed contiguous, explicit
        dtype) and, at ``dropout_p > 0``, wires ``set_dropout``."""
        st = self.state
        hipdnn = st.hipdnn
        hq = st.dtype_map[q_dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hq)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        def _t(name, dims, strides, dtype, uid):
            return g.tensor(
                hipdnn.Tensor()
                .set_name(name)
                .set_dim(list(dims))
                .set_stride(list(strides))
                .set_data_type(dtype)
                .set_uid(uid)
            )

        q_t = _t("Q", q_dims, q_str, st.dtype_map[q_dtype], _Q_UID)
        k_t = _t("K", k_dims, k_str, st.dtype_map[k_dtype], _K_UID)
        v_t = _t("V", v_dims, v_str, st.dtype_map[v_dtype], _V_UID)

        attrs = hipdnn.SdpaAttributes()
        _set_scale = (
            getattr(attrs, "set_attn_scale", None) or attrs.set_attn_scale_value
        )
        _set_scale(float(scale))
        attrs.set_causal_mask(bool(causal))
        attrs.set_generate_stats(True)
        if dropout_p:
            seed_t = _t("seed", [1], [1], hipdnn.DataType.INT64, _SEED_UID)
            offset_t = _t("offset", [1], [1], hipdnn.DataType.INT64, _OFFSET_UID)
            attrs.set_dropout(float(dropout_p), seed_t, offset_t)
        if bias_spec is not None:
            b_dims, b_str, b_dtype = bias_spec
            attrs.set_bias(_t("bias", b_dims, b_str, st.dtype_map[b_dtype], _BIAS_UID))

        o_t, stats_t = g.sdpa(q_t, k_t, v_t, attrs)
        o_t.set_dim(list(o_dims))
        o_t.set_stride(list(o_str))
        o_t.set_data_type(st.dtype_map[o_dtype])
        o_t.set_uid(_O_UID)
        o_t.set_output(True)
        stats_t.set_dim(list(stats_dims))
        stats_t.set_stride(list(stats_str))
        stats_t.set_data_type(hipdnn.DataType.FLOAT)
        stats_t.set_uid(_STATS_UID)
        stats_t.set_output(True)
        return g

    def _bwd_graph(
        self,
        q_dims,
        q_str,
        q_dtype,
        k_dims,
        k_str,
        k_dtype,
        v_dims,
        v_str,
        v_dtype,
        o_dims,
        o_str,
        o_dtype,
        stats_dims,
        stats_str,
        dq_dims,
        dq_str,
        dk_dims,
        dk_str,
        dv_dims,
        dv_str,
        scale,
        causal,
        dropout_p,
        bias_spec,
    ):
        """The paired backward graph. dO is declared in O's own dims/strides;
        dQ/dK/dV are fresh buffers in each source tensor's own memory order and
        dtype (``_output_layout`` generalised; GQA dK/dV keep Hkv)."""
        st = self.state
        hipdnn = st.hipdnn
        hq = st.dtype_map[q_dtype]
        g = hipdnn.Graph()
        g.set_io_data_type(hq)
        g.set_compute_data_type(hipdnn.DataType.FLOAT)

        def _t(name, dims, strides, dtype, uid):
            return g.tensor(
                hipdnn.Tensor()
                .set_name(name)
                .set_dim(list(dims))
                .set_stride(list(strides))
                .set_data_type(dtype)
                .set_uid(uid)
            )

        q_t = _t("Q", q_dims, q_str, st.dtype_map[q_dtype], _Q_UID)
        k_t = _t("K", k_dims, k_str, st.dtype_map[k_dtype], _K_UID)
        v_t = _t("V", v_dims, v_str, st.dtype_map[v_dtype], _V_UID)
        o_t = _t("O", o_dims, o_str, st.dtype_map[o_dtype], _O_UID)
        do_t = _t("dO", o_dims, o_str, st.dtype_map[o_dtype], _DO_UID)
        stats_t = _t("stats", stats_dims, stats_str, hipdnn.DataType.FLOAT, _STATS_UID)

        attrs = hipdnn.SdpaBackwardAttributes()
        _set_scale = (
            getattr(attrs, "set_attn_scale", None) or attrs.set_attn_scale_value
        )
        _set_scale(float(scale))
        attrs.set_causal_mask(bool(causal))
        if dropout_p:
            seed_t = _t("seed", [1], [1], hipdnn.DataType.INT64, _SEED_UID)
            offset_t = _t("offset", [1], [1], hipdnn.DataType.INT64, _OFFSET_UID)
            attrs.set_dropout(float(dropout_p), seed_t, offset_t)
        if bias_spec is not None:
            b_dims, b_str, b_dtype = bias_spec
            attrs.set_bias(_t("bias", b_dims, b_str, st.dtype_map[b_dtype], _BIAS_UID))

        dq_t, dk_t, dv_t = g.sdpa_backward(q_t, k_t, v_t, o_t, do_t, stats_t, attrs)
        dq_t.set_dim(list(dq_dims))
        dq_t.set_stride(list(dq_str))
        dq_t.set_data_type(st.dtype_map[q_dtype])
        dq_t.set_uid(_DQ_UID)
        dq_t.set_output(True)
        dk_t.set_dim(list(dk_dims))
        dk_t.set_stride(list(dk_str))
        dk_t.set_data_type(st.dtype_map[k_dtype])
        dk_t.set_uid(_DK_UID)
        dk_t.set_output(True)
        dv_t.set_dim(list(dv_dims))
        dv_t.set_stride(list(dv_str))
        dv_t.set_data_type(st.dtype_map[v_dtype])
        dv_t.set_uid(_DV_UID)
        dv_t.set_output(True)
        return g

    def _call(
        self,
        real,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        def _native():
            return real(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        torch = self.state.torch
        # Census dims are best-effort; guard the shape reads for exotic call sites.
        try:
            sq = int(query.shape[-2])
            skv = int(key.shape[-2])
            d = int(query.shape[-1])
        except Exception as ex:  # noqa: BLE001
            _reraise_checkpoint_stop(torch, ex)
            return _native()
        base_key = f"Sq={sq},Skv={skv},D={d},dtype={self._tok(query.dtype)}"

        ok, reason = self._gate(query)
        if not ok:
            self.note_native(base_key, reason)
            return _native()

        dropout_p = float(dropout_p)
        grad_enabled = torch.is_grad_enabled()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        capturing = self._safe_bool(
            getattr(torch.cuda, "is_current_stream_capturing", None)
        )
        compiling = self._safe_bool(getattr(torch.compiler, "is_compiling", None))

        sdp_choice = None
        ck_preferred = False
        if dropout_p:
            sdp_choice = self._sdp_choice(
                torch,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
            )
            ck_preferred = self._ck_preferred(torch)

        route, native_reason = _grad_route(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            grad_enabled=grad_enabled,
            deterministic=deterministic,
            warn_only=warn_only,
            capturing=capturing,
            sdp_choice=sdp_choice,
            ck_preferred=ck_preferred,
            compiling=compiling,
        )

        if route == "native":
            self.note_native(GRAD_PREFIX + base_key, native_reason)
            return _native()
        if route == "plain":
            return self._call_plain(
                _native, query, key, value, attn_mask, is_causal, scale, base_key
            )
        return self._call_grad(
            real,
            _native,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            base_key,
            sdp_choice,
            grad_enabled,
        )

    def _call_plain(
        self, _native, query, key, value, attn_mask, is_causal, scale, base_key
    ):
        torch = self.state.torch
        eff_scale = (
            float(scale) if scale is not None else 1.0 / math.sqrt(int(query.shape[-1]))
        )
        try:
            # attn_mask -> additive bias. A bool mask (True=keep) is materialized to
            # a 0/-inf additive tensor; a float mask is used as-is. We own the
            # materialized tensor, so its strides are known.
            bias, bias_spec, extras = self._materialize_bias(torch, query, attn_mask)

            # Output takes Q's dimension order with V's head dim (Dv). We own it,
            # so its strides are known. See the module docstring for why this does
            # not copy any one native backend's layout.
            dv = int(value.shape[-1])
            alloc_dims, inverse = _output_layout(
                tuple(query.shape), tuple(query.stride()), dv
            )
            o = torch.empty(alloc_dims, dtype=query.dtype, device=query.device).permute(
                inverse
            )

            entry = self._cached_graph(
                (
                    tuple(query.shape),
                    tuple(query.stride()),
                    query.dtype,
                    tuple(key.shape),
                    tuple(key.stride()),
                    key.dtype,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.dtype,
                    tuple(o.stride()),
                    eff_scale,
                    bool(is_causal),
                    bias_spec,
                ),
                lambda: self._graph(
                    list(query.shape),
                    list(query.stride()),
                    query.dtype,
                    list(key.shape),
                    list(key.stride()),
                    key.dtype,
                    list(value.shape),
                    list(value.stride()),
                    value.dtype,
                    list(o.shape),
                    list(o.stride()),
                    query.dtype,
                    eff_scale,
                    bool(is_causal),
                    bias_spec,
                ),
                f"{list(query.shape)}x{list(key.shape)}x{list(value.shape)}"
                f"{' causal' if is_causal else ''} {query.dtype}",
            )
            variant = {
                _Q_UID: query.data_ptr(),
                _K_UID: key.data_ptr(),
                _V_UID: value.data_ptr(),
                _O_UID: o.data_ptr(),
            }
            if bias is not None:
                variant[_BIAS_UID] = bias.data_ptr()
            self._execute(entry, variant, query.device)
            self.note_aot(base_key, entry["engine"], **extras)
            return o
        except NotApplicable as na:  # no engine claims this shape -> native
            self.note_native(base_key, str(na))
            return _native()
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            _reraise_checkpoint_stop(torch, ex)
            self.note_native(
                base_key,
                f"exception: {type(ex).__name__}: {ex}",
                level=logging.WARNING,
            )
            return _native()

    def _call_grad(
        self,
        real,
        _native,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        base_key,
        sdp_choice,
        grad_enabled,
    ):
        torch = self.state.torch
        grad_key = GRAD_PREFIX + base_key
        bwd_key = BWD_PREFIX + base_key

        eff_scale = (
            float(scale) if scale is not None else 1.0 / math.sqrt(int(query.shape[-1]))
        )

        decline_reason = None
        try:
            bias, bias_spec, mask_extras = self._materialize_bias(
                torch, query, attn_mask
            )

            dv = int(value.shape[-1])
            alloc_dims, inverse = _output_layout(
                tuple(query.shape), tuple(query.stride()), dv
            )
            o_view_stride = _view_strides(alloc_dims, inverse)
            o_view_dims = tuple(query.shape[:-1]) + (dv,)
            stats_dims = tuple(int(s) for s in query.shape[:3]) + (1,)
            stats_stride = _view_strides(list(stats_dims), list(range(4)))

            q_dq_alloc, q_dq_inv = _output_layout(
                tuple(query.shape), tuple(query.stride()), int(query.shape[-1])
            )
            k_dk_alloc, k_dk_inv = _output_layout(
                tuple(key.shape), tuple(key.stride()), int(key.shape[-1])
            )
            v_dv_alloc, v_dv_inv = _output_layout(
                tuple(value.shape), tuple(value.stride()), int(value.shape[-1])
            )
            dq_stride = _view_strides(q_dq_alloc, q_dq_inv)
            dk_stride = _view_strides(k_dk_alloc, k_dk_inv)
            dv_stride = _view_strides(v_dv_alloc, v_dv_inv)

            try:
                entry_fwd = self._cached_graph(
                    (
                        "fwd+stats",
                        tuple(query.shape),
                        tuple(query.stride()),
                        query.dtype,
                        tuple(key.shape),
                        tuple(key.stride()),
                        key.dtype,
                        tuple(value.shape),
                        tuple(value.stride()),
                        value.dtype,
                        o_view_stride,
                        eff_scale,
                        bool(is_causal),
                        dropout_p,
                        bias_spec,
                    ),
                    lambda: self._fwd_stats_graph(
                        list(query.shape),
                        list(query.stride()),
                        query.dtype,
                        list(key.shape),
                        list(key.stride()),
                        key.dtype,
                        list(value.shape),
                        list(value.stride()),
                        value.dtype,
                        list(o_view_dims),
                        list(o_view_stride),
                        query.dtype,
                        list(stats_dims),
                        list(stats_stride),
                        eff_scale,
                        bool(is_causal),
                        dropout_p,
                        bias_spec,
                    ),
                    f"fwd+stats {list(query.shape)}x{list(key.shape)}x{list(value.shape)} {query.dtype}",
                )
            except NotApplicable as na:
                decline_reason = f"grad: fwd+stats: {na}"

            if decline_reason is None:
                try:
                    entry_bwd = self._cached_graph(
                        (
                            "bwd",
                            tuple(query.shape),
                            tuple(query.stride()),
                            query.dtype,
                            tuple(key.shape),
                            tuple(key.stride()),
                            key.dtype,
                            tuple(value.shape),
                            tuple(value.stride()),
                            value.dtype,
                            o_view_stride,
                            eff_scale,
                            bool(is_causal),
                            dropout_p,
                            bias_spec,
                        ),
                        lambda: self._bwd_graph(
                            list(query.shape),
                            list(query.stride()),
                            query.dtype,
                            list(key.shape),
                            list(key.stride()),
                            key.dtype,
                            list(value.shape),
                            list(value.stride()),
                            value.dtype,
                            list(o_view_dims),
                            list(o_view_stride),
                            query.dtype,
                            list(stats_dims),
                            list(stats_stride),
                            list(query.shape),
                            list(dq_stride),
                            list(key.shape),
                            list(dk_stride),
                            list(value.shape),
                            list(dv_stride),
                            eff_scale,
                            bool(is_causal),
                            dropout_p,
                            bias_spec,
                        ),
                        f"bwd {list(query.shape)}x{list(key.shape)}x{list(value.shape)} {query.dtype}",
                    )
                except NotApplicable as na:
                    decline_reason = f"grad: bwd: {na}"
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- e.g. a wheel missing Graph.sdpa/sdpa_backward
            _reraise_checkpoint_stop(torch, ex)
            self.note_native(
                grad_key, f"exception: {type(ex).__name__}: {ex}", level=logging.WARNING
            )
            return _native()

        if decline_reason is not None:
            self.note_native(grad_key, decline_reason)
            return _native()

        gen = None
        offset_before = None
        seed_t = offset_t = None
        try:
            if dropout_p:
                gen = torch.cuda.default_generators[query.device.index]
                seed, offset, offset_before = self._draw_dropout_state(
                    gen,
                    sdp_choice,
                    int(query.shape[0]),
                    int(query.shape[1]),
                    int(query.shape[2]),
                    int(key.shape[2]),
                )
                seed_t, offset_t = self._dropout_tensors(
                    torch, seed, offset, query.device
                )

            payload = {
                "ov": self,
                "entry_fwd": entry_fwd,
                "entry_bwd": entry_bwd,
                "alloc_dims": list(alloc_dims),
                "inverse": list(inverse),
                "o_dtype": query.dtype,
                "o_view_stride": o_view_stride,
                "stats_dims": stats_dims,
                "dropout_p": dropout_p,
                "is_causal": bool(is_causal),
                "scale": eff_scale,
                "enable_gqa": bool(enable_gqa),
                "real": real,
                "grad_key": grad_key,
                "bwd_key": bwd_key,
                "mask_extras": mask_extras,
                "dq_layout": (q_dq_alloc, q_dq_inv),
                "dk_layout": (k_dk_alloc, k_dk_inv),
                "dv_layout": (v_dv_alloc, v_dv_inv),
            }

            if not grad_enabled:
                # The no-grad first pass of a reentrant checkpoint still needs
                # the fwd-with-stats graph (to draw dropout identically to its
                # later recompute), but never runs the Function -- there is no
                # backward to attach, and the stats are simply discarded.
                device = query.device
                o_dense = torch.empty(alloc_dims, dtype=query.dtype, device=device)
                stats = torch.empty(stats_dims, dtype=torch.float32, device=device)
                variant = {
                    _Q_UID: query.data_ptr(),
                    _K_UID: key.data_ptr(),
                    _V_UID: value.data_ptr(),
                    _O_UID: o_dense.data_ptr(),
                    _STATS_UID: stats.data_ptr(),
                }
                if bias is not None:
                    variant[_BIAS_UID] = bias.data_ptr()
                if seed_t is not None:
                    variant[_SEED_UID] = seed_t.data_ptr()
                    variant[_OFFSET_UID] = offset_t.data_ptr()
                self._execute(entry_fwd, variant, device)
                recompute_extra = {}
                if torch.is_grad_enabled() and torch._C._current_graph_task_id() != -1:
                    recompute_extra["recompute"] = 1
                dropout_extra = {"dropout": 1} if dropout_p else {}
                self.note_aot(
                    grad_key,
                    entry_fwd["engine"],
                    **recompute_extra,
                    **dropout_extra,
                    **mask_extras,
                )
                o = o_dense.permute(inverse)
            else:
                fn = _sdpa_autograd_fn(torch)
                o_dense = fn.apply(query, key, value, bias, seed_t, offset_t, payload)
                o = o_dense.permute(inverse)
        except NotApplicable as na:
            if gen is not None and offset_before is not None:
                gen.set_offset(offset_before)
            self.note_native(grad_key, str(na))
            return _native()
        except (
            Exception
        ) as ex:  # noqa: BLE001 -- any failure -> native, never break the model
            _reraise_checkpoint_stop(torch, ex)
            if gen is not None and offset_before is not None:
                gen.set_offset(offset_before)
            self.note_native(
                grad_key, f"exception: {type(ex).__name__}: {ex}", level=logging.WARNING
            )
            return _native()

        return o
