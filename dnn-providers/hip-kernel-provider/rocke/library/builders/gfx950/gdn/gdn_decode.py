#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side driver for the GDN single-token decode kernel.

The kernel emitter in ``kernels/gfx950/gdn_decode.py`` produces device code.
This module is everything on the host needed to actually exercise it: compile a
spec, build inputs, run it, judge the result against an independent reference,
and time it.

The reference is deliberately *not* a restatement of the kernel. It evaluates
the gated delta rule with whole-tensor fp32 arithmetic and knows nothing about
the kernel's workgroup mapping, warp tiling or cross-lane reductions, so
agreement is evidence about the algorithm rather than a tautology.

Both outputs are checked. The kernel writes ``out`` *and* mutates the recurrent
state in place; a driver that only compares ``out`` would let a corrupted state
write ship silently, because the state is not read back until the next decode
step.

Run::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python \\
        python3 gdn_decode.py --batches 1,16 --variant tiled
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
import struct
import sys
from typing import Dict, Tuple

import torch

from rocke.helpers.activations import SOFTPLUS_THRESHOLD
from kernels.gfx950.gdn_decode import (
    CONV_WIDTH,
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_grid,
    gdn_decode_pointer_alignment,
    gdn_decode_signature,
    is_valid_spec,
)
from rocke.helpers.compile import compile_kernel
from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence
from rocke.runtime import synchronize_and_release

# bf16 inputs against an fp32 reference. Observed error grows with batch (more
# state rows accumulate into one output), so the bound is set well above the
# largest value seen at the batches exercised here while staying far below the
# magnitude of a real indexing or reduction bug.
TOL = 1e-2

_ARCH = "gfx950"
_LAUNCHER_CACHE: Dict[Tuple, KernelLauncher] = {}

# Spec dtype name -> the torch dtype the kernel is compiled against. The kernel
# receives a raw pointer, so this mapping is the only thing tying a caller's
# tensor to the element type frozen into the machine code.
# f32 is a state dtype only; is_valid_spec rejects it as the I/O dtype.
_TORCH_DT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
# Smallest normal f32, 2**-126 (torch.finfo(torch.float32).tiny).
_F32_MIN_NORMAL = 2.0**-126


def launcher_for(spec: GdnDecodeSpec, arch: str = _ARCH) -> KernelLauncher:
    """Compile ``spec`` and wrap it in a launcher, memoised per spec.

    The spec is validated before the cache is read. A flag of the wrong type
    (``dpp_reduce="false"`` is truthy) gets the same ``kernel_name()`` as the
    valid spec it imitates, so a cache hit would hand it that spec's launcher
    instead of raising.

    Keyed on ``kernel_name()`` because that string is what the compiled code
    object is identified by; every field that changes emitted code is encoded
    in it, so two valid specs cannot collide on one cache entry.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid gdn_decode spec for {arch}: {why}")
    key = (spec.kernel_name(), arch)
    cached = _LAUNCHER_CACHE.get(key)
    if cached is not None:
        return cached
    artifact = compile_kernel(build_gdn_decode(spec, arch=arch), arch=arch)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=gdn_decode_signature(spec),
    )
    _LAUNCHER_CACHE[key] = launcher
    return launcher


def _gate_input(spec: GdnDecodeSpec, rnd, batch: int, hv: int, dk: int):
    """The `a` tensor, whose meaning depends on the gate mode.

    * GDN: one raw logit per head, `[B, 1, HV]`.
    * KDA, fuse_gate=True: one raw logit per K channel, `[B, 1, HV, DK]`; the
      kernel turns it into a decay.
    * KDA, fuse_gate=False: the natural-log-domain decay itself. Forced
      non-positive so `exp(a)` lands in (0, 1] -- a positive value would
      amplify the state every step rather than fade it, which the fused path
      cannot produce and so is not worth generating.

    Draws exactly one tensor in every mode, so callers can keep this at a fixed
    position in the input dict and leave the random stream undisturbed.
    """
    if spec.gate_kind != "kda":
        return rnd(batch, 1, hv)
    g = rnd(batch, 1, hv, dk, scale=0.5)
    return g if spec.fuse_gate else -g.abs()


def make_inputs(
    spec: GdnDecodeSpec,
    batch: int,
    seed: int = 0,
    device: str = "cuda",
    pool_depth: int | None = None,
    disjoint_writes: bool = True,
):
    """Deterministic inputs matching the kernel's packed decode contract.

    ``pool_depth`` defaults to ``2 * batch + 1`` and ``disjoint_writes`` to
    True, so the pool is wider than the batch and the read and write slots do
    not overlap. That default is load-bearing, not incidental: a pool sized
    exactly ``batch`` with ``read == write == arange(batch)`` leaves no
    untouched page, and the failure a paged pool invites -- a correct value
    written into the WRONG slot -- is then invisible, because the slot written
    checks out and the slot damaged is never compared. The spare pages are the
    detector that :func:`check` reads.

    Pass ``pool_depth=batch, disjoint_writes=False`` for the exactly-fitted
    pool. That configuration cannot see a misplaced write, so use it only when
    the fit itself is the thing under test.
    """
    torch_dtype = _TORCH_DT[spec.dtype]
    state_dtype = _TORCH_DT[spec.state_dtype]
    depth = 2 * batch + 1 if pool_depth is None else pool_depth
    if depth < (2 * batch if disjoint_writes else batch):
        raise ValueError(
            f"pool_depth {depth} too small for batch {batch} with "
            f"disjoint_writes={disjoint_writes}"
        )
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dtype=torch_dtype, scale=1.0):
        t = torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)
        return (t * scale).to(dtype)

    hk, hv = spec.num_k_heads, spec.num_v_heads
    dk, dv = spec.head_k_dim, spec.head_v_dim
    inp = {
        "query": rnd(batch, 1, hk, dk),
        "key": rnd(batch, 1, hk, dk),
        "value": rnd(batch, 1, hv, dv),
        # `a` stays at this position in the dict so the GDN draw order -- and
        # therefore every seeded GDN input ever recorded -- is bit-for-bit
        # unchanged. See _gate_input for what it holds in each mode.
        "a": _gate_input(spec, rnd, batch, hv, dk),
        "b": rnd(batch, 1, hv),
        # f32 for KDA: the gate is evaluated in f32 and KDA prefill already
        # declares this tensor f32, so the two families share one contract.
        "dt_bias": (
            torch.randn(hv, dk, device=device, generator=gen, dtype=torch.float32) * 0.1
            if spec.gate_kind == "kda"
            else rnd(hv)
        ),
        "A_log": torch.randn(hv, device=device, generator=gen, dtype=torch.float32),
        "read_indices": torch.arange(batch, device=device, dtype=torch.int32),
        "write_indices": (
            torch.arange(batch, 2 * batch, device=device, dtype=torch.int32)
            if disjoint_writes
            else torch.arange(batch, device=device, dtype=torch.int32)
        ),
        "state": rnd(depth, hv, dv, dk, dtype=state_dtype, scale=0.01),
    }
    # Fused-mode tensors are drawn after every unfused draw, so a fused spec's
    # gate and state inputs equal its unfused twin's, seed for seed.
    if spec.fuse_conv:
        conv_dim = spec.conv_dim
        mixed = rnd(batch, conv_dim)
        inp["mixed_qkv"] = mixed
        inp["qkv_stride"] = conv_dim
        # The packed row REPLACES query/key/value in the kernel contract, so
        # the fixture does not carry them: the reference derives q/k/v from
        # the packed row, and a caller that follows the contract is tested.
        for name in ("query", "key", "value"):
            del inp[name]
        inp["conv_state"] = rnd(depth, conv_dim, CONV_WIDTH - 1)
        inp["conv_weight"] = (
            torch.randn(
                conv_dim, CONV_WIDTH, device=device, generator=gen, dtype=torch.float32
            )
            * 0.5
        )
    if spec.fuse_out_norm:
        inp["out_gate"] = rnd(batch, hv * dv)
        inp["og_stride"] = hv * dv
        inp["norm_weight"] = 1.0 + 0.1 * torch.randn(
            dv, device=device, generator=gen, dtype=torch.float32
        )
        inp["norm_eps"] = 1e-6
    return inp


def precompute_kda_log_decay(spec: GdnDecodeSpec, inp) -> torch.Tensor:
    """Natural-log decay for the benchmark-only precomputed KDA mode.

    The fp32 oracle deliberately does not call this helper: benchmark
    preparation and the correctness reference must not share formula code, or
    the same bug can make both arms agree.
    """
    if spec.gate_kind != "kda" or not spec.fuse_gate:
        raise ValueError("precompute_kda_log_decay requires fused KDA raw logits")
    inner = torch.exp(inp["A_log"].float())[None, :, None] * (
        inp["a"][:, 0].float() + inp["dt_bias"].float()
    )
    return spec.lower_bound * torch.sigmoid(inner)


def _ref_conv(spec: GdnDecodeSpec, inp) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 causal conv1d (width ``CONV_WIDTH``) + SiLU over the packed q/k/v row.

    Returns ``(convolved [B, conv_dim], taps_after [B, conv_dim, CONV_WIDTH-1])``:
    tap 0 is the oldest, and the shift drops it and appends the current token.
    Only the first ``spec.conv_dim`` channels of each row belong to the
    kernel, since a row may be a slice of a wider buffer.
    """
    conv_dim = spec.conv_dim
    taps = inp["conv_state"].float()[inp["read_indices"].long()]  # [B, C, W-1]
    x = inp["mixed_qkv"][:, :conv_dim].float()  # [B, C]
    w = inp["conv_weight"].float()  # [C, W]
    if taps.shape[1:] != (conv_dim, CONV_WIDTH - 1) or w.shape != (
        conv_dim,
        CONV_WIDTH,
    ):
        raise ValueError(
            f"conv tensors {tuple(taps.shape[1:])} / {tuple(w.shape)} do not match "
            f"spec.conv_dim {conv_dim} and width {CONV_WIDTH}"
        )
    window = torch.cat((taps, x[..., None]), dim=-1)  # [B, C, W], oldest first
    y = torch.nn.functional.silu((window * w).sum(-1))
    return y, window[..., 1:]


def ref_conv_state_after(spec: GdnDecodeSpec, inp) -> torch.Tensor:
    """Expected conv-state taps written to each write slot: [B, conv_dim, W-1]."""
    return _ref_conv(spec, inp)[1]


def out_error(spec: GdnDecodeSpec, out: torch.Tensor, ref_out: torch.Tensor) -> float:
    """Max output error against the fp32 reference, comparable with ``TOL``.

    The plain decode output is small (|o| << 1), so the absolute error is used.
    With ``fuse_out_norm`` the output is RMS-normalised and gated, |o| reaches
    ~5-7 for these inputs, and one bf16 rounding step of the exact reference
    alone exceeds ``TOL`` there; the error is then taken relative to
    ``max(1, |ref|)`` (identical to the absolute error wherever |ref| <= 1, and
    at most |ref| times smaller than it elsewhere).
    """
    diff = (out.float().reshape(ref_out.shape) - ref_out).abs()
    if spec.fuse_out_norm:
        diff = diff / ref_out.abs().clamp(min=1.0)
    return diff.max().item()


def ref_fp32(spec: GdnDecodeSpec, inp) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whole-tensor fp32 reference for one decode step.

    Returns ``(out, state_after)``. Written directly from the gated delta rule
    with no chunking, tiling or cross-lane structure, so it shares no algebra
    with the kernel beyond the definition itself. Fused specs apply the conv1d
    to q/k/v first (:func:`_ref_conv`) and the gated RMSNorm to ``out`` last;
    the norm runs on the fp32 output.
    """
    if spec.fuse_conv:
        y, _ = _ref_conv(spec, inp)
        hk, dk, dv = spec.num_k_heads, spec.head_k_dim, spec.head_v_dim
        bsz = y.shape[0]
        inp = dict(inp)
        inp["query"] = y[:, : hk * dk].reshape(bsz, 1, hk, dk)
        inp["key"] = y[:, hk * dk : 2 * hk * dk].reshape(bsz, 1, hk, dk)
        inp["value"] = y[:, 2 * hk * dk :].reshape(bsz, 1, spec.num_v_heads, dv)
    hv, g = spec.num_v_heads, spec.v_per_k_head
    scale = 1.0 / math.sqrt(spec.head_k_dim)
    eps = 1e-6  # q/k L2-norm eps (the kernel's NORM_EPS); not the output norm's

    # Each value head reads the key head it is grouped under.
    k_of_v = torch.arange(hv, device=inp["query"].device) // g
    q = inp["query"][:, 0].float()[:, k_of_v]  # [B, HV, DK]
    k = inp["key"][:, 0].float()[:, k_of_v]

    if spec.use_qk_l2norm:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + eps) * scale
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + eps)
    else:
        # l2norm off: the kernel scales q by 1/sqrt(dk) and leaves k raw.
        q = q * scale

    if spec.gate_kind == "kda":
        if spec.fuse_gate:
            # Per-channel gate, bounded in (lower_bound, 0) before the exp.
            #   log_decay[d] = lower_bound * sigmoid(exp(A_log[h]) * (g[d] + dt_bias[h,d]))
            inner = torch.exp(inp["A_log"].float())[None, :, None] * (
                inp["a"][:, 0].float() + inp["dt_bias"].float()
            )
            decay = torch.exp(spec.lower_bound * torch.sigmoid(inner))  # [B, HV, DK]
        else:
            # `a` already carries natural-log-domain decay; only the exp remains.
            # A_log and dt_bias are unread here -- the caller folded them in.
            # This mirrors the competitor recurrence-only kernel, which likewise
            # applies exp(gk) in-kernel and nothing else, which is what makes the
            # two timeable at one work boundary.
            decay = torch.exp(inp["a"][:, 0].float())  # [B, HV, DK]
    else:
        # Scalar gate, unbounded below.
        #   log_decay = -exp(A_log[h]) * softplus(a + dt_bias[h])
        x = inp["a"][:, 0].float() + inp["dt_bias"].float()  # [B, HV]
        softplus = torch.where(x > SOFTPLUS_THRESHOLD, x, torch.log1p(torch.exp(x)))
        decay = torch.exp(-torch.exp(inp["A_log"].float()) * softplus)  # [B, HV]
    beta = torch.sigmoid(inp["b"][:, 0].float())

    state = inp["state"].float()[inp["read_indices"].long()]  # [B, HV, DV, DK]
    # The decay multiplies S from the right, so a per-channel gate scales
    # columns (the DK axis) and a scalar gate scales the whole matrix.
    s = state * (
        decay[..., None, :] if spec.gate_kind == "kda" else decay[..., None, None]
    )

    sk = (s @ k[..., None]).squeeze(-1)  # [B, HV, DV]
    sq = (s @ q[..., None]).squeeze(-1)
    v_new = (inp["value"][:, 0].float() - sk) * beta[..., None]
    kq = (k * q).sum(-1)  # [B, HV]

    out = sq + v_new * kq[..., None]
    s_after = s + v_new[..., None] * k[..., None, :]
    if spec.fuse_out_norm:
        # Gate activation by gate kind, stated here from the models rather
        # than read from the kernel's table, so a wrong table fails the check:
        # SiLU for GDN (Qwen3-Next), sigmoid for KDA (Kimi Linear);
        # ALGORITHM.md section 1.4.
        act = torch.nn.functional.silu if spec.gate_kind == "gdn" else torch.sigmoid
        og = inp["out_gate"][:, : hv * spec.head_v_dim].float().reshape(out.shape)
        rstd = torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + inp["norm_eps"])
        out = out * rstd * inp["norm_weight"].float() * act(og)
    return out.unsqueeze(1), s_after


def _validate_decode_inputs(
    spec: GdnDecodeSpec, inp, batch: int, *, validate_indices: bool = True
):
    """Reject inputs that would make the decode kernel read or write out of bounds.

    The kernel indexes the state pool through ``read_indices``/``write_indices``
    with a raw pointer add and only skips the ``-1`` sentinel; it never
    hardware-bounds the index against the pool depth, so an out-of-range index
    is an out-of-bounds load/store. These host checks catch that before launch.

    Shape and dtype checks are sync-free and always run -- including the state's
    element type, which no device-side check can catch: the kernel derives the
    pool's byte stride from ``spec.state_dtype``. Between ``bf16`` and ``f16``
    (both 16 bits) a mismatched pool computes every address identically and
    merely decodes the bits under the wrong rule, silently corrupting the
    recurrence it feeds back; a 16-bit pool given to an ``f32`` spec is read
    and written twice as far, past the end of the pool. Every pointer must
    also sit on the alignment the emitted vector accesses assume
    (``gdn_decode_pointer_alignment``) and on one device. The index *value*
    range check reads the index extrema, forcing a device->host sync, so it is
    gated by ``validate_indices`` (default on; a hot re-prepare loop whose
    indices are already known good may pass ``False``).
    """
    if spec.gate_kind == "kda":
        # The emitter indexes these buffers from compile-time B/HV/DK extents:
        # unlike the scalar GDN ABI, a legacy-sized allocation is too short and
        # the vector loads would read beyond it. Metadata checks are sync-free,
        # so they always run even when index-value validation is disabled.
        a = inp["a"]
        want_a = (batch, 1, spec.num_v_heads, spec.head_k_dim)
        if tuple(a.shape) != want_a:
            raise ValueError(f"KDA a shape {tuple(a.shape)} != {want_a}")
        want_a_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[spec.dtype]
        if a.dtype != want_a_dtype:
            raise ValueError(f"KDA a dtype {a.dtype} != {want_a_dtype}")
        if not a.is_contiguous():
            raise ValueError("KDA a must be contiguous")

        dt_bias = inp["dt_bias"]
        want_dt_bias = (spec.num_v_heads, spec.head_k_dim)
        if tuple(dt_bias.shape) != want_dt_bias:
            raise ValueError(
                f"KDA dt_bias shape {tuple(dt_bias.shape)} != {want_dt_bias}"
            )
        if dt_bias.dtype != torch.float32:
            raise ValueError(f"KDA dt_bias dtype {dt_bias.dtype} != torch.float32")
        if not dt_bias.is_contiguous():
            raise ValueError("KDA dt_bias must be contiguous")

    state = inp["state"]
    if state.ndim != 4:
        raise ValueError(f"state must be [pool, HV, DV, DK]; got {tuple(state.shape)}")
    pool_depth = state.shape[0]
    want = (spec.num_v_heads, spec.head_v_dim, spec.head_k_dim)
    if tuple(state.shape[1:]) != want:
        raise ValueError(f"state head dims {tuple(state.shape[1:])} != spec {want}")
    want_dt = _TORCH_DT[spec.state_dtype]
    if state.dtype is not want_dt:
        raise ValueError(
            f"state dtype {state.dtype} != spec.state_dtype "
            f"{spec.state_dtype} ({want_dt}); the kernel is compiled with a "
            f"{spec.state_dtype} pointer and would reinterpret these bytes"
        )
    for name in ("read_indices", "write_indices"):
        idx = inp[name]
        if idx.dtype != torch.int32:
            raise ValueError(f"{name} must be int32; got {idx.dtype}")
        if tuple(idx.shape) != (batch,):
            raise ValueError(f"{name} must be [batch={batch}]; got {tuple(idx.shape)}")
    # The remaining buffers are addressed from SPEC constants, never from
    # their own shapes, and the kernel emits no buffer descriptor -- there is
    # no `num_records` to clamp an over-reach. Gate buffers differ by mode:
    # KDA's widened `a` / f32 `dt_bias` were validated above; rechecking them
    # here against GDN's scalar shapes would reject every valid KDA launch.
    hk, hv = spec.num_k_heads, spec.num_v_heads
    dk, dv = spec.head_k_dim, spec.head_v_dim
    io_dt = _TORCH_DT[spec.dtype]
    want_shapes = {
        "query": (batch, 1, hk, dk),
        "key": (batch, 1, hk, dk),
        "value": (batch, 1, hv, dv),
        "b": (batch, 1, hv),
        "A_log": (hv,),
    }
    if spec.gate_kind == "gdn":
        want_shapes.update({"a": (batch, 1, hv), "dt_bias": (hv,)})
    if spec.fuse_conv:
        # The packed row replaces query/key/value in the kernel ABI.
        for name in ("query", "key", "value"):
            want_shapes.pop(name)
        conv_dim = spec.conv_dim
        _check_row_input(inp, "mixed_qkv", "qkv_stride", batch, conv_dim, io_dt)
        row_bytes = inp["qkv_stride"] * inp["mixed_qkv"].element_size()
        row_align = gdn_decode_pointer_alignment(spec)["mixed_qkv"]
        if row_bytes % row_align:
            # Each row is read with 16 B vector loads from row * qkv_stride.
            raise ValueError(
                f"qkv_stride {inp['qkv_stride']} puts mixed_qkv rows off the "
                f"{row_align} B alignment the kernel's vector loads assume"
            )
        _check_dense(inp, "conv_state", (pool_depth, conv_dim, CONV_WIDTH - 1), io_dt)
        _check_dense(inp, "conv_weight", (conv_dim, CONV_WIDTH), torch.float32)
    if spec.fuse_out_norm:
        _check_row_input(inp, "out_gate", "og_stride", batch, hv * dv, io_dt)
        _check_dense(inp, "norm_weight", (dv,), torch.float32)
        eps = inp["norm_eps"]
        # The kernarg packs eps as an f32, so judge the value the kernel gets:
        # 1e-50 rounds to 0.0 and an f32 denormal may be flushed to zero.
        eps32 = math.nan
        if not isinstance(eps, bool) and isinstance(eps, (int, float)):
            try:
                eps32 = struct.unpack("<f", struct.pack("<f", eps))[0]
            except OverflowError:
                pass
        if not (math.isfinite(eps32) and eps32 >= _F32_MIN_NORMAL):
            # rsqrt(mean(o^2) + eps): a zero, negative or NaN eps gives an
            # inf/NaN output for a small or zero row instead of an error.
            raise ValueError(
                "norm_eps must be a finite positive float that stays a normal "
                f"f32 (>= {_F32_MIN_NORMAL!r}); got {eps!r}"
            )
    for name, want_shape in want_shapes.items():
        t = inp[name]
        if tuple(t.shape) != want_shape:
            raise ValueError(
                f"{name} must be {want_shape} for this spec; got {tuple(t.shape)}"
            )
        # A_log is always f32. GDN's remaining gate buffers carry the I/O
        # dtype; KDA's f32 dt_bias was checked in its mode-specific block.
        want_t = torch.float32 if name == "A_log" else io_dt
        if t.dtype is not want_t:
            raise ValueError(f"{name} dtype {t.dtype} != {want_t} for this spec")
    # Every address is computed from the spec's dims, so a non-contiguous
    # tensor is read at the right INDEX in the wrong MEMORY -- the doc promises
    # row-major and nothing enforced it. `.stride()` is metadata: no sync.
    for name in list(want_shapes) + ["state", "read_indices", "write_indices"]:
        if not inp[name].is_contiguous():
            raise ValueError(
                f"{name} must be contiguous (row-major); got strides "
                f"{tuple(inp[name].stride())}"
            )
    # One device for every tensor the kernel receives (``out`` is allocated
    # by prepare() on that device). The launcher packs raw pointers, so a
    # tensor on another device would be dereferenced on this one.
    ref_name = _device_ref(spec)
    device = inp[ref_name].device
    for p in gdn_decode_signature(spec):
        name = p["name"]
        if name == "out" or not isinstance(inp.get(name), torch.Tensor):
            continue
        if inp[name].device != device:
            raise ValueError(
                f"{name} device {inp[name].device} != {ref_name} device {device}"
            )
    for name, align in gdn_decode_pointer_alignment(spec).items():
        if name != "out" and inp[name].data_ptr() % align:
            raise ValueError(
                f"{name} must start on a {align} B boundary (the kernel's vector "
                f"accesses assume it); got address {inp[name].data_ptr():#x}"
            )
    if not validate_indices:
        return
    for name in ("read_indices", "write_indices"):
        idx = inp[name]
        # -1 is the 'skip this slot' sentinel; every other value must land in
        # [0, pool_depth). min()/max() forces one device->host sync.
        lo = int(idx.min().item())
        hi = int(idx.max().item())
        if lo < -1 or hi >= pool_depth:
            raise ValueError(
                f"{name} out of range: [{lo}, {hi}] escapes -1 (skip) or "
                f"[0, {pool_depth})"
            )
    active = (inp["read_indices"] >= 0) & (inp["write_indices"] >= 0)
    active_writes = inp["write_indices"][active]
    if active_writes.numel() != active_writes.unique().numel():
        raise ValueError("write_indices must be unique across active sequences")


def _device_ref(spec: GdnDecodeSpec) -> str:
    """The input whose device every other kernel tensor must share."""
    return "mixed_qkv" if spec.fuse_conv else "query"


def _check_dense(inp, name: str, shape, dtype) -> None:
    """A contiguous tensor of exactly ``shape`` and ``dtype``."""
    t = inp[name]
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} shape {tuple(t.shape)} != {tuple(shape)}")
    if t.dtype is not dtype:
        raise ValueError(f"{name} dtype {t.dtype} != {dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous; got strides {t.stride()}")


def _check_row_input(inp, name: str, stride_name: str, batch: int, width, dtype):
    """``[batch, >= width]`` rows, each contiguous, ``inp[stride_name]`` apart.

    The kernel reads ``width`` elements from ``row * inp[stride_name]``, so the
    row stride must be an integer that matches the tensor's own.
    """
    t, stride = inp[name], inp[stride_name]
    if t.ndim != 2 or t.shape[0] != batch or t.shape[1] < width:
        raise ValueError(
            f"{name} must be [batch={batch}, >= {width}]; got {tuple(t.shape)}"
        )
    if t.dtype is not dtype:
        raise ValueError(f"{name} dtype {t.dtype} != {dtype}")
    if t.stride(1) != 1:
        raise ValueError(f"{name} must have a unit channel stride; got {t.stride()}")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise ValueError(f"{stride_name} must be an int; got {stride!r}")
    if stride != t.stride(0):
        raise ValueError(f"{stride_name} {stride} != {name}.stride(0) {t.stride(0)}")


def prepare(spec: GdnDecodeSpec, inp, batch: int, *, validate_indices: bool = True):
    """Allocate the kernel's outputs and freeze a launch config.

    Split out from :func:`launch` deliberately. Allocating inside a timing loop
    measures the allocator rather than the kernel, and allocating inside a HIP
    graph capture is illegal, so every caller that repeats a launch prepares
    once and then only launches.
    """
    _validate_decode_inputs(spec, inp, batch, validate_indices=validate_indices)
    torch_dtype = _TORCH_DT[spec.dtype]
    out = torch.zeros(
        batch,
        1,
        spec.num_v_heads,
        spec.head_v_dim,
        device=inp[_device_ref(spec)].device,
        dtype=torch_dtype,
    )
    values = dict(inp)
    values["state"] = inp["state"].clone()  # the kernel updates the state in place
    if spec.fuse_conv:
        values["conv_state"] = inp["conv_state"].clone()  # shifted in place too
    values["out"] = out
    values["batch_size"] = batch
    cfg = LaunchConfig(
        grid=gdn_decode_grid(batch, spec),
        block=(spec.block_size, 1, 1),
        stream=0,
    )
    return values, cfg


def launch(launcher: KernelLauncher, values, cfg) -> None:
    """Enqueue one kernel launch. No allocation, no synchronisation."""
    with no_fence():
        launcher(values, config=cfg)


def drain() -> None:
    """Wait for every queued launch and drop the references launches retain.

    :func:`launch` enqueues under ``no_fence()``, so the rocKE runtime keeps each
    launch's packed args and tensor references alive until the stream is
    released. ``torch.cuda.synchronize()`` waits for the GPU but does not release
    them, so a loop that only synchronizes holds every tensor it ever launched: a
    full GDN tile sweep ran the device out of memory that way. Call this once a
    check or measurement is finished, never inside a timed region.
    """
    synchronize_and_release(0)


def run_values(spec: GdnDecodeSpec, inp, launcher: KernelLauncher, batch: int):
    """Prepare, launch once, synchronise. Returns every launch value, including
    the updated ``out``, ``state`` and (``fuse_conv``) ``conv_state``; the
    caller's ``inp`` tensors are left untouched."""
    values, cfg = prepare(spec, inp, batch)
    launch(launcher, values, cfg)
    drain()
    return values


def run(spec: GdnDecodeSpec, inp, launcher: KernelLauncher, batch: int):
    """Prepare, launch once, synchronise. Returns ``(out, state_after)``."""
    values = run_values(spec, inp, launcher, batch)
    return values["out"], values["state"]


def check(
    spec: GdnDecodeSpec, batch: int, seed: int = 0, *, disjoint_writes: bool = True
) -> Tuple[float, float]:
    """Run and compare against the reference. Returns ``(out_err, state_err)``.

    ``state_err`` covers the WHOLE pool, not only the written pages: the pages
    the kernel was not told to touch are compared against their pre-launch
    contents. Comparing written pages alone cannot see a write that landed in
    the wrong slot -- the slot written checks out and the slot damaged is never
    looked at -- and a misplaced write is the failure mode a paged state pool
    invites. For ``fuse_conv`` specs it also covers the conv-state pool, by the
    same written/untouched rule. ``disjoint_writes=False`` updates every
    sequence in place (write slot == read slot); the default pool depth still
    leaves ``batch + 1`` untouched pages.
    """
    inp = make_inputs(spec, batch, seed=seed, disjoint_writes=disjoint_writes)
    ref_out, ref_state = ref_fp32(spec, inp)
    values = run_values(spec, inp, launcher_for(spec), batch)
    out_err = out_error(spec, values["out"], ref_out)
    written = inp["write_indices"].long()
    untouched = torch.ones(
        inp["state"].shape[0], dtype=torch.bool, device=written.device
    )
    untouched[written] = False
    pools = [("state", ref_state)]
    if spec.fuse_conv:
        pools.append(("conv_state", ref_conv_state_after(spec, inp)))
    state_err = 0.0
    for name, want in pools:
        # prepare() cloned the pool, so inp[name] still holds its pre-launch
        # contents: written slots against the reference, the rest unchanged.
        got = values[name].float()
        state_err = max(state_err, (got[written] - want).abs().max().item())
        if untouched.any():
            spill = (got[untouched] - inp[name][untouched].float()).abs().max()
            state_err = max(state_err, spill.item())
    return out_err, state_err


def bench(spec: GdnDecodeSpec, batch: int, reps: int = 200) -> float:
    """Median host-observed launch latency in microseconds."""
    launcher = launcher_for(spec)
    values, cfg = prepare(spec, make_inputs(spec, batch), batch)
    for _ in range(50):
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        launch(launcher, values, cfg)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    drain()
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default="1,16,64", help="comma-separated batch sizes")
    ap.add_argument(
        "--variant",
        choices=("tiled", "simple"),
        default="tiled",
        help="tiled = default warp-tiled path; simple = one-thread-per-row reference",
    )
    ap.add_argument("--bench", action="store_true", help="also report per-launch time")
    ap.add_argument("--no-check", action="store_true", help="skip the correctness gate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2

    spec = GdnDecodeSpec(simple=(args.variant == "simple"))
    ok, why = is_valid_spec(spec, arch=_ARCH)
    if not ok:
        print(f"spec rejected: {why}", file=sys.stderr)
        return 2
    print(f"kernel: {spec.kernel_name()}  block={spec.block_size}")

    worst = 0.0
    for batch in (int(x) for x in args.batches.split(",")):
        grid = gdn_decode_grid(batch, spec)
        line = f"B={batch:<5d} grid={grid[0]:<7d}"
        if not args.no_check:
            out_err, state_err = check(spec, batch, seed=args.seed)
            worst = max(worst, out_err, state_err)
            verdict = "OK" if max(out_err, state_err) <= TOL else "FAIL"
            line += f" out_err={out_err:.3e} state_err={state_err:.3e} {verdict}"
        if args.bench:
            line += f" {bench(spec, batch):8.2f}us"
        print(line)

    if args.no_check:
        return 0
    print(f"worst={worst:.3e} tol={TOL:.1e}")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
