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

from dispatch.gdn.cli_args import check_rotate_mb, parse_batches
from dispatch.gdn.gate_rules import (  # the fp32 pass test and the rotation size
    TOL,  # noqa: F401  (re-exported: the tests and tools read ``gdn_decode.TOL``)
    Gate,
    cold_copy_count,
    touched_bytes_of,
)
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
    return values, launch_config(spec, batch)


def launch_config(spec: GdnDecodeSpec, batch: int) -> LaunchConfig:
    """The launch geometry of ``spec`` at ``batch``: the only part of a launch
    that differs between configurations of one problem (they take the same
    arguments)."""
    return LaunchConfig(
        grid=gdn_decode_grid(batch, spec),
        block=(spec.block_size, 1, 1),
        stream=0,
    )


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
) -> "Gate":
    """Run ``spec`` once on :func:`make_inputs` and score it with :func:`gate`.

    The verdict covers the WHOLE pool, not only the written pages: the pages
    the kernel was not told to touch must keep their pre-launch contents bit
    for bit. Comparing written pages alone cannot see a write that landed in
    the wrong slot -- the slot written checks out and the slot damaged is never
    looked at -- and a misplaced write is the failure mode a paged state pool
    invites. For ``fuse_conv`` specs it also covers the conv-state pool, by the
    same written/untouched rule. ``disjoint_writes=False`` updates every
    sequence in place (write slot == read slot); the default pool depth still
    leaves ``batch + 1`` untouched pages. ``check(...).passed`` is the pass
    test.
    """
    inp = make_inputs(spec, batch, seed=seed, disjoint_writes=disjoint_writes)
    return gate_launch(spec, inp, batch)


def reference(spec: GdnDecodeSpec, inp):
    """``(out, state_after, conv_state_after or None)``: the fp32 reference for
    ``inp``. It depends only on the problem, never on the tile or knobs, so one
    reference scores every configuration of a request."""
    ref_out, ref_state = ref_fp32(spec, inp)
    ref_conv = ref_conv_state_after(spec, inp) if spec.fuse_conv else None
    return ref_out, ref_state, ref_conv


def gate(spec: GdnDecodeSpec, inp, values, ref) -> Gate:
    """Score one launch's ``values`` (from :func:`run_values` on ``inp``)
    against ``ref`` (:func:`reference`): the output, the written state and
    conv-state slots, and every slot the kernel was not told to write. Blind
    spots: with read slot == write slot the gate scores the result by value, but
    cannot tell a kernel that reads its state from the write slot from a
    correct one (both compute the same answer), so the sweep gates disjoint
    slots only; and a ``-1`` skip lane is not modelled.

    The untouched slots are compared bit for bit with ``inp`` (``prepare``
    cloned the pools, so ``inp`` holds their pre-launch contents): the kernel
    must not touch them at all, so a tolerance would hide a small but real
    misplaced write. ``inp`` therefore needs spare pool slots
    (:func:`make_inputs` gives ``2 * batch + 1``).
    """
    ref_out, ref_state, ref_conv = ref
    written = inp["write_indices"].long()
    untouched = torch.ones(
        inp["state"].shape[0], dtype=torch.bool, device=written.device
    )
    untouched[written] = False
    out_err = out_error(spec, values["out"], ref_out)
    state_err = (values["state"].float()[written] - ref_state).abs().max().item()
    spill = bool((values["state"][untouched] != inp["state"][untouched]).any())
    conv_err = 0.0
    if spec.fuse_conv:
        conv = values["conv_state"]
        conv_err = (conv.float()[written] - ref_conv).abs().max().item()
        before = inp["conv_state"]
        spill = spill or bool((conv[untouched] != before[untouched]).any())
    return Gate(out_err, state_err, conv_err, spill)


def gate_launch(
    spec: GdnDecodeSpec, inp, batch: int, launcher: KernelLauncher | None = None
) -> Gate:
    """Launch ``spec`` once on ``inp`` (:func:`run_values`; ``inp`` is left
    untouched) and :func:`gate` it against :func:`reference`."""
    launcher = launcher or launcher_for(spec)
    values = run_values(spec, inp, launcher, batch)
    return gate(spec, inp, values, reference(spec, inp))


# Cold-memory timing. The aim is to time a launch whose state is not already in
# the caches, as when other layers ran since this one last did. Re-launching on
# one buffer set can leave the state cache-resident. The timing
# helpers below therefore cycle through ``k`` independent copies of the inputs
# and state (:func:`cold_copies`). ``rotate_mb`` has one meaning: one rotation
# cycle (all ``k`` copies) touches at least ``rotate_mb`` MiB. A copy is reused
# after the other ``k - 1`` copies ran, so at least ``rotate_mb`` MiB less one
# copy's bytes of other traffic. ``rotate_mb=0`` keeps one buffer set, for warm
# debugging only.
# The default is meant to exceed the caches; that is not measured here. A
# smaller value may leave the state cache-resident, so its times are not
# comparable with cold ones.
ROTATE_MB = 1024
# Launches per captured graph (rounded up to whole cycles of buffer sets).
GRAPH_CHAIN = 64
_POOLS = ("state", "conv_state")


def touched_bytes(spec: GdnDecodeSpec, inp, batch: int) -> int:
    """Distinct bytes one launch reads or writes, counted conservatively low.

    Only the state (and conv-state) slots named by ``read_indices`` /
    ``write_indices`` and the output are counted; the pools are wider than the
    batch, and the small q/k/v/gate inputs are left out on purpose, so the copy
    count derived from this errs towards colder, never warmer.
    """
    slots = torch.unique(
        torch.cat([inp["read_indices"].long(), inp["write_indices"].long()])
    ).numel()
    state = inp["state"]
    conv = inp["conv_state"] if spec.fuse_conv else None
    io_bytes = torch.empty((), dtype=_TORCH_DT[spec.dtype]).element_size()
    return touched_bytes_of(
        slots,
        state[0].numel() * state.element_size(),
        batch * spec.num_v_heads * spec.head_v_dim * io_bytes,
        None if conv is None else conv[0].numel() * conv.element_size(),
    )


def cold_copies(
    spec: GdnDecodeSpec, inp, batch: int, rotate_mb: int = ROTATE_MB
) -> int:
    """Buffer sets ``k`` to rotate through: 1 when warm (``rotate_mb == 0``),
    else at least 2 and enough that one cycle of all ``k`` touches at least
    ``rotate_mb`` MiB. This sizes the cycle, not the reuse distance: a set is
    reused after the other ``k - 1``, which is at least ``rotate_mb`` MiB less
    one set's :func:`touched_bytes`."""
    return cold_copy_count(rotate_mb, lambda: touched_bytes(spec, inp, batch))


def cold_value_sets(spec: GdnDecodeSpec, inp, batch: int, rotate_mb: int = ROTATE_MB):
    """Independent launch value sets to rotate through, each with its own copy
    of every input tensor (``prepare`` clones the state pools). Every
    configuration of one problem takes the same arguments, so one list serves
    all of them; :func:`launch_config` gives each its geometry. Only the first
    copy is validated (the copies are identical), so its validation covers the
    rest."""
    sets = []
    for _ in range(cold_copies(spec, inp, batch, rotate_mb)):
        copy = {
            k: v.clone() if torch.is_tensor(v) and k not in _POOLS else v
            for k, v in inp.items()
        }
        # The copies are identical, so validating the first one covers all.
        sets.append(prepare(spec, copy, batch, validate_indices=not sets)[0])
    return sets


class ColdGraph:
    """A captured HIP graph of ``length`` launches that cycles through
    ``copies`` value sets (:func:`capture_cold`). Each :meth:`replay_us` is one
    timed replay; :meth:`close` frees the graph (the caller owns the sets)."""

    def __init__(self, graph, length: int, copies: int):
        self.graph = graph
        self.length = length
        self.copies = copies

    def replay_us(self) -> float:
        """Per-launch device time of one replay, in microseconds."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / self.length

    def close(self) -> None:
        self.graph.reset()
        self.graph = None


class CaptureUnavailable(RuntimeError):
    """HIP-graph capture failed; the message is the capture's own exception."""


def capture_cold(
    spec: GdnDecodeSpec,
    batch: int,
    value_sets,
    launcher: KernelLauncher | None = None,
    *,
    chain: int = GRAPH_CHAIN,
) -> ColdGraph:
    """Capture ``spec`` cycling through ``value_sets`` (:func:`cold_value_sets`)
    as one graph of at least ``chain`` launches, warmed up.

    Raises :class:`ValueError` when a value set was prepared for a batch other
    than ``batch`` (the grid comes from ``batch`` and the kernel has no batch
    bound, so a mismatch would read and write past the tensors). The check
    trusts each value set's own ``batch_size`` label, not its tensor extents:
    a set whose label disagrees with its tensors is not caught.

    Raises :class:`CaptureUnavailable` carrying the capture's exception text if
    capture fails (it is environment-sensitive, but a launch that is illegal
    under capture fails the same way, so the cause is kept). A failed capture
    leaves the stream invalidated, so the failure path drains before raising
    rather than letting the next caller inherit it.
    """
    if any(v["batch_size"] != batch for v in value_sets):
        raise ValueError(
            "value_sets were prepared for a different batch than the launch grid"
        )
    launcher = launcher or launcher_for(spec)
    cfg = launch_config(spec, batch)
    k = len(value_sets)
    length = k * math.ceil(chain / k)
    for values in value_sets:
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(length):
                launch(launcher, value_sets[i % k], cfg)
    except Exception as exc:
        drain()
        raise CaptureUnavailable(f"{type(exc).__name__}: {exc}") from exc
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return ColdGraph(graph, length, k)


def device_us(
    spec: GdnDecodeSpec,
    inp,
    batch: int,
    launcher: KernelLauncher | None = None,
    *,
    rotate_mb: int = ROTATE_MB,
    rounds: int = 30,
    chain: int = GRAPH_CHAIN,
):
    """Median per-launch device time over ``rounds`` replays of a
    :func:`capture_cold` graph on fresh :func:`cold_value_sets` of ``inp``,
    and the number of value sets it cycles.

    Returns ``(micros, copies)``, or ``None`` if graph capture is unavailable
    (the reason is printed to stderr). The median, not the minimum: the
    minimum rewards one lucky replay.
    """
    sets = cold_value_sets(spec, inp, batch, rotate_mb)
    try:
        cold = capture_cold(spec, batch, sets, launcher, chain=chain)
    except CaptureUnavailable as exc:
        print(f"graph capture unavailable: {exc}", file=sys.stderr)
        return None
    try:
        samples = [cold.replay_us() for _ in range(rounds)]
    finally:
        cold.close()
        del sets
        drain()
    return statistics.median(samples), cold.copies


def eager_us(
    spec: GdnDecodeSpec,
    inp,
    batch: int,
    launcher: KernelLauncher | None = None,
    *,
    rotate_mb: int = ROTATE_MB,
    reps: int = 200,
) -> float:
    """Median host-observed latency of one launch plus synchronisation, in
    microseconds, cycling through :func:`cold_value_sets` like
    :func:`device_us`: the latency a synchronous Python decode loop sees."""
    launcher = launcher or launcher_for(spec)
    cfg = launch_config(spec, batch)
    sets = cold_value_sets(spec, inp, batch, rotate_mb)
    for values in sets:
        launch(launcher, values, cfg)
    torch.cuda.synchronize()
    samples = []
    for i in range(reps):
        start = time.perf_counter_ns()
        launch(launcher, sets[i % len(sets)], cfg)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    del sets
    drain()
    return statistics.median(samples)


def timing_label(rotate_mb: int) -> str:
    """One line saying how the reported times were measured; a warning when
    they are warm, or only partly cold (a rotation cycle below the cold target
    :data:`ROTATE_MB` can still fit in the caches)."""
    check_rotate_mb(rotate_mb)
    if rotate_mb == 0:
        return (
            "WARNING: timing is WARM (one buffer set, --rotate-mb 0): the state "
            "may stay in the caches, so these times are for debugging only and "
            "are not comparable with cold ones"
        )
    if rotate_mb < ROTATE_MB:
        return (
            f"WARNING: timing is partly warm: rotation {rotate_mb} MiB < "
            f"{ROTATE_MB} MiB cold target, so part of the state may stay in the "
            "caches; these times are for debugging only and are not comparable "
            "with cold ones"
        )
    return (
        f"timing: cold memory (buffer sets rotated so one cycle touches >= "
        f"{rotate_mb} MiB; a set is reused only after >= {rotate_mb} MiB less "
        "one set's bytes of other traffic; meant to exceed the caches, not "
        "measured here), median over replays"
    )


def bench(spec: GdnDecodeSpec, batch: int, rotate_mb: int = ROTATE_MB) -> float | None:
    """Median per-launch device time in microseconds, cold memory by default
    (see :func:`device_us`); ``None`` if graph capture is unavailable."""
    timed = device_us(spec, make_inputs(spec, batch), batch, rotate_mb=rotate_mb)
    return None if timed is None else timed[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default="1,16,64", help="comma-separated batch sizes")
    ap.add_argument(
        "--variant",
        choices=("tiled", "simple"),
        default="tiled",
        help="tiled = default warp-tiled path; simple = one-thread-per-row reference",
    )
    ap.add_argument(
        "--bench", action="store_true", help="also report cold per-launch device time"
    )
    ap.add_argument(
        "--rotate-mb",
        type=int,
        default=ROTATE_MB,
        help="MiB one cold-memory rotation cycle touches at least (see "
        "cold_copies); 0 = warm, debugging only",
    )
    ap.add_argument("--no-check", action="store_true", help="skip the correctness gate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    check_rotate_mb(args.rotate_mb, ap)

    try:
        batches = parse_batches(args.batches)
    except argparse.ArgumentTypeError as exc:
        ap.error(f"--batches: {exc}")

    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2

    spec = GdnDecodeSpec(simple=(args.variant == "simple"))
    ok, why = is_valid_spec(spec, arch=_ARCH)
    if not ok:
        print(f"spec rejected: {why}", file=sys.stderr)
        return 2
    print(f"kernel: {spec.kernel_name()}  block={spec.block_size}")
    if args.bench:
        print(timing_label(args.rotate_mb))

    failed = 0
    for batch in batches:
        grid = gdn_decode_grid(batch, spec)
        line = f"B={batch:<5d} grid={grid[0]:<7d}"
        if not args.no_check:
            verdict = check(spec, batch, seed=args.seed)
            line += (
                f" out_err={verdict.out_err:.3e} state_err={verdict.state_err:.3e}"
                f" conv_err={verdict.conv_err:.3e} spill={verdict.spill} "
                + ("OK" if verdict.passed else "FAIL")
            )
            if not verdict.passed:
                failed += 1
                print(line)  # a failing kernel is never timed
                continue
        if args.bench:
            micros = bench(spec, batch, rotate_mb=args.rotate_mb)
            line += f" {micros:8.2f}us" if micros is not None else " n/a"
        print(line)

    if args.no_check:
        return 0
    print(f"{failed} batches failed the fp32 gate (tol={TOL:.1e})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
