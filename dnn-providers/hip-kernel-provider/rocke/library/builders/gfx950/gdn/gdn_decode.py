#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side driver for the GDN single-token decode kernel.

The kernel emitter in ``kernels/gfx950/gdn_decode.py`` produces device code.
This module is everything on the host needed to actually exercise it: compile a
spec, build inputs, run it, judge the result against an independent reference,
and time it.

Inputs are host numpy arrays in the kernel's storage types (bf16 as its raw
``uint16`` bits, see :mod:`rocke.numeric.storage`); :func:`prepare` checks them
and uploads them into device arrays the driver owns, and :func:`read_back`
copies the kernel's outputs back to the host.

The reference is deliberately *not* a restatement of the kernel. It evaluates
the gated delta rule (:mod:`rocke.numeric.gated_delta`) on whole float64
arrays and knows nothing about the kernel's workgroup mapping, warp tiling or
cross-lane reductions, so agreement is evidence about the algorithm rather than
a tautology.

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
import struct
import sys
import time

import numpy as np
from rocke.core.dtypes import dtype_info
from rocke.helpers.compile import compile_kernel
from rocke.numeric import storage
from rocke.numeric.gated_delta import (
    gated_delta_step,
    gdn_log_decay,
    kda_log_decay,
    sigmoid,
)
from rocke.runtime import synchronize_and_release
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.hip_module import Runtime, get_device_count
from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

from dispatch.gdn.cli_args import check_rotate_mb, parse_batches
from dispatch.gdn.gate_rules import (  # the pass test and the rotation size
    TOL,  # also re-exported: the tests and tools read ``gdn_decode.TOL``
    Gate,
    cold_copy_count,
    touched_bytes_of,
)
from kernels.gfx950.gdn_decode import (
    CONV_WIDTH,
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_grid,
    gdn_decode_pointer_alignment,
    gdn_decode_signature,
    is_valid_spec,
)

_ARCH = "gfx950"
_LAUNCHER_CACHE: dict[tuple, KernelLauncher] = {}
# Smallest normal f32, 2**-126.
_F32_MIN_NORMAL = 2.0**-126
# q/k L2-norm eps (the kernel's NORM_EPS); not the output norm's.
_QK_NORM_EPS = 1e-6
# Inputs that are scalars in the kernel ABI, not arrays.
_SCALARS = ("qkv_stride", "og_stride", "norm_eps")


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


def input_dtypes(spec: GdnDecodeSpec) -> dict[str, str]:
    """The rocKE element type of every array input ``spec``'s kernel reads.

    The kernel receives raw pointers, so this map is the only thing tying an
    input to the element type frozen into the machine code: :func:`make_inputs`
    stores each input in it, :func:`validate_inputs` checks it, the reference
    decodes by it and :func:`prepare` uploads by it. The I/O arrays carry
    ``spec.dtype`` and the state pool ``spec.state_dtype``; ``A_log``, KDA's
    ``dt_bias``, the conv weight and the norm weight are always fp32.
    """
    io = dtype_info(spec.dtype).name
    kda = spec.gate_kind == "kda"
    dtypes = {}
    if spec.fuse_conv:
        # The packed row REPLACES query/key/value in the kernel contract.
        dtypes.update(mixed_qkv=io, conv_state=io, conv_weight="fp32")
    else:
        dtypes.update(query=io, key=io, value=io)
    dtypes.update(
        a=io,
        b=io,
        dt_bias="fp32" if kda else io,
        A_log="fp32",
        read_indices="i32",
        write_indices="i32",
        state=dtype_info(spec.state_dtype).name,
    )
    if spec.fuse_out_norm:
        dtypes.update(out_gate=io, norm_weight="fp32")
    return dtypes


def _gate_values(spec: GdnDecodeSpec, normal, batch: int, hv: int, dk: int):
    """The values of the `a` input, whose meaning depends on the gate mode.

    * GDN: one raw logit per head, `[B, 1, HV]`.
    * KDA, fuse_gate=True: one raw logit per K channel, `[B, 1, HV, DK]`; the
      kernel turns it into a decay.
    * KDA, fuse_gate=False: the natural-log-domain decay itself. Forced
      non-positive so `exp(a)` lands in (0, 1] -- a positive value would
      amplify the state every step rather than fade it, which the fused path
      cannot produce and so is not worth generating.

    Draws exactly one array in every mode, so it sits at one fixed position in
    the draw order and the draws after it do not depend on the gate mode.
    """
    if spec.gate_kind != "kda":
        return normal(batch, 1, hv)
    g = normal(batch, 1, hv, dk, scale=0.5)
    return g if spec.fuse_gate else -np.abs(g)


def make_inputs(
    spec: GdnDecodeSpec,
    batch: int,
    seed: int = 0,
    pool_depth: int | None = None,
    disjoint_writes: bool = True,
) -> dict:
    """Deterministic host inputs matching the kernel's packed decode contract.

    Every array is stored in its :func:`input_dtypes` element type (bf16 as raw
    ``uint16`` bits); ``qkv_stride`` / ``og_stride`` / ``norm_eps`` are the
    fused modes' scalars.

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
    dtypes = input_dtypes(spec)
    depth = 2 * batch + 1 if pool_depth is None else pool_depth
    if depth < (2 * batch if disjoint_writes else batch):
        raise ValueError(
            f"pool_depth {depth} too small for batch {batch} with "
            f"disjoint_writes={disjoint_writes}"
        )
    rng = np.random.default_rng(seed)

    def normal(*shape, scale=1.0):
        # float32 draws: the stored types are at most 32 bits wide, and the
        # pool is the largest array a test makes.
        return np.float32(scale) * rng.standard_normal(shape, dtype=np.float32)

    def stored(name, values):
        return storage.encode(values, dtypes[name])

    hk, hv = spec.num_k_heads, spec.num_v_heads
    dk, dv = spec.head_k_dim, spec.head_v_dim
    io = dtype_info(spec.dtype).name
    state_type = dtype_info(spec.state_dtype).name
    # query/key/value are drawn in every mode, so a fused spec's gate and state
    # inputs equal its unfused twin's, seed for seed.
    inp = {
        "query": storage.encode(normal(batch, 1, hk, dk), io),
        "key": storage.encode(normal(batch, 1, hk, dk), io),
        "value": storage.encode(normal(batch, 1, hv, dv), io),
        "a": stored("a", _gate_values(spec, normal, batch, hv, dk)),
        "b": stored("b", normal(batch, 1, hv)),
        # f32 for KDA: the gate is evaluated in f32 and KDA prefill already
        # declares this array f32, so the two families share one contract.
        "dt_bias": stored(
            "dt_bias",
            normal(hv, dk, scale=0.1) if spec.gate_kind == "kda" else normal(hv),
        ),
        "A_log": stored("A_log", normal(hv)),
        "read_indices": np.arange(batch, dtype=np.int32),
        "write_indices": (
            np.arange(batch, 2 * batch, dtype=np.int32)
            if disjoint_writes
            else np.arange(batch, dtype=np.int32)
        ),
        "state": storage.encode(normal(depth, hv, dv, dk, scale=0.01), state_type),
    }
    # Fused-mode arrays are drawn after every unfused draw, for the same reason.
    if spec.fuse_conv:
        conv_dim = spec.conv_dim
        inp["mixed_qkv"] = stored("mixed_qkv", normal(batch, conv_dim))
        inp["qkv_stride"] = conv_dim
        # The packed row REPLACES query/key/value in the kernel contract, so
        # the fixture does not carry them: the reference derives q/k/v from
        # the packed row, and a caller that follows the contract is tested.
        for name in ("query", "key", "value"):
            del inp[name]
        inp["conv_state"] = stored(
            "conv_state", normal(depth, conv_dim, CONV_WIDTH - 1)
        )
        inp["conv_weight"] = stored(
            "conv_weight", normal(conv_dim, CONV_WIDTH, scale=0.5)
        )
    if spec.fuse_out_norm:
        inp["out_gate"] = stored("out_gate", normal(batch, hv * dv))
        inp["og_stride"] = hv * dv
        inp["norm_weight"] = stored("norm_weight", 1.0 + normal(dv, scale=0.1))
        inp["norm_eps"] = 1e-6
    return inp


def _decoded(spec: GdnDecodeSpec, inp, name: str, rows=slice(None)) -> np.ndarray:
    """The values of float input ``name`` (only ``rows`` of its first axis) as
    the kernel reads them, in float64."""
    stored = inp[name][rows]
    return storage.decode(stored, input_dtypes(spec)[name]).astype(np.float64)


def precompute_kda_log_decay(spec: GdnDecodeSpec, inp) -> np.ndarray:
    """Natural-log decay ``[B, HV, DK]`` (float32) for the benchmark-only
    precomputed KDA mode.

    Written out here rather than calling :func:`rocke.numeric.gated_delta.kda_log_decay`:
    benchmark preparation and the correctness reference must not share formula
    code, or the same bug can make both arms agree.
    """
    if spec.gate_kind != "kda" or not spec.fuse_gate:
        raise ValueError("precompute_kda_log_decay requires fused KDA raw logits")
    a_log = inp["A_log"].astype(np.float32)
    logits = storage.decode(inp["a"][:, 0], dtype_info(spec.dtype).name)
    inner = np.exp(a_log)[None, :, None] * (logits + inp["dt_bias"])
    return (spec.lower_bound / (1.0 + np.exp(-inner))).astype(np.float32)


def _silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def _ref_conv(spec: GdnDecodeSpec, inp) -> tuple[np.ndarray, np.ndarray]:
    """Causal conv1d (width ``CONV_WIDTH``) + SiLU over the packed q/k/v row.

    Returns ``(convolved [B, conv_dim], taps_after [B, conv_dim, CONV_WIDTH-1])``:
    tap 0 is the oldest, and the shift drops it and appends the current token.
    Only the first ``spec.conv_dim`` channels of each row belong to the
    kernel, since a row may be wider.
    """
    conv_dim = spec.conv_dim
    taps = _decoded(spec, inp, "conv_state", inp["read_indices"])  # [B, C, W-1]
    x = _decoded(spec, inp, "mixed_qkv")[:, :conv_dim]  # [B, C]
    w = _decoded(spec, inp, "conv_weight")  # [C, W]
    if taps.shape[1:] != (conv_dim, CONV_WIDTH - 1) or w.shape != (
        conv_dim,
        CONV_WIDTH,
    ):
        raise ValueError(
            f"conv arrays {taps.shape[1:]} / {w.shape} do not match "
            f"spec.conv_dim {conv_dim} and width {CONV_WIDTH}"
        )
    window = np.concatenate((taps, x[..., None]), axis=-1)  # [B, C, W], oldest first
    return _silu((window * w).sum(-1)), window[..., 1:]


def ref_conv_state_after(spec: GdnDecodeSpec, inp) -> np.ndarray:
    """Expected conv-state taps written to each write slot: [B, conv_dim, W-1]."""
    return _ref_conv(spec, inp)[1]


def out_error(spec: GdnDecodeSpec, out: np.ndarray, ref_out: np.ndarray) -> float:
    """Max output error of the stored ``out`` against the reference, comparable
    with ``TOL``; NaN when any output is NaN.

    The plain decode output is small (|o| << 1), so the absolute error is used.
    With ``fuse_out_norm`` the output is RMS-normalised and gated, |o| reaches
    ~5-7 for these inputs, and one bf16 rounding step of the exact reference
    alone exceeds ``TOL`` there; the error is then taken relative to
    ``max(1, |ref|)`` (identical to the absolute error wherever |ref| <= 1, and
    at most |ref| times smaller than it elsewhere).
    """
    got = storage.decode(out, dtype_info(spec.dtype).name).reshape(ref_out.shape)
    diff = np.abs(got - ref_out)
    if spec.fuse_out_norm:
        diff = diff / np.maximum(np.abs(ref_out), 1.0)
    return float(diff.max())


def ref_decode_step(spec: GdnDecodeSpec, inp) -> tuple[np.ndarray, np.ndarray]:
    """Whole-array float64 reference for one decode step.

    Returns ``(out [B, 1, HV, DV], state_after [B, HV, DV, DK])``. One token of
    the gated delta rule (:func:`rocke.numeric.gated_delta.gated_delta_step`)
    with no chunking, tiling or cross-lane structure, so it shares no algebra
    with the kernel beyond the definition itself. Fused specs apply the conv1d
    to q/k/v first (:func:`_ref_conv`) and the gated RMSNorm to ``out`` last.
    """
    hk, hv, g = spec.num_k_heads, spec.num_v_heads, spec.v_per_k_head
    dk, dv = spec.head_k_dim, spec.head_v_dim
    batch = inp["read_indices"].shape[0]
    if spec.fuse_conv:
        y, _ = _ref_conv(spec, inp)
        q = y[:, : hk * dk].reshape(batch, hk, dk)
        k = y[:, hk * dk : 2 * hk * dk].reshape(batch, hk, dk)
        v = y[:, 2 * hk * dk :].reshape(batch, hv, dv)
    else:
        q = _decoded(spec, inp, "query")[:, 0]
        k = _decoded(spec, inp, "key")[:, 0]
        v = _decoded(spec, inp, "value")[:, 0]
    # Each value head reads the key head it is grouped under.
    k_of_v = np.arange(hv) // g
    q, k = q[:, k_of_v], k[:, k_of_v]  # [B, HV, DK]
    scale = 1.0 / math.sqrt(dk)
    if spec.use_qk_l2norm:
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + _QK_NORM_EPS) * scale
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + _QK_NORM_EPS)
    else:
        # l2norm off: the kernel scales q by 1/sqrt(dk) and leaves k raw.
        q = q * scale

    a = _decoded(spec, inp, "a")[:, 0]
    dt_bias, a_log = _decoded(spec, inp, "dt_bias"), _decoded(spec, inp, "A_log")
    if spec.gate_kind == "kda" and spec.fuse_gate:
        # Per-channel gate, bounded in (lower_bound, 0).
        log_decay = kda_log_decay(a, dt_bias, a_log[:, None], spec.lower_bound)
    elif spec.gate_kind == "kda":
        # `a` already carries natural-log-domain decay; only the exp remains.
        # A_log and dt_bias are unread here -- the caller folded them in.
        # This mirrors the competitor recurrence-only kernel, which likewise
        # applies exp(gk) in-kernel and nothing else, which is what makes the
        # two timeable at one work boundary.
        log_decay = a  # [B, HV, DK]
    else:
        # Scalar gate, unbounded below; one decay for the whole head.
        log_decay = gdn_log_decay(a, dt_bias, a_log)[..., None]
    beta = sigmoid(_decoded(spec, inp, "b")[:, 0])  # [B, HV]

    # The kernel's pool holds S transposed ([DV, DK]); the rule is written on
    # S itself ([DK, DV]), whose K axis the decay scales.
    state = _decoded(spec, inp, "state", inp["read_indices"])  # [B, HV, DV, DK]
    out, after = gated_delta_step(np.swapaxes(state, -1, -2), q, k, v, log_decay, beta)
    if spec.fuse_out_norm:
        # Gate activation by gate kind, stated here from the models rather
        # than read from the kernel's table, so a wrong table fails the check:
        # SiLU for GDN (Qwen3-Next), sigmoid for KDA (Kimi Linear);
        # ALGORITHM.md section 1.4.
        act = _silu if spec.gate_kind == "gdn" else sigmoid
        og = _decoded(spec, inp, "out_gate")[:, : hv * dv].reshape(out.shape)
        rstd = 1.0 / np.sqrt((out * out).mean(-1, keepdims=True) + inp["norm_eps"])
        out = out * rstd * _decoded(spec, inp, "norm_weight") * act(og)
    return out[:, None], np.swapaxes(after, -1, -2)


_INPUT_FAILED = "gdn_decode input failed"


def validate_inputs(spec: GdnDecodeSpec, inp, batch: int) -> None:
    """Reject host inputs that would make the decode kernel read or write out
    of bounds, or reinterpret bytes. Host checks only: nothing here needs a
    device.

    The kernel indexes the state pool through ``read_indices``/``write_indices``
    with a raw pointer add and only skips the ``-1`` sentinel; it never
    hardware-bounds the index against the pool depth, so an out-of-range index
    is an out-of-bounds load/store. Every other address comes from SPEC
    constants, never from an array's own shape, and the kernel emits no buffer
    descriptor -- there is no ``num_records`` to clamp an over-reach -- so every
    array must be contiguous and exactly the shape the spec implies (KDA's
    per-channel ``a`` and ``dt_bias`` included: a GDN-sized allocation is too
    short for its vector loads).

    The element types (:func:`input_dtypes`) matter as much as the shapes, and
    no device-side check can see them: the kernel derives the pool's byte
    stride from ``spec.state_dtype``. Between ``bf16`` and ``f16`` (both 16
    bits) a mismatched pool computes every address identically and merely
    decodes the bits under the wrong rule, silently corrupting the recurrence
    it feeds back; a 16-bit pool given to an ``f32`` spec is read and written
    twice as far, past the end of the pool.
    """
    dtypes = input_dtypes(spec)
    for name, element in dtypes.items():
        _check_array(inp, name, element)
    state = inp["state"]
    want_head = (spec.num_v_heads, spec.head_v_dim, spec.head_k_dim)
    if state.ndim != 4 or state.shape[1:] != want_head:
        raise ValueError(
            f"{_INPUT_FAILED} - state.shape={state.shape}, expected "
            f"[pool, HV, DV, DK] with (HV, DV, DK)={want_head}"
        )
    pool_depth = state.shape[0]
    hk, hv = spec.num_k_heads, spec.num_v_heads
    dk, dv = spec.head_k_dim, spec.head_v_dim
    shapes = {
        "query": (batch, 1, hk, dk),
        "key": (batch, 1, hk, dk),
        "value": (batch, 1, hv, dv),
        "a": (batch, 1, hv, dk) if spec.gate_kind == "kda" else (batch, 1, hv),
        "b": (batch, 1, hv),
        "dt_bias": (hv, dk) if spec.gate_kind == "kda" else (hv,),
        "A_log": (hv,),
        "read_indices": (batch,),
        "write_indices": (batch,),
        "state": state.shape,
    }
    if spec.fuse_conv:
        conv_dim = spec.conv_dim
        _check_rows(inp, "mixed_qkv", "qkv_stride", (batch, conv_dim))
        row_align = gdn_decode_pointer_alignment(spec)["mixed_qkv"]
        if inp["qkv_stride"] * inp["mixed_qkv"].itemsize % row_align:
            # Each row is read with 16 B vector loads from row * qkv_stride.
            raise ValueError(
                f"{_INPUT_FAILED} - qkv_stride={inp['qkv_stride']}, expected a "
                f"row pitch that is a multiple of {row_align} B: the kernel's "
                "vector loads assume every mixed_qkv row starts on that boundary"
            )
        shapes.update(
            conv_state=(pool_depth, conv_dim, CONV_WIDTH - 1),
            conv_weight=(conv_dim, CONV_WIDTH),
        )
    if spec.fuse_out_norm:
        _check_rows(inp, "out_gate", "og_stride", (batch, hv * dv))
        shapes["norm_weight"] = (dv,)
        _check_norm_eps(inp["norm_eps"])
    for name in dtypes:
        if name not in ("mixed_qkv", "out_gate"):  # checked as rows above
            _check_shape(inp, name, shapes[name])
    _check_indices(inp, pool_depth)


def _check_array(inp, name: str, element: str) -> None:
    """A C-contiguous host numpy array of ``element``'s storage dtype.

    The kernel is compiled with an ``element`` pointer and gets no dtype tag at
    runtime, so other bytes are reinterpreted, not converted; and it computes
    every address from the spec's dims, so a non-contiguous array would be read
    at the right INDEX from the wrong MEMORY."""
    array = inp[name]
    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"{_INPUT_FAILED} - {name}={type(array).__name__}, expected a host "
            "numpy array (prepare uploads it)"
        )
    want = storage.host_dtype(element)
    if array.dtype != want:
        raise ValueError(
            f"{_INPUT_FAILED} - {name}.dtype={array.dtype}, expected {want} "
            f"({element}): the kernel is compiled with a {element} pointer and "
            "would reinterpret these bytes"
        )
    if not array.flags.c_contiguous:
        raise ValueError(
            f"{_INPUT_FAILED} - {name}.strides={array.strides}, expected a "
            "contiguous (row-major) array"
        )


def _check_shape(inp, name: str, shape: tuple) -> None:
    if inp[name].shape != shape:
        raise ValueError(
            f"{_INPUT_FAILED} - {name}.shape={inp[name].shape}, expected {shape}"
        )


def _check_rows(inp, name: str, stride_name: str, rows: tuple) -> None:
    """``[batch, >= width]`` contiguous rows, ``inp[stride_name]`` elements apart.

    The kernel reads ``width`` elements from ``row * inp[stride_name]``, so the
    row stride must be an integer equal to the array's row length; a row may be
    wider than ``width`` and its tail is never read.
    """
    array, stride = inp[name], inp[stride_name]
    batch, width = rows
    if array.ndim != 2 or array.shape[0] != batch or array.shape[1] < width:
        raise ValueError(
            f"{_INPUT_FAILED} - {name}.shape={array.shape}, expected "
            f"[batch={batch}, >= {width}]"
        )
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise TypeError(f"{_INPUT_FAILED} - {stride_name}={stride!r}, expected an int")
    if stride != array.shape[1]:
        raise ValueError(
            f"{_INPUT_FAILED} - {stride_name}={stride}, expected {name}'s row "
            f"length {array.shape[1]}"
        )


def _check_norm_eps(eps) -> None:
    """The kernarg packs eps as an f32, so judge the value the kernel gets:
    1e-50 rounds to 0.0 and an f32 denormal may be flushed to zero.
    rsqrt(mean(o^2) + eps) with a zero, negative or NaN eps gives an inf/NaN
    output for a small or zero row instead of an error."""
    eps32 = math.nan
    if not isinstance(eps, bool) and isinstance(eps, (int, float)):
        try:
            eps32 = struct.unpack("<f", struct.pack("<f", eps))[0]
        except OverflowError:
            pass
    if not (math.isfinite(eps32) and eps32 >= _F32_MIN_NORMAL):
        raise ValueError(
            f"{_INPUT_FAILED} - norm_eps={eps!r}, expected a finite positive "
            f"float that stays a normal f32 (>= {_F32_MIN_NORMAL!r})"
        )


def _check_indices(inp, pool_depth: int) -> None:
    """-1 is the 'skip this slot' sentinel; every other index must land in
    ``[0, pool_depth)``, and no two active sequences may write one slot."""
    for name in ("read_indices", "write_indices"):
        idx = inp[name]
        lo, hi = int(idx.min()), int(idx.max())
        if lo < -1 or hi >= pool_depth:
            raise ValueError(
                f"{_INPUT_FAILED} - {name} spans [{lo}, {hi}], expected -1 (skip) "
                f"or a slot in [0, {pool_depth})"
            )
    active = (inp["read_indices"] >= 0) & (inp["write_indices"] >= 0)
    slots, counts = np.unique(inp["write_indices"][active], return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"{_INPUT_FAILED} - write_indices repeat {slots[counts > 1].tolist()}, "
            "expected unique slots across active sequences"
        )


def prepare(spec: GdnDecodeSpec, inp, batch: int):
    """Validate ``inp``, upload it into fresh device arrays, allocate the
    output, and freeze a launch config. Returns ``(values, config)``.

    Split out from :func:`launch` deliberately. Allocating inside a timing loop
    measures the allocator rather than the kernel, and allocating inside a HIP
    graph capture is illegal, so every caller that repeats a launch prepares
    once and then only launches. The upload is a copy: the kernel updates the
    state pools in place, and ``inp`` keeps their pre-launch contents.
    """
    validate_inputs(spec, inp, batch)
    return _upload(spec, inp, batch), launch_config(spec, batch)


def _upload(spec: GdnDecodeSpec, inp, batch: int) -> dict:
    """Device copies of ``inp``'s arrays plus a zeroed ``out``; scalars pass
    through. Every buffer is a fresh allocation (``hipMalloc`` aligns it to at
    least 256 B); the alignment check below holds the kernel's documented
    assumption (``gdn_decode_pointer_alignment``) on what was allocated."""
    dtypes = input_dtypes(spec)
    values = {name: DeviceArray(inp[name], dtype) for name, dtype in dtypes.items()}
    values.update({name: inp[name] for name in _SCALARS if name in inp})
    values["out"] = DeviceArray.zeros(
        (batch, 1, spec.num_v_heads, spec.head_v_dim), dtype_info(spec.dtype).name
    )
    values["batch_size"] = batch
    for name, align in gdn_decode_pointer_alignment(spec).items():
        if values[name].data_ptr() % align:
            raise RuntimeError(
                f"device allocation for {name} is not {align} B aligned "
                f"({values[name].data_ptr():#x}); the kernel's vector accesses "
                "assume it"
            )
    return values


def _outputs(spec: GdnDecodeSpec) -> tuple[str, ...]:
    """The arrays one launch writes."""
    return ("out", "state", "conv_state") if spec.fuse_conv else ("out", "state")


def read_back(spec: GdnDecodeSpec, values) -> dict[str, np.ndarray]:
    """The arrays a launch wrote (``out``, ``state`` and, for ``fuse_conv``,
    ``conv_state``), copied from ``values``' device arrays to the host in their
    storage types. The copies are blocking, so queued launches have finished."""
    return {name: values[name].values() for name in _outputs(spec)}


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
    launch's packed args and device arrays alive until the stream is released.
    A loop that only waits for the GPU holds every array it ever launched: a
    full GDN tile sweep ran the device out of memory that way. Call this once a
    check or measurement is finished, never inside a timed region.
    """
    synchronize_and_release(0)


def run_values(spec: GdnDecodeSpec, inp, launcher: KernelLauncher, batch: int):
    """Prepare, launch once, synchronise, and :func:`read_back` the arrays the
    launch wrote. ``inp`` is left untouched."""
    values, cfg = prepare(spec, inp, batch)
    launch(launcher, values, cfg)
    drain()
    return read_back(spec, values)


def run(spec: GdnDecodeSpec, inp, launcher: KernelLauncher, batch: int):
    """Prepare, launch once, synchronise. Returns host ``(out, state_after)``."""
    values = run_values(spec, inp, launcher, batch)
    return values["out"], values["state"]


def check(
    spec: GdnDecodeSpec, batch: int, seed: int = 0, *, disjoint_writes: bool = True
) -> Gate:
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
    """``(out, state_after, conv_state_after or None)``: the float64 reference
    for ``inp``. It depends only on the problem, never on the tile or knobs, so
    one reference scores every configuration of a request."""
    ref_out, ref_state = ref_decode_step(spec, inp)
    ref_conv = ref_conv_state_after(spec, inp) if spec.fuse_conv else None
    return ref_out, ref_state, ref_conv


def _same_bits(got: np.ndarray, want: np.ndarray) -> bool:
    """Bitwise equality (a NaN equals the same NaN; -0.0 differs from 0.0)."""
    return got.shape == want.shape and np.array_equal(
        np.ascontiguousarray(got).view(np.uint8),
        np.ascontiguousarray(want).view(np.uint8),
    )


def gate(spec: GdnDecodeSpec, inp, values, ref) -> Gate:
    """Score one launch's host ``values`` (:func:`run_values` /
    :func:`read_back` on ``inp``) against ``ref`` (:func:`reference`): the
    output, the written state and conv-state slots, and every slot the kernel
    was not told to write. Blind spots: with read slot == write slot the gate
    scores the result by value, but cannot tell a kernel that reads its state
    from the write slot from a correct one (both compute the same answer), so
    the sweep gates disjoint slots only; and a ``-1`` skip lane is not modelled.

    The untouched slots are compared bit for bit with ``inp`` (the launch ran
    on device copies, so ``inp`` holds their pre-launch contents): the kernel
    must not touch them at all, so a tolerance would hide a small but real
    misplaced write. ``inp`` therefore needs spare pool slots
    (:func:`make_inputs` gives ``2 * batch + 1``).
    """
    ref_out, ref_state, ref_conv = ref
    dtypes = input_dtypes(spec)
    written = inp["write_indices"]
    untouched = np.ones(inp["state"].shape[0], dtype=bool)
    untouched[written] = False

    def written_error(pool, want):
        got = storage.decode(values[pool][written], dtypes[pool])
        return float(np.abs(got - want).max())

    out_err = out_error(spec, values["out"], ref_out)
    state_err = written_error("state", ref_state)
    spill = not _same_bits(values["state"][untouched], inp["state"][untouched])
    conv_err = 0.0
    if spec.fuse_conv:
        conv_err = written_error("conv_state", ref_conv)
        before = inp["conv_state"][untouched]
        spill = spill or not _same_bits(values["conv_state"][untouched], before)
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


def touched_bytes(spec: GdnDecodeSpec, inp, batch: int) -> int:
    """Distinct bytes one launch reads or writes, counted conservatively low.

    Only the state (and conv-state) slots named by ``read_indices`` /
    ``write_indices`` and the output are counted; the pools are wider than the
    batch, and the small q/k/v/gate inputs are left out on purpose, so the copy
    count derived from this errs towards colder, never warmer.
    """
    slots = np.unique(np.concatenate([inp["read_indices"], inp["write_indices"]]))
    io_bytes = storage.host_dtype(dtype_info(spec.dtype).name).itemsize
    return touched_bytes_of(
        slots.size,
        inp["state"][0].nbytes,
        batch * spec.num_v_heads * spec.head_v_dim * io_bytes,
        inp["conv_state"][0].nbytes if spec.fuse_conv else None,
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
    """Independent launch value sets to rotate through, each its own device
    copy of every input array and its own output. Every configuration of one
    problem takes the same arguments, so one list serves all of them;
    :func:`launch_config` gives each its geometry. ``inp`` is validated once,
    before anything is allocated."""
    validate_inputs(spec, inp, batch)
    copies = cold_copies(spec, inp, batch, rotate_mb)
    return [_upload(spec, inp, batch) for _ in range(copies)]


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
        torch = _graph_torch()
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
    """HIP-graph capture failed or is unavailable; the message says why."""


def import_graph_torch() -> bool:
    """Import torch, when it is installed, for HIP-graph timing; True if it is.

    torch captures and replays the graphs (rocKE's runtime has no graph capture
    of its own yet), and capture only sees launches on torch's own HIP runtime.
    rocKE binds the HIP runtime of a torch that is already imported, so a
    process that times with graphs calls this before its first HIP call. Eager
    timing, the gate and everything else here need no torch.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _graph_torch():
    """The torch :func:`import_graph_torch` imported, for capture and replay."""
    torch = sys.modules.get("torch")
    if torch is None:
        raise CaptureUnavailable(
            "HIP-graph capture needs torch, imported before the first HIP call "
            "(import_graph_torch)"
        )
    return torch


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

    Raises :class:`CaptureUnavailable` when torch (which captures the graph)
    was not imported (:func:`import_graph_torch`), or carrying the capture's exception text if capture fails (it
    is environment-sensitive, but a launch that is illegal under capture fails
    the same way, so the cause is kept). A failed capture
    leaves the stream invalidated, so the failure path drains before raising
    rather than letting the next caller inherit it.
    """
    if any(v["batch_size"] != batch for v in value_sets):
        raise ValueError(
            "value_sets were prepared for a different batch than the launch grid"
        )
    torch = _graph_torch()
    launcher = launcher or launcher_for(spec)
    cfg = launch_config(spec, batch)
    k = len(value_sets)
    length = k * math.ceil(chain / k)
    for values in value_sets:
        launch(launcher, values, cfg)
    drain()
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
    drain()
    runtime = Runtime()
    samples = []
    for i in range(reps):
        start = time.perf_counter_ns()
        launch(launcher, sets[i % len(sets)], cfg)
        runtime.stream_sync(0)
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
    if args.bench:
        import_graph_torch()  # before the first HIP call

    try:
        batches = parse_batches(args.batches)
    except argparse.ArgumentTypeError as exc:
        ap.error(f"--batches: {exc}")

    if not get_device_count():
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
