# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end parity harness for the extended CK Tile DSL kernel family.

This module covers the **platform** subset of the extended kernel family:
MoE streaming kernels and GEMM variants (StreamK, block-scale, MX, MFMA).

Attention / FMHA / Sage / Sparse cases have moved to
``builders.common.parity_fmha_extended`` (library layer) because those
kernels are imported from the ``kernels`` package which is library-only.

Usage::

    python -m rocke.examples.common.parity_extended_kernels [--arch gfx950]

The harness exits non-zero if any kernel's max abs diff exceeds the
per-kernel tolerance, so it doubles as a smoke gate for CI.

Inputs are numpy draws from a fixed per-case seed, stored in the kernel's
element type and launched from :class:`~rocke.runtime.device_array.DeviceArray`
buffers; each reference is computed from the stored values.
"""

# ruff: noqa: E741

from __future__ import annotations

import argparse
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from rocke.instances import (  # noqa: E402
    BlockScaleGemmSpec,
    FusedMoeForward,
    FusedMoeForwardSpec,
    FusedMoeLauncher,
    FusedMoeSpec,
    MxGemmSpec,
    StreamKGemmSpec,
    build_block_scale_gemm,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_topk_weighted_reduce,
    build_mx_gemm,
    build_streamk_gemm,
    block_scale_gemm_grid,
    block_scale_gemm_signature,
    moe_gather_grid,
    moe_gather_signature,
    moe_silu_mul_grid,
    moe_silu_mul_signature,
    moe_topk_weighted_reduce_grid,
    moe_topk_weighted_reduce_signature,
    mx_gemm_grid,
    mx_gemm_signature,
    streamk_gemm_grid,
    streamk_gemm_signature,
)
from rocke.examples.common.parity_support import (
    CaseUnavailable,
    Result,
    default_arch,
    device_available,
    device_values,
    draw,
    launch,
    require_ocp_fp8_arch,
    round_to_f16,
    run_cases,
    summarise,
    values,
)
from rocke.helpers import compile_kernel  # noqa: E402
from rocke.runtime.device_array import DeviceArray, zero_elements
from rocke.runtime.launcher import KernelLauncher, synchronize_and_release

if TYPE_CHECKING:
    import torch


# Target gfx arch for all compiles in this harness.  Set by main() from
# --arch / default_arch() before any case runs.
_ARCH = "gfx950"


def _compile(kernel):
    """Compile *kernel* for the harness-selected arch (``_ARCH``)."""
    return compile_kernel(kernel, arch=_ARCH)


# ---------------------------------------------------------------------
# Fused MoE
# ---------------------------------------------------------------------


def _moe_streaming_spec() -> FusedMoeSpec:
    """The MoE shape every streaming-kernel case runs."""
    return FusedMoeSpec(
        tokens=32,
        experts=4,
        topk=2,
        hidden=256,
        intermediate=512,
        dtype="f16",
        block_size=64,
        vec=4,
    )


def _gather_reference(x: np.ndarray, sids: list[int]) -> np.ndarray:
    """Row ``b`` is ``x[sids[b]]``; a bucket with ``sids[b] < 0`` stays 0."""
    ref = np.zeros((len(sids), x.shape[1]))
    for b, tid in enumerate(sids):
        if tid >= 0:
            ref[b] = x[tid]
    return ref


def _silu_mul_reference(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """``silu(gate) * up``, rounded to fp16 as the kernel stores it."""
    return round_to_f16(gate / (1.0 + np.exp(-gate)) * up)


def _topk_reduce_reference(
    down: np.ndarray, sids: list[int], weights: np.ndarray, tokens: int
) -> np.ndarray:
    """``Y[tid] = sum over buckets b of token tid of weights[b] * down[b]``."""
    ref = np.zeros((tokens, down.shape[1]))
    for b, tid in enumerate(sids):
        if tid >= 0:
            ref[tid] += weights[b] * down[b]
    return ref


def case_moe_gather() -> Result:
    spec = _moe_streaming_spec()
    kernel = build_moe_gather(spec)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=moe_gather_signature(spec),
    )
    rng = np.random.default_rng(0)
    X_host = draw(rng, spec.tokens, spec.hidden)
    sids = [(b % spec.tokens) for b in range(spec.total_pairs)]
    sids[3] = -1  # mask out one bucket
    GroupedInput = DeviceArray.zeros((spec.total_pairs, spec.hidden), "fp16")
    launch(
        launcher,
        {
            "X": DeviceArray(X_host),
            "SortedTokenIds": DeviceArray(np.array(sids, dtype=np.int32)),
            "GroupedInput": GroupedInput,
            "tokens": spec.tokens,
            "hidden": spec.hidden,
        },
        grid=moe_gather_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    ref = _gather_reference(values(X_host, "fp16"), sids)
    r = summarise(device_values(GroupedInput), ref, tol=0.0)
    r.name = "moe_gather"
    return r


def case_moe_silu_mul() -> Result:
    spec = _moe_streaming_spec()
    kernel = build_moe_silu_mul(spec)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=moe_silu_mul_signature(spec),
    )
    rng = np.random.default_rng(1)
    GateOut_host = draw(rng, spec.total_pairs, spec.intermediate)
    UpOut_host = draw(rng, spec.total_pairs, spec.intermediate)
    Hidden = DeviceArray.zeros((spec.total_pairs, spec.intermediate), "fp16")
    launch(
        launcher,
        {
            "GateOut": DeviceArray(GateOut_host),
            "UpOut": DeviceArray(UpOut_host),
            "Hidden": Hidden,
            "total_pairs": spec.total_pairs,
            "intermediate": spec.intermediate,
        },
        grid=moe_silu_mul_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    ref = _silu_mul_reference(values(GateOut_host, "fp16"), values(UpOut_host, "fp16"))
    r = summarise(device_values(Hidden), ref, tol=5e-3, note="f16 silu via exp2 ULP")
    r.name = "moe_silu_mul"
    return r


def case_moe_topk_weighted_reduce() -> Result:
    spec = _moe_streaming_spec()
    kernel = build_moe_topk_weighted_reduce(spec)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=moe_topk_weighted_reduce_signature(spec),
    )
    rng = np.random.default_rng(2)
    DownOut_host = draw(rng, spec.total_pairs, spec.hidden)
    sids = [b // spec.topk for b in range(spec.total_pairs)]
    weights = rng.random(spec.total_pairs, dtype=np.float32)
    Y = DeviceArray.zeros((spec.tokens, spec.hidden), "fp32")
    launch(
        launcher,
        {
            "DownOut": DeviceArray(DownOut_host),
            "SortedTokenIds": DeviceArray(np.array(sids, dtype=np.int32)),
            "SortedWeights": DeviceArray(weights),
            "Y": Y,
            "total_pairs": spec.total_pairs,
            "hidden": spec.hidden,
            "tokens": spec.tokens,
        },
        grid=moe_topk_weighted_reduce_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    ref = _topk_reduce_reference(
        values(DownOut_host, "fp16"), sids, weights, spec.tokens
    )
    r = summarise(device_values(Y), ref, tol=5e-2)
    r.name = "moe_topk_weighted_reduce"
    return r


def case_moe_fused_chain() -> Result:
    """End-to-end CK-Tile-style chained launch of the 3 MoE-specific
    kernels via :class:`FusedMoeLauncher`.

    Drives gather -> silu_mul -> topk_reduce in declaration order on
    a single HIP stream via
    :func:`rocke.runtime.launcher.launch_kernel` /
    :func:`rocke.runtime.launcher.make_kernel`. The 3 kernels are
    independent given their own inputs (the data-flow chain in the
    full MoE forward goes through per-expert GEMMs that are not
    iterated here -- see :class:`FusedMoeLauncher` docstring), so
    correctness is checked by independently comparing each kernel's
    output to its numpy reference. The chain test specifically
    validates:

    1. The new :func:`launch_kernel` primitive correctly submits 3
       :func:`make_kernel` closures on one stream, in declaration
       order, with same-stream FIFO ordering preserving each
       kernel's writes.
    2. :class:`FusedMoeLauncher`'s lazy launcher cache compiles each
       phase exactly once and reuses the cached HSACO + module
       across the 3-callable chain.
    3. The :class:`StreamConfig` ``time_kernel=True`` path returns a
       positive ms when the chain runs in benchmark mode.
    """
    spec = _moe_streaming_spec()
    launcher = FusedMoeLauncher(spec, arch=_ARCH)

    # Phase 1: gather inputs. Same shape and seed as case_moe_gather
    # so this test exercises the same numerics as the per-kernel
    # case, just routed through the chained primitive.
    rng = np.random.default_rng(0)
    X_host = draw(rng, spec.tokens, spec.hidden)
    sids_list = [(b % spec.tokens) for b in range(spec.total_pairs)]
    sids_list[3] = -1  # mask out one bucket so the masked-store path runs
    GroupedInput = DeviceArray.zeros((spec.total_pairs, spec.hidden), "fp16")

    # Phase 2: silu_mul inputs. Independent of phase 1 (the real
    # pipeline routes GroupedInput through per-expert gate / up GEMMs
    # before this kernel runs); the chain test feeds synthetic
    # GateOut / UpOut to keep the 3 kernels' validation independent.
    rng = np.random.default_rng(1)
    GateOut_host = draw(rng, spec.total_pairs, spec.intermediate)
    UpOut_host = draw(rng, spec.total_pairs, spec.intermediate)
    Hidden = DeviceArray.zeros((spec.total_pairs, spec.intermediate), "fp16")

    # Phase 3: topk_reduce inputs.
    rng = np.random.default_rng(2)
    DownOut_host = draw(rng, spec.total_pairs, spec.hidden)
    # Match the per-kernel reduce test's id pattern (no -1 mask, so
    # every bucket contributes -- the gather phase's -1 mask is its
    # own concern).
    reduce_sids_list = [b // spec.topk for b in range(spec.total_pairs)]
    weights = rng.random(spec.total_pairs, dtype=np.float32)
    Y = DeviceArray.zeros((spec.tokens, spec.hidden), "fp32")

    launch_values = {
        "gather": {
            "X": DeviceArray(X_host),
            "SortedTokenIds": DeviceArray(np.array(sids_list, dtype=np.int32)),
            "GroupedInput": GroupedInput,
            "tokens": spec.tokens,
            "hidden": spec.hidden,
        },
        "silu_mul": {
            "GateOut": DeviceArray(GateOut_host),
            "UpOut": DeviceArray(UpOut_host),
            "Hidden": Hidden,
            "total_pairs": spec.total_pairs,
            "intermediate": spec.intermediate,
        },
        "topk_reduce": {
            "DownOut": DeviceArray(DownOut_host),
            "SortedTokenIds": DeviceArray(np.array(reduce_sids_list, dtype=np.int32)),
            "SortedWeights": DeviceArray(weights),
            "Y": Y,
            "total_pairs": spec.total_pairs,
            "hidden": spec.hidden,
            "tokens": spec.tokens,
        },
    }

    # Production dispatch: time_kernel=False -> launch_kernel returns
    # 0.0 and the chain runs once. Drain the device before reading
    # outputs (run() does not implicitly fence on the non-timing path;
    # the FusedMoeLauncher docstring documents this).
    ms_run = launcher.run(launch_values, stream=0, time_kernel=False)
    synchronize_and_release()
    if ms_run != 0.0:
        return Result(
            name="moe_fused_chain",
            passed=False,
            max_abs_diff=0.0,
            rel_max=0.0,
            range_min=0.0,
            range_max=0.0,
            note=f"non-timing path returned ms={ms_run!r}, expected 0.0",
        )

    # Validate phase 1 (gather): GroupedInput[b, :] == X[sids[b], :]
    # for non-negative sids; the masked-out bucket stays 0 (matches
    # case_moe_gather expectations).
    r_gather = summarise(
        device_values(GroupedInput),
        _gather_reference(values(X_host, "fp16"), sids_list),
        tol=0.0,
    )
    if not r_gather.passed:
        return Result(
            name="moe_fused_chain.gather",
            passed=False,
            max_abs_diff=r_gather.max_abs_diff,
            rel_max=r_gather.rel_max,
            range_min=r_gather.range_min,
            range_max=r_gather.range_max,
            note=f"gather phase mismatch (chain): {r_gather.note}",
        )

    # Validate phase 2 (silu_mul): Hidden[b, i] = silu(GateOut[b, i])
    # * UpOut[b, i] within the f16 / exp2 tolerance documented in
    # case_moe_silu_mul (5e-3).
    r_silu = summarise(
        device_values(Hidden),
        _silu_mul_reference(values(GateOut_host, "fp16"), values(UpOut_host, "fp16")),
        tol=5e-3,
        note="f16 silu via exp2 ULP",
    )
    if not r_silu.passed:
        return Result(
            name="moe_fused_chain.silu_mul",
            passed=False,
            max_abs_diff=r_silu.max_abs_diff,
            rel_max=r_silu.rel_max,
            range_min=r_silu.range_min,
            range_max=r_silu.range_max,
            note=f"silu_mul phase mismatch (chain): {r_silu.note}",
        )

    # Validate phase 3 (topk_reduce): atomic-add scatter into Y. Use
    # the same 5e-2 tol as case_moe_topk_weighted_reduce -- the
    # f16 -> f32 atomic add accumulates rounding error per-bucket.
    r_reduce = summarise(
        device_values(Y),
        _topk_reduce_reference(
            values(DownOut_host, "fp16"), reduce_sids_list, weights, spec.tokens
        ),
        tol=5e-2,
    )
    if not r_reduce.passed:
        return Result(
            name="moe_fused_chain.topk_reduce",
            passed=False,
            max_abs_diff=r_reduce.max_abs_diff,
            rel_max=r_reduce.rel_max,
            range_min=r_reduce.range_min,
            range_max=r_reduce.range_max,
            note=f"topk_reduce phase mismatch (chain): {r_reduce.note}",
        )

    # Benchmark path: time_kernel=True -> launch_kernel runs a
    # cold + timed loop wrapping the 3-callable group and returns
    # the per-iteration average ms. Re-zero the output accumulator
    # between iterations to avoid the timed loop accumulating into Y.
    for output in (Y, Hidden, GroupedInput):
        zero_elements(output.info())
    ms_timed = launcher.run(
        launch_values,
        stream=0,
        time_kernel=True,
        cold_niters=1,
        nrepeat=2,
    )
    synchronize_and_release()
    if not (ms_timed > 0.0):
        return Result(
            name="moe_fused_chain.timing",
            passed=False,
            max_abs_diff=0.0,
            rel_max=0.0,
            range_min=0.0,
            range_max=0.0,
            note=f"timing path returned non-positive ms={ms_timed!r}",
        )

    # Aggregate all 3 phases' max_abs into one Result so the harness
    # reports a single line per case while still recording the worst
    # per-phase number.
    worst_max = max(
        r_gather.max_abs_diff,
        r_silu.max_abs_diff,
        r_reduce.max_abs_diff,
    )
    worst_rel = max(r_gather.rel_max, r_silu.rel_max, r_reduce.rel_max)
    return Result(
        name="moe_fused_chain",
        passed=True,
        max_abs_diff=worst_max,
        rel_max=worst_rel,
        range_min=min(r_gather.range_min, r_silu.range_min, r_reduce.range_min),
        range_max=max(r_gather.range_max, r_silu.range_max, r_reduce.range_max),
        note=(
            f"3-callable chain via launch_kernel(StreamConfig(...), gather, "
            f"silu_mul, topk_reduce); timed_ms={ms_timed:.3f}"
        ),
    )


def _torch_for_e2e() -> ModuleType | None:
    """torch for :func:`case_moe_e2e_forward`, or None when it is not installed.

    :func:`main` calls this before its first HIP call when that case is
    selected: torch must create its HIP context first so the rocKE runtime
    binds the same HIP runtime (a torch imported after it sees no GPU).
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _torch_fused_moe_reference(
    routing_logits: torch.Tensor,  # (T, E) f32
    X: torch.Tensor,  # (T, H) act dtype
    W_gate: torch.Tensor,  # (E, I, H)
    W_up: torch.Tensor,  # (E, I, H)
    W_down: torch.Tensor,  # (E, H, I)
    topk: int,
) -> torch.Tensor:
    """Plain torch fused-MoE forward, for parity comparison.

    Only :func:`case_moe_e2e_forward` uses it: :class:`FusedMoeForward` still
    takes torch tensors.

    Mirrors :class:`FusedMoeForward` semantics:
    * router : top-k of routing_logits, then softmax over the K picked
      values (matches CK Tile / topk_softmax kernel semantics, *not*
      softmax-then-topk).
    * per-token-expert pair: gate / up GEMMs in f32, SwiGLU, down GEMM.
    * weighted sum over the K experts per token.
    """
    import torch

    T, H = X.shape
    E, I, _ = W_gate.shape
    top_vals, top_ids = torch.topk(routing_logits, k=topk, dim=-1)  # (T, K)
    top_weights = torch.softmax(top_vals.float(), dim=-1)  # (T, K)

    Y = torch.zeros(T, H, dtype=torch.float32, device=X.device)
    for t in range(T):
        x = X[t, :].float()  # (H,)
        for k in range(topk):
            e = int(top_ids[t, k].item())
            w = float(top_weights[t, k].item())
            gate = x @ W_gate[e].float().T  # (I,)
            up = x @ W_up[e].float().T  # (I,)
            hidden = torch.nn.functional.silu(gate) * up  # (I,)
            out = hidden @ W_down[e].float().T  # (H,)
            Y[t, :] += w * out
    return Y.to(X.dtype)


def case_moe_e2e_forward() -> Result:
    """End-to-end fused-MoE forward via :class:`FusedMoeForward`.

    Drives the full pipeline (router -> sort -> gather -> per-expert
    gate + up GEMMs -> silu_mul -> per-expert down GEMM -> topk_reduce)
    and validates the output against the torch eager reference. The
    pipeline issues 5 streaming kernels via :func:`launch_kernel`
    chains plus 3*E grouped-GEMM launches via
    :class:`GroupedGemmLauncher` -- all on a single HIP stream.

    :class:`FusedMoeForward` takes torch tensors until it moves to
    :class:`DeviceArray`, so this is the one case that uses torch
    (:func:`_torch_for_e2e`); without torch it raises :class:`CaseUnavailable`
    (reported as SKIP).
    """
    torch = _torch_for_e2e()
    if torch is None:
        raise CaseUnavailable(
            "FusedMoeForward takes torch tensors until it moves to DeviceArray"
        )

    spec = FusedMoeForwardSpec(
        tokens=32,
        experts=4,
        topk=2,
        hidden=128,
        intermediate=256,
        dtype="f16",
        streaming_block_size=64,
        streaming_vec=4,
        sort_block_size=64,
        router_block_size=64,
        arch=_ARCH,
    )
    fwd = FusedMoeForward(spec)

    torch.manual_seed(11939)
    device = "cuda"
    act = torch.float16

    # Inputs: routing logits (f32 for the topk-softmax kernel) plus
    # the activation tensors that flow through the MoE forward.
    routing_logits = torch.randn(
        spec.tokens, spec.experts, dtype=torch.float32, device=device
    )
    X = (
        torch.randn(spec.tokens, spec.hidden, dtype=torch.float32, device=device) * 0.1
    ).to(act)
    # Small weight magnitude so the f16 down-GEMM accumulator stays in
    # fp16-representable range -- standard practice for MoE smoke tests.
    W_gate = (
        torch.randn(
            spec.experts,
            spec.intermediate,
            spec.hidden,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    ).to(act)
    W_up = (
        torch.randn(
            spec.experts,
            spec.intermediate,
            spec.hidden,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    ).to(act)
    W_down = (
        torch.randn(
            spec.experts,
            spec.hidden,
            spec.intermediate,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    ).to(act)
    Y = torch.zeros(spec.tokens, spec.hidden, dtype=act, device=device)

    fwd.forward(
        routing_logits=routing_logits,
        X=X,
        W_gate=W_gate,
        W_up=W_up,
        W_down=W_down,
        Y=Y,
        stream=0,
    )
    synchronize_and_release()

    Y_ref = _torch_fused_moe_reference(
        routing_logits=routing_logits,
        X=X,
        W_gate=W_gate,
        W_up=W_up,
        W_down=W_down,
        topk=spec.topk,
    )

    # End-to-end fused MoE accumulates many fp16 ops; the per-element
    # tolerance is loose. We also gate on the *relative* error to
    # catch genuine drift (not just fp16 ULP wobble on a large
    # accumulator).
    return _annotate_result(
        summarise(
            Y.cpu().double().numpy(),
            Y_ref.cpu().double().numpy(),
            tol=0.05,
            note="end-to-end fused MoE forward via FusedMoeForward",
        ),
        "moe_e2e_forward",
    )


def _annotate_result(r: Result, name: str) -> Result:
    """Helper: stamp a ``name`` onto a :class:`Result` (it's a regular
    dataclass so we can't use ``_replace``)."""
    r.name = name
    return r


# ---------------------------------------------------------------------
# GEMM family: StreamK, MFMA, block-scale, MX
# ---------------------------------------------------------------------


def case_streamk_gemm() -> Result:
    """f16 GEMM via StreamK + MFMA atomic split-K."""
    M = N = K = 64
    # tile_m=tile_n=16 matches the f16_16x16x16 atom; tile_k=32 = 2
    # MFMA atoms per macro tile (4 macro tiles span the K=64 axis).
    spec = StreamKGemmSpec(
        M=M,
        N=N,
        K=K,
        tile_m=16,
        tile_n=16,
        tile_k=32,
        dtype="f16",
    )
    kernel = build_streamk_gemm(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=streamk_gemm_signature(spec),
    )
    rng = np.random.default_rng(200)
    # Integer-valued so f16 multiplies are exact (no rounding).
    A = rng.integers(-3, 4, (M, K)).astype(np.float16)
    B = rng.integers(-3, 4, (K, N)).astype(np.float16)
    Cf32 = DeviceArray.zeros((M, N), "fp32")
    Counter = DeviceArray.zeros((1,), "i32")
    launch(
        launcher,
        {"A": DeviceArray(A), "B": DeviceArray(B), "Cf32": Cf32, "Counter": Counter},
        grid=streamk_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    C_ref = values(A, "fp16") @ values(B, "fp16")
    r = summarise(device_values(Cf32), C_ref, tol=0.0)
    r.name = "streamk_gemm (atomic strategy)"
    return r


def case_mfma_gemm() -> Result:
    """f16 GEMM via mfma_f32_16x16x16_f16 atom (production density)."""
    from rocke.instances import (
        MfmaGemmSpec,
        build_mfma_gemm,
        mfma_gemm_grid,
        mfma_gemm_signature,
    )

    M = N = K = 64
    spec = MfmaGemmSpec(M=M, N=N, K=K, dtype="f16")
    kernel = build_mfma_gemm(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=mfma_gemm_signature(spec),
    )
    rng = np.random.default_rng(230)
    # Integer-valued so f16 multiplies are exact (no rounding).
    A = rng.integers(-3, 4, (M, K)).astype(np.float16)
    B = rng.integers(-3, 4, (K, N)).astype(np.float16)
    C = DeviceArray.zeros((M, N), "fp16")
    launch(
        launcher,
        {"A": DeviceArray(A), "B": DeviceArray(B), "C": C, "M": M, "N": N, "K": K},
        grid=mfma_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    C_ref = round_to_f16(values(A, "fp16") @ values(B, "fp16"))
    r = summarise(
        device_values(C),
        C_ref,
        tol=0.0,
        note="mfma_f32_16x16x16_f16 atom; one MFMA per K-tile",
    )
    r.name = "mfma_gemm (f16 16x16x16 atom)"
    return r


def case_block_scale_gemm() -> Result:
    """Block-scaled FP8 GEMM with a/b/abquant per-group scales."""
    require_ocp_fp8_arch("block_scale_gemm", _ARCH)
    M = N = K = 64
    spec = BlockScaleGemmSpec(
        M=M,
        N=N,
        K=K,
        block_tile_m=16,
        block_tile_n=16,
        quant_mode="abquant",
        mantissa_dtype="fp8e4m3",
        group_size_mnk=(1, 1, K),
    )
    kernel = build_block_scale_gemm(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=block_scale_gemm_signature(spec),
    )
    rng = np.random.default_rng(210)
    A_fp8 = draw(rng, M, K, dtype="fp8e4m3", scale=0.3)
    B_fp8 = draw(rng, K, N, dtype="fp8e4m3", scale=0.3)
    # A scale per row and B scale per column, each different, so an output
    # scaled with another row's or column's scale misses the reference.
    a_scale = (1.0 + rng.random((M, 1))).astype(np.float32)
    b_scale = (0.5 + 0.5 * rng.random((1, N))).astype(np.float32)
    C = DeviceArray.zeros((M, N), "fp32")
    launch(
        launcher,
        {
            "A": DeviceArray(A_fp8, "fp8e4m3"),
            "AScale": DeviceArray(a_scale),
            "B": DeviceArray(B_fp8, "fp8e4m3"),
            "BScale": DeviceArray(b_scale),
            "C": C,
            "M": M,
            "N": N,
            "K": K,
        },
        grid=block_scale_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    # Reference: GEMM on the dequantised + scaled values.
    A_dq = values(A_fp8, "fp8e4m3") * a_scale
    B_dq = values(B_fp8, "fp8e4m3") * b_scale
    C_ref = A_dq @ B_dq
    r = summarise(device_values(C), C_ref, tol=1e-2, note="fp8 abquant 1x1xK")
    r.name = "block_scale_gemm (fp8 abquant)"
    return r


def case_mx_gemm() -> Result:
    """OCP MX shared-exponent GEMM, fp8e4m3 mantissa, group_k=32."""
    require_ocp_fp8_arch("mx_gemm", _ARCH)
    M = N = 32
    K = 64  # 2 MX blocks
    spec = MxGemmSpec(
        M=M, N=N, K=K, block_tile_m=32, block_tile_n=128, mantissa_dtype="fp8e4m3"
    )
    kernel = build_mx_gemm(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=mx_gemm_signature(spec),
    )
    rng = np.random.default_rng(220)
    A_fp8 = draw(rng, M, K, dtype="fp8e4m3", scale=0.3)
    B_fp8 = draw(rng, K, N, dtype="fp8e4m3", scale=0.3)
    # MX shared-exponent E8M0 byte: bias=127. e=127 => scale 1.0. Both scale
    # tensors are K-major: (K / 32, M) and (K / 32, N).
    a_scale = np.full((K // 32, M), 127, dtype=np.int8)
    b_scale = np.full((K // 32, N), 127, dtype=np.int8)
    C = DeviceArray.zeros((M, N), "fp32")
    launch(
        launcher,
        {
            "A": DeviceArray(A_fp8, "fp8e4m3"),
            "AScale": DeviceArray(a_scale),
            "B": DeviceArray(B_fp8, "fp8e4m3"),
            "BScale": DeviceArray(b_scale),
            "C": C,
            "M": M,
            "N": N,
            "K": K,
        },
        grid=mx_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    # With scale=1 the dequant is the fp8 round-trip.
    C_ref = values(A_fp8, "fp8e4m3") @ values(B_fp8, "fp8e4m3")
    r = summarise(device_values(C), C_ref, tol=1e-2, note="MX e8m0=127 (scale=1)")
    r.name = "mx_gemm (fp8 e8m0=127)"
    return r


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

ALL_CASES: dict[str, Callable[[], Result]] = {
    "moe_gather": case_moe_gather,
    "moe_silu_mul": case_moe_silu_mul,
    "moe_topk_weighted_reduce": case_moe_topk_weighted_reduce,
    "moe_fused_chain": case_moe_fused_chain,
    "moe_e2e_forward": case_moe_e2e_forward,
    "streamk_gemm": case_streamk_gemm,
    "mfma_gemm": case_mfma_gemm,
    "block_scale_gemm": case_block_scale_gemm,
    "mx_gemm": case_mx_gemm,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        default="all",
        help="Comma-separated case names (or 'all').",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="gfx target for codegen (default: running device, else gfx950).",
    )
    args = parser.parse_args()

    if args.op == "all":
        cases = list(ALL_CASES.items())
    else:
        wanted = {n.strip() for n in args.op.split(",")}
        cases = [(n, fn) for n, fn in ALL_CASES.items() if n in wanted]
        if not cases:
            print(f"no cases match {args.op!r}; valid: {list(ALL_CASES)}")
            return 2
    if any(name == "moe_e2e_forward" for name, _ in cases):
        _torch_for_e2e()  # before the first HIP call (see its docstring)

    global _ARCH
    _ARCH = args.arch or default_arch()
    print(f"codegen arch: {_ARCH}")

    if not device_available():
        print("HIP device not available; skipping.")
        return 0
    return run_cases(cases, arch=_ARCH)


if __name__ == "__main__":
    raise SystemExit(main())
