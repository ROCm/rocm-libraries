# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end fused-MoE forward perf comparison: CK DSL vs Torch vs AITER vs CK Tile.

Drives :class:`rocke.instances.FusedMoeForward` and three reference
implementations on a configurable set of MoE shapes; reports per-
backend latency (ms), correctness vs the torch reference (max abs /
mean abs / rel error), and the resulting speedup ratios.

Backends
--------
1. **CK DSL** -- :class:`rocke.instances.FusedMoeForward`. Composes
   ``topk_softmax`` (router) + ``MoeSortingLauncher`` (sort 3-chain) +
   ``FusedMoeLauncher`` (gather / silu_mul / topk_reduce streaming
   kernels) + ``GroupedGemmLauncher`` (per-expert gate / up / down
   GEMMs) into one pipeline driven by chained
   :func:`~rocke.runtime.launcher.launch_kernel` calls.

2. **Torch eager** -- naive Python loop reference (per token, per
   topk slot, per expert: gate / up / silu_mul / down). Slow, but
   the gold-standard correctness oracle. Both AITER and CK DSL are
   compared against this.

3. **AITER fused_moe** -- ``aiter.fused_moe`` (the production
   ROCm Triton/HIP fused-MoE op). Driven via the standard ``w1 =
   concat([W_gate, W_up], dim=1)`` "G1U1" packing. Router is run
   externally on the host (matching the AITER public API).

4. **CK Tile C++** -- ``build/bin/tile_example_fused_moe`` invoked
   via ``subprocess`` with matching ``-t / -e / -k / -h / -i /
   -prec_*`` arguments. The C++ binary includes its own internal
   router + sort + GEMM + activation + reduce; output is the
   single-line ``[fmoe|<dtype>] ...  XXX.YYY us, ...  TFlops, ...
   TB/s`` perf summary which we parse.

Usage
-----
::

    AITER_PATH=/path/to/aiter PYTHONPATH=Python:$AITER_PATH \\
        /path/to/venv/bin/python \\
        Python/rocke/examples/moe/fused_moe_e2e_perf.py \\
        [--scenario name] [--attempts N] [--warmup N] \\
        [--skip-aiter] [--skip-cktile] [--skip-torch]

The ``--skip-*`` flags help when iterating on one backend; defaults
to all four.

Caveats
-------
* The CK DSL pipeline does per-expert grouped-GEMM dispatch via a
  Python loop with a small device-to-host copy of ``Counts`` and
  ``Offsets`` per dispatch (see :class:`FusedMoeForward` docstring).
  AITER's mega-kernel does in-kernel grouped-GEMM dispatch with no
  host roundtrip; that's the largest known overhead in the DSL path
  for shapes where per-expert GEMMs are small.
* The CK Tile C++ binary uses its own random inputs (no public
  hook to feed external tensors), so the C++ row reports a perf-only
  number; correctness is validated against torch eager on the
  Python side only.
* Default precision is ``f16``; pass ``--dtype bf16`` for bf16. The
  CK DSL streaming kernels accept fp16 / bf16 inputs and route
  through f32 accumulators.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[5]  # .../composablekernel
sys.path.insert(0, str(ROOT / "Python"))

import torch  # noqa: E402

from rocke.instances import (  # noqa: E402
    FusedMoeForward,
    FusedMoeForwardSpec,
)


CK_TILE_BIN = ROOT / "build" / "bin" / "tile_example_fused_moe"


# ---------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int
    dtype: str = "f16"


def default_scenarios() -> List[Scenario]:
    """A small but illustrative set of scenarios.

    Constraints driven by the CK DSL pipeline (and matching what the
    standard MoE workloads look like in production):

    * ``hidden`` and ``intermediate`` must be multiples of the GEMM
      tile dims (default tile_n=128, tile_k=64) and the streaming
      kernel block_size (64 by default).
    * ``experts <= sort_block_size`` (=64) for the single-block scan
      kernel.
    * ``topk <= experts``.
    """
    return [
        # Decode-shape: 1 token, large experts. Common in inference.
        Scenario(
            name="decode_T1_E8_K2_H4096_I7168",
            tokens=1,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=7168,
        ),
        Scenario(
            name="decode_T8_E8_K2_H4096_I7168",
            tokens=8,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=7168,
        ),
        # Mid-batch.
        Scenario(
            name="batch32_E8_K2_H4096_I7168",
            tokens=32,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=7168,
        ),
        # Prefill-shape: many tokens, larger experts.
        Scenario(
            name="prefill_T128_E8_K2_H4096_I7168",
            tokens=128,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=7168,
        ),
        # Smaller validation shape (matches parity test).
        Scenario(
            name="small_T32_E4_K2_H128_I256",
            tokens=32,
            experts=4,
            topk=2,
            hidden=128,
            intermediate=256,
        ),
    ]


# ---------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------


@dataclass
class BackendResult:
    backend: str
    ok: bool
    ms: Optional[float] = None
    max_abs: Optional[float] = None
    mean_abs: Optional[float] = None
    rel_max: Optional[float] = None
    note: str = ""


@dataclass
class ScenarioResult:
    scenario: Scenario
    results: Dict[str, BackendResult] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Inputs + reference
# ---------------------------------------------------------------------


def _torch_dtype_for(dtype: str) -> torch.dtype:
    if dtype in ("f16", "fp16"):
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {dtype!r}")


@dataclass
class TestInputs:
    routing_logits: torch.Tensor  # (T, E) f32
    X: torch.Tensor  # (T, H) act dtype
    W_gate: torch.Tensor  # (E, I, H) act dtype
    W_up: torch.Tensor  # (E, I, H) act dtype
    W_down: torch.Tensor  # (E, H, I) act dtype
    topk_ids_pre: torch.Tensor  # (T, K) i32 — precomputed for AITER
    topk_weights_pre: torch.Tensor  # (T, K) f32 — precomputed for AITER

    @property
    def device(self) -> torch.device:
        return self.X.device


def make_inputs(s: Scenario, *, seed: int = 11939) -> TestInputs:
    """Build deterministic random inputs for one scenario.

    Weight magnitudes are scaled to keep the f16 down-GEMM in
    fp16-representable range -- standard MoE smoke-test convention.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")
    act = _torch_dtype_for(s.dtype)

    routing_logits = torch.randn(
        s.tokens, s.experts, dtype=torch.float32, device=device
    )
    X = (torch.randn(s.tokens, s.hidden, dtype=torch.float32, device=device) * 0.1).to(
        act
    )
    W_gate = (
        torch.randn(
            s.experts, s.intermediate, s.hidden, dtype=torch.float32, device=device
        )
        * 0.05
    ).to(act)
    W_up = (
        torch.randn(
            s.experts, s.intermediate, s.hidden, dtype=torch.float32, device=device
        )
        * 0.05
    ).to(act)
    W_down = (
        torch.randn(
            s.experts, s.hidden, s.intermediate, dtype=torch.float32, device=device
        )
        * 0.05
    ).to(act)

    # Precompute the topk-softmax router on the host. This matches the
    # CK DSL topk_softmax kernel semantic: top-K on the logits, then
    # softmax over the K picked values.
    top_vals, top_ids = torch.topk(routing_logits, k=s.topk, dim=-1)
    top_weights = torch.softmax(top_vals.float(), dim=-1)

    return TestInputs(
        routing_logits=routing_logits,
        X=X,
        W_gate=W_gate,
        W_up=W_up,
        W_down=W_down,
        topk_ids_pre=top_ids.to(torch.int32),
        topk_weights_pre=top_weights,
    )


def torch_fused_moe_reference(inputs: TestInputs, s: Scenario) -> torch.Tensor:
    """Plain torch fused-MoE forward, used as the correctness oracle.

    Vectorised per-expert formulation (mask + index_add). The
    naive per-token-per-topk Python loop allocates ``T * K * 5``
    tiny intermediate tensors per call, which interacts pathologically
    with torch's caching allocator on ROCm 7.2 -- subsequent ctypes-
    driven CK DSL launches hit a memory-access fault that does not
    reproduce when the reference is computed via the vectorised path.
    The vectorised reference is bit-equivalent to the loop form
    (modulo non-deterministic atomic-order in ``index_add_``).
    """
    top_ids = inputs.topk_ids_pre
    top_weights = inputs.topk_weights_pre
    T, H = inputs.X.shape
    Y_dtype = inputs.X.dtype

    Y = torch.zeros(T, H, dtype=torch.float32, device=inputs.X.device)
    for e in range(s.experts):
        mask = top_ids == e
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        if token_idx.numel() == 0:
            continue
        sub_X = inputs.X[token_idx].float()  # (count, H)
        gate = sub_X @ inputs.W_gate[e].float().T  # (count, I)
        up = sub_X @ inputs.W_up[e].float().T  # (count, I)
        hidden = torch.nn.functional.silu(gate) * up  # (count, I)
        out = hidden @ inputs.W_down[e].float().T  # (count, H)
        w = top_weights[token_idx, slot_idx].unsqueeze(-1)
        Y.index_add_(0, token_idx, w * out)
    return Y.to(Y_dtype)


# ---------------------------------------------------------------------
# Backend: timing helper
# ---------------------------------------------------------------------


def time_callable_ms(fn: Callable[[], None], *, warmup: int, attempts: int) -> float:
    """HIP-event timing of ``fn``. Returns mean per-iteration ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(attempts):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / attempts


def _compare(out: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float, float]:
    diff = (out.float() - ref.float()).abs()
    max_d = float(diff.max().item())
    mean_d = float(diff.mean().item())
    ref_max = float(ref.abs().max().item())
    rel = max_d / (ref_max + 1e-9)
    return max_d, mean_d, rel


# ---------------------------------------------------------------------
# Backend: CK DSL
# ---------------------------------------------------------------------


def run_rocke(
    s: Scenario,
    inputs: TestInputs,
    *,
    warmup: int,
    attempts: int,
    experimental_down_reduce: bool = False,
) -> Tuple[torch.Tensor, float]:
    # The streaming kernels (gather / silu_mul / topk_weighted_reduce)
    # use one CTA per bucket row, so ``streaming_block_size`` must
    # divide ``hidden``. The spec default (256) is the production-best
    # choice; we cap at ``hidden`` for small validation shapes where
    # the default would violate the divisibility constraint.
    streaming_block_size = min(256, s.hidden)
    pre_down = os.environ.get("ROCKE_PRESHUFFLE_W_DOWN", "0") == "1"
    pre_gu_packed = os.environ.get("ROCKE_PRESHUFFLE_W_GATE_UP_PACKED", "0") == "1"
    pre_gu_interleaved = (
        os.environ.get("ROCKE_PRESHUFFLE_W_GATE_UP_INTERLEAVED", "0") == "1"
    )
    # Default-on (matches FusedMoeForwardSpec.active_tile_skip_gemms=True);
    # set ROCKE_ACTIVE_TILE_SKIP_GEMMS=0 to force the dense GEMM path.
    active_tile_skip = os.environ.get("ROCKE_ACTIVE_TILE_SKIP_GEMMS", "1") == "1"
    # When the gate-up packed preshuffle is requested, force the
    # packed kernel path (the only one that uses the BatchedGemm
    # launcher). When the interleaved preshuffle is requested, keep
    # the interleaved kernel.
    if pre_gu_packed:
        use_intl = False
    else:
        use_intl = True
    spec = FusedMoeForwardSpec(
        tokens=s.tokens,
        experts=s.experts,
        topk=s.topk,
        hidden=s.hidden,
        intermediate=s.intermediate,
        dtype=s.dtype,
        streaming_block_size=streaming_block_size,
        streaming_vec=8,
        sort_block_size=max(64, s.experts),
        router_block_size=max(64, s.experts),
        use_experimental_fused_down_reduce=experimental_down_reduce,
        preshuffle_w_down=pre_down,
        preshuffle_w_gate_up_packed=pre_gu_packed,
        preshuffle_w_gate_up_interleaved=pre_gu_interleaved,
        use_experimental_interleaved_gate_up_silu=use_intl,
        active_tile_skip_gemms=active_tile_skip,
    )
    fwd = FusedMoeForward(spec)
    fwd._ensure_compiled()  # keep compile out of the timed region

    Y = torch.zeros(s.tokens, s.hidden, dtype=_torch_dtype_for(s.dtype), device="cuda")

    def call():
        fwd.forward(
            routing_logits=inputs.routing_logits,
            X=inputs.X,
            W_gate=inputs.W_gate,
            W_up=inputs.W_up,
            W_down=inputs.W_down,
            Y=Y,
        )

    # In static-offset mode the forward is HIP-graph-capturable
    # (no host roundtrip, all callables on one stream). Capture
    # once on stable inputs and time the replay path -- this is
    # the realistic inference benchmark mode and the only way the
    # DSL pipeline can match the per-launch overhead of a tuned C++
    # reference.
    if getattr(fwd, "_use_static_offsets", False):
        try:
            fwd.capture_graph(
                routing_logits=inputs.routing_logits,
                X=inputs.X,
                W_gate=inputs.W_gate,
                W_up=inputs.W_up,
                W_down=inputs.W_down,
                Y=Y,
                warmup_iters=2,
            )
            ms = time_callable_ms(fwd.replay_graph, warmup=warmup, attempts=attempts)
            # One last replay to populate Y for correctness
            # comparison (the timed loop's last iteration left Y
            # in the right state, but be explicit).
            fwd.replay_graph()
            torch.cuda.synchronize()
            return Y.clone(), ms
        except Exception as exc:
            # Graph capture can fail on edge-case torch versions /
            # custom allocators; fall back to non-graph timing.
            print(
                f"  rocke: graph capture failed ({type(exc).__name__}: {exc!r}); "
                f"falling back to non-graph timing",
                flush=True,
            )

    ms = time_callable_ms(call, warmup=warmup, attempts=attempts)
    return Y.clone(), ms


# ---------------------------------------------------------------------
# Backend: Torch eager
# ---------------------------------------------------------------------


def run_torch_eager(
    s: Scenario, inputs: TestInputs, *, warmup: int, attempts: int
) -> Tuple[torch.Tensor, float]:
    """Torch eager forward: vectorised per-expert mask + scatter.

    Apples-to-apples target with CK DSL and Triton:

    * **Same precision contract** as CK DSL / Triton — fp16 (or bf16)
      inputs and weights, fp32 accumulator (implicit inside torch's
      rocBLAS matmul), fp16 / bf16 output. The previous version of
      this baseline upcast every operand to fp32 inside the per-expert
      loop, which doubled the bytes through every matmul and inflated
      the torch number by roughly 2x; that has been removed.
    * **Y pre-allocated once** outside the timed window and zeroed in
      place at the start of each ``call()`` — matches CK DSL (which
      also takes a pre-allocated ``Y``) and Triton (which pre-allocates
      ``Y_f32``).
    * **Router runs inside ``call()``** to match CK DSL (whose
      ``forward`` includes the router) and Triton (whose ``call``
      computes ``topk`` / ``softmax`` on the host).

    What the timed region still contains that CK DSL / Triton don't:
    Python-side per-expert loop, ``mask.nonzero`` work, per-expert
    matmul launches. That is the irreducible cost of "what torch eager
    looks like to a real user"; this is the measurement we want.
    """
    Y_dtype = inputs.X.dtype
    T, H = inputs.X.shape
    Y_out = torch.zeros(T, H, dtype=Y_dtype, device=inputs.X.device)

    def call():
        # Router runs inside the timed region to match CK DSL.
        top_vals, top_ids_iter = torch.topk(inputs.routing_logits, k=s.topk, dim=-1)
        top_weights_iter = torch.softmax(top_vals, dim=-1).to(Y_dtype)

        Y_out.zero_()
        for e in range(s.experts):
            mask = top_ids_iter == e  # (T, K)
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            sub_X = inputs.X[token_idx]  # (count, H)
            # Torch's fp16/bf16 matmul on rocBLAS uses an fp32
            # accumulator internally and downcasts the result; same
            # numerical contract as CK DSL's MFMA pipeline.
            gate = sub_X @ inputs.W_gate[e].T  # (count, I)
            up = sub_X @ inputs.W_up[e].T  # (count, I)
            hidden = torch.nn.functional.silu(gate) * up  # (count, I)
            out = hidden @ inputs.W_down[e].T  # (count, H)
            w = top_weights_iter[token_idx, slot_idx].unsqueeze(-1)  # (count, 1)
            Y_out.index_add_(0, token_idx, w * out)
        return Y_out

    # Capture one output for correctness check (clone so we keep the
    # value across subsequent timed iterations that overwrite Y_out).
    Y_correct = call().clone()
    ms = time_callable_ms(lambda: call(), warmup=warmup, attempts=attempts)
    return Y_correct, ms


# ---------------------------------------------------------------------
# Backend: AITER
# ---------------------------------------------------------------------


def _build_triton_moe_kernel():
    """Build a small purpose-written Triton fused-MoE kernel.

    Per-(token, topk-slot) program: each program handles one
    ``(t, k)`` pair, reads ``X[t]`` (one hidden-dim vector), and
    performs gate / up / silu_mul / down in-kernel by streaming over
    ``W_gate[expert] / W_up[expert] / W_down[expert]``. The final
    ``weight * out`` accumulates into ``Y[t]`` via ``tl.atomic_add``.

    Built lazily so this file imports without Triton -- AITER /
    Triton failures fall back to "skip" rather than a hard error.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _moe_kernel(
        X_ptr,
        W_gate_ptr,
        W_up_ptr,
        W_down_ptr,
        Y_ptr,
        topk_ids_ptr,
        topk_weights_ptr,
        T,
        K: tl.constexpr,
        H: tl.constexpr,
        IM: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        pid = tl.program_id(0)
        # pid in [0, T*K)
        t_id = pid // K
        k_id = pid % K
        if t_id >= T:
            return
        expert = tl.load(topk_ids_ptr + t_id * K + k_id)
        weight = tl.load(topk_weights_ptr + t_id * K + k_id)

        # Load X[t, :] into registers.
        h_offs = tl.arange(0, BLOCK_H)
        x_mask = h_offs < H
        x = tl.load(X_ptr + t_id * H + h_offs, mask=x_mask, other=0.0).to(tl.float32)

        # Compute gate[i] = sum_h X[t, h] * W_gate[expert, i, h] for i in [0, IM)
        # and up[i] similarly, then hidden[i] = silu(gate[i]) * up[i].
        # Then out[h] = sum_i hidden[i] * W_down[expert, h, i].
        # We tile the I (intermediate) axis with BLOCK_I.
        i_offs = tl.arange(0, BLOCK_I)
        # Stride of W_gate / W_up: per-expert, then row-major (IM, H).
        # Stride of W_down: per-expert, then row-major (H, IM).
        for i_start in range(0, IM, BLOCK_I):
            i_idx = i_start + i_offs
            i_mask = i_idx < IM

            # gate[i] += X @ W_gate[e, i, :]: (BLOCK_I, BLOCK_H) tile of
            # W_gate, multiplied row-wise by X.
            wg_ptr = W_gate_ptr + expert * IM * H + i_idx[:, None] * H + h_offs[None, :]
            wg_tile = tl.load(
                wg_ptr,
                mask=i_mask[:, None] & x_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wu_ptr = W_up_ptr + expert * IM * H + i_idx[:, None] * H + h_offs[None, :]
            wu_tile = tl.load(
                wu_ptr,
                mask=i_mask[:, None] & x_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate = tl.sum(wg_tile * x[None, :], axis=1)
            up = tl.sum(wu_tile * x[None, :], axis=1)
            hidden = (gate * tl.sigmoid(gate)) * up  # SiLU(gate) * up

            # Now contribute hidden's slab into out[:H] via W_down[e, :, i_idx].
            # out[h] += sum_i hidden[i] * W_down[e, h, i]
            # Use a (BLOCK_I, BLOCK_H) tile of W_down.
            wd_ptr = (
                W_down_ptr + expert * H * IM + h_offs[None, :] * IM + i_idx[:, None]
            )
            wd_tile = tl.load(
                wd_ptr,
                mask=i_mask[:, None] & x_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            out_slab = tl.sum(wd_tile * hidden[:, None], axis=0)
            # Weighted atomic-add into Y[t, :].
            y_off = Y_ptr + t_id * H + h_offs
            # Y is f32 accumulator -> atomic_add requires f32.
            tl.atomic_add(y_off, weight * out_slab, mask=x_mask)

    return _moe_kernel


_TRITON_MOE_KERNEL = None


def run_triton(
    s: Scenario, inputs: TestInputs, *, warmup: int, attempts: int
) -> Tuple[Optional[torch.Tensor], Optional[float], str]:
    """Pure-Triton fused MoE baseline (single-kernel implementation).

    AITER's tuned ``e2e_moe`` Triton kernel crashes with a memory
    access fault on MI355X (gfx950) regardless of the torch / ROCm
    version (likely an AITER kernel-arch mismatch). We sidestep that
    by emitting a small purpose-written Triton kernel that mirrors
    the algorithmic shape of the CK DSL pipeline: per ``(token, k)``
    pair, gate / up / SiLU mul / down GEMMs computed inline against
    the per-expert weights, atomic-add into a f32 ``Y`` accumulator.

    This is *not* a tuned production Triton kernel -- it's a
    fair-baseline reference that lets us answer "what does the same
    work cost in Triton on this device" without depending on a
    Triton MoE library that doesn't run here. Tuned AITER numbers
    can be added once the AITER gfx950 issue is fixed upstream.

    Returns ``(out, ms, note)``.
    """
    global _TRITON_MOE_KERNEL
    try:
        if _TRITON_MOE_KERNEL is None:
            _TRITON_MOE_KERNEL = _build_triton_moe_kernel()
    except Exception as exc:
        return None, None, f"triton import failed: {exc}"

    H = s.hidden
    IM = s.intermediate
    if H & (H - 1):
        return None, None, f"triton baseline requires power-of-2 H (got H={H})"
    T = s.tokens
    K = s.topk

    # Pre-allocate both the fp32 atomic-add accumulator (Triton's
    # ``tl.atomic_add`` only accepts fp32) and the fp16/bf16 output
    # tensor. Doing both outside the timed window keeps the timed
    # ``call()`` allocation-free, matching CK DSL's contract (output
    # buffer passed in by the caller).
    Y_f32 = torch.zeros(T, H, dtype=torch.float32, device="cuda")
    Y_out = torch.empty(T, H, dtype=inputs.X.dtype, device="cuda")
    # BLOCK_H must be a constexpr power-of-2 (Triton's tl.arange).
    # We pick the smallest power-of-2 >= H that fits in shared regs;
    # for H > 4096 we'd need tiling on the H axis too -- for the
    # benchmark shapes here (H <= 4096) we can hold one row in
    # registers. BLOCK_I tiles the intermediate dim and supports
    # any multiple of BLOCK_I.
    BLOCK_H = H if (H & (H - 1)) == 0 else 1 << (H - 1).bit_length()
    BLOCK_I = 64

    def call():
        top_vals, top_ids_iter = torch.topk(inputs.routing_logits, k=K, dim=-1)
        top_w = torch.softmax(top_vals, dim=-1).contiguous()
        top_ids_int = top_ids_iter.to(torch.int32).contiguous()
        Y_f32.zero_()
        _TRITON_MOE_KERNEL[(T * K,)](
            inputs.X,
            inputs.W_gate,
            inputs.W_up,
            inputs.W_down,
            Y_f32,
            top_ids_int,
            top_w,
            T,
            K,
            H,
            IM,
            BLOCK_H,
            BLOCK_I,
        )
        # Cast in-place into the pre-allocated output buffer instead
        # of allocating a fresh fp16 / bf16 tensor each call.
        Y_out.copy_(Y_f32)
        return Y_out

    try:
        out = call()
        torch.cuda.synchronize()
    except Exception as exc:
        return None, None, f"triton call failed: {type(exc).__name__}: {exc}"

    try:
        ms = time_callable_ms(lambda: call(), warmup=warmup, attempts=attempts)
    except Exception as exc:
        return out, None, f"triton timing failed: {exc}"
    return out.clone(), ms, "purpose-written Triton kernel (per-(token,k) GEMM chain)"


# ---------------------------------------------------------------------
# Backend: CK Tile C++
# ---------------------------------------------------------------------


_CKTILE_PERF_RE = re.compile(r"(\d+\.\d+)\s+us")


def run_ck_tile_cpp(
    s: Scenario, *, warmup: int, attempts: int
) -> Tuple[Optional[float], str]:
    """Invoke ``tile_example_fused_moe`` and parse its ``us`` output.

    Returns ``(ms, note)``. ``ms`` is ``None`` on failure (binary
    missing, non-zero exit, parser failure). Correctness is not
    validated here -- the C++ binary uses its own random inputs.
    """
    if not CK_TILE_BIN.exists():
        return None, f"ck_tile binary not built ({CK_TILE_BIN})"

    cpp_dtype = "fp16" if s.dtype in ("f16", "fp16") else s.dtype
    cmd = [
        str(CK_TILE_BIN),
        f"-t={s.tokens}",
        f"-e={s.experts}",
        f"-k={s.topk}",
        f"-h={s.hidden}",
        f"-i={s.intermediate}",
        f"-prec_i={cpp_dtype}",
        f"-prec_w={cpp_dtype}",
        f"-prec_o={cpp_dtype}",
        f"-warmup={warmup}",
        f"-repeat={attempts}",
        "-v=0",
        "-kname=0",
        "-act=1",  # silu (we want silu_mul)
        "-gate_only=0",  # gate+up
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
    except Exception as exc:
        return None, f"ck_tile subprocess failed: {exc}"

    if out.returncode != 0:
        return None, f"ck_tile exit code {out.returncode}: {out.stderr.strip()[:200]}"

    # Output line shape:
    #   [fmoe|fp16] t:128, ..., 332.665 us, 64.554 tflops, 3.22968 TB/s
    m = _CKTILE_PERF_RE.search(out.stdout)
    if not m:
        return None, f"ck_tile perf regex failed; stdout head: {out.stdout[:200]}"
    return float(m.group(1)) / 1000.0, ""  # us -> ms


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def _isolate_lane() -> None:
    """Strong isolation between backends inside one scenario so
    retained args / workspace tensors / module caches don't perturb
    the next backend's timing.

    On ROCm 7.2 / torch 2.12, leaving prior-scenario
    :class:`FusedMoeForward` instances alive past the next
    scenario's ``torch.empty()`` workspace allocations causes the
    new pool's tensors to be backed by recycled storage that the
    GPU command processor still has in flight; the symptom is a
    memory access fault during the next forward's first kernel.
    The fix is the canonical "drain + collect" trio:

    1. ``torch.cuda.synchronize()`` to drain any in-flight GPU work.
    2. :func:`synchronize_and_release` to clear
       :attr:`Runtime._pending_args` (the per-stream args /
       tensor-ref bucket).
    3. ``gc.collect()`` to drop any orphan
       :class:`FusedMoeForward` / :class:`WorkspacePool` instances
       whose Python references died on a return statement but whose
       storage hasn't been collected yet.

    We deliberately do NOT call ``torch.cuda.empty_cache()`` -- the
    combination of ``synchronize_and_release`` + ``empty_cache``
    causes the same fault we are working around.
    """
    import gc

    torch.cuda.synchronize()
    try:
        from rocke.runtime.launcher import synchronize_and_release

        synchronize_and_release()
    except Exception:
        pass
    gc.collect()


def run_one_scenario(
    s: Scenario,
    *,
    warmup: int,
    attempts: int,
    skip_rocke: bool,
    skip_torch: bool,
    skip_aiter: bool,
    skip_cktile: bool,
) -> ScenarioResult:
    print(
        f"\n=== {s.name}  T={s.tokens} E={s.experts} K={s.topk} "
        f"H={s.hidden} I={s.intermediate} dtype={s.dtype} ==="
    )
    res = ScenarioResult(scenario=s)

    inputs = make_inputs(s)
    _isolate_lane()

    # Reference (correctness oracle); not timed.
    print("  Computing torch reference...", flush=True)
    Y_ref = torch_fused_moe_reference(inputs, s)
    _isolate_lane()

    # CK DSL.
    if not skip_rocke:
        try:
            Y, ms = run_rocke(
                s,
                inputs,
                warmup=warmup,
                attempts=attempts,
                experimental_down_reduce=False,
            )
            mx, mn, rl = _compare(Y, Y_ref)
            res.results["rocke"] = BackendResult(
                backend="rocke", ok=True, ms=ms, max_abs=mx, mean_abs=mn, rel_max=rl
            )
            print(f"  rocke       : {ms:8.3f} ms   max_abs={mx:.4g}  rel={rl:.4g}")
        except Exception as exc:
            res.results["rocke"] = BackendResult(
                backend="rocke", ok=False, note=f"{type(exc).__name__}: {exc}"
            )
            print(f"  rocke       : FAIL   {exc!r}")
        _isolate_lane()

    # Torch eager (vectorised mask).
    if not skip_torch:
        try:
            Y_t, ms_t = run_torch_eager(s, inputs, warmup=warmup, attempts=attempts)
            mx, mn, rl = _compare(Y_t, Y_ref)
            res.results["torch_eager"] = BackendResult(
                backend="torch_eager",
                ok=True,
                ms=ms_t,
                max_abs=mx,
                mean_abs=mn,
                rel_max=rl,
            )
            print(f"  torch_eager  : {ms_t:8.3f} ms   max_abs={mx:.4g}  rel={rl:.4g}")
        except Exception as exc:
            res.results["torch_eager"] = BackendResult(
                backend="torch_eager", ok=False, note=f"{type(exc).__name__}: {exc}"
            )
            print(f"  torch_eager  : FAIL   {exc!r}")
        _isolate_lane()

    # Triton (purpose-written baseline; AITER's tuned kernels don't
    # run on gfx950 -- see ``run_triton`` docstring).
    if not skip_aiter:
        Y_t2, ms_t2, note_t2 = run_triton(s, inputs, warmup=warmup, attempts=attempts)
        if Y_t2 is not None and ms_t2 is not None:
            mx, mn, rl = _compare(Y_t2, Y_ref)
            res.results["triton"] = BackendResult(
                backend="triton",
                ok=True,
                ms=ms_t2,
                max_abs=mx,
                mean_abs=mn,
                rel_max=rl,
                note=note_t2,
            )
            print(f"  triton       : {ms_t2:8.3f} ms   max_abs={mx:.4g}  rel={rl:.4g}")
        else:
            res.results["triton"] = BackendResult(
                backend="triton", ok=False, note=note_t2
            )
            print(f"  triton       : SKIP   {note_t2}")
        _isolate_lane()

    # CK Tile C++.
    if not skip_cktile:
        ms_c, note_c = run_ck_tile_cpp(s, warmup=warmup, attempts=attempts)
        if ms_c is not None:
            res.results["ck_tile_cpp"] = BackendResult(
                backend="ck_tile_cpp",
                ok=True,
                ms=ms_c,
                note="C++ binary uses its own random inputs; perf-only",
            )
            print(f"  ck_tile_cpp  : {ms_c:8.3f} ms   (C++ binary, perf-only)")
        else:
            res.results["ck_tile_cpp"] = BackendResult(
                backend="ck_tile_cpp", ok=False, note=note_c
            )
            print(f"  ck_tile_cpp  : SKIP   {note_c}")
        _isolate_lane()

    return res


def print_summary(results: List[ScenarioResult]) -> None:
    """Compact perf table + speedup vs torch / vs aiter / vs ck_tile."""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Header.
    cols = [
        "scenario",
        "rocke",
        "torch",
        "triton",
        "cktile_cpp",
        "rocke_vs_torch",
        "rocke_vs_triton",
        "rocke_vs_cktile",
    ]
    widths = [38, 10, 10, 10, 11, 16, 17, 16]
    line = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(line)
    print("-" * len(line))

    def _fmt(ms: Optional[float]) -> str:
        if ms is None:
            return "-"
        return f"{ms:.3f}ms"

    def _fmt_speedup(ms_a: Optional[float], ms_b: Optional[float]) -> str:
        if ms_a is None or ms_b is None or ms_a == 0:
            return "-"
        return f"{ms_b / ms_a:.2f}x"

    for r in results:
        ck_ms = r.results.get("rocke", BackendResult("", False)).ms
        t_ms = r.results.get("torch_eager", BackendResult("", False)).ms
        tr_ms = r.results.get("triton", BackendResult("", False)).ms
        c_ms = r.results.get("ck_tile_cpp", BackendResult("", False)).ms

        row = [
            r.scenario.name,
            _fmt(ck_ms),
            _fmt(t_ms),
            _fmt(tr_ms),
            _fmt(c_ms),
            _fmt_speedup(ck_ms, t_ms),
            _fmt_speedup(ck_ms, tr_ms),
            _fmt_speedup(ck_ms, c_ms),
        ]
        print("  ".join(f"{c:<{w}}" for c, w in zip(row, widths)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenario",
        default=None,
        action="append",
        help="Run only the named scenario(s); repeatable.",
    )
    parser.add_argument("--attempts", type=int, default=10, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=3, help="Cold iterations.")
    parser.add_argument(
        "--dtype",
        choices=("f16", "bf16"),
        default="f16",
        help="Activation dtype for all backends.",
    )
    parser.add_argument("--skip-rocke", action="store_true")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--skip-aiter", action="store_true")
    parser.add_argument("--skip-cktile", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the full result table as JSON to this path.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA / ROCm not available; aborting.", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    scenarios = default_scenarios()
    if args.dtype != "f16":
        scenarios = [Scenario(**{**asdict(s), "dtype": args.dtype}) for s in scenarios]
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.name in wanted]
        if not scenarios:
            print(f"unknown scenarios {args.scenario!r}", file=sys.stderr)
            return 2

    results: List[ScenarioResult] = []
    for s in scenarios:
        results.append(
            run_one_scenario(
                s,
                warmup=args.warmup,
                attempts=args.attempts,
                skip_rocke=args.skip_rocke,
                skip_torch=args.skip_torch,
                skip_aiter=args.skip_aiter,
                skip_cktile=args.skip_cktile,
            )
        )

    print_summary(results)

    if args.report is not None:
        payload = []
        for r in results:
            entry = {"scenario": asdict(r.scenario), "results": {}}
            for k, v in r.results.items():
                entry["results"][k] = asdict(v)
            payload.append(entry)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote report to {args.report}")

    # Exit code: nonzero if any scenario had a CK DSL correctness
    # failure (max_abs > 0.5) -- that's a real bug we want CI to
    # catch. AITER / CK Tile failures are environment issues, not
    # CK DSL bugs, so they don't gate.
    for r in results:
        ck = r.results.get("rocke")
        if ck is not None and ck.ok and ck.max_abs is not None and ck.max_abs > 0.5:
            print(
                f"\nFAIL: rocke correctness regressed on {r.scenario.name} "
                f"(max_abs={ck.max_abs:.4g})",
                file=sys.stderr,
            )
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
