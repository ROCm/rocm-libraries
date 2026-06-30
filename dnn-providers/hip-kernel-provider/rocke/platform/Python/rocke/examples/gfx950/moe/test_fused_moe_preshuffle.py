# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end parity + perf for ``preshuffle_w_down`` on FusedMoeForward.

Builds two :class:`FusedMoeForward` instances (preshuffle_w_down =
False / True) on the same problem shape and weights, replays each
under HIP-graph capture, and reports parity vs the ``False`` baseline
plus end-to-end perf delta.

Run with::

    cd <repo>/dnn-providers/hip-kernel-provider/rocKE/Python
    python rocke/examples/moe/test_fused_moe_preshuffle.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if "rocke" not in sys.modules:
    HERE = Path(__file__).resolve()
    pkg_root = HERE.parents[4]
    sys.path.insert(0, str(pkg_root))

import torch  # noqa: E402

from rocke.instances.common.fused_moe_e2e import (  # noqa: E402
    FusedMoeForward,
    FusedMoeForwardSpec,
)


@dataclass
class Scenario:
    name: str
    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int
    dtype: str = "f16"


def _torch_dtype_for(dtype: str) -> torch.dtype:
    return {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[dtype]


def make_inputs(s: Scenario, *, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dtype = _torch_dtype_for(s.dtype)
    routing_logits = torch.randn(
        s.tokens, s.experts, generator=g, device="cuda", dtype=torch.float32
    )
    X = torch.randn(s.tokens, s.hidden, generator=g, device="cuda", dtype=dtype)
    # Scaled-down init -- production weights are tiny; gemm output
    # then has ~unit scale and the f16 reduction noise stays
    # comparable across the K-axis for the parity check.
    Wgate = (
        torch.randn(
            s.experts,
            s.intermediate,
            s.hidden,
            generator=g,
            device="cuda",
            dtype=dtype,
        )
        / s.hidden**0.5
    )
    Wup = (
        torch.randn(
            s.experts,
            s.intermediate,
            s.hidden,
            generator=g,
            device="cuda",
            dtype=dtype,
        )
        / s.hidden**0.5
    )
    Wdown = (
        torch.randn(
            s.experts,
            s.hidden,
            s.intermediate,
            generator=g,
            device="cuda",
            dtype=dtype,
        )
        / s.intermediate**0.5
    )
    return routing_logits, X, Wgate, Wup, Wdown


def time_replay_ms(replay_fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        replay_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        replay_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _isolate_lane() -> None:
    import gc

    torch.cuda.synchronize()
    try:
        from rocke.runtime.launcher import synchronize_and_release

        synchronize_and_release()
    except Exception:
        pass
    gc.collect()


def run_one(
    s: Scenario,
    *,
    preshuffle_w_down: bool,
    preshuffle_w_gate_up_packed: bool = False,
    preshuffle_w_gate_up_interleaved: bool = False,
    active_tile_skip_gemms: bool = False,
    use_interleaved: bool = True,
    warmup: int,
    iters: int,
):
    streaming_block_size = min(256, s.hidden)
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
        preshuffle_w_down=preshuffle_w_down,
        preshuffle_w_gate_up_packed=preshuffle_w_gate_up_packed,
        preshuffle_w_gate_up_interleaved=preshuffle_w_gate_up_interleaved,
        active_tile_skip_gemms=active_tile_skip_gemms,
        use_experimental_interleaved_gate_up_silu=use_interleaved,
    )
    fwd = FusedMoeForward(spec)
    fwd._ensure_compiled()

    routing_logits, X, Wgate, Wup, Wdown = make_inputs(s)
    Y = torch.zeros(s.tokens, s.hidden, dtype=_torch_dtype_for(s.dtype), device="cuda")
    fwd.capture_graph(
        routing_logits=routing_logits,
        X=X,
        W_gate=Wgate,
        W_up=Wup,
        W_down=Wdown,
        Y=Y,
        warmup_iters=2,
    )
    ms = time_replay_ms(fwd.replay_graph, warmup, iters)
    fwd.replay_graph()
    torch.cuda.synchronize()
    out = Y.clone()
    # Drop the captured graph + pool before releasing the FusedMoeForward
    # so the next config's instance starts on a clean stream / cache.
    del fwd
    _isolate_lane()
    return out, ms


def run_scenario(s: Scenario, warmup: int, iters: int) -> int:
    print(
        f"\n=== {s.name}  tokens={s.tokens} E={s.experts} K={s.topk} "
        f"H={s.hidden} I={s.intermediate} ==="
    )
    # Configurations:
    #   1. baseline_interleaved: production default (no preshuffle).
    #   2. baseline_packed: packed gate-up (slower per docs).
    #   3. preshuf_down_only: only down GEMM preshuffled.
    #   4. preshuf_packed: packed gate-up + down both preshuffled.
    Y_base, base_ms = run_one(
        s, preshuffle_w_down=False, use_interleaved=True, warmup=warmup, iters=iters
    )
    Y_packed, packed_ms = run_one(
        s, preshuffle_w_down=False, use_interleaved=False, warmup=warmup, iters=iters
    )
    Y_pre_d, pre_d_ms = run_one(
        s, preshuffle_w_down=True, use_interleaved=True, warmup=warmup, iters=iters
    )
    Y_pre_full, pre_full_ms = run_one(
        s,
        preshuffle_w_down=True,
        preshuffle_w_gate_up_packed=True,
        use_interleaved=False,
        warmup=warmup,
        iters=iters,
    )
    Y_pre_intl, pre_intl_ms = run_one(
        s,
        preshuffle_w_down=True,
        preshuffle_w_gate_up_interleaved=True,
        use_interleaved=True,
        warmup=warmup,
        iters=iters,
    )
    Y_ats, ats_ms = run_one(
        s,
        preshuffle_w_down=False,
        active_tile_skip_gemms=True,
        use_interleaved=True,
        warmup=warmup,
        iters=iters,
    )
    Y_full, full_ms = run_one(
        s,
        preshuffle_w_down=True,
        preshuffle_w_gate_up_interleaved=True,
        active_tile_skip_gemms=True,
        use_interleaved=True,
        warmup=warmup,
        iters=iters,
    )

    def _drel(Y):
        return (Y - Y_base).abs().max().cpu().item() / max(
            Y_base.abs().max().cpu().item(), 1e-6
        )

    print(f"  baseline_interleaved      = {base_ms * 1000:8.2f} us  (1.000x)")
    print(
        f"  baseline_packed           = {packed_ms * 1000:8.2f} us  ({base_ms / packed_ms:.3f}x)  rel={_drel(Y_packed):.2e}"
    )
    print(
        f"  preshuf_down_only         = {pre_d_ms * 1000:8.2f} us  ({base_ms / pre_d_ms:.3f}x)  rel={_drel(Y_pre_d):.2e}"
    )
    print(
        f"  preshuf_packed_full       = {pre_full_ms * 1000:8.2f} us  ({base_ms / pre_full_ms:.3f}x)  rel={_drel(Y_pre_full):.2e}"
    )
    print(
        f"  preshuf_intl_full         = {pre_intl_ms * 1000:8.2f} us  ({base_ms / pre_intl_ms:.3f}x)  rel={_drel(Y_pre_intl):.2e}"
    )
    print(
        f"  active_tile_only          = {ats_ms * 1000:8.2f} us  ({base_ms / ats_ms:.3f}x)  rel={_drel(Y_ats):.2e}"
    )
    print(
        f"  preshuf_intl + ats_full   = {full_ms * 1000:8.2f} us  ({base_ms / full_ms:.3f}x)  rel={_drel(Y_full):.2e}"
    )
    rels = [
        _drel(Y_packed),
        _drel(Y_pre_d),
        _drel(Y_pre_full),
        _drel(Y_pre_intl),
        _drel(Y_ats),
        _drel(Y_full),
    ]
    if max(rels) > 5e-2:
        print("  [FAIL] one variant drifted from baseline")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    scenarios = [
        Scenario("small", tokens=4, experts=4, topk=2, hidden=128, intermediate=512),
        Scenario(
            "decode_T1_S", tokens=1, experts=8, topk=2, hidden=1024, intermediate=2048
        ),
        Scenario(
            "decode_T8_I4096",
            tokens=8,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=4096,
        ),
        Scenario(
            "decode_T1_I4096",
            tokens=1,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=4096,
        ),
        Scenario(
            "decode_T1_I7168",
            tokens=1,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=7168,
        ),
        Scenario(
            "decode_T8_I7168",
            tokens=8,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=7168,
        ),
    ]
    rc = 0
    for s in scenarios:
        rc |= run_scenario(s, args.warmup, args.iters)
    return rc


if __name__ == "__main__":
    sys.exit(main())
