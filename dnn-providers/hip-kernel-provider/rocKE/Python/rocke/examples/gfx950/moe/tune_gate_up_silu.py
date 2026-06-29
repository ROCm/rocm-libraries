# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tune the experimental fused gate+up+silu MoE GEMM.

This script compares three gate/up activation-barrier variants

    packed gate+up batched GEMM (N=2*I) + packed silu_mul

against two true-fused kernels

    dual-B MFMA gate+up GEMM with SiLU epilogue
    interleaved single-B MFMA gate+up GEMM with SiLU epilogue

across a small grid of MFMA tile shapes. It is intentionally focused on
static-offset shapes (decode / small batch) where HIP graph capture is
valid and per-launch overhead is already amortized. The result tells us
which activation-barrier strategy is worth promoting for a shape.

Usage::

    PYTHONPATH=Python /path/to/venv/bin/python \\
        Python/rocke/examples/moe/tune_gate_up_silu.py \\
        --scenario small --attempts 10 --warmup 3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "Python"))

import torch  # noqa: E402

from rocke.instances import FusedMoeForward, FusedMoeForwardSpec  # noqa: E402
from rocke.instances.common.gemm_universal import TileSpec  # noqa: E402
from rocke.examples.gfx950.moe.fused_moe_e2e_perf import (  # noqa: E402
    Scenario,
    _compare,
    make_inputs,
    time_callable_ms,
    torch_fused_moe_reference,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    tile: TileSpec


def candidate_tiles() -> List[Candidate]:
    """Tile candidates that satisfy the universal GEMM validator.

    Keep this list small: every candidate compiles a fresh HSACO for the
    fused kernel and one for the batched down GEMM. These cover:

    * current MFMA-min default (16x128x64, 1x1 warp grid)
    * wider N tiles
    * more N warps
    * larger 32x32 MFMA atoms where legal
    """
    return [
        Candidate(
            "t16n128k64_w1x1_atom16",
            TileSpec(
                tile_m=16,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
        ),
        Candidate(
            "t16n256k64_w1x2_atom16",
            TileSpec(
                tile_m=16,
                tile_n=256,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
        ),
        Candidate(
            "t32n128k64_w2x1_atom16",
            TileSpec(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=2,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
        ),
        Candidate(
            "t32n256k64_w2x2_atom16",
            TileSpec(
                tile_m=32,
                tile_n=256,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
        ),
        Candidate(
            "t32n128k64_w1x2_atom32",
            TileSpec(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
        ),
    ]


def scenarios() -> dict[str, Scenario]:
    return {
        "small": Scenario(
            name="small_T32_E4_K2_H128_I256",
            tokens=32,
            experts=4,
            topk=2,
            hidden=128,
            intermediate=256,
        ),
        "decode1": Scenario(
            name="decode_T1_E8_K2_H4096_I7168",
            tokens=1,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=7168,
        ),
        "decode8": Scenario(
            name="decode_T8_E8_K2_H4096_I7168",
            tokens=8,
            experts=8,
            topk=2,
            hidden=4096,
            intermediate=7168,
        ),
    }


def run_forward_ms(
    spec: FusedMoeForwardSpec,
    inputs,
    *,
    warmup: int,
    attempts: int,
) -> Tuple[float, float, str]:
    fwd = FusedMoeForward(spec)
    fwd._ensure_compiled()
    Y = torch.zeros(spec.tokens, spec.hidden, dtype=torch.float16, device="cuda")
    Y_ref = torch_fused_moe_reference(
        inputs,
        Scenario(
            name=spec.name,
            tokens=spec.tokens,
            experts=spec.experts,
            topk=spec.topk,
            hidden=spec.hidden,
            intermediate=spec.intermediate,
            dtype=spec.dtype,
        ),
    )

    # Use graph replay whenever static mode is enabled. This keeps the
    # comparison focused on kernel work, not Python launch overhead.
    if getattr(fwd, "_use_static_offsets", False):
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
        fwd.replay_graph()
        torch.cuda.synchronize()
    else:

        def call():
            fwd.forward(
                routing_logits=inputs.routing_logits,
                X=inputs.X,
                W_gate=inputs.W_gate,
                W_up=inputs.W_up,
                W_down=inputs.W_down,
                Y=Y,
            )

        ms = time_callable_ms(call, warmup=warmup, attempts=attempts)
    max_abs, _mean_abs, rel = _compare(Y, Y_ref)
    return ms, max_abs, f"rel={rel:.4g}"


def spec_for(
    s: Scenario, *, tile: TileSpec, path: str, name: str
) -> FusedMoeForwardSpec:
    return FusedMoeForwardSpec(
        tokens=s.tokens,
        experts=s.experts,
        topk=s.topk,
        hidden=s.hidden,
        intermediate=s.intermediate,
        dtype=s.dtype,
        streaming_block_size=64,
        streaming_vec=8,
        sort_block_size=max(64, s.experts),
        router_block_size=max(64, s.experts),
        gemm_tile=tile,
        name=f"rocke_tune_{name}",
        use_experimental_fused_gate_up_silu=(path == "dual"),
        use_experimental_interleaved_gate_up_silu=(path == "interleaved"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(scenarios()), default="small")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA / ROCm not available", file=sys.stderr)
        return 1
    s = scenarios()[args.scenario]
    inputs = make_inputs(s)

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"scenario: {s.name}")
    print()
    print(f"{'candidate':<30} {'path':<12} {'ms':>10} {'max_abs':>12} {'note':<20}")
    print("-" * 92)

    rows = []
    for cand in candidate_tiles():
        for path in ("packed", "dual", "interleaved"):
            try:
                spec = spec_for(
                    s,
                    tile=cand.tile,
                    path=path,
                    name=cand.name + "_" + path,
                )
                ms, max_abs, note = run_forward_ms(
                    spec, inputs, warmup=args.warmup, attempts=args.attempts
                )
                rows.append((ms, cand.name, path, max_abs, note))
                print(
                    f"{cand.name:<30} {path:<12} {ms:10.4f} {max_abs:12.4g} {note:<20}"
                )
            except Exception as exc:
                print(
                    f"{cand.name:<30} {path:<12} {'FAIL':>10} {'-':>12} "
                    f"{type(exc).__name__}: {exc}"
                )
    if rows:
        best = min(rows, key=lambda r: r[0])
        print()
        print(
            f"best: {best[1]} / {best[2]}  {best[0]:.4f} ms  "
            f"max_abs={best[3]:.4g}  {best[4]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
