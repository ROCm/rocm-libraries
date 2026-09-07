# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU numeric verification for rank-staged reduction and routing kernels.

The harness checks both small decode row counts. It intentionally runs through
the Python builder/lowerer so the operation contracts can stabilize before the
C engine emitter is updated.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("ROCKE_BACKEND", "python")


def _launcher(kernel, signature, *, arch: str):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    artifact = compile_kernel(kernel, arch=arch, capture_ir_text=False)
    return KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=signature,
        cache_key=("python-eval", artifact.kernel_name),
    )


def _assert_close(torch, actual, expected, *, label: str, atol: float, rtol: float):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        delta = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{label}: mismatch max_abs={delta.max().item():.6g}, "
            f"mean_abs={delta.mean().item():.6g}"
        )


def verify_rank_reduce(torch, *, rows: int, arch: str) -> None:
    from rocke.instances.common.moe_rank_reduce import (
        MoeRankReduceRMSNormSpec,
        MoeRankReduceScatterSpec,
        build_moe_rank_reduce_rmsnorm,
        build_moe_rank_reduce_scatter,
        moe_rank_reduce_rmsnorm_grid,
        moe_rank_reduce_rmsnorm_signature,
        moe_rank_reduce_scatter_grid,
        moe_rank_reduce_scatter_signature,
    )
    from rocke.runtime.launcher import LaunchConfig

    world_size = 8
    rms_width = 3584
    scatter_width = 7168
    epsilon = 1e-6
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1409 + rows)

    partials = (
        torch.randn(
            world_size,
            rows,
            rms_width,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.1
    ).to(torch.bfloat16)
    gamma = (
        torch.randn(
            rms_width,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.1
        + 1.0
    ).to(torch.bfloat16)
    output = torch.empty(rows, rms_width, dtype=torch.bfloat16, device="cuda")
    rms_spec = MoeRankReduceRMSNormSpec(width=rms_width, world_size=world_size)
    rms_kernel = build_moe_rank_reduce_rmsnorm(rms_spec, arch)
    rms_launcher = _launcher(
        rms_kernel, moe_rank_reduce_rmsnorm_signature(rms_spec), arch=arch
    )
    rms_launcher(
        {
            "Partials": partials,
            "Gamma": gamma,
            "Y": output,
            "rows": rows,
            "width": rms_width,
            "eps": epsilon,
        },
        config=LaunchConfig(
            grid=moe_rank_reduce_rmsnorm_grid(rows, rms_spec),
            block=(rms_spec.block_size, 1, 1),
        ),
    )
    reduced = partials.float().sum(dim=0).to(torch.bfloat16).float()
    reference = (
        reduced
        * torch.rsqrt(reduced.square().mean(dim=-1, keepdim=True) + epsilon)
        * gamma.float()
    ).to(torch.bfloat16)
    torch.cuda.synchronize()
    _assert_close(
        torch,
        output,
        reference,
        label=f"rank-reduce-rmsnorm rows={rows}",
        atol=2e-2,
        rtol=2e-2,
    )

    partials = (
        torch.randn(
            world_size,
            rows,
            scatter_width,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.1
    ).to(torch.bfloat16)
    scatter_spec = MoeRankReduceScatterSpec(width=scatter_width, world_size=world_size)
    scatter_kernel = build_moe_rank_reduce_scatter(scatter_spec, arch)
    scatter_launcher = _launcher(
        scatter_kernel, moe_rank_reduce_scatter_signature(scatter_spec), arch=arch
    )
    full_reference = partials.float().sum(dim=0)
    for rank in (0, 3, world_size - 1):
        output = torch.empty(
            rows, scatter_spec.shard_width, dtype=torch.bfloat16, device="cuda"
        )
        scatter_launcher(
            {
                "Partials": partials,
                "Y": output,
                "rows": rows,
                "width": scatter_width,
                "rank": rank,
            },
            config=LaunchConfig(
                grid=moe_rank_reduce_scatter_grid(rows, scatter_spec),
                block=(scatter_spec.block_size, 1, 1),
            ),
        )
        begin = rank * scatter_spec.shard_width
        expected = full_reference[:, begin : begin + scatter_spec.shard_width].to(
            torch.bfloat16
        )
        torch.cuda.synchronize()
        _assert_close(
            torch,
            output,
            expected,
            label=f"rank-reduce-scatter rows={rows} rank={rank}",
            atol=2e-2,
            rtol=2e-2,
        )


def _routing_reference(torch, logits, bias, *, topk: int, routed_scale: float):
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + bias, k=topk, dim=-1, sorted=True).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids.to(torch.int32), weights * routed_scale


def verify_routing(torch, *, tokens: int, arch: str) -> None:
    from rocke.instances.common.moe_topk_active_pack import (
        MoeTopkActivePackSpec,
        build_moe_topk_active_pack,
        moe_topk_active_pack_grid,
        moe_topk_active_pack_signature,
    )
    from rocke.runtime.launcher import LaunchConfig

    experts = 896
    topk = 16
    routed_scale = 1.25
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2903 + tokens)
    logits = torch.randn(
        tokens,
        experts,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    bias = (
        torch.randn(
            experts,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.01
    )
    spec = MoeTopkActivePackSpec(tokens=tokens, experts=experts, topk=topk)
    sorted_token_ids = torch.empty(
        spec.max_padded_pairs, dtype=torch.int32, device="cuda"
    )
    sorted_topk_ids = torch.empty_like(sorted_token_ids)
    sorted_weights = torch.empty(
        spec.max_padded_pairs, dtype=torch.float32, device="cuda"
    )
    block_expert_ids = torch.empty(spec.max_blocks, dtype=torch.int32, device="cuda")
    counts = torch.empty(experts, dtype=torch.int32, device="cuda")
    block_offsets = torch.empty_like(counts)
    num_blocks = torch.empty(1, dtype=torch.int32, device="cuda")

    kernel = build_moe_topk_active_pack(spec, arch)
    launcher = _launcher(kernel, moe_topk_active_pack_signature(spec), arch=arch)
    launcher(
        {
            "Logits": logits,
            "CorrectionBias": bias,
            "SortedTokenIds": sorted_token_ids,
            "SortedTopkIds": sorted_topk_ids,
            "SortedWeights": sorted_weights,
            "BlockExpertIds": block_expert_ids,
            "Counts": counts,
            "BlockOffsets": block_offsets,
            "NumBlocks": num_blocks,
            "tokens": tokens,
            "experts": experts,
            "routed_scale": routed_scale,
        },
        config=LaunchConfig(
            grid=moe_topk_active_pack_grid(spec),
            block=(spec.block_size, 1, 1),
        ),
    )
    torch.cuda.synchronize()

    expected_ids, expected_weights = _routing_reference(
        torch, logits, bias, topk=topk, routed_scale=routed_scale
    )
    expected_counts = torch.bincount(
        expected_ids.flatten().to(torch.int64), minlength=experts
    ).to(torch.int32)
    expected_blocks = torch.div(
        expected_counts + spec.tile_m - 1,
        spec.tile_m,
        rounding_mode="floor",
    )
    expected_offsets = torch.cumsum(expected_blocks, dim=0) - expected_blocks
    expected_num_blocks = int(expected_blocks.sum().item())

    if not torch.equal(counts, expected_counts):
        raise AssertionError(f"routing counts mismatch for tokens={tokens}")
    if not torch.equal(block_offsets, expected_offsets):
        raise AssertionError(f"routing block offsets mismatch for tokens={tokens}")
    if int(num_blocks.item()) != expected_num_blocks:
        raise AssertionError(
            f"routing NumBlocks mismatch for tokens={tokens}: "
            f"{int(num_blocks.item())} != {expected_num_blocks}"
        )
    expected_block_experts = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device="cuda"),
        expected_blocks.to(torch.int64),
    )
    if not torch.equal(block_expert_ids[:expected_num_blocks], expected_block_experts):
        raise AssertionError(f"BlockExpertIds mismatch for tokens={tokens}")
    if not torch.all(block_expert_ids[expected_num_blocks:] == -1):
        raise AssertionError(f"BlockExpertIds tail is not sentinel for tokens={tokens}")

    seen = torch.zeros(tokens, topk, dtype=torch.bool, device="cuda")
    for expert in range(experts):
        count = int(expected_counts[expert].item())
        if count == 0:
            continue
        begin = int(expected_offsets[expert].item()) * spec.tile_m
        token_slice = sorted_token_ids[begin : begin + count].to(torch.int64)
        slot_slice = sorted_topk_ids[begin : begin + count].to(torch.int64)
        weight_slice = sorted_weights[begin : begin + count]
        if not torch.all(expected_ids[token_slice, slot_slice] == expert):
            raise AssertionError(
                f"expert metadata mismatch for tokens={tokens}, expert={expert}"
            )
        expected = expected_weights[token_slice, slot_slice]
        _assert_close(
            torch,
            weight_slice,
            expected,
            label=f"routing weights tokens={tokens} expert={expert}",
            atol=2e-6,
            rtol=2e-6,
        )
        seen[token_slice, slot_slice] = True
        padded_end = (
            int(expected_offsets[expert].item()) + int(expected_blocks[expert].item())
        ) * spec.tile_m
        if not torch.all(sorted_token_ids[begin + count : padded_end] == -1):
            raise AssertionError(
                f"padding token sentinel mismatch for tokens={tokens}, expert={expert}"
            )
        if not torch.all(sorted_weights[begin + count : padded_end] == 0):
            raise AssertionError(
                f"padding weight mismatch for tokens={tokens}, expert={expert}"
            )
    if not torch.all(seen):
        raise AssertionError(f"not every routed pair was packed for tokens={tokens}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--skip-reduce", action="store_true")
    parser.add_argument("--skip-routing", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("a HIP-visible GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if args.arch not in properties.gcnArchName:
        raise RuntimeError(
            f"requested {args.arch}, running device reports {properties.gcnArchName}"
        )

    for rows in args.rows:
        if not args.skip_reduce:
            verify_rank_reduce(torch, rows=rows, arch=args.arch)
            print(f"PASS rank reductions rows={rows}")
        if not args.skip_routing:
            verify_routing(torch, tokens=rows, arch=args.arch)
            print(f"PASS routing active-pack tokens={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
