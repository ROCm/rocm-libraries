# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end GPU verification of routing, compact quantization, and MegaMoE."""

from __future__ import annotations

import gc
import os

os.environ.setdefault("ROCKE_BACKEND", "python")


def main() -> int:
    import torch

    from rocke.examples.gfx950.fused_mega_moe.reproduce_levels import (
        _build_padded_activation,
        _compare,
        build_static_padded_inputs,
    )
    from rocke.examples.gfx950.moe.fused_moe_e2e_perf import (
        Scenario,
        make_inputs,
    )
    from rocke.examples.gfx950.moe.verify_fused_moe_situ import (
        _make_mxfp4_weights,
        _mxfp4_reference,
        _mxfp4_values,
    )
    from rocke.examples.gfx950.moe.verify_rank_reduce_and_routing import (
        _routing_reference,
    )
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.moe_compact_gather_quant import (
        MoeCompactGatherQuantSpec,
        build_moe_compact_gather_quant,
        moe_compact_gather_quant_grid,
        moe_compact_gather_quant_signature,
    )
    from rocke.instances.common.moe_fused_mega_fp8 import (
        FusedMegaKernelSpecFp8,
        build_moe_fused_mega_gemm_fp8,
        moe_fused_mega_fp8_grid,
        moe_fused_mega_fp8_signature,
    )
    from rocke.instances.common.moe_topk_active_pack import (
        MoeTopkActivePackSpec,
        build_moe_topk_active_pack,
        moe_topk_active_pack_grid,
        moe_topk_active_pack_signature,
    )
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if not torch.cuda.is_available():
        raise RuntimeError("a HIP-visible GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("this verification requires gfx950")

    hidden = 3584
    intermediate = 3072
    local_experts = 16
    global_experts = 896
    topk = 16
    tile_m = 16
    activation_beta = 2.0
    activation_linear_beta = 3.0

    mega_spec = FusedMegaKernelSpecFp8(
        name="rocke_fused_moe_reduction_pipeline",
        tile_m=tile_m,
        gate_up_k=32,
        down_k=32,
        use_dtla=False,
        activation="situ",
        activation_beta=activation_beta,
        activation_linear_beta=activation_linear_beta,
        weight_dtype="mxfp4",
    )
    mega_kernel = build_moe_fused_mega_gemm_fp8(mega_spec, "gfx950")
    mega_artifact = compile_kernel(mega_kernel, arch="gfx950", capture_ir_text=False)
    mega_launcher = KernelLauncher(
        hsaco=mega_artifact.hsaco,
        kernel_name=mega_artifact.kernel_name,
        signature=moe_fused_mega_fp8_signature(mega_spec),
        cache_key=("python-eval", mega_artifact.kernel_name),
    )

    with torch.no_grad():
        for tokens in (1, 8):
            scenario = Scenario(
                name=f"reduction_pipeline_T{tokens}",
                tokens=tokens,
                experts=local_experts,
                topk=topk,
                hidden=hidden,
                intermediate=intermediate,
            )
            inputs = make_inputs(scenario, seed=3301 + tokens)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(4409 + tokens)
            logits = torch.randn(
                tokens,
                global_experts,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            correction_bias = torch.full(
                (global_experts,), -8.0, dtype=torch.float32, device="cuda"
            )
            correction_bias[:local_experts] = 8.0
            expected_ids, expected_weights = _routing_reference(
                torch,
                logits,
                correction_bias,
                topk=topk,
                routed_scale=1.0,
            )
            if int(expected_ids.max().item()) >= local_experts:
                raise AssertionError("routing setup selected a non-local expert")
            inputs.topk_ids_pre = expected_ids
            inputs.topk_weights_pre = expected_weights

            routing_spec = MoeTopkActivePackSpec(
                tokens=tokens,
                experts=global_experts,
                topk=topk,
                tile_m=tile_m,
            )
            routing_kernel = build_moe_topk_active_pack(routing_spec, "gfx950")
            routing_artifact = compile_kernel(
                routing_kernel, arch="gfx950", capture_ir_text=False
            )
            routing_launcher = KernelLauncher(
                hsaco=routing_artifact.hsaco,
                kernel_name=routing_artifact.kernel_name,
                signature=moe_topk_active_pack_signature(routing_spec),
                cache_key=("python-eval", routing_artifact.kernel_name),
            )

            sorted_token_ids = torch.empty(
                routing_spec.max_padded_pairs,
                dtype=torch.int32,
                device="cuda",
            )
            sorted_topk_ids = torch.empty_like(sorted_token_ids)
            sorted_weights = torch.empty(
                routing_spec.max_padded_pairs,
                dtype=torch.float32,
                device="cuda",
            )
            block_expert_ids = torch.empty(
                routing_spec.max_blocks,
                dtype=torch.int32,
                device="cuda",
            )
            counts = torch.empty(global_experts, dtype=torch.int32, device="cuda")
            block_offsets = torch.empty_like(counts)
            num_blocks = torch.empty(1, dtype=torch.int32, device="cuda")
            routing_launcher(
                {
                    "Logits": logits,
                    "CorrectionBias": correction_bias,
                    "SortedTokenIds": sorted_token_ids,
                    "SortedTopkIds": sorted_topk_ids,
                    "SortedWeights": sorted_weights,
                    "BlockExpertIds": block_expert_ids,
                    "Counts": counts,
                    "BlockOffsets": block_offsets,
                    "NumBlocks": num_blocks,
                    "tokens": tokens,
                    "experts": global_experts,
                    "routed_scale": 1.0,
                },
                config=LaunchConfig(
                    grid=moe_topk_active_pack_grid(routing_spec),
                    block=(routing_spec.block_size, 1, 1),
                ),
            )

            gather_spec = MoeCompactGatherQuantSpec(
                tokens=tokens,
                hidden=hidden,
                max_blocks=routing_spec.max_blocks,
                tile_m=tile_m,
                input_dtype=scenario.dtype,
            )
            gather_kernel = build_moe_compact_gather_quant(gather_spec, "gfx950")
            gather_artifact = compile_kernel(
                gather_kernel, arch="gfx950", capture_ir_text=False
            )
            gather_launcher = KernelLauncher(
                hsaco=gather_artifact.hsaco,
                kernel_name=gather_artifact.kernel_name,
                signature=moe_compact_gather_quant_signature(gather_spec),
                cache_key=("python-eval", gather_artifact.kernel_name),
            )
            compact_activation = torch.empty(
                gather_spec.output_rows,
                hidden,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            compact_scale = torch.empty(
                gather_spec.output_rows,
                gather_spec.hidden_blocks,
                dtype=torch.float32,
                device="cuda",
            )
            gather_launcher(
                {
                    "X": inputs.X,
                    "SortedTokenIds": sorted_token_ids,
                    "BlockExpertIds": block_expert_ids,
                    "A": compact_activation,
                    "AScale": compact_scale,
                    "tokens": tokens,
                    "hidden": hidden,
                },
                config=LaunchConfig(
                    grid=moe_compact_gather_quant_grid(gather_spec),
                    block=(gather_spec.block_size, 1, 1),
                ),
            )
            torch.cuda.synchronize()

            host_padded = build_static_padded_inputs(inputs, scenario, tile_m=tile_m)
            fp8_stub = {
                "inputs": inputs,
                "X_f32": inputs.X.float(),
                "nHb": hidden // 128,
            }
            host_activation, host_scale = _build_padded_activation(
                fp8_stub, host_padded, scenario
            )
            active_rows = host_padded["total_padded"]
            scale_delta = (compact_scale[:active_rows] - host_scale).abs().max()
            if float(scale_delta.item()) > 1e-6:
                raise AssertionError(
                    f"compact scale mismatch for tokens={tokens}: "
                    f"{float(scale_delta.item()):.6g}"
                )
            device_dequant = compact_activation[:active_rows].float() * compact_scale[
                :active_rows
            ].repeat_interleave(128, dim=1)
            device_tokens = sorted_token_ids[:active_rows].to(torch.int64)
            valid_rows = device_tokens >= 0
            if not torch.allclose(
                device_dequant[valid_rows],
                inputs.X[device_tokens[valid_rows]].float(),
                atol=2e-2,
                rtol=2e-2,
            ):
                raise AssertionError(f"compact activation mismatch for tokens={tokens}")
            if not torch.all(device_dequant[~valid_rows] == 0):
                raise AssertionError(f"compact padding mismatch for tokens={tokens}")

            mx_weights = _make_mxfp4_weights(torch, fp8_stub)
            output = torch.zeros(tokens, hidden, dtype=torch.float32, device="cuda")
            device_padded = {
                "sorted_token_ids_padded": sorted_token_ids,
                "sorted_weights_padded": sorted_weights,
                "block_expert_ids": block_expert_ids,
                "total_padded": gather_spec.output_rows,
                "slot_size": tile_m,
            }
            mega_launcher(
                _mxfp4_values(
                    fp8_stub,
                    mx_weights,
                    scenario,
                    device_padded,
                    compact_activation,
                    compact_scale,
                    output,
                ),
                config=LaunchConfig(
                    grid=moe_fused_mega_fp8_grid(
                        routing_spec.max_blocks,
                        intermediate,
                        mega_spec,
                    ),
                    block=(mega_spec.block_size, 1, 1),
                ),
            )
            torch.cuda.synchronize()
            reference = _mxfp4_reference(
                torch,
                fp8_stub,
                mx_weights,
                host_padded,
                host_activation,
                host_scale,
                scenario,
                tile_m=tile_m,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
            )
            max_abs, _mean_abs, relative = _compare(output, reference)
            if relative >= 1.5e-2:
                raise AssertionError(
                    f"pipeline parity failed for tokens={tokens}: "
                    f"max_abs={max_abs:.6g}, relative={relative:.6g}"
                )
            print(
                f"PASS routing-to-MegaMoE pipeline tokens={tokens} "
                f"max_abs={max_abs:.6g} relative={relative:.6g}"
            )

            del (
                inputs,
                mx_weights,
                compact_activation,
                compact_scale,
                output,
                reference,
            )
            gc.collect()
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
