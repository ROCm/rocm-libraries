# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU numeric verification for the alternate FP8 fused-MoE activation."""

from __future__ import annotations

import os

os.environ.setdefault("ROCKE_BACKEND", "python")


def _scenario(tokens: int):
    from rocke.examples.gfx950.moe.fused_moe_e2e_perf import Scenario

    return Scenario(
        name=f"alternate_activation_T{tokens}",
        tokens=tokens,
        experts=2,
        topk=2,
        hidden=3584,
        intermediate=3072,
    )


def _quantize_mxfp4(torch, weights):
    """Pack row-major E2M1 weights and return E8M0 scales + f32 oracle."""

    original_shape = weights.shape
    contraction = original_shape[-1]
    if contraction % 32:
        raise ValueError("MXFP4 contraction must be divisible by 32")
    grouped = weights.float().reshape(*original_shape[:-1], contraction // 32, 32)
    amax = grouped.abs().amax(dim=-1)
    nonzero = amax > 0
    exponent = torch.ceil(torch.log2(torch.clamp(amax / 6.0, min=2.0**-126)))
    exponent = torch.clamp(exponent, -126, 127)
    encoded = torch.where(
        nonzero,
        (exponent + 127).to(torch.int32),
        torch.zeros_like(exponent, dtype=torch.int32),
    ).to(torch.uint8)
    scale = torch.where(nonzero, torch.exp2(exponent), torch.zeros_like(exponent))
    normalized = torch.where(
        nonzero.unsqueeze(-1),
        grouped.abs() / scale.unsqueeze(-1),
        torch.zeros_like(grouped),
    )
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=weights.device,
    )
    magnitude_code = torch.bucketize(normalized, boundaries).to(torch.uint8)
    sign = ((grouped < 0) & (magnitude_code != 0)).to(torch.uint8) << 3
    code = magnitude_code | sign
    flat_code = code.reshape(*original_shape)
    packed = (flat_code[..., 0::2] | (flat_code[..., 1::2] << 4)).contiguous()

    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=weights.device,
    )
    dequant = levels[magnitude_code.to(torch.int64)]
    dequant = torch.where((code & 0x8) != 0, -dequant, dequant)
    dequant = (dequant * scale.unsqueeze(-1)).reshape(*original_shape)
    return packed.view(torch.int8), encoded.contiguous().view(torch.int8), dequant


def _make_mxfp4_weights(torch, fp8_inputs):
    inputs = fp8_inputs["inputs"]
    gate, gate_scale, gate_dequant = _quantize_mxfp4(torch, inputs.W_gate)
    up, up_scale, up_dequant = _quantize_mxfp4(torch, inputs.W_up)
    down, down_scale, down_dequant = _quantize_mxfp4(torch, inputs.W_down)
    return {
        "gate": gate,
        "up": up,
        "down": down,
        "gate_scale": gate_scale,
        "up_scale": up_scale,
        "down_scale": down_scale,
        "gate_dequant": gate_dequant,
        "up_dequant": up_dequant,
        "down_dequant": down_dequant,
    }


def _preshuffle_mxfp4_tensor(tensor):
    experts, rows, packed_k = tensor.shape
    if rows % 16 or packed_k % 64:
        raise ValueError("native MXFP4 preshuffle requires N%16=0 and Kbytes%64=0")
    return (
        tensor.view(experts, rows // 16, 16, packed_k // 64, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view_as(tensor)
    )


def _preshuffle_mxfp4_scale(tensor):
    experts, rows, groups = tensor.shape
    if rows % 32 or groups % 8:
        raise ValueError("native MXFP4 scale preshuffle requires N%32=0 and G%8=0")
    return (
        tensor.view(experts, rows // 32, 2, 16, groups // 8, 2, 4)
        .permute(0, 1, 4, 6, 3, 5, 2)
        .contiguous()
        .view_as(tensor)
    )


def _preshuffle_mxfp4_weights(mx_weights):
    result = dict(mx_weights)
    for name in ("gate", "up", "down"):
        result[name] = _preshuffle_mxfp4_tensor(mx_weights[name])
    for name in ("gate_scale", "up_scale", "down_scale"):
        result[name] = _preshuffle_mxfp4_scale(mx_weights[name])
    return result


def _mxfp4_values(
    fp8_inputs,
    mx_weights,
    scenario,
    padded,
    activation,
    activation_scale,
    output,
):
    hidden = scenario.hidden
    intermediate = scenario.intermediate
    return {
        "A": activation,
        "WGate": mx_weights["gate"],
        "WUp": mx_weights["up"],
        "WDown": mx_weights["down"],
        "AScale": activation_scale,
        "WGateScale": mx_weights["gate_scale"],
        "WUpScale": mx_weights["up_scale"],
        "WDownScale": mx_weights["down_scale"],
        "SortedTokenIds": padded["sorted_token_ids_padded"],
        "SortedWeights": padded["sorted_weights_padded"],
        "BlockExpertIds": padded["block_expert_ids"],
        "Y": output,
        "M": padded["total_padded"],
        "N": intermediate,
        "K": hidden,
        "H_out": hidden,
        "stride_a": hidden,
        "stride_b_gate": intermediate * (hidden // 2),
        "stride_b_up": intermediate * (hidden // 2),
        "stride_b_down": hidden * (intermediate // 2),
        "stride_a_scale": hidden // 128,
        "stride_gate_scale": hidden // 32,
        "stride_up_scale": hidden // 32,
        "stride_down_scale": intermediate // 32,
        "stride_gate_scale_e": intermediate * (hidden // 32),
        "stride_up_scale_e": intermediate * (hidden // 32),
        "stride_down_scale_e": hidden * (intermediate // 32),
        "slot_size": padded["slot_size"],
        "tokens": scenario.tokens,
        "MxScaleA": 127,
    }


def _mxfp4_reference(
    torch,
    fp8_inputs,
    mx_weights,
    padded,
    activation,
    activation_scale,
    scenario,
    *,
    tile_m: int,
    activation_beta: float,
    activation_linear_beta: float,
):
    inputs = fp8_inputs["inputs"]
    top_ids = inputs.topk_ids_pre
    top_weights = inputs.topk_weights_pre
    tokens, hidden = inputs.X.shape
    intermediate = scenario.intermediate
    output = torch.zeros(tokens, hidden, dtype=torch.float32, device=inputs.X.device)
    hidden_blocks = hidden // 128
    inter_blocks = intermediate // 128

    for expert in range(scenario.experts):
        token_idx, slot_idx = (top_ids == expert).nonzero(as_tuple=True)
        count = int(token_idx.numel())
        if count == 0:
            continue
        base = padded["expert_base"][expert]
        activation_dequant = activation[base : base + count].float() * activation_scale[
            base : base + count
        ].repeat_interleave(128, dim=1)
        gate = activation_dequant @ mx_weights["gate_dequant"][expert].T
        up = activation_dequant @ mx_weights["up_dequant"][expert].T
        gate = (
            activation_beta * torch.tanh(gate / activation_beta) * torch.sigmoid(gate)
        )
        up = activation_linear_beta * torch.tanh(up / activation_linear_beta)
        gated = gate * up

        expert_output = torch.empty(
            count, hidden, dtype=torch.float32, device=inputs.X.device
        )
        for block_start in range(0, count, tile_m):
            block_end = min(block_start + tile_m, count)
            block = gated[block_start:block_end]
            block_amax = torch.clamp(
                block.reshape(block.shape[0], inter_blocks, 128).abs().amax(dim=(0, 2)),
                min=1e-6,
            )
            block_scale = block_amax / 448.0
            quantized = torch.clamp(
                block / block_scale.repeat_interleave(128),
                -448.0,
                448.0,
            ).to(torch.float8_e4m3fn)
            dequantized = quantized.float() * block_scale.repeat_interleave(128)
            expert_output[block_start:block_end] = (
                dequantized @ mx_weights["down_dequant"][expert].T
            )
        weight = top_weights[token_idx, slot_idx].unsqueeze(-1)
        output.index_add_(0, token_idx, weight * expert_output)

    _ = hidden_blocks
    return output


def main() -> int:
    import torch

    from rocke.examples.gfx950.fused_mega_moe.reproduce_levels import (
        _build_padded_activation,
        _compare,
        _mega_values,
        build_static_padded_inputs,
        make_fp8_inputs,
        torch_fused_moe_fp8_reference,
    )
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.moe_fused_mega_fp8 import (
        FusedMegaKernelSpecFp8,
        build_moe_fused_mega_gemm_fp8,
        moe_fused_mega_fp8_grid,
        moe_fused_mega_fp8_signature,
    )
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if not torch.cuda.is_available():
        raise RuntimeError("a HIP-visible GPU is required")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("this verification requires gfx950")

    activation_beta = 2.0
    activation_linear_beta = 3.0
    spec = FusedMegaKernelSpecFp8(
        name="rocke_fused_moe_mega_fp8_alternate_activation",
        tile_m=16,
        activation="situ",
        activation_beta=activation_beta,
        activation_linear_beta=activation_linear_beta,
    )
    kernel = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
    artifact = compile_kernel(kernel, arch="gfx950", capture_ir_text=False)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=moe_fused_mega_fp8_signature(spec),
        cache_key=("python-eval", artifact.kernel_name),
    )
    mx_spec = FusedMegaKernelSpecFp8(
        name="rocke_fused_moe_mega_mxfp4_alternate_activation",
        tile_m=16,
        gate_up_k=128,
        down_k=128,
        use_dtla=False,
        warp_n=8,
        activation="situ",
        activation_beta=activation_beta,
        activation_linear_beta=activation_linear_beta,
        weight_dtype="mxfp4",
        mxfp4_native=True,
        prefetch_routing_meta=True,
        pipeline_native_down=True,
        mxfp4_preshuffled=True,
        pipeline_native_gateup=True,
    )
    mx_kernel = build_moe_fused_mega_gemm_fp8(mx_spec, arch="gfx950")
    mx_artifact = compile_kernel(mx_kernel, arch="gfx950", capture_ir_text=False)
    mx_launcher = KernelLauncher(
        hsaco=mx_artifact.hsaco,
        kernel_name=mx_artifact.kernel_name,
        signature=moe_fused_mega_fp8_signature(mx_spec),
        cache_key=("python-eval", mx_artifact.kernel_name),
    )

    with torch.no_grad():
        for tokens in (1, 8):
            scenario = _scenario(tokens)
            fp8_inputs = make_fp8_inputs(scenario, seed=1709 + tokens)
            padded = build_static_padded_inputs(
                fp8_inputs["inputs"], scenario, tile_m=spec.tile_m
            )
            activation, activation_scale = _build_padded_activation(
                fp8_inputs, padded, scenario
            )
            output = torch.zeros(
                tokens, scenario.hidden, dtype=torch.float32, device="cuda"
            )
            grid = moe_fused_mega_fp8_grid(
                padded["num_m_blocks"], scenario.intermediate, spec
            )
            launcher(
                _mega_values(
                    fp8_inputs,
                    scenario,
                    padded,
                    activation,
                    activation_scale,
                    output,
                    None,
                ),
                config=LaunchConfig(
                    grid=grid,
                    block=(spec.block_size, 1, 1),
                ),
            )
            torch.cuda.synchronize()
            reference = torch_fused_moe_fp8_reference(
                fp8_inputs,
                padded,
                activation,
                activation_scale,
                scenario,
                tile_m=spec.tile_m,
                activation=spec.activation,
                activation_beta=spec.activation_beta,
                activation_linear_beta=spec.activation_linear_beta,
            )
            max_abs, _mean_abs, relative = _compare(output, reference)
            if relative >= 1.5e-2:
                raise AssertionError(
                    f"alternate activation parity failed for tokens={tokens}: "
                    f"max_abs={max_abs:.6g}, relative={relative:.6g}"
                )
            print(
                f"PASS fused alternate activation tokens={tokens} "
                f"max_abs={max_abs:.6g} relative={relative:.6g}"
            )

            mx_weights = _make_mxfp4_weights(torch, fp8_inputs)
            mx_weights = _preshuffle_mxfp4_weights(mx_weights)
            mx_output = torch.zeros_like(output)
            mx_grid = moe_fused_mega_fp8_grid(
                padded["num_m_blocks"], scenario.intermediate, mx_spec
            )
            mx_launcher(
                _mxfp4_values(
                    fp8_inputs,
                    mx_weights,
                    scenario,
                    padded,
                    activation,
                    activation_scale,
                    mx_output,
                ),
                config=LaunchConfig(
                    grid=mx_grid,
                    block=(mx_spec.block_size, 1, 1),
                ),
            )
            torch.cuda.synchronize()
            mx_reference = _mxfp4_reference(
                torch,
                fp8_inputs,
                mx_weights,
                padded,
                activation,
                activation_scale,
                scenario,
                tile_m=mx_spec.tile_m,
                activation_beta=mx_spec.activation_beta,
                activation_linear_beta=mx_spec.activation_linear_beta,
            )
            mx_max_abs, _mx_mean_abs, mx_relative = _compare(mx_output, mx_reference)
            if mx_relative >= 1.5e-2:
                raise AssertionError(
                    f"MXFP4 alternate activation parity failed for "
                    f"tokens={tokens}: max_abs={mx_max_abs:.6g}, "
                    f"relative={mx_relative:.6g}"
                )
            print(
                f"PASS fused MXFP4 alternate activation tokens={tokens} "
                f"max_abs={mx_max_abs:.6g} relative={mx_relative:.6g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
