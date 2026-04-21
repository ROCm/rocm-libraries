# ASM Kernel Binaries — Provenance

Source repository: https://github.com/ROCm/aiter
Commit: 17d4a33b6f9535e820353ebc6217769efc3766d6

Platform: gfx942 (MI300X / MI300A)

| File | AITER Source Path | Description |
|------|------------------|-------------|
| gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co | hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co | Forward attention, hd128, BF16, round-to-nearest-even, non-causal |
| gfx942/fmha_v3_bwd/MI300/bwd_hd128_odo_bf16.co | hsa/gfx942/fmha_v3_bwd/bwd_hd128_odo_bf16.co | Backward pre-processing: D = sum(O * dO) reduction |
| gfx942/fmha_v3_bwd/MI300/bwd_hd128_bf16_a32_rtne_psskddv.co | hsa/gfx942/fmha_v3_bwd/bwd_hd128_bf16_a32_rtne_psskddv.co | Backward main: dQ/dK/dV, FP32 accumulator, RTNE, parallel split |
| gfx942/fmha_v3_bwd/MI300/bwd_hd128_dq_convert_bf16_rtne.co | hsa/gfx942/fmha_v3_bwd/bwd_hd128_dq_convert_bf16_rtne.co | Backward post-processing: dQ FP32→BF16 conversion |

## Directory layout notes

- **Forward kernels** (`fmha_v3_fwd/`): AITER has separate `MI300/` and `MI308/` subdirectories
  because MI308 (MI300A APU) has different fp8 support. We only include MI300 for the POC.
- **Backward kernels** (`fmha_v3_bwd/`): AITER stores these flat (no MI300/MI308 split) because
  backward kernels only support bf16/fp16 (no fp8), so the same binary works on both MI300X and
  MI300A. We nest them under `MI300/` here for consistency with the forward directory pattern.
