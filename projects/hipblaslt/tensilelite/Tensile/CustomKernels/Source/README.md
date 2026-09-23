# W4A16 unsigned_bias8 kernels

## Decode

`w4a16_decode.hip` is the source for the checked-in
`Custom_W4A16_Decode_G32_W4_UnsignedBias8_gfx1151.s`. Regenerate it with:

```sh
python generate_w4a16_decode.py --compiler /path/to/hipcc --group-size 32 --load-width 4
```

The generator targets gfx1151 and code object version 4, disables compilation-unit
IDs, and attaches the Tensile universal argument layout and selection constraints.
The source also supports group size 128 and eight-dword loads for experimentation;
only the group-32 four-dword variant is shipped as a specialized decode kernel.

The kernel supports FP16 activations/output, asymmetric group scales, sequential
unsigned int4 nibbles, one output column, one batch, and positive K divisible by
256. Four waves compute four output rows. Each wave reduces along K in FP32 after
FP16-rounded dequantization. Rotating the K starting point between rows avoids
concentrating power-of-two row strides on the same memory channels. Streaming
weight loads preserve cache space for activations. The kernel uses no LDS.

## Q27B Equality dispatch

`gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_Q27B.yaml` selects the specialized
decode kernel for N=1 and custom prefill kernels for N=2048, with batch=1:

| M | K | Tuned lda | Prefill tile |
| ---: | ---: | ---: | --- |
| 34816 | 5120 | 5632 | 64x256x64 |
| 5120 | 17408 | 17920 | 128x256x64 |
| 16384 | 5120 | 5632 | 128x256x64 |
| 14336 | 5120 | 5632 | 128x256x64 |
| 5120 | 6144 | 6656 | 128x256x64 |

Equality keys are M, N, batch, and K. Other sizes retain the FreeSize fallback.
The table records the strides used for tuning; it does not require padding.

The prefill kernels are maintained as custom assembly. MT64x256 uses packed FP16
subtract/multiply for dequantization, followed by four byte permutations per eight
weights to restore sequential order before LDS writes. The subtraction is exact;
scaling rounds once to FP16. Its zero-point byte offsets are calculated once and
retained in v189/v190 across the K loop. The kernel declares 191 VGPRs and uses
46,080 bytes of LDS.

## Cold measurement

Use the benchmark from the corresponding host build. Leave
`HIPBLASLT_TENSILE_LIBPATH` unset to test the default build-tree device library,
or point it at an isolated library containing the lazy master and code objects.

```sh
gpu-lock --high hipblaslt-bench -m 34816 -n 1 -k 5120 \
  --lda 5632 --ldb 5120 --ldc 34816 --ldd 34816 \
  --transA T --transB N \
  --a_type i4_r --b_type f16_r --c_type f16_r --d_type f16_r \
  --compute_type f32_r --scaleA 1009 --int4_encoding unsigned_bias8 \
  --alpha 1 --beta 0 --rotating 512 --use_gpu_timer --adaptive -v
```

Use N=2048 to measure prefill. Omit captured `--solution_index` and
`--algo_method index` options to exercise automatic dispatch.
For decode, calculate effective bandwidth from the packed payload, since the
benchmark's generic bandwidth column does not account correctly for int4:

```
bytes = M*K/2 + 2*M*(K/32) + ceil(M/2)*(K/32) + 2*K + 2*M
```

This includes weights, scales, zero points, activations, and output for beta=0.
