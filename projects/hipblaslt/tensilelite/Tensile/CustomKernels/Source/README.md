# W4A16 decode kernel

`w4a16_decode.hip` is the source for
`Custom_W4A16_Decode_G128_ExLlama_gfx1151.s` and
`Custom_W4A16_Decode_G32_ExLlama_gfx1151.s`. Regenerate them with:

```sh
python generate_w4a16_decode.py --compiler /path/to/hipcc --group-size 128
python generate_w4a16_decode.py --compiler /path/to/hipcc --group-size 32
```

The checked-in assembly was generated with AMD clang 23.0.0git,
LLVM revision `0bace1908348b840e6aa1b4b6e12151dae208158`, targeting gfx1151
and code object version 4. Compilation-unit IDs are disabled so temporary
output paths do not change the assembly. The compiler marker is named after
the kernel to allow multiple variants in the same code object. The generator also attaches the
Tensile kernel-argument version and selection constraints.

The kernel supports FP16 activations/output, asymmetric group-32 or group-128 scales,
ExLlama nibble order, one output column, one batch, and positive K divisible
by 256. Four waves compute four output rows. Each wave reduces along K in
FP32 after FP16-rounded dequantization. Rotating the K starting point between
rows avoids concentrating power-of-two row strides on the same memory
channels. Streaming weight loads preserve cache space for activations.
K divisible by 2048 and a row stride divisible by 64 int4 elements use eight
packed dwords per lane; other supported inputs use one dword per lane.
With group size 32, each eight-dword load spans two quantization groups;
each half uses its own scale and zero-point.
The kernel uses no LDS and supports the usual scalar alpha/beta epilogue.

## Cold bandwidth measurement

Point `HIPBLASLT_TENSILE_LIBPATH` at the directory containing the generated
`TensileLibrary_lazy_gfx1151.dat.zlib` and code objects. Use the benchmark
from the corresponding host build:

```sh
hipblaslt-bench -m 6144 -n 1 -k 4096 \
  --lda 4096 --ldb 4096 --ldc 6144 --ldd 6144 \
  --transA T --transB N \
  --a_type i4_r --b_type f16_r --c_type f16_r --d_type f16_r \
  --compute_type f32_r --scaleA 1011 --int4_encoding 2 \
  --alpha 1 --beta 0 --rotating 512 --use_gpu_timer --adaptive -v
```

Use `--algo_method all` to compare the decode and matrix kernels in the same
library. The regular heuristic also selects the decode kernel for this shape.

Calculate bandwidth from the measured time and actual packed payload; the
benchmark's current GB/s column does not account correctly for int4 storage.
For this shape with beta zero:

| Payload | Bytes |
| --- | ---: |
| Packed weights | 12,582,912 |
| FP16 scales | 393,216 |
| Packed zero-points | 98,304 |
| Activations | 8,192 |
| Output | 12,288 |
| Total | 13,094,912 |

For example, 56.48 microseconds corresponds to 231.9 GB/s of total payload
or 222.8 GB/s counting packed weights alone. Report which convention is used.
The rotating allocation exceeds the GPU cache capacity; these are cold weight
measurements, not repeated accesses to a single resident weight matrix.

For group size 32, use `--scaleA 1009` with the same shape and timing flags.
The scales occupy 1,572,864 bytes and zero-points 393,216 bytes; all other
payload sizes above stay the same, for a total of 14,569,472 bytes.
Three cold adaptive runs measured 57.9322, 57.9134, and 57.5432 microseconds
on the Radeon 8060S. Their median is 251.6 GB/s total payload or 217.3 GB/s
packed weights alone. All three runs converged. The monitor recorded package
power and CPU thermal throttling during the combined validation/comparison run.
An alternative four-dword load was slower, so the eight-dword path is retained.

## Q27B Equality dispatch

The gfx1151 Equality logic in
`gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8X_Q27B.yaml` selects measured
solutions for these group-32 FP16 ExLlama projections. Decode uses the
four-dword variant; prefill uses the matrix kernel with workgroup mapping
1 or 4. Regenerate the additional decode assembly with:

```sh
python generate_w4a16_decode.py --compiler /path/to/hipcc --group-size 32 --load-width 4
```

| M | K | Tuned lda | Decode N=1 (us) | Prefill N=2048 (us) | Prefill mapping |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 34816 | 5120 | 5632 | 441.534 | 20603.5 | 1 |
| 5120 | 17408 | 17920 | 216.800 | 10164.9 | 4 |
| 16384 | 5120 | 5632 | 185.530 | 9733.75 | 1 |
| 14336 | 5120 | 5632 | 172.852 | 8482.52 | 1 |
| 5120 | 6144 | 6656 | 74.795 | 3767.01 | 4 |

These are normal C API heuristic selections from a library containing both
Equality and FreeSize logic, with `--rotating 512 --adaptive --use_gpu_timer`.
All ten timings converged. Package power throttling was recorded during the
run. Tuning compared decode load widths 4 and 8 and prefill mappings
1, 4, 8, 16, and 32; the leading prefill mappings were measured again.
Equality keys are M, N, batch=1, and K. Other dimensions retain the FreeSize
fallback. The table records the strides used for tuning; it does not impose
a new requirement that callers pad their weights.

All five exact decode shapes passed numerical validation. Both prefill mappings
passed numerical checks at M=129, N=2048 and each of the three K values,
covering ragged rows and multiple tiles; full-size prefill timings do not run
the CPU reference. Use the benchmark command above with `--scaleA 1009`,
the desired M/N/K/lda from the table, ldb=K and ldc=ldd=M. Omit any captured
`--solution_index` and `--algo_method index` to exercise Equality dispatch.

## Q27B unsigned_bias8 prefill

`generate_w4a16_prefill_unsigned_bias8.py` regenerates the MT64x256x64
unsigned_bias8 assembly from its checked-in ExLlama counterpart:

```sh
python generate_w4a16_prefill_unsigned_bias8.py
```

This kernel serves M=34816, N=2048, K=5120 in the group-32 Equality grid.
It uses the same packed FP16 dequantization, then four byte permutations per
eight weights to restore sequential nibble order before the LDS writes.
The subtraction is exact and the scale multiplication rounds once to FP16.
Tile dimensions, scheduling, LDS use, and VGPR allocation match ExLlama.

The regeneration script also hoists the two zero-point byte-offset calculations
in the ExLlama source before generating unsigned_bias8. Both kernels retain those
lane-dependent offsets in v189/v190 from the initial prefetch through the K loop.
The original nibble offsets remain available for selecting zero points. This
removes two vector shifts per loop iteration; 191 declared VGPRs still fit in
the same 192-register allocation block as the previous 189 on gfx1151.
