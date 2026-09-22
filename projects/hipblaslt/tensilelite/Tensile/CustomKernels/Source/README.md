# W4A16 decode kernel

`w4a16_decode.hip` is the source for
`Custom_W4A16_Decode_G128_ExLlama_gfx1151.s`. Regenerate it with:

```sh
python generate_w4a16_decode.py --compiler /path/to/hipcc
```

The checked-in assembly was generated with AMD clang 23.0.0git,
LLVM revision `0bace1908348b840e6aa1b4b6e12151dae208158`, targeting gfx1151
and code object version 4. Compilation-unit IDs are disabled so temporary
output paths do not change the assembly. The generator also attaches the
Tensile kernel-argument version and selection constraints.

The kernel supports FP16 activations/output, asymmetric group-128 scales,
ExLlama nibble order, one output column, one batch, and positive K divisible
by 256. Four waves compute four output rows. Each wave reduces along K in
FP32 after FP16-rounded dequantization. Rotating the K starting point between
rows avoids concentrating power-of-two row strides on the same memory
channels. Streaming weight loads preserve cache space for activations.
K divisible by 2048 and a row stride divisible by 64 int4 elements use eight
packed dwords per lane; other supported inputs use one dword per lane.
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
