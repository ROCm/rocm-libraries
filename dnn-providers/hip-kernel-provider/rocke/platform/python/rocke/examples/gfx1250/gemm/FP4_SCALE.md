# Packed FP4 scaled GEMM

The [block-scaled GEMM builder](../../../instances/gfx1250/block_scaled_gemm.py)
accepts `dtype_a="fp4", dtype_b="fp4"` for native gfx1250 SCALE and SCALE16.
This path consumes prepacked E2M1 values and E8M0 scale bytes. It does not
quantize floating-point inputs or provide an A16W4 dequantization kernel.

## Input contract

`M`, `N`, and `K` count logical matrix elements. A and B use row-major byte
buffers of shapes `[M, K/2]` and `[N, K/2]`, computing `C = A @ B.T`.
For byte `j`, bits 0–3 encode element `2*j`; bits 4–7 encode element `2*j+1`.
The E2M1 positive codes 0–7 represent `0, 0.5, 1, 1.5, 2, 3, 4, 6`;
bit 3 is the sign, including negative zero. Every nibble is a valid code.
The pointer ABI is `i8`, with 16-byte-aligned matrix buffers.

| Matrix path | K elements per scale | A/B scale operand |
| --- | --- | --- |
| `wmma_scale` | 32 | Four E8M0 bytes packed into i32 |
| `wmma_scale16` | 16 | Eight E8M0 bytes packed into i64 |

Scale memory has shape `[M, K/block_k]` for A and `[K/block_k, N]` for B.
Successive K groups occupy successive bytes, starting at the low byte of the
instruction operand. `scale_dtype="e8m0"` selects this contract; `i8` is its
storage alias. E4M3/E5M3 scale formats are not exposed by this path.
SCALE with block size 32 is MXFP4. SCALE16 here means FP4 with E8M0 scales
and block size 16.

Each wave computes a 16-by-16 output tile. Lane `l` loads A row and B row
`l % 16` within their tiles. Lane half `h = l // 16` owns logical K ranges
`[32*h, 32*h+32)` and `[64+32*h, 96+32*h)` within each K=128 step. These
are two 16-byte loads, yielding eight i32 words padded with eight zero words
for the builtin's sixteen-word argument. Both scale modes use this input map.
Output slot `i` maps to row `8*h+i`, column `l % 16` within the output tile.

M/N must be multiples of 16 and K a multiple of 128. Mixed operand formats,
partial tiles, a logical FP4 IR type, and quantization conversions are outside
this path. The K loop is statically unrolled, as in the existing FP8 builder.

## Verification

The [verifier](block_scaled_gemm_verify.py) builds exact E2M1 code fixtures,
packs them into bytes, and independently decodes the bytes for its reference.
It covers every code point, asymmetric A/B data, independent scales, isolated
K groups, and multiple K steps. FP4 fixtures use finite E8M0 bytes 125–128;
scale extremes and NaN behavior require separate validation.

```sh
python -m rocke.examples.gfx1250.gemm.block_scaled_gemm_verify \
    --dtype fp4 --matrix-path wmma_scale --case all
python -m rocke.examples.gfx1250.gemm.block_scaled_gemm_verify \
    --dtype fp4 --matrix-path wmma_scale16 --m 32 --n 48 --k 256 --compile-route hip
```

COMGR requires a matching LLVM 23 toolchain. With the C++ extension installed,
set `ROCKE_BACKEND=both` to compare Python/C++ LLVM before COMGR compilation.
The HIP verifier invokes the Python HIP lowerer directly. Numerical validation
requires a matching gfx1250 device, NumPy, and ml_dtypes; it does not need torch.
