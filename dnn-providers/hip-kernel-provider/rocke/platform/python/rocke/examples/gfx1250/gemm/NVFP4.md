# Packed NVFP4 GEMM

`NvFp4GemmSpec` consumes E2M1 FP4 inputs, E4M3 scales for each group of
16 values along K, and one FP32 dequantization factor per input tensor.
It uses native gfx1250 SCALE16 for the block-scaled dot product, then applies
the tensor-scale product to the FP32 accumulator before BF16/FP16 conversion.

For row-major A and row-major B used transposed (RCR):

```text
A_real[m,k] = A_tensor_scale * decode(A_scale[m,k//16]) * decode(A[m,k])
B_real[n,k] = B_tensor_scale * decode(B_scale[k//16,n]) * decode(B[n,k])
C = A_real @ B_real.T
```

The implementation computes `float32(A_tensor_scale * B_tensor_scale)` once
per thread and multiplies the final FP32 accumulator by that product. The result
is rounded to FP32 at this multiplication boundary before output conversion;
an FP16 rounding barrier prevents contraction into a single-rounding mixed FMA.
Inputs must use finite tensor factors whose product and scaled result are representable
for the chosen output. Tensor factors are multiplicative dequantization values;
callers holding quantization multipliers must supply their reciprocals.

## Input and launch contract

| Argument | Storage |
| --- | --- |
| A | uint8 `[M, K/2]`, two E2M1 values per byte, low nibble first |
| B | uint8 `[N, K/2]`, same packing |
| A_scale | uint8 `[M, K/16]`, encoded E4M3 block scales |
| B_scale | uint8 `[K/16, N]`, encoded E4M3 block scales |
| C | BF16 or FP16 `[M, N]` |
| M, N, K | existing i32 shape arguments, matching the spec |
| A_tensor_scale, B_tensor_scale | trailing runtime FP32 scalar arguments |

M/N must be positive multiples of 16; K must be a positive multiple of 128.
There is no padding in the packed memory buffers. Each lane's eight meaningful
i32 FP4 words are padded to sixteen words at the compiler builtin boundary.
Block-scale arrays use the simple layouts above, not a library-specific tiled
scale-buffer layout. Reorder externally produced scale buffers before launch.

```python
from rocke.instances.gfx1250.nvfp4_gemm import (
    NvFp4GemmSpec,
    build_nvfp4_gemm,
    nvfp4_gemm_grid,
    nvfp4_gemm_signature,
)

spec = NvFp4GemmSpec(name="nvfp4", M=32, N=48, K=256, dtype_c="bf16")
kernel = build_nvfp4_gemm(spec, arch="gfx1250")
signature = nvfp4_gemm_signature(spec)
grid = nvfp4_gemm_grid(spec)
```

The existing `BlockScaledGemmSpec` defaults and launch ABI are unchanged.
It gains defaulted scale overrides and a `tensor_scale=False` field;
`NvFp4GemmSpec` selects FP4/E4M3/SCALE16 with tensor scaling enabled.
Scale selection is mirrored by Python `IRBuilder.mma` and C
`rocke_b_mma_scaled`; the complete spec-to-kernel builder remains Python-only.
C++ lowers its serialized IR, including the tensor-scale epilogue.

## Validation and scope

The [public NVFP4 description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
defines the E2M1, per-16-value E4M3, and tensor FP32 scaling levels.
The [AMD ISA guide](https://gpuopen.com/amd-isa-documentation/), section 7.12.6,
specifies native FP4 with E4M3 scales on both operands. The machine-readable
ISA defines the instruction and operand fields; the guide supplies the scale
legality table. The [cuBLAS reference](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout)
describes a different scale-buffer layout; this builder does not consume that
layout directly.

CPU tests cover the scale encodings, parameter/signature contract, scale direction,
epilogue placement, invalid specs, and preserved defaults. The GPU verifier covers
K=128/256, BF16/FP16 outputs, independent block scales, non-power-of-two tensor
factors, zero scales, isolated K groups, and all FP4 code products at lane/K
boundaries. Exact fixtures use bounded products and explicitly rounded FP32
tensor scaling; they do not establish arbitrary-input accumulation behavior.

```sh
ROCKE_LLVM_FLAVOR=llvm23 ROCKE_BACKEND=both ROCKE_CPP_STRICT=1 \
ROCKE_REQUIRE_GFX1250=1 python -m pytest -q -s \
    tests/instances/test_gfx1250_scaled_wmma_numeric.py
```

Run from the platform directory with a matching gfx1250 device, LLVM 23
COMGR/HIP, NumPy/ml_dtypes, and a freshly built C++ extension. COMGR compares
Python/C++ LLVM bytes; the HIP route compiles Python-generated HIP source.

This is a packed-input GEMM. Quantization, conversion of foreign scale layouts,
NVFP4 output quantization, partial tiles, a dynamic K loop, and a full C++ builder
remain separate work. It makes no performance claim.
