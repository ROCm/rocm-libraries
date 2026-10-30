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
The shared spec accepts `tensor_scale=True` only for this operand and scale
contract. Use `nvfp4_gemm_verify.run_cases` to verify tensor-enabled kernels;
it supplies both factors and a tensor-aware reference. The generic
`block_scaled_gemm_verify.run_cases` rejects tensor-enabled specs before
device discovery or compilation. The shared launcher and NVFP4 reference require
real integer or floating-point tensor factors with shape `(2,)`, convert them to
FP32, and require the converted values to be finite. String, complex, boolean,
and object arrays are rejected. The launcher rejects malformed factors and
factors supplied for plain specs before allocating GPU memory.
Scale selection is mirrored by Python `IRBuilder.mma` and C
`rocke_b_mma_scaled`; the complete spec-to-kernel builder remains Python-only.
C++ lowers its serialized IR through LLVM, including the tensor-scale epilogue.
HIP compilation uses Python-generated source only; native HIP support is outside
this NVFP4 API's scope. Native inline-assembly lowering remains unsupported, so
the FP16 tensor-epilogue barrier rejects through native HIP.

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
epilogue placement, invalid specs, and preserved defaults. The native barrier
CTest also pins direct/serialized HIP rejection of the FP16 tensor epilogue
while preserving its LLVM support. The GPU verifier covers
K=128/256, BF16/FP16 outputs, independent block scales, non-power-of-two tensor
factors, zero scales, isolated K groups, and all FP4 code products at lane/K
boundaries. Exact fixtures use bounded products and explicitly rounded FP32
tensor scaling; they do not establish arbitrary-input accumulation behavior.

Run these host checks from the platform directory with test dependencies installed
and a fresh `rocke_engine` extension on `PYTHONPATH`:

```sh
ROCKE_BACKEND=python python -m pytest -q \
    tests/core/test_gfx1250_nvfp4.py tests/test_rocke_ci_static.py \
    tests/instances/test_gfx1250_nvfp4_example.py tests/core/test_optimization_barrier.py
ROCKE_BACKEND=cpp ROCKE_CPP_STRICT=1 python -m pytest -q \
    tests/core/test_gfx1250_nvfp4.py tests/test_rocke_ci_static.py \
    tests/instances/test_gfx1250_nvfp4_example.py tests/core/test_optimization_barrier.py
ROCKE_BACKEND=both ROCKE_CPP_STRICT=1 python -m pytest -q \
    tests/core/test_gfx1250_nvfp4.py tests/test_rocke_ci_static.py \
    tests/instances/test_gfx1250_nvfp4_example.py tests/core/test_optimization_barrier.py
python tests/instances/rocke_ir_parity_harness.py \
    --check tests/golden/rocke_representative_ir_sha256
```

The representative registry includes BF16/FP16 outputs at M32/N48 and
K128/K256. The strict `both` registry test builds each case in Python and compares
Python/native LLVM lowering through serialized IR. The golden checks emitted-byte
stability and expected LLVM20/22 rejections; NVFP4 emits at the `llvm23` flavor.
`tools/check_byte_identity.py` uses a separate inventory of paired Python/C
emitters, so its family totals do not establish NVFP4 builder coverage.

The host example test calls `nvfp4_gemm.main`; the GPU test below launches
`nvfp4_gemm_verify` as a subprocess for COMGR/HIP, BF16/FP16, and both shapes:

```sh
ROCKE_LLVM_FLAVOR=llvm23 ROCKE_BACKEND=both ROCKE_CPP_STRICT=1 \
ROCKE_REQUIRE_GFX1250=1 python -m pytest -q -s \
    tests/instances/test_gfx1250_nvfp4_numeric.py
```

Run from the platform directory with a matching gfx1250 device, NumPy/ml_dtypes,
a freshly built C++ extension, and COMGR/HIP supporting the `llvm23` emission
flavor. The flavor describes rocKE's emitted intrinsic ABI; it does not require
the compiler major to be exactly 23. COMGR compares
Python/C++ LLVM bytes; the HIP route compiles Python-generated HIP source.
`ROCKE_REQUIRE_GFX1250=1` makes missing dependencies or the requested device fail
the GPU lane. Host hashes and engine agreement do not establish GPU correctness.
Record the tested source revision, device architecture, compiler/runtime versions,
command, and pass/skip/failure counts for each GPU campaign. If a merge receives
only host validation, state that explicitly and retain the revision of earlier
GPU evidence.

This is a packed-input GEMM. Quantization, conversion of foreign scale layouts,
NVFP4 output quantization, partial tiles, a dynamic K loop, and a full C++ builder
remain separate work. It makes no performance claim.
