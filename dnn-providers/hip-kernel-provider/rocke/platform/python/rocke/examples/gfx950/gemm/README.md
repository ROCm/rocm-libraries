# gfx950 scaled GEMM examples

These untuned correctness examples use the native 32x32x64 scaled MFMA and the
reusable `instances/gfx950/block_scaled_gemm.py` builder. They make no
performance claim.

These entrypoints construct `BlockScaledGemmSpec` and call the builder directly.
The builder selects a `mfma_scaled` catalog atom and emits `IRBuilder.mma`;
there is currently no GEMM dispatcher registration for this family. The
existing gfx950 FP8 fused-MoE dispatcher uses the separate 16x16x128 FP8
atom, with its FP32 block scales applied outside the MFMA. See the
[family contract](../../../../../dsl_docs/instances/gfx950_block_scaled_gemm.md)
and [selection/emission comparison](../../../../../dsl_docs/reference/mma_metadata.md#gfx950-scaled-mfma-paths).

Run from an installed rocKE package or with the platform `python` directory on
`PYTHONPATH`. A gfx950 device, matching LLVM 20, 22 or 23 ROCm compiler/runtime
libraries, NumPy and
`ml_dtypes` are required. The commands below use LLVM 23; select the flavor
matching your compiler.

| Example | A/B matrix formats |
| --- | --- |
| [mxfp8_gemm](mxfp8_gemm.py) | Homogeneous FP8 E4M3 or BF8 E5M2; both by default |
| [mxfp6_gemm](mxfp6_gemm.py) | Homogeneous FP6 E2M3 or BF6 E3M2; both by default |
| [mxfp4_gemm](mxfp4_gemm.py) | Homogeneous FP4 E2M1 |
| [mixed_scaled_gemm](mixed_scaled_gemm.py) | Any of the 20 ordered unequal pairs; defaults to FP8 x FP4 |

```sh
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx950.gemm.mxfp8_gemm --case all
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx950.gemm.mxfp6_gemm --case all
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx950.gemm.mxfp4_gemm --case all
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx950.gemm.mixed_scaled_gemm --dtype-a fp6 --dtype-b fp4 --case all --compile-route hip --output-dtype fp16
```

Short aliases `fp8`, `bf8`, `fp6`, `bf6`, `fp4` normalize to `fp8e4m3`,
`bf8e5m2`, `fp6e2m3`, `fp6e3m2`, `fp4e2m1`. Both matrices are densely packed
row-major, A[M,K] and B[N,K], yielding C[M,N]. [FP6 packing](FP6.md) and
[FP4 packing](FP4_SCALE.md) specify byte storage. A and B may use different
formats and strides. C uses BF16 (default) or FP16, with FP32 accumulation.

Scale tensors contain raw finite E8M0 bytes: A_scale[M,K/32] and
B_scale[K/32,N]. Each scale covers 32 logical K values. The consumer loads
one byte per lane, zero-extends it into an i32 VGPR, and supplies separate A/B
scale operands. Code 127 is unity; codes 125, 126, 128 and 129 represent
1/4, 1/2, 2 and 4. The default `mixed` case uses independent non-unity patterns
on both operands, varying rows, columns, K32 groups, K64 steps and output tiles.
`--case all` adds unity and one-sided controls, signed and positive mantissas,
and every isolated K32 group. A NaN output sentinel detects missing stores.

M and N must be positive multiples of 32; the builder accepts positive K
multiples of 64. The bounded example fixtures accept K64, K128 and K256 and
default to M32/N64/K256. Native comparisons require exact numerical equality
after output-type rounding, with signed zeros treated as equal. Fixture bounds
ensure exact FP32 partial sums. These checks do not establish arbitrary-input
rounding, zero-sign semantics, extreme E8M0 exponents or NaN behavior.

The catalog exposes all 25 ordered pairs as `mfma_scaled` atoms. Each lane
owns 32 logical A/B values; the LLVM ABI uses padded eight-i32 carriers, with
8/6/4 physical VGPR words for F8/F6/F4. F8/BF8 lane payloads interleave 16-value
chunks between K32 halves; F6/BF6/F4 payloads use contiguous 32-value chunks.
C owns 16 FP32 values per lane in a wave64. Scale ownership is one K32 group
per half-wave. The Python and C++ builders and lowerers mirror these contracts.
LLVM 20, 22 and 23 support this family on gfx950 with the same overloaded
nine-argument intrinsic ABI. Select the flavor matching the compiler when
emitting offline IR; other targets reject this family explicitly.

The public [CDNA4 machine-readable ISA](https://gpuopen.com/machine-readable-isa/)
provides the `V_MFMA_SCALE_F32_32X32X64_F8F6F4` encoding (ENC_VOP3PX2, opcode
46) and separate SCALE_SRC0/SCALE_SRC1 operands. LLVM's
[AMDGPU intrinsic definitions](https://github.com/llvm/llvm-project/blob/0bace1908348b840e6aa1b4b6e12151dae208158/llvm/include/llvm/IR/IntrinsicsAMDGPU.td)
provide the overloaded nine-argument ABI. Compiler/disassembly tests check the
instruction, format selectors, physical tuples and VGPR scales; GPU numeric
tests check the loaded-scale interpretation and layout independently.
