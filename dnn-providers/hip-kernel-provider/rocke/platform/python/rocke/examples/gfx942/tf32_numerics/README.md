# Native TF32 numerics on gfx942

Run the [numerical example](../tf32_numerics.py) with NumPy and a built
`rocke_engine` on `PYTHONPATH`, using a ROCm runtime/compiler pair that supports
gfx942:

```console
python -m rocke.examples.gfx942.tf32_numerics --backend both --shape all --output-dir results
```

Set `ROCKE_LLVM_FLAVOR` to the installed COMGR release (`llvm20` for ROCm <7.2,
`llvm22` for 7.2–7.12, `llvm23` for 7.13+). The default runs both engines and both
native shapes. Missing gfx942 hardware or the native extension is an error.
The [spec-driven probe](../../../instances/gfx942/tf32_mma_probe.py) is also
available through the native builder and the paired test emitters.

The example loads dense A and transposed B tiles, accumulates into FP32 C, and
saves D plus the actual prepared A/B bits. Each shape runs raw FP32, the same bits
loaded through I32, explicit GPU RNE preparation, independent host RNE preparation,
host masking, and ordinary FP32 MFMA over the same total K. The C++ lane builds
its own IR and lowers it natively; it also checks exact Python/C++ byte identity.

The corpus includes both signs, even/odd ties and their neighbors, exponent
carry, subnormals, signed zero, infinities, signaling/quiet NaNs, deterministic
dense data, cancellation, asymmetric matrices, and nonzero C. Single products
are checked exactly (NaNs by classification, zero after positive-zero
accumulation). Dense results use float64 products with an explicit FP32
accumulation bound based on the sum of absolute products. No BLAS TF32 policy
can influence the reference.

The output directory contains inputs, prepared words and outputs (`.npz`), LLVM,
HIP source, HSACO, full boundary observations, and `results.json`. The program
exits nonzero on compilation, launch, preparation, parity, or numerical failure.
Boundary JSON preserves hexadecimal results for every variant; the console
prints a representative RNE counterexample. This is a correctness example, not
a performance benchmark.

## Input preparation and logical type

`TF32` is a distinct logical IR type with an **I32 carrier**, one 32-bit word per
element. The dtype names `tf32` and `xf32` select the same catalog format.
`bitcast(fp32_value, TF32)` preserves the raw payload; it does not round.
`cvt_f32_to_tf32(fp32_value)` explicitly rounds to nearest, ties to even, setting
the low 13 fraction bits to zero. It preserves infinities and signed zero and
quiets NaNs while retaining the upper payload bits. A bitcast back to F32
reinterprets the stored bits exactly; it does not truncate a raw payload.

On gfx942 with ROCm 7.14, both native XF32 shapes accepted raw FP32 payloads.
For the tested corpus their outputs matched masking the low 13 bits; I32
transport was bit-preserving. RNE preparation changed rounding-boundary results.
For a single product with B=1 and C=0:

| Path | Input bits | Output bits |
|---|---|---|
| Ordinary FP32 MFMA | `3f801001` | `3f801001` |
| Raw XF32 / I32 carrier | `3f801001` | `3f800000` |
| GPU / host RNE → XF32 | `3f802000` | `3f802000` |

Thus preparation is unnecessary for **instruction acceptance**, but explicit
preparation is necessary when the caller requires **RNE inputs**. Raw payloads
retain native XF32 behavior, including special encodings: in the tested corpus
a signaling NaN whose payload occupied only discarded bits became infinity.
The RNE conversion prevents that by quieting NaNs before clearing low bits.
These observations describe this corpus and toolchain; they are not a guarantee
that raw payloads have general FP32 arithmetic semantics.

The catalog exposes only gfx942 `16x16x8` and `32x32x4` TF32 atoms, with two
logical TF32 elements per lane in A/B and FP32 accumulation. The LLVM intrinsic
ABI takes two floats, so lowering bitcasts the I32 carrier at the intrinsic
boundary. Ordinary FP32 catalog selection is unchanged. Generic TF32 arithmetic
is rejected; bitcasts, vector transport, selection, and the native atoms are
supported. Production GEMM dispatch and cross-target emulation are separate work.
