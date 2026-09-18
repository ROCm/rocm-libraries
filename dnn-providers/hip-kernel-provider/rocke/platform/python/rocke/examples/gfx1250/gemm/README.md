# gfx1250 scaled GEMM examples

Matrix dtype names normalize through the architecture catalog: `fp8e4m3`,
`bf8e5m2`, `fp6e2m3`, `fp6e3m2`, and `fp4e2m1`. The short spellings `fp8`,
`bf8`, `fp6`, `bf6`, and `fp4` remain accepted aliases wherever that matrix
format is supported. Example argument parsing normalizes aliases before
checking the family's supported choices. The FP8 and FP6 examples default to
`both`, which runs their two homogeneous encodings separately.

These matrix names are distinct from scale formats such as `e8m0`. Scales use
byte storage and packed integer instruction operands; a scale-format name does
not introduce a scalar IR type or general conversion support. Existing operation
IDs retain their instruction-format tokens.

Run these modules from an installed rocKE environment, or with the platform's
`python` directory on `PYTHONPATH`. They require a gfx1250 device, matching
LLVM 23 ROCm libraries, NumPy, and `ml_dtypes`. Torch is optional.

| Example | Matrix inputs | Scale contract |
| --- | --- | --- |
| `mxfp8_gemm` | FP8 E4M3 (`fp8`) or BF8 E5M2 (`bf8`) on both operands | E8M0 |
| `mxfp4_gemm` | Packed FP4 E2M1 on both operands | E8M0 |

```sh
python -m rocke.examples.gfx1250.gemm.mxfp8_gemm
python -m rocke.examples.gfx1250.gemm.mxfp4_gemm --compile-route hip --case all
```

Each example constructs a family-specific spec and invokes the shared verifier
to prepare inputs, pack them, compile once, launch, and compare with an
independent decoded reference. M and N must be positive multiples of 16; the
bounded correctness fixtures support K=128 and K=256, with BF16 output.

The default `wmma_scale` instruction uses one E8M0 scale per 32 K elements.
`--matrix-path wmma_scale16` explicitly selects the native 16-element block
variant. Scale bytes are packed in increasing K-group order. FP8 uses one byte
per value; [FP4 uses two values per byte, low nibble first](FP4_SCALE.md).

`--case all` checks neutral scales, independent A/B scales, varying scales, and
every scale group. These bounded exact fixtures do not establish arbitrary-input
rounding, special-value semantics, or performance. The kernel builders and
runtime/packing utilities are shared; each example exposes one input contract.

The generic `block_scaled_gemm_verify` CLI remains available for regression
testing and the older software-scaled WMMA path. That older path is not the
default in the focused MX examples.

## FP6 and mixed matrix formats

`mxfp6_gemm --dtype fp6` and `mxfp6_gemm --dtype bf6` select homogeneous
six-bit inputs. See [FP6.md](FP6.md) for packing. These families have their
own examples and numerical groups.

`mixed_scaled_gemm` selects different A/B formats, with independent packed
row strides and fragments. Its default is FP8 x FP4; it also accepts FP6/BF6.
It rejects equal canonical formats and uses E8M0 scales on both operands.

```bash
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mxfp6_gemm --dtype bf6
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mixed_scaled_gemm --dtype-a fp8 --dtype-b fp4
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mixed_scaled_gemm --dtype-a fp6 --dtype-b bf6 --compile-route hip
```

The verifier's `mixed` case means combined A/B scale variation; the example's
matrix choices determine whether the input dtypes are homogeneous or mixed.

## Independent scale formats

`scale_formats_gemm` defaults to FP4 x FP4 with E4M3 scales. Choose
`--scale-dtype-a` and `--scale-dtype-b` explicitly for mixed inputs. E4M3/E5M3
requires an FP4 operand, and FP4 x FP4 requires matching scale formats.
See [SCALE_FORMATS.md](SCALE_FORMATS.md) for the supported combinations.

```bash
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.scale_formats_gemm
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.scale_formats_gemm --dtype-a fp6 --scale-dtype-a e8m0 --scale-dtype-b e5m3
```

## NVFP4 tensor scaling

`nvfp4_gemm` uses packed FP4 with E4M3 block scales at K=16 and runtime
FP32 tensor factors. It shares the NVFP4 builder and verifier, retaining the
explicit FP32 rounding step before output conversion. See [NVFP4.md](NVFP4.md).

```bash
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.nvfp4_gemm
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.nvfp4_gemm --dtype-c fp16 --compile-route hip
```

`mxfp8_gemm` runs both FP8 E4M3 and BF8 E5M2 by default, as separate
homogeneous cases. Use `--dtype fp8` or `--dtype bf8` to run one encoding,
or `--dtype both` explicitly. The names follow the FP6/BF6 convention.
Both operands use the selected format with E8M0 scales; mixed matrix formats
have a separate example.

`mxfp6_gemm` runs both `fp6` (E2M3) and `bf6` (E3M2) by default, as
separate homogeneous cases. Use `--dtype fp6`, `--dtype bf6`, or explicit
`--dtype both` to select the encodings.
