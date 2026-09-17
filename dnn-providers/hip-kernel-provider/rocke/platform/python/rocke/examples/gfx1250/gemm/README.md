# gfx1250 scaled GEMM examples

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

`mxfp8_gemm` runs both FP8 E4M3 and BF8 E5M2 by default, as separate
homogeneous cases. Use `--dtype fp8` or `--dtype bf8` to run one encoding,
or `--dtype both` explicitly. The names follow the FP6/BF6 convention.
Both operands use the selected format with E8M0 scales; mixed matrix formats
have a separate example.
