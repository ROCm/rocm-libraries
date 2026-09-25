# gfx1250 scaled GEMM examples

Matrix dtype names normalize through `core/dtypes.py`: `fp8e4m3`, `bf8e5m2`,
and `fp4e2m1`, with short aliases `fp8`, `bf8`, and `fp4` for these examples.
Example argument parsing normalizes aliases before checking supported choices.

Run these modules from an installed rocKE environment, or with the platform's
`python` directory on `PYTHONPATH`. They require a gfx1250 device, matching
LLVM 23 ROCm libraries, NumPy, and `ml_dtypes`. Torch is optional.

| Example | Matrix inputs | Scale contract |
| --- | --- | --- |
| `mxfp8_gemm` | FP8 E4M3 (`fp8`) or BF8 E5M2 (`bf8`) on both operands | E8M0 |
| `mxfp4_gemm` | Packed FP4 E2M1 on both operands | E8M0 |

```sh
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mxfp8_gemm
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mxfp4_gemm --compile-route hip --case all
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
or `--dtype both` explicitly. Both operands use the selected format with E8M0
scales. Mixed matrix formats are outside these examples.

The target-independent aliases in `core/dtypes.py` normalize logical format
names. Recognition does not imply that a target supports an atom; use its
catalog to query supported operand formats and shapes. Each example defines
its accepted matrix and scale formats.

`e8m0` identifies a scale format carried in bytes and packed integer operands.
It is not a general scalar IR type or conversion API. Native scaled WMMA uses
one `wmma_scaled` catalog family. Its operation IDs encode the matrix source
and accumulator dtype, each source's scale format, and a shared K-group size; the backend
selects the LLVM intrinsic and packed operand types from that contract.

The native loader uses the atom's `a_scale_layout()` and `b_scale_layout()` for
lane ownership. Global scale tensors have shapes `[M, K/block_k]` and
`[K/block_k, N]`, respectively. K32 packs four E8M0 bytes per lane into i32;
K16 packs eight into i64, first K group in the low byte. Both half-waves carry
the same scales. Matrix A/B lane maps for these scaled atoms are not yet exposed.

CPU tests independently check every scale coordinate, multi-tile addresses and
packed byte order. The opt-in numerical suite exercises both encodings and
scale group sizes, one-sided/group-isolated inputs, multiple tiles and K steps,
and HIP/COMGR compilation. These cases bound the validation; dtype recognition
alone does not establish support for other operand contracts.
