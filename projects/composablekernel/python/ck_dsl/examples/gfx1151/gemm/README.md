# gfx1151 WMMA GEMM — the quantization/precision ladder

A self-contained study of WMMA GEMM on **gfx1151 (RDNA3.5 / Strix Halo APU,
wave32)** across the precision ladder the hardware supports, from the f16 baseline
through int4 weight-only to **int8 storage / f16 compute** — each rung verified for
correctness on silicon.

The point is to show, end to end, (1) the **one-wave-per-16×16-tile WMMA GEMM**
skeleton every rung shares, and (2) how int8 quantization is layered onto it
*without* a DSL core change by dequantizing to f16 and reusing the verified f16
WMMA atom. The **true int8 → int32 native path** (`wmma_i32_16x16x16_iu8`) is owned
upstream — see "Relationship to the native-int path" below.

> RCR layout throughout (A row-major `M×K`, B row-major `N×K`, `C = A @ B.T`),
> one wave (32 lanes) per 16×16 output tile, no LDS — the WMMA fragment ABI does
> the lane distribution. These are minimal **correctness-first reference kernels**
> (4–15 % of peak); ratios, not absolute rates, are the headline.

## The ladder

| # | rung | A·B dtype | accumulate | atom | status |
|---|---|---|---|---|---|
| 01 | f16 baseline | f16 × f16 | f32 | `wmma_f32_16x16x16_f16` | PASS |
| 02 | int4 weight-only (W4A16) | f16 × int4 | f32 | `wmma_f32_16x16x16_f16` (+ int4 dequant) | PASS¹ |
| 03 | int8 storage / f16 compute | int8→f16 × int8→f16 | f32 | `wmma_f32_16x16x16_f16` | PASS |

¹ See the int4 tolerance note under Step 02.

This dir's int8 rung (03) is the **f16-compute** approach: convert int8→f16 in the
K-loop and reuse the verified f16 WMMA. No DSL core change; the win is storage /
memory bandwidth. The **true int8 compute** approach (int8×int8→int32 via the
hardware `v_wmma_i32_16x16x16_iu8` instruction) lives in the upstream native-int
path, not here.

## Hardware / software pin

| | |
|---|---|
| GPU | Radeon 8060S / **gfx1151** (RDNA3.5, Strix Halo APU), wave32 |
| OS | **Windows-native** (this box) |
| comgr / hip | `C:\Windows\System32\amd_comgr_3.dll`, `amdhip64_7.dll` (versioned driver DLLs) |
| f16 WMMA peak | ~59 TFLOP/s |
| int8 WMMA peak | ~118 TOPS (≈2× f16) |
| CK DSL | this repo (`projects/composablekernel/python/ck_dsl`) |

## Reproduce

```bash
cd <ck_dsl>/examples/gfx1151/gemm          # this folder
# Windows-native env: point the loader at the driver DLLs (no PYTHONPATH needed —
# the scripts add the python root themselves):
export CK_DSL_COMGR_LIB="C:\\Windows\\System32\\amd_comgr_3.dll"
export CK_DSL_HIP_LIB="C:\\Windows\\System32\\amdhip64_7.dll"

python scripts/01_f16_verify.py                # f16 baseline
python scripts/02_int4_matmul_nbits_verify.py  # int4 weight-only (W4A16)
python scripts/03_int8_pathb_verify.py         # int8 storage / f16 compute
```

Each script writes its result to `data/0N_*.json`.

## Step 01 — f16 baseline (`01_f16_verify.py`)

The reference WMMA GEMM (`instances/gfx1151/wmma_gemm.py`). Builds, writes a gemm
manifest, and verifies via `ck_dsl.run_manifest --verify` against a numpy
`C = A @ B.T` (small integer inputs; tolerance `1e-2` to absorb the f32
accumulation-order difference vs numpy). Establishes the one-wave-per-16×16-tile
skeleton every rung reuses. Result: **PASS**, `max_abs_diff ≈ 1.8e-5`.

## Step 02 — int4 weight-only, W4A16 (`02_int4_matmul_nbits_verify.py`)

`MatMulNBits` large-N (`instances/common/matmul_nbits.py`): fp16 activations ×
packed-int4 weights with per-group (g=32) fp16 scales, dequantized on load, then
the same f16 WMMA. Verifies against `C = A @ dequant(B, scales)^T`.

> **Int4 tolerance note (pre-existing, also present upstream).** At the default
> `M=128 N=4096 K=4096` shape the gate reports `max_abs_diff=0.125` against the
> absolute `--tol 1e-2` → **FAIL**, but this is *not* an int4 bug. It's one f16
> output ULP (0.0625–0.125 at outputs of magnitude ~128–256), produced by the
> WMMA f32-accumulation order differing from numpy's; relative error is ~0.08 %.
> Confirmed: the *same* kernel at `K=256` passes (`max_abs 1.6e-6`), and a pristine
> upstream checkout reproduces the `0.125` identically. The fix is a relative
> tolerance in the verify (as the int8 scripts use) — a harness change, deferred.

## Step 03 — int8 storage / f16 compute (`03_int8_pathb_verify.py`)

`instances/gfx1151/wmma_gemm_int8.py`. Loads `<16 x i8>` A/B fragments, converts
each element `i8 → sext → sitofp → f16` (lossless for |x|≤127), runs the verified
f16 WMMA, folds `scale_a*scale_b` into the epilogue. **No DSL core change** — it
reuses the proven f16 path. Random asymmetric small int8 + `np.allclose`. Result:
**PASS**.

## Relationship to the native-int path (true int8 → int32)

True int8 *compute* on gfx1151 — `int8×int8→int32` via `v_wmma_i32_16x16x16_iu8`
(and iu4) — is implemented **upstream** (#8091, "native int pipeline"):

- atom: `wmma_i32_16x16x16_iu8` / `wmma_i32_16x16x16_iu4` in the DSL core
  (`core/arch`, `core/ir.py` generalized `mma()`, `core/isa/backend.py`,
  `core/lower_llvm.py`);
- instance: `instances/gfx1151/wmma_gemm_iu8.py` (int8 in, **int32 out**, i32-packed
  operand pointers);
- probe / example: `examples/gfx1151/wmma_iu8_probe.py`,
  `examples/gfx1151/wmma_gemm_compare_orders.py`;
- runner: `_gemm_iu8_problem` in `run_manifest.py`.

So this `gemm/` dir is the **f16-path quant ladder**; the native-int path is the
upstream sibling. The two are complementary: native-int outputs raw i32; this dir's
int8 rung outputs dequantized f16.

### Deferred follow-on (preliminary results retained)

A standalone exploration (on a pre-#8091 base) built an **f16-dequant-output**
true-int8 GEMM and an A-vs-B throughput harness comparing it against the
int8-storage/f16-compute kernel. That work is **deferred** for re-implementation on
upstream's atom (its iu8 GEMM is i32-out; an f16-dequant-output variant + an
adaptive-timing A/B bench are the planned additions). Preliminary A/B numbers from
that exploration (hardened timing, median of 7), kept so the finding isn't lost —
**K, not arithmetic intensity, drives the win**:

| Shape (M×N×K) | regime | A/B (true-int8 / int8→f16) |
|---|---|---|
| 256×256×16384 | K-heavy | ~1.96× |
| 512×512×8192 | K-heavy | ~1.93× |
| 1024×1024×1024 | balanced | ~1.49× |
| 2048×2048×2048 | balanced-large | ~1.21× |
| 8192×512×512 | tall-skinny | ~1.15× |
| 4096×4096×512 | wide-MN | ~1.05× |

K-heavy shapes approach the ideal 2× (many WMMAs/wave amortize the shared per-wave
overhead + true-int8 skips the per-element `sext→sitofp→cast`); wide/skinny shapes
are load/epilogue-bound where both pay the same cost. (512×512×8192 and
4096×4096×512 have near-equal AI ≈ 480 but 1.93× vs 1.05× — AI doesn't predict it.)

## What this doesn't do (and why)

- **No per-row / asymmetric quantization (zero-point).** The int8 rung is
  per-tensor symmetric — enough to exercise the path; per-channel is a follow-on.
- **No int8-output requantization** (the C++ `14_gemm_quantization` `Mul_Clamp`).
  The int8 rung outputs f16; the native-int upstream path outputs i32.
- **No LDS staging / multi-tile-per-wave tuning.** Correctness-first reference
  kernels far from peak.

## File map

```
ck_dsl/examples/gfx1151/gemm/
├── README.md                              # this file
├── scripts/
│   ├── 01_f16_verify.py                   # f16 baseline (run_manifest verify)
│   ├── 02_int4_matmul_nbits_verify.py     # int4 weight-only W4A16
│   └── 03_int8_pathb_verify.py            # int8 storage / f16 compute
└── data/
    └── 0N_*.json                          # per-script result captures
```

Instances under test: `instances/gfx1151/{wmma_gemm, wmma_gemm_int8}.py` and
`instances/common/matmul_nbits.py`. The true-int8 native path
(`instances/gfx1151/wmma_gemm_iu8.py` + the iu8/iu4 core atom) is upstream's.

## CK example that inspired the int8 work

| CK path | what it gave us |
|---|---|
| `example/14_gemm_quantization/gemm_wmma_quantization_int8.cpp` | the true-int8 WMMA target (i8×i8→i32) — realized upstream as `wmma_i32_16x16x16_iu8` |
| `include/ck_tile/core/arch/mma/wmma/wmma_gfx11.hpp` | the gfx11 iu8 builtin signature (signedness/clamp args, i32-packed operands) |
