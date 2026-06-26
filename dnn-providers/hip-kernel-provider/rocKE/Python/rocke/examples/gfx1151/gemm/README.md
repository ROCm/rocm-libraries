# gfx1151 WMMA GEMM — the quantization/precision ladder

A self-contained study of WMMA GEMM on **gfx1151 (RDNA3.5 / Strix Halo APU,
wave32)** across the precision ladder the hardware supports — from the f16
baseline through int4 weight-only to **int8**, with the two int8 compute paths
compared head-to-head — each rung verified for correctness on silicon.

The point is to show, end to end, (1) the **one-wave-per-16×16-tile WMMA GEMM**
skeleton every rung shares, (2) the two ways to do int8 — *int8 storage / f16
compute* (dequant to f16, reuse the f16 atom) vs *true int8 compute* (int8×int8→
int32 on the hardware `wmma_i32_16x16x16_iu8` atom, dequant to f16 in the
epilogue), and (3) an honest A/B throughput read of the two.

> RCR layout throughout (A row-major `M×K`, B row-major `N×K`, `C = A @ B.T`),
> one wave (32 lanes) per 16×16 output tile, no LDS — the WMMA fragment ABI does
> the lane distribution. These are minimal **correctness-first reference kernels**
> (4–15 % of peak); ratios, not absolute rates, are the headline.

> **New to the gfx11 WMMA instruction or the lane/fragment ABI?**
> [`ALGORITHM.md`](ALGORITHM.md) derives the kernels from the math up — the GEMM
> spec, the one-wave-per-16×16-tile mapping, the lane→data layout, and how each
> quantization rung (f16 / int8→f16 / true int8 / int4) is one stage swapped in a
> shared skeleton. Read it first to understand *what* the kernels compute before
> reading *how* they were measured below.

## The ladder

| # | rung | A·B dtype | accumulate | atom | status |
|---|---|---|---|---|---|
| 01 | f16 baseline | f16 × f16 | f32 | `wmma_f32_16x16x16_f16` | PASS |
| 02 | int4 weight-only (W4A16) | f16 × int4 | f32 | `wmma_f32_16x16x16_f16` (+ int4 dequant) | PASS¹ |
| 03 | int8 storage / f16 compute ("Path B") | int8→f16 × int8→f16 | f32 | `wmma_f32_16x16x16_f16` | PASS |
| 04 | true int8 / f16 dequant out ("Path A") | int8 × int8 | **int32** | `wmma_i32_16x16x16_iu8` (upstream atom) | PASS |
| 05 | Path A vs Path B throughput | — | — | — | A wins 1.13–2.51× |

¹ See the int4 tolerance note under Step 02.

**Path A vs Path B** is the central comparison: same int8 quantization and the
same f16 output, different *compute*. Path B converts int8→f16 in the K-loop and
uses the verified f16 WMMA (no DSL core change; win is memory/bandwidth). Path A
keeps int8 all the way into the tensor core via `v_wmma_i32_16x16x16_iu8` (int32
accumulate), dequantizing to f16 only in the epilogue (~2× the tensor-core
ceiling).

## Hardware / software pin

| | |
|---|---|
| GPU | Radeon 8060S / **gfx1151** (RDNA3.5, Strix Halo APU), wave32 |
| OS | **Windows-native** (this box) |
| comgr / hip | `C:\Windows\System32\amd_comgr_3.dll`, `amdhip64_7.dll` (versioned driver DLLs) |
| f16 WMMA peak | ~59 TFLOP/s |
| int8 WMMA peak | ~118 TOPS (≈2× f16) |
| CK DSL | this repo (`dnn-providers/hip-kernel-provider/rocKE/Python/rocke`) |

## Reproduce

```bash
cd <rocke>/examples/gfx1151/gemm          # this folder
# Windows-native env: point the loader at the driver DLLs (no PYTHONPATH needed —
# the scripts add the python root themselves):
export ROCKE_COMGR_LIB="C:\\Windows\\System32\\amd_comgr_3.dll"
export ROCKE_HIP_LIB="C:\\Windows\\System32\\amdhip64_7.dll"

python scripts/01_f16_verify.py                       # f16 baseline
python scripts/02_int4_matmul_nbits_verify.py         # int4 weight-only (W4A16)
python scripts/03_int8_pathb_verify.py                # int8 storage / f16 compute
python scripts/04_iu8_dequant_verify.py --m 16 --n 16 --k 16   # true int8 -> f16
python scripts/04_iu8_dequant_verify.py --m 128 --n 128 --k 128
python scripts/05_int8_perf_a_vs_b.py                 # A-vs-B throughput suite
```

Each script writes its result to `data/0N_*.json`.

## Step 01 — f16 baseline (`01_f16_verify.py`)

The reference WMMA GEMM (`instances/gfx1151/wmma_gemm.py`). Builds, writes a gemm
manifest, and verifies via `rocke.run_manifest --verify` against a numpy
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
> Confirmed: the *same* kernel at a smaller `K` passes well under the gate, and a
> pristine upstream checkout reproduces the `0.125` identically. The fix is a relative
> tolerance in the verify (as the int8 scripts use) — a harness change, deferred.

## Step 03 — int8 storage / f16 compute, "Path B" (`03_int8_pathb_verify.py`)

`instances/gfx1151/wmma_gemm_int8.py`. Loads `<16 x i8>` A/B fragments, converts
each element `i8 → sext → sitofp → f16` (lossless for |x|≤127), runs the verified
f16 WMMA, folds `scale_a*scale_b` into the epilogue. **No DSL core change** — it
reuses the proven f16 path. Random asymmetric small int8 + `np.allclose`. Result:
**PASS**.

## Step 04 — true int8 / f16 dequant out, "Path A" (`04_iu8_dequant_verify.py`)

`instances/gfx1151/wmma_gemm_iu8_dequant.py`. Runs the hardware
`wmma_i32_16x16x16_iu8` atom (int8×int8 → **int32** accumulate), then dequantizes
the i32 accumulator to f16 in the epilogue (`f16 = trunc(sitofp(acc) *
scale_a*scale_b)`). A/B are the *same int8 bytes* as Path B, read as i32-packed
(4 int8/i32 — the iu8 fragment ABI); C is f16. Random asymmetric int8 +
`np.allclose`. Result: **PASS** (`data/04_iu8_dequant_verify.json`: 256×128×256,
`max_abs_diff ≈ 7.7e-7`, `bad=0`) — the i32 accumulation is exact, so the only
error is the final f16 store rounding.

> The `wmma_i32_16x16x16_iu8` atom (and iu4) is **upstream's** native-int landing
> (#8118): the atom lives in the DSL core (`core/arch`, the generalized `mma()` in
> `core/ir.py`, `core/isa/backend.py`, `core/lower_llvm.py`), and upstream ships an
> i32-**output** GEMM at `instances/gfx1151/wmma_gemm_iu8.py` + a probe at
> `examples/gfx1151/wmma_iu8_probe.py`. This step is the **f16-dequant-output**
> sibling built on that atom — a usable quantized-GEMM result (matching the C++
> `14_gemm_quantization` `Mul_Clamp` intent) and the f16-out counterpart that makes
> the A/B comparison apples-to-apples.

## Step 05 — Path A vs Path B throughput (`05_int8_perf_a_vs_b.py`)

Both kernels take the same int8 bytes and output f16 with an identical ABI, so
they're built once and timed back-to-back over a roofline-tagged perf suite.
**Methodology is differentiated from correctness**: one correctness gate up front
(128³), then **adaptive timing** (auto-ramp the iteration count until the measured
interval clears `--min-ms`, then median of `--reps` with min..max spread). Shapes
too small to clear the floor are flagged `[!] launch-bound` rather than reported as
a confident rate.

Result (median of 7; Path A = true-int8→f16, Path B = int8→f16 compute):

| Shape (M×N×K) | regime | AI (op/B) | A TOP/s | B TOP/s | **A/B** |
|---|---|---|---|---|---|
| 512×512×8192 | K-heavy | 482 | 9.64 | 3.84 | **2.51×** |
| 256×256×16384 | K-heavy-narrow | 252 | 13.23 | 6.03 | **2.19×** |
| 1024×1024×1024 | balanced | 512 | 4.98 | 3.08 | 1.62× |
| 8192×512×512 | tall-skinny | 334 | 5.36 | 4.25 | 1.26× |
| 2048×2048×2048 | balanced-large | 1024 | 4.76 | 4.06 | 1.17× |
| 4096×4096×512 | wide-MN | 455 | 6.61 | 5.83 | 1.13× |

**K, not arithmetic intensity, drives the win.** K-heavy shapes meet/exceed the
ideal 2× — many WMMAs per wave amortize the fixed per-wave overhead (address setup
+ 8 epilogue stores, identical for both), and Path A additionally skips Path B's
per-element `sext→sitofp→cast`. Wide-M·N / tall-skinny shapes barely move
(1.13–1.26×): many tiles each do few WMMAs, so the shared load+epilogue cost swamps
the compute difference. Note 512×512×8192 and 4096×4096×512 have near-equal AI
(~470) but 2.51× vs 1.13× — AI doesn't predict it, K does. (Absolute rates carry
GPU-clock/thermal jitter; the A/B ratio measured back-to-back is the stable signal.)

## What this doesn't do (and why)

- **No per-row / asymmetric quantization (zero-point).** Both int8 rungs are
  per-tensor symmetric — enough to exercise the compute paths; per-channel is a
  follow-on.
- **No int8-output requantization.** Both int8 rungs output f16 (the useful
  dequantized form); upstream's `wmma_gemm_iu8.py` is the raw-i32-output sibling.
- **No LDS staging / multi-tile-per-wave tuning.** Correctness-first reference
  kernels far from peak; a compute-bound rewrite would widen the A/B gap on the
  balanced shapes.

## File map

```
rocke/examples/gfx1151/gemm/
├── README.md                              # this file (the ladder + A-vs-B throughput study)
├── ALGORITHM.md                           # the math-up derivation: GEMM spec, WMMA ABI, the rungs
├── scripts/
│   ├── 01_f16_verify.py                   # f16 baseline (run_manifest verify)
│   ├── 02_int4_matmul_nbits_verify.py     # int4 weight-only W4A16
│   ├── 03_int8_pathb_verify.py            # int8 storage / f16 compute
│   ├── 04_iu8_dequant_verify.py           # true int8 -> f16 dequant (upstream iu8 atom)
│   └── 05_int8_perf_a_vs_b.py             # A-vs-B throughput suite (hardened timing)
└── data/
    └── 0N_*.json                          # per-script result captures
```

Instances under test: `instances/gfx1151/{wmma_gemm, wmma_gemm_int8,
wmma_gemm_iu8_dequant}.py` and `instances/common/matmul_nbits.py`. The
`wmma_i32_16x16x16_iu8` core atom and the raw-i32-output `wmma_gemm_iu8.py` are
upstream's (#8118); step 04 builds the f16-dequant variant on that atom.

## CK example that inspired the int8 work

| CK path | what it gave us |
|---|---|
| `example/14_gemm_quantization/gemm_wmma_quantization_int8.cpp` | the true-int8 WMMA target (i8×i8→i32) — the atom is upstream's `wmma_i32_16x16x16_iu8`; step 04 adds the `Mul_Clamp`-style dequant-to-f16 epilogue |
| `include/ck_tile/core/arch/mma/wmma/wmma_gfx11.hpp` | the gfx11 iu8 builtin signature (signedness/clamp args, i32-packed operands) |
```
