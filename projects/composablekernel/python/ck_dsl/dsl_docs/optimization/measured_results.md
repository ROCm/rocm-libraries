# Measured Results From The Validation Pass

This page records the measurements that landed during the documentation validation pass on this checkout. All numbers are MI355X / gfx950 / ROCm 7.0.2 / torch 2.8.0+rocm7.0.2.git245bf6ed / Python 3.12.3. Reproduction commands are below each table; run them from the Composable Kernel repository root with the Python interpreter for your ROCm environment.

These numbers are smoke-grade — they confirm the kernels build, verify, and reach a sane TFLOPS / latency band. They are **not** the hero numbers; for that, run the full `examples/attention/parity_unified_attention.py --attempts 10 --warmup 5` harness or the `RUNBOOK_COMPLIANCE.md` empirical sweeps with median + spread reporting.

## Static Unit Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  python python/test/test_ck_dsl.py
```

Result on this checkout: **286 tests, OK** in ~1.7 s. Covers IR construction, LLVM lowering text shape, transform DAG, helpers, and instance smoke tests.

## Documentation Verifier

`python/ck_dsl/dsl_docs/development/verify_dsl_docs.py` imports every symbol referenced by this documentation tree, exercises every IR builder method, lowers every spec dataclass to LLVM / HIP / CK Tile, builds HSACO via libamd_comgr, and launches small kernels via `KernelLauncher`, `PipelineLauncher`, and `time_launches`. It exits non-zero if any check fails.

```bash
PYTHONPATH=python python \
  python/ck_dsl/dsl_docs/development/verify_dsl_docs.py
```

Result on this checkout: **PASS: 50, FAIL: 0** in ~2 s on MI355X. Coverage includes:

- All public `ck_dsl`, `ck_dsl.helpers`, `ck_dsl.runtime.launcher`, and `ck_dsl.instances` symbols import cleanly.
- Every documented `IRBuilder` method (arith, math, vector, wave / cross-lane, LDS, buffer, MFMA, fp8/bf8/i8 conversion, `s_waitcnt`, `scf_for_iter`, `scf_if`, etc.) emits LLVM IR.
- Buffer-resource ops emit the `0x00027000` (`i32 159744`) DW3 flag verified in the LLVM text.
- `s_waitcnt(vmcnt=16, lgkmcnt=16)` emits the documented encoded `i32 20336`.
- All seven shipped MFMA atoms (`mfma_f32_{16x16x16,16x16x32,32x32x8,32x32x16,4x4x4}_f16` and `mfma_f32_{16x16x16,16x16x32}_bf16`) lower.
- `UniversalGemmSpec` and `ImplicitGemmConvSpec` lower to LLVM, HIP, *and* CK Tile C++.
- All small-op specs (`elementwise / reduce / layernorm / rmsnorm / transpose`) plus `BatchedGemmSpec`, `GroupedGemmSpec`, `DirectConv16cSpec`, `DirectConv4cSpec`, and `UnifiedAttention{2D,3D,Reduce}Spec` build.
- `compile_kernel(...)` returns valid HSACO with the documented `timings` keys.
- Transform-DAG `pad / pad_dynamic / embed / unmerge / merge / indirect / pass_through` all work end-to-end.
- `WorkspacePool`, `DeviceMem`, `KernelLauncher`, `PipelineLauncher`, `time_launches`, `no_fence`, and the sync/release helpers all execute on the GPU.
- The fusion subsystem (`FusedEpilogue`, `BiasAdd`, `ReLU`, `explain_fn`, `default_lowering_registry`, `FusionLegalizer`, `GreedyFusionScheduler`) constructs and walks an FX graph.
- The autotuner (`Autotuner`, `AutotuneConfig`, `AutotuneResult`, `make_autotune_key`, `spec_replace`) constructs.
- Quantization (`quant_max_abs`, `quant_ir_type`, `ir_to_qdtype`, `quantize_scalar_f32`, `dequantize_scalar_to_f32`) round-trips through IR.
- Chiplet swizzle (`chiplet_transform_chunked` + `NUM_XCDS_MI300X/MI325X/MI350X == 8`) emits IR.

A small handful of doc inaccuracies surfaced during this verification and were corrected in-place:

- `BiasAdd` kwargs are `(param_name, dtype)` (was wrongly `bias_dtype` in `fusion/overview.md`).
- `pack_f32_to(b, scalars_f32, *, dtype)` and `store_vec(b, ptr, idx, value, *, n)` are keyword-only with `dtype`/`n`, no `target_dtype` (was wrong in `primitives/intrinsics_and_primitives.md`).
- `BatchedGemmSpec` and `GroupedGemmSpec` both require `name` (and `GroupedGemmSpec` also `trait`).
- `DirectConvProblem` fields are `(N, H, W, groups, cpg, kpg, KH, KW, PAD, stride)` — not the `(Hi, Wi, R, S, sH, sW, pH, pW, dH, dW)` shape used by `ConvProblem` (was wrong in `instances/convolution.md`).
- `QDType` is `Literal["i8", "fp8e4m3", "bf8e5m2"]`, not an enum; values are plain strings (was wrong in `primitives/quantization.md`).

## Generated Example Harness

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  python python/test/test_ck_dsl_examples.py
```

Result: **1 test (multi-subtest), OK** in 204.231 s.

The harness discovers every `example/ck_tile/dsl/<N>_*/gen.py` with an adjacent `expected.json`, builds HSACO + manifest in a subprocess, runs `python -m ck_dsl.run_manifest --verify`, and asserts:

- bit-exact (max_abs_diff = 0) on rounded-input GEMM hero shapes;
- `bad = 0` on conv tolerance (1e-2);
- declared TFLOPS / GB/s lower bounds where present in `expected.json`.

## Implicit-GEMM Conv (Bake-Off 1)

Build + verify in one shot from the README-style entry:

```bash
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
PYTHONPATH=python python \
  -m ck_dsl.examples.bake_off_implicit_gemm --output-dir "$OUT_DIR"
PYTHONPATH=python python \
  -m ck_dsl.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
```

Output:

```text
emitted $OUT_DIR/ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle.hsaco
  (7656 bytes) in 17.21 ms total
  ir_build=0.6ms  lower=0.4ms  comgr_bc=12.8ms  reloc=1.8ms  exe=1.5ms
verify max_abs_diff=7.6293945e-06  bad=0/1605632
Perf: 0.00802084 ms, 230.61 TFlops, 809.922 GB/s
```

Shape: N=8, Hi=Wi=56, C=K=64, R=S=3, pad=1, stride=1, dilation=1. Implicit-GEMM (m=N*Ho*Wo=25088, n=K=64, k=R*S*C=576). Atom: `mfma_f32_32x32x16_f16`. Pipeline `mem`, epilogue `cshuffle`.

`bad = 0` at the conv tolerance (1e-2). `max_abs_diff = 7.6e-06` against the fp32 NumPy reference is within expected fp16-sum noise.

## Distribution-Driven Helpers

```bash
PYTHONPATH=python python \
  python/ck_dsl/examples/distribution_reduce_demo.py --M 32 --N 4096
# -> distribution-driven reduce  M=32 N=4096 bs=256 vec=8  max_abs=0.000e+00

PYTHONPATH=python python \
  python/ck_dsl/examples/distribution_2d_add_demo.py --H 64 --W 128
# -> 2D distribution-driven add  H=64 W=128 tile=(32,64) vec=8  max_abs=0.000e+00
```

Both demos go through the full `TileDistributionEncoding -> make_static_tile_distribution -> load_tile/store_tile -> KernelLauncher` path. Bit-exact vs `torch.sum(dim=-1)` and `A + B` respectively.

## CK Tile Small-Op Parity

```bash
PYTHONPATH=python python \
  python/ck_dsl/examples/ck_tile_parity.py --op all
```

| op                                          | max_abs    | CK lat | torch ref | speedup | ok |
|---------------------------------------------|-----------:|-------:|----------:|--------:|----|
| elementwise.add                             | 0.000e+00  | 12.4us | -         | -       | OK |
| elementwise.sub                             | 0.000e+00  | 10.1us | -         | -       | OK |
| elementwise.mul                             | 0.000e+00  | 12.6us | -         | -       | OK |
| elementwise.max                             | 0.000e+00  | 11.4us | -         | -       | OK |
| elementwise.min                             | 0.000e+00  | 10.1us | -         | -       | OK |
| elementwise.copy                            | 0.000e+00  |  9.8us | -         | -       | OK |
| elementwise.neg                             | 0.000e+00  | 12.5us | -         | -       | OK |
| elementwise.abs                             | 0.000e+00  | 10.3us | -         | -       | OK |
| elementwise.relu                            | 0.000e+00  | 10.8us | -         | -       | OK |
| elementwise.silu                            | 1.221e-04  | 10.3us | -         | -       | OK |
| elementwise.gelu_tanh                       | 6.104e-05  | 10.4us | -         | -       | OK |
| elementwise.exp2                            | 0.000e+00  | 10.0us | -         | -       | OK |
| layernorm2d.512x4096                        | 1.953e-03  | 10.9us |    28.3us |   2.60x | OK |
| rmsnorm2d.512x4096                          | 1.953e-03  | 13.6us |    47.4us |   3.50x | OK |
| reduce.sum.512x4096                         | 3.052e-05  | 10.6us |   730.5us |  69.05x | OK |
| reduce.max.512x4096                         | 0.000e+00  | 10.3us |    16.1us |   1.56x | OK |
| reduce.mean.512x4096                        | 0.000e+00  |  9.9us |    15.2us |   1.53x | OK |
| transpose2d.1024x1024                       | 0.000e+00  |  9.8us |     7.9us |   0.81x | OK |
| batched_gemm.B16_512x512x128                | 3.125e-02  | 12.1us |    20.2us |   1.67x | OK |
| grouped_gemm.5                              | 3.125e-02  | 53.3us |    96.9us |   1.82x | OK |

All passes within the per-op tolerance table (`examples/ck_tile_parity.py::TOL`):

- elementwise linear ops: bit-exact (`max_abs <= 0`);
- silu / gelu_tanh: `<= 2e-4`;
- layer/rmsnorm: `<= 2.5e-3`;
- reduce: `<= 1.5e-3`;
- transpose: bit-exact;
- batched/grouped GEMM: `<= 7e-2`.

`reduce.sum` is fastest vs torch because the kernel reduces with `block_lds_reduce` instead of going through a multi-pass `torch.sum` reduction.

`transpose2d` is below 1x vs torch's vendor path; this is expected for a single 1024x1024 fp16 transpose where torch's call hits a hand-tuned MIOpen / rocBLAS shim. The DSL version is a portable starting point, not the production target.

## Attention Smoke

```bash
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
export AITER_PATH=<aiter-checkout>
PYTHONPATH="python:${AITER_PATH}" \
  python \
  python/ck_dsl/examples/attention/parity_unified_attention.py \
  --scenario decode_d128_b16 --attempts 1 --warmup 0 --paths auto,2d,3d \
  --report "$OUT_DIR"/ckdsl_attention_smoke.json
```

Scenario `decode_d128_b16`: `head_size=128`, `block_size=16`, `num_query_heads=16`, `num_kv_heads=2`, `seq_lens=[(1,1024),(1,2048),(1,4096),(1,512)]`, fp16.

| Lane         | latency      | max_abs vs ref |
|--------------|-------------:|---------------:|
| triton-auto  | 361259.77 us | 0.0001831      |
| ck-auto      |  32834.37 us | 0.0001831      |
| triton-2d    |  51114.26 us | 0.0001831      |
| ck-2d        |  12191.90 us | 0.0001831      |
| triton-3d    |    203.52 us | 0.0001831      |
| ck-3d        |    132.88 us | 0.0001831      |

Per-lane comparisons (CK / Triton):

```text
auto vs auto    : 11.00x   (Triton picks 2D here, CK picks 3D)
2d vs 2d        :  4.19x
3d vs 3d        :  1.53x
```

These are **one** attempt with zero warmup. The README's published speedups (~1.799x auto geomean over 11 scenarios, ~1.743x 3D vs 3D) come from the mean of 5 full harness runs at 10 attempts each. Single-attempt smoke runs should not be cited as headline numbers; they confirm the kernels build, verify, and land in the expected ballpark.

## Build Timings

Median codegen timings observed during this pass (from `KernelArtifact.timings`):

| Kernel                                | Total  | ir_build | ir_lower_llvm | comgr_bc | reloc | exe   |
|---------------------------------------|-------:|---------:|--------------:|---------:|------:|------:|
| implicit-GEMM conv (08 bake-off)      | 17 ms  |  0.6 ms  |    0.4 ms     |  12.8 ms | 1.8 ms| 1.5 ms|
| elementwise.add                       |  ~5 ms |  <0.5 ms |   <0.5 ms     |   ~3 ms  | ~0.5  | ~0.5  |
| reduce.sum (512x4096)                 |  ~6 ms |  <0.5 ms |   <0.5 ms     |   ~4 ms  | ~0.5  | ~0.5  |

These are warm-comgr numbers; the first call in a process pays an extra ~50-200 ms for `libamd_comgr` load.

## What This Validates

- IR construction, IR printing, and LLVM lowering for the IR vocabulary used by all shipped instances.
- libamd_comgr in-process compile path for at least the f16/bf16 stack and the AMDGPU intrinsics declared in `core/lower_llvm.py::_INTRINSIC_DECLS`.
- HIP module load + launch + persistent launcher + workspace path.
- All 13 production instance builders (GEMM, batched GEMM, grouped GEMM, implicit-GEMM conv, direct grouped 16c/4c, img2col, pooling, elementwise, reduce, layernorm, rmsnorm, transpose, scalar 2D/3D attention, tiled 2D/3D attention, attention reduce).
- The transform DAG: `pad`, `pad_dynamic`, `embed`, `unmerge`, `merge`, `pass_through`, `indirect`.
- `TileDistributionEncoding` and the distribution-driven load/store path.
- The chiplet swizzle helper (used by GEMM with `chiplet_swizzle=True`).

## What This Does Not Validate

- BF16 / FP8 / I8 paths beyond what the small-op parity covers.
- Long-context / large-batch attention performance (need full harness with multiple attempts).
- `chiplet_swizzle=True` perf impact (verified only at correctness level here).
- The fusion subsystem end-to-end (covered by `helpers/fusion_validation.py::run_fusion_validation_matrix`).
- The autotuner over a real sweep (`helpers/autotune.py::autotune_sweep`).
- Sweep + median + spread benchmarks (see `sweep_bench.py`).

These are listed not as gaps in the DSL, but as places where additional measurement would strengthen the docs.

## How To Re-Run

Single command to reproduce the full validation pass in this doc:

```bash
cd <composablekernel-checkout>
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python python/test/test_ck_dsl.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python python/test/test_ck_dsl_examples.py

PYTHONPATH=python python -m ck_dsl.examples.bake_off_implicit_gemm --output-dir "$OUT_DIR"
PYTHONPATH=python python -m ck_dsl.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify

PYTHONPATH=python python python/ck_dsl/examples/distribution_reduce_demo.py --M 32 --N 4096
PYTHONPATH=python python python/ck_dsl/examples/distribution_2d_add_demo.py --H 64 --W 128
PYTHONPATH=python python python/ck_dsl/examples/ck_tile_parity.py --op all

export AITER_PATH=<aiter-checkout>
PYTHONPATH="python:${AITER_PATH}" python \
  python/ck_dsl/examples/attention/parity_unified_attention.py \
  --scenario decode_d128_b16 --attempts 1 --warmup 0 --paths auto,2d,3d
```
