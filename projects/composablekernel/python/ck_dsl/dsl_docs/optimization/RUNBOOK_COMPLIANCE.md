# RUNBOOK_COMPLIANCE.md — Optimization Runbook ⇄ DSL Primitive Mapping

This doc maps each section of the `gpu-op-optimization-runbook` Cursor skill
to the DSL primitive(s) that implement it. Use it as a checklist
when authoring a new kernel: "section X says use lever Y → use
helper Z".

## §1 Define the problem precisely

| Runbook | DSL primitive |
|---|---|
| Problem dataclass with shape, dtype, layout | `ConvProblem`, `DirectConvProblem`, plus GEMM `TileSpec`/`DataSpec` and manifest shape fields |
| FLOPs and bytes count | `.flops` property on problem dataclasses |
| Expected memory footprint | `_in_size/_wei_size/_out_size` helpers in launcher's `Problem` subclasses |

## §2 Baselines and limits

| Runbook | DSL primitive |
|---|---|
| Memory-bound floor (B / peak HBM BW) | `ConvFp16Problem::metrics()` returns `gbps` |
| Compute floor (FLOPs / peak fp16 MFMA throughput) | `ConvFp16Problem::metrics()` returns `tflops` |
| Achieved-vs-peak comparison | `ck/dsl/ck_dsl_current_results.md` lays this out per kernel |

## §3 Bottleneck classification

| Runbook | DSL primitive |
|---|---|
| Static IR/ISA inspection | `ck_dsl.analysis.analyze_llvm_ir`, `ck_dsl.analysis.analyze_hsaco` |
| Profiling hooks | Use `rocprof` / `omniperf` against the HSACO produced by `write_artifact(...)`. The launcher's HIP-graph mode amortizes launch overhead so the profile reflects steady-state. |

## §4 Algorithmic mapping

| Runbook | DSL primitive |
|---|---|
| §4.1 GEMM | `instances/gemm_universal.py` (full dispatcher schema) |
| §4.2 Convolution as implicit GEMM | `instances/conv_implicit_gemm.py` + `transforms.py` DAG |
| §4.2 Convolution direct (streaming row) | `instances/conv_direct_grouped.py` (16c and 4c bake-off paths) |
| §4.3 Attention | `instances/attention_unified.py` scalar semantics + `attention_tiled_{2d,3d}.py` MFMA kernels; paged-KV addressing uses `transforms.indirect(...) + unmerge(...)` |
| §4.4 Reductions | `instances/reduce.py`, `layernorm2d.py`, `rmsnorm2d.py`; `block_lds_reduce` + sweep helpers |

## §5 Work decomposition

| Runbook | DSL primitive |
|---|---|
| Block/Warp grid | `WarpGrid` (`ck_dsl.helpers.geometry`) — packs tile + warp grid + bound `tid/lane/warp_*/block_*_off` SSA into one immutable view; `TileSpec` (`ck_dsl.instances.gemm_universal`) is the dispatcher-schema view |
| Per-warp MFMA tile | `MfmaAtom` + `WarpGrid.mfmas_per_warp_m/n/k_atoms_per_tile_k` derived properties |
| Lane-output mapping | `MfmaAtom.lane_to_output(b, lane, i)` for 16x16, 32x32, 4x4 atoms |
| Compile-time loops | `IRBuilder.static_for(...)` / `IRBuilder.unroll(...)` for Python-time unrolling |
| Python-time vs runtime branch split | `Value.__bool__` raises; `IRBuilder.static_if(...)` for host branches, `IRBuilder.scf_if(...)` for runtime branches |
| Persistent kernel | not yet implemented (Phase 2 of the helpers roadmap) |
| Split-K | not yet implemented as a TileSpec field; primitives exist (`global_atomic_add_f32`) |

## §6 Memory hierarchy

| Runbook | DSL primitive |
|---|---|
| §6.1 Buffer rsrc OOB clamp | `buffer_rsrc(ptr, num_bytes)` IR op (DW3 = 0x00027000); verify with `analyze_llvm_ir` / `analyze_hsaco` |
| §6.1 Tail-safe loads via sentinel offset | `select(valid, real_off, oob_sentinel)` pattern, usually with `TensorDescriptor.offset(...)->valid` from `pad` / `pad_dynamic` |
| §6.2 Wide vector loads/stores | `buffer_load_vN_f16`, `buffer_store_vN_f16`, `global_store_vN_f16` (N ∈ {1,2,4} dwords) |
| §6.3 Async DRAM→LDS | `AsyncTileLoader` (wraps `async_buffer_load_lds_addr` with per-wave LDS base) |
| §6.3 LDS bank-conflict avoidance | `ImplicitGemmConvSpec.lds_k_pad` (default `+8` sync, `0` async) plus manual consumer-side read swizzle when async requires packed LDS |
| §6.4 LDS double-buffer | `pipeline="compv4"` in `UniversalGemmSpec`; second `smem_alloc` for the ping-pong slot |

## §7 Matrix-instruction selection

| Runbook | DSL primitive |
|---|---|
| §7.1 MFMA atom shape choice | `MfmaAtom.f16_16x16x16`, `f16_16x16x32`, `f16_32x32x8`, `f16_32x32x16`, `f16_4x4x4` |
| §7.2 K-pack atoms (wider K per atom) | `f16_16x16x32` (vs 16x16x16), `f16_32x32x16` (vs 32x32x8) |
| §7.3 Scheduler hints | `sched_group_barrier(mask, count, sync_id)`, `s_setprio`, `s_waitcnt` |
| §7.3 Pipeline interleave | `SchedulePolicy.for_pipeline(...)` emits canonical DS_READ/MFMA group hints |
| Attention online softmax math | `OnlineSoftmaxState`, `warp_xor_reduce_*`, `apply_softcap_log2`, `exp2`, `rcp`, `fmax` |

## §8 Pipelining

| Runbook | DSL primitive |
|---|---|
| Loop with iter_args | `b.scf_for_iter(lower, upper, step, init_vars, ...)` |
| Manual Python unroll | `b.static_for(...)` / `b.unroll(...)` or direct Python loops in specialized kernels |
| Software pipelining | `SoftwarePipeline.run_ping_pong(...)` for prologue / steady-state / epilogue construction |
| `s_waitcnt` for async DMA gating | `b.s_waitcnt(vmcnt=0)` after `AsyncTileLoader.issue` |

## §9 Epilogue optimization

| Runbook | DSL primitive |
|---|---|
| §9.1 Direct per-lane stores | `DirectEpilogue` (`ck_dsl.helpers.epilogues`) — `vec_in_acc=True` for atoms whose per-lane elements are contiguous (4x4 direct conv); scalar otherwise |
| §9.3 cshuffle (LDS-staged) | `CShuffleEpilogue.from_grid(...)` — LDS-stage + wide `buffer_store_dwordx{2,4}` |
| §9.4 Atomic add for split-K | `global_atomic_add_f32` IR op |
| §9.5 fp32 → fp16 trunc | `vec_trunc_f32_to_f16` IR op (packs 4 f32 → 4 f16 in one fptrunc) |

## §10 Compiler settings

| Runbook | DSL primitive |
|---|---|
| Per-kernel `amdgpu-flat-work-group-size` | `kernel.attrs["max_workgroup_size"]` — emitted automatically by `IRBuilder` for `block_size`-aware kernels |
| Alias/alignment metadata | `IRBuilder.param(..., noalias=True, readonly=True, align=16, dereferenceable=N)` plus load/store `align` attrs |
| Pre-lowering canonicalization | `ck_dsl.core.passes.optimize_kernel` (constant fold, CSE, dead pure-op removal) |
| HIP/clang flags | `runtime/comgr.py` builds with default ROCm 7.0 flags; per-spec overrides via `compile_hints` on the spec object (not yet plumbed) |

## §11 ISA inspection

| Runbook | DSL primitive |
|---|---|
| Disassemble | `analyze_hsaco(hsaco_path).isa` (uses `llvm-objdump -d`) |
| Inspect VGPR/SGPR | `analyze_hsaco(hsaco_path).resources` (uses `llvm-readelf --notes`) |
| Inspect static LDS | `analyze_hsaco(hsaco_path).resources.lds_bytes`; HIP attributes remain optional |

The launcher's `Perf:` line prints TFLOPS and GB/s achieved; cross-
reference with the per-kernel HSACO via `hipFuncGetAttributes()` for
the static analysis.

## §12 Autotuning

| Runbook | DSL primitive |
|---|---|
| Cartesian product over `(TileSpec, TraitSpec)` | `instances.gemm_universal.all_dispatcher_configs(...)` |
| Parallel build with caching | `ck_dsl.sweep` (parallel HSACO build, content-hash-keyed cache) |
| Hygienic benchmark with median + spread | `ck_dsl.benchmark.benchmark_manifest` for one manifest; `ck_dsl.sweep_bench` for cross-config CSV sweeps |
| Hero-subset preselection | `gen.py --subset compute` in `example/.../07_gemm_universal_sweep/` |

## §13 Verification

| Runbook | DSL primitive |
|---|---|
| CPU reference | `ck_dsl.run_manifest` (NumPy fp32 accum; grouped conv aware); legacy C++ launcher remains optional |
| Tolerance | gemm: bit-exact on rounded inputs; conv: `max_abs < 1e-2`; norm/reduce/pointwise/transpose use op-specific NumPy references in `run_manifest` |
| Bit-exact comparison | `python -m ck_dsl.run_manifest ... --verify` (also wrapped by `test_ck_dsl_examples.py`) |
| Cross-check vs torch | conv reference matches `torch.nn.functional.conv2d` semantics |

## Empirical pass — bake-off 1 (Implicit-GEMM conv)

Each lever applied to the implicit-GEMM conv kernel, with the
per-launch TFLOPS that resulted, on MI300X / gfx950:

| Lever | TFLOPS (per-launch) | Notes |
|---|---:|---|
| **baseline** (single-buffer LDS, direct epilogue, 16x16x32) | **111** | `bad=0` at conv tolerance |
| + cshuffle epilogue (§9.3) | 119 | LDS-staged fp16 + wide stores |
| + buffer rsrc DW3=0x00027000 (§6.1) | (correctness fix) | was producing all-zero outputs |
| + 32x32x16 atom (§7.1) | 240 | half the K-loop trip count |
| + K-padded LDS (§6.3) | 248 | break ds_read bank conflicts |
| **graph mode 5x200 (§12)** | **280** | amortize launch overhead |

Cumulative: 111 → 280 TFLOPS = **2.5x** by applying 5 runbook levers
in series. Each lever was empirically verified with a hygienic
benchmark, all preserving `bad=0` correctness at the conv tolerance
against the grouped NumPy reference in `ck_dsl.run_manifest`.

## Empirical pass — bake-off 2 (Direct grouped conv)

Direct grouped conv, `N=32, H=W=200, R=S=3, pad=1`, on MI300X / gfx950:

| Kernel | Lever | Per-launch TFLOPS |
|---|---|---:|
| 16c (`groups=16, cpg=kpg=16`) | baseline (vec=2 direct stores, no K-fold) | ~92 |
| 16c | + K=32 MFMA fold (`mfma_f32_16x16x32_f16` for S=0,1) | ~108 |
| 16c | + wide direct epilogue (1 `buffer_store_dwordx2` per lane = 4 halves) | ~210 |
| 16c | + `BLOCK_GROUPS=4` | **~214** |
| 4c (`groups=64, cpg=kpg=4`) | baseline (4 scalar `buffer_store_short` per lane) | ~44 |
| 4c | + vec2-dword epilogue (1 store/lane, 4 halves fused) | **~48** |

Each lever applied through the helpers: `MfmaAtom.f16_16x16x32` /
`f16_4x4x4`, `TensorDescriptor` / `pad` for H/W-valid input addressing,
and wide vector epilogues via `buffer_store_vN_f16(..., dwords=2)`.

This is the canonical "pull every lever; measure each" example for
the runbook's optimisation methodology — see
`example/ck_tile/dsl/08_bake_off_implicit_gemm/expected.json`,
`09_bake_off_direct_conv_16c/expected.json`, and
`10_bake_off_direct_conv_4c/expected.json` for the gates, and
`ck/dsl/ck_dsl_current_results.md` for the full results.

## CK Tile parity examples

The `example/ck_tile/dsl/` tree now contains Python-generated CK DSL
counterparts for the CK Tile examples that already have DSL instance
builders:

| CK Tile example | CK DSL example | Backing instance |
|---|---|---|
| `02_layernorm2d` | `dsl/02_layernorm2d` | `instances/layernorm2d.py` |
| `05_reduce` | `dsl/05_reduce` | `instances/reduce.py` |
| `10_rmsnorm2d` | `dsl/10_rmsnorm2d` | `instances/rmsnorm2d.py` |
| `16_batched_gemm` | `dsl/16_batched_gemm` | `instances/batched_gemm.py` |
| `21_elementwise` | `dsl/21_elementwise` | `instances/elementwise.py` |
| `37_transpose` | `dsl/37_transpose` | `instances/transpose.py` |
| universal GEMM / conv bake-offs | `dsl/06_*` through `dsl/10_*` | GEMM + conv instances |

`python/test/test_ck_dsl_examples.py` discovers these `gen.py`
wrappers, builds each HSACO, runs `ck_dsl.run_manifest --verify`, and
optionally checks any declared TFLOPS/GB/s lower bounds.
