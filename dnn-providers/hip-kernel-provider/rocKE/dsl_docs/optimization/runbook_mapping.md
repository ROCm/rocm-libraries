# Optimization Runbook Mapping

This page maps the `gpu-op-optimization-runbook` Cursor skill to concrete `ck_dsl` practices and primitives.

The runbook's core rule applies directly:

```text
Do not optimize before correctness, measurement hygiene, and bottleneck classification.
```

In `ck_dsl`, most useful optimizations are explicit choices in specs, helpers, or IR primitives.

## 1. Define The Problem Exactly

Runbook asks for:

- operation contract;
- shapes;
- layouts;
- dtypes;
- boundary behavior;
- side effects;
- tolerance.

DSL locations:

- `ConvProblem`, `DirectConvProblem`, `PoolingProblem`
- `UnifiedAttentionProblem` (selects 2D / 3D via `select_2d_config` / `select_3d_config` / `use_2d_kernel`)
- `TileSpec`, `TraitSpec`, `DataSpec`, `UniversalGemmSpec`
- small-op specs: `Reduce2DSpec`, `LayerNorm2DSpec`, `RMSNorm2DSpec`, `ElementwiseSpec`, `Transpose2DSpec`
- `IOSpecRule` + `validate_io` in `helpers/spec.py`
- manifest metadata (`make_*_manifest`, schema `ck.dsl.example.manifest/v1`)

Checklist:

```text
shape names are explicit
layouts are named
dtypes are explicit
tails/padding/masks are specified
reference path exists
FLOPs/bytes are computable
```

## 2. Establish Baselines

Runbook asks for correctness and performance baselines.

DSL tools:

- `run_manifest.py --verify`;
- NumPy references in manifest runner;
- torch/AITER references for attention paths;
- CK Tile parity examples;
- `benchmark_manifest`;
- `sweep_bench`;
- `time_launches`.

Rules:

- verify first;
- include vendor/library baseline when meaningful;
- do not compare different algorithms without labeling;
- separate compile/module-load from kernel timing;
- report median and spread.

## 3. Bottleneck Classification

Runbook asks for arithmetic intensity, profiler counters, and stall analysis.

DSL tools:

- `analyze_llvm_ir(artifact.llvm_text)`;
- `analyze_hsaco(hsaco_path)`;
- `llvm-objdump`;
- `llvm-readelf`;
- `rocprof`/Omniperf on manifest or persistent launcher runs.

Map symptoms to likely levers:

```text
low MFMA count or wrong atom       -> MfmaAtom/spec issue
high VGPR                          -> atom shape, accumulator count, unroll, cshuffle staging
high LDS bytes                     -> tile size, cshuffle, double buffering
LDS stalls/bank conflicts          -> LdsLayout padding/swizzle/read formula
memory stalls                      -> async load, larger tiles, cache policy, vector width
store bottleneck                   -> cshuffle or direct vector epilogue
launch bound                       -> PipelineLauncher, graph/amortized timing, fusion
```

## 4. Algorithm Mapping

Runbook sections map to instances:

```text
GEMM
  -> instances/gemm_universal.py
  -> instances/batched_gemm.py
  -> instances/grouped_gemm.py

Convolution as implicit GEMM
  -> instances/conv_implicit_gemm.py
  -> transforms.py descriptor DAG

Direct convolution
  -> instances/conv_direct_grouped.py

Attention
  -> instances/attention_unified.py
  -> instances/attention_tiled_2d.py
  -> instances/attention_tiled_3d.py
  -> helpers/attention.py

Reductions/norms
  -> reduce.py, layernorm2d.py, rmsnorm2d.py
  -> block_lds_reduce

Transpose
  -> transpose.py
```

## 5. Work Decomposition

Runbook asks how work maps to blocks, warps, lanes, and registers.

DSL primitives:

- `TileSpec`;
- `WarpGrid`;
- `MfmaAtom`;
- `TileDistributionEncoding`;
- `sweep_row_chunks`;
- grid helper functions;
- chiplet swizzle helper.

Questions to answer:

- How many output elements does one CTA own?
- How many waves per CTA?
- Which tile subregion does each wave own?
- Which accumulator slots does each lane own?
- How many CTAs are needed to fill the GPU?
- Does the grid expose enough parallelism for the shape?

## 6. Memory Hierarchy

Runbook memory levers map to:

```text
buffer rsrc OOB clamp
  -> b.buffer_rsrc
  -> make_buffer_resource
  -> make_buffer_view

wide global/buffer loads and stores
  -> buffer_load_vN_f16
  -> buffer_store_vN_f16
  -> global_load_vN_f16
  -> global_store_vN_f16

async DRAM-to-LDS
  -> AsyncTileLoader
  -> async_buffer_load_lds
  -> raw_ptr_buffer_load_lds

LDS padding and layout
  -> LdsLayout
  -> lds_k_pad
  -> TransposeLdsReader

double buffering
  -> SoftwarePipeline
  -> compv-style pipeline traits
```

Check:

- are invalid lanes OOB-safe before load/store?
- are vector widths aligned?
- does LDS layout avoid bank conflicts?
- does async have correct `s_waitcnt(vmcnt=0)`?
- does double buffering increase LDS enough to reduce occupancy?

## 7. Matrix Instruction Selection

Runbook MFMA levers map to `MfmaAtom`.

Available f16 atoms:

```text
16x16x16
16x16x32
32x32x8
32x32x16
4x4x4
```

Tradeoffs:

- larger atom/output tile can improve MFMA utilization;
- K-packed atoms reduce K-loop trips;
- larger accumulator vectors increase VGPR pressure;
- lane-to-output mapping changes by atom;
- epilogue must match atom layout.

Always verify:

- generated MFMA intrinsic shape;
- A/B operand packing;
- f32 accumulator count;
- numeric tolerance;
- VGPR impact.

## 8. Pipelining

Runbook pipelining maps to:

- `scf_for_iter` for loop-carried values;
- `IRBuilder.unroll` for Python-time unrolling;
- `SoftwarePipeline.run_ping_pong`;
- `AsyncTileLoader`;
- `SchedulePolicy`;
- explicit `s_waitcnt`.

Questions:

- Is the loop compute-bound enough to hide memory?
- Are there enough independent instructions between load and use?
- Is the pipeline prologue/steady/epilogue correct?
- Does added LDS/VGPR reduce occupancy more than overlap helps?

## 9. Epilogue Optimization

Runbook epilogue levers map to:

- `DirectEpilogue`;
- `CShuffleEpilogue`;
- `buffer_store_vN_f16`;
- `global_store_vN_f16`;
- `vec_trunc_f32_to_f16`;
- `global_atomic_add_f32` for future split-K/fixup-style paths.

Decision:

```text
If per-lane accumulator slots map to contiguous output addresses:
  direct vector store can be best.

If accumulator layout is scattered:
  cshuffle LDS staging often wins.
```

Verify output vectorization in LLVM/ISA.

## 10. Compiler Settings

Runbook compiler controls map to:

- `kernel.attrs["max_workgroup_size"]` -> AMDGPU `"amdgpu-flat-work-group-size"="64,N"`;
- `kernel.attrs["waves_per_eu"]` -> AMDGPU `"amdgpu-waves-per-eu"`;
- pointer param attrs: `noalias`, `readonly`, `writeonly`, `align`, `dereferenceable`;
- `optimize_kernel` conservative IR pass (constant fold + CSE + DCE);
- COMGR target ISA (default `amdgcn-amd-amdhsa--gfx950`).

Compiler options can be passed through `build_hsaco_from_llvm_ir(..., options=[...])`. Default is `["-O3"]`. Do not assume flags explain performance until ISA / resource evidence supports it.

## 11. ISA Inspection

DSL flow:

```text
artifact = compile_kernel(kernel)
write artifact.hsaco
analyze_llvm_ir(artifact.llvm_text)
analyze_hsaco(hsaco_path)
```

Inspect:

- MFMA instructions;
- DS reads/writes;
- raw buffer ops;
- `raw_ptr_buffer_load_lds`;
- `s_waitcnt`;
- `s_barrier`;
- vector stores;
- VGPR/SGPR;
- static LDS;
- occupancy hints.

## 12. Autotuning

DSL tuning stack:

- `all_dispatcher_configs`;
- `sweep.py`;
- `sweep_bench.py`;
- `benchmark/summary.py`;
- manifests;
- content-hash cache.

Good search records:

```text
spec
correctness status
latency
TFLOPS/GB/s
median
spread
VGPR/SGPR/LDS
notes
```

Keep failed experiments when they teach something.

## 13. Verification

DSL verification:

- `run_manifest.py --verify`;
- torch/AITER reference for attention;
- NumPy reference for manifest examples;
- shape-specific expected JSON in example trees;
- test discovery for DSL examples.

Verification should cover:

- aligned hero shapes;
- odd/tail dimensions;
- padding;
- stride/dilation;
- masks;
- multiple groups/batches/heads;
- dtype tolerance.

## Empirical Lessons Already Reflected In The DSL

Implicit GEMM conv improvements documented locally:

```text
baseline single-buffer/direct epilogue
  -> cshuffle epilogue
  -> buffer resource correctness fix
  -> 32x32x16 atom
  -> K-padded LDS
  -> graph/amortized launch mode
```

Direct grouped conv improvements:

```text
16c:
  -> K=32 MFMA fold
  -> wide direct epilogue
  -> block_groups tuning

4c:
  -> vectorized 4-channel output store
```

The lesson is not "always use these exact settings." The lesson is that each lever was isolated, verified, inspected, and benchmarked.
