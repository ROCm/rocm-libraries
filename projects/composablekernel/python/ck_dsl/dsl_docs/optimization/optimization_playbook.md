# Optimization Playbook

This is the practical loop for optimizing a `ck_dsl` kernel. It is scoped to
the DSL's compile, launch, verification, and inspection tools.

## The Loop

```text
1. State the hypothesis.
2. Verify correctness baseline.
3. Measure baseline with stable timing.
4. Inspect generated IR/ISA/resources.
5. Change one lever.
6. Re-verify correctness.
7. Re-measure.
8. Explain why it moved.
9. Keep or revert.
10. Record the result.
```

Do not batch several levers unless you are explicitly doing a coarse search and plan to isolate the winner afterward.

## Minimum Result Record

For every experiment record:

```text
kernel/spec name
shape
dtype/layout
GPU and ROCm version
baseline commit/config
variant description
correctness status
max/mean error
latency median
latency spread
TFLOPS or GB/s
VGPR/SGPR/LDS if inspected
notable ISA changes
notes
```

If correctness fails, do not report speed as a win.

## Step 1: Establish Baseline

Build the kernel:

```python
kernel = build_...(spec)
artifact = compile_kernel(kernel)
```

Verify:

```text
python -m ck_dsl.run_manifest artifact.hsaco manifest.json --verify
```

Benchmark:

```text
benchmark_manifest(..., attempts>=5, discard_first=True)
```

Inspect:

```python
ir_stats = analyze_llvm_ir(artifact.llvm_text)
hsaco_stats = analyze_hsaco("kernel.hsaco")
```

## Step 2: Classify Bottleneck

Ask:

- Is arithmetic intensity high enough to be compute-bound?
- Are MFMA instructions present and in the expected shape?
- Are vector stores emitted?
- Is LDS usage reducing occupancy?
- Is VGPR count too high?
- Are there many barriers or waits?
- Is launch overhead significant relative to kernel duration?
- Are tails/padding forcing scalar slow paths for important shapes?

Use profiler counters when static inspection is insufficient.

## Lever: Tile Shape

Changes:

- `tile_m`;
- `tile_n`;
- `tile_k`;
- block size;
- warp grid.

Expected effects:

- larger tiles increase reuse but consume more LDS/registers;
- larger `tile_k` can improve reuse but increases LDS and loop body size;
- unbalanced tiles may help skinny matrices or conv shapes;
- too few CTAs can underfill the GPU.

Check:

- LDS bytes;
- VGPR count;
- occupancy;
- tail fraction;
- grid size.

## Lever: MFMA Atom

Changes:

- 16x16 vs 32x32;
- K-packed vs non-packed;
- 4x4 for small-channel grouped work.

Expected effects:

- K-packed atoms reduce K-loop trips;
- 32x32 atoms can improve compute density;
- accumulator vectors grow and can increase VGPR;
- epilogue mapping changes.

Correctness risks:

- wrong A/B packing;
- stale lane-to-output mapping;
- tolerance drift from accumulation-order change.

Verification:

- generated MFMA intrinsic shape;
- numeric error distribution;
- VGPR change.

## Lever: LDS Layout

Changes:

- K padding;
- packed async layout;
- transpose reader formula;
- swizzle in consumer reads.

Expected effects:

- fewer LDS bank conflicts;
- better DS read bandwidth;
- async compatibility or incompatibility.

Risks:

- async destination swizzle violates lane-contiguous write contract;
- consumer reads use old stride;
- LDS usage increases enough to reduce occupancy.

## Lever: Load Strategy

Choices:

- coalesced load to VGPR then LDS;
- async buffer-to-LDS;
- direct global/buffer load for tiny ops;
- vector width changes.

Expected effects:

- async can reduce VGPR and overlap memory;
- coalesced sync path is simpler and often robust;
- wider vectors reduce instruction count but can worsen tails/alignment.

Risks:

- missing `s_waitcnt(vmcnt=0)`;
- unsupported async dword count;
- vector crosses invalid boundary;
- OOB sentinel in elements instead of bytes.

## Lever: Epilogue

Choices:

- direct scalar stores;
- direct vector stores;
- cshuffle LDS-stage stores;
- fused epilogue transform.

Expected effects:

- cshuffle improves scattered output store coalescing;
- direct vector stores are ideal for naturally contiguous per-lane outputs;
- fused epilogue can remove extra launches/memory passes.

Risks:

- cshuffle consumes LDS and barrier;
- direct stores scalarize;
- output tails mishandled;
- fused math changes tolerance.

## Lever: Schedule And Pipeline

Choices:

- scheduler policy;
- compv-style pipeline;
- static unroll;
- software ping-pong;
- priority hints.

Expected effects:

- better MFMA/DS/VMEM interleave;
- memory latency hiding;
- lower launch-visible stalls.

Risks:

- larger code hurts compile time or icache;
- extra stages increase LDS;
- too many live values increase VGPR;
- incorrect prologue/epilogue produces stale data.

## Lever: Launch And Runtime

Choices:

- persistent `KernelLauncher`;
- `PipelineLauncher`;
- workspace reuse;
- graph/amortized timing;
- fusion of adjacent simple ops.

Expected effects:

- removes module-load and allocation overhead;
- reduces launch overhead for tiny or multi-stage kernels;
- avoids allocator races.

Risks:

- comparing amortized timing against per-launch baseline unfairly;
- workspace lifetime bug;
- wrong torch stream;
- benchmark hides real production launch mode.

## Common Diagnosis Table

```text
Symptom: correct but slow, low MFMA count
Likely: wrong atom/tile, scalar path, missing vectorization
Action: inspect LLVM/ISA, check atom selection and loops

Symptom: fast but incorrect only on padded/tail shapes
Likely: invalid pointer load, bad descriptor valid, vector crosses tail
Action: test tiny adversarial shapes, inspect buffer sentinel path

Symptom: intermittent wrong answers in async path
Likely: missing waitcnt/barrier or workspace lifetime
Action: add/check s_waitcnt(vmcnt=0), stream sync, launcher keepalive

Symptom: atom change improves ISA but regresses runtime
Likely: VGPR/LDS occupancy loss or epilogue bottleneck
Action: inspect resources, try cshuffle/direct epilogue alternatives

Symptom: direct conv close but not within tolerance
Likely: K-packed lane order or accumulator reset bug
Action: compare per-lane small reference, inspect fold order
```

## What To Avoid

- Chasing compiler flags before inspecting IR/ISA.
- Trusting one benchmark run.
- Comparing kernels with different math or masks.
- Comparing compile+launch time to warm launch time.
- Ignoring failed correctness because the speed number is attractive.
- Adding compatibility shims for unshipped experimental branches instead of fixing the builder.

## Done Criteria For An Optimization

An optimization is done when:

- it has a one-sentence hypothesis;
- correctness passes representative and adversarial shapes;
- benchmark improvement is stable across repeated runs;
- generated IR/ISA confirms the intended primitive changed;
- resource usage is recorded;
- docs or comments explain the new invariant if it is non-obvious;
- unsupported configurations are rejected by validation.
