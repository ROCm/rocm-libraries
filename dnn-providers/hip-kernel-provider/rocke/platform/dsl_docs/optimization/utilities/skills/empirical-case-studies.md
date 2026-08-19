# Empirical Case Studies - GPU Kernel Optimization

Real-world empirical data from kernel optimization experiments on AMD MI350X/MI355X (gfx950) hardware.

## Purpose

This document captures concrete, measured findings from optimization experiments. Use these as reference points to understand:
- Typical performance gains from specific optimizations
- Common pitfalls and what didn't work
- Realistic expectations for optimization impact
- Diagnostic techniques that identified root causes

**IMPORTANT**: These are specific case studies from particular experiments, not universal truths. Actual results depend on:
- Hardware architecture (CDNA vs RDNA, generation)
- Problem shape and size
- Data types and precision
- Memory hierarchy behavior
- Compiler version and backend

Always benchmark on your target hardware and workload.

---

## Case Study 1: Direct Convolution on gfx950 (16c, 200x200, fp16)

**Context**: Grouped direct-convolution bakeoff comparing against CK Tile baseline

### Performance Progression (Ordered by Impact)

```text
Baseline (scalar FMA):                 ~1.2   TFLOPS  (reference)
+ MFMA instructions:                   ~54    TFLOPS  (~45x gain)
+ H-row pipeline + 3 circular accs:    ~77    TFLOPS  (~1.4x gain)
+ 8 waves/WG sharing LDS:              ~88-104 TFLOPS  (~1.2x median)
+ Double-buffered LDS ping-pong:       ~93-108 TFLOPS  (~1.05x median)
+ Async DRAM→LDS + vector stores:      ~213-239 TFLOPS (~2.2x gain, BIGGEST)
+ K=32 MFMA folding:                   ~226-233 TFLOPS (marginal win)
+ BLOCK_GROUPS=4, n_fold=4, K32:       ~242-245 TFLOPS (best variant)

CK Tile baseline:                      ~250-254 TFLOPS
```

### Key Insights

1. **Biggest single win**: Vectorizing epilogue from 4× scalar fp16 stores to 1× fp16x8 store per lane
   - Gained ~100-120 TFLOPS in one step
   - Try epilogue vectorization first

2. **MFMA adoption**: Moving from scalar to matrix ops gave 45× speedup
   - Fundamental algorithmic change
   - All modern kernels should use MFMA/WMMA/Tensor Cores

3. **Async loads**: Second-biggest gain after vectorization
   - Enables compute/memory overlap
   - Worth the implementation complexity

### What Didn't Help

| Change | Performance | Why It Failed |
|--------|-------------|---------------|
| BLOCK_Q=32 (double Q per wave) | ~195-197 TFLOPS | Wider LDS row needed 2-pass async loads, doubled accumulator pressure |
| BLOCK_GROUPS=16 | ~206-216 TFLOPS | Insufficient work per group |
| BLOCK_GROUPS=2 | ~224-212 TFLOPS | Too much work per group |
| BLOCK_GROUPS=1 | ~154-152 TFLOPS | Severe underutilization |
| LDS swizzle via register staging | ~92-107 TFLOPS | ALU overhead dominated |
| Naive flat LDS epilogue | ~202-236 TFLOPS | Also correctness issue on one shape |
| Cooperative weight LDS (CK-style) | ~179-215 TFLOPS | Read distribution not optimized |
| Compiler hint sweep | <1% change | waves_per_eu, maxnreg, scheduling modes had minimal impact |
| `enable-post-misched=0` | Miscompiled | Flag worked in CK but broke this kernel |

### Configuration Sensitivity

- Best config was **NOT** CK's reference config
- CK used `waves_per_wg=8`, best was `BLOCK_GROUPS=4 + n_fold=4`
- **Lesson**: Always sweep±1 step around known-good configs
- Register/occupancy tradeoffs differ per implementation

---

## Case Study 2: LDS Bank Conflict Mitigation on gfx950

**Context**: ResNet50 conv3_1, split_image config, comparing XOR vs Padding swizzle

### Measured Impact

| Approach | TFLOPS | Notes |
|----------|--------|-------|
| XOR swizzle (no hotloop scheduling) | 494 | Bank conflicts eliminated, but ALU overhead hurts |
| Padding swizzle (with hotloop scheduling) | 705 | **+43% improvement, +211 TFLOPS gain** |

**Absolute gain**: +100 TFLOPS over XOR

### Why Padding Won on gfx950

1. **Simpler addressing**: 1-2 ALU vs 5-7 ALU instructions per LDS read
2. **Better instruction scheduling**: Reduced address computation allows hardware scheduler to work effectively
3. **Negligible LDS waste**: 3KB / 160KB = 1.9%, doesn't impact occupancy
4. **Same conflict elimination**: Both approaches avoid bank conflicts

### Architecture-Specific Insight

- **gfx950 caveat**: Explicit scheduling barriers (`s_sched_barrier`, `s_sched_group_barrier`) are **silently removed** by LLVM backend
- Hardware scheduler is sophisticated enough without manual hints
- Verify with `grep -c "s_sched_barrier" kernel.s` → should return 0
- Cannot rely on scheduling barriers to mitigate XOR overhead on gfx950

### Decision Heuristic

```python
if LDS_total < 96KB:
    use_xor_swizzle()      # Capacity-constrained (e.g., gfx90a: 64KB)
elif LDS_avail >= 128KB:
    use_padding_swizzle()  # Abundant LDS (e.g., gfx950: 160KB)
else:  # 96KB ≤ LDS < 128KB
    benchmark_both()       # Architecture-dependent
```

**Critical insight from LDS studies**: `ds_write_b128` has 32-dword conflict period while `ds_read_b128` has 64-dword period. Read and write conflict behavior can differ—may need separate swizzle strategies.

### Architecture-Specific LDS Capacities

| Architecture | GPU | LDS per CU | Notes |
|--------------|-----|------------|-------|
| gfx90a | MI210 | 64 KB | LDS capacity is precious - XOR swizzle preferred |
| gfx940 | MI300 | 64 KB | LDS capacity is precious - XOR swizzle preferred |
| gfx950 | MI350X/MI355X | 160 KB | Abundant LDS - padding swizzle often wins |

### Performance Trade-offs by Architecture

**On abundant-LDS architectures (gfx950: 160KB):**
- Padding swizzle can significantly outperform XOR swizzle despite "wasting" LDS
- Simpler addressing (1-2 ALU vs 5-7 ALU) reduces overhead
- Hardware scheduler works more effectively with simpler instruction streams
- LDS waste (e.g., 3KB/160KB = 1.9%) is negligible when total LDS doesn't constrain occupancy
- Both approaches eliminate bank conflicts - difference is addressing complexity

**On capacity-constrained architectures (gfx90a/gfx940: 64KB):**
- XOR swizzle's zero-waste property becomes more important than addressing simplicity
- LDS footprint directly impacts occupancy
- ALU overhead from XOR addressing is acceptable trade-off to avoid LDS waste

### gfx950/CDNA4 Scheduler Behavior

**Explicit scheduling barriers are silently removed** by LLVM backend on gfx950:
- `s_sched_barrier` and `s_sched_group_barrier` instructions → not emitted in final ISA
- Hardware scheduler is sophisticated enough without manual hints
- Cannot rely on scheduling barriers to mitigate XOR addressing overhead
- Always verify ISA output: `grep -c "s_sched_barrier" kernel.s` should return 0

This is why XOR swizzle's ALU overhead is more severe on gfx950 than architectures where scheduling barriers are respected.

**UPDATE — the directives the kernel ALREADY emits can be net-negative.** The point
above ("you cannot *rely on* added scheduling barriers") has a stronger corollary
measured on square GEMM: the `sched_group_barrier` HotLoop + per-cluster
`s_setprio`/`sched_barrier(0)` that the compv4 pipeline emits *over-constrain* comgr's
backend on gfx950, and **removing them is a +1.9–2.5% uplift** (MfmaUtil 63%→68%). So
the guidance is not merely "manual hints don't help" — on gfx950 the hints a kernel
transcribes from a reference can actively hurt. See **Case Study 7**. Lever:
`TraitSpec.emit_sched_hints` (default `None` = arch-resolved, hints OFF on gfx950).

### Opcode-Specific Conflict Behavior (gfx950/MI350X)

From empirical LDS studies on MI350X:

**Conflict periods by opcode:**
- `ds_read_b64`, `ds_read_b128`: **64-dword conflict period**
- All other opcodes (`ds_read_b32`, `ds_write_*`): **32-dword conflict period**
- `ds_write_b128` specifically: 32-dword period (different from its read counterpart)

**Phase decomposition:**
- Lane-pair conflict restrictions vary by opcode
- Example: `ds_write_b128` has 8 phases of 8 lanes each
- Lanes in opposite wave-halves (0-31 vs 32-63) cannot conflict

**Sub-dword behavior:**
- Intra-dword accesses are conflict-free
- Unaligned `ds_read_u16` charges to `SQ_LDS_IDX_ACTIVE` instead of `SQ_LDS_BANK_CONFLICT`

**Critical asymmetry**: Swizzle layouts tuned for reads may not protect the write path. Read and write conflict behavior on the same LDS region can differ and may need separate swizzle strategies.

**For deeper analysis**, see companion documents:
- **LDS_a.md**: "Four Things to Know About LDS Bank Conflicts on MI350X"
- **LDS_b.md**: "Empirically Characterizing LDS Bank Conflicts on AMD MI350X"

---

## Case Study 3: Numerical Tolerance and Bug Detection

**Context**: FP16 inputs, FP32 accumulation, O(100) terms

### Expected Tolerance (Correct Kernel)

| Metric | Typical Value | Range |
|--------|---------------|-------|
| `max_abs_error` | ~8e-6 | 5e-6 to 1e-5 |
| `mean_abs_error` | ~9e-7 | 5e-7 to 2e-6 |

### Bug Signatures

| Symptom | Likely Root Cause |
|---------|-------------------|
| `max_abs ~ 5e-3`, `mean_abs ~ 3e-6`, ~0.1% elements bad | Wrong K=32 MFMA lane mapping |
| `max_abs ~ 1.4e-2`, `mean_abs ~ 1.4e-3`, ~50% bad | Missing accumulator reset (circular slot reuse) |
| `max_abs ~ 1.4e-2`, ~60% bad | Async DMA with invalid per-lane LDS pointers |
| Output is zero or stale | DSL AST rewriter elided stores (missing `const_expr` guard) |
| Correct for one shape, `inf`/`NaN` for another | Wrong wide-store thread mapping in epilogue |

**Rule of thumb**: Errors **two orders of magnitude** higher than expected almost always indicate structural bugs, not "fp16 noise"

---

## Case Study 4: Stability and Measurement Caveats

### Observed Variability

1. **Bimodal latency**: Same kernel showed ~108 vs ~86 TFLOPS within single script
   - Attributed to cache state, thermal, clocking
   - Not a correctness issue

2. **First-run slowdown**: CK Tile config dropped from ~250 to ~80 TFLOPS on first run, then stabilized
   - **Always discard the first run** for benchmarking

3. **Shape-dependent variance**:
   - `groups=32` and `groups=16` individually stable
   - But `groups=32` sometimes throttled
   - **Report median and worst, not just best**

### Diagnostic Techniques That Worked

1. **Count ops in lowered IR**:
   - Counted `vector.store` in MLIR after IR-DUMP
   - Found silently elided LDS stores immediately

2. **ISA inspection**:
   - Count `s_barrier`, `s_waitcnt`, `v_mfma`, `buffer_store_short`
   - Confirmed scalar vs vector epilogue

3. **Assembly metadata**:
   - Check `amdhsa_next_free_vgpr` and `amdhsa_group_segment_fixed_size`
   - Determined if occupancy was VGPR- or LDS-limited

4. **One lever per change**:
   - A/B testing with single-variable changes
   - Combining changes hid which one mattered

---

## Case Study 5: Closing the Last 5%

**Context**: Gap from ~245 to ~254 TFLOPS (3-4%)

### What Didn't Close the Gap

- More LLVM flags
- Additional swizzle patterns
- Weight LDS staging
- Larger Q tiles
- Smaller workgroup sizes
- Naive LDS epilogue

### Likely Remaining Levers (In Order)

1. **Exact CK output writer mapping** with wide-store distribution
2. **CK descriptor swizzle** as descriptor transforms (not ad-hoc arithmetic)
3. **Explicit instruction scheduling** around async loads, MFMA, LDS ops
4. **Different Q-wave decomposition** (split Q across waves vs C across waves)

**Lesson**: At the last few percent, generic levers stop helping. Must port specific implementation knowledge from existing solutions.

---

## Case Study 6: Implicit GEMM vs Pure GEMM Overhead

**Empirical finding**: Implicit GEMM convolution requires 15-20 additional VGPRs vs pure GEMM due to im2col coordinate computation overhead.

**Realistic performance expectations**:
- Convolution: 60-65% of theoretical peak (even with perfect implementation)
- Pure GEMM: 75-85% of theoretical peak

This is an **algorithmic constraint**, not an implementation quality issue.

---

## Case Study 7: Schedule Directives Are Net-Negative on gfx950 GEMM (+2%)

**Context**: Square fp16/bf16 GEMM (8192³, 256×256×64 tile, compv4 "rung F"), chasing
the ~15% gap to rocBLAS. The compv4 path emits a full schedule — a two-stage
`sched_group_barrier` HotLoop plus per-MFMA-cluster `s_setprio(1/0)` + `sched_barrier(0)`
fences — transcribed from the CK Tile C++ reference.

### Finding

Removing ALL of those directives and letting comgr schedule freely is a real,
correctness-preserving uplift — the first positive lever after exhausting AGPR form,
grid/chiplet swizzle, comgr scheduler flags, `iglp_opt`, tile/warp/K sweeps,
hipcc-vs-comgr, MFMA atom shape, `waves_per_eu`, and register read-ahead (all
neutral/negative on this kernel).

| variant (square fp16 8192³)     | /rocBLAS | MfmaUtil |
|---------------------------------|---------:|---------:|
| compv4 with schedule directives | 0.834    | 63.0%    |
| compv4 directives removed       | 0.854    | 67.8%    |

Generality: +2.3% fp16, +2.1% bf16 (0.846→0.864), +1.9–2.5% on rectangular shapes.
Works on the default comgr path (no hipcc).

### Why (measured mechanism)

The kernel is MFMA+LDS **latency-bound**, not memory-bound (`MemUnitStalled` ≈ 0). The
`sched_barrier(0)` (mask 0) is a *full* fence the compiler cannot move loads across, so
the directives simultaneously *describe* a load/compute interleave
(`sched_group_barrier`) and *forbid* it. On gfx950 the hardware scheduler does better
unconstrained; the +5 pt MFMA-utilization IS the +2% wall-clock.

### Proof (statistically definitive)

GPU-event (kernel-only) paired timing, 200 cycles × 80 launches: faster in **200/200**
cycles, median ratio **1.019**, 95% CI [1.0189, 1.0198], worst cycle still +1.1%,
Wilcoxon p = 1.4×10⁻³⁴. `relerr=0` on full 8192³ AND the K-sweep {1,2,3,5,16} race
detector — NOT a racy-fast artifact (an earlier "+4%" warp-group ping-pong WAS such an
artifact and was discarded). Builds are deterministic; the hinted and un-hinted kernels
are distinct code objects.

### Lever

`TraitSpec.emit_sched_hints`: `None` (default) is arch-resolved — hints OFF on gfx950
(takes the uplift), ON elsewhere (preserves the historical emission). `True`/`False`
forces the choice. Keep the directives for the older CDNA the CK Tile reference was
tuned for — this is a gfx950 finding.

### Lesson

Faithfully transcribing a reference kernel's *schedule* can REGRESS on a backend whose
hardware scheduler is already strong. When MfmaUtil is the binding metric and the kernel
is latency-bound, try REMOVING scheduling directives, not adding them. Extends Case
Study 2: it's not just that manual hints can't be relied on — the hints you already emit
can hurt.

---

## Usage Guidelines

### Do:
- Use these case studies as **reference points** for expected gains
- Understand the **context** (hardware, shape, dtype) before extrapolating
- Apply the **diagnostic techniques** to your own kernels
- Check for the **bug signatures** when debugging

### Don't:
- Assume exact numbers will transfer to different hardware/problems
- Skip measuring because "XOR should be faster than padding"
- Ignore stability issues (warmup, multiple runs, median reporting)
- Expect generic levers to close the last 5%

### When to Reference

- Setting expectations for optimization ROI
- Debugging unexpected performance or correctness issues
- Prioritizing which optimizations to try first
- Understanding when you've hit fundamental limits
