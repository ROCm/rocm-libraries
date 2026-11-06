# Slow CPU reference GEMM path investigation (gfx950 per-config rerun 2)

`full_runs_gfx950/per_config_rerun_2/AGGREGATE_analysis.txt` reports **Reference GEMM (CPU)** at
1,080,469 ms (42.1% of wall). This is the largest category. The client's reference GEMM
(`SolveGemmCPU` in `client/src/Reference.cpp`) has two implementations:

- a tiled, vectorised **fast path** (`solveCPUFast`), used when `isFastPathEligible()` accepts the
  problem;
- a scalar **slow path** (`ReferenceSolution<…>::SolveCPU`) for everything else.

The aggregate already splits the two with the `solve_cpu_fast*` and `solve_cpu_slow*` timers:

| Timer | Calls | Total (ms) | Mean (ms) |
| --- | --- | --- | --- |
| `solve_cpu_slow` | 1,947 | 727,756.12 | 373.78 |
| `solve_cpu_slow_per_solution_inputs_changed` | 19 | 13,166.07 | 692.95 |
| **Slow path, total** | **1,966** | **740,922** | |
| `solve_cpu_fast` | 26,656 | 200,931.33 | 7.54 |
| `solve_cpu_fast_per_solution_inputs_changed` | 1,127 | 138,501.63 | 122.89 |
| **Fast path, total** | **27,783** | **339,433** | |

The slow path is 6.6% of reference calls but 68.6% of reference time. This note covers:

- which YAMLs take the slow path;
- why each problem is rejected from the fast path;
- why the slow path is so slow.

Timing claims come from measurements; code reading was only used to decide what to measure.
Inferences are marked as such.

## Summary

1. **Two YAMLs are 81% of the slow path.** `streamk/sk_I8gemm_quick.yaml` (310.7 s) and
   `streamk/sk_f8gemm_quick.yaml` (289.4 s) together take 600 s of the 741 s. All of it is their
   `4111×4111×4111` problems. The fast path rejects them on type:
   - Int8 inputs (all four output types: I8, BF16, Half, I32);
   - Float8 output (`F8 × F8 → F8`).

   In the same YAML, the same-size problems with a Float, Half or BF16 output take the fast path in
   0.47–1.21 s. The slow-path calls each took 45–84 s.
2. **The slow path is serialised by a lock taken once per output element.** After computing each
   output element, every thread calls `omp_set_num_threads(64)` and then enters a global
   `#pragma omp critical`. The critical section exists only for the AMAX-D reduction, but it is
   entered even when `outputAmaxD()` is false; the check is inside it.
   - **Stack samples:** 75 of the roughly 100 thread samples taken during a slow-path GEMM were
     waiting in `__kmp_acquire_queuing_lock`.
   - **Disassembly:** the per-element call sequence is
     `omp_set_num_threads` → `__kmpc_critical` → `outputAmaxD` test → `__kmpc_end_critical`, on
     the path every element takes.
   - **Scaling:** with 25 threads on 25 cores, a 1024³ fp32 GEMM is barely faster than with one
     thread (1,727–2,364 ms against 2,626–2,660 ms).
3. **Entering the critical section only when AMAX-D is requested gives about a 4.6× speed-up**
   (prototype `refpath3`, scratch only).
   - **Driver, 2048³ fp32:** 45,237 ms → 1,507 ms. Other unmodified runs of the same problem took
     6.8–14 s; the contended lock makes the baseline erratic.
   - **Driver, 1024³ fp32:** 1,727 ms → 191 ms.
   - **Replay of all 59 slow-path YAMLs:** slow-path total 546.6 s → 119.5 s. Every config still
     passes, and the driver's results are bit-identical (max diff 0).
4. **Even after that fix, the slow path is about 25–30× slower than the fast path on the same
   problem.** At 4111³ on 25 cores, the guarded slow path took 16.3 s (fp32) and 17.5 s (f8 in).
   The fast path took 0.65 s (f64), 0.76 s (f32) and 0.54 s (f8 in).

   The slow path computes each output element as a separate scalar dot product. In the I8 K loop,
   each MAC is 2 sign-extending loads, an `imul` and an `add`, with no vectorisation or blocking.
   Adding fast-path support for **Int8 inputs** and **Float8 outputs** would move 79% of the
   remaining (post-fix) slow-path time to the fast path.
5. **The other rejection reasons are small.** After the fix, measured in the replay:
   - `gradient`: 8.8 s
   - `useE`: 7.7 s
   - complex types: 4.1 s
   - MX K not a multiple of the MX block: 2.6 s
   - `NumElementsToValidate: 128` (sparse validation, intended): 1.7 s

## Method

- **Production numbers:** the per-config logs in
  `/home/alvasile/timing_runs/full_runs_gfx950/per_config_rerun_2/<cfg>/<cfg>_analysis.txt`. For
  each config, the `solve_cpu_slow*` and `solve_cpu_fast*` rows were summed from the first
  "TIMING BY PHASE" table (`/tmp/refpath/percfg.py` → `/tmp/refpath/percfg_prod.txt`).
  59 configs have at least one slow-path call.
- **Instrumented client (`refpath`):**
  - A copy of the current `client/` sources in `/tmp/refpath/client_src`, built with the
    production flags by `/tmp/bookkeep/build.sh`.
  - `isFastPathEligible()` already has a `rejectFast(reason)` hook (disabled with `if (false)`).
    The copy stores the reason instead, and `SolveGemmCPU` prints one `REFPATH` line per call: the
    path, the reason, M/N/K/batch, the A/B/C/D types, `elementsToValidate`, and the elapsed ms.
  - Reasons are reported in the order `SolveGemmCPU` checks them: `tryFastPath`, then the 20%
    validation-density rule, then `isFastPathEligible()`.
- **Replays:**
  - Each of the 59 configs was rerun through pytest with the production artifacts:
    `--use-cache --artifact-dir` holding a symlink to `/tmp/tl_split_gfx950/artifacts/<cfg>.tar.gz`,
    and `--prebuilt-client`.
  - Same environment as the production rerun: `OMP_NUM_THREADS=25`, 25 pinned idle cores, default
    `KMP_BLOCKTIME`, `TimingInstrumentation=True`, msgpack library format.
  - **Baseline (`refpath`):** cores `28-39,42-46,49-53,55-57`, GPU 4. Logs and per-call `REFPATH`
    lines are in `/tmp/refpath/logs/`.
  - **Guarded critical section (`refpath3`):** cores `64-88`, GPU 5. Results are in
    `/tmp/refpath/logs3/`.
  - The baseline replay has the same 1,966 slow-path calls as production, problem for problem.
    Totals differ (546.6 s against 740.9 s) because the contended lock makes run times erratic;
    see finding 3.
  - Only host-side timers are compared. The GPUs are shared with other users.
- **Same-problem comparisons:**
  - The `cpu-gemm-driver` object from the production build was relinked against the instrumented
    client library: `/tmp/refpath/cpu-gemm-driver`, `-driver2` and `-driver3`.
  - `-driver2` and `-driver3` read the reference's thread cap from `TL_MAXT`. Production hard-codes
    `MAX_OMP_THREADS 64`.
  - Runs used cores `192-216`, whose SMT siblings were idle (2% busy), with `KMP_BLOCKTIME=0`
    unless stated. The timing is the `REFPATH` value for the reference call only; the driver's
    validation step is excluded.
- **Stack sampling:** `perf` is blocked and `ptrace_scope=1` (no attaching), so the driver was
  launched under `rocgdb -batch`. Four `SIGINT`s, 1.5 s apart, each triggered
  `thread apply all bt 4`.
- **Assembly:** `objdump -dr` of `Reference.cpp.o` from the production-flag build, on the
  `.omp_outlined` body of
  `ReferenceSolution<TypedGemm<int8,int8,int8,int8,int,int,int8,int8>,int,int>::SolveCPU` (the I8
  case from `sk_I8gemm_quick`).

## Which YAMLs take the slow path, and why

### By rejection reason

| Reason | Calls | Replay (s) | Replay, guarded critical (s) |
| --- | --- | --- | --- |
| Type: Float8 × Float8 → **Float8** | 6 | 172.4 | 44.1 |
| Type: **Int8** × Int8 → Int8 | 714 | 92.6 | 15.1 |
| Type: **Int8** × Int8 → BFloat16 | 99 | 79.1 | 12.7 |
| Type: **Int8** × Int8 → Int32 | 39 | 70.2 | 10.2 |
| Type: **Int8** × Int8 → Half | 99 | 51.8 | 11.0 |
| `useE` | 18 | 20.1 | 7.7 |
| `gradient` | 47 | 19.9 | 8.8 |
| `K_not_multiple_of_mxBlockA` (MXFP4, K = 333/255/63, block 32) | 36 | 19.8 | 2.6 |
| Type: ComplexFloat | 251 | 9.2 | 2.5 |
| Type: Float8 × **Half** → **Float8** | 24 | 4.6 | 1.5 |
| Type: ComplexDouble | 64 | 4.4 | 1.6 |
| `NumElementsToValidate: 128` (fewer than 20% of D validated) | 569 | 2.4 | 1.7 |
| **Total** | **1,966** | **546.6** | **119.5** |

Notes on the reasons:

- **Type rejections:** the fast path accepts these types:
  - inputs: Float, Double, Half, BFloat16, the four FP8/BF8 variants, and Float4;
  - outputs (C and D): Float, Double, Half, BFloat16.

  Int8 inputs, Int8/Int32/FP8 outputs and complex types fall through.
- **`NumElementsToValidate: 128`:** this is the intended behaviour. The slow path then computes
  only every `NextPrime(total/128)`-th element, so these 569 calls cost 2.4 s in total.
  `elementsToValidate` arrives as `size_t`, so the YAML default `-1` counts as "validate
  everything" and does not trigger this rule.
- **No other reasons occurred** for this set: no `amaxD`, `scaleCD`, `layout`, `multi_index`,
  `mirror_indices`, `no_batch_indices` or `tryFastPath=false` rejections.

### By YAML (all 59)

"Prod" is the production per-config log. "Replay" is the baseline instrumented replay.
"Guarded critical" is the `refpath3` replay; see Option A.

| YAML (`Tests/common/…`) | Slow calls | Reason(s) (calls) | Prod (s) | Replay (s) | Replay, guarded critical (s) |
| --- | --- | --- | --- | --- | --- |
| `streamk/sk_I8gemm_quick.yaml` | 12 | type A=Int8/B=Int8→D=Int8 (3); type A=Int8/B=Int8→D=BFloat16 (3); type A=Int8/B=Int8→D=Half (3); type A=Int8/B=Int8→D=Int32 (3) | 310.66 | 252.50 | 43.20 |
| `streamk/sk_f8gemm_quick.yaml` | 6 | type A=Float8/B=Float8→D=Float8 (6) | 289.45 | 172.42 | 44.06 |
| `gemm/gfx950/mxfp4_mxfp4_bf16_tn_act.yaml` | 36 | K_not_multiple_of_mxBlockA (36) | 25.26 | 19.84 | 2.64 |
| `streamk/gfx950/sk_hgemm_use_e.yaml` | 11 | useE (8); gradient (3) | 22.87 | 17.01 | 7.36 |
| `streamk/gfx950/data_parallel_static_hgemm_use_e.yaml` | 8 | useE (6); gradient (2) | 14.51 | 13.04 | 5.70 |
| `sparse/gfx950/spmm_i8_sb.yaml` | 196 | type A=Int8/B=Int8→D=Int8 (196) | 6.92 | 5.94 | 1.12 |
| `sparse/gfx950/spmm_i8.yaml` | 196 | type A=Int8/B=Int8→D=Int8 (196) | 6.66 | 6.03 | 1.10 |
| `gemm/gfx950/f8f16mix_f8s.yaml` | 24 | type A=Float8/B=Half→D=Float8 (24) | 5.80 | 4.58 | 1.53 |
| `gemm/bufferLoadStore0.yaml` | 189 | type A=ComplexFloat/B=ComplexFloat→D=ComplexFloat (189) | 5.01 | 3.96 | 1.03 |
| `gradient/fp16_gradient_bias.yaml` | 21 | gradient (21) | 4.64 | 5.21 | 2.02 |
| `gemm/cgemm.yaml` | 54 | type A=ComplexFloat/B=ComplexFloat→D=ComplexFloat (54) | 4.47 | 3.94 | 0.62 |
| `gradient/gfx950/bf16_gradient_bias.yaml` | 21 | gradient (21) | 4.39 | 4.69 | 1.35 |
| `gemm/zgemm.yaml` | 54 | type A=ComplexDouble/B=ComplexDouble→D=ComplexDouble (54) | 4.05 | 3.67 | 0.59 |
| `sparse/gfx94x/spmm_i8_mi16_sb.yaml` | 40 | type A=Int8/B=Int8→D=Int8 (40) | 3.99 | 2.22 | 0.19 |
| `sparse/gfx950/spmm_dtl.yaml` | 16 | type A=Int8/B=Int8→D=Int8 (16) | 3.23 | 1.90 | 0.55 |
| `sparse/gfx94x/spmm_i8_mi32_sb.yaml` | 40 | type A=Int8/B=Int8→D=Int8 (40) | 2.96 | 2.22 | 0.24 |
| `sparse/gfx950/spmm_i8hs.yaml` | 64 | type A=Int8/B=Int8→D=Half (64) | 2.75 | 2.70 | 0.28 |
| `sparse/gfx94x/spmm_i8_mi32.yaml` | 40 | type A=Int8/B=Int8→D=Int8 (40) | 2.72 | 2.94 | 0.22 |
| `sparse/gfx94x/spmm_i8_mi16.yaml` | 40 | type A=Int8/B=Int8→D=Int8 (40) | 2.43 | 2.11 | 0.27 |
| `sparse/gfx950/spmm_i8bs.yaml` | 64 | type A=Int8/B=Int8→D=BFloat16 (64) | 2.06 | 2.56 | 0.45 |
| `streamk/sk_cgemm_quick.yaml` | 8 | type A=ComplexFloat/B=ComplexFloat→D=ComplexFloat (8) | 2.03 | 1.31 | 0.87 |
| `sparse/gfx94x/i8_activation.yaml` | 44 | type A=Int8/B=Int8→D=Int8 (44) | 1.96 | 2.01 | 0.16 |
| `gemm/gemm_bias_strided_batched_gfx940.yaml` | 24 | type A=Int8/B=Int8→D=Int8 (24) | 1.50 | 1.53 | 0.13 |
| `gemm/gemm_bias_strided_batched_broadcast_gfx940.yaml` | 24 | type A=Int8/B=Int8→D=Int8 (24) | 1.32 | 1.32 | 0.35 |
| `sparse/gfx94x/spmm_i8bs_sb.yaml` | 16 | type A=Int8/B=Int8→D=BFloat16 (16) | 1.06 | 0.70 | 0.05 |
| `sparse/gfx94x/spmm_i8hs.yaml` | 16 | type A=Int8/B=Int8→D=Half (16) | 0.89 | 1.24 | 0.07 |
| `sparse/gfx94x/spmm_i8is.yaml` | 16 | type A=Int8/B=Int8→D=Int32 (16) | 0.86 | 0.84 | 0.02 |
| `sparse/gfx94x/spmm_i8is_sb.yaml` | 16 | type A=Int8/B=Int8→D=Int32 (16) | 0.72 | 0.75 | 0.07 |
| `sparse/gfx94x/spmm_i8bs.yaml` | 16 | type A=Int8/B=Int8→D=BFloat16 (16) | 0.71 | 0.82 | 0.04 |
| `gemm/gfx950/i8_gsu_gfx950.yaml` | 4 | type A=Int8/B=Int8→D=Int8 (2); type A=Int8/B=Int8→D=Int32 (2) | 0.71 | 0.97 | 0.06 |
| `gemm/i8_gsu_gfx940.yaml` | 4 | type A=Int8/B=Int8→D=Int8 (2); type A=Int8/B=Int8→D=Int32 (2) | 0.70 | 0.79 | 0.13 |
| `sparse/gfx94x/spmm_i8hs_sb.yaml` | 16 | type A=Int8/B=Int8→D=Half (16) | 0.69 | 1.04 | 0.07 |
| `streamk/sk_zgemm_quick.yaml` | 10 | type A=ComplexDouble/B=ComplexDouble→D=ComplexDouble (10) | 0.66 | 0.75 | 1.02 |
| `gemm/fp16_tn.yaml` | 166 | NumElementsToValidate=128 (166) | 0.38 | 0.19 | 0.22 |
| `gemm/gfx950/fp16_use_e_wide_store.yaml` | 4 | useE (4) | 0.32 | 0.12 | 0.02 |
| `gemm/ag.yaml` | 180 | NumElementsToValidate=128 (180) | 0.31 | 1.13 | 0.46 |
| `gemm/gfx950/lds_tr.yaml` | 78 | NumElementsToValidate=128 (78) | 0.29 | 0.38 | 0.43 |
| `gemm/lsu_i8.yaml` | 15 | type A=Int8/B=Int8→D=Int8 (15) | 0.27 | 0.43 | 0.14 |
| `sparse/gfx94x/spmm_vw_lg_one.yaml` | 8 | type A=Int8/B=Int8→D=Int8 (8) | 0.07 | 0.04 | 0.02 |
| `gemm/use_beta_false.yaml` | 6 | NumElementsToValidate=128 (6) | 0.07 | 0.02 | 0.02 |
| `gradient/fp16_gradient_activation.yaml` | 8 | NumElementsToValidate=128 (8) | 0.07 | 0.04 | 0.06 |
| `gemm/fp16fp32mix.yaml` | 8 | NumElementsToValidate=128 (8) | 0.06 | 0.05 | 0.06 |
| `sparse/gfx950/spmm_vw_lg_one.yaml` | 8 | type A=Int8/B=Int8→D=Int8 (8) | 0.06 | 0.05 | 0.06 |
| `gemm/bf16_tn.yaml` | 27 | NumElementsToValidate=128 (27) | 0.06 | 0.02 | 0.07 |
| `gemm/fp16_use_e.yaml` | 18 | NumElementsToValidate=128 (18) | 0.06 | 0.40 | 0.13 |
| `sparse/gfx94x/spmm_vw_lg_one_sb.yaml` | 8 | type A=Int8/B=Int8→D=Int8 (8) | 0.06 | 0.02 | 0.02 |
| `sparse/gfx950/spmm_vw_lg_one_sb.yaml` | 8 | type A=Int8/B=Int8→D=Int8 (8) | 0.06 | 0.03 | 0.02 |
| `gemm/fp16_HH_BHS_bf16mfma.yaml` | 27 | NumElementsToValidate=128 (27) | 0.04 | 0.03 | 0.03 |
| `ductile/gfx950/f8_tn_mx_subtile.yaml` | 2 | NumElementsToValidate=128 (2) | 0.03 | 0.01 | 0.01 |
| `groupedgemm/grouped_gemm.yaml` | 8 | NumElementsToValidate=128 (8) | 0.03 | 0.02 | 0.03 |
| `gemm/fp32_tn.yaml` | 27 | NumElementsToValidate=128 (27) | 0.03 | 0.03 | 0.08 |
| `streamk/sk_bgemm_div.yaml` | 1 | NumElementsToValidate=128 (1) | 0.01 | 0.01 | 0.02 |
| `exception/f16_cf32_nan.yaml` | 3 | NumElementsToValidate=128 (3) | 0.01 | 0.02 | 0.05 |
| `benchmark_runs/fp32_tn.yaml` | 3 | NumElementsToValidate=128 (3) | 0.01 | 0.00 | 0.01 |
| `gemm/gfx950/bf16_cvt.yaml` | 1 | NumElementsToValidate=128 (1) | 0.01 | 0.00 | 0.00 |
| `gemm/large_size.yaml` | 1 | NumElementsToValidate=128 (1) | 0.01 | 0.01 | 0.00 |
| `gemm/fp32_nt.yaml` | 1 | NumElementsToValidate=128 (1) | 0.00 | 0.00 | 0.00 |
| `gemm/nan_bounds_check_odd_padding.yaml` | 3 | NumElementsToValidate=128 (3) | 0.00 | 0.00 | 0.00 |
| `gemm/bf16_tn_predict.yaml` | 1 | NumElementsToValidate=128 (1) | 0.00 | 0.01 | 0.00 |
| **Total** | 1966 | | **740.9** | **546.6** | **119.5** |

Rows below about 0.1 s are within run-to-run noise. Some small rows read slightly higher with
the guarded critical section for that reason.

### The two dominant YAMLs, per call (4111³, one call each)

| YAML | A × B → D | Path | Replay (ms) | Guarded critical (ms) |
| --- | --- | --- | --- | --- |
| `sk_I8gemm_quick` | I8 × I8 → I8 | slow (type) | 60,465 | 9,985 |
| `sk_I8gemm_quick` | I8 × I8 → BF16 | slow (type) | 72,756 | 11,626 |
| `sk_I8gemm_quick` | I8 × I8 → Half | slow (type) | 45,210 | 9,998 |
| `sk_I8gemm_quick` | I8 × I8 → I32 | slow (type) | 66,115 | 9,491 |
| `sk_f8gemm_quick` | F8 × F8 → **F8** (2 calls) | slow (type) | 84,331 / 83,567 | 19,464 / 23,092 |
| `sk_f8gemm_quick` | F8 × F8 → Float (2 calls) | fast | — | 702 / 1,206 |
| `sk_f8gemm_quick` | F8 × F8 → Half | fast | — | 655 |
| `sk_f8gemm_quick` | BF8 × BF8 → Float | fast | — | 636 |
| `sk_f8gemm_quick` | F8 × Half → Float | fast | — | 655 |
| `sk_f8gemm_quick` | BF8 × F8 → BF16 | fast | — | 469 |

In `sk_f8gemm_quick`, the only problem type that takes the slow path is the one whose output is
Float8. Its 4111³ problem is 28× (guarded) to 180× (unguarded) slower than its fast-path siblings
in the same YAML.

## Why the slow path is slow

### Measured: it does not scale with threads

`cpu-gemm-driver`, fp32, `--transA`, slow path, cores `192-216` (25 cores, SMT siblings idle).
The fast path is shown for scale.

| Threads (cores) | Slow path, 1024³ (ms) | Fast path, 1024³ (ms) |
| --- | --- | --- |
| 1 (1 core) | 2,642 | 99 |
| 25 (25 cores), `MAX_OMP_THREADS=25` | 2,329 | 21 |
| 64 on 25 cores (production cap) | 2,364 | 25 |
| 64 (64 cores) | 1,492 | 14 |

Going from 1 to 25 threads speeds the slow path up by only 1.1×. A thread-state trace (every
0.25 s from `/proc/<pid>/task/*/status`) showed all 25 threads running for the whole reference
call, so the threads are busy, not idle.

### Measured: the threads are waiting on one lock

`rocgdb` stack samples were taken during a 2048³ fp32 slow-path call, 25 threads, 4 samples:

| Top of stack | Thread samples |
| --- | --- |
| `__kmpc_critical_with_hint` → `__kmp_acquire_queuing_lock` → `__kmp_wait_4` | 54 |
| … → `__kmp_wait_4` → `sched_yield` (same lock, yielding) | 21 |
| `__kmp_suspend_64` → `pthread_cond_wait` (idle pool threads) | 24 |
| Main thread in `runGemm` | 1 |

### The code and its assembly

`Reference.cpp`, at the end of the per-output-element body of the `#pragma omp parallel` /
`#pragma omp for` loop in `ReferenceSolution<…>::SolveCPU`:

```cpp
                omp_set_num_threads(MAX_OMP_THREADS);
#pragma omp critical
                {
                    if constexpr(notCmplxAmaxD)
                    {
                        if(problem.outputAmaxD())
                        { /* amaxD = max(amaxD, |resultD|) */ }
                    }
                }
```

The I8 instantiation compiles to the following, executed once per output element (`.omp_outlined`
body, `objdump -dr`):

```
62c7a:  mov    $0x40,%edi
62c7f:  call   omp_set_num_threads
62c94:  call   __kmpc_critical            ; .gomp_critical_user_.var (one global lock)
62ca1:  cmpb   $0x1,0x134(%rax)           ; problem.outputAmaxD() — tested inside the lock
62ca8:  jne    62cd2
...
62ce2:  call   __kmpc_end_critical
```

Just before this, the same body calls `operator delete`. That is the per-element
`std::vector<Accumulator> actArgs` built for `Activation()`. Its cost was not measured
separately.

The K loop itself is cheap per instruction. The I8 version (`61da0`–`61dcf`) is unrolled by 2 and
does two `movsbl` loads, an `imul` and an `add` per MAC, stepping A and B by their K strides. With
the lock removed, single-thread throughput is about 0.41 GMAC/s against the fast path's
10.8 GMAC/s (1024³: 2,625 ms against 99 ms). So even an uncontended slow path is about 25× slower
per core.

### Memory layout is a secondary effect

fp32 and f8 inputs, 2048³, slow path, unmodified (64 threads on 25 cores):

| Layout | fp32 (ms) | f8 in (ms) |
| --- | --- | --- |
| NN | 13,900 | 16,704 |
| TN (A contiguous in K) | 11,273 | 12,453 |
| NT (B strided in K) | 20,152 | 18,856 |

The spread is about 1.5–1.8×. That is small next to the lock (a 9–30× change), so layout was not
pursued further.

## Options

### A. Take the critical section only when AMAX-D is requested (measured, prototype `refpath3`)

**Change.** Move `#pragma omp critical` inside `if(problem.outputAmaxD())`, and drop the
per-element `omp_set_num_threads`. The whole diff is in `/tmp/refpath/client_src3/src/Reference.cpp`.

**Further improvement (not measured).** AMAX-D problems would still serialise. They should use a
per-thread maximum combined once, e.g. `reduction(max:amaxD)`. No gfx950 config in this set hit
the slow path with AMAX-D.

| Measurement | Unmodified | Guarded |
| --- | --- | --- |
| Driver 1024³ fp32, 25 threads | 1,727 ms | 191 ms |
| Driver 2048³ fp32, 25 threads | 45,237 ms (other runs: 6.8–14 s) | 1,507 ms |
| Driver 1024³ fp32, 1 thread | 2,660 ms | 2,625 ms |
| Driver results | PASSED, max diff 0 | PASSED, max diff 0 |
| Replay, slow-path total (59 YAMLs) | 546.6 s | 119.5 s (−78%) |
| Replay, `sk_I8gemm_quick` pytest wall | 258 s | 49 s |
| Replay, `sk_f8gemm_quick` pytest wall | 184 s | 56 s |
| Replay, configs passing | 59 / 59 | 59 / 59 |

**Estimated effect on the aggregate.** Applying the replay ratio (119.5 / 546.6) to production's
740.9 s gives about 162 s, saving about 580 s, or 22% of the aggregate wall. This is an
estimate; the production baseline itself varies, as shown above.

**The 64-thread cap.** Once the lock is gone, the hard-coded cap no longer matters on 25 cores.
At 2048³ it measured 1,392 / 1,388 ms with the 64 cap against 1,437 / 1,373 ms with 25 threads.
`omp_set_num_threads` persists for the calling thread, so the cap also changes later OpenMP
regions; using `omp_get_max_threads()` would make it follow `OMP_NUM_THREADS`.

**`KMP_BLOCKTIME`.** With the default (200 ms), the guarded version measured 2,010–2,165 ms
against 1,388–1,437 ms with `KMP_BLOCKTIME=0`. This matches the
[per-solution bookkeeping](per_solution_bookkeeping_investigation.md) finding that
`KMP_BLOCKTIME=0` cut `cpu_reference_gemm` by 2–5×.

### B. Fast-path support for Int8 inputs and Float8 outputs (not prototyped)

These two type gaps are 94.6 s of the 119.5 s left after Option A (79%), and nearly all of the
production slow path. Same-size fast-path calls take 0.47–1.21 s, against 9.5–23 s guarded and
45–84 s unguarded.

For reference, driver measurements of the fast path at 4111³ on 25 cores:

| Inputs | Fast path, 4111³ (ms) |
| --- | --- |
| f64 | 651 |
| f32 | 761 |
| f8 | 541 |

What each extension needs:

- **Int8 inputs.** The accumulation must be exact. A float accumulator is not: a 4111-long I8 dot
  product can reach about 6.7e7, above 2²⁴. Options are the existing `double` instantiation
  (exact below 2⁵³; 651 ms at 4111³) or a dedicated int32/int64 inner kernel.
  - Int8 and Int32 outputs also need `storeFrom` specialisations that use the slow path's
    `SaturateCast<DType>` semantics.
  - Half and BF16 outputs already have write-back.
- **Float8 output.** A `storeFrom<AccumT, Float8>` (and BF8) write-back that matches the slow
  path's `SaturateCast`.
  - The F8 × Half → F8 case (`f8f16mix_f8s`) also needs the mixed-input shadow; the fast path
    already handles mixed F8/Half inputs (`F8 × Half → Float` is fast in `sk_f8gemm_quick`).

**Estimate.** If the 4111³ calls cost what their fast-path siblings do, these YAMLs' reference
time would drop from about 600 s in production to about 10 s. This is an inference from the
sibling measurements; equivalence with the slow path has to be validated per type, as it was when
FP8/FP4 were added to the fast path.

### C. Smaller rejection reasons (not prototyped)

Times are after Option A, from the replay.

| Reason | Remaining (s) | What it would take |
| --- | --- | --- |
| `gradient` | 8.8 | Gradient GEMM with bias reduction. The slow path also runs a separate reduction region. Larger change. |
| `useE` | 7.7 | Store the pre-activation value to E in the fast-path epilogue. |
| `K_not_multiple_of_mxBlockA` | 2.6 | MXFP4 K tails (K = 333, 255, 63 with block 32). Handle a partial last MX block. |
| Complex types | 4.1 | New fast-path variant. Probably not worth it at this size. |
| `NumElementsToValidate: 128` | 1.7 | Intended (sparse validation). Leave as is. |

### D. Make the rejection reason observable

The `rejectFast()` logging is compiled out with `if (false)`, so this audit needed a rebuilt
client. Two cheap ways to keep it reproducible:

- an environment or log-level switch for the reason line;
- a per-reason timer category, e.g. `solve_cpu_slow.<reason>`, so the aggregate shows it directly.

## Not measured / open items

- **The fast path's own 339 s.** It is out of scope here; most of it is the
  `subtile_mxfp8*` YAMLs (`gemm/gfx950/subtile_mxfp8.yaml` 143 s and the
  `*_correctness_regression*` variants).
- **The per-element `actArgs` vector** allocated and freed in the slow path is visible in the
  assembly but was not timed separately.
- **AMAX-D problems** would still take the lock per element after Option A; use a reduction.
- **`ductile/gfx950/f8_tn_mx_subtile.yaml`** reaches the reference with `elementsToValidate=128`,
  although the YAML sets `NumElementsToValidate: 0`. The cost is negligible (0.03 s); the cause
  was not investigated.
- **Run-to-run variance.** Production slow-path time (740.9 s) and the baseline replay (546.6 s)
  differ, because lock contention makes the unmodified slow path erratic; the same 2048³ problem
  measured 6.8 s to 45 s. The guarded version was stable (1.37–1.51 s).

## Scratch files

All in `/tmp/refpath/` (nothing in the repo was changed):

- **Sources:**
  - `client_src/`: `REFPATH` logging.
  - `client_src2/`: adds the `TL_MAXT` thread cap.
  - `client_src3/`: adds the guarded critical section.
- **Builds:** `/tmp/bookkeep/build_refpath{,2,3}/`, plus the relinked `cpu-gemm-driver{,2,3}`.
- **Replays:**
  - Runners: `run.sh` and `run3.sh`; config list `cfgs.txt`.
  - Per-config `.refpath` lines and analyses: `logs/`, `logs3/`.
  - Summaries: `summ.py` → `summ_base.txt`, `summ_fix.txt`; YAML table `table.md`.
- **Driver experiments:** `layout.out`, `threads.out`, `ab.sh`, and `gdb.out` (stack samples).
- **Assembly:** `i8_outlined_r.s`.
