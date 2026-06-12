# Diagnosis — allsweep6144rcrfp16 (bridge vs old-TE fp16/rcr A/B sweep)

Date: 2026-06-11 · Data: `allsweep6144rcrfp16.csv`, `full_sweep.csv`, `full_sweep.log`
Confluence: https://amd.atlassian.net/wiki/spaces/MLSE/pages/1733670075
Fix: commit `4d147777b07` on `muozturk/dispatcher-gemm-bridge` (PR #8123)

---

## 1. Kernel funnel — and the ROOT CAUSE of the 522 (now fixed)

The Confluence report shows `6144 -> 2384 -> 363` and implies 363 is a pure
"comparability intersection" bounded by old-TE legality. The real funnel:

| Stage | Count | Gate |
|---|---|---|
| Raw tile x trait product | 25920 tiles | `default_config.json` |
| `validate_kernel_config` (dispatcher validator) | **6144** | the "expanded configs" |
| codegen arch filter **as it ran** (hard-coded compv4) | **512** | matches ~522 `.so` on disk |
| codegen arch filter using each config's **real** traits | **1520** | what it should emit (post-fix) |
| old-TE binaries built (NO gfx942 arch filter) | **2385** | old-TE instance builder |
| bridge ran on all 5 shapes | **363** | 159 of the built set failed at runtime |

**Root cause (FIXED):** the codegen arch filter
`unified_gemm_codegen.py:_is_tile_arch_valid` hard-coded `pipeline="compv4"` /
`scheduler="intrawave"` and never threaded in the config's actual trait
(`_get_configs_for_variant` and the `_get_tile_configs` pre-filter both called it
trait-blind). `compv4` has the strictest MFMA-geometry constraints, so tiles
legal under `mem`/`compv3` were judged as compv4 and dropped. Two validators
disagreed: the dispatcher's `validate_kernel_config` blessed 6144, but the
codegen arch filter then silently emitted only 512 (returncode 0, no `.hpp`,
logged as "No .hpp matching ... after codegen").

**Signature** — generated-header counts on disk (buggy build) vs old-TE: compv4
roughly preserved, compv3/mem decimated ~5x.

| pipeline | bridge (buggy) | old-TE |
|---|---|---|
| compv4 | 141 | 224 |
| compv3 | 141 | 721 |
| mem | 272 | 1440 |

**Fix (`4d147777b07`):** thread the trait's real pipeline/scheduler into
`_is_tile_arch_valid`; the tile pre-filter keeps a tile if legal under ANY
configured pipeline/scheduler. Verified: emitted set 512 -> 1520 (compv3 464,
compv4 128, mem 928); a previously-rejected `compv3 64x64x192` now codegens a
header end-to-end.

**Residual 1520 vs 2385:** expected — old TE applies no gfx942 arch filter, so
it builds whatever its instance builder enumerates (some may be arch-invalid on
gfx942). The bridge's `ArchFilter` is deliberately conservative; closing the
rest is a separate arch-filter-strictness question, not this bug.

Counts verified on disk (pre-fix build):
- old-TE `benchmark_gemm_universal_fp16_rcr_*` : 2385
- bridge `libgemm_fp16_rcr_*.so` : 522 · generated `.hpp` : 554
- unique kernels in `full_sweep.csv` / `allsweep6144rcrfp16.csv` : 363

---

## 2. The 159 runtime failures (522 -> 363) — root cause

128 of the 159 are the **192-tile family** (`192x64x64`, `64x192x64`). All 128 fail
on all 5 shapes. Root cause is a **tile/shape divisibility mismatch with padding off**.

Code gate — `dispatcher/include/ck_tile/dispatcher/backends/tile_backend.hpp:48-52`:

```cpp
// is_supported(): check dimension divisibility if padding not enabled
if(!pad_m && problem.M % tile_m != 0) return false;
if(!pad_n && problem.N % tile_n != 0) return false;
if(!pad_k && problem.K % tile_k != 0) return false;
```

Chain:
1. Each `libgemm_<stem>.so` registers exactly ONE kernel.
2. Worker calls `dispatcher_run_gemm` -> `select_kernel(problem)`
   (`dispatcher/bindings/ctypes/gemm_ctypes_lib.cpp:222`).
3. The kernel is non-padded (`..._False_False_False_...` = pad_m/n/k all false).
4. `is_supported` runs `M % 192 != 0` -> reject. The 5 sweep shapes
   (512, 1024, 2048, 4096, and 1024x512x256) are **none divisible by 192**.
5. `select_kernel` returns nullptr for every shape ->
   `gemm_ctypes_lib.cpp:229  return -2; // No suitable kernel`
   -> worker emits `status -2` -> log shows `FAILED`.

Corroboration: `full_sweep.log` has `Error: kernel returned status -2` **635 times
= 127 kernels x 5 shapes** — i.e. the whole 192-tile family failing on every shape.

**This is a test-matrix design flaw, not a bridge defect.** old TE applies the
identical divisibility rule, so 192 tiles are simply un-runnable on this shape set
unless padding is enabled or a shape divisible by 192 is added (768/1536/3072).

The other 31 failures are a mixed bag:
- `_grouped` / `_streamk` variants swept in under the `fp16_rcr` name (different op
  types; the plain `dispatcher_run_gemm` path does not drive them).
- a few small-tile kernels with other `IsSupportedArgument`/verify rejections.

---

## 3. tile_k = 192 codegen failures — same root cause as §1 (FIXED)

`64x64x192` (tile_k=192) kernels were logged as `FAIL codegen ... No .hpp
matching ...hpp after codegen` and never produced a `.so`. Originally filed as a
"separate codegen bug," but this was the SAME hard-coded-compv4 arch filter from
§1: those tiles are legal under compv3/mem but were judged as compv4 and
silently skipped. After the fix (`4d147777b07`), a `compv3 64x64x192` config
codegens a header end-to-end. (Not every tile_k=192 combo is necessarily
recovered — confirm with a full rebuild — but the dominant cause was the filter,
not a real codegen defect.)

---

## 4. The suspicious >=20% gaps — old-TE BENCHMARK-BINARY artifact (clock/env), NOT a bridge speedup

16 rows have gap >= 20%; all are `compv4 + intrawave + 1024^3`, reproducible (CV < 1%).
The earlier compiler-asymmetry theory (below) was **investigated and DISPROVEN**. The
real cause is the *measurement harness*, proven on-GPU (gfx942 MI300X) 2026-06-11.

### What it is NOT (each ruled out empirically)
- **Not the kernel.** rocprof shows both paths run the byte-identical symbol
  `ck_tile::kentry<1, GemmKernel<...>>`, 150 dispatches back-to-back, gap=0, no flush
  kernels in either.
- **Not the compiler/flags.** Rebuilt the bridge kernel 4 ways — hipcc current,
  hipcc+`-DNDEBUG`, hipcc without `-mllvm enable-noalias-to-md-conversion=0`, and
  `/opt/rocm/llvm/bin/clang++ -x hip` with old-TE's exact flags
  (`-DNDEBUG -fbracket-depth=1024 -ftemplate-backtrace-limit=0`). All four: **187-189
  TFLOPS** at 1024^3. Compiling old TE's OWN generated header through the bridge path
  also gives **189**, not 156.
- **Not bench knobs.** warmup/repeat/flush_cache/rotating_count/timer all toggled on
  BOTH sides — neither number moves (bridge stays ~188, old-TE binary stays ~156).
  old-TE and bridge defaults are already identical (split_k=1, w=50, r=100, flush=true,
  rot=1000, gpu timer); `stream_config` fields line up field-for-field.
- **Not allocation/placement.** DeviceMem vs raw hipMalloc: both 190. Up to 4 GB of
  decoy device allocations before A/B/C: still 190.
- **Not stale timing code.** All host timing headers (`kernel_launch.hpp`,
  `rotating_buffers.hpp`, `flush_icache.hpp`, `timer.hpp`, `stream_config.hpp`) are
  byte-identical between the bridge tree and the develop-parity worktree; compiling
  the minimal harness against either tree gives 190.
- **Not a measurement bug.** rocprof hardware timestamps confirm the GPU kernel
  *genuinely* runs longer in the old-TE binary process: **13.78 us vs 11.34 us** (the
  reported 156 vs 189 is real device time).

### What it IS
A clean standalone harness (raw hipMalloc, null stream, direct `SelectedKernel::launch`,
same stream_config) measures the old-TE kernel at **189-194**. The old-TE *standalone
benchmark binary* measures the SAME kernel at **156**. The slowdown is a per-process
**GPU clock + execution-environment artifact** of that binary:
- PMC counters (clock-normalized): old-TE kernel = **366k cycles** vs mini **339k**
  (+8%, extra memory-stall cycles).
- Non-PMC wall ratio (1.22) exceeds the cycle ratio (1.08) -> the remaining ~13% is a
  **lower sustained SCLK** in the old-TE binary process.
Why shape-selective: at 2048^3/4096^3 both processes hit the power/thermal cap and clock
converges (small gap); at 1024^3 there is headroom and the two processes' DPM governors
diverge most (peak gap). Pipeline-selective because compv4 is the most compute/clock-
bound of the three.

### The fix (apples-to-apples): measure both kernels through the SAME harness
`ab_same_harness.py` builds the old-TE kernel into a `.so` from old TE's own generated
header and runs BOTH it and the bridge `.so` through the SAME worker
(`run_one_gemm_kernel.py`). The gap then collapses to **~+/-0.5%** at 1024^3 (was
+20..+24% vs the standalone binary). Sample (uniform harness, max of 3):

| shape | bridge | oldTE | gap% |
|---|---|---|---|
| 512^3 | 38.77 | 38.78 | -0.01 |
| 1024^3 | 189.19 | 189.41 | -0.12 |
| 2048^3 | 295.59 | 297.10 | -0.51 |
| 4096^3 | 369.48 | 369.85 | -0.10 |

Conclusion: the >=20% numbers are **not a bridge advantage**. The bridge and old TE run
the same kernel at the same speed; old TE's *standalone benchmark binary* under-extracts
its own kernel by ~18-20% at 1024^3/compv4 due to its process clock/execution state.
Full evidence: `ab_same_harness.py` + `ab_same_harness.out`.

> Historical note: the original theory here was a `hipcc` vs `clang++`-HIP toolchain
> asymmetry ("device kernel not byte-identical at the binary level"). That was wrong --
> all toolchain variants measure ~189, and the device symbol is identical.

---

## 5. `oldte_built` column + does the A/B do a validity check?

`allsweep6144rcrfp16.csv` columns:
`kernel, pipeline, shape, oldTE_median, bridge_median, cv_oldTE, cv_bridge, gap_pct, oldte_built`

- `oldte_built` (harness line 127): `(OLD_BIN_DIR/f"benchmark_gemm_universal_{stem}").exists()`
  — a boolean "the old-TE benchmark binary file exists on disk". A vestigial guard;
  it is **True for all 1815 rows** (every A/B kernel has a built old-TE binary).
  It does NOT mean old TE ran, nor that results matched.
- `gap_pct = (bridge_median - oldTE_median)/oldTE_median * 100`, positive = bridge faster.
  Medians are TFLOPS, median over 3 interleaved repeats (A/B order flipped per repeat).

**The A/B harness does NO correctness/validity check.** It only compares throughput.
It never compares bridge output vs old-TE output and imports no numpy/reference.

Correctness lives in the SEPARATE full sweep: `full_sweep.csv` has `verified` + `max_rel`
columns (bridge output vs an fp32 numpy reference `A@B`, tol fp16). 1815/1815 verified
there. The A/B's 363 kernels are a subset of that verified set, so A/B trusts the prior
verification rather than re-checking.

---

## 6. Recommended actions
- DONE: fix the codegen arch-filter hard-code (commit `4d147777b07`, §1). Next:
  full rebuild on gfx942 to confirm the emitted set lands near 1520 and re-run
  the A/B sweep on the larger comparable population.
- Drop non-power-of-2 (192) tiles from the sweep config, or add shapes divisible by 192,
  or generate 192 tiles with padding on. Removes ~127 noise failures.
- Stop reporting 522->363 as a clean pass; ~127 drops are the divisibility mismatch.
- Re-frame the >=20% compv4 numbers: they measure a compiler difference between two build
  trees, not a bridge speedup. Rebuild matched before claiming a perf win.
- Decide whether the residual 1520-vs-2385 gap matters: it is arch-filter
  conservatism vs old-TE applying no arch filter, not a bridge defect.
