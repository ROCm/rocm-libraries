# hipBLASLt helper-kernel parallel compile — findings & plan

Status: **DELIVERED** (commit on `feature/hipblaslt-helper-split-compile`). The shipped
solution is **overlap**, not split: the monolithic helper compile runs on a worker thread
concurrently with the assembly build AND the end-of-run solution-library writing.
Byte-identical-per-kernel output, no runtime change. Measured ~28% total build-time
reduction (gfx90a). The split/multi-CO analysis below is retained as the
rejected-alternatives record.

## Delivered solution (overlap, caller-level)

`run()` launches `buildSourceCodeObjectFiles` via a `ThreadPoolExecutor(1)` right after
`writeHelpers` + `copyStaticFiles`, runs `writeSolutionsAndKernelsTCL` (now assembly-only)
and the end-of-run library writing (`passPostKernelInfoToLibrary` + `writeMsl`), then joins
the helper future after the master libraries are written (before build_tmp cleanup). So the
helper compile (the link step producing `Kernels.so-000-<arch>.hsaco`) overlaps BOTH the
assembly build AND the library serialization; it is on the critical path only if it
outlasts everything else. Output unchanged, so no `HipSolutionAdapter` change.

Verified (gfx90a, 2765 asm kernels, 222 helper kernels): total 92.8s->66.4s (~28%); helper
compile 28.9s fully hidden behind asm 24.2s + pass-info 2.1s + msl 7.9s. All 222 helper
kernels ISA-identical to the sequential build. Whole-file hashes are non-deterministic in
the baseline itself (amdclang build-id), so per-kernel ISA is the correct equivalence bar.

(An earlier iteration joined inside `writeSolutionsAndKernelsTCL`, overlapping only the
assembly build -> 70.9s. Joining after library writing recovers the residual when the
helper link outlasts the assembly build.)

Safe vs `ParallelMap2` (joblib **loky** backend = separate process pool, not
fork-with-threads). Helper thread and assembly path write distinct filenames into the
pre-created per-arch dirs — no collision.

---

## Rejected alternatives (split/parallel-compile) — measured

This supersedes memory of a "clear winner" — the archived approach was measured and is a
**loser**; splitting buys nothing the overlap doesn't, and costs a runtime change.

## Goal

Cut TensileCreateLibrary cold-build wall-time by parallelizing helper-kernel (HIP
source) compilation. The assembly-kernel path is already parallel; the source path
is one serial `amdclang` invocation on a monolithic `Kernels.cpp`.

## Test setup (reproducible)

- Container `hsc` from image `tensilelite-char:dev`, repo bind-mounted, rocisa built
  via `pip install ./rocisa/`, Tensile run from source via `PYTHONPATH`.
- Config: 4 aldebaran logic yamls → 12 helper objects → **74 `__global__` kernels**.
- Baseline full run: 15.7s, of which **`buildSourceCodeObjectFile` = 9.06s** (serial
  helper compile is the dominant cost even at this tiny scale; real gfx942 = 202 helpers).
- Output today: a single `library/<arch>/Kernels.so-000-<arch>.hsaco` per arch.

## Measured results (the important part)

| Approach | Time | Notes |
|---|---|---|
| **A. Monolithic (current)** | **8.40s** | one TU, one process, serial |
| C1. Split + single `amdclang` invocation, `-fgpu-rdc --offload-jobs=N` | **76s** (9× SLOWER) | identical for N=1,8,32,74 — `--offload-jobs` had **zero** effect |
| **C3. Split + process-parallel, non-rdc** | **~2.1–3.5s (≈3.5–4×)** | each TU runs its own backend in its own process |

### Why C1 (the archived approach) loses
`-fgpu-rdc` emits **LLVM bitcode** and defers ISA codegen to a **serial link-time
backend** over the combined module. The parallelizable part (front-end parse) is
cheap; the expensive part (instruction selection / regalloc) stays serial — and rdc
adds overhead on top. `-parallel-jobs` (the flag the archive relied on) does not exist
in amdclang 23; its successor `--offload-jobs` only fans out across offload *archs*,
not within a single-arch TU, so it does nothing here.

The archive's `-fgpu-rdc -shared` host-link path also no longer links in amdclang 23
(`undefined hidden symbol: __hip_gpubin_handle_*`) — it worked under amdclang 18.

### Why C3 wins, and its cost
Non-rdc `--offload-device-only -c` makes each TU do its **own full codegen** → real
per-process parallelism. But each TU then emits its **own finalized code object** ELF.
These cannot be cheaply merged back into one (merging = re-running the serial backend,
i.e. back to C1). So the fast path inherently produces **N code objects per arch**.

## Equivalence (corrects the "byte-equivalent" premise)

- **Whole-file byte-equivalence is impossible** with any parallelization: the
  monolithic output is one serial TU by definition; splitting changes file layout,
  ordering, and padding.
- **Per-kernel machine code IS byte-identical.** Verified: kernel
  `Cijk_SB_GG_BiasS_HAS_ScaleAlphaVec_PostGSU8_VW2` disassembles to the **exact same
  702-instruction stream** whether compiled independently or inside the monolith
  (only load address + trailing section padding differ). Expected — helpers are
  independent `__global__`s with no cross-kernel optimization.
- Achievable acceptance bar: **identical per-kernel ISA for every helper** (offline,
  no GPU needed) + functional/numerical check on hardware.

## Runtime impact (the one non-obvious cost)

The helper loader hardcodes a single filename:
`HipSolutionAdapter.cpp:477` → `"Kernels.so-000-" + arch + ".hsaco"`.
C3 emits multiple helper code objects, so this must change to load a **set** (glob
`Kernels*.so-000-<arch>.hsaco`). Low risk: `<glob.h>` is already included there, and
the **assembly** loader (`tensile_host.cpp:2831`) already directory-iterates and loads
every arch-matching `.co` — the pattern is established.

## Recommended design

1. **Bucket, don't shatter.** Split the helper set into `min(numHelpers, cpuThreads)`
   buckets, each written as one `.cpp`. #code-objects ≈ #cores (not 74+), keeping the
   loader change small while still saturating cores. Per-kernel ISA stays identical
   (within-bucket is still monolithic-equivalent per independent kernel).
2. Generate bucket sources via `ParallelMap2` (reuse existing infra).
3. Compile buckets with the **existing** non-rdc flags (`--offload-device-only -c`),
   one process per bucket via `ParallelMap2`, each unbundled per-arch as today.
4. Emit `Kernels-<b>.so-000-<arch>.hsaco`; update `HelperKernelCache` key to the
   bucket set; update `HipSolutionAdapter` helper load to glob.
5. Verification gate: for a fixed config, assert the **union of per-kernel ISA** equals
   the monolithic baseline's, kernel-for-kernel (script already prototyped).

## Files to touch

- `tensilelite/Tensile/TensileCreateLibrary/Run.py` — `writeHelpers`→bucketed
  `writeHelper`s; `writeSolutionsAndKernelsTCL` helper-compile section (~line 652).
- `tensilelite/Tensile/Toolchain/Source.py` — `buildSourceCodeObjectFiles` to take
  multiple kernel sources / parallelize.
- `tensilelite/Tensile/Toolchain/HelperKernelCache.py` — multi-file cache key.
- `tensilelite/src/hip/HipSolutionAdapter.cpp:477` — glob helper COs.

## Open decision for David

Whole-file byte-equivalence is off the table (physically impossible with parallelism).
The viable win **requires a runtime loader change** to load multiple helper code
objects. Confirm: proceed with C3 (bucketed multi-CO + loader glob, per-kernel-ISA
equivalence gate)? It's the only measured path that actually reduces build time.
