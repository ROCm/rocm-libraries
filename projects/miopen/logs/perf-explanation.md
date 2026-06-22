# Performance Analysis: hipDNN Forwarding Redirect vs. Native MIOpen

Analysis of `FourRunComparison.txt` — why repeated runs speed up dramatically
with the redirect flag **off** but stay slow with it **on**.

## What the four runs show

| Run | Flag | `real` | Dominant cost |
|-----|------|--------|---------------|
| 1 | off, cold | 4.860s | MIOpen kernel compile + tuning (one-time) |
| 2 | off, warm | **0.213s** | everything served from disk cache |
| 3 | on, cold | 4.863s | compile cost (hidden from the hipDNN timers) |
| 4 | on, warm | **4.121s** | `conv plan build (cold): 3.908 s` — explicit in the log |

## Why flag-off collapses 4.860s -> 0.213s

The ~4.6s on the first flag-off run is one-time GPU kernel compilation
(hiprtc/comgr) plus solver find/tuning. MIOpen **persists the compiled kernel
binaries to an on-disk cache** (`~/.cache/miopen/...`). The second process loads
those binaries instead of recompiling, so the entire compile cost evaporates
-> 0.213s.

Note the `gfx94250.HIP.fdb.txt ... unreadable` warning fires on *both* flag-off
runs, so the find-db text file is irrelevant here — the speedup comes from the
separate compiled-kernel cache, which survives across processes.

## Why flag-on stays ~4.8s -> 4.1s

The hipDNN forwarding path caches its finalized plan in an **in-memory,
process-lifetime map only** — not on disk. From
`projects/miopen/src/private/wrapper.cpp`:

- `conv_plan_cache()` is a `static std::unordered_map<std::string, ConvPlanEntry>`
  (wrapper.cpp:2145) — lives and dies with the process.
- The cold/warm split (`build_conv_plan` vs. cached lookup) is keyed in that map
  at wrapper.cpp:2388-2400. The comment at :2199-2200 even says
  "process-lifetime cache."

So every fresh process misses the cache and pays the full hipDNN plan build,
which includes the backend's own kernel compilation/autotuning. Run 4 captures
it cleanly: **`conv plan build (cold): 3,908,480,225 ns ~= 3.9 s`**, which is
essentially the entire 4.1s wall time. The warm executes after that are ~27us —
the actual conv is trivially fast; the cost is rebuilding the plan from scratch
each launch.

## The run-3 vs run-4 plan-build discrepancy

In run 3 the instrumented `conv plan build (cold)` reads only 1.33ms, yet wall
time is still 4.863s — the compile cost landed *outside* the plan-build timer
(likely warmed lazily, or shared with the comgr/hiprtc cache from the
immediately preceding flag-off runs, then triggered at first execute). In run 4
the same ~3.9s of work fell squarely inside the plan-build timer. Either way the
wall clock confirms each flag-on process pays the full multi-second tax.

## Bottom line

Flag-off is fast on repeat runs because MIOpen has a **persistent (disk) kernel
cache**. The hipDNN redirect path has only an **in-process plan cache**, so it
recompiles/retunes on every launch and never benefits from warm-start across
processes. To make the redirect competitive on repeated invocations, the plan
build needs a persistent, on-disk cache (or the hipDNN backend's own kernel
cache must be enabled/shared) — otherwise the per-process ~4s plan build will
dominate every cold start.

## "But hipDNN feeds back into MIOpen — why doesn't MIOpen's cache help?"

It's a reasonable question: hipDNN's backend *does* ultimately call into MIOpen,
so some MIOpen caching is available to the redirect path. The reason it doesn't
rescue warm-run performance is that "MIOpen caching" is not one monolithic thing.
There are three distinct caches at two layers, and the one that makes the native
warm run fast is **not** the one the redirect's bottleneck depends on.

### The three caches in play

**Layer 1 — inside MIOpen (shared by both paths in principle):**

- **(a) Compiled-kernel binary cache** — `~/.cache/miopen/...`, the hiprtc/comgr
  output. Persistent across processes. This is what collapses native run 1 -> 2
  (4.860s -> 0.213s).
- **(b) Find-db / solver-selection cache** — the `.fdb.txt` files. The *system*
  one is broken on this box (the CU-count fallback bug, `gfx94250` vs
  `gfx942e4`, see the side note below), so MIOpen has no system find-db to lean
  on for the redirect path.

**Layer 2 — the redirect shim itself:**

- **(c) `conv_plan_cache()`** — `wrapper.cpp:2145`, an in-process
  `unordered_map` of finalized hipDNN plans. It is a function-local `static`, so
  it lives and dies with the process.

### Why feeding back into MIOpen doesn't save you

Cache (a) *does* partially engage on the redirect path — that's exactly why run 3's
`conv plan build` timer read only 1.33ms: the kernel binaries were already on disk
from the preceding native runs, so no recompile happened.

But the expensive ~3.9s in run 4 (`conv plan build (cold): 3.9s`, captured inside
`build_conv_plan`) is **not** kernel compilation. It's the hipDNN
plan-construction chain — `EngineHeur` -> `EngineConfig` -> `Finalize`
(wrapper.cpp:2258-2291), running with `HIPDNN_HEUR_MODE_FALLBACK` — which under
the hood drives MIOpen's **solver find / heuristic selection / autotune**. That
step is the job of the find-db cache (b), and on this path the persistence isn't
engaging: the system find-db is broken, and the redirect path isn't landing
reusable find results the way the native warm run does.

So each cold process on the redirect path:

1. Misses the shim's plan cache (c) — it's in-process, gone with the last process.
2. Re-enters `build_conv_plan`, which re-runs the find/heuristic/autotune work.
3. Gets *no* persistent shortcut for that work, because the find-db (b) that
   would cache it isn't functioning here.

The compiled-kernel cache (a) is the only fully-working persistent cache, and it
only covers the compile step — not the multi-second solver-selection step that
dominates the redirect's cold build.

### Net

"hipDNN feeds back into MIOpen" is true, and MIOpen's *kernel* cache is genuinely
shared. But the redirect's cost is in plan/solver selection, and the two caches
that could persist *that* across processes are both unavailable here — the shim's
own cache is in-memory only (wrapper.cpp:2145), and MIOpen's find-db is broken by
the separate CU-count fallback bug. Fix either one (give the shim a disk-backed
plan cache, or fix the find-db fallback so solver selection persists) and the
warm redirect runs should drop toward native warm times.

**Certainty note:** the in-process plan cache (c) is verified directly from the
code. The exact attribution of the 3.9s to MIOpen solver-find vs. hipDNN-layer
finalization is inferred from the timers and log behavior — instrumenting inside
`build_conv_plan` around the `EngineHeur`/`Finalize` calls would confirm which
sub-step owns the time.

## Side note: the "fdb.txt is unreadable" warning

Every run logs:

```
MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable:
".../build/share/miopen/db/gfx94250.HIP.fdb.txt"
```

...yet the file actually on disk is named `gfx942e4.HIP.fdb.txt`. The discrepancy
is the **compute-unit count**, not the architecture. Both names share the
`gfx942` arch prefix; the differing suffix is MIOpen's encoding of the GPU's CU
count.

From `src/include/miopen/handle.hpp:229-240`:

```cpp
static std::string GetDbBasename(const TargetProperties& target, size_t num_cu)
{
    auto ret = target.DbId() + [&]() {        // DbId() == "gfx942"
        std::ostringstream ss;
        if(num_cu <= 64)
            ss << '_' << num_cu;              // decimal, with underscore
        else
            ss << std::hex << num_cu;         // hex, no underscore
        return std::string(ss.str());
    }();
    return ret;
}
```

The suffix is the CU count: `_<decimal>` if <= 64, otherwise bare **hex**.
Decoding both names (both take the hex branch, hence no underscore):

- Looked-for `gfx94250` -> `0x50` = **80 CUs**
- On-disk `gfx942e4` -> `0xe4` = **228 CUs**

The log reports `GPU Model: 1 x AMD Instinct MI308X`. At runtime
`GetMaxComputeUnits()` returns **80** for this MI308X, so MIOpen builds the
basename `gfx94250` and looks for `gfx94250.HIP.fdb.txt`. But the find-db that
ships in the build tree was tuned for a **228-CU** gfx942 part (MI300-class),
named `gfx942e4.HIP.fdb.txt`. The MI308X's CU count doesn't match the count
baked into the shipped filename, so the exact-name lookup misses.

MIOpen *does* have a nearest-CU fallback that should select `gfx942e4` for this
80-CU part, but it fails to engage for on-disk databases due to a filter bug, so
MIOpen ends up with no system find-db and relies on the compiled-kernel cache for
the warm-run speedup. This warning is orthogonal to the redirect. The fallback
bug is written up separately for handoff in
[`find-db-fallback-bug.md`](./find-db-fallback-bug.md).
