# hipDNN vs native MIOpen — conv plan-build caching reproducer

This folder compares running the same convolution two ways, each as repeated
**separate processes**, to show a difference in cross-process caching:

- **hipDNN backend C API** (`hipdnn_conv_cache_repro.cpp`) — builds a conv
  forward execution plan and times it. Every fresh process re-pays a multi-second
  plan build; it is never amortized across processes.
- **native MIOpen** (`MIOpenDriver`) — the same conv called directly. On a GPU
  that needs runtime kernel compilation, the second run is dramatically faster
  thanks to MIOpen's persistent on-disk compiled-kernel cache.

Conv problem (identical for both paths): `N16 C16 H16 W16 K16 R3 S3 pad1 stride1
dil1`, NCHW, fp32.

## Prerequisites

A built `rocm-libraries` checkout that includes:

- hipDNN backend (`libhipdnn_backend.so`) and its generated `hipdnn_backend_export.h`,
- the hipDNN miopen engine plugin (`libmiopen_plugin.so`),
- MIOpen (`libMIOpen.so`) and `MIOpenDriver`,
- ROCm with `hipcc` (default `/opt/rocm`, override with `ROCM_PATH`).

The scripts **auto-discover** these artifacts by name anywhere under the repo, so
they do not care what your build directory is called or how deep this folder is
nested. They locate the repo root with `git rev-parse` from the script's own
location, so they work regardless of your current working directory.

If auto-discovery picks the wrong thing (e.g. you keep several builds), override
any path explicitly — see the env vars documented at the top of `common.sh`
(`ROCM_LIBRARIES_ROOT`, `HIPDNN_BACKEND_LIB`, `MIOPEN_DRIVER`, etc.).

## hipDNN backend path

The reproducer uses **only** the hipDNN backend C API (`hipdnnBackend*`): tensor
descriptors → conv-fwd op → op graph → `EngineHeur(FALLBACK)` → `EngineConfig` →
`ExecutionPlan` → finalize, then a VariantPack execute. It does not call MIOpen's
public API.

```bash
./build_repro.sh           # compiles the reproducer against the discovered hipDNN build
./run_repro.sh             # two fresh processes, warm disk caches
./run_repro.sh --cold      # wipe MIOpen find-db + kernel cache before process 1 (true cold start)
```

> **Cache location note (network homes):** MIOpen keeps two on-disk user
> databases:
>
> - the **compiled-kernel cache** — normally `~/.cache/miopen`
> - the **user find-db / perf-db** — normally `~/.config/miopen` (records which
>   solver/kernel to use, so MIOpen can skip the `find` search on later runs)
>
> **If the home directory is on a network drive (NFS, etc.), MIOpen falls back to
> local `/tmp` copies instead:** `/tmp/.cache/miopen`, `/tmp/.config/miopen`, and
> `/tmp/miopen-lockfiles`. On such a machine, wiping only `~/.config/miopen` will
> *not* produce a cold start — the real databases survive under `/tmp`. The
> `--cold` flag wipes **all** of these (home and `/tmp` fallbacks, cache and
> find-db) so a cold start is genuinely cold regardless of where the databases
> live and which solver the conv resolves to.
>
> **Which database actually drives the native cold→warm speedup?** For this conv
> it is the **find-db (`~/.config/miopen` → `/tmp/.config/miopen`), not the kernel
> cache.** Isolating each on the MI300X confirms it (`MIOpenDriver`, same conv):
>
> | Cold start by clearing… | 1st run | 2nd run |
> |-------------------------|---------|---------|
> | both databases | 4.06 s | 0.53 s |
> | **only the find-db** (`.config`) | **4.12 s** | 0.51 s |
> | **only the kernel cache** (`.cache`) | **0.54 s** | 0.51 s |
>
> Clearing the find-db alone reproduces the full ~4 s cold cost; clearing the
> kernel cache alone has essentially no effect. The cold cost is the **`find`
> step** (solver search/benchmark), not kernel compilation — this conv's winning
> solver is a precompiled Winograd binary shipped inside `libMIOpen.so`, so
> nothing is compiled at runtime and the kernel cache has nothing to save. The
> kernel cache only matters on a part/conv whose solver *does* require a runtime
> hiprtc/comgr build (e.g. a CU count without matching precompiled kernels), which
> is why `--cold` still clears both.

`run_repro.sh` points `HIPDNN_PLUGIN_DIR` at the discovered `libmiopen_plugin.so`
and puts the directories of `libhipdnn_backend.so` and `libMIOpen.so` on
`LD_LIBRARY_PATH`.

### What it shows

Each invocation is a separate process that builds the plan exactly once. Typical
result on gfx942 (MI300X):

| Metric | Process 1 | Process 2 (fresh) |
|--------|-----------|-------------------|
| `conv plan build` | ~2.4 s | ~2.4 s — **not amortized** |
| `conv execute (cold)` | ~2 ms* | ~2 ms |
| `conv execute (warm)` | ~27 µs | ~27 µs |

\* On a true cold start (`--cold`, empty `~/.cache/miopen`) the first process's
`execute (cold)` is multi-second (one-time hiprtc/comgr kernel compile) and the
plan build can reach ~14 s; once the compiled-kernel disk cache is populated,
`execute (cold)` collapses to ~2 ms **across processes** — but the plan build
stays ~2.4 s in every fresh process.

**Conclusion:** MIOpen's compiled-kernel disk cache (`~/.cache/miopen`) *does*
cross processes — that's why `execute (cold)` drops to milliseconds on the second
process. The hipDNN **plan build (heuristic/solver-find) does not persist across
processes**, so it re-pays ~2.4 s on every fresh launch. The non-caching is
inherent to the hipDNN backend plan-build path.

## Native MIOpen comparison (MIOpenDriver)

To contrast against calling MIOpen directly — and to reproduce the native
**second-run speedup** — run the same conv through `MIOpenDriver`:

```bash
./run_miopen_driver.sh           # native MIOpen, two fresh processes (run 2 should be faster)
./run_miopen_driver.sh --cold    # wipe the MIOpen kernel cache before run 1 (true cold start)
```

(`--cold` wipes the MIOpen kernel cache *and* find-db, including their `/tmp`
fallbacks — see the cache location note above; on a network-mounted home the real
databases live under `/tmp`, and the find-db must be cleared to see the native
cold→warm collapse.)

The exact command it runs (same conv as the C++ reproducer) is:

```bash
MIOpenDriver conv -n 16 -c 16 -H 16 -W 16 -k 16 -y 3 -x 3 \
                  -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1 -V 0 -i 100 -t 1
```

(`-F 1` forward-only, `-V 0` no verify, `-i 100` iters, `-t 1` timed). Wrap it in
`time` and run it twice to see wall-clock cold vs. warm. The script does this for
you and auto-resolves the driver + library paths.

### Expected native result — and an important caveat

The dramatic second-run speedup is **GPU-dependent**. Previously observed on an
MI308X (80 CU):

| Native run | wall time |
|------------|-----------|
| Run 1 (cold) | 4.860 s |
| Run 2 (warm) | **0.213 s** |

The collapse comes from MIOpen's persistent on-disk compiled-kernel cache
(`~/.cache/miopen`). It only shows up when run 1 actually triggers hiprtc/comgr
**compilation** — which happens when the GPU's compute-unit (CU) count does not
match the kernels precompiled/shipped for the part.

On a part that already has matching precompiled kernels, run 1 is *already* fast
and there is little to no second-run delta. For example, on the 192-CU MI300X
this folder was last run on, the conv resolves to a precompiled Winograd binary
solver (`ConvBinWinogradRxSf2x3g1`), `~/.cache/miopen` stays empty, and both runs
are ~0.5 s. To observe the big cold→warm collapse, run on a
part whose CU count forces compilation (e.g. the MI308X), or otherwise force a
solver/path that requires a runtime kernel build.
