# hipDNN Redirect Investigation — Summary

**Ticket:** [ALMIOPEN-1965](https://amd-hub.atlassian.net/browse/ALMIOPEN-1965)
— *Investigation that MIOpen shim can successfully redirect to hipDNN* (Story,
status: Implementation)
**Scope:** De-risk redirecting MIOpen API calls to hipDNN through the MIOpen
shim/wrapper. Prove handle lifecycle, measure overhead, and (stretch) run a real
operation end-to-end through `MIOpenDriver`.

Implementation lives in `projects/miopen/src/private/wrapper.cpp`. The redirect
is entirely opt-in via environment variables and always falls back to native
MIOpen on any failure, so default behavior is unchanged.

## Implementation

**Feature flags** (all off by default):

| Variable | Effect |
|---|---|
| `MIOPEN_HIPDNN_FORWARDING=enabled` | Master switch; pairs a hipDNN handle to each MIOpen handle |
| `MIOPEN_HIPDNN_FORWARDING_CONV=enabled` | Routes `miopenConvolutionForward` to hipDNN |
| `MIOPEN_HIPDNN_FORWARDING_TIMING=1` | Emits `[MIOpen->hipDNN]` cold/warm timings to stderr |
| `MIOPEN_HIPDNN_FORWARDING_MEMINFO=1` | Emits HIP free/total memory around handle init |

**Handle lifecycle.** `miopenCreate` / `miopenCreateWithStream` open a paired
hipDNN handle via `hipdnnCreate` and stash it in an
`unordered_map<miopenHandle_t, hipdnnHandle_t>`; `miopenDestroy` releases it via
`hipdnnDestroy`. hipDNN open failure is non-fatal — the MIOpen handle is still
returned (a stderr line is emitted; per RFC §4.4 this becomes a routing-policy
decision in a later phase). `miopenCreateWithStream` also has a reverse-order
init mode (hipDNN before MIOpen) to isolate init-order/memory effects.

**Convolution forwarding.** `miopenConvolutionForward` dispatches to
`try_forward_conv_to_hipdnn` when both forwarding + conv flags are set. That
helper builds an equivalent hipDNN **backend descriptor** convolution graph by
hand (the same cuDNN-style API the handle path uses — no frontend dependency),
finalizes an execution plan, caches it per problem shape, and executes through
the paired handle. Guards return `miopenStatusUnsupportedOp` (→ native fallback)
for: non-identity alpha/beta, `groupCount != 1`, spatial dim > 5, or any hipDNN
API failure. The plan cache is an in-process `unordered_map` keyed on tensor
dims/strides/types + conv params — **process-lifetime only, not persisted to
disk.**

## How it was tested

Single conv repeated 100 iterations (`-i 100`) via `MIOpenDriver`:

```
MIOpenDriver conv -n 16 -c 16 -H 16 -W 16 -k 16 -y 3 -x 3 \
    -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 1 -V 0 -i 100 -t 1
```

Four runs (`FourRunComparison.txt`): flag **off** then **on**, each run twice
(cold process / warm process) to expose cross-process caching behavior. Hardware:
1× MI308X (gfx942), ROCm 7.14.60850, MIOpen 3.5.2. Native solution selected:
`ConvBinWinogradRxSf2x3g1` (GPU kernel ~0.017 ms).

## Results

Per-phase overhead (redirect on, from the `[MIOpen->hipDNN]` timers):

| Phase | Cold | Warm |
|---|---|---|
| `hipdnnCreate` | ~2.1 ms | n/a (once per process) |
| conv plan build | **1.3 ms – 3.9 s** (highly variable) | cached (0) |
| conv execute | ~1.4 ms | ~27 µs / call |
| `hipdnnDestroy` | ~1 µs | n/a |

Wall-clock (`real`), cold vs. warm **process**:

| | 1st run (cold) | 2nd run (warm) |
|---|---|---|
| Redirect **off** (native) | 4.860 s | **0.213 s** |
| Redirect **on** (hipDNN) | 4.863 s | **4.121 s** |

Native MIOpen drops to 0.213 s on the second process because it reuses its
**persistent on-disk kernel cache**. The redirect path has only an in-process
plan cache, so every fresh process re-pays the cold plan build (dominated by the
multi-second `conv plan build`), and warm-process wall time stays ~4 s. The warm
*per-call* execute overhead itself is small (~27 µs).

## Acceptance criteria

- **Open/close of hipDNN handle in the shim — MET.** Paired `hipdnnCreate` on
  handle creation, `hipdnnDestroy` on destroy; cold create ~2.1 ms, destroy
  ~1 µs.
- **Timing/benchmarking overhead, cold + warm — MET.** Instrumented via
  `MIOPEN_HIPDNN_FORWARDING_TIMING`; numbers above. (Note: the driver's own GPU
  kernel timing does not flow through the redirect — it reports 0.000 ms / inf
  GFLOPs — so the shim's env-var timers are the usable measurement path.)
- **Stretch: full operation + run via MIOpenDriver — LARGELY MET.**
  `miopenConvolutionForward` is fully implemented over the hipDNN backend API and
  runs end-to-end through `MIOpenDriver` with `[MIOpen->hipDNN]` trace lines.
  Caveats: numerical verification was not enabled in these runs (`-V 0`), and
  hipDNN's *own* internal logging was not captured here (only the shim's trace).

## Concerns found

1. **No persistent cache for the redirect (highest impact).** The hipDNN plan is
   cached per process only. Native MIOpen persists compiled kernels to disk and
   so collapses repeat runs to 0.2 s; the redirect re-pays cold setup every
   process and stays ~4 s. Cross-process persistence (or enabling/sharing
   hipDNN's own kernel cache) is needed before the redirect is competitive on
   repeated invocations.
2. **Plan-build time is unstable** (1.3 ms on one cold run, 3.9 s on another).
   The multi-second case implies kernel compilation/autotuning happening inside
   plan finalize on some runs; the attribution/trigger needs to be understood.
3. **Driver benchmarking blind spot.** GPU kernel time / GFLOPs report
   0.000 / inf under the redirect, so standard driver perf reporting is unusable;
   only the shim timers work today.
4. **Limited coverage.** Conv forward only, identity scaling, group=1. Because
   the path falls back silently on any unsupported case, "redirect enabled" does
   not guarantee the redirect was actually taken — future work should surface
   when fallback occurs.
5. **Correctness not yet verified.** These runs used `-V 0` (no numerical
   verification), so the hipDNN path was only proven to execute, not to produce
   correct results. Follow-up: re-run with verification enabled (`-V 1`) to
   confirm the redirected output matches native MIOpen before relying on the
   path.

## Performance impact (summary)

- **Per handle:** +~2.1 ms cold create, +~1 µs destroy; negligible thereafter.
- **Per conv (warm):** ~27 µs host-side overhead per call (native GPU kernel is
  ~17 µs; not directly comparable, but the redirect adds modest steady-state
  cost).
- **Cold start / repeated processes:** dominated by the non-persisted plan build
  (up to ~3.9 s), keeping warm-process wall time at ~4 s vs. native 0.2 s. This
  is the headline regression for the redirect as currently implemented and the
  main item to resolve in the RFC.

## Related notes

- The `gfx94250.HIP.fdb.txt ... unreadable` warning seen in the logs is a native
  find-db CU-count resolution issue on MI308X, **unrelated to the redirect** and
  tracked separately. It does not affect the conclusions above.
