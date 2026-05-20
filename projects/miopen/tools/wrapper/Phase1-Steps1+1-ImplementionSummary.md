# RFC 0001 Phase 1 Investigation — Implementation Summary

> **Historical.** This file describes the *investigation phase* deliverables
> only. The actual public/private split that this investigation gated landed
> in the same change set that produced `RemainingWork.md`. For current
> status, see `tools/wrapper/RemainingWork.md` and the updated §6 of
> `docs/rfcs/0001_HipdnnForwardingWrapper_phase1_investigation.md`.

Investigation infrastructure for RFC 0001 Phase 1 is in place, gated entirely behind the existing `MIOPEN_ENABLE_HIPDNN_WRAPPER` flag.

## What landed (during the investigation)

- `cmake/InvestigationHipdnnWrapper.cmake` — included from top-level CMakeLists only when the flag is ON. Registers CTest checks under the `investigation` label. No effect on the flag-off build, and (per RFC §6 Q1, Q4) no longer runs any code generators at configure time.
- `src/private/miopen_private_rename.h` (Q1; was bootstrapped at `tools/wrapper/`, moved to `src/private/` in the implementation step) — hand-maintained `#define miopenFoo miopenFoo_impl` block for all **263** public entry points (corrects the stale "172 distinct" inventory in §2). Bootstrapped one-shot from a now-retired generator (see RFC §6 Q1). Intentionally NOT installed (Q5 mitigation).
- `src/private/wrapper.cpp` (Q4; was bootstrapped at `tools/wrapper/`, moved to `src/private/` in the implementation step) — hand-maintained Phase 1 pass-through wrapper, one `extern "C"` stub per entry point. Bootstrapped one-shot from a now-retired generator (see RFC §6 Q4); from here on, edits to the public C surface are mirrored here by hand and enforced by the count-parity CTest.
- `tools/wrapper/consumer_smoke.c` + `check_consumer_smoke.sh` (Q5) — installs-tree consumer test that asserts no `_impl` symbols leak.
- `tools/wrapper/symbol_diff.sh` (Q2/Q6) — `dump`/`diff` for public symbols, SONAME, DT_NEEDED.
- `tools/wrapper/wrapper_overhead.sh` + `microbench_settensor.cpp` (Q7) — three-workload CSV harness against flag-off and flag-on prefixes.
- CTest checks `investigation_q4_stub_count` (covers Q1 + Q4 drift between miopen.h, the rename header, and wrapper.cpp) and (opt-in) `investigation_q5_consumer_smoke`.
- §6 of the RFC investigation doc filled in with status for each Q.

## What was deferred to the implementation step (now landed)

Building `MIOpen_private`, applying the rename via `-include`, packaging a `libMIOpen.so` wrapper artifact, and the empirical Q2/Q5/Q6 wiring all landed in the implementation step. Q7 numbers still pend a hardware run (see `RemainingWork.md`).

## How to exercise it

```
cmake -B build -DMIOPEN_ENABLE_HIPDNN_WRAPPER=ON
ctest --test-dir build -L investigation
```
