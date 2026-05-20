# ALMIOPEN-1900 — Remaining Work

Gap analysis between [ALMIOPEN-1900 — Investigate public/private split of MIOpen library](https://amd-hub.atlassian.net/browse/ALMIOPEN-1900) and the work landed on `users/nhanna/miopen-hipdnn-shim-investigation-1`.

## Acceptance criteria → status (post-implementation)

| AC | Status |
|---|---|
| 1. Feature Flag everything | ✅ `MIOPEN_ENABLE_HIPDNN_WRAPPER` gates the entire split: variable + wrapper target in `src/CMakeLists.txt`, drift/Q5/Q6 CTests, and the miopen-provider link-target switch. Flag-off configure is byte-equivalent by construction. |
| 2. Private MIOpen lib exposing `_impl` | ✅ `MIOpen_private` target compiles existing MIOpen sources with `MIOPEN_BUILDING_PRIVATE` defined and `-include src/private/miopen_private_rename.h` — the 263 public symbols become `_impl` in `libMIOpen_private.so`. |
| 3. Public MIOpen lib forwarding to private | ✅ `MIOpen` target rebuilds as a thin wrapper from `src/private/wrapper.cpp` (263 `extern "C"` stubs), PUBLIC-links `MIOpen_private`. Same SOVERSION as today; `find_package(miopen)` consumers don't relink. |
| 4. Convert hipDNN's miopen-provider to private | ✅ `dnn-providers/miopen-provider/CMakeLists.txt` detects `TARGET MIOpen_private` and prefers it for all three link sites (`miopen_plugin_impl`, `miopen_plugin_private`, `miopen_plugin`). Falls back to `MIOpen` for legacy installs. |
| 5. MIOpenDriver + logging demo with flag on | ⏳ Wired but not run — needs GPU hardware. Run a representative driver workload with `MIOPEN_ENABLE_LOGGING=1`, capture trace showing wrapper stub then `_impl` entry, attach to the Jira ticket. |
| 6. Flag off still builds | ✅ Enforced by the `public_api_symbol_check` CTest (in `src/CMakeLists.txt`, runs in BOTH flag states whenever `BUILD_TESTING` is on): asserts the built library exports exactly the set of `MIOPEN_EXPORT` names declared in `miopen.h`. Catches accidental ABI changes from either direction. |

## Per-item status

### Build system — Private library (AC #2)

1. ✅ `MIOpen_private` CMake target added in `src/CMakeLists.txt` with all existing MIOpen sources, `MIOPEN_BUILDING_PRIVATE` define, `-include` rename header, and `MIOpen_EXPORTS` define so `MIOPEN_EXPORT` resolves to default visibility on the renamed `_impl` symbols.
2. ✅ `miopen_private_rename.h` and `wrapper.cpp` moved to `src/private/`. `src/private/` is not in any install include list.

### Build system — Public wrapper (AC #3)

3. ✅ `MIOpen` target compiles only `src/private/wrapper.cpp`, PUBLIC-links `MIOpen_private`, sets `rocm_set_soversion(MIOpen ${MIOpen_SOVERSION})` (SOVERSION matches today), reuses the same hidden-visibility / `--exclude-libs,ALL` settings as today's `MIOpen`. Both targets flow through the same `rocm_install_targets`/`rocm_export_targets` (`_MIOPEN_INSTALL_TARGETS` list).
4. ✅ `MIOpen_with_plugins` continues to link `MIOpen` — flag-off: same target as today; flag-on: the wrapper, so tests/driver/fin/speedtests exercise the real dispatch path. New `MIOpen_private_with_plugins` INTERFACE alias is the bypass hook for in-tree consumers (Phase 4 short-circuit, benchmarks).

### Validation (closes RFC §6 outstanding items + AC #6)

5. ✅ Q2 superset diff CTest registered: `investigation_q2_dump_candidate` + `investigation_q2_symbol_superset` (in `cmake/InvestigationHipdnnWrapper.cmake`), gated on `-DMIOPEN_WRAPPER_FLAGOFF_BASELINE=<prefix>` from a flag-off `symbol_diff.sh dump` baseline. Compares public symbols (must be superset) and SONAME (must match).
6. ✅ Q5 consumer-smoke CTest already wired (`investigation_q5_consumer_smoke`, opt-in via `MIOPEN_INVESTIGATION_INSTALL_PREFIX`). Run after `cmake --install` to confirm no `_impl` leakage and rename header is absent from the installed include tree.
7. ✅ Q6 find_package smoke added: `investigation_q6_find_package_smoke` runs the new `tools/wrapper/find_package_smoke/` external project against the install prefix, verifies both `MIOpen` and `MIOpen_private` are consumable via `find_package(miopen)`, and checks DT_NEEDED on the wrapper.
8. ⏳ Q7 overhead — harness in tree (`tools/wrapper/wrapper_overhead.sh`); needs a hardware run. Numbers go into RFC §6 Q7 section once available; merge gated on the §3 Q7 acceptance criteria.
9. ✅ `public_api_symbol_check` CTest added (in `src/CMakeLists.txt`, runs in BOTH flag states whenever `BUILD_TESTING` is on): the built library's exported `miopen*` symbol set must equal the set of `MIOPEN_EXPORT` declarations in `miopen.h`. No baseline file to maintain — `miopen.h` is the source of truth, and the existing drift CTest already keeps `miopen.h` in sync with the rename header and wrapper.

### hipDNN provider conversion (AC #4)

10. ✅ `dnn-providers/miopen-provider/CMakeLists.txt` switches its three link sites (`miopen_plugin_impl`, `miopen_plugin_private`, `miopen_plugin`) to `MIOpen_private` when that target is present in the consumed `find_package(miopen)`. Falls back to `MIOpen` for legacy installs. A `message(STATUS ...)` records which path was chosen at configure time.
11. ⏳ Build + run of the miopen-provider tests against the privately-linked variant — needs GPU hardware and a flag-on MIOpen install for the provider build to consume.

### Demonstration (AC #5)

12. ⏳ MIOpenDriver + logging trace — needs GPU hardware. Procedure: build flag-on, install, run a representative shape (e.g. `MIOpenDriver conv -n 32 -c 64 -H 56 -W 56 -k 64 …`) with `MIOPEN_ENABLE_LOGGING=1`, attach the captured trace to the Jira ticket. The wrapper currently produces no log line of its own (Phase 1 is intentionally silent on the wrapper boundary); add a one-line `MIOPEN_LOG_FUNCTION_TRACE` to a representative stub if a visible "wrapper hit" entry is desired for the demo.

### Closeout

13. ✅ RFC §6 sections updated with implementation status (Q1/Q3/Q4 PASS, Q2/Q5/Q6 wired with empirical exit pending the first install run, Q7 still hardware-blocked).
14. ✅ §5 Decision: Go on Option A (RFC investigation doc, end of §5).
15. ✅ `Phase1-Steps1+2-Summary.md` updated to reflect that the wrapper is now built (no longer "investigation only"). `Phase1-Steps1+1-ImplementionSummary.md` left as a historical record of what shipped in the investigation phase.

## What still requires hardware/install

- AC #5 (item 12): MIOpenDriver+logging demo on a real GPU.
- Item 8 (Q7): `wrapper_overhead.sh` run on a representative GPU.
- Item 11: miopen-provider tests against a flag-on install.
- The empirical Q2/Q5/Q6 CTests are wired but only run when their opt-in
  `-D…` variables point at real artifacts (a flag-off baseline dump for Q2,
  an installed prefix for Q5 and Q6).

Everything else needed to close ALMIOPEN-1900 is implemented and
uncommitted in the working tree.
