# RFC 0001 — Phase 1 Steps 1 & 2 Summary

> **Superseded.** The investigation framing in the original draft of this
> document ("the wrapper is not yet built") no longer applies — the
> public/private split landed in the same change set that produced the
> `RemainingWork.md` follow-up. See:
>
> - `docs/rfcs/0001_HipdnnForwardingWrapper_phase1_investigation.md` §5
>   (Decision: Go on Option A) and §6 (per-Q empirical status)
> - `tools/wrapper/RemainingWork.md` (per-AC status table)
> - `tools/wrapper/README.md` (which validations run when, and how)
>
> The questions answered below are kept verbatim as historical context for
> what the original investigation phase looked like, with one corrected
> line per section flagging what changed.

## Does this cover all API endpoints?

**Surface: yes. Compilation/linking: now also yes (was: no).**

Both Phase 1 source artifacts live in tree as hand-maintained files (RFC §6 Q1, Q4 — generators retired):

- `src/private/miopen_private_rename.h` — `#define miopenFoo miopenFoo_impl` for all 263 entry points.
- `src/private/wrapper.cpp` — one `extern "C"` pass-through stub for all 263 entry points.

The CTest `investigation_q4_stub_count` check enforces three-way count parity: `miopen.h`, the rename header, and `wrapper.cpp` must all agree on the public entry-point count. **Both files are now compiled into the build** when `MIOPEN_ENABLE_HIPDNN_WRAPPER=ON`: the rename header is applied to MIOpen_private sources via `-include`, and `wrapper.cpp` is the source of the wrapper `MIOpen` target.

## When the flag is off, does this compile MIOpen exactly as it was before?

**Yes.** The new code is gated by `if(MIOPEN_ENABLE_HIPDNN_WRAPPER)` in `CMakeLists.txt:124-126` and by the `_MIOPEN_IMPL_TARGET` variable in `src/CMakeLists.txt:870-876`. With the flag off:

- `_MIOPEN_IMPL_TARGET` evaluates to `MIOpen` and `_MIOPEN_INSTALL_TARGETS` evaluates to just `MIOpen` — every existing `target_*` call binds to the same `MIOpen` target it always did
- `InvestigationHipdnnWrapper.cmake` isn't read
- The wrapper `add_library(MIOpen …)` block is skipped
- No new targets, no new sources, no link-line changes, no new install artifacts
- The `MIOpen` target is byte-equivalent to the pre-split build

## When the flag is on, does this create both a libMIOpen.so and libMIOpen_private.so?

**Yes** (was: no). With the flag on:

- `MIOpen_private` is added with all the existing MIOpen sources, the rename header applied via `target_compile_options(... PRIVATE "SHELL:-include …/src/private/miopen_private_rename.h")`, and `MIOPEN_BUILDING_PRIVATE` defined.
- `MIOpen` is added separately, compiles `src/private/wrapper.cpp`, PUBLIC-links `MIOpen_private`, and gets the same SOVERSION as today (so consumers don't need to relink).
- Both libraries flow through the same `rocm_install_targets` and `rocm_export_targets` calls; `find_package(miopen)` consumers see both.

## How can each be tested?

### Flag-off build is the same as before

```bash
git checkout users/nhanna/miopen-hipdnn-shim-pre-investigation-1
cmake -B build-baseline -S . && cmake --build build-baseline -j -t MIOpen
sha256sum build-baseline/lib/libMIOpen.so > /tmp/baseline.sha

git checkout users/nhanna/miopen-hipdnn-shim-investigation-1
cmake -B build-flagoff -S . && cmake --build build-flagoff -j -t MIOpen
sha256sum build-flagoff/lib/libMIOpen.so > /tmp/flagoff.sha

diff /tmp/baseline.sha /tmp/flagoff.sha   # expected: identical
```

If timestamps or build-IDs make `sha` differ, the symbol-level fallback is `tools/wrapper/symbol_diff.sh dump build-baseline/lib/libMIOpen.so --out /tmp/base` then the same on the flag-off build, then `symbol_diff.sh diff --baseline /tmp/base --candidate /tmp/flagoff` — clean superset with matching SONAME.

A stronger header-derived check ships with the build: `ctest -R public_api_symbol_check` asserts that the built library exports exactly the set of `MIOPEN_EXPORT` names declared in `miopen.h`. This runs in BOTH flag states whenever `BUILD_TESTING` is on.

### Flag-on build has two .so's

Now testable.

```bash
cmake -B build-flagon -S . -DMIOPEN_ENABLE_HIPDNN_WRAPPER=ON
cmake --build build-flagon -j
ls build-flagon/lib/libMIOpen.so build-flagon/lib/libMIOpen_private.so
readelf -d build-flagon/lib/libMIOpen.so | grep NEEDED   # mentions libMIOpen_private.so.1
```

The investigation CTests:

```bash
ctest --test-dir build-flagon -L investigation
```

run the drift parity check (`investigation_q4_stub_count`) and, if the user passes `-DMIOPEN_INVESTIGATION_INSTALL_PREFIX=…`, the consumer-smoke (Q5) and find_package smoke (Q6) checks against the install tree.

### Flag-on properly feeds through the wrapper

Now testable.

1. `nm -D --defined-only build-flagon/lib/libMIOpen_private.so | grep miopenConvolutionForward` — must show `miopenConvolutionForward_impl`, NOT `miopenConvolutionForward`.
2. `nm -D --defined-only build-flagon/lib/libMIOpen.so | grep miopenConvolutionForward` — must show `miopenConvolutionForward` (the wrapper) and NOT `miopenConvolutionForward_impl`.
3. Run a tiny consumer with `LD_DEBUG=symbols` or set a breakpoint in the wrapper stub and confirm a real call into MIOpen passes through it. (This is part of AC #5 — see `RemainingWork.md` item 12; needs hardware.)

The `tools/wrapper/check_consumer_smoke.sh` harness automates parts (1) and (2) once you have an installed prefix.

### Flag-off does not feed through any wrapper

Trivially true: `nm -D build-flagoff/lib/libMIOpen.so | grep _impl` returns nothing — flag-off has no `_impl` symbols at all because the rename header is never applied.

## Status

| Question | Status |
|---|---|
| All endpoints covered (in the hand-maintained wrapper source + rename header)? | Yes — 263 entry points, three-way drift-checked against miopen.h |
| All endpoints covered (in shipped binaries)? | Yes — wrapper.cpp compiled into MIOpen, rename applied to MIOpen_private |
| Flag off = byte-equivalent? | Yes by construction; enforced at symbol level by `public_api_symbol_check` |
| Flag on builds two `.so`s? | Yes |
| Wrapper actually intercepts calls? | Yes by construction; runtime demonstration is RemainingWork item 12 (hardware) |
