# Phase 1 wrapper tooling — RFC 0001

Validation harnesses and bootstrap scripts for the public/private split
implemented per `docs/rfcs/0001_HipdnnForwardingWrapper_phase1_investigation.md`.

These tools are **only** wired up when MIOpen is configured with
`-DMIOPEN_ENABLE_HIPDNN_WRAPPER=ON`. With the flag off (the default), this
directory is inert — the top-level CMakeLists.txt skips
`cmake/InvestigationHipdnnWrapper.cmake` entirely, preserving §1's
byte-equivalence constraint.

The two source-of-truth artifacts live in `src/private/`, not here:

| File | Investigation Q | Purpose |
|---|---|---|
| `src/private/miopen_private_rename.h` | Q1 | Hand-maintained `#define miopenFoo miopenFoo_impl` block, applied to MIOpen_private sources via the compiler's `-include` flag. Intentionally NOT installed (Q5). |
| `src/private/wrapper.cpp` | Q4 | Hand-maintained Phase 1 pass-through wrapper — one `extern "C"` stub per public entry point, each forwarding to its `_impl` symbol. Compiled into the wrapper `libMIOpen.so`. |

Validation scripts and CTest helpers in this directory:

| File | Investigation Q | Purpose |
|---|---|---|
| `check_stub_count.cmake` | Q1, Q4 | CTest that asserts `miopen.h`, the rename header, and `wrapper.cpp` all agree on the count of public entry points. The drift check that survives both generators' retirement. |
| `check_public_api.sh` | §1, item 9 | Asserts the *built* libMIOpen.so exports exactly the set of `MIOPEN_EXPORT` names declared in `miopen.h` (no missing symbols, no escaped private symbols). Wired as `public_api_symbol_check` in `src/CMakeLists.txt` — runs in BOTH flag states whenever BUILD_TESTING is on. |
| `consumer_smoke.c`, `check_consumer_smoke.sh` | Q5 | Tiny consumer that links an *installed* prefix; asserts no `_impl` symbol references and that the rename header is absent from the install include tree. |
| `find_package_smoke/`, `check_find_package_smoke.sh` | Q6 | External CMake project that uses `find_package(miopen)` to consume both `MIOpen` and (when present) `MIOpen_private`; verifies wrapper DT_NEEDED references `libMIOpen_private.so`. |
| `symbol_diff.sh` | Q2, Q6 | `dump`/`diff` modes for the public symbol set, SONAME, and DT_NEEDED of `libMIOpen.so`. Used to validate that flag-on remains a superset of flag-off. |
| `wrapper_overhead.sh`, `microbench_settensor.cpp` | Q7 | Three-workload overhead harness (microbench + short conv + ResNet50-style conv) emitting a CSV of deltas. Requires GPU hardware. |

## Running

Configure with the wrapper flag on:

```bash
cmake -B build -S . -DMIOPEN_ENABLE_HIPDNN_WRAPPER=ON
cmake --build build -j
```

Validations through CTest:

```bash
ctest --test-dir build -L investigation        # Q1/Q4 drift, opt-in Q5/Q6
ctest --test-dir build -R public_api_symbol    # §1/item-9 equivalence (any flag state)
```

For the install-tree validations (Q5, Q6):

```bash
cmake --install build --prefix /tmp/miopen-prefix
cmake -B build -S . -DMIOPEN_ENABLE_HIPDNN_WRAPPER=ON \
                    -DMIOPEN_INVESTIGATION_INSTALL_PREFIX=/tmp/miopen-prefix
ctest --test-dir build -L investigation
```

For the Q2 superset diff:

```bash
# Dump a flag-off baseline once.
cmake -B build-off -S . && cmake --build build-off -j -t MIOpen
tools/wrapper/symbol_diff.sh dump build-off/lib/libMIOpen.so --out /tmp/baseline

# Configure flag-on with the baseline path; the test compares both at run time.
cmake -B build-on -S . -DMIOPEN_ENABLE_HIPDNN_WRAPPER=ON \
                       -DMIOPEN_WRAPPER_FLAGOFF_BASELINE=/tmp/baseline
cmake --build build-on -j
ctest --test-dir build-on -R investigation_q2
```

## Adding a new public entry point

When a new `MIOPEN_EXPORT` function is added to `include/miopen/miopen.h`,
update both source-of-truth files in lock-step:

1. Add a `#define miopenNewFn miopenNewFn_impl` line to
   `src/private/miopen_private_rename.h`.
2. Add an `extern "C"` pass-through stub for `miopenNewFn` to
   `src/private/wrapper.cpp` (forward-declare the `_impl` symbol, then a
   one-line stub that forwards arguments).

The `investigation_q4_stub_count` CTest fails loudly if any of the three
counts (header, rename, wrapper) disagree.

## Re-bootstrapping

If `miopen.h` ever undergoes a large refactor that warrants regenerating
both files from scratch, the original generators
(`gen_rename_header.py`, `gen_wrapper_source.py`) are retrievable from
git history at commit `a827879e67`. Neither is part of the build.
