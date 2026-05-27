# MIOpen hipDNN Provider Plugin — Linkage & Runtime Verification

Date: 2026-05-27
Branch: `users/nhanna/miopen-hipdnn-shim-investigation-1`
Hardware: AMD Instinct MI300X (gfx942), ROCm 7.13, MIOpen 3.5.2
Builds under test: `build-flagoff` (wrapper OFF), `build-flagon` (wrapper ON)
Synthetic harness: `perf-results/plugin-test/`

## 1. Purpose

The hipDNN MIOpen provider plugin selects its MIOpen link target at CMake time:

```cmake
# dnn-providers/miopen-provider/CMakeLists.txt:149-163
find_package(miopen CONFIG REQUIRED)

# RFC 0001 Phase 4 short-circuit: when the MIOpen install was built with
# MIOPEN_ENABLE_HIPDNN_WRAPPER=ON, it exports a MIOpen_private target whose
# libMIOpen_private.so contains the renamed _impl symbols. Linking the plugin
# against MIOpen_private bypasses the wrapper, so this provider's calls don't
# round-trip back through hipDNN. When the MIOpen install is the legacy
# single-library build, MIOpen_private won't exist and we fall back to MIOpen.
if(TARGET MIOpen_private)
    set(_MIOPEN_PROVIDER_LINK_TARGET MIOpen_private)
    message(STATUS "miopen-provider: linking MIOpen_private (Phase 4 short-circuit)")
else()
    set(_MIOPEN_PROVIDER_LINK_TARGET MIOpen)
    message(STATUS "miopen-provider: linking MIOpen (legacy single-library install)")
endif()
```

The intent: when MIOpen is installed in **wrapper mode** (flagon), the provider plugin links directly to `libMIOpen_private.so.1` so its calls don't take the public-API wrapper hop and don't risk re-entering hipDNN. When MIOpen is installed in **legacy mode** (flagoff), the provider links to `libMIOpen.so.1` like any normal MIOpen consumer.

This document verifies, via structural ELF inspection, symbol resolution, dynamic-linker tracing, and a negative test, that this wiring **selects the correct library at link time** and that the loader **routes the runtime calls as designed** — and documents one gap that surfaced during the runtime test.

## 2. Test artifacts

Because no pre-built provider plugin in the working tree is linked against `libMIOpen_private.so.1` (the canonical one in `/data/jlichtne/TheRock/therock-build/lib/hipdnn_plugins/engines/libmiopen_plugin.so` was built against an older MIOpen install that lacked the `MIOpen_private` target — see §3), a synthetic plugin that mirrors the real plugin's link wiring was built:

- `perf-results/plugin-test/synthetic_plugin.c` — calls three public MIOpen entry points (`miopenGetVersion`, `miopenCreate`, `miopenDestroy`). Compiled with `-fvisibility=hidden`, exports only `plugin_run` via a linker version script — matches the real plugin's symbol-export discipline.
- `perf-results/plugin-test/build.sh` — produces two variants from the same source, against `build-flagon/lib`:
  - `libsynth_plugin_public.so`  → linked `-lMIOpen` (legacy / fallback target)
  - `libsynth_plugin_private.so` → linked `-lMIOpen_private` (Phase-4 target)
- `perf-results/plugin-test/host.c` — a generic `dlopen(RTLD_NOW | RTLD_LOCAL) + dlsym("plugin_run")` host, mimicking how hipDNN loads engine plugins at runtime. The host does not link MIOpen itself; all MIOpen exposure comes from the loaded plugin's `DT_NEEDED`.

## 3. Structural verification (`readelf -d`)

`DT_NEEDED` / `DT_RUNPATH` of each artifact:

| Artifact | NEEDED MIOpen entry | RUNPATH |
| --- | --- | --- |
| `libsynth_plugin_public.so`  (new) | `libMIOpen.so.1` | `build-flagon/lib` |
| `libsynth_plugin_private.so` (new) | `libMIOpen_private.so.1` | `build-flagon/lib` |
| TheRock pre-built `libmiopen_plugin.so` | `libMIOpen.so.1` | `$ORIGIN/...` |
| `libMIOpen.so.1` (flagon build) | `libMIOpen_private.so.1`, `libamd_comgr.so.3`, `librocblas.so.5`, `libamdhip64.so.7`, … | `build-flagon/lib:/opt/rocm/lib` |

**Findings:**

- The two synthetic variants differ only in their `NEEDED` MIOpen entry — exactly the difference the Phase-4 short-circuit produces. This is the structural fingerprint we'd see on any real provider plugin built from the same CMakeLists.
- `libMIOpen.so.1` itself has a hard `NEEDED` on `libMIOpen_private.so.1`. This is by design — the wrapper stubs in `libMIOpen.so.1` call `<name>_impl` symbols defined in the private library — but it means that in a flagon install **every consumer of `libMIOpen.so.1` transitively loads `libMIOpen_private.so.1`** whether they want to or not.
- The TheRock pre-built `libmiopen_plugin.so` is `NEEDED libMIOpen.so.1` (no MIOpen_private), confirming it was built against a non-flagon MIOpen install. It is not useful for validating Phase 4 and cannot be substituted for the synthetic test.

## 4. Symbol cross-check

How many `miopen*` symbols each library exports, and what form (`nm -D --defined-only | grep " T miopen"`):

| Library | Plain `miopen*` exports | `miopen*_impl` exports | Total |
| --- | ---: | ---: | ---: |
| `libMIOpen.so.1` (wrapper) | 263 | 0 | 263 |
| `libMIOpen_private.so.1` | 4 † | 263 | 267 |

† The 4 non-renamed survivors in `libMIOpen_private.so.1` are: `miopenConvolutionABBackwardWeightsGetWorkSpaceSize`, `miopenHiddenGetConvolutionFindMode`, `miopenHiddenSetConvolutionFindMode`, `miopen_sqlite3_memvfs_init`. They are not in the rename header and are unrelated to the wrapper hop.

Undefined `miopen*` symbols in each synthetic plugin (`nm -D --undefined-only`):

| Plugin | Undefined `miopen*` symbols |
| --- | --- |
| `libsynth_plugin_public.so` | `miopenCreate`, `miopenDestroy`, `miopenGetVersion` |
| `libsynth_plugin_private.so` | `miopenCreate`, `miopenDestroy`, `miopenGetVersion` |

Note the identical undefined-symbol set: the plugin source uses the unrenamed public names (`#include <miopen/miopen.h>`) regardless of which library it is linked against. The linker tolerates this for the PRIVATE variant because shared-object linking is lazy — undefined symbols in a `.so` are not errors at link time; they are recorded as a request the dynamic loader will fulfill later.

**Where each undefined symbol resolves:**

| Plugin | `miopenCreate` resolves in | `miopenGetVersion` resolves in | `miopenDestroy` resolves in |
| --- | --- | --- | --- |
| PUBLIC  | `libMIOpen.so.1` ✓ | `libMIOpen.so.1` ✓ | `libMIOpen.so.1` ✓ |
| PRIVATE | `libMIOpen_private.so.1` exports only `miopenCreate_impl` ✗ | only `miopenGetVersion_impl` ✗ | only `miopenDestroy_impl` ✗ |

The PRIVATE variant has no way to bind its undefined symbols at load time — the names it asks for are not the names the private library exports. This is the gap explored in §5 and §7.

## 5. Runtime verification — `LD_DEBUG=bindings`

Captures: `perf-results/plugin-test/ld_public.<pid>`, `ld_private.<pid>`.

### PUBLIC variant — `./host ./libsynth_plugin_public.so`

```
plugin_run returned 10        # = 3 + 5 + 2 (MIOpen 3.5.2)
```

Bindings of interest (counts from the LD_DEBUG output):

| Source library | Target library | Bindings | Notes |
| --- | --- | ---: | --- |
| `libsynth_plugin_public.so` | `libMIOpen.so.1` | **3** | exactly the three API calls in `plugin_run` |
| `libMIOpen.so.1` | `libMIOpen_private.so.1` | **263** | every wrapper stub binds its matching `<name>_impl` at load time (RTLD_NOW) |

The three explicit plugin-side bindings:

```
binding file ./libsynth_plugin_public.so [0]
    to .../libMIOpen.so.1 [0]: normal symbol `miopenGetVersion'
binding file ./libsynth_plugin_public.so [0]
    to .../libMIOpen.so.1 [0]: normal symbol `miopenCreate'
binding file ./libsynth_plugin_public.so [0]
    to .../libMIOpen.so.1 [0]: normal symbol `miopenDestroy'
```

This is the canonical wrapper-hop trace: plugin code calls `miopenCreate`, which is a forwarding stub in `libMIOpen.so.1`, which calls `miopenCreate_impl` in `libMIOpen_private.so.1`. The 263 wrapper→private bindings are the entire C-ABI surface — they get resolved up front because the host uses `RTLD_NOW`, but only three of them are actually called during this run.

### PRIVATE variant — `./host ./libsynth_plugin_private.so`

```
dlopen failed: ./libsynth_plugin_private.so: undefined symbol: miopenGetVersion
```

Bindings of interest:

| Source library | Target library | Bindings | Notes |
| --- | --- | ---: | --- |
| `libsynth_plugin_private.so` | `libgcc_s.so.1`, `libc.so.6` | 6 | only C runtime / unwind glue |
| `libsynth_plugin_private.so` | any `libMIOpen*.so.1` | **0** | dlopen aborted before any MIOpen binding could be attempted |

(The trace does record ~2.8k bindings *internal to* `libMIOpen_private.so.1` and from ROCm helpers into it — those are part of loading the library itself and would happen before the plugin's own relocations are processed.)

This is the runtime expression of the §4 gap: the linker happily built `libsynth_plugin_private.so` with `U miopenGetVersion`, but the only thing `libMIOpen_private.so.1` exports under that name discipline is `miopenGetVersion_impl`, so dlopen fails at the plugin's relocation step. The Phase-4 short-circuit, *as currently written in the provider's CMakeLists*, produces a `.so` that **links but does not load**.

## 6. Negative test — hide `libMIOpen_private.so.1`

To confirm that the loader is actually pulling `libMIOpen_private.so.1` (not picking up a stale copy from `/opt/rocm/lib` or similar):

```
$ mv build-flagon/lib/libMIOpen_private.so.1{,.hidden_for_test}

$ ./host ./libsynth_plugin_public.so
dlopen failed: libMIOpen_private.so.1: cannot open shared object file: No such file or directory

$ ./host ./libsynth_plugin_private.so
dlopen failed: libMIOpen_private.so.1: cannot open shared object file: No such file or directory

$ mv build-flagon/lib/libMIOpen_private.so.1{.hidden_for_test,}

$ ./host ./libsynth_plugin_public.so
plugin_run returned 10
```

Both variants fail with the same "cannot open shared object" error, then PUBLIC recovers immediately after the file is restored. This confirms:

- The PUBLIC plugin's runtime dependency on `libMIOpen_private.so.1` is real and transitive (via `libMIOpen.so.1`'s `DT_NEEDED`) — there is no possible code path through the wrapper variant that avoids loading the private library.
- The PRIVATE plugin's runtime dependency on `libMIOpen_private.so.1` is real and direct.
- Both plugins resolve `libMIOpen_private.so.1` from `build-flagon/lib` via the `DT_RUNPATH` baked into each plugin's ELF (and into `libMIOpen.so.1`'s ELF for the PUBLIC case) — nothing else on the system is being silently substituted.

## 7. Gap finding — Phase 4 short-circuit is incomplete

§4–§6 together establish that the provider's CMake snippet **chooses the link target correctly** (the PRIVATE variant does link `libMIOpen_private.so.1` rather than `libMIOpen.so.1`), but the resulting plugin **cannot be loaded** because the symbol names visible to the plugin's compilation unit don't match the symbol names exported by `libMIOpen_private.so.1`.

The mechanism the wrapper build uses to make MIOpen's own translation units call `_impl` names is a build-time include:

```cmake
# projects/miopen/src/CMakeLists.txt — MIOpen Private build options
target_compile_options(MIOpen_private PRIVATE
    -DMIOPEN_BUILDING_PRIVATE
    -include ${CMAKE_CURRENT_SOURCE_DIR}/private/miopen_private_rename.h)
```

The header (`projects/miopen/src/private/miopen_private_rename.h`) `#define`s every `miopenFoo` to `miopenFoo_impl` and is gated by `MIOPEN_BUILDING_PRIVATE` — and the comment at the top reads:

> This header is intentionally NOT installed (RFC 0001 §4.6, Q5).

So the rename header is not in the install tree; the provider plugin, being a downstream consumer, has no way to apply it as written today. The PRIVATE link target therefore only delivers value if **something on the consumer side** also rewrites the names. Options:

1. **Install the rename header** under `private/` in the MIOpen install tree, and have `find_package(miopen)` set up the `MIOpen_private` imported target's `INTERFACE_INCLUDE_DIRECTORIES` / `INTERFACE_COMPILE_OPTIONS` to apply `-DMIOPEN_BUILDING_PRIVATE -include miopen_private_rename.h` automatically. The provider plugin would then transparently get the renamed names whenever it links against `MIOpen_private`. This is the lowest-friction option for plugin authors.
2. **Ship a parallel public header** that declares the API in `_impl` form, and have the provider's source code include that header when targeting `MIOpen_private`. This avoids the macro-rename trick but requires the provider to be conditionally compiled.
3. **Keep the rename header private, change the private library's exports** so it also publishes the unrenamed names as additional symbols (aliases). This eliminates the need for any consumer-side rename, at the cost of `libMIOpen_private.so.1` having two symbols per API. It re-creates the risk Phase 4 was trying to remove (the private library being callable through the public name path).

Option (1) most closely matches the existing build's mechanism and was the assumption the provider's CMake snippet was written against.

The runtime evidence collected here can be used as the regression bar for any chosen fix: once it's in, `./host ./libsynth_plugin_private.so` must succeed, and `LD_DEBUG=bindings` must show plugin→`libMIOpen_private.so.1` bindings on `miopenCreate_impl` / `miopenDestroy_impl` / `miopenGetVersion_impl` with **zero** bindings into `libMIOpen.so.1`.

## 8. Summary

| Verification | PUBLIC variant | PRIVATE variant |
| --- | --- | --- |
| `DT_NEEDED` matches CMake intent | ✓ `libMIOpen.so.1` | ✓ `libMIOpen_private.so.1` |
| `DT_RUNPATH` finds the lib | ✓ `build-flagon/lib` | ✓ `build-flagon/lib` |
| Symbols exist in target lib | ✓ all 3 found | ✗ only `_impl` forms exported |
| `dlopen` succeeds | ✓ | ✗ undefined symbol: `miopenGetVersion` |
| `plugin_run` executes | ✓ returns 10 (MIOpen 3.5.2) | ✗ never reached |
| Wrapper hop observed (LD_DEBUG) | 3 plugin→wrapper + 263 wrapper→private | n/a |
| Negative test (hide private lib) | fails with cannot-open (transitive) | fails with cannot-open (direct) |

- The **wrapper path** (PUBLIC plugin against flagon MIOpen) is the legacy/fallback case and works end-to-end. The wrapper-hop overhead is the ≈1 ns per call documented in `PerformanceComparison.md`.
- The **short-circuit path** (PRIVATE plugin) selects the correct library at link time but its plugin source can't call any API in that library because the consumer has no access to the rename mechanism. This is a **gap in Phase 4's wiring**, not a bug in the test setup — every API call in the synthetic plugin is also present in the real plugin and would fail the same way.
- The negative test confirms `libMIOpen_private.so.1` is always loaded in a flagon install, even when only the public library is linked. Wrapper mode and short-circuit mode do not differ in *which* files are mapped, only in which library's exports the plugin's relocations bind to.

## 9. Files

- `perf-results/plugin-test/synthetic_plugin.c`, `host.c`, `build.sh` — sources
- `perf-results/plugin-test/libsynth_plugin_{public,private}.so`, `host` — built artifacts
- `perf-results/plugin-test/ld_public.<pid>`, `ld_private.<pid>` — raw LD_DEBUG bindings traces
