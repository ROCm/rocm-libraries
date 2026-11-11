# Building the rocKE engine

This is the canonical build and artifact-hygiene reference for the **rocKE C++
engine**: the tree rooted at this `rocke/platform/` directory whose sources live under
`cpp/` and which compiles to a static archive `librocke_core.a`. The engine lowers
the rocke IR to LLVM IR and is consumed by a hipDNN provider plugin (which links
the archive and loads it at runtime).

All paths below are relative to the `rocke/platform/` root (written `<rocke/platform>`), so they
are correct wherever this tree lives.

There is **one supported engine build path**. Use it. The historical failures it
prevents are all variations of "a stale or mis-flagged build product was used
silently".

## TL;DR: build the engine + prove it

```bash
# build the engine fresh AND run the byte-identity gate (the definition-of-done):
python <rocke/platform>/tools/check_byte_identity.py

# the portable-IR replay contract, carried through to object code (what CI runs):
python <rocke/platform>/tools/run_portable_ir_gates.py
```

The first proves the C++ engine emits the same LLVM-IR as the Python engine. The
second proves the *replay* paths — the C++ importer and the recipe VM, including
parametric recipes replayed at values the roller never recorded — reproduce the
Python lowerer byte for byte, at `.ll` and then at HSACO. Both are device-free;
the second needs comgr.

`check_byte_identity.py` builds the engine fresh as a Release static archive
(`librocke_core.a`) and then proves its LLVM-IR emission is byte-identical to the
Python engine across every kernel family. A green run means the dual-engine
contract holds. It writes everything under a build root (default
`$TMPDIR/rocke_verify`); nothing is written back into the source tree.

Common options:

| Option | Meaning |
|---|---|
| `--build-root DIR` | Where the engine is built (use local disk; never NFS). |
| `--only SUBSTR` | Restrict the gate to families containing SUBSTR (comma-separated). |
| `--ir` | Also run the IR-canonical diff (diagnostic). |
| `--ref-pyroot DIR` / `--ref-shim DIR` | Compare against another tree's Python engine. |

> Build on a **local filesystem**. NFS makes the compiler and `comgr`
> pathologically slow.

To run the full validation suite (relative-path guard + byte-identity gate +
pytest + ctest when a build dir exists), use the test runner:

```bash
python <rocke/platform>/tests/run_all.py                  # guard + gate + pytest (+ ctest)
python <rocke/platform>/tests/run_all.py --only gemm      # restrict the gate to one family
```

Multi-arch coverage (gfx950 baseline plus gfx942/gfx1151/gfx1201) is intrinsic to
the gate: the parity emitters enumerate the architectures per `(spec, arch)`
config, so the standard run above already exercises every supported arch.

## Building the engine archive by hand

```bash
cmake -S <rocke/platform> -B <build> -DCMAKE_BUILD_TYPE=Release
cmake --build <build> --target rocke_core -j$(nproc)
# -> <build>/librocke_core.a   (the archive a provider links)
```

The top-level `<rocke/platform>/CMakeLists.txt` globs `cpp/**/*.cpp` (excluding
`cpp/bindings/`) into `rocke_core`, with the public ABI headers at `cpp/include`.

Configuring needs a Python 3 interpreter. The engine's copy of the intrinsic
arch-domain table is generated at build time from the JSON columns under
`python/rocke/core/arch/data/`, which the Python engine reads directly, by
`tools/gen_arch_domain_cpp.py` (standard library only). The output goes to
`<build>/generated/` and is never committed, so regenerating a column with
`tools/gen_arch_domain.py` reaches both engines with no other step.

Optional sanitizer build for diagnostics (not for shipping): `-DROCKE_SANITIZE=ON`.

### The shared engine (`librocke.so`)

The portable-IR replay tooling reaches the C++ importer, recipe VM and lowerer
through ctypes, so it needs a shared library rather than the archive:

```bash
cmake -S <rocke/platform> -B <build> -DCMAKE_BUILD_TYPE=Release \
      -DROCKE_BUILD_SHARED_ENGINE=ON
cmake --build <build> --target rocke_shared -j$(nproc)
# -> <build>/librocke.so   (point ROCKE_ONLINE_LIB at this)
```

`--whole-archive` is applied for you. It is not optional: nothing in the `.so`
references the engine's entry points, so without it the linker discards every
object as unused and emits a valid library with no symbols in it.

`online.build_lib()` produces an equivalent library on demand, which is more
convenient interactively. Prefer the target when the resulting library needs to
be identifiable — CI points `ROCKE_ONLINE_LIB` at it, and a named library that
does not exist is now an error rather than a cue to go and find another one.

> **Toolchain/runtime flags.** AUTO queries the loaded COMGR compiler’s LLVM
> version to select the emitted IR flavor. An explicit `ROCKE_LLVM_FLAVOR`
> supports offline emission; it does not select COMGR or bypass compiler checks.
> Full flag list: [`dsl_docs/reference/env_flags.md`](dsl_docs/reference/env_flags.md).
> The two engines must stay byte-identical — see the parity rule in
> [`dsl_docs/development/engine_parity.md`](dsl_docs/development/engine_parity.md).

## Runtime: finding ROCm (libamd_comgr / libamdhip64)

In-process compile + launch needs the ROCm shared libs at runtime. The Python
runtime resolves them WITHOUT importing torch
(`python/rocke/runtime/runtime_coexistence._candidate_lib_paths` /
`_rocm_root_libdirs`), in priority order:

1. explicit full-path override env var: `ROCKE_COMGR_LIB`, `ROCKE_HIP_LIB`;
2. the lib torch uses, only if torch is already imported (the resolver never
   imports torch to obtain it): an unambiguous mapped copy, else
   `<torch>/lib/lib*.so`, else TheRock's `_rocm_sdk_core/lib/lib*.so*` (only
   when `rocm_sdk` is imported, i.e. the torch is TheRock's). Exception for
   comgr only: a stale torch comgr moves below every step 3 candidate (see
   below);
3. a real ROCm install discovered without torch: `$ROCM_PATH` / `$ROCM_HOME` ->
   `<root>/lib`, then globbed `/opt/rocm*/core-*/lib` and `/opt/rocm*/lib`,
   newest version first (a packaged ROCm 7.2 keeps the runtime under a versioned
   `core-*/lib`, not always plain `/opt/rocm/lib`);
4. bare `lib<name>.so` on the dynamic linker's search path (last resort).

With multiple distinct copies in `/proc/self/maps`, the mapped tier prefers
the sole copy under imported torch or the active `_rocm_sdk_core`. It skips
ambiguous mappings rather than treating address order as load order.

Stale torch comgr (`_torch_comgr_is_stale`, comgr only, never HIP): the release
of torch's comgr is read from its location (in `_rocm_sdk_core`: the wheel's
release, `rocm_sdk.__version__` or the installed `rocm-sdk-core` version;
elsewhere: `torch.version.hip`), the same way `resolved_lib_rocm_version` does,
and compared with the release of the first step 3 install whose version is
readable. If both are known and torch's is older, torch's comgr is tried after
every step 3 candidate (still before step 4), because an older comgr rejects
ISAs newer than its ROCm. A `_rocm_sdk_core` comgr already mapped into the
process is never demoted: TheRock's `rocm_sdk.initialize_process` loaded it
`RTLD_GLOBAL`, and a second comgr beside it aborts the process with `LLVM
ERROR: support is already registered for analysis`. It stays first; if it
cannot handle the target ISA, the compile fails at `set_isa`.

On Linux, the Python COMGR loader rejects `ROCKE_COMGR_LIB` before `dlopen` if
it names an existing file with an ELF header different from the only mapped
COMGR path. An alias of the same file remains valid. Missing or non-ELF
overrides retain the existing load-failure fallback. With multiple mapped
paths or unreadable maps, override precedence remains unchanged. Compile
with another COMGR in a separate process before importing torch.

torch is therefore **optional** — required only for the `torch.fx` fusion
frontend, torch-tensor launch (`runtime/torch_interop.py`), and on-GPU torch-eager
numeric checks. Building the engine, lowering, `comgr` compile, numpy launch, and
the byte-identity gate need no torch. If a torch-less process reports `cannot load
libamd_comgr.so`, set `ROCM_PATH` (or `ROCKE_COMGR_LIB`) to your ROCm install.

## The rocke_engine Python binding (optional)

The `cpp` backend of the Python frontend reaches the engine through the
`rocke_engine` pybind module, built from `cpp/bindings/` against a prebuilt
archive:

```bash
cmake -S <rocke/platform>/cpp/bindings -B <bld> -DCMAKE_BUILD_TYPE=Release \
  -DROCKE_ENGINE_ARCHIVE=<build>/librocke_core.a \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$(which python)"
cmake --build <bld> -j$(nproc)
# put <bld> on PYTHONPATH so `import rocke_engine` works; otherwise the cpp
# backend falls back to the Python lowerer (see core/backend.py).
```

## Consuming the engine from a provider plugin

A hipDNN provider links `librocke_core.a` `--whole-archive` and resolves it
strictly to avoid stale-archive failures:

- **`-DROCKE_LIB=/path/to/librocke_core.a`** — use that specific archive. If the file
  does not exist, configuration **fails immediately** (it never searches for a
  fallback).
- **no `-DROCKE_LIB`** — the engine is built **fresh** as an isolated sub-build and
  that archive is linked, so the linked archive is always in lockstep with the
  engine source.

There is intentionally no path that uses a checked-in `build/` archive.

## Artifact hygiene (do not commit build products)

Build products are regenerated by every build. A stale one checked into the tree
silently shadows a fresh build and produces failures that look like code bugs.
**Never commit build artifacts.**

This includes anything matching: `build*/`, `cmake-build*/`, `CMakeFiles/`,
`CMakeCache.txt`, `_deps/`, `__pycache__/`, `*.a`, `*.o`, `*.so`, `*.dylib`,
`*.dll`, `*.lib`, `*.cpython-*.so`. Always build into an out-of-tree directory
(`-B /tmp/rocke`, never inside `rocke/platform/`) so these never land in a commit.

## Freshness stamps

The portable-IR native bundle carries an explicit provenance stamp so a consumer
can detect a mismatch rather than rely on rebuild discipline.
`cmake/rocke_build_id.cmake` computes a deterministic, git-independent content
hash over the repository-owned inputs to the shared engine and replay CLI: the
linked C++ sources and headers (excluding pybind-only sources), the replay CLI
source, the CMake recipes that build and stamp both artifacts, and the inputs of
the generated arch-domain table (the committed columns, `arch_domain_waivers.json`,
`domain.py` and `tools/gen_arch_domain_cpp.py`).

CMake computes the ID while configuring for diagnostics, then
`cmake/generate_rocke_build_id.cmake` recomputes it on every build. A source or
recipe edit therefore refreshes the generated header before
`cpp/core/rocke_build_id.cpp` is rebuilt; `copy_if_different` avoids recompiling
that object when the ID is unchanged. No lowering/emission translation unit
includes the generated header, so the LLVM-IR byte-identity contract is
unchanged.

The engine exposes the stamp through `rocke_build_id()` and
`rocke_engine_version()` (`cpp/include/rocke/rocke_build_id.h`), and the replay
CLI exposes the same identity through `--provenance`. Source-less installed
tests use the independently generated provenance manifest installed beside the
CLI. Source checkouts validate a native artifact against the current source ID
before admitting it to the process; subsequent online calls reuse that immutable
loaded image without re-hashing the source tree. Native source changes therefore
require a rebuild and a new process.

`tools/check_byte_identity.py` prints the engine stamp on every run, and the
pybind extension surfaces it through the `build_id` attribute. Remaining
follow-on work is provider-side validation that a loaded prebuilt-HSACO bundle
matches the engine build ID it was built against.
