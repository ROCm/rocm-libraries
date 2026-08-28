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
```

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

Optional sanitizer build for diagnostics (not for shipping): `-DROCKE_SANITIZE=ON`.

> **Toolchain/runtime flags.** Codegen is driven by the `comgr` in use, and the
> emitted IR flavor must match it: set `ROCKE_LLVM_FLAVOR=llvm22` for a ROCm 7.2
> `comgr` if `/opt/rocm` is older (avoids a `COMPILE_SOURCE_TO_BC` rejection).
> Full flag list: [`dsl_docs/reference/env_flags.md`](dsl_docs/reference/env_flags.md).
> The two engines must stay byte-identical — see the parity rule in
> [`dsl_docs/development/engine_parity.md`](dsl_docs/development/engine_parity.md).

## Runtime: finding ROCm (libamd_comgr / libamdhip64)

In-process compile + launch needs the ROCm shared libs at runtime. The Python
runtime resolves them WITHOUT importing torch
(`Python/rocke/runtime/runtime_coexistence._candidate_lib_paths` /
`_rocm_root_libdirs`), in priority order:

1. explicit full-path override env var: `ROCKE_COMGR_LIB`, `ROCKE_HIP_LIB`;
2. torch-bundled `<torch>/lib/lib*.so` — only if torch is already imported (the
   resolver never imports torch to obtain it);
3. a real ROCm install discovered without torch: `$ROCM_PATH` / `$ROCM_HOME` ->
   `<root>/lib`, then globbed `/opt/rocm*/core-*/lib` and `/opt/rocm*/lib`,
   newest version first (a packaged ROCm 7.2 keeps the runtime under a versioned
   `core-*/lib`, not always plain `/opt/rocm/lib`);
4. bare `lib<name>.so` on the dynamic linker's search path (last resort).

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

The engine carries an explicit freshness stamp so a consumer can *detect* a
mismatch rather than rely on rebuild discipline. At CMake configure time
`cmake/rocke_build_id.cmake` computes a deterministic, git-independent content hash
of the engine sources (`cpp/**`) and injects it (plus a human `engine_version`)
into `cpp/core/rocke_build_id.cpp` as compile definitions — scoped to that single
TU so no emission object is touched (the `.ll` byte-identity contract holds). The
stamp is exposed by `rocke_build_id()` / `rocke_engine_version()`
(`cpp/include/rocke/rocke_build_id.h`), printed by `tools/check_byte_identity.py` on
every run, and surfaced through the pybind `build_id` attribute. Changing any
tracked source byte changes the build-id, so a stale or mixed-build archive is
detectable.

Remaining follow-on: a manifest/bundle version + provider-side validation that a
loaded prebuilt-HSACO bundle matches the engine build-id it was built against.
