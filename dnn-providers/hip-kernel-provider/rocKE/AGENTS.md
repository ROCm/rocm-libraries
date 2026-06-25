# rocKE - agent onboarding

rocKE is a **dual-engine CK-DSL kernel stack**: a Python authoring frontend and a
C++ backend that emit **byte-identical AMDGPU LLVM IR**. Read this before editing.

## What this is

`ck_dsl` lets you author CK-Tile-style AMDGPU kernels in Python: build a typed SSA
`KernelDef`, lower it to AMDGPU LLVM IR, compile to HSACO in-process via
`libamd_comgr`, and launch through HIP. The same lowering exists as a C++ engine
(`libckc_core.a`) so kernels can be served with **no Python at runtime**.

```
Spec dataclass -> build_*() -> KernelDef -> lower -> .ll -> comgr -> HSACO -> launch
```

## Layout (layers mirror across languages)

```
rocKE/
  Python/ck_dsl/        # authoring frontend (import ck_dsl)
    core/               # IR, passes, lower_llvm/lower_hip, arch, isa, backend, serialize
    helpers/            # tensor views, atoms, epilogues, schedules, manifest, ...
    instances/<arch>/   # spec-driven kernels (common, gfx942, gfx950, gfx1151, gfx1201, gfx1250)
    runtime/            # comgr, hip_module, launcher (Python-only)
    dispatch/ analysis/ benchmark/ heuristics/ examples/   # Python-only
  Cpp/                  # C++20 engine (mirrors the Python layers)
    include/ckc/        # public extern "C" ABI (flat) - the provider/bindings contract
    core/ helpers/ instances/ support/
    bindings/           # ckc_engine pybind module -> links libckc_core.a
  tests/                # by-layer, language-agnostic (see tests/README.md)
  tools/                # check_byte_identity.py and other cross-platform entry points
  cmake/  CMakeLists.txt  pyproject.toml
```

## The #1 invariant: byte-identity

The Python engine (`core/lower_llvm.py`) and the C++ engine (`Cpp/`) MUST emit the
**same LLVM-IR bytes** for every kernel family. Any op/instance/atom/fusion/arch
change must be made in **both** engines in the same change. Prove it:

```bash
python tools/check_byte_identity.py            # build engine fresh + gate (llvm20)
CK_DSL_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py   # and llvm22
```

A change is done only when the gate is GREEN for every family at both flavors.

## Build / test / run

```bash
# PYTHONPATH for the Python package
export PYTHONPATH=<rocKE>/Python                 # then: import ck_dsl

# build the C++ engine + the C++ unit-test binaries (so ctest has something to run)
cmake -S <rocKE> -B /tmp/rocke -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke -j

# relative-path guard + byte-identity gate + pytest; also runs ctest only when
# --build-root already holds the built C++ test binaries (default temp run skips it)
python tests/run_all.py                          # guard + gate + pytest
python tests/run_all.py --build-root /tmp/rocke  # + ctest from the build above
python -m pytest tests/instances/test_ck_dsl_multiarch.py   # a fast CPU-only suite
```

Before updating a PR, run the top-level pre-commit hooks over the PR range from
the repository root:

```bash
pre-commit run --from-ref origin/develop --to-ref HEAD
```

Requirements: `pip install -r requirements.txt` (torch must be the ROCm build,
installed separately). On-GPU lanes need a HIP-visible device + ROCm `comgr`.

## Hard rules

- **Byte-identity**: mirror every emission change in both engines; re-run the gate.
  If you intend to change emitted output, re-bless the golden in the same change.
- **Relative paths only**: no file under `rocKE/` may hardcode an absolute repo
  path or a path escaping `rocKE/`. `tests/run_all.py` enforces this with a grep
  guard; keep anchors derived from `__file__` / `CMAKE_CURRENT_SOURCE_DIR`.
- **Never `ruff check --fix` emitter code** (`core`, `helpers`, `instances`): the
  IR builder is side-effecting (`b.const_i32(8)` emits an op even if its handle is
  unused), so F841 autofix silently changes kernels. Lint with `ruff check` (no
  `--fix`).
- **Cross-platform**: scripts are Python (no `.sh` under `rocKE/`); use `tempfile`,
  `os.cpu_count()`, `pathlib`, `shutil.which` - no `/tmp`, `nproc`, `sudo`.
- **Default arch is `gfx950`** (the byte-identity baseline). Do not change the
  codegen default; for on-GPU runs, prefer the local device via
  `ck_dsl.runtime.hip_module.get_device_arch()` and fall back to `gfx950`.

## Key env flags

| flag | meaning |
|---|---|
| `CK_DSL_BACKEND` | `cpp` (default) \| `python` \| `both` (differential assert) |
| `CK_DSL_CPP_STRICT` | `1` = raise instead of silently falling back to Python when `ckc_engine` isn't built |
| `CK_DSL_LLVM_FLAVOR` | `llvm20` \| `llvm22` (must match the ROCm `comgr` in use) |
