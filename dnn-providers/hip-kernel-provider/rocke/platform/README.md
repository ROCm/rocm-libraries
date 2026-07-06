# rocKE

**rocKE** is a dual-engine rocke kernel stack for AMDGPU: a **Python authoring
frontend** (`rocke`) and a **C++ engine** (`Cpp/` → `librocke_core.a`) that emit
**byte-identical** AMDGPU LLVM IR. You author kernels in Python (build a typed
SSA `KernelDef`), lower to LLVM IR, compile to HSACO in-process via
`libamd_comgr`, and launch through HIP. The same lowering exists in C++ so
kernels can be served with **no Python at runtime**.

```text
Spec dataclass -> build_*() -> KernelDef -> lower -> .ll -> comgr -> HSACO -> launch
```

> Deeper docs: agent/onboarding notes in [`AGENTS.md`](AGENTS.md), the canonical
> build + artifact-hygiene reference in [`BUILD.md`](BUILD.md), the test-tree map
> in [`tests/README.md`](tests/README.md), and architecture/runtime/development
> docs under [`dsl_docs/`](dsl_docs).

## Layout

```text
rocKE/
  Python/rocke/   # authoring frontend (import rocke): core, helpers, instances/<arch>,
                   # runtime, dispatch, analysis, benchmark, heuristics, examples
  Cpp/             # C++20 engine (mirrors the Python layers): core, helpers, instances,
                   # support; include/ckc (public extern "C" ABI); bindings (rocke_engine)
  tests/           # by-layer, language-agnostic; run_all.py is the entrypoint
  dsl_docs/        # merged architecture / runtime / development docs
  tools/           # cross-platform Python tooling (check_byte_identity.py, ...)
  cmake/  CMakeLists.txt  pyproject.toml  requirements.txt
```

`Python/rocke/<layer>` and `Cpp/<layer>` mirror each other by layer name. The
two engines must stay byte-identical (see [`dsl_docs/development/engine_parity.md`](dsl_docs/development/engine_parity.md)).

## Prerequisites

- **ROCm 7.x** with `libamd_comgr` and `libamdhip64` on the dynamic-linker path
  (7.2 recommended → `ROCKE_LLVM_FLAVOR=llvm22`; older → `llvm20`). ROCm is a
  system dependency, not a pip package.
- **Python 3.10+**, a C++20 compiler (`amdclang++`/`clang++`/`g++`), CMake 3.16+
  (3.18+ for the `Cpp/bindings` pybind module).
- Python deps: `pip install -r requirements.txt` (adds `numpy`, plus `pytest` /
  `pybind11` in the `dev` extra). **torch** must be the ROCm build for your
  system, installed separately from the ROCm wheel index (not via this repo),
  for example: `pip install --index-url https://download.pytorch.org/whl/rocm7.0 torch`.
- GPU lanes need a HIP-visible device; on a shared box that may require running
  under `sudo -E` (or membership in the `video`/`render` groups).

```bash
export ROCKE=$(pwd)        # run from this rocKE/ directory
export PYTHONPATH=$ROCKE/Python
```

## Build

### The engine (`librocke_core.a`)

```bash
cmake -S "$ROCKE" -B /tmp/rocke -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke --target rocke_core -j$(nproc)
# -> /tmp/rocke/librocke_core.a
```

The top-level `CMakeLists.txt` globs `Cpp/**/*.cpp` (excluding `Cpp/bindings/`)
into `rocke_core`, with the public ABI headers at `Cpp/include`. Optional
diagnostic build: add `-DROCKE_SANITIZE=ON` (ASan/UBSan; not for shipping).

### The `rocke_engine` Python binding (optional, enables the `cpp` backend)

```bash
cmake -S "$ROCKE/Cpp/bindings" -B /tmp/rocke_pybind -DCMAKE_BUILD_TYPE=Release \
  -DROCKE_ENGINE_ARCHIVE=/tmp/rocke/librocke_core.a \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$(which python)"
cmake --build /tmp/rocke_pybind -j$(nproc)
export PYTHONPATH="$ROCKE/Python:/tmp/rocke_pybind"   # so `import rocke_engine` resolves
```

If `rocke_engine` is not importable, the default `cpp` backend transparently falls
back to the native Python lowerer (set `ROCKE_CPP_STRICT=1` to make that an
error). See [`dsl_docs/reference/env_flags.md`](dsl_docs/reference/env_flags.md).

## Run

Author + lower a kernel in Python (no GPU needed to build/lower):

```bash
python -c "
from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec, TileSpec, TraitSpec, DataSpec, build_universal_gemm)
from rocke.core.lower_llvm import lower_kernel_to_llvm
spec = UniversalGemmSpec(name='demo',
    tile=TileSpec(tile_m=128, tile_n=128, tile_k=32, warp_m=2, warp_n=2, warp_k=1,
                  warp_tile_m=16, warp_tile_n=16, warp_tile_k=16),
    trait=TraitSpec(pipeline='compv3', epilogue='default'),
    data=DataSpec(dtype_a='fp16', dtype_b='fp16', dtype_c='fp16', dtype_acc='fp32'),
    wave_size=64, block_size=256, batched=False)
print(lower_kernel_to_llvm(build_universal_gemm(spec, arch='gfx950'), arch='gfx950')[:400])
"
```

Compile + launch on a GPU uses `rocke.runtime` (comgr + HIP). The default arch
is `gfx950`; on-GPU paths can target the local device via
`rocke.runtime.hip_module.get_device_arch()`.

## Test

One cross-platform entrypoint runs the relative-path guard → byte-identity gate
→ pytest. It **also** runs `ctest`, but only when the `--build-root` already
contains the built C++ test binaries — the gate itself builds just `rocke_core`,
so a default run (fresh temp build-root) does the guard + gate + pytest and
skips ctest. To include the C++ unit tests, point `--build-root` at a full
build (see step 3 below):

```bash
python "$ROCKE/tests/run_all.py"                       # guard + gate + pytest
python "$ROCKE/tests/run_all.py" --only gemm           # scope the gate to some families
python "$ROCKE/tests/run_all.py" --no-gate             # skip the engine build/gate
python "$ROCKE/tests/run_all.py" --build-root /tmp/rocke  # also run ctest from this build
```

Or run the pieces individually:

```bash
# 1) Byte-identity gate: build the engine fresh + prove C-engine == Python-engine .ll
python "$ROCKE/tools/check_byte_identity.py"
ROCKE_LLVM_FLAVOR=llvm22 python "$ROCKE/tools/check_byte_identity.py"   # also the llvm22 flavor

# 2) Python unit/integration suite
python -m pytest "$ROCKE/tests"

# 3) C++ unit tests (after a build that includes the test targets)
cmake -S "$ROCKE" -B /tmp/rocke && cmake --build /tmp/rocke -j$(nproc)
ctest --test-dir /tmp/rocke --output-on-failure

# 4) On-GPU numeric (needs a HIP device; use sudo -E on a shared box)
sudo -n -E env PYTHONPATH="$ROCKE/Python" python \
  "$ROCKE/tests/instances/differential/numeric.py" --arch gfx950
```

Arch coverage (gfx942 / gfx1151 / gfx1201 in addition to gfx950) rides on the
emitter config set the byte-identity gate enumerates; there is no separate arch
sweep. See [`tests/README.md`](tests/README.md) for the per-layer test map and
the dedup/coverage notes.

## License

Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
