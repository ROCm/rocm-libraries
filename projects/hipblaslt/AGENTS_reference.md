# hipBLASLt Reference

Supplementary reference for `AGENTS.md` — load this when you need raw cmake commands, cmake presets, or test instructions.

## Raw CMake Build

After activating `build/venv`:

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx942 \
  -DPython_EXECUTABLE="$(pwd)/build/venv/bin/python" \
  -DPython3_EXECUTABLE="$(pwd)/build/venv/bin/python"
cmake --build build --parallel
```

CMake presets (`cmake --list-presets`): `default:release`, `hipblaslt` (host lib only, no device libs), `gemm-libs` (device libs only), `hipblaslt-clients`, `tensilelite`, `rocisa`, plus pinned configs `rocm-7.0.0` / `rocm-7.0.2`.

## Running Tests

Test data is **generated at build time** from `clients/tests/data/*.yaml` by `clients/tests/hipblaslt_gentest.py` into `build/clients/hipblaslt_gtest.data`. The binary looks for that file in its own directory, so:

```bash
cmake --build build --target tensilelite-device-libraries  # one-time per GPU config; slow
cmake --build build --target hipblaslt-test
cd build/clients && ./hipblaslt-test --gtest_filter='*quick*'
```

Runtime device-library lookup: `HIPBLASLT_TENSILE_LIBPATH` must point **directly** at the dir containing `TensileLibrary_lazy_<arch>.dat` and the code objects (e.g. `build/Tensile/library` or `/opt/rocm/lib/hipblaslt/library`). If unset, the loader resolves relative to the loaded shared library. Matmul tests will fail without a device library for the running GPU.

To skip a test on a known-broken platform, add an entry under `clients/tests/data/known_bugs.yaml` (matched by `function`/`initialization`/`known_bug_platforms`).

`rtest.py` at the repo root is a build-driven test runner — not the usual entry point for local dev.
