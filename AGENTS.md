# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
This is the ROCm **rocm-libraries** super-repo: a monorepo of C++/HIP GPU
libraries (`projects/*`, e.g. rocblas, rocsparse, miopen, ...) plus shared
components (`shared/*`). See `README.md` and `CONTRIBUTING.md` for the full
layout and the CMake superbuild instructions.

### Hardware / toolchain reality in the Cloud VM
The Cloud Agent VM is **CPU-only**: there is **no AMD GPU** (`/dev/kfd` absent)
and **no ROCm toolchain** (`amdclang++`, `hipcc`, `hipconfig`, `rocminfo` are
not installed). Consequently the C++/HIP superbuild
(`cmake --preset ...` / `cmake --build`) and any GPU correctness/performance
tests documented in `CONTRIBUTING.md` **cannot be built or run here** — they
require the ROCm compiler and AMD hardware. Do not spend time trying to run the
full superbuild in this environment.

### What CAN be developed/tested offline: `shared/tensile`
[Tensile](shared/tensile) is a pure-Python GEMM/tensor-contraction kernel
generator and auto-tuner. Its Python code, lint suite, and `unit`-marked
pytest suite run fully on CPU and are the practical dev target in this VM.

The update script installs Tensile (editable) plus its runtime/dev/test/lint
deps into the user site (`pip --user`). Note **`~/.local/bin` is not on `PATH`**
by default, so invoke tools as Python modules:

```bash
cd shared/tensile
python3 -m pytest Tensile/Tests -m unit -n 4        # unit tests
python3 -m flake8 Tensile                            # lint
# format/import-order checks (file list mirrors tox.ini black/isort envs):
python3 -m black --check --line-length=100 docs/ Tensile/Parallel.py Tensile/TensileCreateLibrary.py Tensile/TensileCreateLib/ Tensile/BuildCommands/ Tensile/Utilities/Toolchain.py Tensile/Tests/unit/test_TensileCreateLibrary.py Tensile/Tests/unit/test_KernelFileContext.py Tensile/Tests/unit/test_AsmRegisterPool.py
python3 -m isort --check --profile=black <same file list as black>
```

The canonical command definitions live in `shared/tensile/tox.ini`
(`unittest`, `lint`, `format`, `isort` envs) and `shared/tensile/pytest.ini`.
`tox` itself is not used here because several tox envs build the C++ client /
assemble kernels, which need ROCm.

### Expected unit-test result without ROCm
`python3 -m pytest Tensile/Tests -m unit` yields roughly **5454 passed** with
~**13 failed + 6 errored**. Every one of those failures/errors is a
`FileNotFoundError` for `amdclang++` (the ROCm assembler) from tests that
assemble kernels or parse library-logic files (e.g. `test_tryAssembler`,
`test_useGlobalParameters`, `test_LibraryIO::test_parseLibraryLogic*`,
`test_TensileCreateLibrary::test_prepAsm`). These are **expected** on a
GPU-less VM and are not regressions.

### Generating a GEMM kernel offline (no GPU)
Tensile normally probes `hipconfig` / the ROCm assembler at startup. To
initialize it and generate a **Source** (HIP C++) kernel without ROCm:
- put a stub `hipconfig` (prints a version string, e.g. `6.3.0`) and a stub
  `rocm_agent_enumerator` (prints e.g. `gfx90a`) on `PATH`, and
- pass `Tensile.AsmCaps.getCapabilitiesCache(<hipVersion>)` as the
  `capabilitiesCache` arg to `Tensile.Common.assignGlobalParameters(...)`.

This uses Tensile's built-in offline capabilities cache so `GetAsmCaps` never
invokes the assembler. Assembly-kernel generation still needs the real ROCm
assembler and cannot be done here.
