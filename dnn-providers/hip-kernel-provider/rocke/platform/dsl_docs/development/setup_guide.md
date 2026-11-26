# Setup Guide (Linux & Windows)

How to install prerequisites, create a virtual environment, build the optional
C++ engine, and configure the environment flags for the CK DSL. For a guided
*learning* path once you're set up, see [`onboarding.md`](./onboarding.md); for
engine-internals work see [`engine_contributing.md`](./engine_contributing.md).

---

## 1. Recommended stack

The DSL generates AMDGPU LLVM IR text and compiles it to a HSACO with
`libamd_comgr` (no hipcc, no MLIR). The IR flavor follows the compiler the loaded
COMGR runs, which rocKE finds and queries itself (§5,
[Compiler detection and provenance](../reference/env_flags.md#compiler-detection-and-provenance)).

| Component | Recommended |
|---|---|
| ROCm | **10.x** |
| Python | **3.12** (3.10+ works) |
| OS | Linux (full GPU support); Windows (build/lowering + RDNA execution; see §4) |

rocKE does not use PyTorch: references are numpy (`rocke.numeric`) and device
data is `DeviceArray`.

**Supported GPUs**

| Family | ISA | Notes |
|---|---|---|
| CDNA3/4 | `gfx942`, `gfx950` | Linux only; wave64 |
| RDNA3.5/4 | `gfx1151`, `gfx1201` | Linux + Windows (HIP SDK); wave32 |

> **Why the stack matters.** Codegen depends on the compiler the loaded COMGR
> runs: the generated IR must match its LLVM version (§5), and a few kernels
> behave differently on older compilers (e.g. a persistent-kernel path that hangs
> an older comgr). Record which compiler a measurement used
> (`rocke.runtime.comgr.loaded_compiler_info().describe()`).

---

## 2. Get the source

```bash
git clone <your-rocm-libraries-remote> rocm-libraries
cd rocm-libraries/dnn-providers/hip-kernel-provider/rocke/platform
export PYTHONPATH=python          # the DSL lives under python/rocke
```

All commands below assume you are in the `rocke/platform/` directory with
`PYTHONPATH=python` unless noted.

---

## 3. Linux setup (full support)

### 3.1 Install ROCm

Install ROCm 10.x from AMD's packages for your distro (see the ROCm install
guide). Confirm the GPU is visible and note your `gfx` target:

```bash
rocminfo | grep -E "Name:|gfx"
ls /opt/rocm/lib/libamd_comgr.so        # the in-process compiler the DSL uses
```

### 3.2 Create a virtual environment

```bash
python3.12 -m venv ~/.venv
source ~/.venv/bin/activate
pip install --upgrade pip
pip install numpy ruff           # numpy for references; ruff for lint
python -c "from rocke.runtime import comgr; print(comgr.loaded_compiler_info().describe())"
```

The last line names the COMGR rocKE loads and the compiler it runs
(`ROCKE_COMGR_LIB` selects another one; see
[Compiler detection and provenance](../reference/env_flags.md#compiler-detection-and-provenance)).

> ⚠️ **Lint with `ruff check` — never `ruff check --fix` — on emitter code**
> (`rocke/core`, `helpers`, `instances`). The IR builder is **side-effecting**:
> `b.const_i32(8)` emits an op even when its handle is unused, so ruff's `F841`
> autofix deletes the assignment and silently changes the emitted kernel. Use
> `# noqa: F841` for intentional unused handles, and **re-run the byte-identity
> gate after any lint/format pass**. See `dsl_docs/development/invariants.md`
> (rule 9).

### 3.3 GPU access

Kernel **lowering** (`comgr`) needs no GPU and runs anywhere. Kernel **launch**
needs access to the GPU device nodes (`/dev/kfd`, `/dev/dri/renderD*`). If
`rocminfo` lists no `gfx` agent, your user is not in the device groups (e.g.
`video`/`render`). Either add yourself
(`sudo usermod -aG video,render $USER` then re-login) or run launch steps with
elevated privileges; the benchmark harness honours `ROCKE_USE_SUDO=1` (§5).

### 3.4 (Optional) Build the C++ engine for the `cpp` backend

The default backend (`ROCKE_BACKEND=cpp`) lowers Python-authored kernels through
the C++ engine. If the `rocke_engine` module isn't built/importable, the DSL
**automatically falls back to the native Python lowerer** (byte-identical output)
and records why — so this step is optional, but build it to exercise the C
engine:

```bash
# 1. engine archive (no GPU needed); the top-level CMakeLists builds rocke_core:
cmake -S . -B /tmp/rocke_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke_build --target rocke_core -j"$(nproc)"
# 2. the pybind module (see cpp/bindings/ for its CMake):
#    build it into a dir, then put that dir on PYTHONPATH:
export PYTHONPATH=python:/path/to/rocke_engine_build_dir
python -c "import rocke_engine; print('engine build_id:', rocke_engine.build_id())"
```

### 3.5 Verify

```bash
# import + backend:
python -c "import rocke, rocke.core.backend as b; print('backend:', b.resolve_backend())"
# build + run a GEMM on the GPU (prints TFLOPS + PASS/FAIL):
python -m rocke.examples.common.universal_gemm_verify --arch gfx950 --m 1024 --n 1024 --k 1024
# the cross-engine byte-identity gate (no GPU):
tools/check_byte_identity.py
```

---

## 4. Windows setup (build + lowering; RDNA execution)

Windows support targets **RDNA (gfx1151 / gfx1201)** via the AMD **HIP SDK for
Windows**. CDNA/MI parts are Linux-only. The IR-generation and `comgr` build path
work on Windows (the DSL loads `amd_comgr.dll`); GPU *execution* depends on the
Windows ROCm/HIP runtime, which is newer and less complete than Linux — treat
launch/perf on Windows as experimental.

### 4.1 Install the HIP SDK for Windows

Install the AMD HIP SDK for Windows. It provides `amd_comgr.dll` (the in-process
compiler the DSL loads) and the HIP runtime. Note the install root, e.g.
`C:\Program Files\AMD\ROCm\<version>`.

### 4.2 Python

```bat
py -3.12 -m venv %USERPROFILE%\.venv
%USERPROFILE%\.venv\Scripts\activate
pip install --upgrade pip
pip install numpy ruff
```

### 4.3 Make `amd_comgr.dll` discoverable

The DSL's compiler wrapper loads `amd_comgr.dll`. Ensure the HIP SDK `bin`
directory is on `PATH` (or copy the DLL beside your interpreter):

```bat
set PATH=C:\Program Files\AMD\ROCm\<version>\bin;%PATH%
set PYTHONPATH=python
python -c "from rocke.runtime import comgr; print(comgr.loaded_compiler_info().describe())"
```

### 4.4 Build the C++ engine (optional)

Use CMake with the Visual Studio or Clang toolchain:

```bat
cmake -S . -B build_win -DCMAKE_BUILD_TYPE=Release
cmake --build build_win --target rocke_core --config Release -j
```

If the `rocke_engine` pybind module isn't built, the DSL falls back to the Python
lowerer automatically (byte-identical), so the engine is optional on Windows too.

### 4.5 Verify (Windows)

```bat
set PYTHONPATH=python
:: lowering only (no GPU): generate .ll for a kernel
python -c "from rocke.instances.common.gemm_universal import *; from rocke.core.lower_llvm import lower_kernel_to_llvm; print('lowered ok')"
```

For GPU launch on Windows, target an RDNA `gfx` (e.g. `--arch gfx1201`) and
expect runtime maturity to vary; the lowering/build path is the supported part.

---

## 5. Environment variables

Set these to configure the DSL. On Linux use `export NAME=value`; on Windows use
`set NAME=value`.

### Core

| Variable | Values (default) | Purpose |
|---|---|---|
| `ROCKE_BACKEND` | `cpp` \| `python` \| `both` (**cpp**) | Which engine lowers Python-authored kernels. `cpp` = the C++ engine (falls back to Python if no shared `librocke` is found); `python` = the native lowerer; `both` = run both and assert byte-identical output (the differential check). |
| `ROCKE_LLVM_FLAVOR` | `llvm20` \| `llvm22` \| `llvm23` \| `llvm24` (automatic when unset) | Select the LLVM IR flavor (datalayout/intrinsics). Automatic selection lazily queries the loaded COMGR binary's LLVM version and fails if compiler evidence is unavailable. An explicit flavor supports offline IR emission without loading COMGR; it does not change the compiler consuming the IR. COMGR compilation still requires compiler-version evidence and checks the IR's p8 generation against that compiler. See [Compiler detection and provenance](../reference/env_flags.md#compiler-detection-and-provenance) for library selection and lifetime details. |
| `ROCKE_CPP_STRICT` | `1` (unset) | Make `ROCKE_BACKEND=cpp` **raise** instead of silently falling back to the Python lowerer when the C++ engine cannot lower a kernel or is not found. Use to guarantee you are exercising the C engine. |
| `ROCKE_USE_SUDO` | `1` (unset) | The benchmark/sweep harness launches kernels via `sudo -n -E` (for boxes where the user lacks GPU device-group access). |

### hipDNN provider

| Variable | Values (default) | Purpose |
|---|---|---|
| `ROCKE_C_JIT` | `1` (unset) | Provider generates kernels from C source at runtime (C-JIT mode) instead of loading prebuilt HSACOs. |
| `ROCKE_PROVIDER_C_JIT` | CMake `ON`/`OFF` | Build-time switch to compile the provider's C-JIT path. |
| `ROCKE_ALLOW_ENGINE_MISMATCH` | `1` (unset) | Downgrade the engine build-id freshness check from a hard error to a warning when a kernel bundle's stamp doesn't match the linked engine. Default behaviour is to fail loudly on a stale/mismatched bundle. |

### Tooling / examples

| Variable | Purpose |
|---|---|
| `ROCKE_PARITY_BUILD` / `ROCKE_PARITY_EMIT` | The binding-parity harness's `.so` dir and prebuilt-emitter dir — **must point at the same fresh build** (a fresh `.so` with a stale emitter dir reports false mismatches). |
| `ROCKE_CI_RUN_DIR` | Out-of-tree directory for local/cluster CI runs (keeps build products off the source tree). |
| `AITER_PATH` | Path to an AITER checkout used by some examples for external-baseline comparison. |

### Advanced / diagnostic (leave unset)

A number of `ROCKE_FP8_*`, `CK_WSP3_*`, `CK_SWZ_*`, `HIPDNN_GFX942_*`, and
`ATOM_*` flags are experimental kernel-development knobs, all **off by default**.
They are read at the relevant instance/builder sites; don't set them unless you
are working on that specific kernel path.

---

## 6. Common setup pitfalls

| Symptom | Cause / fix |
|---|---|
| `do_action(COMPILE_SOURCE_TO_BC): status=1` | Inspect `comgr.loaded_compiler_info().describe()` and the COMGR diagnostic. AUTO uses loaded compiler evidence; do not choose an override from a ROCm release label (§5). |
| `No module named 'rocke'` | `PYTHONPATH` not pointing at `rocke/platform/python`. |
| `rocminfo` lists no `gfx` agent | Not in GPU device groups (§3.3). |
| `cpp` backend silently uses Python | no shared `librocke` found; build the `rocke_shared` target and point `ROCKE_ONLINE_LIB` at it, and/or set `ROCKE_CPP_STRICT=1` to surface it. |
| A persistent-kernel example hangs the compiler | Known on older comgr; use ROCm 10.x. |

For deeper failure modes (stale artifacts, the differential gate), see
[`troubleshooting.md`](./troubleshooting.md).
