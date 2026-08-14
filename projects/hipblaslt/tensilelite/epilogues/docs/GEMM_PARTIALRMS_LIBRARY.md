# Building a TensileLite device library for the GEMM+PartialRMS kernel

This document explains how to compile the GEMM+PartialRMS (K1) kernel into a
TensileLite device library (`.hsaco` / `.co` / `.dat`) that hipBLASLt can load
at runtime. It covers the two-step process: generating LibraryLogic YAMLs from
the benchmark YAML, then invoking `TensileCreateLibrary` to compile them.

---

## Background

At build time, hipBLASLt compiles kernels via `TensileCreateLibrary`, which takes
**LibraryLogic YAML** files as input (already-selected solutions, not benchmark
configs) and produces:

| Artifact | Purpose |
|---|---|
| `Kernels.so-000-<arch>.hsaco` | Compiled AMDGPU ELF containing all kernels |
| `TensileLibrary_..._<arch>.co` | Clang-offload-bundled code object |
| `TensileLibrary_..._<arch>.dat.zlib` | msgpack solution library for runtime dispatch |
| `TensileLibrary_lazy_<arch>.dat.zlib` | Lazy-load dispatch index |
| `TensileLiteLibrary_lazy_<arch>_Mapping.dat.zlib` | Lazy filename mapping |

The LibraryLogic YAML is a 12-element YAML sequence encoding the kernel
parameters, the problem type, and a size→solution selection table.

---

## Quick start: one-shot build

Use `epilogues/scripts/build_library.sh` to go from the benchmark YAML to a
compiled device library in a single command:

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite

bash epilogues/scripts/build_library.sh \
    --yaml "$PWD/epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml" \
    --chip gfx950 \
    --out-dir /tmp/partialrms_lib \
    --group-by PartialRMSResidualAdd
```

The script calls `python -m Tensile.Tensile` to benchmark and emit
LibraryLogic YAMLs, then `python -m Tensile.TensileCreateLibrary` to compile them.

---

## Prerequisites

- `~/.tensile` venv with `rocisa` and `Tensile` importable (run `invoke rocisa`
  once if not already installed).
- `amdclang++` on PATH (provided by ROCm).
- A `gfx950` GPU is not required for compilation, only for running the kernel.

---

## Step 1: Generate the LibraryLogic YAMLs

The native Tensile pipeline benchmarks all K1 solutions from
`epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml` and writes one LibraryLogic
YAML per distinct `ProblemType` (plain, `PRMS`, `PRMS_RA`).

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite

python -m Tensile.Tensile \
    epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml \
    /tmp/partialrms_out
```

This produces:

```
/tmp/partialrms_out/3_LibraryLogic/
  partialrms_k1_Cijk_..._PRMS_UserArgs.yaml      (no-residual solutions)
  partialrms_k1_Cijk_..._PRMS_RA_UserArgs.yaml   (residual-add solutions)
```

Both variants coexist in one merged library and are dispatched at runtime by
the `UsePartialRMS` and `UsePartialRMSResidualAdd` predicates.

### How distinct LibraryLogic filenames are produced

`ProblemType.__str__()` now appends `PRMS` when `UsePartialRMS=True` and
`PRMS_RA` when `PartialRMSResidualAdd=True`, giving each variant a unique
problem-type string. This is what prevents the two variants from colliding in
the output directory.

---

## Step 2: Compile with TensileCreateLibrary

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite

mkdir -p /tmp/partialrms_output

python -m Tensile.TensileCreateLibrary \
    --architecture=gfx950 \
    --cxx-compiler=$(which amdclang++) \
    --jobs=$(nproc) \
    /tmp/partialrms_logic/ \
    /tmp/partialrms_output/ \
    HIP
```

Typical output (10 unique solutions across both files):

```
TensileCreateLibrary
...
no type mismatches found
10 unique solutions
10 kernels processed
```

Compilation takes approximately 6 seconds per kernel.

### Output layout

```
/tmp/partialrms_output/
  library/
    gfx950/
      Kernels.so-000-gfx950.hsaco          (~870 KB)
      TensileLibrary_..._gfx950.co         (~85 KB)
      TensileLibrary_..._gfx950.dat.zlib   (~11 KB)
      TensileLibrary_lazy_gfx950.dat.zlib
      TensileLiteLibrary_lazy_gfx950_Mapping.dat.zlib
```

---

## Step 3: Verify (optional, requires gfx950 GPU)

```bash
source ~/.tensile/bin/activate
cd /path/to/tensilelite

python epilogues/bench/bench_gemm_rms.py
# Expected: "verification PASSED", ~870+ TFLOPS K1, ~530 GB/s epilogue
```

The existing unit tests also exercise the kernel end-to-end:

```bash
pytest epilogues/unittests/test_gemm_partial_rms.py -v          # 980 tests
pytest epilogues/unittests/test_gemm_partial_rms_epilogue.py -v  # 6 tests
```

---

## Kernel parameters

The K1 kernel is defined in `epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml`. Key parameters:

| Parameter | Value |
|---|---|
| Data type | bf16 (TN GEMM, `DataType=b`) |
| MacroTile variants | 128×64, 256×64, 512×64, 128×128, 512×128 (5 tile configs) |
| DepthU | 64 |
| MatrixInstruction | `[16, 16, 32, 1, 1, 4, 8, wg0, wg1]` (MI16x16x32) |
| StreamK | 3 (persistent kernel, ForceDPOnly mode) |
| StreamKForceDPOnly | 1 |
| PartialRMS | True (subtile PartialRMS epilogue enabled) |
| UseSubtileImpl | True |
| PrefetchGlobalRead | 2 |

`MIWaveGroup[1]` sets the number of M-dimension waves. `MacroTile0 = 16 * 8 * MIWaveGroup[0]`
(N_hidden tiles); `MacroTile1 = 16 * 4 * MIWaveGroup[1]` (M-token tiles). The `--wg-n`
flag to `bench_gemm_rms.py` controls `MIWaveGroup[1]` only:

```bash
python epilogues/bench/bench_gemm_rms.py --wg-n 1   # MT1=64  (MIWaveGroup[1]=1)
python epilogues/bench/bench_gemm_rms.py --wg-n 2   # MT1=128 (MIWaveGroup[1]=2, default)
```

---

## How hipBLASLt invokes TensileCreateLibrary at build time

hipBLASLt's CMake build (`cmake/HipBLASLtCodegen.cmake`) chains two custom
commands:

```bash
# Step 1: validate logic files
python -m Tensile.TensileLogic <LOGIC_PATH> --known-bugs known_bugs.yaml --check-all

# Step 2: compile kernels
python -m Tensile.TensileCreateLibrary \
    --architecture=gfx942;gfx90a \
    --cxx-compiler=$(which amdclang++) \
    --jobs=$(nproc) \
    [--logic-filter="gfx942/Equality/*"] \
    projects/hipblaslt/library/ \
    <build>/Tensile/ \
    HIP
```

The logic YAMLs under `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/`
were produced from offline benchmarking runs and committed to the repository.
`TensileCreateLibrary` only compiles them — it does not benchmark or select solutions.

The recommended build path is `invoke build -a gfx942 -f 'gfx942/Equality/*'`
from `projects/hipblaslt`, which handles the venv, rocisa install, and
`--logic-filter` automatically.
