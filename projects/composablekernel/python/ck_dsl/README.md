# ck_dsl

## Why
CK Tile kernels are expressive and fast, but editing them through C++
templates can make iteration slow. `ck_dsl` keeps the CK Tile programming
model while moving kernel authoring into Python so developers can build,
inspect, launch, and tune a kernel without waiting on a full C++ compile.

The goal is not to replace CK Tile. The goal is to shorten the loop between
an idea, a generated GPU kernel, a correctness check, and a performance
measurement, while still using CK-style tensor transforms, tile distributions,
MFMA operations, and launch semantics.

## What
`ck_dsl` is a Python authoring layer for Composable Kernel Tile kernels on
AMDGPU. It builds a small SSA IR, lowers directly to AMDGPU LLVM IR, compiles
HSACO in-process with `libamd_comgr`, and launches through the HIP runtime.

The public shape of most kernels is:

```text
Spec dataclass -> build_*() -> KernelDef -> compile_kernel() -> KernelLauncher
```

Instance modules generally expose:

```text
<Op>Spec
build_<op>(spec)
<op>_signature(spec)
<op>_grid(spec)
```

## Layout
```text
python/ck_dsl/
├── core/         # IR, printing, optimization passes, LLVM/HIP lowering
├── helpers/      # authoring helpers: manifests, MFMA atoms, epilogues, layouts
├── instances/    # spec-driven CK Tile parity kernels
├── runtime/      # COMGR, HIP module loading, launchers, timing helpers
├── analysis/     # LLVM/ISA/resource inspection helpers
├── benchmark/    # repeated-run benchmark summaries
├── examples/     # maintained Python examples and parity harnesses
├── dsl_docs/     # detailed architecture, runtime, and development docs
├── run_manifest.py
├── sweep.py
└── sweep_bench.py
```

## Quick Start
Run from the Composable Kernel repository root:

```bash
export PYTHONPATH=python
python -m ck_dsl
```

Build and verify one generated example:

```bash
export PYTHONPATH=python
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"

python -m ck_dsl.examples.bake_off_implicit_gemm --output-dir "$OUT_DIR"
python -m ck_dsl.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
```

Run the core static test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  python python/test/test_ck_dsl.py
```

Run generated example tests when a ROCm GPU is available:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  python python/test/test_ck_dsl_examples.py
```

## Minimal Kernel Build
```python
from ck_dsl import compile_kernel
from ck_dsl.instances import TileSpec, TraitSpec, UniversalGemmSpec
from ck_dsl.instances import build_universal_gemm

spec = UniversalGemmSpec(
    name="my_gemm",
    tile=TileSpec(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    ),
    trait=TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="cshuffle",
    ),
)

kernel = build_universal_gemm(spec)
artifact = compile_kernel(kernel)
```

`artifact.hsaco` contains the code object bytes, `artifact.llvm_text` contains
the lowered LLVM IR, and `artifact.timings` records the compile stages.

## Running And Timing
For Python-owned examples, prefer the manifest runner:

```bash
PYTHONPATH=python python -m ck_dsl.run_manifest \
  "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
```

For in-process launchers, use `KernelLauncher` and `time_launches` from
`ck_dsl.runtime.launcher`. `time_launches` performs warmup, records HIP events,
runs without per-launch fences inside the timed loop, and returns average
milliseconds per launch.

## Requirements
- ROCm with HIP runtime and `libamd_comgr` available through the default ROCm
  installation or the dynamic linker.
- Python with the dependencies used by CK's Python tests.
- A supported AMDGPU target for GPU launch tests.

Static IR and lowering tests do not require a GPU. Runtime, parity, and
benchmark paths do.

## More Documentation
- `dsl_docs/architecture/mental_model.md`
- `dsl_docs/runtime/compile_launch_and_manifest.md`
- `dsl_docs/runtime/manifest_schema.md`
- `dsl_docs/development/testing.md`
- `dsl_docs/instances/index.md`
