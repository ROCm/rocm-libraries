# rocke

> ⚠️ **Contributors — never run `ruff check --fix` on emitter code** (`rocke/core`,
> `helpers`, `instances`). The IR builder is **side-effecting**: `b.const_i32(8)`
> emits an instruction even when its Python handle is unused, so ruff's `F841`
> autofix **deletes the op and silently changes the kernel**. Lint with `ruff
> check` (no `--fix`), use `# noqa: F841` for intentional handles, and re-run the
> byte-identity gate after any lint/format pass. See
> [`dsl_docs/development/invariants.md`](../../dsl_docs/development/invariants.md) rule 9.

## Why
CK Tile kernels are expressive and fast, but editing them through C++
templates can make iteration slow. `rocke` keeps the CK Tile programming
model while moving kernel authoring into Python so developers can build,
inspect, launch, and tune a kernel without waiting on a full C++ compile.

The goal is not to replace CK Tile. The goal is to shorten the loop between
an idea, a generated GPU kernel, a correctness check, and a performance
measurement, while still using CK-style tensor transforms, tile distributions,
MFMA operations, and launch semantics.

## What
`rocke` is a Python authoring layer for Composable Kernel Tile kernels on
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
rocKE/Python/rocke/
├── core/         # IR, printing, optimization passes, LLVM/HIP lowering
├── helpers/      # authoring helpers: manifests, MFMA atoms, epilogues, layouts
├── instances/    # spec-driven CK Tile parity kernels
├── runtime/      # COMGR, HIP module loading, launchers, timing helpers
├── analysis/     # LLVM/ISA/resource inspection helpers
├── benchmark/    # repeated-run benchmark summaries
├── examples/     # maintained Python examples and parity harnesses
├── run_manifest.py
├── sweep.py
└── sweep_bench.py

# architecture/runtime/development docs live at the engine root: rocKE/dsl_docs/
```

## Quick Start
Run from the `rocKE/` root (the package lives under `Python/`):

```bash
export PYTHONPATH=Python
python -m rocke
```

Build and verify one generated example:

```bash
export PYTHONPATH=Python
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"

python -m rocke.examples.common.bake_off_implicit_gemm --output-dir "$OUT_DIR"
python -m rocke.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
```

Run the core unit test suite (the IR/lowering tests need no GPU; ~20
harness/timer tests require a GPU):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Python \
  python tests/test_rocke.py
```

For the full validation suite (relative-path guard + cross-engine byte-identity
gate + pytest, plus ctest when the C++ tests are built), use the entrypoint:

```bash
python tests/run_all.py
```

## Minimal Kernel Build
```python
from rocke import compile_kernel
from rocke.instances import TileSpec, TraitSpec, UniversalGemmSpec
from rocke.instances import build_universal_gemm

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
PYTHONPATH=Python python -m rocke.run_manifest \
  "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
```

For in-process launchers, use `KernelLauncher` and `time_launches` from
`rocke.runtime.launcher`. `time_launches` performs warmup, records HIP events,
runs without per-launch fences inside the timed loop, and returns average
milliseconds per launch.

## Requirements
Recommended stack (what the example READMEs' numbers are reproduced on):
- **ROCm 7.2** with HIP runtime and `libamd_comgr` (also works on ROCm 7.0.x —
  set `ROCKE_LLVM_FLAVOR=llvm22` if your `comgr` is 7.2 but `/opt/rocm` is older).
- **PyTorch 2.12 (+rocm7.2 build)** in a virtual environment — the wheel bundles
  the `libamd_comgr` the DSL prefers, so the venv selects the build toolchain.
- **Python 3.12**; plus `numpy` (references) and `ruff` (lint).
- A supported AMDGPU target for GPU launch tests: CDNA `gfx942`/`gfx950`
  (Linux) or RDNA `gfx1151`/`gfx1201` (Linux + Windows HIP SDK).

Static IR and lowering tests do not require a GPU. Runtime, parity, and
benchmark paths do.

**Full setup (venv, building the C++ engine, env-variable flags, Linux &
Windows):** see [`dsl_docs/development/setup_guide.md`](../../dsl_docs/development/setup_guide.md).

## More Documentation
- `dsl_docs/development/setup_guide.md` — prerequisites, venv, env flags (Linux & Windows)
- `dsl_docs/reference/env_flags.md` — every environment variable
- `dsl_docs/development/engine_parity.md` — the Python⇄C++ parity rule (every optimization needs both)
- `dsl_docs/architecture/mental_model.md`
- `dsl_docs/runtime/compile_launch_and_manifest.md`
- `dsl_docs/runtime/manifest_schema.md`
- `dsl_docs/development/testing.md`
- `dsl_docs/instances/index.md`
