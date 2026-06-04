# CK DSL – Composable Kernel Tile DSL

## Overview

`ck_dsl` is a Python DSL for authoring GPU kernels targeting AMD GPUs (AMDGPU). It provides a high-level Python interface for writing CK Tile-style kernels that lower in-process to AMDGPU LLVM IR and compile to HSACO (HSA Code Object) without requiring `hipcc`, template metaprogramming, or MLIR round-trips.

**Key characteristics:**
- ~34,000 lines of Python code
- Pure Python authoring surface with SSA IR representation
- Direct LLVM IR emission via in-process lowering
- Runtime compilation using `libamd_comgr` and `libamdhip64`
- Targets AMD MI300 series (gfx940/942/950)
- Wave size fixed at 64

## Architecture

The package is organized into 4 main layers:

### 1. Core Layer (`core/`)
**Purpose:** SSA IR, lowering passes, and IR transformation

- **`ir.py`** – Python SSA IR definitions (`KernelDef`, `Op`, `Value`, `Region`, `IRBuilder`)
  - All kernels are first-class Python data structures, not C++ text
  - Operations produce SSA values with explicit types
  - Control flow operations carry nested regions
  - No external dependencies (stdlib only)

- **`lower_llvm.py`** – Production lowering path: IR → AMDGPU LLVM IR
  - Target ISA: `amdgcn-amd-amdhsa--gfx950` (default)
  - Buffer-resource descriptors use `flags=0x00027000` for OOB-safe reads
  - This is the ONLY path used at runtime

- **`lower_hip.py`** – Debug/inspection backend: IR → readable HIP C++
  - NOT used in production runtime
  - Narrower op coverage than LLVM lowering
  - Useful for human inspection and debugging

- **`lower_cktile.py`** – CK Tile C++ emission for parity/reference
  - Consumes high-level specs (`UniversalGemmSpec`, `ImplicitGemmConvSpec`)
  - Does NOT consume `KernelDef`
  - Used for parity validation only

- **`ir_print.py`** – MLIR-style textual IR printer (human inspection only)

- **`passes.py`** – Conservative optimization passes on IR

- **`arch/`** – Architecture-specific metadata (MFMA atoms, ISA details)

- **`isa/`** – ISA encoding and intrinsic definitions

### 2. Runtime Layer (`runtime/`)
**Purpose:** In-process HSACO build, HIP module load, kernel launch

- **`comgr.py`** – `libamd_comgr` ctypes wrapper
  - Compiles LLVM IR → HSACO bytes in-process
  - Returns `ComgrTimings` for codegen profiling

- **`hip_module.py`** – `libamdhip64` ctypes wrapper
  - `Runtime` class: loads HSACO via `hipModuleLoadData`
  - Manages HIP module lifecycle

- **`launcher.py`** – Kernel execution and timing
  - `KernelLauncher`: persistent HSACO loader, called repeatedly
  - `PipelineLauncher`: chains multi-stage kernels on one stream
  - `WorkspacePool`: long-lived torch workspace management
  - `time_launches`: canonical HIP-event timer

- **`torch_module.py`** – PyTorch argument packing utilities

### 3. Helpers Layer (`helpers/`)
**Purpose:** High-level kernel authoring abstractions

Key components:
- **`compile.py`** – One-shot `IR → LLVM IR → HSACO` pipeline (`compile_kernel`)
- **`manifest.py`** – Kernel manifest schema for batch execution
- **Geometry & layout:**
  - `MfmaAtom`, `mfma_atom`: Matrix-multiply accumulate primitives
  - `WarpGrid`: Warp-level tiling configuration
  - `LdsLayout`: LDS (shared memory) layout descriptors
  - `TensorDescriptor`: Coordinate-transform DAG (see `transforms.py`)
- **Data movement:**
  - `CoalescedTileLoader`: Coalesced global → register tile loads
  - `AsyncTileLoader`: Async global → LDS tile loads
  - `AsyncTileLoaderSlot`: Multi-buffered async load slots
- **Epilogues:**
  - `DirectEpilogue`: Direct register → global writeback
  - `CShuffleEpilogue`: C-shuffle pattern for non-coalesced writes
- **Scheduling:**
  - `SchedulePolicy`: Loop scheduling policies
  - `SoftwarePipeline`: Software pipelining helpers
- **Attention:**
  - `mfma_attention.py`, `mfma_attention_bwd.py`: Attention kernel helpers
  - `attention.py`: 2D/3D attention configs
- **Fusion:**
  - `fusion_ir.py`, `fusion_lowering.py`, `fusion_scheduler.py`: Kernel fusion infrastructure
  - `fusion_legalize.py`, `fusion_memory.py`, `fusion_validation.py`: Fusion validation
  - `fuse.py`: High-level fusion API
- **Utilities:**
  - `activations.py`: Activation functions
  - `autotune.py`: Autotuning infrastructure
  - `codebook.py`: Quantization codebooks
  - `distribution.py`: Data distribution helpers
  - `gather_scatter.py`: Gather/scatter operations
  - `i4_dequant.py`: INT4 dequantization
  - `io.py`: I/O utilities
  - `mx_scale.py`: MX-format scaling
  - `persistent.py`: Persistent kernel helpers
  - `pipeline.py`: Pipeline construction

### 4. Analysis & Benchmark (`analysis/`, `benchmark/`)
**Purpose:** LLVM IR, ISA, and HSACO inspection; performance measurement

- **`analysis/ir.py`** – LLVM IR statistics extraction
- **`analysis/isa.py`** – ISA parsing and instruction counting
- **`analysis/report.py`** – Variant comparison reports
- **`benchmark/summary.py`** – Repeated-run summaries (median, spread)

### 5. Instances (`instances/`)
**Purpose:** Parametric kernel builders for common operations

- **`common/`** – Shared instance utilities
  - `conv_implicit_gemm.py`: Implicit GEMM convolution builder
  - `gemm_universal.py`: Universal GEMM builder

- **`gfx950/`** – gfx950-specific instances
  - `attention_tiled_2d.py`: 2D tiled attention
  - `attention_tiled_2d_fastkv_regp.py`: Fast KV 2D attention variant
  - `attention_tiled_3d.py`: 3D tiled attention (paged KV-cache)
  - `deep_fused_conv_pool.py`: Deep-fused conv+pool prototype

- **`gfx1151/`**, **`gfx942/`** – Other architecture variants

### 6. Examples (`examples/`)
**Purpose:** Reference implementations and validation harnesses

- **`common/`** – Shared example infrastructure
- **`gfx950/`** – gfx950 example kernels
  - `deep_conv_fusion/`: Deep fusion convolution examples
  - `gemm_perf_*/`: GEMM performance case studies
  - `attention/`: Attention examples
  - `moe/`: Mixture-of-experts examples
  - `qwen3_30b_a3b/`: Qwen model examples

### 7. Documentation (`dsl_docs/`)
**Purpose:** Comprehensive implementation and usage documentation

See [dsl_docs/README.md](dsl_docs/README.md) for the full reading order.

Key sections:
- **`architecture/`** – Mental models, authoring patterns, design decisions
- **`ir_lowering/`** – IR model, lowering pipeline, backend details
- **`primitives/`** – Intrinsics, memory layouts, wave operations, quantization
- **`instances/`** – Instance-specific documentation (GEMM, conv, attention)
- **`runtime/`** – Compile, launch, manifest schema, limitations
- **`optimization/`** – Optimization runbook, measured results, compliance
- **`fusion/`** – Kernel fusion overview
- **`autotune/`** – Autotuning infrastructure
- **`development/`** – Testing, extending the DSL
- **`reference/`** – API index, file index, glossary, MFMA atom catalog

## Compilation Pipeline

```
Spec dataclass (e.g., UniversalGemmSpec)
  └─> Instance builder (e.g., build_universal_gemm)
      └─> KernelDef SSA IR (core/ir.py)
          ├─> Optional passes (core/passes.py)
          ├─> MLIR-style text (core/ir_print.py, inspection only)
          └─> AMDGPU LLVM IR (core/lower_llvm.py)
              └─> libamd_comgr (runtime/comgr.py)
                  └─> HSACO bytes
                      └─> hipModuleLoadData → hipModuleLaunchKernel
                          (runtime/hip_module.py + launcher.py)
```

**Critical path:** LLVM IR text → libamd_comgr → HSACO → hipModule

There is **NO** MLIR pipeline at runtime. `print_ir()` emits MLIR-style text for humans only.

## Common Workflows

### 1. Write and compile a kernel

```python
from ck_dsl import (
    IRBuilder, F16, F32, I32,
    compile_kernel, MfmaAtom, WarpGrid,
    CoalescedTileLoader, DirectEpilogue,
    make_gemm_manifest, write_artifact,
)

# Build kernel IR
kernel = build_my_kernel(...)

# Compile to HSACO
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")

# Save artifact
write_artifact(artifact, "output/my_kernel")
```

### 2. Run a kernel from manifest

```bash
python -m ck_dsl.run_manifest output/my_kernel.hsaco output/manifest.json --verify
```

### 3. Benchmark a sweep

```python
from ck_dsl import benchmark_manifest

results = benchmark_manifest(
    hsaco_path="output/sweep.hsaco",
    manifest_path="output/sweep_manifest.json",
    warmup=10,
    iterations=100
)
```

### 4. Analyze generated code

```python
from ck_dsl import analyze_hsaco, analyze_llvm_ir

# Analyze LLVM IR
ir_stats = analyze_llvm_ir(artifact.llvm_text)
print(f"VGPR usage: {ir_stats.vgpr_count}")

# Analyze HSACO
hsaco_stats = analyze_hsaco(artifact.hsaco)
print(f"ISA instruction count: {hsaco_stats.isa_stats.total_instructions}")
```

## Kernel Optimization Using `optimization/utilities`

Use `dsl_docs/optimization/optimization_runbook.md` as the full checklist and
`dsl_docs/optimization/runbook_mapping.md` to map each optimization idea to a
specific CK DSL primitive. The helper material under
`dsl_docs/optimization/utilities/` is the practical toolbox: focused skills in
`utilities/skills/` and runnable probes/scripts in `utilities/tools/`.

The optimization loop is:

1. State the hypothesis and exact shape/layout/dtype contract.
2. Verify the correctness baseline before timing.
3. Measure with stable timing (`benchmark_manifest`, `sweep_bench`, or
   `time_launches`) and report median plus spread.
4. Inspect generated IR, ISA, and resources before changing code.
5. Change one lever, re-verify, re-measure, and explain the movement.
6. Keep or revert the change, then record the result.

Start with the focused skill docs when the bottleneck is known:

- `utilities/skills/gemm-optimization-ckdsl.md` for tile geometry, MFMA atom,
  pipeline, epilogue, LDS, and occupancy tradeoffs.
- `utilities/skills/lds-optimization-ckdsl.md` for LDS padding, swizzles,
  bank conflicts, and `ds_read`/`ds_write` behavior.
- `utilities/skills/prefetch-data-load-ckdsl.md` for async global-to-LDS
  staging, wait counts, and data-load overlap.
- `utilities/skills/capture-kernel-trace-ckdsl.md` and
  `utilities/skills/kernel-trace-analysis.md` for rocprof/ATT workflows.
- `utilities/skills/bisect-perf-regression.md` when a performance change is
  suspected but the responsible commit or lever is unclear.

Use the DSL probes before paying for full profiling. From
`projects/composablekernel/python`:

```bash
export PYTHONPATH=.

python ck_dsl/dsl_docs/optimization/utilities/tools/dsl_probes/probe_occupancy.py \
  --demo attention_tiled_2d
python ck_dsl/dsl_docs/optimization/utilities/tools/dsl_probes/probe_isa_inspect.py \
  --demo attention_tiled_2d
python ck_dsl/dsl_docs/optimization/utilities/tools/dsl_probes/probe_intrinsic_counts.py \
  --demo attention_tiled_2d
```

The most useful probes answer narrow questions:

- `probe_occupancy.py`: VGPR/AGPR/SGPR/LDS usage and the apparent occupancy
  limiter.
- `probe_intrinsic_counts.py`: whether expected AMDGCN intrinsics were emitted
  before COMGR/HSACO generation.
- `probe_isa_inspect.py`: MFMA, vector memory, LDS, waitcnt, SALU, and VALU
  opcode mix from disassembly.
- `probe_config_sweep.py`: dataclass-based spec sweeps with optional latency
  measurement supplied by your harness.
- `probe_targeted_bench.py`: single-window CUDA-event comparison against a
  baseline for production-shaped inputs.
- `probe_rocprof_single.py`: steady-state rocprof runs with build and warmup
  outside the profiled window.

After a static probe narrows the candidate set, use rocprof-driven tools under
`utilities/tools/stage4_analyze/` and `utilities/tools/stage5_compare/` to
analyze LDS conflicts, prefetch efficiency, kernel traces, and counter deltas.
ATT trace analysis requires `rocprof-trace-decoder`; if it is unavailable, use
the PMC profiling path documented in
`utilities/skills/capture-kernel-trace-ckdsl.md`.

Never report a speedup unless correctness still passes for the same shape set.
Record the operation contract, spec overrides, timing method, IR/ISA/resource
signals, profiler counters, and final keep/revert decision.

## Target Hardware

- **Primary target:** AMD Instinct MI355X (gfx950)
- **Also supports:** gfx940, gfx942
- **Wave size:** 64 (fixed)
- **MFMA atoms:** See [dsl_docs/reference/mfma_atom_catalog.md](dsl_docs/reference/mfma_atom_catalog.md)

### Known Hardware Constraints

- **Async tile loader:** Only supports `{4, 3, 1}` dwords (AMDGPU `raw_ptr_buffer_load_lds` intrinsic limitation)
- **Buffer descriptors:** Use `flags=0x00027000` for OOB-safe loads (out-of-bounds lanes return zero)
- **Cache coherency hints:** `CACHE_ALL (0)`, `CACHE_GLOBAL (1)`, `CACHE_STREAM (2)`, `NON_TEMPORAL (3)`

## GPU Access Requirements

When running on AMD Instinct MI355X (gfx950):

- User must be member of groups that can access `/dev/kfd` (GID 506) and `/dev/dri/renderD144` (GID 109)
- On this host: `kfdhost(506)`, `renderhost(109)`, and `video(44)`
- **Symptom of missing groups:** `hipMalloc` returns code 100 (`hipErrorOutOfMemory`) on all allocations
- **Fix:** Add user to required groups, then launch via `sudo -u <user> bash -lc '...'` to refresh group membership

## Validation & Testing

Run from repository root:

```bash
export PYTHONPATH=python

# Unit tests
PYTHONDONTWRITEBYTECODE=1 python python/test/test_ck_dsl.py
PYTHONDONTWRITEBYTECODE=1 python python/test/test_ck_dsl_examples.py

# Example validation
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
python -m ck_dsl.examples.common.bake_off_implicit_gemm --output-dir "$OUT_DIR"
python -m ck_dsl.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
```

See [dsl_docs/development/testing.md](dsl_docs/development/testing.md) for detailed procedures.

## Remote GPU Testing (slurm harness + standalone boards)

Architectures the local host can't run are validated on remote hardware. When the
local GPU matches the target arch, run directly; otherwise use the remote paths
below. The gfx1151 vs. gfx1201 split is **deliberate** — do not run gfx1151
through slurm:

- **gfx1201 (RDNA4 / Navi 48):** alola slurm cluster, via the remote-test harness.
- **gfx1151 (RDNA3.5 / Strix Halo):** standalone Windows board (not alola).
- **gfx942 / gfx950 (CDNA):** alola slurm, or locally when the local GPU matches.

### Harness layout (`ck_dsl/benchmark/remote_test/`)

Build an HSACO locally, rsync it (plus a slim copy of `ck_dsl/`) to the alola
login node, then `srun` `run_manifest --verify` inside a ROCm docker container on
a compute node matching an arch feature constraint (`GFX1201&MARKHAM`, etc.).

- `config.py` — `ARCHES` profiles (example module, constraint, time). Reads env:
  `CKDSL_REMOTE_HOST`, `CKDSL_DOCKER_IMAGE`, `CKDSL_DOCKER_EXTRA_FLAGS`,
  `CKDSL_SLURM_EXTRA`. Local stage root `/tmp/ckdsl_remote/<arch>/`.
- `slurm.py` — `run_arch(arch)` is the real entry point. Reads
  `<stage>/run_spec.json`, pushes the tree + artifacts, builds the `srun` command
  (rocminfo preflight exits 42 if no GPU agent), streams output back.
- `transport.py`, `persistent.py` — rsync/ssh helpers and long-lived
  (sbatch-holder) allocations for scarce nodes.

> **The `cli.py` front-end is currently broken** (`from . import build` fails —
> `build.py` is untracked). Do **not** use `python -m ...remote_test`. Drive
> `slurm.run_arch` directly until `build.py` is restored.

### Running gfx1201 on alola (working path)

```bash
source ~/.ckdsl_env        # CKDSL_REMOTE_HOST=ckdsl-login, docker image, etc.
export PYTHONPATH=$(pwd)
python3 - <<'PY'
import json
from pathlib import Path
from ck_dsl.helpers import compile_kernel, make_gemm_manifest, write_artifact
from ck_dsl.instances.gfx1201.wmma_gemm import WmmaGemmSpec, build_wmma_gemm
from ck_dsl.benchmark.remote_test import slurm

arch, shape = "gfx1201", (256, 512, 128)
spec = WmmaGemmSpec(name=f"wmma_gemm_{arch}")
art = compile_kernel(build_wmma_gemm(spec, arch=arch), arch=arch)
out = Path(f"/tmp/ckdsl_remote/{arch}"); out.mkdir(parents=True, exist_ok=True)
manifest = make_gemm_manifest(artifact=art, block_m=16, block_n=16, block_k=16,
    threads_per_block=spec.block_size, default_shape=shape, grid_order="MN",
    atoms=["wmma_gfx12_f32_16x16x16_f16"])
write_artifact(art, out, manifest)
(out / "run_spec.json").write_text(json.dumps(
    {"shape": {"m": shape[0], "n": shape[1], "k": shape[2]},
     "hsaco": f"{art.kernel_name}.hsaco", "manifest": "manifest.json"}))
print("rc =", slurm.run_arch(arch))   # PASS => "verify max_abs_diff=0 bad=0/N"
PY
```

`run_arch` runs exactly **one** shape (from `run_spec.json`) per call — rewrite
`run_spec.json` and call again to sweep. Harmless rsync warnings
(`cannot delete non-empty directory: .venv`) may appear; ignore them.

### Running gfx1151 on the standalone board

Build a `gfx11-generic` HSACO locally (the board reports a cosmetic stepping the
toolchain can't name), `scp` the `.hsaco` + `manifest.json` to the board, and run
`ck_dsl.run_manifest --verify` there with
`CK_DSL_HIP_LIB=C:\Windows\System32\amdhip64_7.dll`. See the board's resident
sweep driver for the canonical shape list (square + non-square M≠N coverage).

```bash
python3 -m ck_dsl.examples.gfx1151.wmma_gemm_verify \
  --arch gfx11-generic --m 256 --n 512 --k 128 --no-verify --output-dir <dir>
```

## Entry Points

### CLI Entry Points

1. **Top-level CLI:** `python -m ck_dsl`
   - Prints available runnable modules

2. **Manifest runner:** `python -m ck_dsl.run_manifest <hsaco> <manifest.json> [--verify]`
   - Loads HSACO and executes all manifest entries
   - Optional verification against reference

3. **Sweep bench:** `python -m ck_dsl.sweep_bench <sweep_manifest.json> [--csv ...]`
   - Benchmarks multiple kernel configurations

4. **Example generators:**
   ```bash
   python -m ck_dsl.examples.common.bake_off_implicit_gemm --output-dir <dir>
   python -m ck_dsl.examples.common.bake_off_direct_conv_16c --output-dir <dir>
   python -m ck_dsl.examples.common.bake_off_direct_conv_4c --output-dir <dir>
   ```

### PyTorch Integration

```python
# torch.compile backend
import torch
from ck_dsl.torch_backend import compile as ck_dsl_compile

torch._dynamo.config.disable = False
torch.compile(model, backend="ck_dsl")
```

## Important Design Constraints

1. **SSA values cannot drive Python control flow**
   - `Value.__bool__` raises by design
   - Use `IRBuilder.static_if(...)` for Python booleans
   - Use `IRBuilder.scf_if(...)` for runtime predicates

2. **No MLIR at runtime**
   - Production path: IR → LLVM IR → HSACO
   - MLIR-style text is for inspection only

3. **Buffer-safe by default**
   - OOB lanes return zero (descriptor encoding)
   - Canonical tail-safe primitive for conv and attention

4. **Persistent runtime**
   - `KernelLauncher` loads HSACO once, reused for multiple launches
   - `WorkspacePool` maintains long-lived allocations

## File Organization Reference

```
ck_dsl/
├── __init__.py              # Package exports
├── __main__.py              # CLI entry point
├── run_manifest.py          # Manifest execution
├── sweep.py                 # Parallel sweep driver
├── sweep_bench.py           # Benchmark sweep runner
├── torch_backend.py         # torch.compile backend
├── core/                    # SSA IR + lowering
│   ├── ir.py               # IR definitions
│   ├── ir_print.py         # Text printer
│   ├── lower_llvm.py       # → LLVM IR (production)
│   ├── lower_hip.py        # → HIP C++ (debug)
│   ├── lower_cktile.py     # → CK Tile C++ (parity)
│   ├── passes.py           # Optimization passes
│   ├── arch/               # Architecture metadata
│   └── isa/                # ISA encodings
├── runtime/                 # HSACO build + launch
│   ├── comgr.py            # libamd_comgr wrapper
│   ├── hip_module.py       # libamdhip64 wrapper
│   ├── launcher.py         # Kernel launch + timing
│   └── torch_module.py     # PyTorch utilities
├── helpers/                 # High-level authoring
│   ├── compile.py          # compile_kernel entry point
│   ├── manifest.py         # Manifest schema
│   ├── atoms.py            # MFMA atom definitions
│   ├── geometry.py         # WarpGrid, tiling
│   ├── layouts.py          # LDS layouts
│   ├── loads.py            # Tile loaders
│   ├── epilogues.py        # Output epilogues
│   ├── pipeline.py         # Software pipelining
│   ├── transforms.py       # Coordinate transforms
│   ├── attention*.py       # Attention helpers
│   ├── fusion_*.py         # Fusion infrastructure
│   └── ...
├── analysis/                # IR/ISA inspection
│   ├── ir.py               # LLVM IR stats
│   ├── isa.py              # ISA parsing
│   └── report.py           # Variant reports
├── benchmark/               # Performance measurement
│   └── summary.py          # Benchmark summaries
├── instances/               # Kernel builders
│   ├── common/             # Shared builders
│   ├── gfx950/             # gfx950 instances
│   ├── gfx942/             # gfx942 instances
│   └── gfx1151/            # gfx1151 instances
├── examples/                # Reference implementations
│   ├── common/             # Shared examples
│   ├── gfx950/             # gfx950 examples
│   └── data/               # Test data
└── dsl_docs/                # Documentation
    ├── README.md           # Documentation index
    ├── architecture/       # Design & mental models
    ├── ir_lowering/        # Lowering details
    ├── primitives/         # Primitive operations
    ├── instances/          # Instance docs
    ├── runtime/            # Runtime docs
    ├── optimization/       # Optimization guide
    ├── fusion/             # Fusion docs
    ├── autotune/           # Autotuning docs
    ├── development/        # Development guide
    ├── reference/          # API reference
    └── hipdnn_provider/    # HIP DNN provider
```

## Getting Started

1. **New to CK DSL?** Read [dsl_docs/README.md](dsl_docs/README.md) for the recommended reading order
2. **Want to write a kernel?** Start with [dsl_docs/architecture/authoring_model.md](dsl_docs/architecture/authoring_model.md)
3. **Want to optimize?** See [dsl_docs/optimization/optimization_runbook.md](dsl_docs/optimization/optimization_runbook.md)
4. **Need API reference?** Check [dsl_docs/reference/api_index.md](dsl_docs/reference/api_index.md)

## Quick Reference

### Common Imports

```python
from ck_dsl import (
    # IR + types
    IRBuilder, F16, F32, I32, BF16, I8, I16, I64,
    # Helpers
    compile_kernel, MfmaAtom, mfma_atom, WarpGrid,
    CoalescedTileLoader, AsyncTileLoader,
    DirectEpilogue, CShuffleEpilogue,
    make_gemm_manifest, make_conv_manifest, write_artifact,
    # Transforms
    TensorDescriptor, unmerge, embed, pad, merge, pass_through,
)
```

### Typical Kernel Structure

```python
builder = IRBuilder()
kernel = builder.kernel_def(
    name="my_kernel",
    grid_size=(grid_x, grid_y, grid_z),
    block_size=(block_x, block_y, block_z),
    signature=signature,
)

with builder.region(kernel.body):
    # Allocate LDS
    lds = builder.smem_alloc(elem=F16, shape=(lds_rows, lds_cols))
    
    # Create MFMA atom and warp grid
    atom = mfma_atom("mfma_f32_16x16x16_f16")
    grid = WarpGrid(warp_m=2, warp_n=2, atom=atom)
    
    # Tile loading
    loader = CoalescedTileLoader(...)
    tile_data = loader.load(...)
    
    # Compute
    result = builder.mfma(...)
    
    # Epilogue
    epilogue = DirectEpilogue(...)
    epilogue.store(result, ...)
    
    builder.return_()

return kernel
```

## Contact & Support

This is an internal AMD project. For issues or questions:
- Check [dsl_docs/](dsl_docs/) for detailed documentation
- Run validation tests (see [Validation & Testing](#validation--testing))
- Review examples in [examples/](examples/)

## License

Copyright (c) Advanced Micro Devices, Inc., or its affiliates.  
SPDX-License-Identifier: MIT
