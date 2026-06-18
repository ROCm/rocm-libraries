# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`ck_dsl` is a Python authoring layer for CK Tile-style GPU kernels on AMDGPU. It builds a typed SSA IR in Python, lowers directly to AMDGPU LLVM IR text, compiles to HSACO in-process via `libamd_comgr`, and launches through the HIP runtime. No `hipcc`, no MLIR, no C++ template metaprogramming.

## Build & Run

All commands run from the composablekernel repository root.

```bash
# Set up Python path (required for all commands)
export PYTHONPATH=python

# Static tests (no GPU required) — IR construction, lowering, transforms
PYTHONDONTWRITEBYTECODE=1 python python/test/test_ck_dsl.py

# Run a single test class or method
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python -m unittest test.test_ck_dsl.TestCoreIR
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python -m unittest test.test_ck_dsl.TestCoreIR.test_method_name

# GPU runtime tests (requires ROCm GPU)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python python/test/test_ck_dsl_examples.py

# Multi-architecture tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python python/test/test_ck_dsl_multiarch.py

# Numeric/correctness tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python python/test/test_ck_dsl_numeric.py

# IR parity check against golden reference
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:python/test \
  python -m ck_dsl_ir_parity_harness --compare test/golden/ck_dsl_representative_ir_sha256.json

# Build and verify a generated kernel
OUT_DIR=$(mktemp -d)
PYTHONPATH=python python -m ck_dsl.examples.common.bake_off_implicit_gemm --output-dir "$OUT_DIR"
PYTHONPATH=python python -m ck_dsl.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
```

Test framework is `unittest` (no pytest). GPU tests auto-skip when no GPU is available.

## Architecture

The compilation pipeline flows through four layers:

```
instances/  →  helpers/  →  core/ir.py  →  core/lower_llvm.py  →  runtime/comgr.py  →  runtime/hip_module.py
(Spec+build)   (authoring)   (SSA IR)      (LLVM IR text)         (HSACO bytes)         (GPU launch)
```

**Key mental model:** Python builds a typed SSA `KernelDef` object — it is NOT string generation. `Value` objects are first-class IR nodes with type checking. `Value.__bool__` raises `TypeError` to prevent SSA values from accidentally controlling Python flow; use `static_if(python_bool)` for compile-time decisions and `scf_if(ssa_value)` for runtime predicates.

### Layer Details

- **`core/`** — IR type system (`ir.py`), LLVM IR lowering (`lower_llvm.py` — production path), HIP C++ lowering (`lower_hip.py` — debug/readable output), CK Tile C++ lowering (`lower_cktile.py` — parity reference), conservative optimization passes (`passes.py` — constant fold, CSE, DCE), architecture targets (`arch/target.py`)

- **`helpers/`** — High-level kernel-authoring abstractions: `TensorDescriptor` and coordinate transforms (`transforms.py`), `MfmaAtom` catalog (`atoms.py`), `WarpGrid` block/warp/lane decomposition (`geometry.py`), tile loaders (`loads.py`), LDS layouts (`layouts.py`), scheduling policies (`schedule.py`), software pipelining (`pipeline.py`), epilogues (`epilogues.py`), `compile_kernel()` one-call compilation (`compile.py`), manifest generation (`manifest.py`)

- **`instances/`** — Parametric kernel builders: GEMM family (`gemm_universal.py`, batched, grouped, streamk, block-scale, MX), convolution (`conv_implicit_gemm.py`, `conv_direct_grouped.py`), attention (`attention_unified.py`, FMHA variants, paged KV, split-KV), small ops (layernorm, rmsnorm, reduce, transpose, pooling, elementwise), MoE infrastructure, quantization variants

- **`runtime/`** — `comgr.py` wraps `libamd_comgr` via ctypes for LLVM IR → HSACO compilation. `hip_module.py` wraps `libamdhip64` for module load and kernel launch. `launcher.py` provides `KernelLauncher` (persistent compiled kernel) and `time_launches()` (HIP-event benchmarking)

- **`dispatch/`** — Operator-to-kernel selection: request/candidate/registry pattern for choosing kernel configs at runtime

- **`analysis/`** — LLVM IR and ISA inspection: `analyze_llvm_ir()`, `analyze_hsaco()`, `parse_isa()`, `parse_resources()`

### Instance Pattern

Every instance module follows the same contract:

```python
@dataclass
class FooSpec:     # tile dimensions, traits, data types
    ...

def build_foo(spec: FooSpec) -> KernelDef:    # build the IR
    ...

def foo_signature(spec: FooSpec) -> dict:     # kernel argument layout
    ...

def foo_grid(spec: FooSpec, problem) -> tuple: # grid dimensions
    ...
```

The typical workflow: `Spec → build_*() → KernelDef → compile_kernel() → KernelArtifact`

### Lowering Backends

- **`lower_kernel_to_llvm`** — Production path. Generates AMDGPU LLVM IR text. Auto-detects LLVM 20 (ROCm 7.0) vs LLVM 22+ (ROCm 7.2+) for datalayout compatibility.
- **`lower_kernel_to_hip`** — Debug path. Readable HIP C++ output. Narrower op coverage.
- **`lower_spec_to_cktile`** — Parity reference. Emits CK Tile C++. Does NOT consume KernelDef (operates on specs directly).

### Architecture Targets

Primary target: `gfx950` (CDNA3+, MI355X). Also tested: `gfx940`, `gfx942` (CDNA2), `gfx1151`, `gfx1201`. Wave size is fixed at 64.

## Documentation

Comprehensive docs live in `dsl_docs/` (44 markdown files). Key reading order:
- `dsl_docs/architecture/mental_model.md` — critical conceptual foundation
- `dsl_docs/runtime/compile_launch_and_manifest.md` — compile-to-launch pipeline
- `helpers/README.md` — comprehensive helpers reference
- `instances/SUPPORT_MATRIX.md` — instance support across architectures
