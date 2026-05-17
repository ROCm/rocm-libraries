# CK DSL Documentation

This folder is a deep, code-adjacent guide to `ck_dsl`, the Python authoring layer for CK Tile-style GPU kernels on AMDGPU. The package README is the quick tour. These notes are the field manual: how kernels are described, how the Python SSA IR works, how the lowering stack maps DSL operations to AMDGPU LLVM IR and HSACO, what each primitive means, how the shipped instances execute step by step, what limits matter, and how to validate changes.

The implementation tree is:

```text
python/ck_dsl/
├── core/ # SSA IR, IR printer, conservative passes, LLVM/HIP/CK Tile lowering
├── runtime/ # libamd_comgr + libamdhip64 ctypes; launcher, workspace, timing
├── helpers/ # CK Tile-like authoring helpers and the high-level compile entrypoint
├── analysis/ # LLVM IR + HSACO/ISA + resource inspection
├── benchmark/ # repeated-run benchmark summaries (median, spread)
├── instances/ # spec-driven kernel builders (gemm, conv, attention, small ops)
├── examples/ # Python-owned example generators and parity harnesses
├── transforms.py # coordinate-transform DAG (pad/embed/unmerge/merge/indirect)
├── run_manifest.py # python -m ck_dsl.run_manifest (HSACO + manifest runner)
├── sweep.py # parallel build-on-the-fly sweep driver
├── sweep_bench.py # benchmark driver over a sweep manifest
└── torch_backend.py # torch.compile backend (ck_dsl.compile)
```

## Reading Order

New to the DSL? Read in this order:

1. `architecture/mental_model.md`
2. `architecture/authoring_model.md`
3. `ir_lowering/ir_model.md`
4. `ir_lowering/lowering_pipeline.md`
5. `ir_lowering/backend_details.md`
6. `primitives/intrinsics_and_primitives.md`
7. `primitives/memory_layout_and_transforms.md`
8. `primitives/wave_and_cross_lane.md`
9. `primitives/quantization.md`
10. `instances/index.md`
11. `instances/gemm.md`
12. `instances/convolution.md`
13. `instances/attention.md`
14. `instances/small_ops.md`
15. `runtime/compile_launch_and_manifest.md`
16. `runtime/manifest_schema.md`
17. `runtime/comgr_and_hipmodule.md`
18. `runtime/limitations.md`
19. `optimization/runbook_mapping.md`
20. `optimization/optimization_playbook.md`
21. `optimization/measured_results.md`
22. `fusion/overview.md`
23. `autotune/overview.md`
24. `development/testing.md`
25. `development/extending.md`
26. `reference/file_index.md`
27. `reference/api_index.md`
28. `reference/op_vocabulary.md`
29. `reference/mfma_atom_catalog.md`
30. `reference/glossary.md`

## One-Screen Summary

`ck_dsl` lets a kernel author write Python objects and IR-builder calls instead of C++ template metaprogramming. A typical path is:

```text
spec dataclass
 └─> instance builder
 └─> KernelDef SSA IR (core/ir.py)
 ├─> optional passes (core/passes.py)
 ├─> MLIR-style text (core/ir_print.py, for inspection only)
 └─> AMDGPU LLVM IR (core/lower_llvm.py)
 └─> libamd_comgr (runtime/comgr.py)
 └─> HSACO bytes
 └─> hipModuleLoadData -> hipModuleLaunchKernel
 (runtime/hip_module.py + launcher.py)
```

Hard facts:

- The production compile path is **LLVM IR text -> libamd_comgr -> HSACO -> hipModule**. There is no MLIR pipeline at runtime. `print_ir()` emits MLIR-style text for humans only.
- `core/lower_hip.py` (`lower_kernel_to_hip`) is a debugging/inspection backend that emits readable HIP C++. It is not the production runtime path. Op coverage is narrower than LLVM lowering.
- `core/lower_cktile.py` emits CK Tile C++ from selected high-level specs (`UniversalGemmSpec`, `ImplicitGemmConvSpec`). It does not consume `KernelDef`. It exists for parity/reference.
- Default target ISA is `amdgcn-amd-amdhsa--gfx950` (CDNA3+). The DSL also runs on `gfx940/gfx942` where the chosen MFMA atoms and waitcnt encoding are valid. The datalayout in `core/lower_llvm.py::_DATALAYOUT` is the clang-emitted gfx950 string.
- Wave size is fixed at 64 in current MFMA lane mappings and helpers.
- Kernel authors usually compose helpers (`TensorDescriptor`, `TensorView`, `TileWindow`, `MfmaAtom`, `WarpGrid`, `CoalescedTileLoader`, `AsyncTileLoader`, `SchedulePolicy`, `SoftwarePipeline`, `DirectEpilogue`, `CShuffleEpilogue`, `block_lds_reduce`, `sweep_row_chunks`).
- Non-bijective addressing (convolution, paged attention, indirection) is expressed with the transform DAG in `transforms.py`.
- Runtime is persistent: `KernelLauncher` loads HSACO once and is called repeatedly. `PipelineLauncher` chains stages on one stream. `WorkspacePool` keeps long-lived torch workspaces alive across launches. `time_launches` is the canonical HIP-event timer.
- Buffer-resource descriptors are constructed with `flags=0x00027000` (= 159744). The DW3 encoding is TYPE=2 (BUFFER_RESOURCE), DATA_FORMAT=4 (32-bit dword), NUM_FORMAT=4 (UINT). With this, OOB lanes silently return zero — the canonical tail-safe primitive used by conv and attention.
- `AsyncTileLoader.choose_dwords` selects `{4, 3, 1}` only (the AMDGPU `raw_ptr_buffer_load_lds` intrinsic on this target does not accept 2 dwords).
- `Value.__bool__` raises by design. SSA values cannot drive Python branches; use `IRBuilder.static_if(...)` for Python booleans and `IRBuilder.scf_if(...)` for runtime predicates.

## Validation Status

The docs in this folder were written against the current code and validated against the live repo using the following commands (see `development/testing.md` for the full procedure and measured results):

```bash
# Static, in-process unit suite (no GPU required).
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
 /workspace/dsl_bake_off/venv/bin/python python/test/test_ck_dsl.py
# -> 359 tests, OK (~1.8 s)
# (count grows with each wave; +10 in for fused MoE,
# +28 in for FMHA expansion + infra,
# +20 in for Sage attention + sparse attention + infra)

# Generated example harness (builds HSACO, runs HIP).
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
 /workspace/dsl_bake_off/venv/bin/python python/test/test_ck_dsl_examples.py
# -> 1 test (multi-subtest), OK (~204 s)

# Read-style implicit-GEMM conv repro: build HSACO then verify.
PYTHONPATH=python /workspace/dsl_bake_off/venv/bin/python \
 -m ck_dsl.examples.bake_off_implicit_gemm --output-dir /tmp/ex08
PYTHONPATH=python /workspace/dsl_bake_off/venv/bin/python \
 -m ck_dsl.run_manifest /tmp/ex08/*.hsaco /tmp/ex08/manifest.json --verify
# -> verify max_abs_diff=7.6e-06 bad=0/1605632 Perf: 230.61 TFlops

# Distribution-driven helpers (validates TileDistributionEncoding path).
PYTHONPATH=python /workspace/dsl_bake_off/venv/bin/python \
 python/ck_dsl/examples/distribution_reduce_demo.py --M 32 --N 4096
PYTHONPATH=python /workspace/dsl_bake_off/venv/bin/python \
 python/ck_dsl/examples/distribution_2d_add_demo.py --H 64 --W 128
# -> max_abs=0.000e+00 (both)

# CK Tile parity (small-op family vs torch reference).
PYTHONPATH=python /workspace/dsl_bake_off/venv/bin/python \
 python/ck_dsl/examples/ck_tile_parity.py --op all
# -> all OK on MI355X (see optimization/measured_results.md for the table)

# Attention smoke (CK DSL vs AITER Triton, 2D + 3D paths).
PYTHONPATH=python:/workspace/aiter /workspace/dsl_bake_off/venv/bin/python \
 python/ck_dsl/examples/attention/parity_unified_attention.py \
 --scenario decode_d128_b16 --attempts 1 --warmup 0 --paths auto,2d,3d
# -> max_abs=0.0001831 in all lanes; CK 3D 132.88us vs Triton 3D 203.52us
```

Numbers are MI355X / gfx950 / ROCm 7.0.2 + torch 2.8.0+rocm7.0.2.

## Source Material

These docs are written against the current code. When this file and code disagree, trust code. The most important source files are listed in `reference/file_index.md`.

Conventional anchors:

- Package re-exports: `python/ck_dsl/__init__.py`, `python/ck_dsl/helpers/__init__.py`.
- IR + builder: `python/ck_dsl/core/ir.py`.
- Production lowering: `python/ck_dsl/core/lower_llvm.py`.
- HIP debug lowering: `python/ck_dsl/core/lower_hip.py`.
- CK Tile parity emission: `python/ck_dsl/core/lower_cktile.py`.
- Conservative passes: `python/ck_dsl/core/passes.py`.
- COMGR: `python/ck_dsl/runtime/comgr.py`.
- HIP runtime: `python/ck_dsl/runtime/hip_module.py`.
- Launcher / workspace / timing: `python/ck_dsl/runtime/launcher.py`.
- Torch arg packing: `python/ck_dsl/runtime/torch_module.py`.
- High-level compile: `python/ck_dsl/helpers/compile.py`.
- Manifest schema: `python/ck_dsl/helpers/manifest.py`.
- Optimization runbook: `/workspace/OPTIMIZATION_RUNBOOK (1).md`.
- DSL runbook compliance table: `python/ck_dsl/RUNBOOK_COMPLIANCE.md`.
- Coordinate-transform DAG walkthrough: `python/ck_dsl/TRANSFORM_DAG.md`.
- Helpers reference: `python/ck_dsl/helpers/README.md`.
