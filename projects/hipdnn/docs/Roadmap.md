# hipDNN Roadmap

This document outlines the development roadmap for hipDNN, a comprehensive graph-based deep learning library for AMD GPUs. For current operation support details, refer to the [Operation Support documentation](./OperationSupport.md).

> [!NOTE]
> 📝 This roadmap is subject to change based on project priorities, community feedback, and technical requirements. The hipDNN team will endeavor to keep the roadmap up to date but the further out the quarter, the more speculative our plans. 😅
>
> ✅ = Done
>
> ⏳ = In progress

## P0 ~ Q1 2026

**Focus:** Stable foundation & core operations

### Conv
- **Convolution MIOpen plugin support** ✅
  - Including basic fusions ✅
- **Convolution Fusilli plugin support** ✅

### Normalization
- **Batch normalization MIOpen plugin support** ✅
  - Including basic fusions ✅
- **Batch normalization Fusilli plugin support** ✅
- **LayerNorm & RMSNorm frontend API** ✅

### GEMM
- **Initial frontend GEMM API support** ✅
- Fusilli plugin integration (see note) ✅
- hipBLASLt plugin initial enablement ✅

### SDPA
- **SDPA frontend API & backend descriptors** ✅

### Core
- **Stable, robust library to build upon** ✅
- Kernel engine settings (Engine knob configurations API + implementation) ✅
  - Ex. Flag for enabling benchmarking mode on MIOpen plugin
- Initial Python bindings POC ✅
- Initial benchmarking & performance tooling ✅

> **Notes:**
> - Fusilli plugin is opt-in, and not defaulted on yet.
> - PyTorch integration was moved to early Q2

## P1 ~ Q2 2026 (Current milestone)

**Focus:** SDPA forward path, GEMM with MX low-precision data types, client auto-tuning, performance-testing CI, and a generated support matrix.

### PyTorch
- **PyTorch integration for opt-in hipDNN backend** ⏳

### SDPA
- First-wave SDPA forward kernels callable end-to-end through the graph API ⏳
- Overridable tensor shapes (required for variable sequence lengths) ⏳
- Note: backward-pass production quality and SDPA feature-flag gating tracked in later quarters

### GEMM
- **hipBLASLt plugin expanded operation & datatype support** ⏳
- MX GEMMs through the hipBLASLt provider plugin ⏳
- Documented constraints surfaced for graph builders (alignment, batch, epilogues)

### Auto-tuning
- **Client auto-tuning API** ⏳
- Build N alternative execution plans for a single graph ⏳
- Sampling run that ranks plans by wall-time and selects a winner ⏳
- Export auto-tuning result to a config file for reuse across runs ⏳

### Benchmarking & performance testing
- **Benchmarking & performance Python tools** ⏳
- Installable as wheels
- Set up CICD for the project

### Support matrix
- Integration tests emit structured pass/fail per op × datatype × engine × architecture ✅
- Generation step produces a human-readable support matrix from those results ✅
- Matrix published as a regular CI artifact ⏳

### Heuristics
- **Engine selection config file support** ⏳

### Core
- Kernel engine tagging & filtering ✅
  - Behavioral notes for filtering ✅

## P2 ~ Q3 2026

**Focus:** SDPA, better heuristics & improved kernel provider selection

### SDPA
- Wider SDPA support
- **CK SDPA plugin for hipDNN** — Composable Kernel-backed SDPA provider plugin

### Heuristics
- Heuristic plugin API
- Plugin architecture ⏳
- **Phase 1 heuristic plugin: providing heuristic engine selection for limited architectures**

### Normalization
- Expanded LayerNorm & RMSNorm kernel coverage in the HIP kernel provider
- Expanded layout & datatype coverage for batchnorm

### Core
- Add **hipRTC & caching support** to plugin SDK (Empowers plugin developers, and standardizes caching of artifacts)
- Kernel engine tagging & filtering
  - Numeric notes for filtering
  - Client API to enable filtering
- Python API wrappers (general availability beyond POC)
- Plugin SDK utility expansion to further streamline new-provider development


## P3 ~ Q4 2026 & beyond

**Focus:** Q4 and beyond is far enough out, that there is substantial uncertainty on what will be the most important features at this time. We value community input on what you would like to see!

### Increase operational support coverage
- Additional high performance static fusion support for priority use cases
- Additional JIT graph support for operations
- Improve general operational support for operations:
  - Additional layout support
  - Additional datatype support

### More framework integrations
- Currently discussing timelines for various framework integrations. Roadmap will be updated as they are defined.

### Normalization
- **Distributed normalization support**

### Core
- Expanded performance and validation suites for hipDNN full install (using real user workloads and benchmarks to drive testing)
- AOT graph compilation without devices present (Pre-compile graph support)
- **hipGraph support**
- Support dynamic linking to backend (enables forwards and backwards compatible client libraries)
  - Save/Load Execution plans
- Non-standard tensor support (ragged, non-packed, vectorized)

## Contributing

hipDNN is an open-source project that welcomes community contributions. Your feedback shapes the project's direction.

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md). For questions or suggestions, please open an issue in the [hipDNN repository](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn).
