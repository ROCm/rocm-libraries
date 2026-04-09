# hipDNN Roadmap

This document outlines the development roadmap for hipDNN, a comprehensive graph-based deep learning library for AMD GPUs. For current operation support details, refer to the [Operation Support documentation](./OperationSupport.md).

> [!NOTE]
> 📝 This roadmap is subject to change based on project priorities, community feedback, and technical requirements. The hipDNN team will endevor to keep the roadmap up to date but the further out the quarter, the more speculative our plans. 😅
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

**Focus:** PyTorch integration, more operations, basic engine selection heuristic & core improvements

### PyTorch
- **PyTorch integration for opt-in hipDNN backend** ⏳(Early Q2)

### GEMM
- **hipBLASLt plugin expanded operation & datatype support** ⏳

### SDPA
- Limited SDPA provided by Fusilli, CK & ASM ⏳

### Normalization
- Adding new **HIP kernel provider plugin** to expand normalization support ⏳
- Expanded operation API & coverage to support Layernorm & RMS ⏳
- Expanded layout & datatype coverage for batchnorm

### Heuristics
- Heuristic plugin API
- **Initial heuristic plugin**

### Core
- Plugin SDK utilities to streamline plugin development for new providers ⏳
- Benchmarking & performance python tools ⏳
- Python API wrappers ⏳
- **Client auto-tuning API**

## P2 ~ Q3 2026

**Focus:** SDPA, better heuristics & improved kernel provider selection

### SDPA
- Wider SDPA support

### Heuristics
- **Heuristics Plug-in implementation** Phase 2 (Refining and expanding heuristic capabilities)

### Core
- Add **hipRTC & caching support** to plugin SDK (Empowers plugin developers, and standardizes caching of artifacts)
- Kernel engine tagging & filtering
  - Behavioral & numeric notes for filtering
  - Client API to enable filtering


## P3 ~ Q4 2026 & beyond

**Focus:** Q4 and beyond is far enough out, that there is substantial uncertainty on what will be the most important features at this time. We value community input on what you would like to see!

### Increase operational support coverage
- Additional high performance static fusion support for priority use cases
- Additional JIT graph support for operations
- Improve general operational support for operations:
  - Additional layout support
  - Additional data-type support

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
