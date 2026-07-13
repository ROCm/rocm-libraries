# hipDNN Roadmap

This document outlines the development roadmap for hipDNN, a comprehensive graph-based deep learning library for AMD GPUs. For current operation support details, refer to the [Operation Support documentation](./OperationSupport.md).

> [!NOTE]
> 📝 This roadmap is subject to change based on project priorities, community feedback, and technical requirements. The hipDNN team will endeavor to keep the roadmap up to date but the further out the quarter, the more speculative our plans. 😅
>
> ✅ = Done
>
> ⏳ = In progress

## Q1 2026

**Focus:** Stable foundation & core operations

### Conv
- **Convolution MIOpen plugin support** ✅
  - Including basic fusions ✅

### Normalization
- **Batch normalization MIOpen plugin support** ✅
  - Including basic fusions ✅
- **LayerNorm & RMSNorm frontend API** ✅

### GEMM
- **Initial frontend GEMM API support** ✅
- hipBLASLt plugin initial enablement ✅

### SDPA
- **SDPA frontend API & backend descriptors** ✅

### Core
- **Stable, robust library to build upon** ✅
- Kernel engine settings (Engine knob configurations API + implementation) ✅
  - Ex. Flag for enabling benchmarking mode on MIOpen plugin
- Initial Python bindings POC ✅
- Initial benchmarking & performance tooling ✅

## Q2 2026 (Current milestone)

**Focus:** SDPA forward path, client auto-tuning, performance-tooling, and a generated support matrix.

### SDPA
- First-wave SDPA forward kernels callable end-to-end through the graph API ⏳
  - SDPA forward golden-reference data + tests landed ✅
  - Forward GPU reference, LSE/stats output, and causal-grid fixes in progress ⏳
- Overridable tensor shapes API (required for variable sequence lengths) ✅
  - Phase 1 override-shape plumbing landed (RFC 0008) ✅
- Note: SDPA backward pass is now in active development (mask support, FP32 gradient accumulation); production-quality backward and SDPA feature-flag gating remain tracked in later quarters

### Auto-tuning
- **Client auto-tuning API** ⏳
  - Autotune RFC merged ✅
  - Initial implementation and config op-matching in progress ⏳
- Build N alternative execution plans for a single graph ⏳
- Sampling run that ranks plans by wall-time and selects a winner ⏳
- Export auto-tuning result to a config file for reuse across runs ⏳

### Benchmarking & performance testing
- **Benchmarking & performance Python tools** ⏳
  - Core dnn-benchmarking tool (engine comparison, SDPA/PyTorch references, HIP-event timing) landed ✅
  - Cross-platform (Windows) port in progress ⏳

### Support matrix
- Integration tests emit structured pass/fail per op × datatype × engine × architecture ✅
- Generation step produces a human-readable support matrix from those results ✅
- Per-graph engine support-claims model defined (RFC 0015) ✅

### Heuristics
- **Engine selection config file support** ✅
  - Frontend heuristic policy enumeration API landed (RFC 0007) ✅

### Core
- Kernel engine tagging & filtering ✅
  - Behavioral notes for filtering ✅
- **Graph + execution-plan binary serialize/deserialize** ✅

## Q3 2026

**Focus:** SDPA, GEMM with MX low-precision data types, better heuristics & improved kernel provider selection

### SDPA
- Wider SDPA support
- **rocKE SDPA plugin for hipDNN** — ROCm Kernel Engine SDPA provider plugin
  - ROCm Kernel Engine (rocKE) proof-of-concept underway, including a tiled C-JIT SDPA path ⏳

### GEMM
- **hipBLASLt plugin expanded operation & datatype support** ⏳
  - FP8 (OCP) dequantize + GEMM path in progress ⏳
- MX GEMMs through the hipBLASLt provider plugin ⏳
- Documented constraints surfaced for graph builders (alignment, batch, epilogues)

### MOE (Mixture of Experts)
- MOE frontend and backend POC (limited coverage)

### Heuristics
- Heuristic plugin API
- Plugin architecture ⏳
- Phase 1 heuristic plugin: providing heuristic engine selection for limited architectures or team may pivot to support heuristics on rocKE

### Normalization
- Expanded LayerNorm & RMSNorm kernel coverage in the HIP kernel provider
- Expanded layout & datatype coverage for batchnorm

### Support matrix
- Matrix published as a regular CI artifact ⏳
- Bundled Integration Tests ⏳

### Benchmarking & performance testing
- Set up CICD for the project ⏳

### PyTorch
- **PyTorch integration for opt-in hipDNN backend** ⏳

### cuDNN compatibility
- **cuDNN v9 frontend compatibility shim** behind `HIPDNN_ENABLE_CUDNN_COMPATIBILITY` ⏳
  - Shim skeleton landed ✅
  - Enum coverage in progress ⏳

### MIOpen integration
- **MIOpen ↔ hipDNN shim** enabling MIOpen to route through hipDNN (RFC stage)
  - MIOpen superbuild integration ✅

### Core
- Add **hipRTC & caching support** to plugin SDK (Empowers plugin developers, and standardizes caching of artifacts)
- Kernel engine tagging & filtering
  - Numeric notes for filtering
  - Client API to enable filtering
- Python API wrappers (general availability beyond POC) ⏳
- Plugin SDK utility expansion to further streamline new-provider development

## Q4 2026 & beyond

**Focus:** Q4 and beyond is far enough out, that there is substantial uncertainty on what will be the most important features at this time. We value community input on what you would like to see!

### Increase operational support coverage
- Additional high performance static fusion support for priority use cases
- Additional JIT graph support for operations
- Improve general operational support for operations:
  - Additional layout support
  - Additional datatype support

### Benchmarking & performance testing
- Bindings installable as wheels
- App installable as wheel

### More framework integrations
- Currently discussing timelines for various framework integrations. Roadmap will be updated as they are defined.

### Normalization
- **Distributed normalization support**

### Core
- Fallback (graph splitting) engine
- Expanded performance and validation suites for hipDNN full install (using real user workloads and benchmarks to drive testing)
- AOT graph compilation without devices present (Pre-compile graph support)
- **hipGraph support**
- Support dynamic linking to backend (enables forwards and backwards compatible client libraries)
  - Save/Load Execution plans (binary serialize/deserialize of graph + execution plan landed in Q2; full save/load across dynamic linking tracked here)
- Non-standard tensor support (ragged, non-packed, vectorized)

## Contributing

hipDNN is an open-source project that welcomes community contributions. Your feedback shapes the project's direction.

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md). For questions or suggestions, please open an issue in the [hipDNN repository](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn).
