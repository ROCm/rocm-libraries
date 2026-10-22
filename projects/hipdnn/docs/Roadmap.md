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

## Q2 2026

**Focus:** SDPA forward path, client auto-tuning, performance-tooling, and a generated support matrix.

### SDPA
- **First-wave SDPA forward kernels callable end-to-end through the graph API** ✅
  - Forward path on gfx942 & gfx950: BF16, head dims 128 and 192x128, GQA/MQA ✅
  - Bottom-right causal forward on gfx942 (gfx950 causal tracked in Q3) ✅
  - SDPA forward golden-reference data + tests landed ✅
- Forward LSE/softmax-stats output and a GPU forward reference ✅
- Overridable tensor shapes API (required for variable sequence lengths) ✅
  - Phase 1 override-shape plumbing landed (RFC 0008) ✅
- SDPA backward pass on gfx942 ⏳
  - FP32 gradient accumulation ✅
  - No-mask, causal and sliding-window mask modes ✅
  - BF16 and FP16 at head dim 128 ✅
  - Note: wider architecture and head-dim coverage tracked in Q3

### Auto-tuning
- **Client auto-tuning API** ✅
  - Autotune RFC merged ✅
  - Implementation and config op-matching landed ✅
- Build N alternative execution plans for a single graph ✅
- Sampling run that ranks plans by wall-time and selects a winner ✅
- Export auto-tuning result to a config file for reuse across runs ✅

### Benchmarking & performance testing

> [!IMPORTANT]
> The dnn-benchmarking tool has moved to its own repository: [ROCm/dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking).

- **Benchmarking & performance Python tools** ✅
  - Core dnn-benchmarking tool (engine comparison, SDPA/PyTorch references, HIP-event timing) landed ✅
  - Cross-platform (Windows) support landed ✅

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

> [!NOTE]
> Q3 2026 ended on 2026-09-30. Items that were not complete moved to [Carried over from Q3 2026](#carried-over-from-q3-2026) under Q4 2026. Only completed Q3 items remain in this section.

### Heuristics
- Heuristic plugin API (RFC 0007) ✅

### Support matrix
- Whole-graph bundled integration tests ✅
  - Default authoring format for new graph-verification tests ✅
  - Enabled as the default execution path in CI ✅

## Q4 2026 & beyond (Current milestone)

**Focus:** Broader operation, layout & datatype coverage; deeper framework integrations; and the graph-execution, tensor and packaging work that moves hipDNN from tech preview toward general availability. We still value community input on what you would like to see prioritized!

### Carried over from Q3 2026

> [!NOTE]
> These items were planned for Q3 2026. They were not complete when the quarter ended, so they moved here. Markers show the state at the time of the move.

#### SDPA
- Wider SDPA coverage: FP16 and FP8 forward, head dims up to 256, broader GQA configurations ⏳
- Production-quality backward pass ⏳ (the forward GPU reference landed in Q2)
- gfx950 backward and gfx950 causal forward
- **rocKE SDPA plugin for hipDNN** — ROCm Kernel Engine SDPA provider plugin
  - ROCm Kernel Engine (rocKE) in-progress, including a JIT SDPA path ⏳
- Ragged tensor support for variable sequence lengths (RFC 0014) ⏳

#### GEMM
- **hipBLASLt plugin expanded operation & datatype support** ⏳
  - FP8 (OCP) dequantize + GEMM path landed behind `HIPBLASLTPROVIDER_ENABLE_MX_GEMM` (off by default) ✅
- MX GEMMs through the hipBLASLt provider plugin ⏳
- Documented constraints surfaced for graph builders (alignment, batch, epilogues)

#### MOE (Mixture of Experts)
- MOE frontend and backend POC (limited coverage) ⏳
  - Frontend API, schema & backend descriptors landed ✅
  - Initial provider kernel dispatch ⏳

#### Heuristics
- Plugin architecture ⏳
- Phase 1 heuristic plugin: providing heuristic engine selection for limited architectures or team may pivot to support heuristics on rocKE

#### Normalization
- Expanded LayerNorm & RMSNorm kernel coverage in the HIP kernel provider ⏳
- Expanded layout & datatype coverage for batchnorm ⏳

#### Support matrix
- Matrix published as a regular CI artifact ⏳

#### Testing
- **Op coverage: detailed per-op integration tests**, in order Attention, then Convolution, then Normalization
  - A test plan per tier (quick, standard, comprehensive, full) for each op
  - Pairwise feature coverage per op, so the smallest set flags obvious breaks across providers
  - Bundles that execute the plans
  - Golden-reference baseline per op

#### Universal descriptors
- Data-driven kernel ingestion via universal kernel, match, dispatch, engine and heuristic descriptors (RFC 0017) ⏳

#### Benchmarking & performance testing
- Benchmarking CI running on AMD GPUs in [ROCm/dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking) ⏳

#### PyTorch
- **PyTorch integration for opt-in hipDNN backend** ⏳

#### cuDNN compatibility
- **cuDNN v9 frontend compatibility shim** behind `HIPDNN_ENABLE_CUDNN_COMPATIBILITY` ⏳
  - Shim core node coverage landed ✅
  - Node and enum coverage expansion in progress ⏳

#### MIOpen integration
- **MIOpen ↔ hipDNN shim** enabling MIOpen to route through hipDNN, behind `MIOPEN_ENABLE_HIPDNN_WRAPPER` (off by default) ⏳
  - MIOpen superbuild integration ✅

#### Core
- Add **hipRTC & caching support** to plugin SDK (Empowers plugin developers, and standardizes caching of artifacts)
- Kernel engine tagging & filtering
  - Numeric notes for filtering
  - Client API to enable filtering
- Python API wrappers (general availability beyond POC) ⏳
- Plugin SDK utility expansion to further streamline new-provider development

### Testing
- Unified gate for a new integration or kernel source, built on the per-op coverage, giving high confidence in its quality
- Fuzzy mode for release testing (Q1 2027): exploratory runs within an affordable pre-release budget to find uncovered cases, grow the set, and show where coverage stands

### Increase operational support coverage
- Additional high performance static fusion support for priority use cases
- Additional JIT graph support for operations
- Improve general operational support for operations:
  - Additional layout support
  - Additional datatype support

### Benchmarking & performance testing
- Bindings installable as wheels ⏳
- App installable as wheel
- Automated benchmark runs weekly or daily off `develop`, with release-by-release regression tracking (Q1 2027)

### More framework integrations
- vLLM backend
- Currently discussing timelines for various framework integrations. Roadmap will be updated as they are defined.

### Normalization
- **Distributed normalization support**

### Core
- Fallback (graph splitting) engine
- Expanded performance and validation suites for hipDNN full install (using real user workloads and benchmarks to drive testing)
- AOT graph compilation without devices present (Pre-compile graph support)
- **hipGraph support**
- Support dynamic linking to backend (enables forwards and backwards compatible client libraries) ⏳
  - Save/Load Execution plans (binary serialize/deserialize of graph + execution plan landed in Q2; full save/load across dynamic linking tracked here)
- Non-standard tensor support (non-packed, vectorized)

## Contributing

hipDNN is an open-source project that welcomes community contributions. Your feedback shapes the project's direction.

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md). For questions or suggestions, please open an issue in the [hipDNN repository](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn).
