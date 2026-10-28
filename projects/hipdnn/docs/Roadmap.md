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
- SDPA backward pass on gfx942 ✅
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

### SDPA
- Forward GPU reference ✅
- gfx950 causal forward ✅
- **rocKE SDPA in the HIP kernel provider** — ROCm Kernel Engine SDPA kernels served through hipDNN ✅
  - gfx950 dense attention engine ✅
  - gfx1151 (Strix Halo) inference SDPA engine ✅

### GEMM
- **hipBLASLt plugin expanded operation & datatype support** ✅
  - FP8 (OCP) dequantize + GEMM ✅
- MX GEMMs (MX FP4 and MX FP6) through the hipBLASLt provider plugin ✅
- Documented constraints surfaced for graph builders (alignment, batch, epilogues) ✅

### Heuristics
- Universal Heuristic Descriptor design (RFC 0019) ✅
- Benchmarked engine-ranking cache, in-process and persisted to disk ✅

### Normalization
- Expanded LayerNorm & RMSNorm kernel coverage in the HIP kernel provider ✅

### Support matrix
- Whole-graph bundled integration tests ✅
  - Default authoring format for new graph-verification tests ✅
  - Enabled as the default execution path in CI ✅

### Universal descriptors
- **Data-driven kernel ingestion** via universal kernel, match, heuristic and engine descriptors (RFCs 0017–0020) ✅
  - Generic kernel ingestor with build-time kernel packaging ✅

### cuDNN compatibility
- **cuDNN v9 frontend compatibility shim** behind `HIPDNN_ENABLE_CUDNN_COMPATIBILITY` ✅
  - Shim core node coverage landed ✅
  - Node and enum coverage expansion ✅

### Core
- Runtime pass-by-value scalars (RFC 0016) ✅

## Q4 2026 (Current milestone)

**Focus:** Broad SDPA coverage, kernel-ingestor maturity, deeper integration testing, and expanded benchmarking.

### SDPA
- **Broad SDPA forward and backward coverage** ⏳
  - rocKE integration ⏳
  - Ahead-of-time (AOT) compiled FlyDSL kernels for broad coverage ⏳
- Wider SDPA coverage: FP16 and FP8 forward, head dims up to 256, broader GQA configurations ⏳
  - FP16 forward and broader GQA configurations ✅
  - FP8 forward path wired through the SDPA engine ✅
  - Head dim 256 on gfx1151 ✅ (CDNA head dim 256 still in progress)
- Production-quality backward pass, including gfx950 ⏳
- Ragged tensor support for variable sequence lengths (RFC 0014) ⏳
  - API, descriptor lowering and GPU reference ✅

### Kernel ingestor
- Performance improvements ✅
- Execution plan save/load ✅
- Expression handling improvements ⏳
- Runtime improvements ⏳

### MIOpen provider
- Close the operation-coverage gap in the hipDNN MIOpen provider ⏳

### GEMM
- **MX GEMM in the hipBLASLt provider** ⏳
  - gfx1250 enablement ⏳
  - Enabled by default ⏳

### Convolution
- Expanded convolution integration from rocKE ⏳

### Integration testing
- Per-op review and enhancements ⏳
- "Gauntlet" mode for deeply proving integrations

### Benchmarking & performance testing
- Improved op support in [ROCm/dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking), including SDPA backward and other backward ops
- Graph metadata for improved tracking and filtering of the benchmark corpus
- Expanded set of benchmarked graphs

### cuDNN compatibility
- Continued cuDNN v9 frontend compatibility, including a user guide and samples ⏳
- cuDNN v8 frontend compatibility, scope to be confirmed

### Heuristics
- Learned engine and kernel ranking via Universal Heuristic Descriptors, with shipped trained models ⏳
- Heuristic generation pipeline for custom engines, letting custom engines compete with built-in engines in selection ⏳

### MOE (Mixture of Experts)
- MOE frontend and backend POC (limited coverage) ⏳
  - Frontend API, schema & backend descriptors landed ✅
  - Initial provider kernel dispatch ⏳

### Support matrix
- Matrix published as a regular CI artifact ⏳

### PyTorch
- **PyTorch integration for opt-in hipDNN backend** ⏳
  - `hipdnn_torch` injection layer ✅

### MIOpen integration
- **Opt-in runtime-feature flag to route MIOpen → hipDNN for convolutions** ⏳
  - MIOpen superbuild integration ✅
  - Convolution entry points forwarded to hipDNN behind `MIOPEN_ENABLE_HIPDNN_WRAPPER` ✅

### Core
- Python API wrappers (general availability beyond POC) ⏳
  - Engine discovery, auto-tuning and DLPack interoperability ✅
  - Frontend installable as a wheel ✅
- Plugin SDK utility expansion to further streamline new-provider development ⏳

## Beyond Q4 2026

**Focus:** Broader operation, layout & datatype coverage; deeper framework integrations; and the graph-execution, tensor and packaging work that moves hipDNN from tech preview toward general availability. We still value community input on what you would like to see prioritized!

### Increase operational support coverage
- Additional high performance static fusion support for priority use cases
- Additional JIT graph support for operations
- Improve general operational support for operations:
  - Additional layout support
  - Additional datatype support
- Expanded layout & datatype coverage for batchnorm

### Benchmarking & performance testing
- Benchmarking CI running on AMD GPUs in [ROCm/dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking)
- App installable as wheel

### More framework integrations
- vLLM backend
- Deeper PyTorch integration, using hipDNN for attention
- Currently discussing timelines for various framework integrations. Roadmap will be updated as they are defined.

### Normalization
- **Distributed normalization support**

### MIOpen integration
- Harden the MIOpen → hipDNN convolution reroute until it can be the default-on path

### Core
- Add **hipRTC & caching support** to plugin SDK (Empowers plugin developers, and standardizes caching of artifacts)
- Kernel engine tagging & filtering
  - Numeric notes for filtering
  - Client API to enable filtering
- Fallback (graph splitting) engine
- Expanded performance and validation suites for hipDNN full install (using real user workloads and benchmarks to drive testing)
- AOT graph compilation without devices present (Pre-compile graph support)
- **hipGraph support**
- Support dynamic linking to backend (enables forwards and backwards compatible client libraries)
  - Save/Load Execution plans (binary serialize/deserialize of graph + execution plan landed in Q2; full save/load across dynamic linking tracked here)
- Non-standard tensor support (non-packed, vectorized)

## Contributing

hipDNN is an open-source project that welcomes community contributions. Your feedback shapes the project's direction.

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md). For questions or suggestions, please open an issue in the [hipDNN repository](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn).
