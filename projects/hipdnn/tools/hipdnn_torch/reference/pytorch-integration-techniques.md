# PyTorch → hipDNN integration techniques

> **Point-in-time design reference.** This analysis was captured in 2026. Roadmap items,
> quarter dates, provider maturity, and version numbers below reflect that snapshot and
> will age — treat them as direction, not current status.

> **How this relates to the samples in this package.** The `hipdnn_torch` package and its
> `samples/` implement one form of Tier-1 injection: functional-level interception of
> `F.linear` / `F.rms_norm` / `F.scaled_dot_product_attention`, with native fallback and a
> census of what routed vs. fell back. This document covers that form alongside the
> ATen-dispatch form of Tier 1 (§3.1 sets the two side by side, with the tradeoffs of each)
> and the more integrated tiers above it — a C++ extension, TorchInductor fusion, and a
> native backend.

A survey of approaches for routing PyTorch operations through hipDNN on ROCm, with
fallback to the default implementation when hipDNN doesn't support a configuration.

## Why hipDNN

hipDNN routes operations through a plugin architecture in which multiple kernel providers
can serve the same operation. Four properties motivate integrating with it:

1. **Multi-engine architecture**: hipDNN's plugin architecture supports multiple kernel providers for the same operation. For SDPA, this means AITER ASM kernels, rocKE kernels, and RDNA-tuned kernels can all coexist — the framework doesn't need to track AMD's kernel ecosystem. Today, engine selection uses configurable static ordering (`HIPDNN_HEUR_FALLBACK_ENGINE_ORDER`) or auto-tuning (which benchmarks all options but has warmup cost). Smarter heuristic-based or analytic selection is planned but not yet implemented — this is an active area of development, not a solved problem.

2. **Integrate once, benefit later**: When new kernel capabilities land — a faster SDPA, improved convolutions from a new CK version, FP8 GEMM optimizations in hipBLASLt — they can be pulled into hipDNN as provider plugins and delivered to every framework already integrated with hipDNN. No new API calls, no Python bindings to build, no model code to adapt. The framework keeps calling the same hipDNN graph API; the new capability appears as a better engine option. Similarly, new fusion patterns (e.g., attention+norm, matmul+activation) don't require framework-side changes — they're served through hipDNN's graph API consistently. Once auto-tuning with persistent caching is in place, the best engine across all providers is selected and cached, so each new provider addition improves performance without framework-side integration effort.

3. **RDNA enablement**: AITER is MI3XX-only with no RDNA roadmap, and CK Flash Attention does not currently compile on RDNA (Wave64 ASM boundaries), so on some paths consumer AMD GPUs (RX 7900, RX 9070 series) fall back to unoptimized kernels or have no working kernel today. hipDNN with RDNA-tuned kernel providers is a path to running well-optimized Flash Attention and other DNN operations on Wave32 consumer GPUs.

4. **Graph-level fusion**: hipDNN's graph API enables multi-op fusion (conv+bn+relu, matmul+bias+activation, attention+norm) that individual library calls cannot achieve. New fusion patterns added to providers are automatically available — frameworks don't need to learn a new API or adapt model code for each fusion.

Each integration tier is a progressively deeper path to these properties:
- **Phase 1 (Python injection)** gives access to hipDNN's multi-engine architecture and RDNA support for inference workloads
- **Phase 2 (C++ extension)** provides production-grade performance with proper graph caching
- **Phase 3 (TorchInductor fusion)** enables hipDNN's graph-level fusion
- **Phase 4 (native backend)** is the long-term PyTorch integration (roadmap item)

---

## 1. Executive Summary & Decision Matrix

### Integration Tiers

| Tier | Approach | Effort | torch.compile | Fallback | Use Case |
|------|----------|--------|---------------|----------|----------|
| **1 — Python** | `torch.library.Library("aten", "IMPL")` + `hipdnn_frontend` | Days | Yes (below Dynamo) | `get_kernel` + `call_boxed` | Fast validation of rocKE kernels against real models |
| **2 — C++ ext** | `torch.utils.cpp_extension.load()` + `TORCH_LIBRARY_IMPL` | Weeks | Yes | `at::redispatch` | Production perf with C++-level graph caching |
| **3 — Inductor** | Pattern matcher fusion passes (`hipdnn_fusion.py`) | Weeks-Months | Native | N/A (pattern match) | Multi-op fusion (conv+bn+relu, matmul+bias+act) |
| **4 — Backend** | cuDNN v9 shim (SDPA) + MIOpen shim (conv/BN) + native backend | Months | Native | N/A | Upstream PyTorch integration |

### Per-Operation Feasibility

| Operation | hipDNN API | Provider | Maturity | Phase 1 Feasible | Key ATen Op |
|-----------|-----------|----------|----------|-----------------|-------------|
| Convolution | `conv_fprop` / `conv_dgrad` / `conv_wgrad` | MIOpen | Production (Q1 2026) | Yes | `aten::convolution` |
| GEMM/Matmul | `matmul` | hipBLASLt | Production (Q1 2026) | Yes | `aten::mm`, `aten::addmm`, `aten::bmm` |
| BatchNorm | `batchnorm` / `batchnorm_inference` / `batchnorm_backward` | MIOpen + hip-kernel | Production (Q1 2026) | Yes | `aten::native_batch_norm` |
| LayerNorm | `layernorm` / `layernorm_backward` | hip-kernel | Expanding (Q3 2026) | Yes | `aten::native_layer_norm` |
| RMSNorm | `rmsnorm` / `rmsnorm_backward` | hip-kernel | Expanding (Q3 2026) | Yes | `aten::_fused_rms_norm` |
| SDPA | `sdpa` / `sdpa_backward` | rocKE | Active dev (Q2-Q3 2026) | Yes | `aten::_scaled_dot_product_flash_attention` |

---

## 2. Architecture Overview

### 2.1 hipDNN Architecture

hipDNN is AMD's graph-based deep learning library — the ROCm analog to NVIDIA's cuDNN v9. It is NOT a wrapper around MIOpen. It is a higher-level abstraction with a plugin architecture that dispatches to multiple backend kernel providers.

```
Frameworks (PyTorch, vLLM, SGLang)
                |
        hipDNN Frontend (C++ header-only, cuDNN v9-style graph API)
                |
        hipDNN Backend (C API, libhipdnn_backend.so)
                |
    +-----------+-----------+-----------+
    |           |           |           |
 MIOpen     hipBLASLt    rocKE/HIP    Custom
 Plugin     Plugin       Kernel       Plugins
 (conv,     (GEMM,       Provider
 batchnorm) FP8 MX)      (SDPA, norms)
    |           |           |
 MIOpen lib hipBLASLt    CK/rocKE/HIP
             lib          kernels
```

**Multi-engine architecture**: hipDNN doesn't lock you into one kernel library. Multiple providers can coexist for the same operation: MIOpen for mature conv workloads, hipBLASLt for GEMM with fused epilogues, rocKE kernels for SDPA and norms, or AITER kernels when available via provider plugins. Today, provider selection uses configurable static ordering or auto-tuning (which benchmarks all options at a warmup cost). As AMD ships faster kernels in any library, they can be added as providers and prioritized via configuration — frameworks don't need to change. Smarter automatic heuristic selection is planned but remains future work.

**RDNA enablement**: The hip-kernel-provider and other providers can include RDNA-tuned kernels (WMMA-based for Wave32) alongside CDNA kernels (MFMA-based for Wave64). hipDNN's architecture-aware dispatch routes to the right kernel per GPU — transparent to the framework. This is what lets one integration serve both CDNA and RDNA parts.

**Graph lifecycle** (build once, execute many):
```
Graph() → add ops → validate() → build_operation_graph(handle)
       → create_execution_plans() → check_support() → build_plans()
       → execute(handle, variant_pack, workspace)  // repeat with different data
```

Key source files:
- Frontend Graph API: `projects/hipdnn/frontend/include/hipdnn_frontend/Graph.hpp`
- Backend C API: `<rocm-sdk>/include/hipdnn/backend/hipdnn_backend.h`
- Python bindings: `projects/hipdnn/python/frontend_bindings/build/hipdnn_frontend_python.abi3.so`
- Installed library: `<rocm-sdk>/lib/libhipdnn_backend.so`

### 2.2 PyTorch ROCm Dispatch

PyTorch uses a dispatcher with dispatch keys. On ROCm, the key hierarchy is:

```
Python → Autocast → Autograd → Backend (HIP)
```

Operations decompose through a chain. High-level ops (e.g., `F.conv2d`) are `CompositeImplicitAutograd` — they decompose to lower-level ATen ops that have actual CUDA/HIP kernel implementations.

**Current dispatch paths**:
- **Convolution**: `F.conv2d` → `aten::convolution` (CompositeExplicitAutograd) → `aten::miopen_convolution` (HIP) → MIOpen
- **GEMM**: `F.linear` → `aten::addmm` (HIP) → hipBLASLt/rocBLAS
- **BatchNorm**: `F.batch_norm` → `aten::native_batch_norm` (HIP) → MIOpen or native kernels
- **LayerNorm**: `F.layer_norm` → `aten::native_layer_norm` (HIP) → native HIP kernels
- **RMSNorm**: `F.rms_norm` → `aten::_fused_rms_norm` (HIP) → native HIP kernels
- **SDPA**: `F.scaled_dot_product_attention` → `aten::_scaled_dot_product_flash_attention` (HIP) → Flash Attention (AOTriton/CK)

**Critical dispatch key detail**: On ROCm builds, the dispatch key is `"HIP"`, not `"CUDA"`. Use `torch._C._dispatch_key_for_device("cuda")` for portable code that returns the correct key on either platform.

### 2.3 Integration Tiers Defined

**Tier 1 — Python-only** (days): Override ATen ops at the HIP dispatch key using `torch.library.Library("aten", "IMPL")`. Call hipDNN through the `hipdnn_frontend` Python bindings. Save original kernels via `torch.library.get_kernel()` for fallback.

**Tier 2 — C++ extension** (weeks): Build a HIP C++ extension with `torch.utils.cpp_extension.load()`. Register implementations via `TORCH_LIBRARY_IMPL(aten, CUDA, m)` in C++. Call hipDNN's C++ frontend directly. Use `at::redispatch` for fallback. `hipcc` is auto-selected on ROCm.

**Tier 3 — TorchInductor fusion** (weeks-months): Register pattern matcher passes modeled after `mkldnn_fusion.py` to match multi-op sequences (conv+bn+relu) and replace with hipDNN fused graph calls.

**Tier 4 — Native backend** (months): Three parallel paths: (A) hipify PyTorch's cuDNN v9 SDPA code via hipDNN's compatibility shim, (B) the MIOpen ↔ hipDNN shim that transparently routes MIOpen's conv/BN calls through hipDNN (zero PyTorch changes), and (C) a full native hipDNN backend. See Section 6.3 for detailed comparison.

---

## 3. Phase 1: Python-Only Full Injection

Python-only injection is a fast path to running models (e.g., ComfyUI) on hipDNN/rocKE
kernels with no model or PyTorch-source changes, with clean fallback to the default
implementation. It comes in two forms — this package implements one, and this section
details the other. They are complementary, not mutually exclusive.

### 3.1 Two forms of Python injection

| | Functional monkeypatch (implemented in this package) | ATen-dispatch override (detailed in the rest of §3) |
|---|---|---|
| **Mechanism** | Replace `torch.nn.functional.linear` / `.rms_norm` / `.scaled_dot_product_attention` | Register `torch.library.Library("aten", "IMPL")` overrides at the HIP dispatch key |
| **Call sites caught** | Calls that resolve `F.<op>` at call time — `nn.Linear`, `nn.RMSNorm`, `F.sdpa`. Does not catch GEMMs written as `torch.matmul` / `@` / `torch.bmm`. | The ATen op regardless of how it was spelled — `@`, `torch.matmul`, `bmm`, and direct `torch.ops.aten.*` all route through it. |
| **torch.compile** | Not caught — Dynamo resolves function references at trace time | Caught — the override runs below Dynamo at the dispatcher |
| **Autograd / training** | Returns a plain tensor; inference-only unless you add an `autograd.Function` | Registered below autograd; cover training by also overriding the backward ATen ops |
| **Reading op intent** | Direct — the high-level args (`is_causal`, `scale`, `normalized_shape`) are in hand | Indirect — you see decomposed ops; SDPA is post-backend-selection and returns a tuple |
| **Fallback** | Call the saved original functional | `get_kernel()` + `call_boxed()` |
| **Opt-in** | Via `install()`; importing the package has no side effects | Overrides register when the injection module is imported |

Both are "Tier 1." Which one fits depends on the goal. The functional monkeypatch is
simpler, and its high-level view of each call makes the fallback census straightforward to
read — a fit for bring-up and correctness/coverage work in eager-mode frameworks (ComfyUI,
Hugging Face with `attn_implementation="sdpa"`). The ATen-dispatch form reaches more call
sites and is caught by `torch.compile`, so it fits broader op coverage, compiled models, and
training. The two can also be combined. The remainder of this section details the
ATen-dispatch form; the functional-monkeypatch form is in this package's `hipdnn_torch/`
modules and `samples/`.

### 3.2 The ATen-dispatch injection mechanism

```python
import torch
import hipdnn_frontend as hipdnn

# Determine the correct dispatch key for this platform
_DISPATCH_KEY = torch._C._dispatch_key_for_device("cuda")  # "HIP" on ROCm, "CUDA" on NVIDIA

# Step 1: Capture original kernels BEFORE any overrides
_originals = {}
_OPS_TO_OVERRIDE = [
    "convolution", "mm", "addmm", "bmm",
    "native_batch_norm", "native_layer_norm",
    "_fused_rms_norm", "native_group_norm",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
]

for op_name in _OPS_TO_OVERRIDE:
    _originals[op_name] = torch.library.get_kernel(f"aten::{op_name}", _DISPATCH_KEY)

# Step 2: Register overrides
_lib = torch.library.Library("aten", "IMPL")

def _make_hipdnn_override(op_name, hipdnn_impl_fn):
    """Create an override that tries hipDNN, falls back to original on any failure."""
    original = _originals[op_name]
    def impl(keyset, *args, **kwargs):
        try:
            return hipdnn_impl_fn(*args, **kwargs)
        except Exception:
            # hipDNN doesn't support this config — fall back to original kernel
            return original.call_boxed(keyset, *args, **kwargs)
    return impl

# Register each override
_lib.impl("convolution",
          _make_hipdnn_override("convolution", _hipdnn_convolution),
          _DISPATCH_KEY, with_keyset=True)
_lib.impl("mm",
          _make_hipdnn_override("mm", _hipdnn_mm),
          _DISPATCH_KEY, with_keyset=True)
# ... etc for each op
```

### 3.3 Complete ATen Op Override List

These are the exact lowest-level ATen operators with HIP kernels. Higher-level ops (e.g., `conv2d`, `linear`, `matmul`) auto-decompose into these.

| Category | ATen Op | Schema |
|---|---|---|
| **Convolution** | `aten::convolution` | `(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor` |
| **GEMM** | `aten::mm` | `(Tensor self, Tensor mat2) -> Tensor` |
| **GEMM** | `aten::addmm` | `(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor` |
| **GEMM** | `aten::bmm` | `(Tensor self, Tensor mat2) -> Tensor` |
| **BatchNorm** | `aten::native_batch_norm` | `(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)` |
| **LayerNorm** | `aten::native_layer_norm` | `(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)` |
| **RMSNorm** | `aten::_fused_rms_norm` | `(Tensor input, int[] normalized_shape, Tensor? weight, float? eps) -> (Tensor, Tensor)` |
| **GroupNorm** | `aten::native_group_norm` | `(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)` |
| **SDPA** | `aten::_scaled_dot_product_flash_attention` | `(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, ...) -> (Tensor, Tensor, ...)` |
| **SDPA** | `aten::_scaled_dot_product_efficient_attention` | `(Tensor query, Tensor key, Tensor value, Tensor? attn_bias=None, ...) -> (Tensor, Tensor, ...)` |

Note: `aten::_fused_rms_norm` IS a real ATen op — `torch.nn.functional.rms_norm` → `aten::rms_norm` (CompositeImplicitAutograd) → `aten::_fused_rms_norm` (HIP kernel). ComfyUI uses this at `comfy/ops.py:637`.

### 3.4 Graph/Plan Caching — Required for Performance

hipDNN's expensive steps (graph build, engine heuristic selection, plan compilation) must happen only once per unique operation configuration. On subsequent calls with the same shapes/dtypes/attributes, we skip straight to `execute()` with new tensor pointers.

**Cache key design**: `(op_type, input_shapes, input_dtypes, frozen_attributes)`

**Cache value**: `(built_graph, handle, tensor_uids, workspace_buffer)`

**On cache hit**: build `variant_pack` with new tensor `data_ptr()` values, call `graph.execute()`. This is the hot path.

```python
_graph_cache = {}   # cache_key -> CachedPlan
_handle = None      # single hipDNN handle, reused

class CachedPlan:
    __slots__ = ('graph', 'input_uids', 'output_uids', 'workspace_buf', 'workspace_size')
    def __init__(self, graph, input_uids, output_uids, workspace_buf, workspace_size):
        self.graph = graph
        self.input_uids = input_uids
        self.output_uids = output_uids
        self.workspace_buf = workspace_buf
        self.workspace_size = workspace_size

def _get_handle():
    global _handle
    if _handle is None:
        _handle = hipdnn.create_handle()
    return _handle

def _hipdnn_convolution(input, weight, bias, stride, padding, dilation,
                         transposed, output_padding, groups):
    cache_key = ("conv_fprop", input.shape, weight.shape, input.dtype,
                 tuple(stride), tuple(padding), tuple(dilation),
                 transposed, groups, bias is not None)

    if cache_key in _graph_cache:
        # HOT PATH: reuse compiled plan, just swap tensor pointers
        cached = _graph_cache[cache_key]
        variant_pack = {}
        variant_pack[cached.input_uids['x']] = input.data_ptr()
        variant_pack[cached.input_uids['w']] = weight.data_ptr()
        if bias is not None:
            variant_pack[cached.input_uids['bias']] = bias.data_ptr()

        # Allocate output tensor (same shape as before)
        output = torch.empty(cached.output_shape, dtype=input.dtype, device=input.device)
        variant_pack[cached.output_uids['y']] = output.data_ptr()

        ws_ptr = cached.workspace_buf.ptr() if cached.workspace_buf else 0
        result = cached.graph.execute(_get_handle(), variant_pack, ws_ptr)
        if not result.is_good():
            raise RuntimeError(f"hipDNN execute failed: {result.get_message()}")
        return output

    # COLD PATH: build graph, compile plans, cache for reuse
    graph = hipdnn.Graph()
    graph.set_io_data_type(_torch_to_hipdnn_dtype(input.dtype))
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)

    x_tensor = hipdnn.Tensor.create(list(input.shape), _torch_to_hipdnn_dtype(input.dtype))
    x_tensor.set_name("x").set_uid(0)

    w_tensor = hipdnn.Tensor.create(list(weight.shape), _torch_to_hipdnn_dtype(weight.dtype))
    w_tensor.set_name("w").set_uid(1)

    attrs = hipdnn.ConvFpropAttributes()
    attrs.set_padding(list(padding))
    attrs.set_stride(list(stride))
    attrs.set_dilation(list(dilation))

    y_tensor = graph.conv_fprop(x_tensor, w_tensor, attrs)
    y_tensor.set_name("y").set_output(True).set_uid(2)

    # Build plans
    handle = _get_handle()
    assert graph.validate().is_good()
    assert graph.build_operation_graph(handle).is_good()
    assert graph.create_execution_plans().is_good()
    assert graph.check_support().is_good()
    assert graph.build_plans().is_good()

    # Allocate workspace
    workspace_size = graph.get_workspace_size()
    workspace_buf = hipdnn.DeviceBuffer(workspace_size) if workspace_size > 0 else None

    # Cache it
    cached = CachedPlan(
        graph=graph,
        input_uids={'x': 0, 'w': 1},
        output_uids={'y': 2},
        workspace_buf=workspace_buf,
        workspace_size=workspace_size,
    )
    cached.output_shape = list(y_tensor.get_dim())
    _graph_cache[cache_key] = cached

    # Execute
    output = torch.empty(cached.output_shape, dtype=input.dtype, device=input.device)
    variant_pack = {0: input.data_ptr(), 1: weight.data_ptr(), 2: output.data_ptr()}
    ws_ptr = workspace_buf.ptr() if workspace_buf else 0
    result = graph.execute(handle, variant_pack, ws_ptr)
    if not result.is_good():
        raise RuntimeError(f"hipDNN execute failed: {result.get_message()}")
    return output
```

**Why caching matters**: For models like Stable Diffusion in ComfyUI, the same conv/matmul/norm shapes repeat across every denoising step (hundreds of times per image generation). After the first forward pass warms the cache, every subsequent call hits the hot path — just pointer swaps and `execute()`.

### 3.5 Per-Operation hipDNN Mapping

#### Convolution → `graph.conv_fprop()`

hipDNN C++ API:
```cpp
std::shared_ptr<TensorAttributes> conv_fprop(
    std::shared_ptr<TensorAttributes> x,
    std::shared_ptr<TensorAttributes> w,
    ConvFpropAttributes attributes)  // returns y
```

Python binding: `hipdnn.ConvFpropAttributes()` with `.set_padding()`, `.set_stride()`, `.set_dilation()`.

Backward: `conv_dgrad(dy, w, attrs)` → dx, `conv_wgrad(dy, x, attrs)` → dw.

Provider support (MIOpen): FP16, BF16, FP32 / NCHW, NHWC, NCDHW, NDHWC. Deterministic engine available. Fused conv+bias+activation supported.

#### GEMM → `graph.matmul()`

hipDNN C++ API:
```cpp
std::shared_ptr<TensorAttributes> matmul(
    std::shared_ptr<TensorAttributes> a,
    std::shared_ptr<TensorAttributes> b,
    MatmulAttributes attributes)  // returns c
```

Python binding: `hipdnn.MatmulAttributes()` — no op-specific params beyond tensor I/O; shapes come from tensors.

**Handling `addmm`**: `addmm(bias, mat1, mat2, beta=1, alpha=1)` is `alpha * mat1 @ mat2 + beta * bias`. With hipDNN, use `graph.matmul()` for the GEMM, then `graph.pointwise()` with ADD for the bias — hipDNN will fuse these into a single kernel via the hipBLASLt epilogue.

Provider support (hipBLASLt): FP16, BF16, FP32. Transposed inputs, batched matmuls. Fused epilogues: bias, ReLU, clamp, GELU (tanh approx), Swish, and combinations.

#### BatchNorm → `graph.batchnorm()` / `graph.batchnorm_inference()`

hipDNN C++ API:
```cpp
// Training forward — returns [y, meanOut, invVarianceOut, nextRunningMean, nextRunningVar]
std::array<std::shared_ptr<TensorAttributes>, 5>
    batchnorm(std::shared_ptr<TensorAttributes> x,
              std::shared_ptr<TensorAttributes> scale,
              std::shared_ptr<TensorAttributes> bias,
              BatchnormAttributes attributes)

// Inference — returns y only
std::shared_ptr<TensorAttributes>
    batchnorm_inference(std::shared_ptr<TensorAttributes> x,
                        std::shared_ptr<TensorAttributes> mean,
                        std::shared_ptr<TensorAttributes> invVariance,
                        std::shared_ptr<TensorAttributes> scale,
                        std::shared_ptr<TensorAttributes> bias,
                        BatchnormInferenceAttributes attributes)

// Backward — returns [dx, dscale, dbias]
std::array<std::shared_ptr<TensorAttributes>, 3>
    batchnorm_backward(std::shared_ptr<TensorAttributes> dy,
                       std::shared_ptr<TensorAttributes> x,
                       std::shared_ptr<TensorAttributes> scale,
                       BatchnormBackwardAttributes attributes)
```

Provider support: MIOpen (spatial mode, FP16/BF16/FP32, all layouts) + hip-kernel-provider. Both training and inference. Fused patterns: BN+activation, BN+DReLU+backward.

#### LayerNorm → `graph.layernorm()`

hipDNN C++ API:
```cpp
// Returns [y, mean, invVariance]
std::array<std::shared_ptr<TensorAttributes>, 3>
    layernorm(std::shared_ptr<TensorAttributes> x,
              std::shared_ptr<TensorAttributes> scale,
              std::shared_ptr<TensorAttributes> bias,
              LayernormAttributes attributes)

// Backward — returns [dx, dscale, dbias]
std::array<std::shared_ptr<TensorAttributes>, 3>
    layernorm_backward(std::shared_ptr<TensorAttributes> dy,
                       std::shared_ptr<TensorAttributes> x,
                       std::shared_ptr<TensorAttributes> scale,
                       LayernormBackwardAttributes attributes)
```

Provider: hip-kernel-provider. Kernel coverage expanding in Q3 2026.

#### RMSNorm → `graph.rmsnorm()`

hipDNN C++ API:
```cpp
// Returns [y, invRms]
std::array<std::shared_ptr<TensorAttributes>, 2>
    rmsnorm(std::shared_ptr<TensorAttributes> x,
            std::shared_ptr<TensorAttributes> scale,
            RMSNormAttributes attributes)

// Backward — returns [dx, dscale, dbias]
std::array<std::shared_ptr<TensorAttributes>, 3>
    rmsnorm_backward(std::shared_ptr<TensorAttributes> dy,
                     std::shared_ptr<TensorAttributes> x,
                     std::shared_ptr<TensorAttributes> scale,
                     std::shared_ptr<TensorAttributes> inv_rms,
                     RMSNormBackwardAttributes attributes)
```

Provider: hip-kernel-provider. hipDNN/rocKE provides an optimized RMSNorm where PyTorch only has a basic native kernel.

#### SDPA → `graph.sdpa()`

hipDNN C++ API:
```cpp
// Returns [o, stats]
std::array<std::shared_ptr<TensorAttributes>, 2>
    sdpa(std::shared_ptr<TensorAttributes> q,
         std::shared_ptr<TensorAttributes> k,
         std::shared_ptr<TensorAttributes> v,
         SdpaAttributes attributes)

// Backward — returns [dQ, dK, dV]
std::array<std::shared_ptr<TensorAttributes>, 3>
    sdpa_backward(std::shared_ptr<TensorAttributes> q,
                  std::shared_ptr<TensorAttributes> k,
                  std::shared_ptr<TensorAttributes> v,
                  std::shared_ptr<TensorAttributes> o,
                  std::shared_ptr<TensorAttributes> dO,
                  std::shared_ptr<TensorAttributes> stats,
                  SdpaBackwardAttributes attributes)
```

The SdpaAttributes API defines: GQA/MQA, causal masking, diagonal band bounds, additive attention bias, dropout (seed/offset), ALiBi positional encoding, paged attention (page_table_k/v), FP8 quantization (descale/scale tensors), sliding window, BHSD/BSHD layouts. **Important**: the API defines these features but not all are supported by current engines. The existing ASM SDPA engine supports basic forward (BF16/FP8_E4M3, causal, gfx942/gfx950). Paged attention, ALiBi, dropout, attention bias, and RDNA are defined in the API but awaiting engine implementation — rocKE is the expected path to filling these gaps as its kernel coverage expands.

Providers: Multiple engines compete for best performance per-configuration:
- **rocKE SDPA engine** — high-performance kernels authored via rocKE's agentic kernel authoring platform
- **AITER kernels** (when available via plugin) — proven MI3XX performance
- **RDNA-tuned kernels** — Wave32/WMMA-based for consumer GPUs (gfx1100, gfx1201)
- hipDNN's engine selection picks from available providers based on configurable static ordering or auto-tuning. Smarter heuristic selection is planned.

### 3.6 Fallback Strategy

The `get_kernel()` + `call_boxed()` pattern provides clean fallback. The fallback triggers when:

1. **Unsupported dtype/shape**: hipDNN provider doesn't support the combination (e.g., INT8 convolution, non-spatial batch norm)
2. **Provider not loaded**: The required plugin isn't available (e.g., hipBLASLt plugin for matmul)
3. **Graph build failure**: `validate()`, `build_operation_graph()`, or `build_plans()` returns an error
4. **Execution failure**: `execute()` returns an error at runtime

In all cases, the exception handler catches the failure and calls `original.call_boxed(keyset, *args, **kwargs)` to run the original MIOpen/hipBLASLt/native kernel.

**Logging**: When fallback occurs, log the op name, shapes, dtype, and reason. This data identifies which configurations need rocKE kernel coverage.

```python
import logging
_log = logging.getLogger("hipdnn_injection")

def _make_hipdnn_override(op_name, hipdnn_impl_fn):
    original = _originals[op_name]
    def impl(keyset, *args, **kwargs):
        try:
            return hipdnn_impl_fn(*args, **kwargs)
        except Exception as e:
            _log.debug(f"hipDNN fallback for {op_name}: {e}")
            return original.call_boxed(keyset, *args, **kwargs)
    return impl
```

### 3.7 ComfyUI-Specific Notes

ComfyUI does NOT use `torch.compile` by default. All `@torch.compile` decorators in the codebase are either commented out or behind dead code (`assert False`). ComfyUI has an opt-in torch.compile API (`comfy_api/torch_helpers/torch_compile.py`) but it is not called by default.

The `Library("aten", "IMPL")` approach works perfectly for ComfyUI because:
- It registers at the dispatcher level (below TorchDynamo)
- It survives if a user later opts into torch.compile
- It intercepts all ATen ops regardless of whether they come from `nn.Module`, `F.conv2d`, or direct `torch.ops.aten` calls

**Deployment**: The injection can be a single Python file imported before model loading:
```python
# In ComfyUI startup or custom node:
import hipdnn_torch_injection  # registers all overrides on import
```

---

## 4. Per-Operation Deep Dives

### 4.1 Convolutions

**Current dispatch**: `F.conv2d` → `aten::convolution` (CompositeExplicitAutograd) → selects `aten::miopen_convolution` on ROCm → MIOpen's `miopenConvolutionForward`

**hipDNN mapping**: `graph.conv_fprop(x, w, ConvFpropAttributes)`. The MIOpen provider wraps the same MIOpen calls but adds graph-level optimization.

**Tier 2 — C++ extension**:
```cpp
#include <hipdnn_frontend.hpp>
#include <torch/extension.h>

// Register via TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(aten, HIP, m) {
    m.impl("convolution", [](at::Tensor input, at::Tensor weight,
                              std::optional<at::Tensor> bias,
                              at::IntArrayRef stride, at::IntArrayRef padding,
                              at::IntArrayRef dilation, bool transposed,
                              at::IntArrayRef output_padding, int64_t groups) {
        // Build or lookup cached hipDNN graph
        auto result = hipdnn_conv_cached(input, weight, bias,
                                          stride, padding, dilation, groups);
        if (result.has_value()) return *result;
        // Fallback to original
        return at::native::convolution(input, weight, bias, stride, padding,
                                        dilation, transposed, output_padding, groups);
    });
}
```

**Tier 4 — cuDNN shim**: The cuDNN v9 compatibility shim does NOT yet expose conv operations — only SDPA. Adding `conv_fprop`/`conv_dgrad`/`conv_wgrad` to the shim's `graph_wrapper.h` would enable hipify-based integration.

**Provider maturity**: MIOpen provider conv is production-ready (Q1 2026). Supports FP16, BF16, FP32, NCHW/NHWC layouts, 2D and 3D, depthwise, groups, deterministic engine.

### 4.2 GEMM / Matmul

**Current dispatch**: PyTorch on ROCm has four GEMM backends, selected differently in eager mode vs torch.compile:

*Eager mode* (`torch.mm()` / `F.linear` without torch.compile):
- `at::cuda::blas::gemm()` checks `BlasBackend` preference (set via `torch.backends.cuda.preferred_blas_library()` or `TORCH_BLAS_PREFER_CUBLASLT=1`):
  - `Default` or `Cublas` → rocBLAS/hipBLAS (the default)
  - `Cublaslt` → hipBLASLt
  - `Ck` → Composable Kernel GEMM (compiled in via `USE_ROCM_CK_GEMM` but NOT default — must be explicitly selected; uses CDNA-oriented `DeviceGemmMultiD_Xdl_CShuffle_V3`)
- **TunableOp** (`PYTORCH_TUNABLEOP_ENABLED=1`) intercepts before backend selection, benchmarks all rocBLAS AND hipBLASLt algorithms per problem size, caches results to CSV. This is the closest existing analog to hipDNN's multi-engine auto-tuning.

*torch.compile* (TorchInductor with `max_autotune`):
- Inductor collects candidates from backends in `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` (default: `ATEN,TRITON,CPP`)
- Available ROCm backends: **ATEN** (falls through to eager dispatch), **TRITON** (Triton-generated GEMM), **CK** (CK universal GEMM instances), **CKTILE** (newer CK Tile API)
- CK/CKTILE are effectively off by default — not in the default backend list, require the separate `ck4inductor` package to be installed, and are hardcoded CDNA-only (`ck_supported_arch = ["gfx90a", "gfx942", "gfx950"]`)

**hipDNN mapping**: `graph.matmul(a, b, MatmulAttributes)`. The hipBLASLt provider wraps hipBLASLt with graph-level epilogue fusion.

**Key value**: Given the already-rich GEMM backend landscape in PyTorch, standalone GEMM through hipDNN adds limited value. The advantage comes from **fused epilogues** (matmul+bias+activation as a single kernel via hipDNN's graph API) and **future multi-engine dispatch** when hipDNN has more than one GEMM provider.

**Provider maturity**: hipBLASLt provider is production-ready for FP32/FP16/BF16. FP8 OCP BlockScale GEMM (MX format) is in progress for gfx950/gfx1250.

**RDNA GEMM opportunity**: rocBLAS (Tensile) is **1.7x faster** than hipBLASLt for standard GEMM on RDNA 3/3.5 (gfx1151 benchmark: 115ms vs 194ms) because hipBLASLt lacked properly tuned kernels. CK GEMM is effectively unavailable on RDNA — hardcoded CDNA-only in Inductor, and the eager-mode CK backend uses CDNA-oriented kernels. In practice, RDNA users get only rocBLAS (default, better perf) or hipBLASLt (worse on RDNA 3/3.5) — no CK, no Triton GEMM by default. A hipDNN rocBLAS GEMM provider favored on RDNA alongside hipBLASLt on CDNA would deliver real multi-engine value. rocBLAS lacks FP8/FP4/FP6, fused epilogues, and grouped GEMM — but the missing narrow types are irrelevant for current RDNA hardware. For FP16/BF16 GEMM, which dominates RDNA workloads, rocBLAS is the better choice today. Note: on RDNA 4 (gfx12xx), hipBLASLt is now the default within rocBLAS, so the gap may be closing for the newest architectures.

### 4.3 BatchNorm

**Current dispatch**: `F.batch_norm` → `aten::native_batch_norm` → MIOpen or native HIP kernels

**hipDNN mapping**: `graph.batchnorm()` for training, `graph.batchnorm_inference()` for inference. Multi-return: training returns `[y, mean, invVar, nextRunningMean, nextRunningVar]`.

**Key consideration**: The ATen op returns `(Tensor, Tensor, Tensor)` — output, save_mean, save_invstd. The hipDNN API returns 5 values for training. The mapping must extract the correct subset.

**Provider maturity**: Production-ready. Both MIOpen and hip-kernel-provider. Spatial mode, FP16/BF16/FP32, all layouts.

### 4.4 LayerNorm

**Current dispatch**: `F.layer_norm` → `aten::native_layer_norm` → native HIP kernels (hipified `layer_norm_kernel.cu`)

**hipDNN mapping**: `graph.layernorm(x, scale, bias, LayernormAttributes)` → `[y, mean, invVariance]`

**Provider maturity**: API exists (Q1 2026). hip-kernel-provider kernel coverage expanding in Q3 2026.

### 4.5 RMSNorm

**Current dispatch**: `F.rms_norm` → `aten::_fused_rms_norm` → native HIP kernel

**hipDNN mapping**: `graph.rmsnorm(x, scale, RMSNormAttributes)` → `[y, invRms]`

hipDNN/rocKE provides optimized RMSNorm kernels where PyTorch only has a basic native implementation. RMSNorm runs in every transformer layer in modern architectures like LLaMA, so it is called many times per forward pass.

**Provider maturity**: API exists (Q1 2026). hip-kernel-provider expanding in Q3 2026.

### 4.6 SDPA (Scaled Dot-Product Attention)

**Current dispatch**: `F.scaled_dot_product_attention` → backend selection → `aten::_scaled_dot_product_flash_attention` → Flash Attention (AOTriton/CK) or `aten::_scaled_dot_product_efficient_attention` → Memory-Efficient (CK)

**hipDNN mapping**: `graph.sdpa(q, k, v, SdpaAttributes)` → `[o, stats]`

**rocKE connection**: The SDPA engine in hip-kernel-provider is a key rocKE deliverable. rocKE is an agentic kernel authoring platform where kernels are defined as high-level Python code using primitives, then lowered to an IR and compiled to optimized binaries. The Phase 1 injection is the fastest path to validate these kernels against real transformer models in frameworks like ComfyUI and vLLM.

**API features**: GQA, MQA, causal masking, paged attention, FP8, ALiBi, sliding window, BHSD/BSHD layouts. Note: the API defines these but current engine coverage is narrower (basic forward on gfx942/gfx950, BF16/FP8, causal). Wider coverage depends on rocKE engine progress.

**cuDNN shim**: SDPA is the ONLY operation with cuDNN v9 shim support today (gated behind `HIPDNN_ENABLE_SDPA`). PyTorch has existing cuDNN v9 SDPA code (`aten/src/ATen/native/cudnn/MHA.cpp`) that could be hipified through the shim — this is the most mature Tier 4 path.

**Provider maturity**: Q2-Q3 2026 active development. rocKE SDPA plugin in POC stage.

---

## 5. Fusion Opportunities

hipDNN's graph API supports multi-op fusion, which individual per-op library calls cannot express. This section covers how to reach fusion through PyTorch's mechanisms.

### 5.1 TorchInductor Pattern Matcher (with torch.compile)

The `mkldnn_fusion.py` in TorchInductor is the exact template to follow. It demonstrates how to register patterns that match op sequences and replace them with fused backend calls.

**Architecture**:
- `_register_unary_fusion()` matches patterns like `CallFunction(aten.relu, conv_call)` and replaces with a single fused op
- `register_lowering_pattern(pattern, extra_check, pass_dict)` registers the pattern at lowering time
- Fused IR nodes (`ConvolutionUnary`, `ConvolutionBinary`) represent the fused computation

**For hipDNN**, create analogous files:

```python
# hipdnn_fusion.py — modeled after mkldnn_fusion.py

from torch._inductor.pattern_matcher import (
    CallFunction, register_lowering_pattern, Arg, KeywordArg
)

def _register_hipdnn_conv_relu():
    """Match conv + relu and replace with hipDNN fused conv+activation graph."""
    pattern = CallFunction(
        aten.relu,
        CallFunction(aten.convolution, *_conv_args, _users=1)
    )

    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_hipdnn_conv_fusion,
        pass_dict=hipdnn_pass_dict,
    )
    def handler(match, *args, **kwargs):
        # Build hipDNN graph: conv_fprop + pointwise(RELU)
        return L[hipdnn_ops.conv_activation](*args, activation="relu")
```

**Patterns to register**:
- Conv + unary activation (relu, gelu, silu, sigmoid, tanh, hardswish, leaky_relu)
- Conv + add (residual connection) + optional activation
- Linear/matmul + bias + activation (via hipBLASLt epilogue)
- BatchNorm + activation (relu, etc.)

Key reference file: `torch/_inductor/fx_passes/mkldnn_fusion.py` (1591 lines)

### 5.2 Custom Graph Pass (non-invasive torch.compile hook)

For a non-invasive approach that doesn't require modifying PyTorch source:

```python
import torch._inductor.config as config
from torch._inductor.custom_graph_pass import CustomGraphPass

class HipDNNFusionPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        # Walk graph nodes, identify fusible sequences, replace with hipDNN calls
        for node in graph.nodes:
            if node.op == 'call_function' and node.target == torch.ops.aten.convolution.default:
                # Check if next op is relu/activation
                users = list(node.users.keys())
                if len(users) == 1 and users[0].target == torch.ops.aten.relu.default:
                    _replace_with_fused_conv_relu(graph, node, users[0])

    def uuid(self):
        return "hipdnn-fusion-v1"

# Register without modifying PyTorch source:
config.post_grad_custom_pre_pass = HipDNNFusionPass()
```

### 5.3 Eager-Mode Fusion (without torch.compile — for ComfyUI)

Three approaches for eager mode:

**Weight-level conv-BN fusion** (simplest, no kernel change):
```python
from torch.nn.utils.fusion import fuse_conv_bn_eval

# Fold BN params into conv weights at model load time
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        bn = _find_following_bn(model, name)
        if bn is not None:
            fused = fuse_conv_bn_eval(module, bn)
            _replace_module(model, name, fused)
```

This eliminates BN as a separate op. hipDNN then handles just conv+relu (if activation follows).

**`make_fx()` graph capture** (trace once, optimize, run many):
```python
from torch.fx.experimental.proxy_tensor import make_fx

# Trace model into FX graph
traced = make_fx(model, tracing_mode="real")(sample_input)

# Apply fusion passes to the FX graph
apply_hipdnn_fusion_passes(traced.graph)

# Use traced model for inference (reuses fused graph)
output = traced(real_input)
```

**Module-level fused wrappers** (requires model modification):
```python
class HipDNNConvBnReLU(torch.nn.Module):
    def __init__(self, conv, bn, relu):
        super().__init__()
        self.conv = conv; self.bn = bn; self.relu = relu
    def forward(self, x):
        # Build hipDNN graph: conv + bn + relu as single fused kernel
        return _hipdnn_fused_conv_bn_relu(x, self.conv, self.bn)
```

### 5.4 Supported Fused Patterns in hipDNN

Patterns that hipDNN providers support today:

| Pattern | Provider | Implementation |
|---------|----------|---------------|
| Conv + Bias + Activation (ReLU) | MIOpen | `MiopenConvFwdBiasActivPlan` |
| BatchNorm + Activation | MIOpen | Fused BN+Activation plan |
| BatchNorm + DReLU + Backward | MIOpen | Fused backward plan |
| Matmul + Bias | hipBLASLt | Epilogue: `HIPBLASLT_EPILOGUE_BIAS` |
| Matmul + Bias + ReLU | hipBLASLt | Epilogue: `HIPBLASLT_EPILOGUE_RELU_BIAS` |
| Matmul + Bias + GELU | hipBLASLt | Epilogue: `HIPBLASLT_EPILOGUE_GELU_BIAS` |
| Matmul + Bias + Swish | hipBLASLt | Epilogue: fused bias+swish |
| Multi-op arbitrary graphs | hipDNN native | Build any DAG of operations |

**Future**: As rocKE builds more operation kernels and the graph splitting/fallback engine matures (Q4 2026 roadmap), more complex fusion patterns become possible.

### 5.5 HIP Graphs (Complementary)

HIP/CUDA graphs reduce kernel launch overhead by recording and replaying sequences of GPU operations. They are orthogonal to hipDNN's kernel fusion but complementary:

- hipDNN fusion: merges operations into fewer, more efficient kernels
- HIP graphs: eliminates launch overhead by replaying pre-recorded sequences

Both can be used together — hipDNN fused kernels captured inside a HIP graph.

---

## 6. Cross-Cutting Concerns

### 6.1 torch.compile Compatibility

| Mechanism | Survives torch.compile? | Notes |
|---|---|---|
| `Library("aten", "IMPL")` at HIP key | **Yes** | Operates below Dynamo. Best for Phase 1. |
| C++ `TORCH_LIBRARY_IMPL` | **Yes** | Same dispatcher-level registration. |
| `custom_op` + `register_fake` | **Yes** | Op is opaque to compiler (prevents cross-op fusion). |
| `TorchDispatchMode` | **No** | Dynamo skips tracing inside handlers (`_should_skip_dynamo()` defaults to True). |
| Module hooks | **Yes** | Preserved in compiled modules. |
| Monkey-patching | **No** | Dynamo resolves function references at trace time. |
| Pattern matcher (`register_replacement`) | **Native** | Runs as part of Inductor compilation. |
| Custom graph pass (`config.post_grad_custom_pre_pass`) | **Native** | Runs as part of Inductor compilation. |

### 6.2 Autograd Compatibility

hipDNN provides backward variants for all operations: `conv_dgrad`, `conv_wgrad`, `batchnorm_backward`, `layernorm_backward`, `rmsnorm_backward`, `sdpa_backward`.

**Strategy for Phase 1**: Register the forward override at the HIP dispatch key. PyTorch's autograd engine will decompose the backward through the same dispatch — if you also override the backward ATen ops, hipDNN handles both directions.

For inference-only use cases (ComfyUI image generation), backward is not needed. For training, ensure backward ops are also overridden or implement `torch.autograd.Function` with custom forward/backward.

### 6.3 Three Native Backend Paths

There are three distinct paths for deep PyTorch integration, each targeting different parts of the codebase:

#### Path A: cuDNN v9 Shim → SDPA (and future ops)

PyTorch has existing cuDNN v9 graph API code for SDPA (`aten/src/ATen/native/cudnn/MHA.cpp`). hipDNN's cuDNN v9 compatibility shim (`projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/`) allows this code to be hipified with minimal changes — just swapping include paths so `cudnnHandle_t` maps to `hipdnnHandle_t`, etc.

**Status**: The shim skeleton has landed (Q3 2026 in-progress). SDPA is the only operation with shim support today (gated behind `HIPDNN_ENABLE_SDPA`). `graph_wrapper.h` exposes `sdpa()` and `sdpa_backward()` but NOT conv, matmul, or norm methods yet.

**What it gives end users**: hipDNN's multi-engine SDPA architecture (AITER ASM, rocKE, RDNA kernels all available as providers) accessible through PyTorch's native SDPA path — no external injection needed. Engine selection via static ordering or auto-tuning; automatic heuristic selection is future work.

**Risk**: Low. Reuses existing, tested cuDNN v9 code. RFC 0012 explicitly prefers this path.

#### Path B: MIOpen Shim → Convolutions, BatchNorm (transparent upgrade)

PyTorch's ROCm backend calls MIOpen directly for convolutions and batch norm via `aten/src/ATen/native/miopen/Conv_miopen.cpp` and `BatchNorm_miopen.cpp`. The **MIOpen ↔ hipDNN shim** (Q3 2026 roadmap, RFC stage) would make MIOpen route through hipDNN behind the scenes — when a framework calls the MIOpen C API, the shim translates to hipDNN graph operations internally.

**Status**: RFC stage. MIOpen superbuild integration is done. The shim itself is planned but not yet implemented.

**What it gives end users**: hipDNN's multi-engine architecture and graph-level fusion for conv and batch norm **without changing a single line of PyTorch code**. PyTorch continues calling MIOpen as it always has, but MIOpen's implementation is backed by hipDNN. RDNA users benefit because hipDNN can route to RDNA-tuned kernels where MIOpen's current solvers may underperform.

**Risk**: Very low for PyTorch. All changes are inside MIOpen/hipDNN — PyTorch is unaware. However, the MIOpen shim itself is a significant engineering effort.

#### Path C: Native hipDNN Backend → Full Coverage

Build a new native hipDNN backend in PyTorch, either replacing or coexisting with the MIOpen backend. Register implementations for all ATen ops (conv, GEMM, norms, SDPA) that call hipDNN's C++ frontend directly.

**Status**: Not started. The Q3 2026 roadmap item "PyTorch integration for opt-in hipDNN backend" likely refers to a combination of Paths A and B rather than a full native backend.

**What it gives end users**: Full hipDNN API access including graph-level fusion across operations. Not constrained by cuDNN v9 or MIOpen API semantics.

**Risk**: Higher — requires new PyTorch source files, upstream review, and long-term maintenance.

#### Comparison

| Aspect | Path A (cuDNN Shim) | Path B (MIOpen Shim) | Path C (Native) |
|--------|---------------------|---------------------|-----------------|
| Operations | SDPA (today), others later | Conv, BatchNorm | All |
| PyTorch changes | Minimal (hipify include swap) | **None** | New backend files |
| Risk | Low | Very low (for PyTorch) | Higher |
| Status | Shim skeleton landed | RFC stage | Not started |
| RDNA benefit | SDPA on RDNA | Conv/BN on RDNA | All ops on RDNA |
| Multi-engine dispatch | SDPA only (today) | Conv/BN | All ops |
| Timeline | Q3 2026 | Q3-Q4 2026 | Q4 2026+ |

**Recommended approach**: Pursue Paths A and B in parallel — they are complementary and together cover the most compute-intensive operations (SDPA via cuDNN shim, conv/BN via MIOpen shim) with minimal PyTorch changes. Path C is the long-term goal but has the highest bar.

The shim paths are at:
- cuDNN v9 shim: `projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/`
- MIOpen shim: RFC stage (Q3 2026 roadmap)

### 6.4 hipDNN Python Bindings

The `hipdnn_frontend` package exists as a POC (Q1 2026) with a compiled `.so` and tests for conv, matmul, batchnorm, normalization, pointwise.

Key classes and functions:
- `hipdnn.Graph()` — create a new graph
- `hipdnn.Tensor.create(dims, dtype)` — create a tensor descriptor
- `hipdnn.create_handle()` — create a hipDNN handle
- `hipdnn.DeviceBuffer(nbytes)` — allocate device memory
- `hipdnn.DataType.FLOAT`, `.HALF`, `.BFLOAT16` — data type enums
- `hipdnn.ConvFpropAttributes()`, `MatmulAttributes()`, etc. — operation attributes
- Graph methods: `.validate()`, `.build_operation_graph()`, `.create_execution_plans()`, `.check_support()`, `.build_plans()`, `.execute()`, `.get_workspace_size()`

General availability (beyond POC) is planned in the roadmap.

### 6.5 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HIPDNN_PLUGIN_DIR` | `hipdnn_plugins/engines/` (relative to cwd) | Engine plugin search directory |
| `HIPDNN_HEURISTIC_PLUGIN_DIR` | `hipdnn_plugins/heuristics/` | Heuristic plugin search directory |
| `HIPDNN_HEUR_POLICY_ORDER` | `Config,StaticOrdering` | Heuristic policy priority order |
| `HIPDNN_HEUR_CONFIG_PATH` | (unset) | JSON rule file for config-based engine selection |
| `HIPDNN_HEUR_FALLBACK_ENGINE_ORDER` | MIOpen-first | Engine preference for static ordering |
| `HIPDNN_LOG_LEVEL` | `off` | Minimum log severity: `off`, `info`, `warn`, `error`, `fatal` |
| `HIPDNN_LOG_FILE` | stderr | Log output file path |
| `HIPDNN_LOG_GRAPH_DIR` | (disabled) | Write graph JSON files during finalization |

---

## 7. Recommendations & Phasing

### Phase 1 — Python Injection (Days)

**Goal**: Give models access to hipDNN's multi-engine architecture and RDNA support with no model code changes.

> **Two forms of Tier 1.** This package ships the functional-monkeypatch form; the
> ATen-dispatch form is what §3 details. See §3.1 for the two set side by side and when
> each fits. The recommendations in this section apply to Tier 1 in either form.

**Model changes required**: **NONE**. In the ATen-dispatch form, importing the injection
module registers the overrides, and every subsequent PyTorch operation is intercepted
transparently — no model code modifications, no wrapper classes, no config changes. (The
functional-monkeypatch form this package ships is opt-in via `install()` rather than
import-time registration, and is side-effect-free on import; see the package README.)

**Deliverable (ATen-dispatch form)**: A Python module that, when imported, registers ATen op
overrides via `torch.library.Library("aten", "IMPL")` on top of the `hipdnn_frontend` Python
bindings. Graph/plan caching keeps overhead low after warmup. Such a module would:
1. On import: capture the original kernels via `get_kernel()`, then register overrides for each supported ATen op
2. Offer selective control: enable/disable, or restrict to a subset of ops (e.g. `["convolution", "mm"]`)
3. Expose logging to show which ops routed through hipDNN vs. fell back
4. Report statistics: hipDNN hits, fallbacks, and cache size

For the functional-monkeypatch form, the shipped `hipdnn_torch` package provides the
equivalent controls — `install()` / `uninstall()`, op selection, `enable_logging()`, and
`report()` (a per-shape census plus ranked fallback reasons). See the package README.

**What each form covers**:
- On RDNA, routes the covered ops to hipDNN's RDNA-tuned providers where PyTorch's default paths may be unavailable or unoptimized (e.g. Flash Attention, norms)
- On CDNA, exposes hipDNN's multiple providers (AITER, rocKE, MIOpen, hipBLASLt) for the covered ops, selectable via static ordering or auto-tuning
- Graph-level fusion is available where providers support it (conv+activation, matmul+bias+act)
- Fallback logging records which configurations fell back — the data that identifies gaps in provider kernel coverage
- Enables A/B comparison of hipDNN provider routing vs. direct single-library calls

**Limitations**: Python overhead on the cache-miss path. No multi-op fusion. If the hipDNN
Python bindings cannot accept a PyTorch tensor's `data_ptr()` directly, DeviceBuffer memory
management adds copies.

### Model Changes Required — Comparison Across Phases

| Phase | Model Code Changes | How to Activate |
|-------|-------------------|-----------------|
| **Phase 1 — Python** | **None** | `import hipdnn_torch` before model load |
| **Phase 2 — C++ ext** | **None** | `import hipdnn_torch_native` (loads .so) |
| **Phase 3 — Inductor** | **None** | `config.post_grad_custom_pre_pass = HipDNNFusionPass()` before `torch.compile()` |
| **Phase 4 — Backend** | **None** | Built into PyTorch ROCm — works automatically |

All phases are transparent to model code. The only difference is what you import or configure before running the model.

### Phase 2 — C++ Extension (Weeks)

**Goal**: Production-grade performance with C++-level graph caching.

**Model changes required**: **NONE**. Same import-based activation as Phase 1, but the import loads a compiled C++ extension instead of pure Python overrides.

**Deliverable**: A HIP C++ extension built with `torch.utils.cpp_extension.load()`. Registers via `TORCH_LIBRARY_IMPL(aten, HIP, m)`. Uses `std::unordered_map` for graph cache with near-zero lookup overhead. Direct pointer passing between PyTorch tensors and hipDNN — no copies.

**Additions**:
- Eager-mode conv-BN weight fusion (`fuse_conv_bn_eval`)
- Optional module-level fused wrappers for common patterns
- Performance profiling integration (HIP event timing)

### Phase 3 — TorchInductor Fusion Passes (Weeks-Months)

**Goal**: Enable hipDNN's graph-level fusion.

**Deliverable**: `hipdnn_fusion.py` + `hipdnn_ir.py` modeled after `mkldnn_fusion.py`. Pattern matcher registrations for conv+activation, linear+activation, matmul+bias+activation. Custom graph pass via `config.post_grad_custom_pre_pass` for non-invasive deployment.

**What it adds**: Multi-op fused kernels that cannot be achieved by overriding individual ATen ops. This is what hipDNN's graph API expresses that the per-op PyTorch dispatch path cannot.

### Phase 4 — Native Backend (Months)

**Goal**: hipDNN becomes part of PyTorch's ROCm backend — all models use it automatically, just like cuDNN on NVIDIA.

**Model changes required**: **NONE**.

**Three parallel paths** (see Section 6.3 for full analysis):

| Path | Target Ops | PyTorch Changes | Timeline |
|------|-----------|----------------|----------|
| **A — cuDNN v9 shim** | SDPA (today), expand to others | Minimal (hipify include swap) | Q3 2026 |
| **B — MIOpen shim** | Conv, BatchNorm | **None** (transparent to PyTorch) | Q3-Q4 2026 |
| **C — Native backend** | All operations | New source files | Q4 2026+ |

**Recommended approach**: Pursue Paths A and B in parallel:
- **Path A**: Hipify PyTorch's cuDNN v9 SDPA code (`aten/src/ATen/native/cudnn/MHA.cpp`) via hipDNN's cuDNN compatibility shim. Shim skeleton has landed (Q3 2026 in-progress). Immediately gives RDNA users optimized Flash Attention and CDNA users multi-engine SDPA dispatch.
- **Path B**: The MIOpen ↔ hipDNN shim (RFC stage, Q3 2026 roadmap) makes MIOpen route through hipDNN behind the scenes. PyTorch continues calling MIOpen as always, but conv and batch norm gain access to hipDNN's multi-engine architecture and RDNA-tuned kernels — zero PyTorch changes required.
- **Path C** is the long-term goal for full coverage and graph-level fusion, but has the highest upstream review bar.

---

## 8. Operation Gap Analysis: What hipDNN Needs for Modern Models

Beyond integrating hipDNN's current operations into PyTorch and vLLM, there are operations and fusion patterns that hipDNN should add to deliver best performance for modern LLM and diffusion model architectures. This analysis also applies to the vLLM integration (see `vllm-integration-techniques.md`).

### 8.1 Critical Gaps — Blocking Modern LLM Support

#### RoPE (Rotary Position Embeddings)

**Used by**: LLaMA, Mistral, Qwen, Phi, Gemma, DeepSeek — virtually all modern LLMs.

**hipDNN status**: NOT supported. No dedicated operation, no mention in the codebase.

**Why it can't be expressed with existing ops**: RoPE applies `x * cos(theta) + rotate_half(x) * sin(theta)` where `rotate_half` interleaves/swaps pairs of elements in the head dimension. While MUL, ADD, SIN exist as pointwise modes, the `rotate_half` permutation requires element reordering that no current pointwise mode can express. Even with creative striding, it would be unacceptably slow without a fused kernel.

**Recommendation**: Add a dedicated `HIPDNN_OPERATION_TYPE_ROPE_EXT` with support for both interleaved and non-interleaved formats, position offset (for KV cache continuation), and multi-frequency base (for NTK-aware RoPE scaling). Ideally fused with QKV projection or KV cache update.

#### MoE (Mixture of Experts)

**Used by**: Mixtral, DeepSeek-MoE/V2/V3, Qwen-MoE, Grok, Arctic.

**hipDNN status**: NOT supported. MOE frontend/backend POC is planned for Q3 2026.

**What's needed**: Router gating (matmul + softmax/top-k), token dispatch (scatter to experts), grouped/variable-batch GEMM (each expert processes different token counts), weighted aggregation (scatter-add back). This is a multi-operation pattern that needs new operation types or a composite graph pattern.

#### Per-Token / Per-Group Quantization and Dequantization

**Used by**: FP8 inference (per-token dynamic quantization), GPTQ (per-group INT4), AWQ (per-group INT4).

**hipDNN status**: Only MX block-scale (per-32-element block) quantize/dequantize is supported. Traditional per-tensor, per-channel, per-token, and per-group quantization modes are NOT available as dedicated operations.

**What's needed**: Quantize/dequantize operations with configurable granularity (per-tensor, per-channel, per-token, per-group) and support for INT4 unpacking (for GPTQ/AWQ weight dequantization). Without this, hipDNN cannot serve quantized models — the dominant deployment scenario.

### 8.2 Critical Gaps — Blocking Modern Diffusion Model Support

#### AdaLayerNorm (Adaptive Layer Normalization)

**Used by**: DiT, Stable Diffusion 3, Flux, Hunyuan — all modern diffusion transformers.

**Formula**: `scale * LayerNorm(x) + shift` where scale/shift come from conditioning (timestep or class embedding).

**hipDNN status**: Expressible as 3-node graph (`layernorm` → `pointwise MUL` → `pointwise ADD`), but without guaranteed kernel fusion this has 3x the kernel launch overhead and 3x the memory traffic vs. a fused kernel.

**Recommendation**: Either a dedicated `HIPDNN_OPERATION_TYPE_ADALAYERNORM_EXT` or guaranteed graph fusion of layernorm + pointwise_mul + pointwise_add. The former is cleaner; the latter would also benefit other patterns.

#### GroupNorm

**Used by**: U-Net diffusion models (SD 1.x, SDXL, ControlNet).

**hipDNN status**: NOT supported. No operation type, no graph method. Not expressible via existing ops because it requires reshaping `[B, C, H, W]` → `[B*G, C/G, H, W]` before normalization.

**Recommendation**: Add `HIPDNN_OPERATION_TYPE_GROUPNORM_EXT`. While the trend is toward DiT (which uses AdaLayerNorm), U-Net models remain widely deployed.

#### Upsampling (Nearest-Neighbor, Bilinear)

**hipDNN status**: Only pooling modes (maxpool, avgpool) exist in resample. No nearest-neighbor or bilinear upsampling.

**Recommendation**: Add upsampling modes to the existing resample operation type. Needed for U-Net architectures.

### 8.3 Important Performance Optimizations

#### Fused Gate+Up Projection (Split-Output GEMM)

Modern LLM FFN blocks (LLaMA SwiGLU) compute `SiLU(x @ W_gate) * (x @ W_up)` where `W_gate` and `W_up` are often concatenated into a single weight matrix. A split-output GEMM that produces both halves in one kernel launch would halve the GEMM launches in every FFN block.

**hipDNN status**: Not expressible. hipBLASLt would need split-output GEMM or hipDNN would need a way to express "one matmul, two output column ranges."

#### Fused Residual + Norm

`rms_norm(x + residual)` or `layer_norm(x + residual)` appears at every layer boundary. Fusing the pointwise ADD with the subsequent normalization eliminates one full activation read/write.

**hipDNN status**: Expressible as 2-node graph (pointwise ADD → rmsnorm), but fusion is not guaranteed.

**Recommendation**: Ensure the graph fusion engine can fuse pointwise ADD followed by normalization into a single kernel.

#### Fused RMSNorm + Quantize

For FP8 inference: normalize activations and quantize to FP8 in one pass, avoiding a full-precision intermediate write.

**hipDNN status**: RMSNorm and block_scale_quantize exist separately. Graph-level fusion between them is not documented.

#### Fused Attention + Output Projection

SDPA produces output O, then a separate matmul projects it. Fusing these eliminates one memory round-trip per layer.

**hipDNN status**: Not expressible as a single fused op. Graph fusion might compose these if supported.

### 8.4 Nice-to-Have Additions

| Operation | Notes |
|-----------|-------|
| Quick GELU (`x * sigmoid(1.702x)`) | Used in CLIP/ViT. Expressible as 3 pointwise ops but a dedicated mode would be cleaner. |
| Softmax as a dedicated op | Currently must be built from reduction + exp + div. Needed for MoE router. |
| Speculative decoding primitives | Token verification + acceptance sampling. Can be external. |

### 8.5 cuDNN v9 Parity Analysis

Before deciding whether each gap needs a custom hipDNN design or can match cuDNN's approach, here's what cuDNN v9 (frontend 1.26.0, backend 9.24+) supports for these operations:

| Operation | cuDNN v9 Status | hipDNN Strategy |
|-----------|----------------|-----------------|
| **RoPE** | No standalone op — composed via pointwise or within SDPA | Custom hipDNN op (neither library has it natively) |
| **MoE Grouped GEMM** | Extensively supported — grouped GEMM + SwiGLU, + GLU, + quant, + Hadamard, per-expert reduction | **Match cuDNN** — add MoE grouped matmul op (Q3 2026 POC planned) |
| **AdaLayerNorm** | Enum defined (`CUDNN_ADA_LAYER_NORM = 5`), support status unclear | **Match cuDNN** — add norm mode enum value, implement kernel |
| **GroupNorm** | Enum defined (`CUDNN_GROUP_NORM = 3`) but NOT yet supported | **Match cuDNN** — add enum, implement kernel (cuDNN hasn't shipped this yet either) |
| **Fused RMSNorm + SiLU** | Supported as open-source CuTe DSL kernel, auto-detected pattern | **Match cuDNN** — graph fusion pattern or dedicated op |
| **Fused RMSNorm + RHT + amax** | Supported (Blackwell SM100+) | Future — architecture-specific |
| **Fused GEMM + SwiGLU** | Supported via MoE grouped GEMM + SwiGLU fusion | **Match cuDNN** — part of MoE grouped GEMM work |
| **Per-token/per-group quant** | Supported via per-row gating in grouped GEMM + quant | **Match cuDNN** — add quantization granularity modes |
| **Causal Conv1d** | On Q3 2026 cuDNN roadmap | Watch and match if needed (Mamba/state-space models) |
| **Reshape modes** | On Q3 2026 cuDNN roadmap | Watch and match |
| **Strided slice** | On Q3 2026 cuDNN roadmap | Watch and match |

**Key takeaway**: cuDNN is expanding its MoE support (grouped GEMM with many fusion patterns) and adding graph manipulation operations (reshape, slice, concat). MoE grouped GEMM is where the feature gap between the two libraries is currently widest, and it is what the cuDNN compatibility shim will need matching ops for. RoPE is absent from both libraries.

Sources:
- [cuDNN Q3 2026 Roadmap](https://github.com/NVIDIA/cudnn-frontend/issues/442)
- [cuDNN Backend Release Notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/release-notes.html)
- [cuDNN Frontend Graph API](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/graph-api.html)
- [cuDNN Frontend GitHub](https://github.com/NVIDIA/cudnn-frontend)
- [Fused RMSNorm + SiLU](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/rmsnorm_silu.html)

### 8.6 Recommended Priority for New Operations

| Priority | Operation | Impact | cuDNN Has It? | Roadmap Status |
|----------|-----------|--------|--------------|---------------|
| **P0** | MoE grouped GEMM + fusions | Dominant scaling architecture (Mixtral, DeepSeek) | **Yes** (extensive) | Q3 2026 POC |
| **P0** | RoPE | Every modern LLM, every token, every layer | **No** (neither library) | Not planned |
| **P0** | Per-token/per-group quant + GPTQ/AWQ dequant | Required for serving quantized models | **Partial** (per-row in GEMM) | Not planned |
| **P0** | AdaLayerNorm | Every modern diffusion transformer (DiT, SD3, Flux) | **Enum defined**, unclear support | Not planned |
| **P1** | Fused RMSNorm + SiLU | LLM FFN activation fusion | **Yes** (auto-detected pattern) | Not planned |
| **P1** | Fused gate+up GEMM (split-output) | Halves FFN GEMM launches in LLMs | **Yes** (via MoE grouped GEMM + SwiGLU) | Not planned |
| **P1** | Fused residual+norm | Every layer boundary in transformers | Implicit via graph fusion | Not planned |
| **P1** | GroupNorm | U-Net diffusion models | **Enum defined, not yet supported** | Not planned |
| **P1** | Fused RMSNorm+quantize | FP8 inference pipelines | Partial (GEMM + quant) | Not planned |
| **P1** | Upsample (nearest/bilinear) | U-Net diffusion models | Via resample | Not planned |
| **P1** | FP8 convolution | Diffusion model acceleration | Via runtime fusion engine | Not planned |
| **P2** | Quick GELU mode | CLIP/ViT | No dedicated mode | Not planned |
| **P2** | Softmax dedicated op | MoE router, general use | Via graph pattern | Not planned |
| **P2** | Causal Conv1d | Mamba/state-space models | **On Q3 2026 cuDNN roadmap** | Not planned |
| **P2** | Reshape / strided slice / concat ops | Graph manipulation | **On Q3 2026 cuDNN roadmap** | Not planned |

### 8.6 What's Already Well Covered

For balance — hipDNN's current coverage is solid for the core compute patterns:

- **SDPA** with extensive API surface (GQA, paged attention, causal, ALiBi, sliding window, FP8) — current engine coverage is basic forward (BF16/FP8, causal, gfx942/gfx950); wider feature coverage depends on rocKE engine progress
- **Dense GEMM** with fused epilogues (bias + ReLU/GELU/Swish)
- **Convolution** (forward, dgrad, wgrad) with fused conv+bias+activation
- **RMSNorm and LayerNorm** (forward and backward)
- **BatchNorm** (training, inference, backward) with fused BN+activation
- **47 pointwise operations** covering most activation functions and elementwise ops
- **Block-scale FP8/FP4/FP6 quantize/dequantize** (MX formats)
- **Reduction** operations (sum, mean, max, norms)
- **Resample** (pooling)

The gaps are primarily in (1) operations specific to modern architecture innovations (RoPE, MoE, AdaLN), (2) quantization formats used in production inference (GPTQ, AWQ, per-token FP8), and (3) guaranteed graph-level fusion of common multi-op patterns.

---

## Appendix A: hipDNN Provider Support Matrices

### MIOpen Provider

| Operation | Datatypes | Layouts | Notes |
|-----------|-----------|---------|-------|
| Conv Forward | FP16, BF16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Deterministic engine available |
| Conv Backward Data | FP16, BF16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Deterministic engine available |
| Conv Backward Weights | FP16, BF16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Deterministic engine available |
| Conv + Bias + Activation | FP16, BF16, FP32 | NCHW, NHWC, NCDHW, NDHWC | Fused; activation = ReLU only |
| BatchNorm Inference | FP16, BF16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Spatial mode only |
| BatchNorm Training | FP16, BF16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Spatial mode only |
| BatchNorm Backward | FP16, BF16, FP32 | NCL, NLC, NCHW, NHWC, NCDHW, NDHWC | Spatial mode only |
| BN + Activation (fused) | FP16, BF16, FP32 | All above | Fused graph |
| BN + DReLU + Backward (fused) | FP16, BF16, FP32 | All above | Fused graph |

### hipBLASLt Provider

| Operation | Datatypes | Notes |
|-----------|-----------|-------|
| Stand-alone Matmul | FP32, FP16, BF16 (compute: FP32) | Transposed, batched (equal or broadcast) |
| Matmul + Bias | FP32, FP16, BF16 | Fused epilogue |
| Matmul + ReLU | FP32, FP16, BF16 | Fused epilogue |
| Matmul + Bias + ReLU | FP32, FP16, BF16 | Fused epilogue |
| Matmul + Bias + GELU (tanh) | FP32, FP16, BF16 | Fused epilogue |
| Matmul + Bias + Swish | FP32, FP16, BF16 | Fused epilogue (unit beta) |
| FP8 OCP BlockScale + GEMM | FP8_E4M3, FP8_E5M2 | gfx950/gfx1250 only; K%128==0 |

### hip-kernel-provider (rocKE)

| Operation | Status | Notes |
|-----------|--------|-------|
| SDPA Forward | Q2-Q3 2026 active dev | Current engine: BF16/FP8, causal, gfx942/gfx950. Wider coverage (paged, ALiBi, RDNA) depends on rocKE progress. |
| SDPA Backward | Q2-Q3 2026 active dev | |
| BatchNorm (HIP MLOPS) | Available | Forward train/infer, backward |
| LayerNorm (HIP MLOPS) | Expanding Q3 2026 | Forward |
| RMSNorm (HIP MLOPS) | Expanding Q3 2026 | Forward + backward |
| Resample | Available | Forward + backward |

## Appendix B: Key File Paths

| Purpose | Path |
|---------|------|
| hipDNN Frontend Graph API | `projects/hipdnn/frontend/include/hipdnn_frontend/Graph.hpp` |
| hipDNN Backend C API | `<rocm-sdk>/include/hipdnn/backend/hipdnn_backend.h` |
| Operation types enum | `<rocm-sdk>/include/hipdnn/backend/HipdnnOperationType.h` |
| SDPA attributes | `projects/hipdnn/frontend/include/hipdnn_frontend/attributes/SdpaAttributes.hpp` |
| cuDNN v9 shim | `projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/` |
| cuDNN shim RFC | `projects/hipdnn/docs/rfcs/0012_CuDNN_Shim.md` |
| MIOpen provider support | `dnn-providers/miopen-provider/docs/OperationSupport.md` |
| hipBLASLt provider support | `dnn-providers/hipblaslt-provider/docs/OperationSupport.md` |
| hip-kernel-provider (rocKE) | `dnn-providers/hip-kernel-provider/` |
| Python bindings | `projects/hipdnn/python/frontend_bindings/` |
| Python binding tests | `projects/hipdnn/python/frontend_wheel_package/tests/` |
| hipDNN Roadmap | `projects/hipdnn/docs/Roadmap.md` |
| Environment variables | `projects/hipdnn/docs/Environment.md` |
| MKLDNN fusion template | `torch/_inductor/fx_passes/mkldnn_fusion.py` |
| Pattern matcher API | `torch/_inductor/pattern_matcher.py` |
| Custom graph pass API | `torch/_inductor/custom_graph_pass.py` |
| torch.library API | `torch/library.py` |
| ComfyUI ops | `comfy/ops.py` |
