# vLLM → hipDNN integration techniques

> **Point-in-time design reference.** This analysis was captured in 2026. Roadmap items,
> quarter dates, provider maturity, backend names, and version numbers below reflect that
> snapshot and will age — treat them as direction, not current status.

> **How this relates to the samples in this package.** The `hipdnn_torch` package targets
> PyTorch's `torch.nn.functional` path. vLLM deliberately bypasses that path for its
> hottest operations (see *Key Differences from PyTorch Integration* below), so the
> functional monkeypatch the samples ship does **not** cover vLLM — a vLLM-specific
> integration is required. This document charts that path, from a `ROCM_HIPDNN`
> attention-backend plugin to a first-class backend.

A survey of approaches for routing vLLM operations through hipDNN on ROCm — from quick
validation to a production backend.

## How vLLM's structure maps to hipDNN

vLLM currently maintains several separate AMD attention backends on ROCm (e.g. ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, ROCM_AITER_MLA, ROCM_ATTN, TRITON_ATTN). Each wraps a different kernel library (AITER/CK, Triton, native HIP), and coverage differs by architecture — AITER is MI3XX-only, so RDNA consumer GPUs have no AITER path.

hipDNN's design lets a single `ROCM_HIPDNN` backend sit in front of these cases. Its relevant properties:

1. **Multi-engine architecture (capability vs. today)**: hipDNN's plugin architecture *can* host multiple kernel providers for the same operation — for SDPA, that could eventually include AITER ASM, rocKE, and RDNA-tuned kernels. In practice today the provider set for vLLM's operations is thin: the rocKE ASM SDPA engine covers basic forward on gfx942/gfx950, and AITER-as-a-hipDNN-plugin and RDNA providers do not yet exist. The multi-engine benefit is therefore largely prospective — it depends on those plugins being written. Engine selection today has two modes, neither of which automatically picks the best engine per problem: (a) configurable static ordering (`HIPDNN_HEUR_FALLBACK_ENGINE_ORDER`), and (b) auto-tuning, which benchmarks the available engines for a given problem, writes an output file with the results, which you then convert into a rules file to make the selection persistent. Automatic heuristic/analytic selection is planned but not implemented.

2. **Integrate once, add providers over time**: When new kernel capabilities land — a faster SDPA, improved FP8 GEMM from hipBLASLt, new attention variants — they can be added to hipDNN as provider plugins and reach vLLM without vLLM-side code changes: no new API calls, no Python bindings to build, no model adaptation. vLLM keeps calling the same `ROCM_HIPDNN` backend; the new capability appears as an additional engine option. New fusion patterns (attention+norm, matmul+activation) are served through the same graph API. To actually select a newly added provider, you re-run auto-tuning and regenerate the rules file — there is no automatic re-selection. This is a looser version of how cuDNN works on NVIDIA, where the framework integrates once and kernel improvements shipped in cuDNN updates reach it without framework changes.

3. **RDNA enablement**: On RDNA consumer GPUs (RX 7900, RX 9070 series), AITER has no support or roadmap and CK Flash Attention does not currently compile, so several of vLLM's ROCm attention backends do not run there. hipDNN with RDNA-tuned providers is a path to running optimized attention on these Wave32 GPUs through the same backend.

4. **Graph-level fusion**: hipDNN's graph API enables multi-op fusion (attention+norm, matmul+bias+activation) that reduces kernel launches; this matters most in the decode phase, where launch overhead is a larger fraction of total time. New fusion patterns added to providers are available through the same API — no vLLM changes needed per fusion.

### Key Differences from PyTorch Integration

The PyTorch ATen injection approach (`torch.library` overrides of `aten::convolution`, `aten::mm`, etc.) does **NOT** automatically cover vLLM. vLLM deliberately bypasses ATen dispatch for performance-critical operations:

| Operation | Goes through ATen? | vLLM Custom Path |
|-----------|-------------------|------------------|
| Dense GEMM (F.linear) | **Yes** | `aten::addmm` → hipBLASLt |
| Attention | **No** | `torch.ops._C.*` / `torch.ops._rocm_C.*` custom ops |
| RMSNorm | **No** | Custom fused ops (often fused with quant or residual) |
| Quantized GEMM | **No** | Custom Triton/HIP/AITER kernels |
| RoPE | **No** | Custom fused kernels |
| MoE dispatch | **No** | Custom fused CUTLASS/AITER kernels |

A vLLM-specific integration strategy is required. The PyTorch injection provides partial coverage (dense GEMM, some standalone norms) but misses the most compute-intensive operations, such as attention and quantized GEMM.

---

## 1. Executive Summary & Decision Matrix

### Integration Tiers

Note: Standalone GEMM through hipDNN is not a recommended integration point for vLLM today. hipDNN's only GEMM provider is hipBLASLt — the same library vLLM already calls directly. Routing GEMM through hipDNN adds indirection with no performance uplift.

GEMM through hipDNN becomes interesting once there are multiple providers to choose between. There is one concrete, near-term opportunity:

**rocBLAS provider for RDNA**: Investigation confirms rocBLAS (Tensile) significantly outperforms hipBLASLt on current RDNA GPUs. On gfx1151, rocBLAS was **1.7x faster** for a 4096x4096 GEMM (115ms vs 194ms) because hipBLASLt lacked properly tuned kernels. rocBLAS internally has a passthrough to hipBLASLt (`ROCBLAS_USE_HIPBLASLT` env var) — on RDNA 4 (gfx12xx) hipBLASLt is now the default within rocBLAS, but on RDNA 3/3.5 rocBLAS Tensile remains faster.

What rocBLAS lacks vs hipBLASLt: no FP8/FP4/FP6 types, no fused epilogues (bias+activation), no grouped GEMM (MoE). However, the missing narrow types (FP8, FP4, FP6) require gfx950/gfx1250 hardware and are irrelevant for current RDNA. For standard FP16/BF16 GEMM — the dominant precision on RDNA — rocBLAS is the better choice today.

A hipDNN rocBLAS GEMM provider favored on RDNA alongside hipBLASLt on CDNA would be a practical, near-term example of multi-engine GEMM delivering real uplift. The architecture-based routing maps naturally to hipDNN's heuristic system. hipBLASLt remains the provider for advanced features (fused epilogues, grouped GEMM, FP8).

Other future paths:
- **rocKE GEMM kernels** targeting specific shapes or areas of poor performance in existing libraries
- **MoE grouped GEMM** when hipDNN adds MoE support

None of these exist today as hipDNN providers, so standalone GEMM as a first integration step still brings no value for vLLM. But the rocBLAS-on-RDNA performance gap is concrete and documented — adding a rocBLAS GEMM provider should be considered.

| Tier | Approach | Effort | Model Changes | Primary Value |
|------|----------|--------|--------------|---------------|
| **1 — Attention plugin** | `ROCM_HIPDNN` backend via `register_backend` | 1-2 weeks | None (config flag) | Single SDPA entry point (multi-engine as providers land) + RDNA path |
| **2 — Multi-op coverage** | Attention plugin + RMSNorm/LayerNorm overrides | 2-4 weeks | None | Extends hipDNN across more of the forward pass |
| **3 — Fusion integration** | Graph-level fusions (attn+norm, matmul+act) | Weeks-Months | None | Performance uplift from reduced kernel launches |
| **4 — Full backend** | First-class hipDNN platform backend in vLLM | Months | None | Complete integration with vLLM IR dispatch |

### Per-Operation Coverage

| Operation | hipDNN API | Provider | Maturity | Recommended Tier |
|-----------|-----------|----------|----------|-----------------|
| SDPA/Flash Attention | `graph.sdpa()` | rocKE | Q2-Q3 2026 | Tier 1 (attention plugin) |
| Paged Attention | `graph.sdpa()` + `set_paged_attention_k/v_table()` | rocKE | Q2-Q3 2026 | Tier 1 |
| RMSNorm | `graph.rmsnorm()` | hip-kernel | Expanding Q3 2026 | Tier 2 |
| LayerNorm | `graph.layernorm()` | hip-kernel | Expanding Q3 2026 | Tier 2 |
| Fused matmul+activation | `graph.matmul()` + `graph.pointwise()` | hipBLASLt epilogue | Production | Tier 3 (only as fused pattern) |
| FP8 GEMM (OCP) | `graph.matmul()` + block_scale_dequantize | hipBLASLt | gfx950/gfx1250 | Tier 3 (when FP8 adds value over direct hipBLASLt) |
| Quantized GEMM (AWQ/GPTQ) | N/A | N/A | N/A | Stays in custom kernels |
| Fused RoPE | N/A | N/A | N/A | Stays in custom kernels |
| MoE dispatch | MOE POC Q3 2026 | N/A | Planned | Future |

---

## 2. LLM Inference Operations & hipDNN Coverage

### 2.1 Common LLM Architectures and Their Operation Profiles

**Dense Transformer (LLaMA, Qwen, Mistral)**:
Each layer: RMSNorm → QKV Linear → RoPE → Attention → O Linear → RMSNorm → Gate/Up Linear → SiLU → Down Linear → Residual Add

Key ops: SDPA (with GQA), dense GEMM, RMSNorm, SiLU activation, RoPE

**Mixture of Experts (Mixtral, DeepSeek-V2 dense+MoE)**:
Same as dense but FFN replaced with: Router → Top-K dispatch → Expert GEMMs → Aggregation

Key ops: Everything from dense + MoE routing + grouped/batched GEMM

**Multi-Latent Attention (DeepSeek-V3/R1 MLA)**:
Compressed KV heads with latent projection: Q/KV compression → Attention with reduced head dim → Up-projection

Key ops: Specialized attention pattern, different head dimensions for K vs Q

### 2.2 What hipDNN Covers Today

**Important distinction**: hipDNN has two layers — the **frontend graph API** (which defines what operations can be expressed) and the **engine providers** (which implement actual GPU kernels). An operation in the API with no engine that supports it cannot be executed. The analysis below distinguishes between these.

**Supported (API + engine available)**:
- SDPA/Flash Attention — basic forward path on gfx942/gfx950 (BF16 and FP8_E4M3 inputs, BF16 output, causal masking, rank-4 tensors). The rocKE ASM SDPA engine is the current provider.
- Dense GEMM (FP16/BF16/FP32) with fused epilogues (bias+activation) — via hipBLASLt provider
- Convolution (forward/dgrad/wgrad) — via MIOpen provider (FP16/BF16/FP32)
- BatchNorm (training/inference/backward) — via MIOpen and hip-kernel providers
- RMSNorm, LayerNorm (forward and backward) — via hip-kernel provider (coverage expanding Q3 2026)
- 47 pointwise operations — via graph fusion patterns
- Block-scale FP8 quantize/dequantize (MX formats) — via hipBLASLt provider

**In the API but NOT yet supported by any engine**:
- Paged attention — `set_paged_attention_k/v_table()` exists in SdpaAttributes but the current ASM SDPA engine explicitly rejects it
- ALiBi positional encoding — API exists, rejected by current engine
- Attention bias (additive mask) — API exists, rejected by current engine
- Dropout in SDPA — API exists, rejected by current engine
- Stats output (logsumexp for backward) — API exists, rejected by current engine
- Sliding window — API exists, engine support unverified
- SDPA on RDNA — current ASM SDPA engine requires gfx942 or gfx950 only

**The rocKE factor**: rocKE is the expected path to filling these engine gaps — its agentic kernel authoring platform is designed to produce SDPA kernels with wide configuration coverage (paged attention, ALiBi, variable sequence lengths, RDNA architectures, etc.). However, rocKE is a work in progress and what specific capabilities will be available when remains an open question. The hipDNN graph API is ready to accept these features the moment rocKE engines land — no API-side work is needed, only engine implementation.

**Implications for vLLM integration planning**: The viability of each integration phase depends on rocKE engine availability:
- A `ROCM_HIPDNN` attention backend is feasible today for basic SDPA (non-paged, BF16/FP8, causal, gfx942/gfx950) — useful for prefill-only benchmarking
- Paged attention engine support is the prerequisite for decode-phase coverage — and decode is the majority of inference compute. This is the key capability to track from rocKE.
- RDNA SDPA engine support is what enables the consumer-GPU use case
- Integration work can proceed in parallel with rocKE development: build the vLLM backend framework, caching, and plumbing now, and plug in rocKE engines as they become available

**Not covered** (stays in custom kernels):
- Fused RoPE (not in hipDNN's operation set)
- MoE token dispatch/routing/aggregation (MOE POC planned Q3 2026)
- Quantized GEMM formats: AWQ (INT4), GPTQ (INT4), GGUF formats
- Fused RMSNorm + static quantization (custom vLLM fusion)
- Fused all-reduce + RMSNorm (distributed-specific)
- KV cache block allocation/management (memory management, not compute)


---

## 3. vLLM Architecture on ROCm

### 3.1 Dispatch Flow: Request to GPU Kernel

```
User request (prompt text)
  │
  ▼
vLLM Scheduler (CPU) ── batch requests, allocate KV cache blocks
  │
  ▼
ModelRunner.execute_model() ── prepare inputs, select prefill/decode
  │
  ▼
Model forward pass (e.g., LlamaForCausalLM)
  │
  ├── Embedding lookup (standard PyTorch)
  │
  ├── For each transformer layer:
  │     │
  │     ├── RMSNorm ─────────── AITER custom op (MI3XX) or fused_add_rms_norm
  │     │                        or PyTorch aten::_fused_rms_norm (fallback)
  │     │
  │     ├── QKV Linear ──────── F.linear → aten::addmm → hipBLASLt/rocBLAS
  │     │                        or AITER linear (if VLLM_ROCM_USE_AITER_LINEAR=1)
  │     │
  │     ├── RoPE ────────────── Custom fused kernel (torch.ops._C.*)
  │     │
  │     ├── Attention ───────── Backend selection:
  │     │     │                  ROCM_AITER_FA (MI3XX: CK prefill + ASM decode)
  │     │     │                  ROCM_AITER_UNIFIED_ATTN (MI3XX: unified path)
  │     │     │                  ROCM_AITER_MLA (MI3XX: DeepSeek MLA)
  │     │     │                  TRITON_ATTN (all ROCm: Triton kernels)
  │     │     │                  ROCM_ATTN (legacy: Triton prefill + HIP decode)
  │     │     └────────────────  Paged KV read/write via block tables
  │     │
  │     ├── O projection ────── Same as QKV (F.linear path)
  │     ├── RMSNorm ─────────── Same as above
  │     ├── Gate+Up Linear ──── Same as QKV (or MoE dispatch for MoE models)
  │     ├── SiLU activation ─── Custom silu_and_mul kernel
  │     ├── Down Linear ─────── Same as QKV
  │     └── Residual add ────── Standard PyTorch
  │
  ├── Final RMSNorm
  └── LM Head (Linear) → Logits → Sampling → Output tokens
```

### 3.2 Attention Backend System

vLLM's attention backend selection on ROCm (as of mid-2026):

| Backend | Architectures | Prefill Path | Decode Path |
|---------|--------------|-------------|-------------|
| ROCM_AITER_FA | gfx942/gfx950 (MI3XX) | flash_attn_varlen_func (CK) | AITER ASM paged attention |
| ROCM_AITER_UNIFIED_ATTN | gfx942/gfx950 | Unified kernel (preferred on MI3XX) | Same kernel |
| ROCM_AITER_MLA | gfx942/gfx950 | MLA-specific attention | MLA decode |
| TRITON_ATTN | All ROCm | Triton Flash Attention | Triton decode |
| ROCM_ATTN | All ROCm (legacy) | Triton prefill | HIP paged attention (`torch.ops._rocm_C.paged_attention`) |

**Backend registration API** (vLLM 0.13.0+):
```python
# Via CLI
--attention-config.backend ROCM_HIPDNN

# Via API
from vllm.attention.backends.registry import register_backend
register_backend("ROCM_HIPDNN", HipDNNAttentionBackend)

# Via plugin system
class MyPlugin:
    def get_attn_backend_cls(self):
        return HipDNNAttentionBackend
```

### 3.3 AITER Integration

AITER (AI Tensor Engine for ROCm) provides optimized kernels for MI300/MI325/MI355:
- Attention: CK-based Flash Attention + optimized ASM decode kernels
- GEMM: Optimized hipBLASLt wrapper + FP8 BMM
- RMSNorm: Fused kernels (standalone and fused with quantization)
- MoE: Fused expert dispatch

**Critical limitation**: AITER is MI3XX-only. No RDNA support, no roadmap for RDNA. This means consumer AMD GPUs (RX 7900, RX 9070 series) get no AITER acceleration.

Control: `VLLM_ROCM_USE_AITER=1` (master switch) with sub-flags:
- `VLLM_ROCM_USE_AITER_LINEAR` — AITER for linear layers
- `VLLM_ROCM_USE_AITER_MOE` — AITER for MoE
- `VLLM_ROCM_USE_AITER_RMSNORM` — AITER for RMSNorm

### 3.4 Why vLLM Needs Native Integration

vLLM's performance-critical operations bypass PyTorch's ATen dispatch, using custom ops (`torch.ops._C.*`, `torch.ops._rocm_C.*`) instead. This means a vLLM-specific integration is required — the PyTorch ATen injection technique from the companion document does not reach vLLM's attention, norms, or quantized GEMM paths. The primary integration point is vLLM's attention backend plugin system, with custom op overrides for norms as the next tier.

---

## 4. Integration Approaches — Quick to Production

### 4.1 Reference: the functional-monkeypatch samples in this package

The `hipdnn_torch` package (see the package README and `samples/`) patches
`F.linear` / `F.rms_norm` / `F.scaled_dot_product_attention` to route onto a hipDNN
engine, with native fallback and a per-shape census. It shares the graph-build →
cache → `execute()` pattern this document describes. The relevant mechanics:

- A per-shape graph cache holds the built hipDNN `Graph` so the expensive build/plan
  steps run once per distinct shape; the hot path only swaps tensor pointers and calls
  `graph.execute()`.
- The variant pack maps each tensor uid to a device pointer (`tensor.data_ptr()`), and
  the workspace is a `torch.empty(ws, dtype=torch.uint8, device=...)` buffer.
- Any call the engine cannot serve falls back to the real PyTorch functional and is
  logged with a reason.

**Coverage relative to vLLM**: this patches `torch.nn.functional` entry points. vLLM
routes attention and (often) RMSNorm through custom ops rather than those functionals,
so the functional patch reaches vLLM's dense GEMM (via `F.linear`) but not its attention
or fused-norm paths. That is the gap the vLLM-specific integration below addresses.

### 4.2 Recommended Fast Path: vLLM Attention Backend Plugin

**Effort**: 1-2 weeks
**Model changes**: None — activated via config flag
**Coverage**: Prefill attention today; decode with paged KV once a paged-attention engine exists (the current ASM SDPA engine rejects paged attention — see §2.2)

Attention is the most compute-intensive operation in vLLM on ROCm and the one with the most per-library backend variation, which makes it the natural first integration point. A `ROCM_HIPDNN` backend puts vLLM's several per-library backends behind a single entry point that routes to whatever SDPA providers exist. Today that is the rocKE ASM engine (basic forward, gfx942/gfx950); AITER-as-a-plugin and RDNA providers are not yet available, so the multi-provider aspect is prospective. Provider selection uses configurable static ordering or auto-tuning (benchmark then generate a rules file); there is no automatic heuristic selection. It also enables vLLM on RDNA, where AITER backends do not run — once an RDNA SDPA engine exists.

#### Backend Structure

```python
from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl

class HipDNNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_HIPDNN"

    @staticmethod
    def get_impl_cls():
        return HipDNNAttentionImpl

    @staticmethod
    def validate_configuration(head_size, num_heads, num_kv_heads, dtype, ...):
        # Advertise supported configs: head_dim 64/128, FP16/BF16, GQA
        supported_head_sizes = [64, 128]
        if head_size not in supported_head_sizes:
            raise ValueError(f"hipDNN SDPA supports head sizes {supported_head_sizes}")
        return True


class HipDNNAttentionImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads, ...):
        self._handle = hipdnn.create_handle()
        self._graph_cache = {}
        self._num_heads = num_heads
        self._head_size = head_size
        self._scale = scale
        self._num_kv_heads = num_kv_heads

    def forward(self, query, key, value, kv_cache, attn_metadata, ...):
        if attn_metadata.is_prefill:
            return self._prefill_attention(query, key, value, attn_metadata)
        else:
            return self._decode_attention(query, key, value, kv_cache, attn_metadata)

    def _prefill_attention(self, query, key, value, attn_metadata):
        """Prefill: variable-length sequences, no paging needed."""
        cache_key = self._prefill_cache_key(query, key, attn_metadata)
        cached = self._graph_cache.get(cache_key)

        if cached is None:
            # COLD PATH: build hipDNN SDPA graph
            graph = hipdnn.Graph()
            graph.set_io_data_type(_to_hipdnn_dtype(query.dtype))
            graph.set_compute_data_type(hipdnn.DataType.FLOAT)

            q_t = graph.tensor(...)  # [B, H, S_q, D]
            k_t = graph.tensor(...)  # [B, H_kv, S_kv, D]
            v_t = graph.tensor(...)  # [B, H_kv, S_kv, D]

            attrs = hipdnn.SdpaAttributes()
            attrs.set_attn_scale(self._scale)
            attrs.set_causal_mask(attn_metadata.is_causal)

            outputs = graph.sdpa(q_t, k_t, v_t, attrs)
            o_t = outputs[0]

            # Build and cache
            graph.build_operation_graph(self._handle)
            graph.create_execution_plans([hipdnn.HeuristicMode.FALLBACK])
            graph.check_support()
            graph.build_plans()

            cached = CachedSDPAPlan(graph, uids, workspace)
            self._graph_cache[cache_key] = cached

        # HOT PATH: execute with new tensor pointers
        output = torch.empty_like(query)
        variant_pack = {
            cached.q_uid: query,
            cached.k_uid: key,
            cached.v_uid: value,
            cached.o_uid: output,
        }
        graph.execute(self._handle, variant_pack, cached.workspace)
        return output

    def _decode_attention(self, query, key, value, kv_cache, attn_metadata):
        """Decode: paged KV cache, seq_len_q=1 always."""
        cache_key = self._decode_cache_key(query, kv_cache, attn_metadata)
        cached = self._graph_cache.get(cache_key)

        if cached is None:
            graph = hipdnn.Graph()
            # ... setup tensors ...

            attrs = hipdnn.SdpaAttributes()
            attrs.set_attn_scale(self._scale)
            attrs.set_causal_mask(True)
            # PAGED ATTENTION: pass block tables directly
            attrs.set_paged_attention_k_table(k_page_table_tensor)
            attrs.set_paged_attention_v_table(v_page_table_tensor)
            attrs.set_paged_attention_max_seq_len_kv(max_context_len)

            outputs = graph.sdpa(q_t, k_t, v_t, attrs)
            # ... build and cache ...

        # HOT PATH
        output = torch.empty_like(query)
        # ... execute with paged KV cache block pointers ...
        return output
```

#### Registration

```python
# As a vLLM plugin (no vLLM source changes):
from vllm.attention.backends.registry import register_backend
register_backend("ROCM_HIPDNN", HipDNNAttentionBackend)

# Activate:
# vllm serve model --attention-config.backend ROCM_HIPDNN
```

This coexists with existing backends for A/B benchmarking. Performance must meet or exceed ROCM_AITER_FA on MI3XX before any backend deprecation.

### 4.3 Medium Path: Custom Op Override for RMSNorm

After the attention backend, extend coverage to RMSNorm and GEMM.

**RMSNorm**: vLLM registers custom ops like `torch.ops._C.rms_norm` and `torch.ops._C.fused_add_rms_norm`. These can potentially be overridden using the same `torch.library.Library` technique targeting the `_C` namespace:

```python
# Capture original
original_rms = torch.library.get_kernel("_C::rms_norm", _DISPATCH_KEY)

# Override
lib = torch.library.Library("_C", "IMPL")
lib.impl("rms_norm", _hipdnn_rms_norm_override, _DISPATCH_KEY, with_keyset=True)
```

Whether this works depends on how vLLM registers its custom ops — if they have CUDA/HIP dispatch keys, the override pattern applies. If they're registered as Python-level functions without dispatch keys, monkeypatching the function reference is the alternative.

### 4.4 Deeper Path: Integrated hipDNN Provider Module

Build a `hipdnn_vllm` package that provides:

1. **Attention backend** (Section 4.2)
2. **Norm layer overrides** that route RMSNorm/LayerNorm through hipDNN
3. **Graph-level fusion** across operations (attention → norm) via hipDNN's graph API — this is where hipDNN adds value beyond individual op replacement

The module registers everything at import time:
```python
import hipdnn_vllm  # Registers attention backend + op overrides
# Then start vLLM with --attention-config.backend ROCM_HIPDNN
```

### 4.5 Production Path: First-Class hipDNN Backend in vLLM

Full integration where hipDNN is a recognized platform backend alongside AITER:

- Attention backend in vLLM's core registry
- Integrated with vLLM's `torch.compile` fusion passes
- Participates in the emerging vLLM IR per-op dispatch priority system
- RDNA support (where AITER is MI3XX-only)
- Auto-selected based on GPU architecture and model capabilities

**Model changes required across all tiers**: **NONE**. Every tier is activated via configuration (import, CLI flag, or env var), never by modifying model code.

---

## 5. Graph/Plan Caching for vLLM

LLM inference has different caching dynamics than image generation (ComfyUI). These differences shape the hipDNN graph cache design.

### 5.1 Prefill Phase (Prompt Processing)

- **Sequence lengths vary** per request (32 to 128K+ tokens)
- Each unique `(batch_size, seq_len_q, seq_len_kv)` combination requires a separate graph
- Continuous batching means batch_size changes frequently
- **Challenge**: Many unique shapes → large cache, frequent cold-path hits

### 5.2 Decode Phase (Token Generation)

- **Highly repetitive**: `seq_len_q=1` always (generating one token at a time)
- `seq_len_kv` increments by 1 each step, but with paged attention the physical layout doesn't change — only the page table grows
- Batch size is relatively stable within a scheduling window
- **Opportunity**: After initial warmup, nearly 100% cache hits

### 5.3 Caching Strategy

```python
def _prefill_cache_key(self, query, key, attn_metadata):
    return (
        "prefill",
        query.shape[0],    # batch_size * seq_len (flattened)
        self._num_heads,
        self._head_size,
        query.dtype,
        attn_metadata.is_causal,
        _bucket(key.shape[1]),  # bucket seq_len_kv to reduce entries
    )

def _decode_cache_key(self, query, kv_cache, attn_metadata):
    return (
        "decode",
        query.shape[0],    # batch_size (seq_len=1 always)
        self._num_heads,
        self._head_size,
        query.dtype,
        _bucket(attn_metadata.max_context_len),  # bucketed max context
        kv_cache.block_size,
    )

def _bucket(seq_len):
    """Bucket sequence lengths to reduce cache entries."""
    if seq_len <= 512: return (seq_len + 63) // 64 * 64
    if seq_len <= 4096: return (seq_len + 255) // 256 * 256
    return (seq_len + 1023) // 1024 * 1024
```

### 5.4 Overridable Tensor Shapes (RFC 0008)

hipDNN's overridable tensor shapes API (RFC 0008, Q2 2026 — Phase 1 done) may allow reusing a single compiled graph across different sequence lengths by overriding shapes at execute time:

```cpp
Error execute(hipdnnHandle_t handle,
              std::unordered_map<int64_t, void*>& variantPack,
              void* workspace,
              const std::vector<int64_t>& overrideUids,
              const std::vector<std::vector<int64_t>>& overrideShapes,
              const std::vector<std::vector<int64_t>>& overrideStrides) const
```

If this works for SDPA, a single graph compiled for `max_seq_len` could serve all shorter sequences — dramatically reducing cache size and warmup time.

### 5.5 Pre-Warming Strategy

At model load time, pre-build graphs for the most common configurations:
```python
def _prewarm_graphs(self):
    """Pre-build graphs for common decode batch sizes."""
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        for max_ctx in [512, 1024, 2048, 4096]:
            self._build_decode_graph(batch_size, max_ctx)
```

---

## 6. Fusion Opportunities in vLLM

### 6.1 Attention + Norm Fusion

A hipDNN graph can express SDPA followed by RMSNorm as a single graph, reducing kernel launches:

```python
# Instead of separate attention + rmsnorm:
graph = hipdnn.Graph()
sdpa_out = graph.sdpa(q, k, v, sdpa_attrs)
norm_out = graph.rmsnorm(sdpa_out[0], scale, rmsnorm_attrs)
# hipDNN's engine can fuse these into fewer launches
```

This matters most in decode, where kernel launch overhead is a larger fraction of total time.

### 6.2 GEMM + Activation Fusion

The hipBLASLt provider supports fused epilogues for FFN layers:

| FFN Pattern | hipBLASLt Epilogue | Description |
|------------|-------------------|-------------|
| Linear + SiLU | `HIPBLASLT_EPILOGUE_SWISH_BIAS` | Gate projection in LLaMA FFN |
| Linear + GELU | `HIPBLASLT_EPILOGUE_GELU_BIAS` | FFN in GPT-style models |
| Linear + ReLU | `HIPBLASLT_EPILOGUE_RELU_BIAS` | Older architectures |
| Linear + Bias | `HIPBLASLT_EPILOGUE_BIAS` | Standard bias add |

Express via hipDNN graph:
```python
graph = hipdnn.Graph()
matmul_out = graph.matmul(input_t, weight_t, matmul_attrs)
silu_out = graph.pointwise(matmul_out, pointwise_attrs)  # SWISH_FWD mode
# hipBLASLt provider fuses matmul + activation into single kernel
```

### 6.3 vLLM's torch.compile Fusion Passes

vLLM already implements fusion passes under `torch.compile`. hipDNN patterns could be registered alongside these using:

- `torch._inductor.config.post_grad_custom_pre_pass = HipDNNVLLMFusionPass()`
- Patterns: attention→norm, matmul→bias→activation
- Modeled after `mkldnn_fusion.py` (see PyTorch integration doc for details)

### 6.4 Fused RMSNorm + Residual Add

vLLM's `fused_add_rms_norm` computes `rms_norm(x + residual)` in one kernel. This can be expressed in hipDNN as:

```python
graph = hipdnn.Graph()
add_out = graph.pointwise(x_t, residual_t, add_attrs)  # ADD mode
norm_out = graph.rmsnorm(add_out, scale_t, norm_attrs)
```

Whether hipDNN's engine fuses these into a single kernel depends on provider support.

---

## 7. RDNA vs. CDNA Considerations

### 7.1 The Architecture Gap

| Library | CDNA (MI300/MI325/MI355) | RDNA (RX 7900/9070) |
|---------|------------------------|---------------------|
| AITER | Full support | **No support, no roadmap** |
| CK Flash Attention | Works (MFMA, Wave64) | **Fails to compile** (Wave64 ASM) |
| vLLM Flash Attention | Recommended | AMD docs: `BUILD_FA=0` for RX 7900 |
| Triton | Works | Works (default on all AMD) |
| hipDNN | Works (CDNA providers) | **Potential first-class RDNA path** |

### 7.2 hipDNN on RDNA

hipDNN with RDNA-tuned kernel providers has the following characteristics on consumer GPUs:

- **Architecture-aware dispatch**: MFMA kernels on CDNA, WMMA kernels on RDNA, selected per GPU and transparent to vLLM — one backend across architectures.
- **Coverage gap it addresses**: on Wave32 consumer GPUs, AITER has no support and CK Flash Attention does not currently compile, so there is no well-optimized Flash Attention path from those libraries. RDNA-tuned hipDNN providers are a path to one.
- **Applicable workloads**: local LLM inference on consumer GPUs (Ollama, LM Studio, Open WebUI, etc.), which run on RDNA.

**What's needed**: RDNA-specific kernel providers within rocKE. The hipDNN dispatch infrastructure and provider plugin architecture already exist; the outstanding work is the kernels themselves (WMMA-tuned SDPA, norms, etc. for Wave32).

### 7.3 Integration Impact

A `ROCM_HIPDNN` attention backend in vLLM works on both CDNA and RDNA through hipDNN's architecture-aware dispatch, without vLLM needing separate code paths.

---

## 8. Recommendations & Phasing

### Phase 1 — Attention Backend Plugin (1-2 Weeks)

**Goal**: Deliver hipDNN's multi-engine SDPA architecture and RDNA enablement to vLLM users.

**What**: Implement `ROCM_HIPDNN` attention backend as a vLLM plugin. Use hipDNN's SDPA graph API with paged attention support. Graph/plan caching with sequence length bucketing.

**What it delivers to end users**:
- **MI3XX users**: A single backend entry point for SDPA. Today it routes to the rocKE ASM engine (basic forward); additional providers (AITER ASM, etc.) become available as they are written as hipDNN plugins. Selection is via static ordering or auto-tuning (benchmark → rules file); automatic heuristic selection is future work.
- **RDNA users**: Working, optimized Flash Attention on consumer GPUs where no viable path exists today
- **All users**: Single backend that works across AMD architectures — no more manual `VLLM_ATTENTION_BACKEND` selection
- hipDNN paged attention maps directly to vLLM's KV cache block tables
- A/B benchmarking against existing backends validates performance parity

**Model changes**: None — `--attention-config.backend ROCM_HIPDNN`.

Attention is the largest compute cost in LLM inference and the operation with the most per-library backend variation on ROCm, which is why it is the first phase.

### Phase 2 — Multi-Op Coverage (2-4 Weeks)

**Goal**: Extend hipDNN coverage beyond attention to norms and fused patterns.

**What**:
- Override vLLM's custom RMSNorm/LayerNorm ops with hipDNN implementations
- Implement attention+norm graph fusion (SDPA → RMSNorm as a single hipDNN graph)

**What it delivers**: hipDNN covers more of the transformer layer, reducing kernel launches and memory round-trips between ops.

**Model changes**: None.

### Phase 3 — Full Backend Integration (Months)

**Goal**: Production-ready hipDNN backend in vLLM.

**What**:
- First-class registration in vLLM's attention backend registry
- Integration with vLLM IR per-op dispatch priorities
- Integration with vLLM's torch.compile fusion passes
- Graph-level fusions (attention+norm, future matmul+activation when hipDNN gains multiple GEMM providers or MoE grouped GEMM)
- RDNA auto-selection (hipDNN on RDNA where AITER is unavailable)
- Performance parity or better vs. AITER on MI3XX

**Model changes**: None — built into vLLM's ROCm platform support.

---

## 9. Operation Gap Analysis

A comprehensive analysis of operations hipDNN needs to add for modern LLM and diffusion models — including cuDNN v9 parity analysis — is in Section 8 of the companion document `pytorch-integration-techniques.md`. The gaps most relevant to vLLM are:

| Priority | Operation | vLLM Impact | cuDNN Has It? |
|----------|-----------|------------|--------------|
| **P0** | MoE grouped GEMM + fusions | Mixtral, DeepSeek — dominant scaling architecture | **Yes** (extensive) |
| **P0** | RoPE | Every LLM, every token, every layer | **No** (neither library) |
| **P0** | Per-token/per-group quant + GPTQ/AWQ dequant | Required for serving quantized models | **Partial** |
| **P1** | Fused RMSNorm + SiLU | LLM FFN activation pattern | **Yes** (auto-detected) |
| **P1** | Fused gate+up GEMM (split-output) | Halves FFN GEMM launches in SwiGLU | **Yes** (via MoE + SwiGLU) |
| **P1** | Fused residual+norm | Every layer boundary — eliminates memory round-trip | Implicit via graph |
| **P1** | Fused RMSNorm+quantize | Critical for FP8 inference pipelines | Partial |

**cuDNN parity note**: cuDNN is expanding MoE support (grouped GEMM + SwiGLU, + GLU, + quant, per-expert reduction). MoE is therefore the area where the feature gap between the two libraries is currently widest. RoPE is absent from both libraries.

Adding RoPE and MoE support would shift hipDNN from covering ~60% of a LLaMA forward pass to ~90%+, leaving only KV cache management and framework-specific scheduling outside hipDNN's scope.

---

## Appendix A: hipDNN SDPA Feature Matrix vs. vLLM Requirements

The table below distinguishes between what the hipDNN **graph API** defines and what **engines currently implement**. Features with API support but no engine are awaiting rocKE kernel development — the API is ready, the integration plumbing can be built, but the feature won't execute until an engine lands.

| Feature | vLLM Needs | API Defined? | Engine Support? | Notes |
|---------|-----------|-------------|----------------|-------|
| Basic SDPA forward | Yes | Yes | **Yes** (gfx942/gfx950, BF16/FP8_E4M3) | Current ASM SDPA engine |
| Causal masking | Yes | Yes | **Yes** | Supported by current engine |
| GQA (Grouped Query) | Yes (LLaMA 3, Mistral) | Yes | **Partial** | K/V heads < Q heads; verify engine support |
| MQA (Multi-Query) | Yes (some models) | Yes | **Partial** | K/V heads = 1; verify engine support |
| Paged KV cache | Yes (core feature) | Yes | **No** — engine rejects | Awaiting rocKE; critical for vLLM decode |
| ALiBi positional encoding | Yes (MPT, BLOOM) | Yes | **No** — engine rejects | Awaiting rocKE |
| Attention bias (additive mask) | Yes (some models) | Yes | **No** — engine rejects | Awaiting rocKE |
| Dropout | Training only | Yes | **No** — engine rejects | Awaiting rocKE |
| Stats output (for backward) | Training | Yes | **No** — engine rejects | Awaiting rocKE |
| Variable sequence lengths | Yes | Yes (via shape override) | **Unknown** | Needs verification |
| Sliding window | Yes (Mistral) | Yes | **Unknown** | Needs verification |
| FP8 attention | Emerging | Yes | **Partial** (FP8_E4M3 input) | BF16 output only currently |
| Head dimensions 64/128 | Yes | Yes | **Likely** | Standard sizes; verify |
| Head dimension 256 | Some models | Yes | **Unknown** | Needs verification |
| MLA (Multi-Latent) | Yes (DeepSeek) | Unknown | **Unknown** | May need custom support |
| Block masking | Some use cases | Yes | **Unknown** | Needs verification |
| RDNA architecture | Growing use case | N/A | **No** — engine requires gfx942/gfx950 | Awaiting rocKE RDNA kernels |

## Appendix B: Key File Paths

Repo-relative paths (from the rocm-libraries root); `<rocm-sdk>` is an installed ROCm SDK.

| Purpose | Path |
|---------|------|
| hipDNN SDPA attributes (paged attn) | `projects/hipdnn/frontend/include/hipdnn_frontend/attributes/SdpaAttributes.hpp` |
| hipDNN Graph API | `projects/hipdnn/frontend/include/hipdnn_frontend/Graph.hpp` |
| hipDNN Roadmap | `projects/hipdnn/docs/Roadmap.md` |
| hipDNN Backend C API | `<rocm-sdk>/include/hipdnn/backend/hipdnn_backend.h` |
| hipBLASLt provider support | `dnn-providers/hipblaslt-provider/docs/OperationSupport.md` |
| hip-kernel-provider (rocKE) | `dnn-providers/hip-kernel-provider/` |
| PyTorch integration techniques | `pytorch-integration-techniques.md` |

## Appendix C: Related Resources

- [vLLM ROCm Attention Backend Blog](https://vllm.ai/blog/2026-02-27-rocm-attention-backend) — Details the 7 AMD backend architecture
- [vLLM V1 ROCm Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html) — Performance tuning guide
- [vLLM Triton Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) — Platform-portable Triton attention
- [vLLM Plugin System](https://docs.vllm.ai/en/v0.13.0/design/plugin_system/) — Plugin interface documentation
- [vLLM Attention Backend Features](https://docs.vllm.ai/en/latest/design/attention_backends/) — Feature support matrix
- [vLLM-ATOM: AMD-Optimized Plugin](https://rocm.blogs.amd.com/software-tools-optimization/vllm-atom/README.html) — Example AMD plugin
- [AITER: AI Tensor Engine for ROCm](https://github.com/rocm/aiter) — AMD's optimized operator library
- [vLLM IR RFC (Issue #32358)](https://github.com/vllm-project/vllm/issues/32358) — Emerging per-op dispatch framework
