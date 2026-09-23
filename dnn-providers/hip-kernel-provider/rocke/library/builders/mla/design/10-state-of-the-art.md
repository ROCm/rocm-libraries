[← MLA design doc index](../DESIGN.md)

## 10. State of the art — public MLA kernel implementations

This section documents the public implementations reviewed during the design spike.
They inform the architectural decisions in §2–§5; nothing here is prescriptive for
rocKE implementation.

### 10.1 AITER (AMD Inference Toolkit)

The canonical ROCm MLA reference. Lives at `github.com/ROCm/aiter`; integrated into
vLLM and SGLang as the AMD attention backend.

**Decode kernel** (`mla_decode_fwd`):
- Hand-written assembly (not open-sourced in detail).
- Implements MQA in latent space (head_dim=576) exactly as described in §2.4.
- Absorbed weights (`w_kc = W_abs`, `w_vc = W_UV`) pre-computed at model load.
- Paged KV cache with split-KV scheduling across CUs.
- Reported **17× speedup** over non-absorbed naive MLA on gfx942
  (`rocm.blogs.amd.com/software-tools-optimization/aiter-mla/`).
- On gfx942: `ROCM_AITER_TRITON_MLA` is reported to slightly outperform the ASM
  backend. Directional only — no public benchmark table to cite, so treat the two as
  comparable on gfx942 and measure locally rather than planning around a margin.
- On gfx950: the ASM backend is reported to match or beat Triton MLA — also
  directional, with no public benchmark table to cite.

**Prefill**:
- `ROCM_AITER_MLA`: dispatches to CK or ASM; ASM prefill limited to $S_q < 160$
  (chunked-prefill regime).
- `ROCM_AITER_TRITON_MLA`: uses Triton MHA path for prefill (standard flash with
  expanded head_dim=192 for nope+rope).

### 10.2 FlashMLA (DeepSeek)

`github.com/deepseek-ai/FlashMLA` — the reference MLA decode kernel from DeepSeek
(CUDA/Hopper only, not ported to ROCm). Architecturally relevant:

- Confirms the MQA-in-latent-space approach (§2.4): Q shape `[Sq, N, r_KV+d_rope]`,
  KV cache `[Skv, 1, r_KV+d_rope]` with paged block size 64.
- Uses TMA on Hopper for async KV load; AMD equivalent would be `raw_ptr_buffer_load_lds`
  with `async_buffer_load_lds_addr` (already in `attention_tiled_2d.py`).
- Split-KV across SMs with partial-result merging (same structure as
  `attention_tiled_3d`).
- Fuses KV buffer writes with decode attention; upstream reports roughly +12% from
  removing the PyTorch-side overhead this eliminates.
- Upstream reports a ceiling of ~3000 GB/s memory-bound and ~660 TFLOPS
  compute-bound on H800.

> The two figures above are self-reported in the FlashMLA repo
> (`github.com/deepseek-ai/FlashMLA`), not independently verified, and are Hopper
> numbers on a different memory system. They are quoted for order of magnitude and
> for the *shape* of the optimization (fusing the KV write), not as targets for
> gfx942/gfx950.

### 10.3 TileLang MLA

`github.com/tile-ai/tilelang` — open-source composable tiled DSL with a complete,
readable MLA kernel for gfx942.

- Reported at **95% of AITER assembly performance** on gfx942, **1.98× over Triton**
  MLA and **3.76× over PyTorch** baseline — upstream's own figures
  (`tilelang.com/deeplearning_operators/deepseek_mla.html`), not independently
  reproduced.
- ~80 lines of Python, fully open-source. The tiling strategy is the most
  transparent public reference for MLA on gfx942.
- Handles gfx942's 64 KB LDS (vs Hopper's 228 KB shared memory) explicitly; tile sizes not
  constrained to multiples of 64; swizzling for bank conflicts handled automatically.
- Recommended as a **parity baseline** in addition to AITER Triton MLA (§8.3).

### 10.4 CK (Composable Kernels)

CK has no dedicated MLA kernel (no `mla`, `kv_lora_rank`, `absorbed` code paths),
but it contains relevant MLA-adjacent infrastructure added explicitly for DeepSeek
V3:

**`(hdim_q=192, hdim_v=128)` — officially supported:**
- Commit `4399ad79029` (March 2025): "support hdim=192/128 pair for deepseekv3".
- Registered as a supported `[hdim_q, hdim_v]` pair for fp16/bf16/fp8/fp8bf16/bf8 in
  `dispatcher/codegen/fmha/fmha_arch_specs.json` (`supported_hdims`). **Instantiation is
  narrower than registration:** in the `01_fmha` example codegen only plain
  `fmha_fwd.py` carries a `(192,128)` tile. `fmha_fwd_splitkv.py` has no 192 entry at
  all, `fmha_pagedkv_prefill.py:587` has its `"192"` tile **commented out**, and
  `fmha_batch_prefill.py` carries only 128/256 — so neither split-KV nor either prefill
  generator emits a `(192,128)` kernel today.
- This is absorbed-form MLA: Q/K head_dim = qk_nope(128) + qk_rope(64) = 192,
  V head_dim = v_head_dim = 128. The CK FMHA kernel with these dims is a direct
  viable substrate for the MLA prefill kernel without fusing the latent
  expansion — the expansion is a separate prior GEMM and CK handles the attention.
- Restrictions at this pair are per-variant, and the paged conclusion is stronger than
  a per-hdim gate. `fmha_fwd.py:820` (`check_hdim`) skips `(192,128)` only when
  `bias != "no"` or `dropout == "t"` — **LSE is available** there — but `fmha_fwd.py`
  has no paged-KV pipeline at all. The `(192,128)` guards in
  `fmha_pagedkv_prefill.py:703-706` (`bias != "no" or lse == "t"`) and
  `fmha_batch_prefill.py:846-853` (same, plus dropout) are **dead code today**, because
  neither generator has a 192 tile to reach them. On the dispatcher codegen the paged
  family *is* enumerated over `supported_hdims`, so `(192,128)` is reachable there — but
  `get_pagedkv_pipelines` (`dispatcher/codegen/fmha/instance_gen.py:1030`) passes
  `lse="f"` positionally for **every** `qr_pagedkv` spec. CK therefore emits no paged-KV
  forward kernel with LSE at any head dim: chunked-prefill via `softmax_lse` is blocked
  on the path MLA needs (§3), and bias is unavailable on every variant at this pair.

**`MLA_H128xH576_Asymmetric` test case — planned but not instantiated:**
- `tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml` (added 2026-05-17, `a1834d2b22`) contains
  a test entry: `hdim_q=128, hdim_v=576`, seqlen_q=4096, described as "Multi-latent
  attention fusion; asymmetric Q/KV (128 vs 576)." The 576-dim V corresponds to the
  MQA-in-latent-space approach (r_KV=512 + d_rope=64 = 576 from §2.4).
- **No kernel instance exists** for this pair in `fmha_arch_specs.json`. It is a
  future target, not a usable kernel.

**Partial RoPE via `rotary_dim`:**
- `include/ck_tile/ops/fmha/block/block_rotary_embedding.hpp` supports a
  `rotary_dim` parameter that applies RoPE only to the first `rotary_dim` elements
  of the head vector, leaving the nope slice unrotated. This directly models the MLA
  RoPE pattern (RoPE on `d_rope=64` of a 192-dim head). Already wired into
  `fmha_fwd_appendkv_kernel.hpp` at runtime.

**Hard cap `hdim_q <= 256`:**
- All CK FMHA pipelines contain `static_assert(kSubQKHeaddim <= 256)`. CK cannot
  directly handle the Q latent dimension (r_Q=1536) without structural changes.

**Bottom line for rocKE:** CK's (192,128) FMHA is a ready-made *measurement baseline*
for MLA prefill in the "separate expansion" mode — not a shippable rocKE instance: per
§3, `lower_cktile.py` accepts no attention spec, so a wrapped CK kernel yields no
`KernelDef` and no `ATTENTION_REGISTRY` candidate. The prefill implementation should
prototype wrapping it early and measure the in-loop fusion approach against it; the
custom rocKE kernel is still the deliverable (§9).

### 10.5 FlashInfer ROCm MLA

`github.com/ROCm/flashinfer` — ROCm port of FlashInfer. Decode via
`trtllm_batch_decode_with_kv_cache_mla`:

- KV cache layout: `[num_pages, page_size, r_KV + d_rope]` — confirms the
  concatenated layout described in §4.
- Decode kernel: 128-head MQA in latent space, reusing `c_KV` as both K and V for
  the score + value step (same as §2.4).
- CK backend for gfx942 prefill; supports gfx942 and gfx950.
- **The MLA API carries `kv_lora_rank` and `qk_rope_head_dim` as separate parameters
  rather than a single `head_dim`**, and its two modes are
  `(head_dim_k, head_dim_v) = (576, 512)` for decode-absorb and **`(192, 128)` for
  prefill** — i.e. exactly the asymmetric score/value split §7.2 proposes to admit and
  §3.1 forbids padding away. The upstream engine (arXiv:2501.01005, §11) also states the
  anti-pattern directly: a naive implementation decompresses the latent to full KV and
  then runs standard attention, wasting bandwidth, where the fused form keeps the latent
  projection inside the tiled kernel. That is §2.5's two regimes seen from the kernel
  side.

### 10.6 SGLang weight absorption

`github.com/sgl-project/sglang` — the weight absorption technique (§2.4) was first
deployed at scale in SGLang (PR #905, #1138). Key implementation details:

- Absorbed weights stored as `w_kc` and `w_vc` at model load; the SGLang MLA
  module never calls `W_UQ` or `W_UK` during serving.
- Prefill crossover threshold ~171–228 tokens (§2.5): below threshold → Triton
  absorbed decode; above → materialize K/V → standard flash prefill.
- Open issue (#4615): avoid materializing `w_kc`/`w_vc` to save GPU memory — still
  open, relevant to the hipDNN graph tensor representation question in §7.4.

---

