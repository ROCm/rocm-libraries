[← MLA design doc index](../DESIGN.md)

## 11. References

**Internal (rocKE codebase):**
- `library/builders/gfx950/attention/ALGORITHM.md` — unified attention on gfx950; template for this doc's structure
- `library/builders/gfx942/attention/ALGORITHM.md` — gfx942 narrow/flash math
- `library/builders/gfx1250/attention/gfx1250_universal_attention_plan.md` — phased plan analog
- `library/kernels/common/fmha_fwd_fp8.py` — sync-dequant fp8 pattern (gfx950+ fp8 kernels)
- `library/dispatch/attention/` — `AttentionRequest`, `ATTENTION_REGISTRY`, the arch candidate modules (the dispatch layer to extend; `common.py` holds the request and its gates, `generic.py:49-57` the unified `Capability`)
- `platform/python/rocke/dispatch/core.py` — `Capability`, `ShapeRange`, `KernelCandidate`, `CandidateRegistry` (the matching machinery §7.3 builds on; not re-exported through `dispatch.attention`)
- `library/kernels/common/attention_unified.py` — `UnifiedAttentionProblem`, `UNIFIED_HEAD_SIZES`, flash building blocks
- `library/kernels/gfx942/attention_tiled_2d.py`, `attention_tiled_3d.py` — arch baselines
- `library/kernels/gfx950/attention_tiled_2d.py`, `attention_tiled_2d_fastkv_regp.py` — gfx950 baselines; `register_pv` and `ds_read_tr` patterns
- `library/kernels/common/attention_arch.py` — arch gating (`_NARROW_TILED_2D_ARCHES`, `validate_tiled_attention_arch`)
- `platform/cpp/instances/gfx950/attention_tiled_2d_kv_body_pv_epilogue.cpp` — `ds_read_tr16_b64` usage in V staging

**Papers — the primary sources for §0–§2 and §5.**

Every geometry value in §1, the per-head shape of $W_{UK}$ (§0), the head-shared
$K_{\text{rope}}$ (§2.1), the $1/\sqrt{192}$ scale (§0) and the absorption identity
(§2.4) are checkable against these; §9's trap list cites them. Where this doc and a
paper disagree, the paper wins.

- **DeepSeek-V2** — `arXiv:2405.04434`. §2.1 is the defining MLA specification.
  §2.1.2 gives $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ (the **per-head**
  up-projection, §0) and the absorption statement (*"$W^{UK}$ can be absorbed into
  $W^Q$, and $W^{UV}$ can be absorbed into $W^O$"*, §2.4). §2.1.3 Eq. 15 defines
  $k^R_t = \text{RoPE}(W^{KR}h_t)$ as *a shared key* $\in \mathbb{R}^{d^R_h}$ — the
  post-rotation storage and head-sharing of §2.1. Eq. 18 divides the score by
  $\sqrt{d_h + d^R_h}$ — the $1/\sqrt{192}$ of §0, not $1/\sqrt{576}$. §3.1.2 gives
  $n_h{=}128,\ d_h{=}128,\ d_c{=}512,\ d'_c{=}1536,\ d^R_h{=}64$ — the DeepSeek row of §1.
- **DeepSeek-V3** — `arXiv:2412.19437`. Confirms the same MLA geometry at V3 scale and
  the 61-layer depth used in §4's `W_abs` working-set argument.
- **Kimi-K2** — `arXiv:2507.20534`. Table 2 gives 64 attention heads against
  DeepSeek-V3's 128 at equal depth (61 layers) — the Kimi row of §1 — and the rationale
  (at 128 K context, 64→128 heads is *"an 83% increase in inference FLOPs"*), which is
  why $H_q$ is the only axis this kernel family parameterizes over.
- **Yun et al., *Rethinking LLM Inference Bottlenecks: Insights from Latent Attention
  and Mixture-of-Experts*** — `arXiv:2507.15465`. Independent hardware analysis of the
  §2.5 split, and the strongest external support for two distinct kernels: it
  recommends *"the prefill stage uses MLA without reordering and the decode stage uses
  MLA with reordering"* ("reordering" = absorption). Quantifies both directions —
  prefill attention 2.02× **worse** with absorption at $B{=}1, L{=}4096$; decode 119×
  **better** at $B{=}256, L{=}4096$; decode score-layer arithmetic intensity ≈1 → ≈100 —
  and prices the prefill penalty at $d_{KV_{co}}/d_{hd} = 512/128 = 4\times$ the
  score-layer compute. It fixes **no token threshold**, which is why §2.5's ~200 stays a
  citation until §8.2's family-2 sweep measures it.
- **FlashInfer** — `arXiv:2501.01005`. See §10.5 for the `(192, 128)` prefill mode and
  the fused-vs-decompressed statement.
- **MLA sequence-parallelism training regression** — `arXiv:2607.17644`. Not
  load-bearing here (training-side), but it independently states the inference win as
  *never materializing or recomputing $K^C$/$V^C$ over the growing cache* — the §2.5
  $1/S_q$ argument from the memory side.

**External (state of the art — see §10):**
- AITER MLA: `github.com/ROCm/aiter` / `rocm.blogs.amd.com/software-tools-optimization/aiter-mla/`
- FlashMLA (DeepSeek, CUDA): `github.com/deepseek-ai/FlashMLA`
- TileLang MLA on gfx942: `github.com/tile-ai/tilelang` / `tilelang.com/deeplearning_operators/deepseek_mla.html`
- FlashInfer ROCm: `github.com/ROCm/flashinfer`
- SGLang weight absorption: PR #905, #1138 at `github.com/sgl-project/sglang`
- FlyDSL: `github.com/ROCm/FlyDSL` — MLIR-based Python DSL used in AITER for MoE/GEMM; a rocKE-comparable Python→HSACO path through MLIR rather than LLVM IR. No MLA attention kernel, so it informs no decision in §2–§5
- CK FMHA (192,128): `github.com/ROCm/composable_kernel` — commit `4399ad79029`; `include/ck_tile/ops/fmha/`, `dispatcher/codegen/fmha/fmha_arch_specs.json`, `tile_engine/ops/fmha/ck_fmha_testing_matrix.yaml`
