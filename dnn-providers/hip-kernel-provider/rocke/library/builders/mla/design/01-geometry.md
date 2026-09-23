[← MLA design doc index](../DESIGN.md)

## 1. MLA geometry and model variants

| Model | $H_q$ | $d_{\text{nope}}$ | $d_{\text{rope}}$ | $d_V$ | $r_{KV}$ | $r_Q$ |
|---|---|---|---|---|---|---|
| DeepSeek V2/V3/V3.1/R1 | 128 | 128 | 64 | 128 | 512 | 1536 |
| Kimi-K2 | 64 | 128 | 64 | 128 | 512 | 1536 |

> **GLM-5 is out of scope for this kernel family**, despite being an obvious third
> candidate alongside DeepSeek and Kimi-K2. Its published config
> (`zai-org/GLM-5`, `config.json`) is `model_type: glm_moe_dsa` /
> `GlmMoeDsaForCausalLM` with `index_n_heads: 32`, `index_head_dim: 128`,
> `index_topk: 2048` — GLM-5 is **DeepSeek Sparse Attention**, so decode is a
> lightning-indexer top-k gather over the latent cache, not the dense flash loop
> specified here. No value of $H_q$ makes it servable by these kernels. Four of its six
> MLA-geometry values also differ from the DeepSeek row
> ($H_q$ 64 — which happens to match Kimi-K2 — $d_{\text{nope}}$ 192, $d_V$ 256,
> $r_Q$ 2048; only $d_{\text{rope}}$ and $r_{KV}$ match both rows). The sparse decode
> path, not the geometry, is the reason it is out of scope: a corrected-geometry GLM-5
> row would still not be servable by these kernels. Sparse-attention support is tracked
> as separate work; DeepSeek V2/V3/V3.1/R1 and Kimi-K2 are unaffected and the design
> below stands for both.

---

