[← MLA design doc index](../DESIGN.md)

## 0. Notation

| symbol | shape | meaning |
|---|---|---|
| $Q$ | $S_q \times H_q \times (d_{\text{nope}} + d_{\text{rope}})$ | query; split into nope and rope slices |
| $c_q$ | $S_q \times r_Q$ | compressed query latent, **shared across heads** (called `q_latent` in the prefill spec, `c_q` in the decode spec — one tensor, two names) |
| $c_{KV}$ | $S_k \times r_{KV}$ | compressed KV latent (the KV cache payload) |
| $K_{\text{rope}}$ | $S_k \times d_{\text{rope}}$ | separately-stored RoPE key component |
| $W_{UK}$ | $H_q \times r_{KV} \times (d_{\text{nope}} + d_V)$ | up-projection: latent → K\_nope ‖ V. **Per head** — DeepSeek's `kv_b_proj` is $[r_{KV},\, H_q(d_{\text{nope}}{+}d_V)]$; the latent is shared across heads, its expansion is not. 128 × 512 × 256 × 2 B = **32 MiB** total; the **256 KB per-head slice** is what the §5 tiling streams |
| $W_{UV}$ | $H_q \times r_{KV} \times d_V$ | V slice of $W_{UK}$ (column partition), per head. 16 MiB total, 128 KB per head |
| $W_{UK,K}$ | $H_q \times r_{KV} \times d_{\text{nope}}$ | K\_nope slice of $W_{UK}$ (column partition), per head. 16 MiB total, 128 KB per head |
| $W_{UQ}$ | $H_q \times r_Q \times (d_{\text{nope}} + d_{\text{rope}})$ | query up-projection |
| $W_{UQ,\text{nope}}$ | $H_q \times r_Q \times d_{\text{nope}}$ | nope slice of $W_{UQ}$ (column partition, per head: $W_{UQ}^{(h)}[:, :d_{\text{nope}}]$) |
| $W_{UQ,\text{rope}}$ | $H_q \times r_Q \times d_{\text{rope}}$ | RoPE slice of $W_{UQ}$ (per head: $W_{UQ}^{(h)}[:, d_{\text{nope}}:]$); named `W_rope_proj` in the kernel specs |
| $W_{\text{abs}}$ | $H_q \times r_Q \times r_{KV}$ | absorbed weight (decode only) — $W_{UQ,\text{nope}} \cdot W_{UK,K}^{\top}$, see §2.4 |
| $\text{scale}$ | scalar | softmax scale $= 1/\sqrt{d_{\text{nope}} + d_{\text{rope}}} = 1/\sqrt{192}$ — see the note below |
| $d_{\text{nope}}$ | 128 | content head dimension (qk\_nope) |
| $d_{\text{rope}}$ | 64 | RoPE head dimension (qk\_rope) |
| $d_V$ | 128 | value head dimension |
| $r_{KV}$ | 512 | KV lora rank |
| $r_Q$ | 1536 | query lora rank |
| $H_q$ | 128 (DeepSeek), 64 (Kimi-K2) | query heads |
| $H_k$ | 1 | KV heads (MLA always has Hk=1) |
| `Bq`, `Bk` | tuning knobs (§5) | kernel query-tile and KV-**tile** sizes; `Bk` is independent of the paged-cache `block_size` — see the note in §5.2 |
| `r_KV_tile` | 64–128 (§5) | $r_{KV}$ K-step for the latent expansion (prefill) and for `W_UV` streaming (decode epilogue) |
| `BLOCK_H` | sized at impl (§9) | query **heads** per workgroup — the MFMA M-tile. §3 (prefill) and §4 (decode) each carry one and they are **not the same value**: raising it in decode amortizes the shared KV read (§2.4), raising it in prefill multiplies in-loop `W_UK` traffic (§3). Read every use as scoped to its section |
| `NUM_SEGMENTS` | sized at impl (§9) | split-KV segments per sequence in the 3D decode path (§4); coupled to `BLOCK_H` through grid parallelism, not independent of it |
| `num_warps` | sized at impl (§9) | wave64 waves per workgroup; workgroup size is `64 * num_warps` threads |

> **Weight layout convention.** Every weight matrix in this doc is stored `[in, out]`
> (row-major). An up-projection is therefore a plain `x · W` with **no transpose**:
> `K_nope = c_KV · W_UK_K`, `q = c_q · W_UQ[h]`, `V = c_KV · W_UV`. A `^T` appears in
> this doc only where two *activation* tensors are contracted to form a score matrix
> (`q_nope · K_nope^T`), or in the one weight-times-weight product that defines
> $W_{\text{abs}}$ (§2.4), where the shared $d_{\text{nope}}$ axis is contracted.

> **Softmax scale.** `scale` is $1/\sqrt{d_{\text{nope}} + d_{\text{rope}}} = 1/\sqrt{192}$
> for **both** kernels. It does **not** become $1/\sqrt{576}$ in the decode-absorb kernel:
> weight absorption is an exact algebraic rewrite of the same 192-wide content score
> (§2.4), not a widening of the head dimension, so the scale must not follow the
> `head_dim = 576` framing used to describe the decode kernel's *memory* structure.
> This is an easy mistake to make, and the existing tooling makes it easy: the decode
> harness derives `scale = shape.head_size**-0.5`
> (`benchmarks/gfx*/attention/decode/benchmark_decode_live.py`), so an MLA shape
> labelled `head_size = 576` would silently get $1/\sqrt{576}$ — one more reason the
> MLA shape files carry no `head_size` key today (§8.2). `scale` must therefore
> be a host-supplied runtime scalar, never derived from `head_dim` by the kernel or the
> harness. A second, independent reason: DeepSeek-V3 with YaRN rope scaling folds an
> extra `mscale²` factor into `scale` at the model level, so no geometry-derived value
> is correct for that model even at the right head dimension.

---

