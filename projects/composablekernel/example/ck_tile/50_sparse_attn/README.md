# Sparge Attention (Composable Kernel)

A Composable Kernel port of [SpargeAttn](https://github.com/thu-ml/SpargeAttn) for AMD GPU. Both the block-map pipeline (mean-pool → cosine sim → pooled QK → top-k LUT) and the sparse FMHA stage run on-GPU. Two attention backends are exposed via `-pipeline=vsa` (default, faster) and `-pipeline=jenga`.

## Status vs Upstream

Implemented:
- per-block mean-pool, cosine similarity, pooled QK
- top-k / `cdfthreshd` block selection, BlockMap LUT
- sparse FMHA (both `vsa` and `jenga` backends)
- per-head `topk` / `simthreshd1` / `cdfthreshd`
- **is_causal mask in pooled score** (top-left only at block-map grain) ([spas_sage_attn/utils.py:L338](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/utils.py#L338))
- **attention_sink** — block-map column 0 force-on ([spas_sage_attn/autotune.py:L355](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/autotune.py#L355))

Not yet ported (upstream pinned to commit [`ae5b629`](https://github.com/thu-ml/SpargeAttn/tree/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a)):
- **K smoothing** — pre-pool `k -= km`; required for diffusion / video checkpoints (CogVideoX, Mochi-1, Flux, OpenSora, SD 3.5) ([spas_sage_attn/core.py:L53](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/core.py#L53))
- **SageAttention-style Q/K int8 quant fusion** — fuses Sage int8 quant into the pool kernel, enables a downstream int8 GEMM0 in the attn kernel (matches `49_sageattention`'s quant recipe) ([spas_sage_attn/utils.py:L371](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/utils.py#L371))

## Performance

![V1 sage-sparge comparison](docs/pv_skip_mode_comparison.png)

*MI300X, b=2 h=16 s=8192 d=128, 5 seeds × 9 sparsity points, `-pv_mode=warp`. Two hlines (`fmha_dense` fp16, `fmha_sage` fp8bf16 BLOCKSCALE) and two sweeps (`sparge_fp16`, `sparge_sage` = V1 sage⊕sparge with int8 BLOCKSCALE Q/K + fp16 V); labels are sparge_sage speedup vs dense.*

V1 (`sparge_sage`) sits +15..+19% above `sparge_fp16` at every sparsity, and crosses the dense baseline near sparsity 0.31 vs sparsity 0.40 for fp16.

## PV-skip modes

`pv_threshold` per-Q-tile skip in the attention kernel is implemented in three variants, selectable at runtime via `-pv_mode={none|warp|block}`:

- **`none`** — skip disabled; baseline matching the no-PV-skip codegen instance.
- **`warp`** (per-wavefront) — each wavefront votes locally via `__shfl_xor` butterfly AND; SGPR-resident flag. CK-tile-specific variant, not in upstream.
- **`block`** (per-block) — block-wide consensus vote via LDS broadcast; aligned with upstream sm80 ([`qk_int_sv_f16_cuda_sm80.cuh:L334`](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh#L334)). V loads stay unconditional in all modes — the guard wraps the PV MMA only, matching upstream and paper Algorithm 1.

Default is `-pv_mode=warp`; `none` disables the skip and `block` selects the upstream-aligned block-wide vote. On the `kM0=64` tile bucket of the recipe shape, `warp` wins — `block` adds +33..+35 VGPR which depresses occupancy.

## Usage

```bash
ninja tile_example_sparge
./bin/tile_example_sparge -pipeline=vsa -b=2 -h=32 -s=16384 -d=128 -topk=0.4 -simthreshd1=0.001
```

Select a PV-skip variant with `-pv_mode={none|warp|block}` (default `warp`); finite `-pv_threshold=20` lets the per-Q-tile skip predicate fire.

Mask + attention sink:
- `-mask` accepts the `01_fmha` grammar (`0` / `t` / `b` / `t:l,r` / `xt:N` / `g:y,x`, default `0`). The block-map selection prunes past-diagonal blocks only under `mask_top_left` (`t`); `b` / SWA / generic are forwarded to the attention kernel and emit a stderr WARN that the block-map selection is unchanged.
- `-attention_sink {0,1}` forces block-map column `kb=0` ON for every Q-block (default `0`). Under `-mask t` this is degenerate since `kb=0` is always causal-valid.

Add `-v=1` for CPU validation; use a small shape (`-b=1 -h=2 -s=512`), since full-shape CPU reference scales O(s²) and runs 30+ minutes at s=8k, hours at s=16k. When `-mask != 0` or `-attention_sink == 1`, the `[block_map cross-check]` and `[VSA LUT self-consistency]` cells are SKIPPED (the CPU reference does not model causal mask or sink); the `[attention output]` cell still runs but the dense reference applies no mask, so it will report FAIL on the kernel-correct output. Treat `-v=1` correctness as **block-map level only** in those configurations.

## References

- [SpargeAttn upstream](https://github.com/thu-ml/SpargeAttn) (pinned to [`ae5b629`](https://github.com/thu-ml/SpargeAttn/tree/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a))
- [Paper — Zhang et al., arXiv:2502.18137](https://arxiv.org/abs/2502.18137)
