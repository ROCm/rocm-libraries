# Sparge Attention (Composable Kernel)

A Composable Kernel port of [SpargeAttn](https://github.com/thu-ml/SpargeAttn) for AMD GPU. Both the block-map pipeline (mean-pool → cosine sim → pooled QK → top-k LUT) and the sparse FMHA stage run on-GPU. Two attention backends are exposed via `-pipeline=vsa` (default, faster) and `-pipeline=jenga`.

## Status vs Upstream

Not yet ported (upstream pinned to commit [`ae5b629`](https://github.com/thu-ml/SpargeAttn/tree/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a)):
- **K smoothing** — pre-pool `k -= km`; required for diffusion / video checkpoints (CogVideoX, Mochi-1, Flux, OpenSora, SD 3.5) ([spas_sage_attn/core.py:L53](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/core.py#L53))

## Performance

![SpargeAttn + SageAttn comparison](docs/pv_skip_mode_comparison.png)

*MI300X, b=2 h=16 s=8192 d=128, 5 seeds × 9 sparsity points, `-pv_mode=warp`. Two baselines (Dense FP16, Dense + SageAttn FP8 BLOCKSCALE) and two sparse sweeps: SpargeAttn (sparse + FP16) and SpargeAttn + SageAttn (sparse + INT8 BLOCKSCALE Q/K, FP16 V). Timing is the binary's in-program GPU timer (hipEvent) measured end-to-end over the full pipeline — all three sparse kernels (K-stats + block-map selection + attention) — not an attention-only profile. Point labels are the SpargeAttn + SageAttn speedup vs Dense FP16.*

SpargeAttn + SageAttn (int8 Q/K) sits ~+12% above the FP16-only sparse sweep (median; +4..+14% across sparsity), and crosses the dense baseline near sparsity 0.47 vs sparsity 0.53 for FP16; at sparsity 0.91 it reaches ~3.5x dense. These numbers are end-to-end across all three GPU kernels (K-stats + block-map selection + attention), so the break-even sparsity is higher than an attention-only timing would suggest.

### Reproducing the chart

Scripts live in [`docs/`](docs/). Two steps — measure on your own GPU, then plot:

```bash
# 1. sweep on an MI300-class GPU (needs an MI300-class GPU; ~5 min for the full 5-seed sweep)
python3 docs/run_bench.py --bin-dir build/bin --csv sparge_bench.csv

# 2. render the figure from the CSV you just produced (needs matplotlib)
python3 docs/plot.py --csv sparge_bench.csv --out sparge_chart.png
```

`run_bench.py` is the readable reference for how each curve's data is produced:
it documents the four curves, the bench shape, and the CSV schema it writes
(one row per curve/sparsity/seed). Timing is read from the binary's in-program
GPU timer (hipEvent, parsed from stdout), which brackets all three sparse kernels
(K-stats + block-map selection + attention) end-to-end — no rocprof required.
The exact measured numbers are not vendored —
re-run step 1 to generate them on your own hardware. `run_bench.py --smoke` does
a single-sparsity quick check; `--launcher "srun --jobid=<id> --overlap"` wraps
each run for schedulers like SLURM. Data uses random tensors (uniform
[-0.5, 0.5]), so the measured sparsity per point varies slightly with `--seeds`.

## PV-skip modes

`pv_threshold` per-Q-tile skip in the attention kernel is implemented in three variants, selectable at runtime via `-pv_mode={none|warp|block}`:

- **`none`** — skip disabled; baseline matching the no-PV-skip codegen instance.
- **`warp`** (per-wavefront) — each wavefront votes locally via `__shfl_xor` butterfly AND; SGPR-resident flag. Maps to upstream `PVThresholdMode::kPerWarp` ([`attn_utils.cuh`](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/attn_utils.cuh#L59)); the per-warp granularity is upstream's, only the butterfly-AND-of-bool implementation is CK-tile-specific.
- **`block`** (per-block) — block-wide consensus vote via LDS broadcast; upstream `PVThresholdMode::kPerBlock` ([`qk_int_sv_f16_cuda_sm80.cuh:L303`](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh#L303)). V loads stay unconditional in all modes — the guard wraps the PV MMA only, matching upstream and paper Algorithm 1.

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
