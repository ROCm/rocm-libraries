# Running FlyTape on alola (gfx950), and its relation to ATOM

Status (2026-10-30): FlyTape has **not** been connected to ATOM. It runs standalone, through its own benchmark
script, on random weights at the checkpoint's shapes. ATOM is only present because FlyTape borrows the q38
container's Python environment. This page records how it was run, what the numbers mean, and what connecting it
to ATOM would take. Absolute numbers live in
`hackathon-prep/private-perf/qwen38-true-mono/flytape-baseline.md`; ratios only here.

## 1. What FlyTape is

`dev/flytape-main/flytape-main`: a FlyDSL persistent "occupancy" decode megakernel for gfx950. Its "qwen4" model
(`flytape/models/qwen4/`) is the Qwen3.8-Flash-Next decoder: 48 layers (36 GDN, 12 QSA), 512 experts top-10, the
gated-residual (HC) mixers, and the n-gram PLE table. One launch serves a closed batch until EOS, the token budget
or a host cancel. Details: `research/flytape.md`.

## 2. Environment used

- Nodes: the `rocke` partition (`smci355-ccs-aus-n07-17`, gfx950 MI355 OAM), one GPU for TP1, two for TP2.
- Container: the q38 enroot image `rocm-pytorch-rocm10.sqsh` (ROCm 10.0, torch 2.13) with ATOM's repro venv at
  `/work/venv`. It is used because that venv already has **FlyDSL 0.3.4.1**, the only version FlyTape is
  validated with. FlyTape imports nothing from ATOM.
- Node seeding: the image, venv and checkpoint live in node-local `/tmp/vanantha-q38`, which admins wipe. A
  wiped node is re-seeded with `~/rocke-sdpa-claude/q38/seed.sh` (one sbatch per node, about 11 minutes; markers
  in `~/rocke-sdpa-claude/q38/seeded/`).
- Tools: `hackathon-prep/tools/q38/q38_sync.sh` packs a local tree as a stage tarball in the Austin home;
  `q38_job.sh --mode container --stage <name>` unpacks it into the job's node-local directory and runs a command
  inside the container.

## 3. Steps

```bash
T=/c/Users/vanantha/Documents/dev/hackathon-prep/tools/q38
# stage the FlyTape tree and the harness
$T/q38_sync.sh /c/Users/vanantha/Documents/dev/flytape-main/flytape-main flytape
$T/q38_sync.sh /c/Users/vanantha/Documents/dev/flytape-rocke/harness fth
# TP1 (configs: precision:BATCHxPROMPT list; trace configs get a phase table)
Q38_NODE=smci355-ccs-aus-n07-17 $T/q38_job.sh --tag ft-tp1-a --gpus 1 --time 01:30:00 --mode container \
  --stage flytape --stage fth -- 'bash /job/stages/fth/ft_bench_tp1.sh'
# TP2
Q38_NODE=smci355-ccs-aus-n07-17 $T/q38_job.sh --tag ft-tp2-a --gpus 2 --time 01:30:00 --mode container \
  --stage flytape --stage fth --env TP=2 --env "CONFIGS=mxfp4:1x128,2x128,4x128,16x128,128x128" \
  --env TRACE_CFG=2x128 -- 'bash /job/stages/fth/ft_bench_tp1.sh'
```

`dev/flytape-rocke/harness/ft_bench_tp1.sh` does, inside the container:
1. sets `FLYDSL_RUNTIME_CACHE_DIR` job-local, disables core dumps;
2. runs FlyTape's CPU spec tests and the toy GPU tests as a smoke check (`tests/test_spec.py`, `tests/test_toy.py`);
3. runs `scripts/bench_qwen4_full.py --tp $TP --precision <p> --configs <list> --reps 3` per precision, and once
   with `--trace` for the per-phase table;
4. writes the JSON lines to `$Q38_OUT` (copied to `~/rocke-sdpa-claude/q38/results/<tag>/`).

## 4. What the benchmark measures

- Weights: `random_weights` at the full checkpoint shape (seed 0), with the ~51 GB n-gram table drawn on the GPU
  in chunks. No checkpoint is loaded.
- State: a zero decode state at position `prompt_len - 1`; no prefill. Prompts are random token ids. Routing is
  therefore near-uniform, not the served model's.
- Metric: the median gap between consecutive device step stamps (`s_memrealtime`), one token per row per step.
  It excludes the host, the server, the scheduler and sampling outside the kernel.
- Precision: FlyTape's `mxfp4` preset quantizes the routed and shared experts only (W4A4 in the tiled MoE, decoded
  W4A16 on the GEMV path at batch <= 4). Dense projections, the HC mixers, the router and the head stay bf16.
  ATOM's MXFP4 checkpoint runs those dense layers as MXFP4 GEMVs, so FlyTape reads more bytes than the mono.
- Phase traces come from a separately compiled stamped kernel; a phase's time includes the next sync's wait.

## 5. Results in brief (ratios)

- TP1 B1 device step is about 1.7x the plain FlyDSL mono's end-to-end per-token time at K=0 (and about 2x the
  hybrid stack's); TP2 B1 is still slower than the one-GPU hybrid. FlyTape's large-batch tiled tape scales to
  B128 at TP2, a regime the ATOM mono does not serve (ATOM uses its native path above CONC 4).
- B4 phase shares (TP1): HC mixers about 0.34, MoE about 0.31, GDN about 0.20, QSA about 0.10, head about 0.03.
- These comparisons are not like-for-like: device step vs served time, random vs real weights, bf16 vs MXFP4
  dense layers, independent rows vs a sequential verify, and TP2 (two GPUs) vs TP1. The fair-comparison rule is
  PLAN.md decision 5 (matched GPU count, batch, precision and metric; DP2 arms for every TP2 cell).

## 6. What connecting FlyTape to ATOM would take

Not done; listed so the gap is explicit.

1. **Weights:** a loader from the `Qwen3.8-Flash-Next-MXFP4` checkpoint into FlyTape's layouts
   (`flytape/models/qwen4/weights.py`, `precision.py`), including the 16x16 expert shuffle and E8M0 scale
   shuffle, and a preset whose dense layers are MXFP4 to match the checkpoint.
2. **Prefill hand-off:** FlyTape prefills on the host in PyTorch (`prefill.py`, out of memory past roughly 8K-16K
   tokens). ATOM would have to hand its prefilled KV cache, GDN state and PLE history to FlyTape's layouts.
3. **Serving:** a model runner in ATOM that launches FlyTape's occupancy per closed batch and reads tokens from
   its mailbox. That conflicts with ATOM's continuous batching and its per-step scheduler, and FlyTape has no MTP.
4. **Numerics:** an accuracy run (GSM8K, HumanEval, RULER) against the mono, since FlyTape's reduction orders and
   W4A4 MoE differ from ATOM's.

For this project the comparison target remains the ATOM hybrid stack (small batch) and FlyTape's own benchmark
plus ATOM's native path (large batch, `DESIGN-LB.md`). The true rocKE mono connects to ATOM through the
`ATOM-truemono` hook (`probes/WP0a.md`), not through FlyTape.
