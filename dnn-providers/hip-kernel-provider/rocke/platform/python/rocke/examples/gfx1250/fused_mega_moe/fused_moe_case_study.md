# Case Study: Fused-MoE on gfx1250 (Qwen3-30B-A3B day-0)

**Goal.** Bring up the Qwen3-30B-A3B fused-MoE operator on gfx1250 (gfx1250-class,
wave32, WMMA) and reach **performance parity with gfx950** (CDNA4, wave64,
MFMA) on the same model, while everything stays numerically correct.

**Status at checkpoint.**
- Single-launch WMMA fused-MoE mega-kernel: **implemented, GPU-validated correct**.
- **Decode parity (M ≤ 256): ACHIEVED** (0.98–1.05× of gfx950).
- **Batched parity (M ≥ 512): 0.72–0.93×** — much improved from the 0.55–0.68
  starting point, not yet fully at parity in mid-batch.

All numbers below are HIP-event-timed (200 iters, 30 warmup), bf16, A3B GEMM
dims `H=2048, I=768, H_out=2048`, gfx950 local + gfx1250 remote (GPU 2).

---

## 1. The kernel

A **single launch** computes the full per-expert MoE path with **no HBM
intermediates** (`instances/gfx1250/fused_moe_mega_wmma.py`):

```
gate GEMM + up GEMM  (shared LDS-resident A, dual-B)
  -> SiLU(gate)*up    (in registers)
  -> Hidden           (staged into PERSISTENT LDS, never HBM)
  -> down GEMM        (reads Hidden from LDS as its A operand)
  -> weighted, token-masked ATOMIC reduce into Y
```

It mirrors the gfx950 MFMA mega (`instances/common/moe_fused_mega.py`)
structurally and swaps only the arch-specific primitive: the **WMMA 16×16×32**
atom (a/b frag = 16, c frag = 8, wave32, 128-thread block) instead of MFMA
(wave64, 256-thread block). The cshuffle→down "reshape" is implicit (Hidden's
write address == the down A-read address in a row-major LDS tile).

---

## 2. The eval recalibration (what we got wrong first)

The single most important lesson:

- **TFLOPs at a synthetic `M=512` shape is the wrong metric.** A3B decode has
  `batch=2, topk=8` → 16 active (token,expert) pairs → the MoE M-dimension is
  ~16–256 rows. At that size the kernel is **memory-/latency-bound**, not
  compute-bound, so TFLOPs is meaningless.
- The right metric is **wall-clock µs at the real A3B shapes** (decode M≈16–256
  *and* batched M≥512), scored as **parity = gfx950_µs / gfx1250_µs**,
  **best-config vs best-config** on each arch.
- A subtle trap: an early "gfx1250 wins at large M (1.48×)" result was an
  artifact of comparing **tuned gfx1250 vs *untuned* gfx950**. Always tune both
  sides before claiming parity.

Harness: `examples/gfx1250/qwen3_30b_a3b/a3b_parity_scoreboard.py` (+
`examples/gfx1250/fused_mega_moe/fused_mega_moe_bench.py` for the config sweeps).

---

## 3. Knobs & levers tried

| Lever | What it does | Regime it helps | Verdict |
|---|---|---|---|
| **`tile_n_down = H_out`** (single down tile) | removes the down output-tile loop + repeated atomic-reduce passes | decode | ✅ **biggest decode win** (down to 63 µs) |
| **`tile_n_inter ↓`** (more grid.x workgroups) | fixes occupancy starvation (only 3 WGs at tni=256) | decode | ✅ **big** (188→113 µs at tni 256→64) |
| **`tile_m = 32`** | 2 WMMA M-atoms/block → amortizes weight stream over more rows → fewer m-blocks | batched | ✅ **big** (M=512 0.60→0.93, M=4096 0.68→0.86) |
| **`double_buffer`** (LDS ping-pong, 1 barrier/K-tile) | overlap next-tile global→LDS with current WMMAs | decode + batched | ✅ small/consistent (~3–5%) |
| **`wmma_v1`** (WMMA intrawave ds_read/wmma scheduler) | keep the matrix pipe fed | **compute-bound GEMM** | ✅ **+2.6% on square GEMM**; ❌ regresses the memory-bound MoE |
| `waves_per_eu` (2/4) | AMDGPU occupancy hint | — | ❌ neutral / negative |
| `warp_n = 8` (256-thread block) | more threads/block | — | ❌ regressed (tile_m=16 can't feed 8 warps) |
| **`tile_m = 64`** | 4 M-atoms/block | batched | ❌ **spills** on gfx1250's 128-thread wave32 blocks (33–45 TFLOPs); works on gfx950 wave64 |

### What worked
- **Occupancy + tile-shape tuning dominates the decode regime.** The decode MoE
  is latency-bound (~1% of peak HBM BW at 1 m-block); the wins came from giving
  the GPU more parallel work (`tile_n_inter↓` → more grid.x) and cutting
  fixed-overhead loops (`tile_n_down = H_out`).
- **`tile_m = 32`** is the key batched lever: more rows per weight load.
- **`double_buffer`** is a small, free, always-on-able win.
- The **WMMA scheduler** (`helpers/schedule.py: WmmaHotLoopInstList` +
  `SchedulePolicy.emit_wmma_compute_schedule`) is genuinely useful — but only
  for **compute-bound** WMMA matmuls (square GEMM, prefill, LM head), not the
  memory-bound decode MoE.

### What did not work
- **`waves_per_eu`** and **`warp_n=8`**: no benefit; the block is too small to
  use more waves/threads at `tile_m=16`.
- **`wmma_v1` on the MoE**: regressed it (~25.8→21.0 TFLOPs at M=512) — forcing
  a `ds_read→wmma` order over-constrains the scheduler when the bottleneck is
  VMEM, not LDS latency.
- **`tile_m=64` on gfx1250**: register spill on 128-thread wave32 blocks. This
  is the structural reason gfx1250 lags gfx950 in mid/large batch — gfx950's
  256-thread wave64 blocks absorb `tile_m=64` (→ 183 TFLOPs at M=4096).

---

## 4. Final performance scoreboard (MoE, bf16, best-vs-best)

| M | ~decode batch | gfx950 µs | gfx1250 µs | **parity** | gfx950 TFLOPs | gfx1250 TFLOPs | gfx1250 best config |
|---|---|---|---|---|---|---|---|
| 16 | — | 55.1 | 56.3 | **0.98** | 2.7 | 2.7 | tm16 tni64 tnd2048 db |
| 32 | — | 60.4 | 57.4 | **1.05** | 5.0 | 5.3 | tm16 tni64 tnd2048 db |
| 64 | — | 66.7 | 66.0 | **1.01** | 9.1 | 9.2 | tm16 tni64 tnd2048 db |
| 128 | b≈1 | 66.9 | 64.2 | **1.04** | 18 | 19 | tm16 tni64 tnd2048 db |
| 256 | **b=2 (decode)** | 66.8 | 65.8 | **1.02** | 36 | 37 | tm16 tni64 tnd2048 db |
| 512 | b≈4 | 70.0 | 75.6 | **0.93** | 69 | 64 | tm32 tni64 tnd2048 db |
| 1024 | b≈8 | 111.2 | 145.6 | **0.76** | 87 | 66 | tm32 tni64 tnd2048 db |
| 2048 | b≈16 | 132.9 | 185.9 | **0.72** | 145 | 104 | tm32 tni256 tnd512 db |
| 4096 | b≈32 | 210.6 | 246.2 | **0.86** | 183 | 157 | tm32 tni256 tnd512 db |

Config legend: `tm`=tile_m, `tni`=tile_n_inter, `tnd`=tile_n_down, `db`=double_buffer.
gfx950 best configs: decode `tni128/tnd512`; batched `tni256/tnd512` with
`tm16/32/64` rising with M.

### Decode-parity progression (M=256, the real decode point)
| stage | gfx1250 µs | parity |
|---|---|---|
| baseline (tni256, single-buffer) | 187.9 | 0.59 |
| + occupancy (tni64) + single down tile (tnd2048) + double_buffer | 63–66 | **1.02–1.06** |

### Batched-parity progression (`tile_m` added)
| M | before (tm16) | after (tm tuned) |
|---|---|---|
| 512 | 0.60 | **0.93** |
| 1024 | 0.69 | **0.76** |
| 2048 | 0.71 | **0.72** |
| 4096 | 0.68 | **0.86** |

---

## 5. Two regimes (the mental model)

- **Decode (M ≤ 256, latency-bound, ~1% peak BW):** win with **parallelism +
  fewer fixed-overhead loops** (grid.x, single down tile). Bytes-reduction (fp8)
  and compute-scheduling do **not** help here. → **parity reached.**
- **Batched (M ≥ 512, compute/BW-bound):** win with **`tile_m`** (amortize
  weights) and, in principle, compute scheduling. gfx950 pulls ahead because its
  wave64/256-thread blocks + `compv4` pipeline + `tile_m=64` capacity extract
  more compute; gfx1250's wave32/128-thread blocks spill before they get there.

Note: gfx1250's best decode configs use `tnd = H_out` (single down tile), which
only fits because gfx1250 has **more LDS** than gfx950 (gfx950 can't build it) —
a legitimate architectural advantage, not an unfair comparison.

---

## 6. Correctness

GPU-validated on gfx1250 (`examples/gfx1250/fused_mega_moe_wmma_verify.py`):
single-TG and multi-TG (grid.x inter-split + atomic reduce), multiple experts,
`double_buffer` on/off, `tile_m` 16/32/64 — all PASS (max rel ≤ 1.4e-6 vs a bf16
reference). No-GPU structure/lowering tests: `tests/test_moe_fused_mega_wmma.py`
(8) + `tests/test_wmma_schedule.py` (4), all green.

---

## 7. Remaining work (next levers, in priority order)

1. **Mid-batch register pressure (M=1024–2048, parity ≈ 0.72–0.76).** Make
   `tile_m=64` fit on wave32 (tighter accumulator scheduling / fewer live
   fragments) so gfx1250 can match gfx950's `tile_m=64` compute efficiency.
2. **fp8 / bf8 MoE.** Halves weight traffic — a day-0 quant requirement and a
   win in the BW-bound batched regime (reuse the validated
   `instances/gfx1250/block_scaled_gemm.py` WMMA fp8 core; mirror the scale
   plumbing from the gfx950 `moe_fused_mega_fp8` levels). Note: fp8 will *not*
   move decode parity (decode is latency-bound, already at parity).
3. **Whole-layer scoreboard.** Extend `a3b_parity_scoreboard.py` to the
   attention + dense-GEMM + norm + routing rows for an end-to-end layer parity
   number (MoE is ~60% of decode time).

---

## 8. Artifacts

| File | Role |
|---|---|
| `instances/gfx1250/fused_moe_mega_wmma.py` | the WMMA fused-MoE mega-kernel (+ double_buffer, tile_m, waves_per_eu, wmma_v1) |
| `helpers/schedule.py` | reusable WMMA scheduler (`WmmaHotLoopInstList`, `emit_wmma_compute_schedule`/`emit_wmma_hotloop`) |
| `examples/gfx1250/qwen3_30b_a3b/a3b_parity_scoreboard.py` | gfx950↔gfx1250 parity scoreboard harness |
| `examples/gfx1250/fused_mega_moe/fused_mega_moe_bench.py` | config-sweep bench (both arches) |
| `examples/gfx1250/fused_mega_moe/fused_mega_moe_wmma_verify.py` | GPU numeric verify |
| `tests/test_moe_fused_mega_wmma.py`, `tests/test_wmma_schedule.py` | no-GPU structure/lowering tests |
