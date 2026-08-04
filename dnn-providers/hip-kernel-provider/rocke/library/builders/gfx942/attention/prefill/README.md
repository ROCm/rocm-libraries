# gfx942 dense flash-attention prefill (`attention_dense`) — AICK-1664

Port of the gfx950 dense prefill kernel (PR #9480 / AICK-1663) to **gfx942 (CDNA3)**.
This folder mirrors the gfx950 layout (`builders/gfx950/attention/prefill/`): the
kernel body lives in `kernels/gfx942/attention_dense.py`; this builder owns the host
path (spec → compile → launch → SDPA parity + benchmark).

## Status

**P0 landed.** `build_attention_dense(arch="gfx942")` emits the correctness-first
body (`_build_attention_dense_p0`): 32×32×8 atom + K-loop doubling, transposed-QK,
non-pipelined single-buffer LDS, element-wise V read. Validated against an fp32 SDPA
reference across the in-scope cohort, 0 register spill within the VGPR/LDS budget.

**P0 is deliberately not perf-competitive.** Being non-pipelined with an element-wise
V read, it is expected to be slower than the generic `attention_tiled_2d` gfx942
kernel until P1–P3 land. Opt in for correctness and bring-up, not for throughput.

Deferred, each rejected with a structured reason by `supports_attention_dense`:
persistent grid (P4), varlen, ragged, sliding-window. See the AICK-1664 plan for the
full ordered work list and the deferred-findings backlog.

## Scope

gfx942 only · forward-inference prefill · dense causal (no paging / bias / SWA / sinks)
· bf16 + fp16 · head dims **D64 & D128** · MHA + GQA incl. non-power-of-2 (40/8, 28/4).
The **persistent** variant is in the story scope but is P4 — only the default grid is
implemented today. D256 is out of scope (AICK-1495/1496).

`block_n` must divide the 256-row query tile, and `K_lds + V_lds` must fit the 64 KB
gfx942 LDS — both are enforced by `supports_attention_dense`, so an out-of-budget
config is rejected with a reason rather than aborting in codegen.

## Why a separate kernel (not an arch branch in the gfx950 file)

The gfx950 body bakes in CDNA4-only primitives; the algorithm genuinely diverges on
CDNA3, so the DSL convention (`dsl_docs/architecture/multi_arch_data_layout.md`) puts
it in a per-gfx module. This also keeps the gfx950 golden IR byte-identical by
construction. The CDNA3 deltas:

| Concern | gfx950 (CDNA4) | gfx942 (CDNA3) |
|---|---|---|
| MFMA atom | `mfma_f32_32x32x16` (K=16) | `mfma_f32_32x32x8` (K=8) → 2× per K=16 tile, A/B repack; **C-layout identical** |
| Conflict-free V | `ds_read_b64_tr_b16` (transpose read) | **no `ds_read_tr16`** → `perm_b32` store-path transpose |
| Cross-half exchange | `permlane32_swap` | **absent** → `perm_b32` / `ds_bpermute` |
| LDS / CUs | 160 KB, one CU count | 64 KB, two CU counts (228 / 304) → retune occupancy, `num_persistent` |
| Tile barrier | bare `s_barrier` is safe (NBUF=2 double buffer) | single buffer → the tile-end barrier **must** drain `lgkmcnt` (`sync_lds_only`) |

## Problem category (drives the optimization order)

The shipped gfx942 tiled-2D prefill kernel is **LDS-bank-conflict-bound →
MFMA-starved** (not compute-bound, not HBM-bound): the V-read conflict rate dominates
and leaves the MFMA pipe mostly idle, rather than sitting at the HBM-bound roofline
this problem should reach on this part. So conflict-free V and softmax-VALU reduction
rank first; compute-side scheduling levers that win on the compute-bound gfx950 dense
kernel (`s_setprio`, diagonal two-phase peel) are **proven-negative on gfx942** and are
not ported.

Per `AGENTS.md`, measured conflict rates, pipe utilisation and per-lever deltas are
recorded outside the repo — see the AICK-1664 plan and the protected results page.

## Lever port map (gfx950 dense → gfx942)

Impact is ranked qualitatively; magnitudes live outside the repo.

| gfx950 lever | Expected gfx942 impact | gfx942 status | port action |
|---|---|---|---|
| 32×32 atom + K-loop doubling | enablement (required) | already in tiled_2d (`use_mfma_32x32x8`) | **lifted** (P0, done) |
| CK-1 transposed PV | **dominant** | vehicle prototyped in tiled_2d; re-verified correct on `develop` | **lift** (P1, highest impact) |
| LDS K bank-conflict pad | large | shipped at pad=8 for D128; D64 unpadded by construction | **retune width** for the gfx942 bank geometry, extend to D64 (P3) |
| `exp2_fast` | moderate | ABSENT on gfx942 | **new, portable** (P2) |
| lazy_rescale | marginal | ABSENT on gfx942 | **new, portable** (P2) |
| wide4 WG=256 | large | shipped/proven on gfx942 | **retune** (P3) |
| K single-buffer | **prerequisite** | NBUF=2 does not fit 64 KB at D128 | required before any pipelining (P3) |
| persistent grid-stride | moderate | ABSENT in tiled_2d | **new outer loop** (P4) |
| PV-only `s_setprio` | **proven-negative** | tested and disabled on gfx942 | **DO NOT PORT** |
| diagonal two-phase peel | **proven-negative** | reverted on gfx942 | re-test only if the bound shifts (P5) |

## Bench

```
python attention_dense_prefill.py                 # parity + bench, default shapes
python attention_dense_prefill.py --dtype fp16 --d 64
```

`--persistent` / `--sw` are accepted by the CLI but rejected at build time until P4/P1.

Full-cohort parity and perf are driven by the live harness at
`benchmarks/gfx942/attention/prefill/benchmark_dense_prefill_live.py` (`--mode all`),
which is the numeric gate for this kernel — the same role the bench plays on gfx950.
Per AGENTS.md, **measured throughput lives only in the protected results page, never
in the repo.**
