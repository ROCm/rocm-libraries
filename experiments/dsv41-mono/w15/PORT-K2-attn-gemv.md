# W15 PORT K2-attn-gemv: ATOM's K2a chain and K1's GEMV / mHC stages as rocKE stage functions

W15, 2026-11-04 / 05. MI355X (gfx950), TP2 no-EP, DSpark 5, the B0 recipe (bf16 KV, fp8 index plane, `--level 0` +
FULL graphs). Ratios only. Absolute numbers are in
`hackathon-prep/private-perf/dsv41-mono/w15/PORT-K2-attn-gemv-abs.md`; job logs and result tarballs are on BeeGFS
`/scratch/users/vanantha/w15ka/{logs,results}` (n01-09); n07-17's ran from its node-local `/tmp/vanantha-w15ka` and
were read from there (part of them was lost with the node, §3.5).

This continues the port the first W15 agent left after the 2026-11-04 network drop (PLAN decision 14). Its code
(`stage_gate.py`, `stage_norm.py`, `stage_slice.py`, `stage_router.py`, `stage_attn.py`, `seam_common.py`, the ATOM
hooks in `rk_extern.py` / `attn_pre.py` / `attn_post.py` / `layer_post.py`) and its rig (`hackathon-prep/dsv41/w15/rig/`)
were kept. Its K1 rig and ATT jobs (a1-a3) are reused here; nothing was redone.

## 0. Verdict

- **Ported, bitwise, every served width, both nodes.** The whole K2a chain runs as rocKE stage functions: `attn_score`
  (`attn`), `irq` (PV + combine + inverse RoPE + FP8 quant), `wo_a`, `wo_b`, the FFN seam's `slice` (with
  `slice_prefetch` + `stage_reduce`), `norm` and `gate`, and K2b's `router`; with K1's GEMV family (`wqkv_a`, `wq_b`)
  and K1's mHC seam (`slice`, `norm`, `gate`). In ATOM's served engine (eager, in place, every byte a launch writes
  compared, LDS poisoned between reference and candidate) the full set is **bitwise** with ATOM's FlyDSL stages on
  every K1 and K2 launch: 3 DSpark seeds (target 6 / 12 / 18 / 24 / 36 / 42 / 48 rows, draft 5 / 10 / 15 / 20 / 25 /
  30 / 35 / 40) plus the native path (1 and 4 rows), and with the `sop=2,ra=1` GEMV build. 0 differing launches in
  67,714 per rank on n01-09 and 68,770 on n07-17 (§2).
- **Mutants.** 12 must-fail mutants over the six new stages, each in its own engine run, on both nodes: all caught (10
  by byte differences in exactly the regions the mutated stage writes, 2 by deadlocking the layer: a `router` logit
  put to the wrong part slot and an `irq` payload with a flipped bit both leave a consumer polling forever). The K1
  rig catches the seam mutants on both nodes at 1-12 rows, 3 seeds.
- **Knob-off identity, both nodes.** With the environment unset, `.pre.ll` and `.text` of K1 (s6 ix0 / ix1, s12, s24,
  draft s5) and K2 (s6, s12, s48, draft s5) equal pristine ATOM `df86c482` on n01-09 (a4b) and n07-17 (a5n). The tree
  checked is the current `dev/ATOM-dsv41`, which also holds the K2-moe agent's hooks.
- **In situ (§3):**
  - **Graph-mode GPU decode step (HTL, n01-09, 4 rotated passes over 2 jobs): the full set is a WIN at C1
    (0.987-0.993), C2 (0.984-0.991) and C8 (0.990-0.996), every pass below every A-vs-A twin (0.991-1.007); a TIE at
    C4** (0.988-0.996). With the K1 `norm` restructure (`rkwv`) 0.984-0.991. The n07-17 confirmation ran but was lost
    when n07-17 went down at 05:46 UTC (§3.5): a one-node result until it is re-run.
  - K1 graph rig, both nodes: the parity seam lost 0.5-1.2 % at 3-6 rows; diagnosed with ATT (`norm`'s bf16 weight
    loads) and restructured (`wvec=1`, one `dwordx4` a chunk). **The full K1 port with it (`allwv`) is a WIN on both
    nodes: 0.988-0.992 at 5-6 rows and 0.960-0.967 at 11-12 rows**, a tie at 3 rows (§3.1b).
  - K2 in-place eager A/B against a recompiled no-op control: TIE at every width (§3.2). Graph-mode stage timeline:
    K2 0.965-0.983 up to 24 rows in one pass, with twins up to 1.7 % off (§3.3): not a verdict on its own.
- **WaveScope ledger of OUR code (§4).** At K2 6 rows our stages are wait- and stream-bound like ATOM's: the K2 slice's
  TP reduce poll (0.42 of all our traced latency; the hand-over from both ranks' `wo_b`), `irq`'s CTA barriers, the
  weight / key streams of `wo_a` / `wo_b` / `attn_score`, the `norm` and `router` polls. No idle-wave or LDS problem
  except `gate` (structural, 0.001 of K2). PMC: the same MFMA work, fewer VMEM instructions, 16.5 % more dynamic SALU
  in K2 (poll spinning, not a calling cost: the functions are inlined).
- **COMGR vs Rust direct (§5).** The direct compiler builds every stage as a standalone kernel (no refusal), but it
  cannot build a device function linked into a FlyDSL kernel, so in situ the auto-select is FlyDSL's LLVM for all
  stages. Statically, direct loses on every stage form (more instructions, more `vmcnt(0)` drains except in `attn`,
  more VGPRs); the defects are named in §5 (DPP not folded into the VALU op, no `ds_read2` pairing, unfolded address
  arithmetic, a full drain on both sides of every barrier). None of 7 lever sets moves them by more than 3 %.
- **First knob surface (§6):** every hard-coded choice in the port is listed, default = parity; 25 are already knobs
  and the first climb step on them (`wvec=1`) is the K1 win above.

## 1. What was ported

All in `copy-rocm-lib/dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx950/dsv41_mono/` (new files; nothing in
`qwen38_mono` changed). Each module builds an `lds="arg"` rocKE `KernelDef` lowered as an LLVM device function; ATOM
links it with `flydsl.compiler.extern_link` in place of one FlyDSL stage body (`ATOM_DSV41_RK_EXTERN=<stage>=<dir>;..`,
`atom/models/deepseek_v41/mono/kernels/rk_extern.py`), additive and default off.

| stage | module | symbol / variants (served set generated) | ATOM call site | knobs | mutants |
|---|---|---|---|---|---|
| K2a `attn_score` | `stage_attn.py` | `rocke_dsv41_attn_p<per>x<ix>n<S>`, rows 1-48, ix 0 / 1 | `attn_post._score_task` | SeamKnobs + `askip`, `stat_w0` | `put_bit`, `key` |
| K2a `irq` (PV, combine, inverse RoPE, MXFP8 quant) | `stage_attn.py` | `rocke_dsv41_irq_g<groups>x<ix>n<S>` | `attn_post._rk_irq` | as above | `put_bit`, `alpha`, `sink` |
| K2a `wo_a` (+ `quantize_fp8`) | `stage_woa.py` (W13 WP4) | `rocke_dsv41_woa_n<S>`, one tile | `attn_post` | `WqkvKnobs` (`sop`, `ra`, poll, pause, ..) | WP4 set |
| K2a `wo_b` | `stage_wob.py` (W13 WP4) | `rocke_dsv41_wob_n<S>`, one tile | `attn_post` | `WqkvKnobs` | WP4 set |
| K2a FFN seam `slice` (+ prefetch + reduce) | `stage_slice.py` | `rocke_dsv41_slice_k2n<S>` | `attn_post._rk_slice` | SeamKnobs + `batch`, `fn_early_max` | `put_bit`, `res_bit`, `mix_drop` |
| K2a FFN seam `norm` | `stage_norm.py` | `rocke_dsv41_norm_k2c16n<S>` | `attn_post._rk_norm` | SeamKnobs + `wvec`, `wearly` | `put_bit`, `row`, `rsq_drop` |
| K2a / K1 seam `gate` | `stage_gate.py` | `rocke_dsv41_gate_n<S>` | `attn_post._rk_gate`, `attn_pre` | SeamKnobs + `early` | `put_bit`, `sink`, `slice_drop`, `row`, `order` |
| K2b `router` (+ `await_x`) | `stage_router.py` | `rocke_dsv41_router_n<S>e<128/384>` | `moe` via `layer_post` | SeamKnobs + `early_from`, `w_cpol`, `x_cpol`, `await_` | `put_bit`, `chain_drop`, `part` |
| K1 seam `slice` | `stage_slice.py` | `rocke_dsv41_slice_k1f<F>a<A>n<S>` | `attn_pre` | as K2 slice | + `pend0` |
| K1 seam `norm` (+ MXFP8 quant) | `stage_norm.py` | `rocke_dsv41_norm_k1n<S>` | `attn_pre` | as K2 norm | + `code` |
| K1 `wqkv_a` | `stage_wqkv.py` (P2) | `rocke_dsv41_wqkv_n<S>`, one tile | `attn_pre` | `WqkvKnobs` | P2 set |
| K1 `wq_b` (+ GPT-J RoPE) | `stage_wqb.py` (W13 WP4) | `rocke_dsv41_wqb_n<S>`, one tile, TP2 | `attn_pre` | `WqkvKnobs` | WP4 set |

The DPP reductions (`shfl=dpp`: ATOM's two bank-masked `update_dpp`s in VALU form, plus `ror` = one `row_ror:8` and
`swz` = `ds_swizzle` as knobs) live in `seam_common.py` and are shared by every seam stage. The FP8 GEMV core with
`sop` / `ra` is `stage_wqkv`'s. The mHC seams follow the Qwen WP2a HC port pattern (one CTA a token for gate / norm,
32-column tasks for slice, f32 partials put as tagged pairs, the 8-wave sums in wave order).

When a stage applies:
- the seams, `router`, `attn` and `irq`: every width, every layer kind, TP2 (`attn` / `irq`) and not with
  `ATOM_MONO_DEBUG`, NUMA or CCL builds;
- the GEMV family: one-token-tile builds (own rows <= 16), as in WP4.
A missing variant raises unless `ATOM_DSV41_RK_EXTERN_MISSING=mono` (that build keeps the FlyDSL stage).

## 2. Gates

### 2.1 In place in the served engine, n01-09 (job a4b)

Offline engine, `--enforce-eager`, `ATOM_DSV41_RK_LEV_CHECK`: every K1 / K2 launch is run as ATOM's build and as the
extern build from the same snapshot, LDS poisoned in between; every written byte (26.5 MB a speculative launch, 42.7 MB
native) compared. Counts per rank (both ranks identical):

| arm | set | plan (conc x requests) | launches | rows seen (K1 and K2) | differing |
|---|---|---|---|---|---|
| g0 | all 10 stages | 1x4, 2x4, 4x8, 8x8 (seed 0) | 12,050 | 5, 6, 10, 12, 15, 18, 20, 24, 35, 40, 42, 48 | 0 |
| g1 | all 10 stages | same, seed 1 | 11,940 | + 30, 36 | 0 |
| g2 | all 10 stages | same, seed 2 | 12,394 | + 25, 30 | 0 |
| r0 | all 10 stages | native 1x3, 4x4 | 19,200 | 1, 4 | 0 |
| gs | the GEMV family with `sop=2,ra=1` | 1x4, 2x4, 4x8, 8x8 | 12,130 | as g0 | 0 |

Draft launches (`DraftMonoRunner`, the e128 / top-3 K2) are in every speculative arm (900-954 a run).

Mutants (one engine run each, only the mutated stage linked, 1x2 + 4x4):

| mutant | stage | launches | differing | regions that differ |
|---|---|---|---|---|
| `key` | attn | 5,110 | 2,555 (every K2) | scratch, peers, `res_ffn0`, `gates_ffn0..2` |
| `put_bit` | attn | 5,110 | 2,555 | same |
| `alpha` | irq | 4,944 | 2,472 | same |
| `sink` | irq | 5,460 | 2,730 | same |
| `put_bit` | irq | HANG (layer deadlocks) | 49 of 99 before the hang | caught |
| `part` | router | HANG (route waits for the empty part slot) | - | caught |
| `chain_drop` | router | 5,196 | 2,598 | scratch, peers, `ffn_out0` |
| `put_bit` | router | 5,374 | 2,687 | scratch (+ 8 `ffn_out0`) |
| `order` | gate (K1 and K2) | 5,288 | 5,288 | `gates_attn0..2`, `gates_ffn0..2` |
| `rsq_drop` | norm (K1 and K2) | 5,196 | 4,716 | `q0`, `ffn_normed0`, `ffn_out0`, scratch |
| `res_bit` | slice (K1 and K2) | 5,110 | 4,868 | `res_attn0`, `res_ffn0` |
| `mix_drop` | slice (K1 and K2) | 5,202 | 5,202 | `gates_*`, scratch |

`order` (the slice sum reversed, an f32 reassociation) is labelled report-only in the module; in the engine it does
change bits, so it counts as a caught mutant here. The two hangs are caught by the arm's timeout (a mutant that
deadlocks the layer cannot pass).

### 2.2 Second node, n07-17 (job a5n2)

Same arms, node-local harness under `/tmp/vanantha-w15ka` (n07-17 has no BeeGFS since its 2026-11-04 reboot):

| arm | launches a rank | differing |
|---|---|---|
| g0 / g1 / g2 (DSpark, 3 seeds) | 12,658 / 12,388 / 12,566 | 0 / 0 / 0 |
| r0 (native 1 and 4 rows) | 19,200 | 0 |
| gs (`sop=2,ra=1` GEMV family) | 11,958 | 0 |

All 12 mutants caught again, with the same regions and the same two hangs (`irq` `put_bit`, `router` `part`):
`attn` `key` 2,687 / 5,374, `put_bit` 2,601 / 5,202; `irq` `alpha` 2,472 / 4,944, `sink` 2,558 / 5,116; `router`
`chain_drop` 2,515 / 5,030, `put_bit` 2,684 / 5,368; `gate` `order` 5,030 / 5,030; `norm` `rsq_drop` 4,716 / 5,196;
`slice` `res_bit` 5,037 / 5,288, `mix_drop` 5,030 / 5,030.

### 2.3 K1 rig (synthetic weights, every output buffer and the whole mailbox scratch, 3 seeds)

Bitwise for `all` (seams + GEMV pair), `seam`, `wq` and `wv` at 1 / 3 / 5 / 6 / 11 / 12 rows on n01-09 (a4e) and
n07-17 (a5n, a3: 14 knob arms too: `ror`, `swz`, `asm` poll, `p0` pause, `lds` barrier, `e1` gate early, `wv`, `we0`,
`b2` / `b8` slice batch). Mutants `g_ord`, `n_rsq`, `s_res`, `s_mix` (and in a3 `g_put`, `g_sink`, `g_drop`, `g_row`,
`n_put`, `n_row`, `n_code`, `s_put`, `s_pend`, `s_sq`) are caught at every width they apply to.

## 3. In-situ timing

### 3.1 K1 graph rig (whole K1, ratio to ATOM's K1, median of 24 paired rounds; controls: recompiled FlyDSL K1 `rc`, A-vs-A graph twin `bb`)

| rows | node | all | seam (gate + norm + slice) | wq (wqkv_a + wq_b) | wv (seam, norm `wvec=1`) | rc / bb |
|---|---|---|---|---|---|---|
| 6 | n01-09 | 0.997 | 1.012 | **0.991** | - | 1.003 / 1.002 |
| 6 | n07-17 | 0.999 | 1.012 | **0.988** | 1.004 | 0.999 / 0.999 |
| 5 | n01-09 | 1.001 | 1.009 | **0.992** | - | 0.999 / 1.000 |
| 5 | n07-17 | 1.001 | 1.008 | **0.993** | 0.998 | 1.000 / 1.001 |
| 3 | n01-09 | 1.002 | 1.005 | 1.000 | - | 1.001 / 1.002 |
| 3 | n07-17 | 1.009 | 1.012 | 0.999 | 0.998 | 1.001 / 0.998 |
| 11 | n01-09 | **0.966** | 1.002 | **0.969** | - | 1.003 / 1.002 |
| 11 | n07-17 | **0.963** | 0.998 | **0.966** | 0.997 | 0.999 / 0.999 |
| 12 | n01-09 | **0.969** | 1.008 | **0.964** | - | 1.002 / 1.001 |
| 12 | n07-17 | **0.967** | 1.003 | **0.961** | 0.999 | 1.001 / 0.999 |
| 1 | both | 1.002-1.021 | 0.996-1.015 | 1.001-1.020 | 1.000 | 0.987-1.014 (controls spread over 2 %: no verdict at 1 row) |

Per-stage arms from a3 (n07-17, 6 rows): `gate` 0.997, `slice` 0.996, `norm` 1.011; the seam's loss is `norm`.

- **Verdict:** `wq` WIN at 5-6 rows (0.988-0.993) and 11-12 rows (0.961-0.969) on both nodes, the arm's range below
  both controls. `seam` LOSS at 3-6 rows (1.005-1.012), both nodes; diagnosed in §4.2 and restructured (`wv`: 0.997-1.004,
  a TIE). `all` = `wq` + `seam`: the seam loss eats the GEMV win at 3-6 rows, so `all` with `wv` is the K1 build to
  carry: §3.1b.

### 3.1b K1 combination screen with the restructured norm (rig, both nodes: a8 n01-09, a8n n07-17)

All arms = `gate` + `norm` + `slice` + `wqkv_a` + `wq_b`, bitwise (3 seeds), `n_rsq` caught. Values n01-09 / n07-17:

| rows | all (parity norm) | **allwv** (`wvec=1`) | allwvs (+ GEMV `sop=2,ra=1`) | allwvr (+ `shfl=ror`) | allwve (+ gate `early`) | allwvl (+ `red_bar=lds`) | wq only | rc / bb (n01-09; n07-17) |
|---|---|---|---|---|---|---|---|---|
| 6 | 1.000 / 0.997 | **0.990 / 0.988** | 0.999 / 0.996 | 0.991 / 0.989 | 0.993 / 0.990 | 0.992 / 0.987 | 0.991 / 0.990 | 1.002 / 1.001; 1.001 / 1.000 |
| 5 | 1.003 / 1.001 | **0.992 / 0.988** | 0.995 / 0.992 | 0.990 / 0.988 | 0.991 / 0.990 | 0.990 / 0.987 | 0.990 / 0.992 | 0.998 / 0.999; 1.000 / 0.999 |
| 3 | 0.999 / 1.008 | 1.000 / 0.997 | 1.007 / 1.004 | 1.000 / 0.999 | 0.997 / 1.002 | 0.996 / 0.999 | 0.997 / 0.999 | 0.998 / 1.000; 1.002 / 1.001 |
| 11 | 0.961 / 0.968 | **0.963 / 0.967** | 0.966 / 0.970 | 0.969 / 0.968 | 0.968 / 0.968 | 0.968 / 0.966 | 0.964 / 0.968 | 0.998 / 0.999; 1.002 / 1.002 |
| 12 | 0.969 / 0.965 | **0.963 / 0.960** | 0.967 / 0.962 | 0.963 / 0.958 | 0.963 / 0.959 | 0.963 / 0.959 | 0.965 / 0.959 | 1.003 / 1.002; 0.998 / 0.998 |

- **`allwv` is the K1 build to carry:** a WIN at 5, 6, 11 and 12 rows on both nodes (0.988-0.992 and 0.960-0.967,
  every arm range below the controls), a TIE at 3 rows. With the restructured norm the seam no longer costs anything
  and the whole K1 keeps the GEMV pair's gain.
- On top of `allwv`, `ror`, gate `early` and `red_bar=lds` are ties (within 0.4 %): kept as knobs for the climb.
  `sop=2,ra=1` on the K1 GEMVs is a small loss here (+0.3-0.9 % against `allwv` at 3-6 rows), as in WP4 for K1.

### 3.2 K2 (and K1) in-place eager A/B (rk_bench, n01-09, a4b)

Every K1 / K2 launch of the benched layers is run as ATOM's build and as each variant from the same state, 3 rounds in a
shuffled order, each launch bracketed by a TP barrier. Values are the per-launch medians of variant / ATOM (1,920 pairs a
cell); the column `aa` is the recompiled no-op control (`nop=1`: ATOM's own code under another name), which sits
0.7-1.3 % below 1.000 for K1 (a bias of the reference slot). Read every variant against `aa`.

| kernel | rows | aa | all | seam (gate, norm, slice) | att (attn, irq) | rtr (router) | gemv (wqkv, wqb, woa, wob) |
|---|---|---|---|---|---|---|---|
| K2 | 6 | 0.993 | 0.993 | 0.990 | 0.990 | 0.993 | 0.990 |
| K2 | 12 | 0.998 | 1.001 | 0.997 | 0.998 | 0.999 | 1.001 |
| K2 | 24 | 0.997 | 0.995 | 0.996 | 0.997 | 0.996 | 0.997 |
| K2 | 48 | 0.999 | 0.998 | 0.996 | 0.997 | 0.997 | 0.999 |
| K2 native | 1 / 3 / 4 | 1.001 / 0.994 / 0.995 | 1.005 / 0.995 / 0.995 | 1.002 / 0.993 / 0.992 | 1.002 / 0.992 / 0.995 | 1.004 / 0.995 / 0.996 | 1.001 / 0.994 / 0.994 |
| K1 | 6 | 0.987 | 0.998 | 0.999 | (0.995) | (0.995) | 0.987 |
| K1 | 12 | 0.990 | 1.000 | 1.000 | (0.999) | (0.998) | 0.991 |
| K1 | 24 | 0.990 | 0.996 | 0.998 | (0.996) | (0.996) | 0.986 |
| K1 | 48 | 0.988 | 0.998 | 0.998 | (0.998) | (0.997) | 0.989 |

(`att` and `rtr` do not touch K1: their K1 cells are extra no-op controls, and they show the K1 control spread, about
0.8 %.)

- **Verdict:** K2, every subset and the full set: TIE at every width (within 0.4 % of `aa`, inside the spread). K1:
  `gemv` ties the controls here (eager, warm weights), `seam` and `all` are 0.6-1.1 % above `aa` and inside the
  no-op spread (0.987-0.999), i.e. the seam loss from §3.1 is visible but not separable in eager mode.

### 3.3 Graph-mode stage timeline (gtl: timeline builds, FULL graphs, DSpark C1 / C2 / C4 / C8 + warm, n01-09, a4b)

Sum over the step's K1 / K2 launches of one kind (both ranks), the full set (`rk`) against the mean of ATOM (`b1`) and
its A-vs-A twin (`b1b`), one pass:

| kernel | 6 rows | 12 | 18 | 24 | 42 | 48 |
|---|---|---|---|---|---|---|
| K2 rk / base | 0.976 | 0.966 | 0.983 | 0.965 | 1.012 | 1.002 |
| K2 twin | 1.003 | 0.999 | 0.995 | 1.017 | 0.994 | 0.987 |
| K1 rk / base | 0.992 | 0.990 | 0.991 | 0.972 | 0.991 | 0.987 |
| K1 twin | 0.999 | 0.997 | 1.000 | 0.999 | 1.000 | 1.001 |

- K1: a WIN at every width in this pass (0.972-0.992 against a twin within 0.3 %), consistent with the rig.
- K2: 0.965-0.983 up to 24 rows against twins 0.995-1.017: promising, NOT a verdict (one pass; the 24-row twin is off by
  1.7 %; the eager same-state A/B in §3.2 says tie). The HTL (no-profiler host timeline, rotated passes) is §3.4-3.6;
  the n07-17 gtl was lost (§3.5).

### 3.4 Graph-mode GPU decode step (HTL: no-profiler host timeline, FULL graphs, DSpark, fresh engine per arm-run)

The whole GPU decode step (target K1 / K2 for every layer + draft + sampling), per CONC cell and rank, over the batch
sizes every run of the cell has; arm / mean(ATOM `b1`, its A-vs-A twin `b1b`), two rotated passes (values: pass 1,
pass 2; both ranks agree to 0.001):

| node | CONC | rk (the full set) | twin / base |
|---|---|---|---|
| n01-09 (a6d) | C1 | **0.987, 0.989** | 0.997, 1.004 |
| n01-09 | C2 | **0.988, 0.988** | 1.004-1.007 |
| n01-09 | C4 | 0.988, 0.996 | 0.991, 0.998 |
| n01-09 | C8 | **0.990, 0.991** | 0.999-1.001 |
| n07-17 (a7n) | C1-C8 | ran; lost with the node (§3.5) | |

- **Verdict (n01-09):** a WIN of 1.0-1.3 % of the whole GPU decode step at C1, C2 and C8 (both passes below the
  twin's range); C4 is inside the twin's spread (0.988 / 0.996 against 0.991 / 0.998): TIE. The eager same-state
  bench (§3.2) cannot see this (K2 ties there): the gain is in graph mode, where the launches run back to back and the
  stage latencies, not the per-launch overheads, set the step. Second node: §3.5.

### 3.5 Second-node confirmation

n07-17 ran the full gate (§2.2), the K1 rig screens (§3.1, §3.1b), and then the K2 bench + gtl (a5n2) and the HTL
(a7n, 2 passes). The node went `down*` (not responding) at 05:46 UTC on 2026-11-05, before those results were copied
off its node-local `/tmp` (it has had no BeeGFS since its 2026-11-04 reboot): the K2 bench, gtl and HTL of n07-17 are
lost and slurm requeued a5n2 (cancelled). A second n01-09 HTL job with 2 more passes and the `wvec=1` arm (a9b) is §3.6,
and a replacement second node (m07-13, partition `miopen`) is §3.7. **n07-17 itself is unseeded and without BeeGFS since; the second-node confirmation runs on m07-13 (§3.7).**

### 3.6 More n01-09 passes, with the restructured norm (a9b)

A second job, same node, other GPUs, arms `b1`, `rk`, `rkwv` (the full set with `norm` `wvec=1`), `b1b`, 2 rotated
passes (values pass 1, pass 2):

| CONC | rk | rkwv | twin / base |
|---|---|---|---|
| C1 | 0.993, 0.987 | 0.991, 0.986 | 1.001-1.003 |
| C2 | 0.991, 0.984 | 0.988, 0.984 | 1.000-1.006 |
| C4 | 0.996, 0.991 | 0.991 (pass 1) | 0.996-1.001 |
| C8 | 0.996, 0.993 | 0.990 (pass 1) | 0.998-1.004 |

Pass 2 of `rkwv` lost its C4 / C8 cells: the node's `/tmp` filled up (other workflows' jobs) and the HTL writer died
with `ENOSPC` (rc 124); not a fault of the arm.

**n01-09, 4 passes over 2 jobs:** `rk` is 0.987-0.993 at C1, 0.984-0.991 at C2, 0.988-0.996 at C4 and 0.990-0.996 at
C8, against twins at 0.991-1.007. C1, C2 and C8: every pass below every twin (WIN, 0.4-1.6 % of the GPU decode
step). C4: two passes inside the twin range (TIE). `rkwv` is equal to or below `rk` in every cell it has (0.984-0.991):
the K1 `wvec=1` gain carries into the step, inside the noise of this many passes.

### 3.7 Second node for the graph-mode verdict: m07-13 (partition `miopen`, a10m)

Attempted and failed for an environment reason: m07-13 (and n14-05, same setup) has BeeGFS and a dsv41 seed (W16's),
but its pyxis containers see no GPU (`/dev/kfd` maps to `nobody`, `rocminfo` has no gfx agent), so every arm of a10m
died at engine start. At the time of writing n07-17 is unseeded without BeeGFS and n01-09 is draining (`SlurmdSpoolDir
is full`): no second MI355X node can run the container engine. **The graph-mode step verdict (§3.4, §3.6) is a
one-node result; everything else in this note (gates, identity, K1 rig wins) holds on both nodes.** The harness now
stops a GPU job whose container sees no GPU (exit 125) instead of running 0-GPU arms.

## 4. WaveScope ledger of OUR code

Method: `ROCKE_DEBUG_LOC=1` builds of the six new stages (`rk_dbg`; the GEMV family plain), stage-marked debug dumps of
the same K1 / K2 builds split with `rv_isa split` (the linked rocKE code carries its DWARF, so the ATT `code.json`
lines map to `stage_*.py` / `seam_common.py` source lines), engine ATT (`rocprofv3 --att`, eager, DSpark C1, K2 s6
non-draft indexer layer and K1 s6 f1 ix0, iterations 30-31, one target CU a run), `w15_att.py analyze` (per-stage
ledger + per-source-line ledger) and the new `h/ledger_cmp.py` (ours vs ATOM per stage). Dispatches whose wall time is
dominated by the first stage's wait for the previous launch / peer (5-10x the median dispatch) are excluded. The K1 rig
ATT (a3, 16 CUs, synthetic weights) gives the K1 seam lines.

### 4.1 K2 at 6 rows (engine, n01-09: a4e CU 1 + a6c CUs 0-3; 11 clean dispatches a arm out of 20)

Fractions of each stage's own traced latency (`poll` and `vmcnt` overlap: a poll loop's wait is a `vmcnt` wait;
`VMEM` = load issue latency; `idle` = wave-slot cycles without a wave). `share` = the stage's part of all traced K2
latency in our build; `lat/A` = our stage's latency over ATOM's, pooled over the same CUs.

| rank | stage | share | lat/A | vmcnt | lgkmcnt | barrier | poll | MFMA | VALU | LDS | VMEM | idle | ATOM: vmcnt / barrier / poll / VMEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `slice` (+ reduce) | 0.128 | 0.82 | 0.60 | 0.03 | 0.05 | **0.83** | 0.00 | 0.09 | 0.01 | 0.06 | 0.12 | 0.58 / 0.04 / 0.85 / 0.06 |
| 2 | `irq` | 0.093 | 1.06 | 0.13 | 0.05 | **0.42** | 0.05 | 0.00 | 0.16 | 0.07 | 0.09 | 0.07 | 0.15 / 0.39 / 0.04 / 0.09 |
| 3 | `wo_b` | 0.090 | 0.93 | 0.28 | 0.03 | 0.07 | 0.00 | 0.00 | 0.07 | 0.01 | **0.46** | 0.10 | 0.28 / 0.07 / 0.00 / 0.41 |
| 4 | `wo_a` | 0.043 | 1.16 | 0.20 | 0.09 | 0.07 | 0.00 | 0.01 | 0.12 | 0.02 | **0.42** | 0.08 | 0.21 / 0.08 / 0.00 / 0.39 |
| 5 | `router` | 0.019 | 0.63 | **0.68** | 0.01 | 0.03 | **0.40** | 0.00 | 0.05 | 0.00 | 0.13 | 0.08 | 0.69 / 0.03 / 0.51 / 0.12 |
| 6 | `attn_score` | 0.018 | 0.89 | 0.37 | 0.02 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | **0.46** | 0.12 | 0.21 / 0.00 / 0.00 / 0.63 |
| 7 | `norm` | 0.015 | n/a | 0.16 | 0.00 | 0.01 | 0.00 | 0.00 | 0.09 | 0.00 | **0.61** | 0.17 | 0.10 / 0.19 / 0.00 / 0.19 |
| 8 | `gate` | 0.001 | 0.91 | 0.12 | 0.01 | **0.48** | 0.18 | 0.00 | 0.04 | 0.00 | 0.05 | **0.53** | 0.13 / 0.45 / 0.00 / 0.07 |

(Rest of K2, ATOM's code in both builds: `ug` 0.335, `down` 0.194, `shared` 0.031, entry 0.023, `route` / `reduce` /
`xq` under 0.005 each. Their `lat/A` of 0.97-1.03 (`ug` 1.03, `down` 1.01, entry 0.98) is the noise floor of this
column; `route` 1.24 and `shared` 0.49 show what CTA placement does to small stages. Our `norm` gets n/a: ATOM's norm
tasks barely land on the traced CUs.)

- Our code has no idle-wave or LDS problem except `gate` (0.53 idle: wave 0 alone runs the Sinkhorn while the other 7
  waves wait at the barrier). MFMA busy is near zero everywhere: these are latency / stream stages.
- Ranked bubbles of OUR code at K2 6 rows:
  1. `slice`'s poll (0.83 of its latency, the same as ATOM's 0.85): `stage_reduce` waits for both ranks' `wo_b` partials.
     This is the TP hand-over on the critical path, not a code inefficiency; the lever is upstream (`wo_b` finishing
     earlier, or the decision-10 schedule surface).
  2. `irq`'s barriers (0.42; ATOM 0.39): 2 CTA barriers for the split stats, then 2 a column group around the LDS
     combine.
  3. The weight / key streams of `wo_b`, `wo_a`, `attn_score` (VMEM issue + `vmcnt` 0.6-0.8): at the byte floor this
     is bandwidth. `attn_score` waits on `vmcnt` 0.37 vs ATOM's 0.21 while issuing less (0.46 vs 0.63): our split DMA
     keeps one split in flight; a deeper DMA is a knob to build (§6).
  4. `router`'s x poll (0.40, ATOM 0.51): the wait for the FFN seam's `x`.
  5. `norm`'s loads (0.61 VMEM: the bf16 weight loads again; `wvec=1` applies here too).
  6. `gate`'s barrier and idle waves.
- **Source lines** (the `rk_dbg` functions' DWARF through the ATT `code.json`; share of all traced latency of our
  source lines, the 11 dispatches pooled):

  | share | source line (stage line -> inner helper) | what it is | stall / latency |
  |---|---|---|---|
  | 0.421 | `stage_slice.py:191 reduce_one -> seam_common.py:221 poll_aux` | K2 slice's TP reduce: polls both ranks' `wo_b` partials (system scope) | 0.83 |
  | 0.127 | `stage_attn.py:302 -> barrier` | `irq`: the barrier after wave 0's split-stats poll | 1.00 |
  | 0.054 | `stage_norm.py:120 -> poll` | `norm` waits for slice's `lin` pairs | 0.82 |
  | 0.029 | `stage_router.py:102 await_x -> poll` | `router` waits for `x` | 0.83 |
  | 0.028 | `stage_router.py:131` | the router MFMA chain (waits on its weight / x operands) | 1.00 |
  | 0.025 | `stage_attn.py:192 rows_to_lds` | `attn_score` key-row DMA into LDS | 0.97 |
  | 0.024 | `stage_attn.py:310` | `irq`'s 12-pair tagged poll (the stats + the 8 splits' p) | 0.81 |
  | 0.019 | `stage_attn.py:234` | `attn_score`'s `vmcnt(0)` before reading a split's keys from LDS (the DMA wait) | 1.00 |
  | 0.019 | `stage_attn.py:301 -> poll` | `irq` wave 0's split-stats poll | 0.93 |
  | 0.018 | `stage_attn.py:147 key_row` | the window-ring row index of a key (stalls on an operand load) | 0.97 |

  The three poll lines carry 2.8 M, 0.37 M and 0.18 M instruction hits: the polls spin. That is where the extra
  dynamic SALU of §4.3 comes from.
- Static (dbgsplit, K2 s6): our `wo_a` / `wo_b` / `router` / `norm` are 0.70-0.93x ATOM's instruction count, `irq`
  0.99x with 14 more `s_waitcnt` (92 vs 78) and 16 fewer `sc1` loads, `attn_score` 1.02x. With all stages linked the
  K2 s6 build has +5 % SGPR spills (830 vs 790; 805 vs 824 at s12, 1,175 vs 1,209 at s48), the draft K2 s5 far fewer
  (80 vs 220).

### 4.2 K1 at 6 rows (engine ATT, n01-09, a4e: CU 1, 4 dispatches a arm; non-indexer `f1 ix0` build)

| stage | share | lat/A | vmcnt | lgkmcnt | barrier | VALU | VMEM issue | note |
|---|---|---|---|---|---|---|---|---|
| `wq_b` (rocKE) | 0.574 | 0.90 | 0.44 | 0.03 | 0.12 | 0.10 | 0.19 | poll 0.41: waits for its q-latent input |
| `wqkv_a` (rocKE) | 0.256 | (1.44, CTA placement) | 0.31 | 0.04 | 0.05 | 0.10 | 0.40 | weight stream |
| `slice` (rocKE) | 0.125 | 0.87 | 0.18 | 0.16 | 0.10 | 0.19 | 0.25 | |
| `norm` (rocKE) | 0.006 | 0.56 | 0.11 | 0.02 | 0.10 | 0.24 | 0.30 | |

K1 rig ATT source lines (a3, 16 CUs, the seam stages built `rk_dbg`), ranked by latency:
`stage_slice.py:134 row_loads -> seam_common.ld_bf` (the residual rows, stall 0.93 of its latency),
`stage_slice.py:120 fn_loads -> ld_f32` (the mix weights), `stage_slice.py:269-274` (the MFMA mix and its
barrier), `stage_norm.py:120 -> stage_wqkv.poll` (norm waits for slice's `lin` pairs), `stage_norm.py:109 weights ->
ld_bf` (0.97 stall).

- **The `norm` loss, diagnosed and restructured.** ATOM loads the norm weights as 8 bf16 loads a chunk (the dword each
  half sits in). Our parity form does the same, and the ATT line `stage_norm.py:109 weights` shows them as a stall
  behind the poll. Restructure (a different load structure, not a schedule tweak): `wvec=1`, one `dwordx4` a chunk
  (bitwise: the same bytes). Rig: the seam goes from 1.008-1.012 to 0.997-1.004 at 3-6 rows on n07-17 (§3.1). Kept as a
  knob, default 0 (parity); the climb takes it.
- The K1 `slice` residual loads (`row_loads`, bf16 a half dword) are the next target: the same `dwordx4` restructure
  applies (knob to build, §6).

### 4.3 PMC (rocprofv3 `--pmc`, eager DSpark C1, every K1 / K2 s6 dispatch, median per dispatch, ours / ATOM; a6d)

| kernel | MFMA | VALU | SALU | LDS | LDS bank conflicts | VMEM rd | VMEM wr |
|---|---|---|---|---|---|---|---|
| K2 s6 (indexer layer) | 0.999 | 0.957 | **1.165** | 1.009 | 1.002 | 0.971 | 0.968 |
| K1 s6 f1 ix0 | 1.000 | 1.004 | 1.021 | 1.005 | 1.006 | 0.979 | 0.834 |
| K1 s6 f1 ix1 | 1.000 | 0.935 | 0.974 | 1.004 | 1.006 | 0.994 | 1.000 |
| K1 s6 f0 ix0 / ix1 | 1.000 / 1.000 | 1.011 / 0.917 | 1.051 / 0.956 | 1.005 / 1.004 | 1.006 / 1.006 | 1.005 / 0.965 | 0.759 / 1.000 |

- Same MFMA work everywhere (the ports are bitwise); fewer VMEM instructions (wider loads / stores); LDS bank
  conflicts unchanged.
- K2 runs **16.5 % more SALU instructions**, dynamically. It is not a calling-convention cost: the linked functions
  are inlined (no `s_swappc` in the K2 s6 code object) and the static SALU count of our K2 is 0.96x ATOM's. So the
  extra SALU is executed in loops: mailbox-poll iterations (a poll iteration is SALU: compare, branch, `s_sleep`). The
  ATT source lines agree (§4.1): our three poll lines carry 2.8 M, 0.37 M and 0.18 M instruction hits. The spin
  count follows how long the peer rank / the upstream stage takes (and under `--pmc` the dispatches serialize), so
  it measures waiting, not work on the critical path.
- `FETCH_SIZE` / `WRITE_SIZE` hang the profiled engine (as for K2-moe), and the wave-cycle counters are dominated by
  the serialized TP waits under `--pmc`; achieved bandwidth therefore comes from the stage times (§3.3) against the
  stage bytes, not from counters.

## 5. COMGR vs Rust direct

The stage functions are LLVM device functions linked into ATOM's FlyDSL K1 / K2. The direct compiler emits a whole
HSACO with one kernel and cannot produce a linkable device function, so **in situ every stage is built by FlyDSL's
LLVM (auto-select: COMGR/LLVM for all, by construction)**. To still compare the compilers on our code, each stage's
`KernelDef` is compiled standalone both ways (`h/direct_census2.py`, cluster container COMGR, crate `so/a2`), default
levers and 7 lever sets (`dppcombine`, `ldspairs`, `storewait`, `vmemfirst`, `nofence`, `minmax3`,
`dppcombine+ldspairs+storewait`). Direct / COMGR:

| stage form | insts | `s_waitcnt` | `vmcnt(0)` | VGPR | SGPR |
|---|---|---|---|---|---|
| gate n5-n48 | 1.16 | 0.84 | 2.0 | 1.84 | 1.12 |
| norm k1 n6 | 1.31 | 2.47 | 6.0 | 1.20 | 1.23 |
| norm k2 n6-n48 | 1.43 | 2.47 | 6.0 | 1.05 | 1.25 |
| slice k1 n6 | 1.25 | 1.84 | 4.0 | 1.39 | 1.46 |
| slice k2 n6 / n24 / n48 | 1.21 / 1.33 / 1.36 | 1.36 / 1.60 / 1.50 | 3.3 / 4.7 / 3.0 | 1.20 / 1.27 / 1.94 | 1.14 / 0.96 / 0.96 |
| router n6 / n24 / n48 e384, n5 e128 | 1.03 / 1.01 / 1.03 / 1.04 | 1.33 / 1.17 / 1.11 / 1.35 | 3.0 / 2.3 / 2.4 / 3.0 | 1.25 / 1.65 / 1.21 / 1.47 | 1.14 / 1.07 / 1.07 / 1.14 |
| attn p8 n6 / p4 n24 / p2 n48 | 1.15 / 1.18 / 1.19 | 0.88 / 0.90 / 0.97 | 0.5 / 0.6 / 0.7 | 1.64 / 1.09 / 0.96 | 0.79 / 1.38 / 1.50 |
| irq g1 n6 / g4 n24 / g8 n48 | 1.16 / 1.19 / 1.19 | 0.79 / 0.99 / 0.95 | 2.0 / 3.1 / 5.1 | 1.23 / 1.19 / 1.19 | 1.44 / 1.21 / 1.17 |
| wqkv n6 / n12 | 1.01 / 0.98 | 0.91 / 0.74 | 4.5 / 3.3 | 0.97 / 1.05 | 1.00 / 1.00 |
| wqb n6 / n12 | 1.26 / 1.22 | 1.72 / 1.59 | 10 / 10 | 1.06 / 0.98 | 0.96 / 0.96 |
| woa n6 / n12 | 1.01 / 0.99 | 0.94 / 0.81 | 2.7 / 3.0 | 0.91 / 0.99 | 1.24 / 1.09 |
| wob n6 / n12 | 1.07 / 1.03 | 1.09 / 0.93 | 4.0 / 4.3 | 1.00 / 1.01 | 1.20 / 1.06 |

The best lever set changes instruction counts by at most 3 % (irq) and under 1 % on every other form. Defects found in the gate n6 ISA diff
(mnemonic histograms, direct vs COMGR), each a compiler-core item (decision 18):
1. **DPP not folded into the consumer.** COMGR emits 40 `v_add_f32_dpp` (the row-shift reduction folded into the add);
   direct emits `v_mov_b32_dpp` + `v_add_f32` pairs (122 vs 82 DPP movs, 167 vs 128 `v_add_f32`). The `dppcombine`
   lever leaves this form unchanged (same instruction count); why it does not match is not isolated yet.
2. **No `ds_read2` pairing.** 40 `ds_read_b32` vs COMGR's 20 `ds_read2_b32` for adjacent LDS words (`ldspairs` leaves
   the form unchanged).
3. **Unfolded address arithmetic.** 28 `v_add_lshl_u32` + 8 `v_mul_lo_u32` + 11 `v_add3_u32` where COMGR folds the
   constants into DS / buffer offsets.
4. **Waits.** `vmcnt(0)` 2-10x COMGR's on every form but `attn` (0.5-0.7x there): direct puts a full
   `vmcnt(0) lgkmcnt(0)` drain on both sides of every `s_barrier` (norm k2: COMGR has one `vmcnt(0)` in the whole
   form, direct six, four of them around its two barriers) and splits counted waits into a full drain inside load
   sequences.
5. **VGPRs** up to 1.9x on the small seam forms (cause not isolated yet).
Timing direct vs COMGR in situ is not possible until the direct compiler can emit a linkable function (or a whole-K2
rocKE build exists, PLAYBOOK idea 12). Recorded: "direct: rejected on base parity-port standalone (static), in situ
not applicable"; the defects go to the crate's ledger (rust-track).

## 6. First knob surface (every hard-coded choice of the port; default = parity, byte-identical)

Already knobs (all built, all bitwise in the K1 rig where they apply; numerics class B = bitwise by construction):

| knob | stages | values (default first) | class | first measurement |
|---|---|---|---|---|
| `poll` | all seam stages | ir, asm | B | asm: K1 1.029-1.041 LOSS (a3), kept as a knob |
| `pause` | all | 1 (`s_sleep 1`), 0 (`s_nop 0`), -1, N | B | p0: tie (0.992-1.007) |
| `poll_max` | all | 12, 1-16 | B | - |
| `shfl` | all | dpp, ror, swz | B | K1 rig 1-11 rows: ror tie (0.993-1.007), swz 0.994-1.009 |
| `red_bar` | all | wg, lds | B | lds: tie (0.993-1.008) |
| `early` | gate | 0, 1 | B | e1: tie (0.995-1.008) |
| `wvec` | norm | 0, 1 | B | **1: fixes the K1 seam loss** (§4.2); `allwv` K1 WIN on both nodes (§3.1b) |
| `wearly` | norm | 1, 0 | B | 0: 1.036-1.057 LOSS (a must-lose order control) |
| `batch` | slice | 4, 1-8 | B | 2 / 8: 0.996-1.009 / 0.995-1.013 (tie / slight loss at 3-6 rows) |
| `fn_early_max` | slice k2 | 24, 0-48 | B | - |
| `early_from` | router | 12, 1-48 | B | - |
| `w_cpol` / `x_cpol` | router | 2 (nt) / 16 (sc1), any cpol | B | - |
| `askip` | attn | 0, 1 | B | (W13 lever: 0.992-0.995 in FlyDSL) |
| `stat_w0` | irq | 1, 0 | B | - |
| `sop` / `ra` | GEMV family | 0 / 0, sop 1-2, ra 1 | B | gs: bitwise; WP4 tie |
| `x_first`, `w_cpol`, `poll*`, `pause`, `red_bar` | GEMV family | WqkvKnobs | B | P2 / WP4 |

Hard-coded today, to expose (default = parity):

| choice | where | default | candidate values | class | why it may matter |
|---|---|---|---|---|---|
| residual row load width | slice `row_loads` | bf16 half-dword loads | `dwordx4` | B | top ATT line in K1 (§4.2) |
| mix-weight load width / timing | slice `fn_loads` | 6 f32 loads a thread, first | `dwordx4`, after the poll | B | second ATT line |
| irq barrier structure | irq combine | 2 CTA barriers a column group (+ 2 for the stats) | named barriers per wave pair, LDS flags instead of barriers | B | 0.42 of irq latency (§4.1) |
| irq split-tile staging | irq | through LDS, 8 splits | direct register combine for groups=1 | B | |
| Sinkhorn on wave 0 only | gate | wave 0, 7 waves idle | overlap with other stages' work (schedule surface) | B | 0.53 idle in gate, but gate is 0.001 of K2 |
| score DMA depth | attn | 1 split in flight | 2-3 | B | `vmcnt` 0.37 vs ATOM's 0.21 (§4.1) |
| score LDS row stride | attn | 260 words | 256 + 8, 272 | B | bank conflicts (PMC to check) |
| `exp2` form | attn | `exp2_fast` | precise | T | - |
| PV key-row loads | irq | 64-bit `global_load_dwordx4` | buffer loads with soffset | B | VGPR / address cost |
| router K steps per wave | router | 5 steps of 32, waves w, w+8 | 4 waves x 10 | B | - |
| router token groups | router | 1 group <= 16 rows, 2 above | 1 / 3 | B | - |
| norm token mapping | norm | 256 threads, 3 chunks | 512 threads, 2 tokens a CTA | B | |
| slice MFMA mix | slice | `16x16x4 f32`, 8-wave sum in wave order | `32x32x2 f32` | T (sum order) | |
| poll wave | all | every wave polls (or wave 0 for irq stats) | one poll wave + LDS broadcast | B | Qwen idea 9 |
| store cache bits | seams | `sc1` puts | `nt`, `sc0 sc1` | B | |
| code-object backend | all | FlyDSL LLVM (only option in situ) | direct when linkable | B | §5 |

Interaction groups to screen together: (`wvec`, slice load widths, `batch`) on the K1 seam; (`shfl`, `red_bar`,
`pause`) on every seam; (`askip`, score DMA depth, `stat_w0`, irq barrier structure) on attention; (`early_from`,
`x_cpol`, `w_cpol`) on router; (`sop`, `ra`, `x_first`) on the GEMV family; and every port knob with the decision-10
schedule surface once it exists.

## 7. The rig (reusable)

`hackathon-prep/dsv41/w15/rig/` (new files only):
- `a4_submit.sh <tag> <node> <phases> [eng phases] [VAR=value ..]`: one call builds the whole round's environment
  (stage function dirs `rk`, `rk_dbg`, `wq`, `wqs`, one dir per mutant, extra dirs via `FN_EXTRA`; gate arms; bench
  variants plus `BV_EXTRA`; gtl / HTL arms; ATT / PMC specs). `SUBMIT=n07/submit_local.sh` runs it node-locally,
  `HDIR` picks the harness copy.
- `submit.sh` / `n07/submit_local.sh`, `w15_job.sh` / `n07/w15_job_local.sh`, `n07/push.sh` / `n07/pull.sh`
  (node-local stages, harness, `.so`, results for a node without BeeGFS), `n07/release_w15.sh` (releases held jobs
  while ALL W15 jobs stay within the 8-GPU cap).
- `h/w15_run.sh` phases: `gen` (stage functions, `gen_fn.sh`), `ident`, `census` (COMGR vs direct, `direct_census2.py`),
  `k1rig` (`k1_rig.py`, seconds a candidate), `dbgsplit`, `att` (K1 rig ATT), `att2` (engine ATT, one run per CU in
  `W15_ATT2_CUS`), `pmc` (rocprofv3 `--pmc` groups, `pmc_cmp.py`), `eng` (`wp4_run.sh`: `gate`, `bench`, `gtl`,
  `htl`, ...). `h/ledger_cmp.py` prints the ours-vs-ATOM stage ledger from two `w15_att.py ledger` files.
- Screening cost: the K1 rig runs 7 arms x 3 seeds + paired graphs at one width in about 20 s (6 widths in about
  2 minutes after a 2-minute function generation); a K2 candidate is a bench variant (a 6-variant bench over C1-C8
  takes about 30 minutes), a gtl arm (about 12 minutes) or an HTL arm-run (about 10 minutes, engine start included).

## 8. Next steps (ranked)

1. Second-node confirmation of the graph-mode step win as soon as a second MI355X node can run the container engine
   (n07-17 reseeded, or n01-09 back): HTL `b1` / `rk` / `rkwv` / `b1b`, 2 passes, plus the K2 bench and gtl. One call:
   `HDIR=<harness dir> bash a4_submit.sh <tag> <node> gen:eng htl:bench:gtl "WP4_HTL_ARMS=..." WP4_HTL_PASSES=2`
   (node-local: `SUBMIT=n07/submit_local.sh`, then `n07/pull.sh` right after the job). Until then it is a one-node
   result.
2. Carry `wvec=1` into every arm (`allwv` is the K1 build), screen the K2 `norm` with it, and build the slice
   load-width knobs (§6, top K1 ATT lines); re-screen K1 in the rig.
3. Attack `irq`'s barriers (the biggest bubble of our K2 code that is not a TP wait): an LDS-flag combine instead of
   the 2 CTA barriers a column group, gated bitwise, ATT before / after.
4. `attn_score`'s split DMA depth (one split in flight today; our `vmcnt` share 0.37 vs ATOM's 0.21). `gate`'s idle
   waves (0.53) are structural (one token's 4x4 Sinkhorn on 16 lanes) and `gate` is 0.001 of K2: the fix is the
   schedule surface (give the CTA's other waves work), not this stage.
5. The K2 slice's TP reduce poll (0.42 of our traced K2 latency) is the hand-over from `wo_b`: it moves only with
   `wo_b` finishing earlier on both ranks or with the decision-10 schedule surface (holds, CTA placement) - put the
   port's knobs and that surface into the qwen38_mono/tune V4.1 backend with signed lookback, per layer kind x width,
   scored on the graph-mode K2 critical path.
6. Rust direct: fix defects 1-4 of §5 in the crate (generic: every seam form shows them), and design a
   linkable-function output (or the whole-K2 rocKE build, PLAYBOOK idea 12) so direct can be timed in situ.
