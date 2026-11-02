# WP-L2e: the large-batch QSA layer pieces with ATOM's semantics (E9), rows 5-128, TP1 / TP2, contexts 128 / 8K / 32K

Work package WP-L2e of `../DESIGN-LB.md` 15.6 (wave LB-2), workflow W10, 2026-11-02. alola `rocke` partition
(MI355X, 256 CUs) for every verdict, both rocke nodes; Markham MI350X for WaveScope only. Ratios only in this file;
absolute times are in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2e.md`. Below 1 is faster.

Status: **done; gate met on both nodes for the product (tolerance-class) arm.** Numerics pass on MI355X on both nodes
with every must-fail mutant caught. The bitwise-ATOM-scores arm (`score=f64`) is built and verified, but it loses to
the FlyTape-structure port at 8K / 32K. It is kept as a knob, not as the default. The product arm's indexer scores
are tolerance class (bf16 MFMA in most long-context cells, the f32 VALU chain `score=valu` at TP1 5x8K, TP1 8x8K and
TP2 8x8K; fix round V2): the RULER gate is owed for both score forms before it serves (open item 1).

The independent verification ran (verdict partial, section "Verification"); the fix round answers its findings V1-V9
(section "Fix round").

## 0. Summary

1. **What was built** (`rocke/library/kernels/gfx950/qwen38_mono/large_batch/qsa_lb2.py`, Python engine,
   `q38lb_qsa2_*`; nothing served imports it):
   - The QSA core of P-L5 (`qsa_lb.py`, which stays byte-identical) forked and extended with ATOM's model semantics
     (WP2c's Q3-Q7 plus the Q2 pieces; details in section 1).
   - With every new knob off, a build lowers to the same LLVM IR as the matching P-L5 build (kernel name aside; CPU
     test). P-L5's numerics and verdicts therefore carry over to the default arm.
   - The two E1 GEMMs around the core: `qsa_proj` (E1 `qsa_in`) and `qsa_out`, from WP-L2a's plan table.
     - The core reads `qsa_proj`'s output in E1's Y layout (`pio=e1`).
     - It writes its output as MXFP8 in the activation layout of `qsa_out` (`oq=mx`), so `qsa_out` reads it with no
       repack.
   - Host references: `qsa_ref.py` (FlyTape semantics, P-L5) and `qsa_ref2.py` (ATOM semantics, f64 golden).
   - CPU tests: `library/tests/test_qwen38_lb_qsa2.py` (57 pass).
   - Harness: `hackathon-prep/qwen38/truemono-lb/wp-l2e/` (`l2e_bench.py`, `l2e_climb.py`, `l2e_report.py`,
     `l2e_summary.py`, `l2e_tables.py`, `isa_cmp.py`, `isa_loops.py`, `att_l2e.py`, `prebuild.sh`, `job.sh`,
     `sub.sh`).
2. **Numerics gate met on both nodes** (section 2).
   - 8 cases (TP1 / TP2; rows 4-40; dense, budget-boundary, 8K, 32K and planted-tie positions; 2 layers) x about 60
     arms, plus the composed chain E1 `qsa_in` -> core -> E1 `qsa_out`.
   - Every ATOM-exact stage is bitwise: q / k / v, the raw index key, the block key, the indexer query, and the f64
     scores. The 31-bit selection is exact from the device's own scores. The output is within tolerance of the f64
     golden.
   - All 23 must-fail mutants are caught on n01-09 (22 on n07-17, where `hneed_qs` did not exist yet). The stale-read
     twin is caught in 7 of 8 cases.
   - Every timing job also checked each timed code object (stream and phase twin) on random data at the cell's exact
     shape.
3. **The product arm beats FlyTape in situ in every bracketed cell** (random data; zero data with `heqs`) (phase metric, same job, FlyTape traced before and
   after; section 4). The product arm is `tpick`: ATOM semantics, MFMA indexer scores (`score=valu` at TP1 5x8K, TP1
   8x8K and TP2 8x8K; at p128 nothing is scored), both tolerance class (as DESIGN-LB E9
   specifies), paged KV, MXFP8 output.
   - **Core / FlyTape's `qsa_core`, random data:**

     | TP | p128 | 8K | 32K |
     |---|---|---|---|
     | 2 | 0.77-0.85 | 0.73-0.93 | 0.78-0.92 |
     | 1 | 0.75-0.80 | 0.78-0.89 | 0.81-0.90 |

   - **QSA family** (E1 `qsa_in` + core + E1 `qsa_out`) / FlyTape's `qsa_proj + qsa_core + qsa_out`, plus
     `qsa_out_reduce` at TP1 (TP2 has no such phase; fix round V6):

     | TP | p128 | 8K | 32K |
     |---|---|---|---|
     | 2 | 0.57-0.62 | 0.66-0.77 | 0.72-0.80 |
     | 1 | 0.49-0.55 | 0.64-0.67 | 0.67-0.71 |

   - FlyTape's bench uses zero state. On zero data (like for like with it), the product arm with `heqs` reaches 0.70-0.91
     of FlyTape's core at 8K / 32K on both TPs. Without `heqs`, the 32K zero cells at rows 8-32 sit at 0.91-1.13.
   - **TP2 B128 8K**, the 10k target's batch: core 0.73-0.74, QSA family 0.66 of FlyTape's.
4. **ATOM's semantics cost 1.06-1.31x over P-L5's FlyTape-semantics pick at 8K / 32K** (random data), and 0.94-1.09
   at p128. The causes:
   - `index_cmp` is bf16 (2x FlyTape's E4M3 key bytes in the scan);
   - the full 31-bit top-512 select (FlyTape's 16-bit keys are coarser);
   - ATOM's norm and RoPE forms.

   The exact-score arm (`xpick`, f64 sums, ATOM's bits) pays more: 1.08-1.57x the P-L5 pick, and 0.94-1.13 of the port
   at 8K / 32K (random data).
5. **Three climb rounds** (section 3) of hill climbing with lookback (drops plus every pair of moves, losers included):
   - 6 jobs; 36 cells x 2 families (x = exact scores, t = any score form) x 1-2 starts;
   - 646 climb steps; 9,068 measured candidate moves on the climb steps (each step's ledger record also holds one
     control entry: 9,714 records). The end-of-climb lookback measured 21,746 drops / pairs more, plus 258 final and
     149 confirm measurements (fix round V8).
   - Round 3 re-tested every move, earlier rejects included, on the round-2 picks with the new levers.
   - It accepted `s64dpp` in 22 of the 24 long-context exact picks and `sst4` in 10 tolerance picks.
   - Re-checked on both nodes in the timing jobs: round-3 x pick / round-2 x pick 0.84-1.00 (WIN in 116 of 132
     cell-jobs, never a loss); round-3 t pick / round-2 t pick 0.95-1.02 (WIN 34, TIE 33, LOSS 5 of 72).
6. **New levers this round, all default-off knobs:**
   - `s64dpp`: the scan's f64 butterflies on DPP moves.
   - `sst4`: 16-byte score stores.
   - `heq`: an all-equal test of the tied keys before the restricted 15-bit search.
   - `heqs`: `heq` without its pass when the tied set is small.
   - Both nodes give the same `heqs` verdict, but it splits by data:
     - zero / tied data: WIN in every cell-job, 0.79-0.97;
     - random data: TIE in 40 of 42 cell-jobs of the tolerance arm (two losses of 0.6-0.7 % at rows 5), and a 0.7-3 %
       loss in 14 of 44 cell-jobs of the exact arm.
   - So `heqs` is retained, and recommended for the tolerance arm wherever states can tie (fix round: on random data
     8 losses of 80 cell-jobs, rows 5-8, at most 1.7 %) (fresh / zero caches, the
     FlyTape-comparable bench state). `heq` without `s` loses 1-2 % on random data everywhere.
7. **DESIGN 16 IR diff closed** (section 6). The port is faster than FlyTape in situ at 8K / 32K because of FlyTape's
   tape-level SGPR budget and cache policy, not because of a different algorithm. Static loop-level comparison of
   FlyTape's own `qsa_core` (dumped from its tape at B16 x 8K / 32K) against the port:
   - FlyTape's scan loop body is 1.31-1.35x the port's instructions. It carries 16-44 SGPR-spill `v_readlane` /
     `v_writelane` per iteration (the whole tape: 1,593 SGPR spills); the port's has none.
   - Its 32 key loads are `sc1`; the port's are plain.
   - Its attention stage is three ~260-instruction copies with readlanes around them.
   - At p128 (no scan, no select) the two are at parity, as P-L5 measured.
8. **WaveScope before / after** (section 5): `s64dpp` took the exact pick's traced latency at TP2 rows 128 32K to 0.88
   of before. The tolerance pick moved within noise (0.96-1.01). What remains:
   - exact arm: VALU (f64 FMA) issue;
   - tolerance arm: the scan's key loads and the attention K / V waits;
   - both: the per-layer barrier.

## 1. What `qsa_lb2` adds to P-L5's port

Every addition is a default-off knob (the kernel name lists only deviations). The full list is in the module
docstring.

| knob | what | source of the semantics |
|---|---|---|
| `kvl=page16` | ATOM's paged K / V (`kv_cache_block_size` 16), per-layer tables, block table in LDS (`pgt=lds`) or global (`pgt=glb`) | ATOM's cache layout |
| `sem=atom` | Q2 + Q3-Q7 as ATOM computes them. Q2: direct-scale RMSNorm of q / k with the f64 inverse, ATOM's bf16 RoPE tables, v raw, k / v / raw index key appended to the paged caches. Indexer query / block keys: zero-centred norm, ATOM's RoPE, block key pooled from 4 raw rows, stored bf16 in `index_cmp`. Scores: relu of 4 heads' dots / fl(sqrt(128)). Selection: full 31-bit keys, ties to the lower block. Output: bf16(bf16(o) * sigmoid) | `wave2/WP2c.md` (Q3-Q7), ATOM `qsa.py` / `qsa_fast.py` |
| `score=f64` / `mfma` / `valu` | exact f64 sums (ATOM's bits) / bf16 MFMA with f32 accumulation / f32 FMA chain (both tolerance class, DESIGN N10) | |
| `oq=mx` / `both` | core output as MXFP8 in E1's activation layout (`rocke.helpers.mx_quant`'s rule = ATOM's Q7 pack) | |
| `pio=e1` | the projection read from E1's Y layout | WP-L2a |
| `hsel` | hierarchical 31-bit select: FlyTape's 16-bit search on `key >> 15`, then the tied keys' low 15 bits (one wave with ballots up to 256 tied, else the restricted search) | |
| `fma64`, `s64dpp` | f64 scoring: query in f64 with one `v_fma_f64` per (dim, head); 16-lane f64 butterflies on DPP | |
| `sst4` | MFMA scorer: one 16-byte store per lane | |
| `heq`, `heqs` | `hsel`: min / max test of the tied low bits before the restricted search; skipped when the tied set is small | |
| mutants | `page`, `qnorm`, `ikey`, `gate`, `oqs`, `sel16`, `cmp`, `hneed` (new), plus P-L5's | |

The attention arithmetic stays MFMA (bf16 P, f32 accumulation, online softmax): tolerance class against ATOM's
sequential f32 chains (DESIGN-LB E9).

## 2. Numerics (MI355X, both nodes)

Jobs:
- v1 (n01-09) and v2 (n07-17): the round-1 code;
- v5 / v6 (n07-17, two GPUs): the current code incl. `s64dpp` / `sst4` / `heq`;
- v7 (n01-09): incl. `heqs`, plus the chain.

Each case runs from the device's own inputs, stage by stage, against `qsa_ref` (FlyTape semantics, row and paged
layouts) and `qsa_ref2` (ATOM semantics, f64 golden).

| check | result |
|---|---|
| ATOM-exact stages (q, k, v, raw index key, block key, indexer query) | bitwise in every arm, every case |
| scores | bitwise (`score=f64`), within 1e-4 (`mfma` / `valu`) |
| selection | exact from the device scores (31-bit rule) in every arm |
| core output | within 2^-6 of the f64 golden (max error 6.0e-3 to 8.9e-3) |
| MXFP8 pack | bitwise against the host quantizer of the device's bf16 output |
| order-keeping groups | bitwise equal: `atx` (12 arms incl. `fma64`, `s64dpp`, `hsel`, `heq`, `heqs`, `v1` / `v2`, `its4`), `atmf` (6 arms incl. `sst4`, `heq`, `heqs`), `at`, `pg`, `ft` |
| determinism | 2 launches bitwise equal |
| must-fail mutants (23) | all caught on both nodes: `rope`, `scale`, `merge`, `tie`, `tie_h`, `window`, `kvpos`, `iqrope`, `scan`, `scan_mf`, `page`, `page_ft`, `qnorm`, `ikey`, `gate`, `oqs`, `oqs_ft`, `sel16`, `cmp`, `hneed`, `hneed_q`, `hneed_qs`, `stale_plain`. Each is caught in 5-8 of 8 cases; the `tie` / `hneed` families only bite in tie / sparse cases |
| stale-read twins | `acq=plain` + stress: caught in 7 of 8 cases (missed in the dense 2K case); the `sc1` twin passes and is bitwise equal to its group |
| chain E1 `qsa_in` -> core -> E1 `qsa_out` | PASS at TP1 8x8K, TP1 64x128, TP1 128x32K, TP2 16x32K, TP2 128x8K (v7, `--chain-cells` set to 5 cells; the default list runs 4). Fix round: also on the timed picks at all 36 cells (V4). `qsa_in` error about 3e-5, core as above, `qsa_out` read straight from the core's MXFP8 (no repack) about 2e-5 |
| per-timed-object check (every timing job) | every `xpick` / `tpick` / `*_qs` / `*_r2` / `ft` / P-L5 pick code object, stream AND phase twin, on random data at the cell's shape: PASS, twins equal, in all 6 timing jobs |
| compile gate | scratch 0, no asm hazards (`hz` pad) on every timed object. Refused (scratch 8-80 B, venv COMGR): 92 (TP1) / 48 (TP2) derived `waves=8` paged arms per timing job, nearly all with ATOM semantics (TP1 85 + 7 `sem=ft`;
TP2 46 + 2): TP1 at every prompt, TP2 at p128 rows 8-128, 8K rows 64 / 128 and 32K rows 128 (scratch 8-68 B at TP2;
fix round V7); none of them is a climb pick (the climbs only time gate-clean builds) |

## 3. Climbs (decisions 13 / 15)

`l2e_climb.py`:
- **Families:** x (exact f64 scores kept) and t (the score form is a knob too).
- **Starts:** P-L5's per-cell pick moved to ATOM semantics, FlyTape's structure, and in later rounds the previous
  round's pick.
- **Each step** re-tests every move on the current base, earlier rejects included. A move is accepted only under the
  win rule against the base's own controls (a second graph and a recompiled twin).
- **Lookback on the end:** every lever dropped, every pair of moves (losers included), then the 4 best re-measured with
  twice the rounds.
- **Product knobs fixed:** `kvl=page16, sem=atom, oq=mx, pio=e1`.
- **Data:** random; 8-16 rounds; climbs on one node, verdicts on both (section 4).

| round | base | jobs | steps / moves measured | lookback accepts | most-accepted moves |
|---|---|---|---|---|---|
| r1 | P-L5 pick (ATOM semantics) and FlyTape structure | v1 (TP2, n01-09), v2 (TP1, n07-17) | 399 / 5,314 | 26 | `mpad` 81, `sel=v1/v2` 57, `qloc` 28, `pf` 21, `waves=8` 19, `scan_its` 34, score form 14, `gang=4` 8 |
| r2 | r1 picks, + `hsel`, `fma64` | v3 (TP2), v4 (TP1) | 139 / 2,059 | 9 | `hsel` 34, `fma64` 17 |
| r3 | r2 picks, + `s64dpp`, `sst4`, `heq` | v5 (TP2), v6 (TP1) | 108 / 1,695 | 11 | `s64dpp` 22, `sst4` 3 (+ 5 via lookback pairs), `scan_its`, `sela`, `qloc`, `spf` |

- Round 3 picks over the atom-semantics FlyTape structure (climb metric, random data): x 0.69-0.88, t 0.60-0.87.
- `heq` was never accepted on random data. `heqs` came after round 3 (section 4.3).
- Picks: `wp-l2e/l2e_picks_final.json` (`xpick`, `tpick`, `tpick_qs`, `xpick_qs` per cell).

## 4. Timing (DESIGN 9.1)

Six jobs, both nodes for each TP:
- **t3:** round-3 picks.
- **t4:** round-3 picks plus `heqs`.
- **Container jobs (t3, t4):** FlyTape traced before and after in the same job.
- **Venv jobs (t5):** the same arm set without FlyTape, for paired verdicts on the other node.

| TP | n01-09 | n07-17 |
|---|---|---|
| 1 | t4 (container) | t3 (container), t5 (venv) |
| 2 | t3 (container), t5 (venv) | t4 (container, 3 GPUs to skip bus e5) |

Method:
- **Prebuilt objects only.** Every timed binary is a venv-COMGR code object, built in a GPU-less job and loaded by
  name with `L2E_HSACO_STRICT=1`. None is compiled by the container COMGR (alola rules 217 / 219).
- **Rounds and controls.** 12 shuffled rounds; arms and metrics shuffled every round; controls `ft2` (second graph)
  and `ft_rc` (recompiled).
- **Metrics.** `phase` is the primary metric (a striped grid sync per layer); `stream` and `launch` are also recorded.
- **Data.** Random data and zero data in every cell.
- **Verdict counts** are (WIN / TIE / LOSS) over rows 5-128 x jobs.
- **FlyTape coverage.** FlyTape has no B5 and no TP1 B128 (index limit), so those cells have no FlyTape column.

### 4.1 Against the FlyTape-structure port (`ft`) and P-L5's FlyTape-semantics pick, paired per round

| TP | prompt | data | P-L5 pick / ft | xpick / ft | tpick / ft | tpick / P-L5 pick | xpick / P-L5 pick |
|---|---|---|---|---|---|---|---|
| 1 | 128 | rand | 0.72-0.82 (15/0/0) | 0.75-0.84 (18/0/0) | 0.75-0.83 (18/0/0) | 1.00-1.05 (0/5/10) | 0.99-1.04 (1/5/9) |
| 1 | 128 | zero | 0.72-0.82 (15/0/0) | 0.74-0.84 (18/0/0) | 0.75-0.82 (18/0/0) | 1.00-1.05 (0/5/10) | 0.99-1.04 (0/4/11) |
| 1 | 8K | rand | 0.85-0.89 (15/0/0) | 0.94-1.01 (13/4/1) | 0.92-0.97 (18/0/0) | 1.06-1.13 (0/0/15) | 1.08-1.17 (0/0/15) |
| 1 | 8K | zero | 0.85-0.94 (15/0/0) | 0.91-1.00 (12/6/0) | 0.89-0.98 (18/0/0) | 0.95-1.13 (5/3/7) | 0.97-1.18 (3/2/10) |
| 1 | 32K | rand | 0.77-0.84 (15/0/0) | 0.99-1.05 (0/6/12) | 0.92-0.99 (16/2/0) | 1.14-1.22 (0/0/15) | 1.20-1.35 (0/0/15) |
| 1 | 32K | zero | 0.77-0.83 (15/0/0) | 1.09-1.22 (0/0/18) | 0.97-1.17 (3/0/15) | 1.24-1.41 (0/0/15) | 1.37-1.49 (0/0/15) |
| 2 | 128 | rand | 0.78-0.83 (15/0/0) | 0.76-0.88 (18/0/0) | 0.77-0.88 (18/0/0) | 0.94-1.09 (7/4/4) | 0.93-1.09 (6/6/3) |
| 2 | 128 | zero | 0.79-0.84 (15/0/0) | 0.76-0.89 (18/0/0) | 0.77-0.88 (18/0/0) | 0.95-1.07 (5/7/3) | 0.94-1.06 (7/5/3) |
| 2 | 8K | rand | 0.84-0.89 (15/0/0) | 0.97-1.06 (6/2/10) | 0.94-1.02 (11/4/3) | 1.09-1.19 (0/0/15) | 1.15-1.23 (0/0/15) |
| 2 | 8K | zero | 0.84-0.88 (15/0/0) | 1.00-1.11 (0/5/13) | 0.94-1.06 (6/6/6) | 1.09-1.24 (0/0/15) | 1.15-1.28 (0/0/15) |
| 2 | 32K | rand | 0.70-0.84 (15/0/0) | 0.98-1.13 (2/4/12) | 0.92-1.02 (14/1/3) | 1.12-1.31 (0/0/15) | 1.18-1.57 (0/0/15) |
| 2 | 32K | zero | 0.71-0.84 (15/0/0) | 1.12-1.30 (0/0/18) | 1.03-1.26 (0/0/18) | 1.28-1.54 (0/0/15) | 1.34-1.71 (0/0/15) |

The six `tpick` losses against the port on random data (TP2 8K and 32K, three jobs each) are all at rows 16 (1.02, both
nodes). The table covers the builder's six timing jobs. The verification job (n01-09) measured `tpick` / `ft` at TP2
8K zero rows 16 at 1.065, so over all jobs that cell's range is 0.94-1.07 (fix round V9; the fix-round jobs' figures
are in "Fix round"). The
FlyTape-semantics port is not a product arm (16-bit keys, E4M3 index keys, FlyTape's norm and RoPE); the product bar is
FlyTape in situ (4.2).

### 4.2 Against FlyTape in situ (same job, median of the before / after brackets; container jobs t3 + t4)

| TP | prompt | data | ft / FT core | xpick / FT core | tpick / FT core | family x / FT | family t / FT | E1 `qsa_in` / FT `qsa_proj` | E1 `qsa_out` / FT `qsa_out` (+ `qsa_out_reduce` at TP1) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | rand | 0.96-0.99 | 0.75-0.78 | 0.75-0.80 | 0.49-0.54 | 0.49-0.55 | 0.26-0.30 | 0.37-0.58 |
| 1 | 128 | zero | 0.96-1.00 | 0.74-0.78 | 0.75-0.79 | 0.49-0.54 | 0.49-0.55 | 0.25-0.30 | 0.37-0.58 |
| 1 | 8K | rand | 0.83-0.93 | 0.83-0.90 | 0.78-0.89 | 0.64-0.70 | 0.64-0.67 | 0.26-0.30 | 0.37-0.62 |
| 1 | 8K | zero | 0.87-0.94 | 0.81-0.95 | 0.78-0.90 | 0.66-0.70 | 0.64-0.68 | 0.24-0.30 | 0.37-0.61 |
| 1 | 32K | rand | 0.88-0.92 | 0.89-0.94 | 0.81-0.90 | 0.70-0.77 | 0.67-0.71 | 0.25-0.30 | 0.38-0.60 |
| 1 | 32K | zero | 0.87-0.92 | 0.99-1.10 | 0.91-1.05 | 0.78-0.84 | 0.73-0.80 | 0.25-0.30 | 0.37-0.61 |
| 2 | 128 | rand | 0.97-1.03 | 0.78-0.85 | 0.77-0.85 | 0.58-0.62 | 0.57-0.62 | 0.32-0.36 | 0.59-0.84 |
| 2 | 128 | zero | 0.95-1.04 | 0.78-0.85 | 0.78-0.85 | 0.57-0.62 | 0.57-0.62 | 0.31-0.37 | 0.59-0.84 |
| 2 | 8K | rand | 0.75-0.93 | 0.77-0.97 | 0.73-0.93 | 0.69-0.80 | 0.66-0.77 | 0.32-0.37 | 0.49-0.78 |
| 2 | 8K | zero | 0.76-0.93 | 0.77-1.02 | 0.72-0.98 | 0.68-0.83 | 0.65-0.80 | 0.32-0.37 | 0.49-0.76 |
| 2 | 32K | rand | 0.82-0.91 | 0.91-0.98 | 0.78-0.92 | 0.77-0.86 | 0.72-0.80 | 0.31-0.37 | 0.47-0.78 |
| 2 | 32K | zero | 0.82-0.91 | 1.01-1.17 | 0.85-1.13 | 0.86-0.99 | 0.77-0.95 | 0.31-0.38 | 0.46-0.79 |

- FlyTape's bracket spread inside one job was at most 1.04 (1.28 in one TP2 B128 32K job): the cells read ±0.02.
- FlyTape runs its bench from a zero state (no prefill at prompt 8K / 32K). So the "zero" rows are like for like, and
  the "rand" rows compare rocKE on random caches with FlyTape on zero caches.
- FlyTape's out-projection phases (corrected in the fix round, V6; FlyTape `model.py` `attn_out`, `bodies_tiled.py`,
  `docs/multi-gpu.md`):
  - TP1: `qsa_out` (split-K GEMM writing f32 partials) plus a separate `qsa_out_reduce` phase that sums them. The
    FlyTape column is the sum of both. rocKE's E1 `qsa_out` folds its splits in the kernel.
  - TP2: no `qsa_out_reduce` phase exists. The GEMM's last arriver folds the split partials inside `qsa_out` and pushes
    bf16 partials into the peer's mailbox (`attn_y`); the cross-rank add runs in the next `mlp_mix_down` as it loads
    `y`. So FlyTape's TP2 `qsa_out` includes the fold and the push but not the rank add. rocKE's E1 `qsa_out` has
    neither the push nor the add: that exchange is WP-C5 / WP-L5's in-situ all-reduce, which the composed TP2 pass
    adds (as in section 8).
- Per-row ranges: `wp-l2e/tmp/sum_byrows.md` (a ratio table).

### 4.3 `heq` / `heqs` (data-dependent select shortcut; t4 + t5, both nodes per TP)

| arm / pick (paired) | random data | zero data |
|---|---|---|
| `tpick` + `heq` | 1.00-1.03 (0/18/45) | 0.79-0.97 (63/0/0) |
| `tpick` + `heq` + `heqs` | 1.00-1.01 (0/40/2) | 0.80-0.97 (42/0/0) |
| `xpick` + `heq` | 1.00-1.05 (0/16/50) | 0.81-0.97 (66/0/0) |
| `xpick` + `heq` + `heqs` | 1.00-1.03 (0/30/14) | 0.82-0.97 (44/0/0) |

- With `heqs`, the product arm / FlyTape core on zero data, round-4 jobs (both TPs):
  - 8K: 0.70-0.91;
  - 32K: 0.78-0.91;
  - family: 0.63-0.78.
- Without it, the 32K zero cells at rows 8-32 sit at 0.91-1.13.
- Decision: `heqs` stays a default-off knob in the module and is part of the recommended product pick (`tpick_qs`)
  wherever the select is hierarchical. Rule: WIN in every zero / tied cell-job, TIE in 40 of 42 random cell-jobs (the
  2 losses are 0.6-0.7 % at rows 5). For the exact arm it is not recommended (random losses of 0.7-3 %).

### 4.4 Round 3 against round 2 (decision 15, re-checked on both nodes in the timing jobs)

- `xpick` r3 / r2: 0.84-1.00 (WIN in 116, TIE in 16, LOSS in 0 of 132 cell-jobs). That is `s64dpp`, with
  `scan_its` / `sela` / `sel` moves on some cells.
- `tpick` r3 / r2, cells where the pick changed: 0.95-1.02 (WIN 34, TIE 33, LOSS 5 of 72). The losses are TP1 8K zero
  at rows 32 (1.02, 3 jobs; its climb ran on random data, where it wins) and TP2 p128 rows 32 (`pgt=glb`, 1.01, 2 jobs).

## 5. WaveScope (ATT, Markham MI350X, one traced CU, 2 launches x 2 layers; triage only)

Before = att2 (round-2 picks); after = att3 (round-3 picks). Traced-wave totals are relative to before.

| cell / arm | before -> after (total latency) | top stalls after |
|---|---|---|
| TP2 128 x 32K, exact pick (+ `s64dpp`) | 0.88 | attention K / V waits (`vmcnt`) 29 + 15 %, K / V load issue 16 %, barrier 13 %; EXEC 62 % (f64 FMA issue: VALU 54 % of latency) |
| TP2 128 x 32K, tolerance pick (+ `sst4`, `its8`) | 1.01 | load issue 29 % (scan keys 14 %, K / V 15 %), `vmcnt(0)` 28 % (scan 11 %, attention 7 %), barrier 14 %, score stores 6 % |
| TP2 16 x 8K, exact pick | 0.99 | barrier 34 %, KV loads 16 %, `vmcnt(0)` 16 % |
| TP2 16 x 8K, tolerance pick | 0.96 | barrier 32 %, `vmcnt(0)` 20 %, KV loads 16 % |
| TP1 64 x 32K (after only) | - | exact: attention waits 20 % + K / V load issue 19 %, barrier 13 %; tolerance: attention waits 27 %, barrier 12 %, scan key loads 8 % |

The ledgers (`wp-l2e/att_items*.txt`, private `att/`) name the source lines:
- **The barrier:** `qsa_lb2.py` line 605, the per-layer phase sync.
- **The attention stage loads:** lines 2171-2172 / 2235-2237.
- **The scan key loads:** 1200 / 1225.

The tolerance arm at long context is bound by the bf16 `index_cmp` scan bytes (2x FlyTape's E4M3). That is the
`kcf=fp8` lever of section 7.

## 6. DESIGN 16: why the port is faster than FlyTape at 8K / 32K

Tools:
- `rv_ft_job.sh`, a new FlyTape dump at TP1 B16 x 8K and x 32K (job 68250362; the `.text` of the debug and plain
  builds is identical, 2 of 2);
- `wp-l2e/isa_cmp.py` (instruction classes) and `wp-l2e/isa_loops.py` (per-loop: backward SOPP branches inside the
  region).

Comparison: FlyTape's `qsa_core` (copy 0 of the phase in the tape) against the venv-COMGR port at the same shape
(`q38lb_qsa2_h24k2_b16c8256` / `c32832`; default knobs = P-L5's port, the same IR).

| | FlyTape `qsa_core` 8K / 32K | port 8K / 32K |
|---|---|---|
| instructions | 11,951 / 12,605 | 7,675 / 7,476 |
| `v_readlane` / `v_writelane` (SGPR spill traffic) | 616 / 260, 561 / 244 | 368 / 179, 372 / 178 (port SGPR spills: 179; tape: 1,593) |
| vector loads, `sc1` / plain | 443 / 21, 441 / 21 | 127 / 65, 129 / 65 |
| MFMA | 96 (attention stage in three ~260-instruction loop copies) | 48 (one 772-instruction 2-stage loop) |
| `s_waitcnt vmcnt(0)` | 77 / 75 | 49 / 50 |
| scan loop body | 1,755 / 1,708 instr., 32 `sc1` key loads, 44 / 16 readlanes + 8 writelanes per iteration | 1,304 instr., 32 plain key loads, 0 readlanes |
| p128 tape (rv-ft1 B16 x 128) | 10,096 instr., 560 readlanes | 7,486 instr. |

- **At p128** the phase is heads prep plus 2-3 attention stages. Both sides do the same work there, and P-L5 measured
  parity (0.95-1.04).
- **At 8K / 32K** the scan (2,048 / 8,192 blocks per row) and the select loops dominate.
  - FlyTape's loop bodies there carry SGPR-spill readlanes. The tape's 106-SGPR budget is shared by 30+ phases, so
    loop-invariant scalars are reloaded from VGPR spill slots inside the loops.
  - Its key loads also carry `sc1`, and it has about 35 % more instructions per scan iteration.
  - The port, a standalone kernel per phase, keeps its scalars in SGPRs and loads keys plain.
- This matches the measured gap: port / FlyTape 0.75-0.93 at 8K and 0.82-0.92 at 32K, growing with rows (more scan work
  per CTA).
- It is a property of FlyTape's single-tape build, not an LB lever. A composed LB pass (WP-L3 / L5) must keep each
  phase's scalars out of a shared SGPR budget, or it will inherit the same cost.

## 7. Lever ledger (every verdict names its base; losers kept as knobs)

| lever | verdict (base) | note |
|---|---|---|
| `kvl=page16` | product (required) | ATOM's caches; the paged FlyTape-semantics arm `pg` was 0.94-1.09 of the P-L5 pick (row layout) in the first timing round |
| `sem=atom` | product (required) | the semantics; its cost is section 4.1's pick column |
| `score=mfma` (t) | product | DESIGN-LB E9's `q3=mfma`; tolerance class; **RULER gate owed before it serves** |
| `score=valu` (t) | product at TP1 5x8K, TP1 8x8K, TP2 8x8K (and every p128 cell, where nothing is scored) | f32 FMA chain; tolerance class; **RULER gate owed as well** (fix round V2) |
| `score=f64` (x) | bitwise arm, kept | loses to the port at 8K / 32K (x / ft 0.97-1.30) |
| `hsel` | accepted r2 (r1 picks) | in every long-context pick |
| `fma64` | accepted r2 for x (r1 picks) | |
| `s64dpp` | accepted r3 for x (r2 picks), both nodes | 0.84-1.00 |
| `sst4` | accepted r3 for t in 10 cells (r2 picks) | small (0.98-0.99 via lookback pairs) |
| `heq` | rejected r3 (r2 picks, random data); WIN on zero data | superseded by `heqs` |
| `heqs` | TIE (random) / WIN (zero), both nodes (r3 picks) | recommended for the tolerance arm (4.3) |
| `waves=8` + ATOM semantics at TP1 | refused by the scratch gate (venv COMGR, 8-80 B) | register-lean 512-thread variant still open (P-L5) |
| `gang`, `pf`, `qloc`, `qlds`, `mpad`, `sela`, `spf`, `gearly`, `pgt`, `kvcp`, `scan_its`, `sel=v1/v2/bits` | per-cell climb moves | ledgers in the v1-v6 results (`climb` records, base per step) |
| **`kcf=fp8`** (not built) | - | an E4M3 shadow of `index_cmp` written at append time (FlyTape's key format), scored with fp8 MFMA. It halves the scan bytes that bound the tolerance arm at 32K (section 5). It changes ATOM's cache representation (tolerance class, RULER-gated) and needs a prefill-side conversion. Estimate: <= 0.3 % of a TP2 B128 8K step (QSA family share 0.041 of FlyTape's step). Not built this round because of that bound. |
| **exact-by-filtering** (not built) | - | MFMA scores with a rigorous error bound, exact f64 rescoring only of blocks within the bound of the top-512 boundary: ATOM's bits at about the tolerance arm's cost. Worth building only if the RULER gate rejects `score=mfma` |

## 8. Budget consequence (DESIGN-LB 15.7, derived)

- 15.7 used **QSA family 0.60-0.85** of FlyTape's (P-L5's conservative core plus WP-L2a's `qsa_in`). Measured now,
  same job, with ATOM's semantics (random data):
  - TP2: 0.57-0.62 (p128), 0.66-0.77 (8K), 0.72-0.80 (32K);
  - TP1: 0.49-0.55, 0.64-0.67, 0.67-0.71.
  - On zero state (FlyTape's bench state), with `heqs`: 0.63-0.78 at 8K / 32K.
- At FlyTape's QSA share of the step (0.041 TP2 B128, 0.055 TP2 B32, 0.064 TP1 B32), the 15.7 central K0 ratios move by
  at most -0.005 (the new range is inside the old one). The QSA family is not on the critical path of the 10k target.
- The TP2 `qsa_out` exchange is carried by WP-C5's in-situ figure, as before.

## 9. Open items

1. **RULER gate for `score=mfma` and `score=valu`** (the product arm is tolerance class; `valu` serves TP1 5x8K,
   TP1 8x8K and TP2 8x8K). Until it passes, `xpick` (bitwise scores) is
   the safe arm. The cost of falling back: x / t 1.02-1.18 at 8K / 32K.
2. ~~Independent verification of this WP.~~ Done (partial; section "Verification"). Its findings V1-V9 are
   answered in "Fix round", including the measured bytes-floor A/B.
3. The register-lean 512-thread variant (TP1 `waves=8` with ATOM semantics spills on the venv COMGR).
4. `kcf=fp8` and exact-by-filtering (section 7), if the composed pass shows the QSA family on the critical path at long
   context.
5. Climbs on zero data (all climbs ran on random data). `heqs` was decided from the timing jobs' paired arms instead.

## 10. Rules and records

- alola skill rule 234 (this WP):
  - the slow login-node stage tar of about 1,300 prebuilt objects;
  - the `waves=8` scratch refusals (TP1 and, per fix round V7 and rule 240, TP2) and the misleading per-cell FAIL
    summary they cause;
  - the long-context FlyTape ISA dump recipe;
  - the time split of container timing jobs.
- Harness change: `job.sh` takes `L2E_HSACO_STAGE=<stage>` (prebuilt objects in venv and container mode alike; a
  `--env` value cannot expand `$Q38_STAGES`).
- Absolute numbers and raw data: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2e.md` and `wp-l2e-raw/`.

## Verification

Independent verifier (W10), 2026-11-02. **Verdict: partial.** The numerics gate and the in-situ timing reproduce on
fresh jobs, including the 10k batch cell. Two things stop a plain "confirmed":
- One element of the group-L2 gate, the A/B against the bytes floor (DESIGN-LB 15, group L2), was never run.
- The product arm is tolerance class and its RULER gate is still owed. That gate also covers `score=valu` (V2).

**Re-runs (the cheapest decisive checks).** Both used fresh stages of the local trees: `w10-vWP-L2e-rk` (copy-rocm-lib
rocKE) and `w10-vWP-L2e-h` (a copy of the builder's harness). The file hashes match the builder's final jobs: `qsa_lb2.py`
6563b131, `qsa_lb.py` 142e0e3f, `qsa_ref2.py` 93284aa7, `l2e_bench.py` 18bd0b19.
- **Job 68251037 `w10-vWP-L2e-numn07`** (n07-17, venv, GPU bus 0x75; bus e5 not allocated): gate, numerics and chain
  phases.
  - The compile gate passes 489 of 489 builds. Scratch is 0 and the `hz` hazard scan finds 0.
  - Every one of the roughly 60 arms in all 8 cases passes. Every order-keeping group is bitwise equal, and so are the
    MXFP8 packs.
  - **All 23 must-fail mutants are caught, now also on n07-17.** That includes `m_hneed_qs`, which the builder had on
    n01-09 only. The per-case counts are identical to v5 / v6 / v7, e.g. `tie` 5 of 8 and `scan_mf` 7 of 8.
  - The stale-read twin `stale_plain` is caught in 7 of 8 cases. It misses `tp2_dense`, where no row reaches the scan,
    so the miss is structural. `stale_sc1` passes and is bitwise equal to `atx`.
  - The chain passes at 4 cells: E1 `qsa_in` error about 3e-5, `qsa_out` about 2e-5.
- **Job 68251031 `w10-vWP-L2e-ftn01`** (n01-09, container, 2 GPUs): FlyTape TP2 traced before and after the timing
  phase.
  - Cells: TP2 B16 8K, B128 8K and B128 32K, on zero and random data, 12 shuffled rounds.
  - Code objects: the builder's prebuilt `w10-l2e-hsaco7` under `L2E_HSACO_STRICT=1`, with the final picks
    (`l2e_picks_final.json`).
  - This adds the `heqs` product arm (`tpick_qs`) in situ on the TP2 node where the builder's bracket (t3) predates
    `heqs`.
  - Every timed arm passed its own random-data check, with stream and phase twins equal and both checker controls
    passing. The 10 FAIL lines are refused non-pick arms whose phase twin is missing (rule 234).

| TP2, phase metric | B16 8K rand / zero | B128 8K rand / zero | B128 32K rand / zero | report (4.2, 4.3) |
|---|---|---|---|---|
| `tpick` / FlyTape `qsa_core` | 0.93 / 0.97 | **0.74 / 0.74** | 0.79 / 0.87 | 0.73-0.93 (8K), 0.78-0.92 (32K rand), 0.85-1.13 (32K zero) |
| `tpick_qs` / FlyTape `qsa_core` | 0.93 / 0.90 | 0.75 / 0.71 | 0.79 / 0.79 | 0.70-0.91 (zero) |
| QSA family (`tpick_qs`) / FlyTape | 0.77 / 0.75 | **0.66 / 0.63** | 0.71 / 0.71 | 0.66 at B128 8K |
| `tpick` / `ft` port (in-job verdict) | 1.016 LOSS / 1.065 LOSS | 0.968 WIN / 0.935 WIN | 0.922 WIN / 1.033 LOSS | rows-16 losses (1.02) / 32K zero losses as in 4.1 |
| `tpick_qs` / `tpick` | 1.008 TIE / 0.929 WIN | 1.006 TIE / 0.966 WIN | 1.003 TIE / 0.903 WIN | TIE (rand) / WIN (zero) |

- Controls `ft2` and `ft_rc` sit at 0.993-1.004.
- FlyTape's two brackets agree within 5 % on `qsa_core`: 1.05x between pre and post at B128 8K, nearly flat at 32K. The
  spread of 1.24 that the report script prints for B128 8K comes from the small `qsa_out` phase.
- The headline cell reproduces on the other node: TP2 B128 8K core 0.74, QSA family 0.63-0.66. The section 8 budget
  conclusion holds: the QSA family is off the critical path of the 10k target.

**Static checks (local).**
- `test_qwen38_lb_qsa2.py`: 57 passed. It includes `test_default_ir_equals_pl5`, IR identity with the new knobs off over
  5 configurations.
- `qsa_lb.py` is unmodified against git.
- Per-arm audit of all 6 timing jobs' `cellcheck` records: no timed arm, in any cell or job, failed its check. `ft2` is
  the second graph of the checked `ft` object.
- Every recorded build in the timing jobs, the v7 job and both re-runs has `hazards = 0`. The value is never missing.
- Climb steps: 646, which matches.

**Findings (V1 affects the gate; V2-V9 are corrections and gaps):**

- **V1. The bytes-floor A/B was never run.**
  - Group L2's gate asks for an isolated A/B against the bytes floor as well as against FlyTape.
  - `l2e_bench.py` builds the floor streamer (`jobs[("floor",)]`) but never times it, and no floor ratio appears in
    this report.
  - Derived figure, not measured: P-L5's measured TP2 floor (fix round, n01-09), scaled by
    `cell_bytes(atom=True) / cell_bytes(atom=False)`. That ratio is 1.107 at 8K and 1.318 at 32K, from the bf16 index
    keys.
  - The product arm then sits at about 1.7x the floor at B128 8K / 32K and about 6.3x at B16 8K.
  - To close V1, time the floor at `cell_bytes(atom=True)` in the timing jobs and report arm / floor per cell.
- **V2. The RULER gate is owed for `score=valu` as well as `score=mfma`.**
  - The report calls the product arm "MFMA indexer scores". But `tpick` uses `score=valu` at three long-context cells:
    TP1 5x8K, TP1 8x8K and TP2 8x8K.
  - It also uses `valu` at every p128 cell, but nothing is scored there (no scan), so those do not matter.
  - Open item 1 should name both tolerance-class score forms.
- **V3. `heqs` in situ on the other node: TP1 still missing.**
  - In the builder's data, the in-situ ratios for `tpick_qs` come from one node per TP: t4a1n01 for TP1 and t4a2n07
    for TP2.
  - This verification adds TP2 on n01-09 for 3 cells. TP1 `tpick_qs` against FlyTape on n07-17 is still unmeasured.
  - The paired `tpick_qs` / `tpick` ratio is on both nodes (t4 + t5).
- **V4. The timed E1 GEMMs are not checked in the timing jobs.**
  - The E1 `qsa_in` / `qsa_out` objects that make up the QSA family ratio get no numerics check there: `check_cell`
    covers core arms only.
  - Their only check in this WP is the chain at 4-5 shapes. Those are venv builds, not the strict container objects,
    and rows 5 / 8 / 32 / 64 are not covered.
  - The chain's core is `score=f64` with default knobs, not the product pick. The product pick's numerics come from
    the per-cell checks instead.
- **V5. Neighbours are not recorded in the container timing jobs.**
  - None of t3, t4 or this job records neighbours: there is no `squeue` inside the container (rule 235).
  - So every FlyTape in-situ figure lacks DESIGN 9.1's neighbour record. The venv jobs t5 have it.
  - The same-job brackets limit the damage, but the rule is not met as written.
- **V6. FlyTape's "`qsa_out` (+ reduce)" phase does not exist in the TP2 traces.**
  - The traced TP2 phases are `qsa_proj`, `qsa_core`, `qsa_out` and the mixer / MoE phases. There is no
    `qsa_out_reduce`, in this job or in t4a2n07.
  - Sections 4.2 and 8 say FlyTape's side includes its reduce while rocKE's has no exchange. In fact neither side
    shows a separate reduce phase.
  - The note should be corrected, or the place where FlyTape's out-proj reduction runs should be identified.
- **V7. `waves=8` refusals also happen at TP2.** Section 2 and rule 234 say only TP1 builds are refused. TP2 builds
  were refused too: rows 64 / 128 at 8K, both `sem=ft` and `sem=atom`, with 8-40 B of scratch, per the GATE FAIL lines
  of t3a2n01 and t4a2n07. None of them is a pick.
- **V8. The climb move counts are lower than the ledgers.**
  - The climb ledgers hold 9,714 move results: r1 5,713, r2 2,198, r3 1,803.
  - The report says 9,068: 5,314, 2,059 and 1,695.
  - The step count, 646, matches. The report should define its count, e.g. whether lookback and refused moves are
    excluded.
- **V9. Small wording points.**
  - In 4.1, `tpick` / `ft` at TP2 8K zero reaches 1.065 here, just outside the reported 0.94-1.06.
  - "Chain at 5 cells" depends on the `--chain-cells` argument: the default runs 4.

Records:
- Raw data (absolute numbers) for both jobs and a ratio JSON (`report_vftn01.json`):
  `hackathon-prep/private-perf/qwen38-true-mono/large-batch/vwp-l2e-raw/`.
- Harness copy: `hackathon-prep/qwen38/truemono-lb/vwp-l2e/`.
- alola-cluster skill rule 237.
- No git changes; no files deleted.

## Fix round

W10, 2026-11-02. The fix round answers the verification's findings V1-V9. Two new jobs ran on fresh stages with the
same strict prebuilt code objects (`w10-l2e-hsaco7`, `L2E_HSACO_STRICT=1`) and the same final picks
(`l2e_picks_final.json`). The kernel files are unchanged (`qsa_lb2.py` 6563b131, `qsa_lb.py` 142e0e3f, `e1.py`
debca5f9, `e1_plans.py` b8552fe8). The rocKE stage is the verifier's `w10-vWP-L2e-rk`.

The harness is `hackathon-prep/qwen38/truemono-lb/wp-l2e-fix/` (stage `w10-l2e-hfix`), a copy of `wp-l2e/` with
these additions:
- `l2e_bench.py`:
  - floor graphs in `time_cell`;
  - `e1_check`, a numerics check of each timed E1 object;
  - `--chain-picks`, which runs the chain on the timed picks;
- `fx_report.py`;
- `neighbours.py`.

The builder's `wp-l2e/` files are untouched.

| job | node | mode | what |
|---|---|---|---|
| 68251229 `w10-l2efx-An07` | n07-17 (bus 0x75; e5 not allocated) | container, 2 GPUs | Steps: FlyTape TP1 traced (the 12 t3 / t4 configs) -> gate, chain and timing -> FlyTape TP1 traced again. Timing: all 36 cells, zero and random data, 12 shuffled rounds |
| 68251232 `w10-l2efx-Bn01` | n01-09 | venv, 1 GPU | the same gate, chain and timing, all 36 cells (no FlyTape) |

In both jobs:
- The compile gate passes 566 of 566 builds (every one prebuilt, with scratch 0 and no hazards).
- Every timed core object passes its per-cell check: 254 of 254 arms over 36 cells, with all 36 checker-control pairs
  firing.

### V1: the bytes-floor A/B (now measured)

How it was measured:
- **The streamer.** P-L5's floor is a read streamer of the same bytes: nt + sc1 loads from 256 CTAs. It now runs in
  every timed cell, in the same shuffled rounds as the arms.
- **Which bytes:**
  - the core at ATOM's bytes (`cell_bytes(atom=True)`, bf16 index keys);
  - the `ft` port at FlyTape's bytes;
  - each E1 GEMM at its own bytes (MXFP4 weights and scales, MXFP8 X, bf16 Y);
  - the family floor is the sum of the three.
- **The ratio.** Arm (phase metric) over its floor, paired per round. Ranges are over rows 5-128 and both nodes.

| TP | prompt | product core / floor | `xpick` / floor | `ft` port / its floor | E1 `qsa_in` / floor | E1 `qsa_out` / floor | QSA family (product) / floor |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 3.9-86 | 3.9-86 | 4.7-112 | 2.2-3.4 | 3.9-7.2 | 4.0-8.0 |
| 1 | 8K | 1.5-14 | 1.5-15 | 1.7-16 | 2.3-3.5 | 4.0-7.5 | 1.6-8.0 |
| 1 | 32K | 1.5-13 | 1.8-16 | 1.9-17 | 2.2-3.5 | 3.7-7.3 | 1.6-8.4 |
| 2 | 128 | 4.7-96 | 4.7-98 | 5.5-127 | 2.9-4.7 | 6.2-8.4 | 5.1-9.8 |
| 2 | 8K | 1.7-18 | 1.8-19 | 1.9-21 | 3.0-4.6 | 6.0-8.4 | 1.9-10.8 |
| 2 | 32K | 1.6-15 | 2.0-18 | 2.3-21 | 3.0-4.6 | 6.0-8.3 | 1.7-10.4 |

The low end of each range is rows 128 and the high end is rows 5. The two nodes agree within 0.05 at rows 128.

| cell (random / zero data, both nodes) | product core / floor | QSA family / floor |
|---|---|---|
| TP2 B128 8K (the 10k batch) | 1.67-1.69 / 1.67-1.72 | 1.85-1.89 / 1.87-1.91 |
| TP2 B128 32K | 1.64-1.68 / 1.61-1.62 | 1.76-1.80 / 1.73-1.74 |
| TP2 B16 8K | 6.3-6.4 / 6.2 | 5.8-5.9 / 5.7 |
| TP1 B128 8K / 32K | 1.51 / 1.58 (random) | 1.65-1.66 / 1.69 |
| TP1 B64 8K / 32K | 1.84-1.87 / 1.87-1.93 (random) | 2.06-2.10 / 2.05-2.10 |

- **The verifier's derived figures hold.** It estimated about 1.7x at TP2 B128 and about 6.3x at B16 8K from P-L5's
  floor scaled by bytes. The measured values are 1.67-1.72 and 6.2-6.4. The measured TP2 floors are 1-5 % above the
  derived ones (private notes).
- **Reading.**
  - At rows 128 and long context the product core runs at 1.5-1.7x its bytes floor. The `ft` port runs at 1.65-2.35x
    of its own (smaller) floor.
  - At small rows and at p128 the core is latency- and sync-bound: the bytes per layer are a few MB. The ratio grows
    roughly as 1 / rows, as P-L5 found for FlyTape's semantics.
  - The E1 GEMMs are at 2.2-8.4x their floors (0.3-3 MB per layer, latency-bound, as WP-L2a found).
- **What the floor says about headroom.** The 10k cell (TP2 B128 8K) has at most 1 - 1/1.68, about 0.40, of the core's
  time left above the bandwidth floor. That headroom is the wait and issue stalls WaveScope names (section 5): K / V
  waits, scan key loads and the per-layer barrier.
- **Group L2 gate.** The A/B against the bytes floor is now measured in every cell on both nodes, alongside the A/B
  against FlyTape.

### V2: two tolerance-class score forms

`tpick` scores with `score=valu` (the f32 FMA chain) at TP1 5x8K, TP1 8x8K and TP2 8x8K. It also uses `valu` at every
p128 cell, where nothing is scored. Everywhere else at 8K / 32K it uses `score=mfma`.

Both forms are tolerance class. The status line, section 0, the lever ledger (section 7, new `score=valu` row) and open
item 1 now name both forms. The RULER gate is owed for both before the product arm serves.

### V3: the product arm with `heqs` against FlyTape in situ, TP1 on n07-17 (closed)

Job 68251229 is the TP1 FlyTape bracket on n07-17. It times the product arm (`tpick_qs` where the cell selects
hierarchically, else `tpick`) against FlyTape's `qsa_core` from the same job (median of the before and after brackets).

| TP1, n07-17 | p128 (rows 8-64) | 8K (rows 8-64) | 32K (rows 8-64) |
|---|---|---|---|
| product / FlyTape `qsa_core`, random | 0.75-0.80 | 0.77-0.89 | 0.82-0.90 |
| product / FlyTape `qsa_core`, zero | 0.76-0.80 | 0.76-0.89 | 0.80-0.88 |
| QSA family / FlyTape (`qsa_proj + qsa_core + qsa_out + qsa_out_reduce`) | 0.49-0.55 | 0.63-0.66 | 0.66-0.71 |

- The values are inside the report's TP1 ranges (section 4.2), now on the other node.
- **`heqs` against `tpick`, paired, n07-17:**
  - zero data: WIN in every cell, 0.84-0.96;
  - random data: TIE in every cell, 1.002-1.009.
- **FlyTape's brackets** (before / after) agree within 1.02 on `qsa_core`, `qsa_proj` and `qsa_out`. The small
  `qsa_out_reduce` phase spreads up to 1.13.
- **Coverage.** The product arm against FlyTape in situ now has, from the builder's jobs plus this round, both nodes for
  TP1 (n01-09 t4, n07-17 here) and for TP2 (n07-17 t4, n01-09 in the verification).

### V4: the timed E1 objects are numerics-checked, and the chain runs on the timed picks (closed)

- **Per-cell E1 check (`e1_check`).** Every timing cell now checks both timed E1 objects (`qsa_in`, `qsa_out`)
  before they are timed.
  - **What is checked:** the strict prebuilt code object the cell times, in both launch forms it times (stream over all
    layers; phase with the layer walk's grid sync).
  - **Data:** the cell's own random weights and X.
  - **Layers:** 0, Le/2 and Le-1, against the f64 reference.
  - **Tests:** determinism (two launches bitwise equal) and stream / phase twins bitwise equal.
  - **Must-fail checker controls:** no launch, and one output element moved by 2^-6 of the maximum.
  - **Result:** 144 of 144 PASS on each node (36 cells x 2 data x 2 GEMMs). The largest error is 4.9e-5 against the
    2^-9 bound. Determinism, twins and both controls hold in 144 of 144.
  - A failing E1 object would now be dropped from the timing.
- **The chain on the timed picks (`--chain-picks tpick_qs|tpick,xpick_qs|xpick`, all 36 cells).**
  - **What is chained:** the timed E1 `qsa_in` -> the cell's timed product pick, and separately its exact pick ->
    the timed E1 `qsa_out`, reading the core's MXFP8 output with no repack.
  - **Objects:** the same kernel names as the timed objects, so in strict mode the same code objects.
  - **Result:** 72 of 72 PASS on each node (`tpick` 15, `tpick_qs` 21, `xpick` 14, `xpick_qs` 22), at every row
    count, rows 5 / 8 / 32 / 64 included.
  - The container run is the strict container-loaded objects, not venv builds.

### V5: neighbours of the container timing jobs (recorded from outside the container)

A container has no `squeue`, so `neighbours.py` now records the neighbours from the login node. It reads `sacct -a -r
rocke` over each job's run interval and lists every other job on the same node whose interval overlaps.

| job | node | neighbours (jobs / GPUs, overlapping the run interval) |
|---|---|---|
| t3a2n01 68249974 | n01-09 | 10 / 10 |
| t3a1n07 68249976 | n07-17 | 7 / 9 |
| t4a2n07 68250497 | n07-17 | 9 / 9 |
| t4a1n01 68250499 | n01-09 | 13 / 14 |
| verification ftn01 68251031 | n01-09 | 5 / 6 |
| fix An07 68251229 | n07-17 | 7 / 7 |
| fix Bn01 68251232 (venv; `squeue` in the job as well) | n01-09 | 6 / 6 |

- The counts are over the whole interval, so jobs that ran one after another are each counted. Some of the count is
  agorenko's and tkoermen's `sacct` records that say RUNNING since 10-29. The venv job's own `squeue` does not show the
  two tkoermen records, so they look stale.
- All neighbours are other users' or other lanes' single-GPU jobs. None shares this job's GPU (GRES-exclusive).
- With these records, DESIGN 9.1's neighbour requirement is met for every in-situ FlyTape figure of this WP. The
  records are in the private raw dir.

### V6: FlyTape's out-projection phases (corrected)

FlyTape's TP2 tape has no `qsa_out_reduce` phase. Its TP1 tape has one. Sources: FlyTape `model.py` `attn_out`,
`bodies_tiled.py` `_out_reduce`, and `docs/multi-gpu.md` ("two all-reduces per layer, and no phase of their own").

- **TP2.** `qsa_out`'s last arriver folds the split partials and pushes bf16 partials to the peer's mailbox. The rank
  add runs inside the next `mlp_mix_down`.
- **TP1.** `qsa_out` writes f32 split partials and `qsa_out_reduce` sums them.

Section 4.2's note, the section 0 wording and the table header are corrected:
- **TP1:** the FlyTape out column is `qsa_out + qsa_out_reduce`.
- **TP2:** it is `qsa_out` alone. That includes FlyTape's fold and its push, but not its rank add.
- **rocKE's E1 `qsa_out`:** neither push nor add. The TP2 exchange stays budgeted separately via WP-C5's in-situ figure
  (section 8).

The numbers do not change: `l2e_report.py` already used `qsa_out_reduce` only where it exists.

### V7: `waves=8` refusals at TP2 too (corrected)

Per timing job, the compile gate refused 92 TP1 builds and 48 TP2 builds. All of them are `waves=8` builds with paged
KV.
- TP1: 85 with ATOM semantics plus 7 `sem=ft`.
- TP2: 46 plus 2. The TP2 refusals are at p128 rows 8-128, 8K rows 64 / 128 and 32K rows 128, with 8-68 B of scratch.
- None is a pick.

Section 2's compile-gate row and section 10 are corrected. alola rule 240 amends rule 234.

### V8: the climb counts (defined)

| round | climb steps | candidate moves measured on the steps | step ledger records incl. the per-step control entry | end-of-climb finals | lookback drops / pairs | lookback confirms |
|---|---|---|---|---|---|---|
| r1 | 399 | 5,314 | 5,713 | 114 | 6,115 | 71 |
| r2 | 139 | 2,059 | 2,198 | 72 | 7,177 | 39 |
| r3 | 108 | 1,695 | 1,803 | 72 | 8,454 | 39 |
| total | 646 | **9,068** | 9,714 | 258 | 21,746 | 149 |

- The report's 9,068 counts candidate moves measured on the climb steps.
- The verifier's 9,714 also counts the one control record (`_ctl`) each step stores.
- Moves refused by the gate or the pre-check are not measured, so they appear in neither count.
- Counting every measured configuration (steps + finals + lookback + confirms), the three rounds timed 31,221.

### V9: wording points (corrected)

- **4.1, `tpick` / `ft` at TP2 8K zero.** Rows 16 measured 1.057 on both nodes in the fix jobs, and 1.065 in the
  verification. Over all jobs the range is 0.94-1.07; a note is added under the table.
- **The product arm at that cell (`tpick_qs`, zero data)** does not lose: 0.90-0.98 against `ft` at TP2 8K zero over
  rows 5-128 (11 WIN, 1 TIE).
- **The chain cell count** is now stated with its argument: v7 ran 5 cells (`--chain-cells`), the default is 4, and the
  fix round chains all 36 cells on the timed picks.

### Fix-round paired ratios (both nodes, 36 cells, zero and random data, 12 rounds; phase metric)

The fix jobs re-measure section 4.1 / 4.3. Each entry is the range with (WIN / TIE / LOSS) counts.

| TP | prompt | data | product / `ft` | `tpick` / P-L5 pick | `tpick_qs` / `tpick` (`heqs`) |
|---|---|---|---|---|---|
| 1 | 128 | rand | 0.75-0.82 (12/0/0) | 1.00-1.06 (0/3/7) | - |
| 1 | 128 | zero | 0.75-0.82 (12/0/0) | 1.00-1.06 (0/3/7) | - |
| 1 | 8K | rand | 0.93-0.99 (12/0/0) | 1.06-1.15 (0/0/10) | 1.00-1.01 (0/6/0) |
| 1 | 8K | zero | 0.86-0.97 (12/0/0) | 0.92-1.14 (2/2/6) | 0.94-0.96 (6/0/0) |
| 1 | 32K | rand | 0.93-0.99 (10/2/0) | 1.13-1.22 (0/0/10) | 1.00-1.01 (0/10/2) |
| 1 | 32K | zero | 0.91-0.98 (12/0/0) | 1.24-1.41 (0/0/10) | 0.83-0.95 (12/0/0) |
| 2 | 128 | rand | 0.76-0.87 (12/0/0) | 0.95-1.08 (4/4/2) | - |
| 2 | 128 | zero | 0.76-0.86 (12/0/0) | 0.95-1.07 (4/4/2) | - |
| 2 | 8K | rand | 0.95-1.02 (8/2/2) | 1.07-1.19 (0/0/10) | 1.00-1.01 (0/11/1) |
| 2 | 8K | zero | 0.90-0.98 (11/1/0) | 1.09-1.23 (0/0/10) | 0.92-0.97 (12/0/0) |
| 2 | 32K | rand | 0.94-1.02 (10/0/2) | 1.11-1.32 (0/0/10) | 1.00-1.02 (0/9/3) |
| 2 | 32K | zero | 0.92-1.02 (10/1/1) | 1.28-1.53 (0/0/10) | 0.80-0.90 (12/0/0) |

- **The product arm against the FlyTape-structure port.** It loses only at rows 16:
  - TP2 8K random: 1.02, both nodes;
  - TP2 32K random: 1.018-1.025, both nodes;
  - TP2 32K zero: 1.016, n07-17.

  These are the same rows-16 cells as section 4.1. FlyTape in situ remains the product bar.
- **`heqs` on random data, now over more cell-jobs.** It loses 6 of 38 here:
  - rows 5 / 8 at 32K on both TPs, plus TP2 5x8K;
  - the losses are 0.5-1.7 %.

  Together with section 4.3 that is 8 losses in 80 random tolerance-arm cell-jobs, all at rows 5-8, all 1.7 % or less.
- **The recommendation is refined.** Use `heqs` where states can tie: fresh or zero caches, and FlyTape's bench state.
  There it wins 0.79-0.97 in every cell-job (section 4.3 and this round). At rows 5-8 on long-context random data it costs up to 1.7 %.
- **The other ratios match section 4.1 within ±0.02.**
- Absolute times are in the private notes.

### Status after the fix round

- **Every V finding is answered.**
  - **Measured or re-run:** V1 (floor A/B), V3 (TP1 n07-17 in situ), V4 (E1 checks, chain on the timed picks).
  - **Recorded:** V5 (neighbours).
  - **Corrected in the text:** V2, V6, V7, V8, V9.
- **The gate holds.** It is met on both nodes for the product arm against FlyTape in situ, and the bytes-floor A/B is
  now measured.
- **What still blocks serving the product arm:**
  - open item 1, the RULER gate for both tolerance score forms (`mfma`, `valu`); this is model-level accuracy work,
    outside the kernel WP;
  - open items 3-5 (register-lean 512-thread variant, `kcf=fp8` / exact-by-filtering, zero-data climbs).
- **For the 10k target:** at TP2 B128 8K the QSA family is 0.63-0.66 of FlyTape's in situ and 1.85-1.91x its bytes
  floor. Section 8's conclusion stands: the QSA family is not on the 10k critical path.
- Records:
  - absolute numbers: `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2e.md` ("Fix round");
  - raw data: `wp-l2e-fix-raw/` (An07, Bn01, `report_fx.json`, `summary_fx.md`, the `sacct` dumps and the neighbour
    lists);
  - alola-cluster skill rule 240.
- No git changes; no files deleted.
