# P-L5: the large-batch QSA core (sparse attention) at rows 8-128, TP1 and TP2, contexts 128 / 8K / 32K (gfx950)

Probe P-L5 of `../DESIGN-LB.md` 15.4 / 15.6: "QSA core at LB (row-gang, MFMA QK / PV, indexer, select) at p128 and 8K
vs FlyTape's QSA phases; the last unmeasured family". Workflow W7, 2026-11-01, alola `rocke` partition (MI355X, 256 CUs)
for every verdict, Markham MI350X for triage only (numerics, phase stamps, WaveScope). This file has ratios only;
absolute times are in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/P-L5.md`.

Status: **done. Gate met at TP1 and TP2 on both nodes (after the fix round, section "Fix round").** Numerics passed on
MI355X on every timed code object, stream build and `walk=phase` twin, on random data (every stage against a float64
host reference; 8 source mutants + 2 path variants all caught; stale-read twins caught; bounded-wait timeout control
fires; checker must-fail controls caught). Every timed binary of the verdict jobs is a venv-COMGR code object loaded by
name (`PL5_HSACO_STRICT=1`: none compiled by the container COMGR). The FlyTape-structure port reproduces FlyTape's
in-situ `qsa_core` phase at p128, and the per-cell LB pick beats the port with the win rule in every cell-job on both
nodes (TP1 and TP2) and FlyTape in situ in every bracketed cell (TP1 and TP2, both nodes, FlyTape traced before and after
in the same job). The first round's TP2-vs-FlyTape table (job c2n01, container-COMGR binaries, failed numerics) is
withdrawn and replaced (4.3).

## 0. Summary

1. **What was built** (`rocke/library/kernels/gfx950/qwen38_mono/large_batch/qsa_lb.py`, Python engine, `q38lb_qsa_*`):
   - One kernel family that first reproduces FlyTape's row-gang `qsa_core` (`flytape/models/qwen4/bodies_qsa.py` with
     the indexer / scan / select helpers of `bodies.py`) in rocKE: unit = (layer, row, gang member); heads prep (q / k
     zero-centred RMSNorm + partial RoPE, KV append), indexer ring / block compression to E4M3 + power-of-two scale,
     dense path within the 2048-token budget, sparse path past it (indexer query published by member 0, chunked VALU
     scoring, last arriver selects the top 512 blocks with FlyTape's 16-step bit search and tie rule, release flag,
     token-position table), MFMA attention (`S^T = K Q^T` on 16x16x32 bf16, `O^T += V^T P^T` on 16x16x16 bf16 with V
     read transposed from LDS, next stage's K / V in flight), wave merge in LDS, gang merge by last arrivers (two levels
     past 8), sigmoid gate, bf16 output. Same gang rule (`ft_gang` = FlyTape's `gang_size`), same 64-token stage, same
     LDS layouts.
   - Cross-CTA payloads use `sc1` stores / loads (P5 rules), counters are agent-scope atomics reset by their last
     arriver with an atomic add, every wait is bounded and sets the launch-fatal word, every asm VMEM with an SGPR input
     carries the `s_nop 4` hazard pad (`hz`, compile-gated by the ISA scan).
   - 21 knobs, every LB lever default-off (section 5); `walk = stream | phase` (phase = a striped grid sync per layer,
     FlyTape's phase form); a bytes-floor streamer (`build_floor_kernel`); stamps (informational).
   - Host references `large_batch/qsa_ref.py` (float64 on the device's own inputs, stage by stage); CPU tests
     `library/tests/test_qwen38_lb_qsa.py` (22 pass); harness `hackathon-prep/qwen38/truemono-lb/p-l5/`
     (`pl5_bench.py`, `pl5_climb.py`, `pl5_summ.py`, `pl5_tables.py`, `pl5_condense.py`, `job.sh`, `prebuild.sh`,
     `att_qsa.py` / `att_job.sh`, `mk_job.sh`).
2. **Numerics gate met** (section 2): 6 cases (TP1 / TP2, rows 4-40, positions within the budget, at the budget boundary
   2050-2055, 8K, 32K, block-completing and ring steps mixed in one launch, 2 layers), about 50 arms each, plus (fix
   round) a per-cell check of every timed code object, stream AND phase twin, on random data at the cell's exact shape
   (layers 0-1, 3 rows, 2 launches, stream / phase twins bitwise equal, checker must-fail controls): all PASS on MI355X,
   both nodes, 207 / 207 arm pairs over all 30 cells per job; 8 distinct source mutants plus 2 re-runs on other code
   paths (`scan_mf`, `merge_2lvl`) caught; the 2 stale-read twins (`acq=plain` + pretouch / delayed publish) caught in 5
   of 6 cases; order-keeping knob groups bitwise equal; 2 launches bitwise equal.
3. **The port is at parity at p128 and faster than FlyTape in situ at 8K / 32K (unexplained)** (phase metric, same job, FlyTape bracketed before
   and after): FlyTape structure in rocKE / FlyTape's traced `qsa_core` = **0.97-1.02 at p128, 0.82-0.94 at 8K,
   0.86-0.92 at 32K** (TP1, both nodes; TP2 both nodes, rows 8-128, fix round: 0.95-1.04 p128, 0.76-0.94 8K, 0.82-0.91
   32K). The gap grows with bandwidth demand (rows x context) and is **unexplained** until DESIGN 16's IR diff of
   FlyTape's own build (open item); every `best / FT` below therefore includes up to 0.25 that no LB lever earned, and
   the budget carries the port-relative `best / ft` as its conservative figure (item 7).
4. **The QSA core is a latency chain, not a bandwidth stream** (stamps, WaveScope; section 4.4): one unit takes almost
   the whole phase. At p128 a unit is heads prep (0.16-0.23) + attention of 2-3 stages and the gang merge (0.77-0.83);
   at 8K / 32K the single-CTA select dominates at small rows (0.46-0.59 of a unit at rows 8) and the scan at rows 128
   (0.35-0.47). Even the best builds sit 1.5-1.8x the bytes floor at rows 128 long context and 4-66x at p128.
5. **Per-cell LB picks** (a hill climb with lookback over every knob, then a same-job FlyTape-bracketed confirmation,
   section 4): best / FlyTape-structure-in-rocKE = **0.72-0.74 (p128, rows 8-32), 0.79-0.84 (p128, rows
   64-128), 0.82-0.91 (8K), 0.77-0.84 (32K)**, WIN in every TP1 cell-job; best / FlyTape in situ = **0.72-0.79 (p128),
   0.70-0.82 (8K), 0.67-0.77 (32K)** at TP1, both nodes, zero and random data. TP2 (fix round, both nodes, prebuilt
   venv binaries, FlyTape bracketed in each job): best / ft 0.80-0.84 (p128), 0.85-0.88 (8K), 0.72-0.84 (32K), WIN in
   60 / 60 cell-jobs; best / FT **0.79-0.84 (p128), 0.66-0.79 (8K), 0.60-0.77 (32K)**. The launch metric (one launch per
   layer) gives 0.79-0.88 / 0.72-0.85 / 0.68-0.80 of FlyTape in situ at TP1 and 0.88-0.99 / 0.67-0.82 / 0.61-0.80 at
   TP2.
6. **The levers that carry it** (single-lever ledger on the FlyTape base, 30 cells, section 5): `mpad` (the waves' LDS
   merge with a padded row: FlyTape's 256-word rows put 16 head rows of a lane group in one bank) WIN in 30 / 30, 0.88-0.99;
   `qloc` (q kept in LDS on the dense path) WIN 10 / 10 at p128, 0.94-0.98; `sel=v1 / v2` (FlyTape's bit search with the
   counts done on the VALU instead of a ballot + `s_bcnt1` per key, the same selection) WIN 39 / 40 long-context cells,
   0.91-0.99; `score=mfma` + `kw16` (indexer scores on MFMA, bf16 query: tolerance class) WIN 19 / 20, 0.90-0.99;
   `waves=8` with `qlds` (512-thread CTA, 128-token stage) WIN 9 of 20 long-context cells, up to 0.87; `gang=4` at p128
   rows 8-64 (twice FlyTape's 2) WIN 8 / 8 (it loses at rows 128, where it needs two unit rounds). `pf=1` loses alone
   (0.97-1.09) but is in 57 of the 67 picks (PLAN decision 13).
7. **Budget consequence** (section 6): FlyTape's `qsa_core` is 0.015-0.043 of its step at p128 and 0.064-0.117 at 8K /
   32K. **Conservative (port-relative, what the LB levers earned): best / ft = 0.72-0.91 at TP1 and 0.72-0.88 at TP2
   (both nodes), saving 0.002-0.012 of FlyTape's step at p128 and 0.007-0.033 at 8K / 32K.** Against FlyTape in situ the
   core measures 0.60-0.84 (all cells, both TPs, both nodes), saving 0.003-0.012 at p128 and 0.012-0.047 at 8K / 32K, but
   that figure includes the unexplained port gap (item 3). DESIGN-LB 15.3 took the QSA family at 0.6-1.2 with the core
   unmeasured.

## 1. The kernel (`qsa_lb.py`)

| piece | FlyTape (`bodies_qsa.py` / `bodies.py`) | rocKE port (`ft`, every knob at its default) | LB levers (knobs) |
|---|---|---|---|
| unit | (row, member), gang = `gang_size` (power of two <= 32 dividing the grid, at most the 64-token rounds of the context) | identical; units (layer, row, member), layers runtime | `gang` (any power of two <= 64) |
| CTA | 256 threads, 64-token stage (16 per wave) | identical | `waves=8`: 512 threads, 128-token stage (needs `pf=1` + `qloc` + `qlds` to fit 256 VGPRs) |
| heads prep | every member preps every head within the budget (identical stores); past it each member preps `member + i * gang` | identical | `qloc`: q to LDS (dense rows skip the global round trip); `qlds`: S MFMAs read q from LDS each stage |
| indexer | wave 0 appends to the ring or completes a block (mean, norm, RoPE at block start, E4M3 + scale); the block completed past the budget is scored by member 0 | identical | - |
| scan | 16-lane rows, 8 dims per lane, f32 FMAs, row reduce, ReLU sum over 4 heads; chunks of 256 blocks dealt over the gang | identical | `scan_its` (chunk 2-16 x CTA/16 blocks), `spf` (next chunk in flight), `score=mfma` (16x16x32 bf16, keys E4M3 -> bf16 exact, query bf16), `kw16` (32-byte key loads, K order permuted in both operands) |
| select | last arriver: keys = f32 bits 30..15 in LDS, 16 single-bit rounds (ballot + popcount), runs + block scan, ascending list; release flag | identical | `sel`: `v1` / `v2` (VALU counts, 1 / 2 bits per round), `bits2` (2 bits, ballots), `nib` (4-bit digits, packed counters), `hist` (byte histograms): all give the same boundary; `sela` (every member selects, list in LDS, no release handshake) |
| attention | 8 x 16x16x32 S MFMAs, wave-private online softmax (exp2 domain, scale log2e/16), 16 x 16x16x16 PV MFMAs, V transposed through LDS (`ds_read_b64_tr_b16`), next stage prefetched | identical (unrolled by 2 so the prefetch never copies at the back edge) | `pf=1` (load per stage), `kvcp=ntsc1` (nt on the KV loads), `gearly` (gate loads before the stages) |
| merges | waves through LDS (`[wave][16 heads][256]` f32), members by last arrivers in groups of 8 then 8 groups | identical | `mpad` (260-word rows) |
| coherence | FlyTape buffers / fences | `sc1` stores + loads on every cross-CTA payload, atomics agent scope, bounded polls (launch-fatal word) | `acq=plain` = the stale-read twin |

Must-fail mutants (`MUTANTS`): `rope` (no RoPE partner), `scale` (softmax scale x2), `merge` (last partial dropped),
`tie` (ties to the higher block), `window` (the current token dropped), `kvpos` (append one slot early), `iqrope`
(indexer query not rotated), `scan` (one indexer head dropped; also on the MFMA path).

## 2. Numerics (MI355X; every timed binary)

Cases (2 layers, synthetic bf16 inputs N(0, 1), norm weights N(0, 0.01), random E4M3 keys with power-of-two scales):
`tp2_dense` (rows 8, positions 127-2050), `tp2_bound` (2050-8199: last dense position, first sparse, block-completing
steps), `tp1_mixed` (dense and 8K rows in one launch), `tp2_32k` / `tp1_32k` (32767-32800), `tp2_r40` (40 rows, a
non-power-of-two bucket, gang 4, mixed). Each (layer, row) is checked stage by stage against `qsa_ref` on the device's
own inputs:

| check | rule | result |
|---|---|---|
| q (qb) / k (KV append) | max err / max \|ref\| <= 2^-7 (bf16 storage) | 3.0e-3-3.9e-3 every arm |
| v append, ring append | bit exact | exact |
| block key (E4M3 x scale) | <= 0.13 of max (E4M3 has 3 mantissa bits) | 4.4e-2-5.4e-2 |
| indexer query | <= 1e-4 | 1.2e-7-1.5e-7 |
| scores | <= 1e-4 of the max (`mfma`: against the bf16-rounded query; the block completed this step is scored in f32) | 1.1e-7-1.5e-7 |
| selection | exact, from the device's own scores (FlyTape's tie rule) | 0 mismatches |
| gated attention output | per (row, head) max err / max \|ref\| <= 2^-6 (bf16 P and bf16 output) | 3.6e-3-4.8e-3 |
| determinism | 2 launches bitwise equal | every arm |
| order-keeping knob groups | bitwise equal: {`ft`, `sel=hist/nib/bits2/v1/v2`, `kvcp`, `walk=phase`, `pf`, `sela`, `spf`, `qloc`, `mpad`+`gearly`, `scan_its`, recompiled twin, stress}, {`waves=8` variants}, {`score=mfma` variants}, {`kw16` variants} | EQUAL (on the venv COMGR's binaries; see 2.1) |
| must-fail mutants | each caught in >= 1 case | 8 distinct source mutants (`MUTANTS`) plus 2 re-runs on other code paths (`scan` on the MFMA scorer, `merge` at gang 32 = two-level merge): all caught; `tie`, `iqrope`, `scan` miss only the dense case, which has no selection (tie in 3 cases, the others in 5-6) |
| stale-read twin (`acq=plain`, member 0 delays its publish, the others pretouch) | reported | caught in 5 of 6 cases per job; `stress` with `sc1` stays bitwise equal |
| timeout control (stress 200 + 1 poll iteration) | the launch-fatal word must be set | set in every sparse case |
| timed binaries in the timing jobs (fix round: `check_cell`) | every timed arm, its stream build AND its `walk=phase` twin (the primary metric), random data at the cell's exact shape, layers 0-1, rows 0 / B/2 / B-1, 2 launches bitwise, twins bitwise equal; checker controls (no launch, 5 % perturbed row) must be caught | 207 / 207 arm pairs over all 30 cells (TP2 timed + TP1 code objects) PASS per job, both nodes (f2n01, f2n07); max core error 4.9e-3; controls caught in 30 / 30 cells. The first round's per-cell check covered only the stream builds, and its zero-data half was vacuous (zero in, zero out) |

### 2.1 The container COMGR finding

The FlyTape brackets run in the q38 container (ROCm 10 COMGR); the climbs ran in the venv (torch 2.12 / ROCm 7.2
COMGR). The two COMGRs build different code from the same rocKE IR. **Container-COMGR 512-thread builds (`waves=8`) gave
wrong, nondeterministic outputs (core error 0.36-1.6, two launches different) with AND without spills** (job c2n01:
`w8` pf1, `w8g64`, `w8mf`, `w8_hist` at 32K were wrong with scratch 0; the spilled `w8_pf2`, `w8g8`, `w8g16`, `w8` at
rows 40 were correct). The first round said only the spilled ones were wrong and relied on the scratch == 0 gate; that
was false (verifier, corrected in alola rule 209 by rules 217 (P-L5 verification) / 219). Venv-COMGR `waves=8` builds are correct with and
without spills (MI355X numerics of c1n01p, c2n07v, f2n01, f2n07; MI350X 5-launch check of all 354 timed objects).
Rule since the fix round: **no container-COMGR QSA build is ever timed.** The verdict jobs run the venv-built code
objects inside the container (`prebuild.sh` in a GPU-less venv job, shipped as the q38 stage `w7-pl5-hsaco`, loaded by
kernel name: `PL5_HSACO_DIR`) with `PL5_HSACO_STRICT=1` (a missing object is a refused build, never a container
compile; must-fail control: an empty dir refuses every build), and each object's numerics are checked again in the same
job on the MI355X (`check_cell`).

## 3. Timing method

- **Cells:** TP1 (24 q / 2 kv heads per rank) and TP2 (12 / 1) x rows 8, 16, 32, 64, 128 x prompt 128, 8192, 32768;
  every row at position prompt - 1 + 32 (FlyTape's mid-step of its 64 decode steps). TP1 rows 128 has no FlyTape bar
  (FlyTape refuses TP1 B128); it is reported against the rocKE port.
- **Data:** random (bf16 N(0, 1), random keys: scattered selections) and zero (FlyTape's bench state is zero: all scores
  tie, the lowest 512 blocks are selected, contiguous). Every verdict holds in both; the tables give the range.
- **Cold data:** layers rotated so attended + scanned bytes >= 4x MALL, at least 12 (the model's QSA layers).
- **Metrics:** `phase` = one launch walks the layers with a striped grid sync after each (FlyTape's phase form; primary
  metric), `launch` = one launch per layer (the `layers=1` bring-up), `stream` = one launch walks the layers with no
  barrier (the overlap bound of a dataflow pass; not comparable to an in-situ phase at small rows, where it overlaps up
  to 50 layers' latency chains).
- **Arms:** `ft` (FlyTape structure), `ft_rc` (recompiled twin), `ft2` (second graph), LB arms; HIP graphs, arms and
  metrics shuffled every round, 12 rounds (climb: 8). Win rule DESIGN 9.1 (median below both controls' minima by 0.002
  and >= 3/4 rounds below 1).
- **FlyTape in situ:** `bench_qwen4_full.py --trace` (mxfp4, its random weights and zero state) in the same container job
  before and after the rocKE process, `qsa.qsa_core` / 12 layers; bracket spread 1.000-1.028.
- **Bytes floor:** a read streamer of the cell's attended KV + scanned keys + q-proj row + output bytes (nt sc1 loads,
  256 x 512, one launch over the rotated layers).
- **Jobs (MI355X):** screens 68241973, 68241990, 68242023; brackets 68242060 (TP2, n01-09, bus 05), 68242156 (TP1, n07-17,
  bus 117); climbs 68242226 (TP2, n01-09), 68242290 (TP2, n01-09), 68242294 (TP1, n07-17); confirmations 68242343 (TP2,
  n01-09 bus 133, container COMGR), 68242399 (TP1, n01-09, container COMGR), **68242821 (TP1, n01-09 bus 101, prebuilt)**,
  **68242876 (TP1, n07-17 bus 101, prebuilt)**, 68243956 (TP2, n07-17 bus 05, venv, no FlyTape; superseded);
  **fix round: 68244896 (TP2 + TP1 checks, n01-09 bus 0x75, FlyTape TP2 on 0x75 + 0x15, prebuilt strict)**, **68244900
  (same, n07-17 bus 0x05, FlyTape TP2 on 0x05 + 0x15)**; prebuild 68242600 (GPU-less, 1,056 code objects). Node + bus per row in the
  private tables; the bad n07-17 GPU (bus e5) was skipped by the health check. MI350X triage: 68242054, 68242147,
  68242062, 68242105, 68242162, 68242203 (numerics + stamps; 68242054 failed on an argument typo), 68242252 / 68242418
  (WaveScope).

## 4. Results

### 4.1 Port fidelity: the FlyTape structure in rocKE vs FlyTape in situ (phase metric)

| cells | ft / FT (TP1 rows 8-64, n01-09 + n07-17, zero + random) | ft / FT (TP2 rows 8-128, n01-09 + n07-17, fix round, prebuilt) |
|---|---|---|
| p128 | 0.974-1.016 | 0.954-1.038 |
| 8K | 0.818-0.942 | 0.756-0.936 |
| 32K | 0.860-0.923 | 0.823-0.913 |

(The first round's TP2 column, 0.91-0.99 / 0.75-0.95 / 0.84-0.92 from jobs 68242060 and c2n01, used container-COMGR
`ft` builds; `ft` is a 256-thread build that passed numerics there, and the fix-round numbers agree within 0.05.)

- At p128 (latency chain, little data) the port is at parity, which says the structure was reproduced faithfully: same
  units, same waits, same merges. The gap grows with rows and context (more bytes per unit). Candidate causes (code
  generation, cache bits, FlyTape's in-situ phase starting right after its `qsa_proj` phase with other phases' traffic
  just finished) are **untested**: the gap is unexplained until the DESIGN 16 IR diff, so `best / FT` is reported next
  to the port-relative `best / ft`, and the budget uses the latter as its conservative figure.
- Random vs zero data moves `ft / FT` by up to 0.06 at 8K rows 32-64 (scattered selections read more lines than the
  zero state's contiguous blocks); FlyTape only runs zero state, so the zero-data rows are the like-for-like ones.

### 4.2 TP1: per-cell LB pick vs the port and vs FlyTape in situ (prebuilt, both nodes)

Jobs 68242821 (n01-09) and 68242876 (n07-17); each cell: min-max over the two nodes and zero / random data; every
verdict is the in-job win rule against `ft` (controls `ft2`, `ft_rc`).

| prompt | rows | ft/FT phase | best/ft phase | verdicts vs ft | best/FT phase | best/FT launch | best/floor | best arm(s) |
|---|---|---|---|---|---|---|---|---|
| 128 | 8 | 0.995-1.016 | 0.715-0.733 | WIN 4 | 0.719-0.734 | 0.795-0.814 | 50.83-51.08 | pick3 |
| 128 | 16 | 0.983-1.002 | 0.715-0.737 | WIN 4 | 0.717-0.725 | 0.786-0.802 | 24.31-26.04 | pick, pick3 |
| 128 | 32 | 0.988-1.004 | 0.726-0.742 | WIN 4 | 0.728-0.737 | 0.793-0.814 | 12.77-13.22 | pick3 |
| 128 | 64 | 0.974-0.991 | 0.794-0.808 | WIN 4 | 0.787-0.788 | 0.845-0.878 | 7.08-7.42 | pick, pick2 |
| 128 | 128 | - | 0.808-0.842 | WIN 4 | - | - | 4.01-4.12 | pick, pick2 |
| 8192 | 8 | 0.924-0.936 | 0.817-0.827 | WIN 4 | 0.764-0.766 | 0.788-0.790 | 8.25-8.45 | pick3 |
| 8192 | 16 | 0.899-0.942 | 0.853-0.861 | WIN 4 | 0.774-0.806 | 0.786-0.827 | 4.67-4.91 | pick |
| 8192 | 32 | 0.859-0.938 | 0.872-0.877 | WIN 4 | 0.751-0.823 | 0.770-0.851 | 2.83-3.10 | pick3 |
| 8192 | 64 | 0.818-0.874 | 0.857-0.899 | WIN 4 | 0.703-0.785 | 0.722-0.811 | 1.87-2.10 | pick2, pick3 |
| 8192 | 128 | - | 0.877-0.905 | WIN 4 | - | - | 1.47-1.67 | pick, pick2 |
| 32768 | 8 | 0.914-0.923 | 0.830-0.837 | WIN 4 | 0.762-0.767 | 0.792-0.798 | 8.81-9.03 | pick, pick2 |
| 32768 | 16 | 0.900-0.908 | 0.825-0.832 | WIN 4 | 0.747-0.749 | 0.771-0.779 | 4.92-5.04 | pick |
| 32768 | 32 | 0.877-0.887 | 0.776-0.780 | WIN 4 | 0.682-0.689 | 0.716-0.725 | 2.78-2.86 | pick |
| 32768 | 64 | 0.860-0.873 | 0.768-0.773 | WIN 4 | 0.665-0.672 | 0.679-0.691 | 1.89-1.93 | pick |
| 32768 | 128 | - | 0.781-0.799 | WIN 4 | - | - | 1.55-1.59 | pick |

`pick` / `pick2` are the two climb ends (FlyTape start and LB start), `pick3` the lookback's winning variant; the knobs
of each are in section 5.2. The two nodes agree within 0.02 on every best/ft ratio.

### 4.3 TP2: per-cell LB pick vs the port and vs FlyTape in situ (fix round: prebuilt, both nodes)

Jobs 68244896 `f2n01` (n01-09, bench bus 0x75) and 68244900 `f2n07` (n07-17, bench bus 0x05), q38 container, 2 GPUs:
FlyTape TP2 traced before and after the rocKE process (bracket spread 1.000-1.041), every timed arm a venv-built code
object loaded by name with `PL5_HSACO_STRICT=1` (1,056 / 1,056 prebuilt), each object (stream + phase) checked on random
data in the same job (section 2), 12 shuffled rounds, controls `ft2` / `ft_rc` (phase medians 0.987-1.022). Each cell:
min-max over the two nodes and zero / random data (4 cell-jobs).

| prompt | rows | ft/FT phase | best/ft phase | verdicts vs ft | best/FT phase | best/FT launch | best/floor | best arm(s) |
|---|---|---|---|---|---|---|---|---|
| 128 | 8 | 1.004-1.027 | 0.810-0.827 | WIN 4 | 0.826-0.835 | 0.969-0.990 | 64.99-66.14 | pick, pick2 |
| 128 | 16 | 1.030-1.038 | 0.798-0.810 | WIN 4 | 0.828-0.836 | 0.901-0.928 | 32.05-33.41 | pick |
| 128 | 32 | 0.979-1.021 | 0.811-0.839 | WIN 4 | 0.822-0.829 | 0.940-0.957 | 16.63-17.17 | pick, pick2 |
| 128 | 64 | 0.993-0.996 | 0.817-0.821 | WIN 4 | 0.813-0.818 | 0.939-0.949 | 8.49-8.68 | pick, pick2 |
| 128 | 128 | 0.954-0.966 | 0.821-0.837 | WIN 4 | 0.793-0.807 | 0.884-0.899 | 4.63-4.70 | pick, pick2 |
| 8192 | 8 | 0.926-0.936 | 0.845-0.851 | WIN 4 | 0.785-0.793 | 0.814-0.822 | 11.26-11.54 | pick |
| 8192 | 16 | 0.911-0.917 | 0.862-0.867 | WIN 4 | 0.786-0.791 | 0.808-0.819 | 5.94-6.08 | pick |
| 8192 | 32 | 0.875-0.899 | 0.851-0.858 | WIN 4 | 0.745-0.770 | 0.776-0.799 | 3.41-3.66 | pick, pick2 |
| 8192 | 64 | 0.810-0.847 | 0.867-0.884 | WIN 4 | 0.716-0.734 | 0.752-0.786 | 2.25-2.46 | pick |
| 8192 | 128 | 0.756-0.828 | 0.859-0.870 | WIN 4 | 0.655-0.711 | 0.671-0.733 | 1.65-1.82 | pick |
| 32768 | 8 | 0.906-0.913 | 0.835-0.840 | WIN 4 | 0.762-0.766 | 0.796-0.802 | 11.06-11.53 | pick, pick2 |
| 32768 | 16 | 0.890-0.906 | 0.826-0.833 | WIN 4 | 0.736-0.751 | 0.771-0.790 | 6.05-6.19 | pick, pick2 |
| 32768 | 32 | 0.886-0.904 | 0.815-0.827 | WIN 4 | 0.722-0.748 | 0.760-0.782 | 3.59-3.74 | pick |
| 32768 | 64 | 0.823-0.832 | 0.797-0.801 | WIN 4 | 0.656-0.663 | 0.672-0.680 | 2.38-2.47 | pick3 |
| 32768 | 128 | 0.825-0.845 | 0.720-0.731 | WIN 4 | 0.596-0.609 | 0.613-0.627 | 1.68-1.73 | pick |

The two nodes agree within 0.03 on every best / ft and within 0.05 on every best / FT (the widest: 8K rows 128, where
`ft / FT` itself moves 0.76-0.83 between nodes). The rows-128 32K pick is a `waves=8` build: venv-built, checked per cell
on both nodes, so the 0.60 point now stands on verified binaries.

**Withdrawn (first round):** the TP2 table of job 68242343 (c2n01: container-COMGR binaries, its own numerics FAILED,
14 of 30 cell-jobs' best arm a container `waves=8` build of the class that gave wrong output) and the derived cross-job
estimate. The port-relative n07-17 job 68243956 (venv binaries, best / ft 0.72-0.88, WIN 30 / 30) agrees with the fix
round's best / ft within 0.04 and is superseded by it.

### 4.4 Where a unit's time goes (stamps, informational, FlyTape-structure arm)

Fractions of the median unit span (stamped twins; the stamp is another binary, never timed):

| cells | heads + indexer + scan | select (incl. waiting for the last scanner and the release) | attention kv head 0 | kv head 1 (TP1) |
|---|---|---|---|---|
| p128, all rows | 0.16 (TP1) / 0.23 (TP2) | - | 0.42 / 0.77 | 0.41 / - |
| 8K, rows 8 | 0.08 / 0.12 | 0.46 / 0.51 | 0.20 / 0.31 | 0.21 |
| 8K, rows 128 | 0.16 / 0.26 | 0.06 / 0.10 | 0.38 / 0.62 | 0.39 |
| 32K, rows 8 | 0.12 / 0.17 | 0.54 / 0.59 | 0.16 / 0.21 | 0.16 |
| 32K, rows 128 | 0.35 / 0.47 | 0.10 / 0.14 | 0.27 / 0.37 | 0.27 |

- Inside the selecting CTA (TP2 rows 8, 8K; MI350X stamps): keys to LDS 0.04, **boundary search 0.21**, runs + block scan 0.04, list
  writes 0.05 of the unit; the rest of "select" is waiting for the slowest scanner (FlyTape's 256-block chunks leave 24
  of 32 members idle at rows 8) and the release poll. The bit search's cost is the per-key ballot + `s_bcnt1` chain,
  not the barriers: two- and four-bit variants with ballots (`bits2`, `nib`) lose 1.01-1.05; the same search with the
  counts on the VALU (`v1`, `v2`) wins 0.91-0.99.
- Attention of kv head 0 at p128 TP2 rows 128 (MI350X stamps): q operand loaded 0.07, stages 0.34, partial store +
  arrival 0.28, merge 0.05.
- **WaveScope** (MI350X, `att_job.sh`, CU 1): FlyTape structure at TP2 rows 128 8K: `vmem_load` issue stalls 40 % and
  `s_waitcnt vmcnt(0)` 28 % of stall cycles (the KV stream saturates the CU's memory pipe), barrier 9 %; the LDS merge
  store (`qsa_lb.py` 1594) stalls 2.7-12 % (16-way bank conflict, fixed by `mpad`). At rows 8 8K the barrier is 47 %
  of stall cycles (waves waiting while one CTA selects and wave 0 polls). After (the TP2 rows 128 32K pick): VALU
  latency share 38.8 % -> 17.5 % (MFMA scoring), executing 49.8 % -> 22.9 %, `s_waitcnt vmcnt(0)` 32.6 % -> 11.7 %.

## 5. Knob ledger

### 5.1 Single levers on the FlyTape base (climb step 0, phase metric, random data, 30 cells, both TPs)

| lever | p128 (10 cells) | 8K (10) | 32K (10) |
|---|---|---|---|
| `mpad` | 0.876-0.902 (W10) | 0.931-0.986 (W10) | 0.948-0.987 (W10) |
| `qloc` | 0.935-0.976 (W10) | 0.989-1.011 (W1/T8/L1) | 0.993-1.006 (T10) |
| `sel=v2` | - | 0.959-0.990 (W10) | 0.913-0.976 (W10) |
| `sel=v1` | - | 0.972-0.992 (W9/T1) | 0.930-0.982 (W10) |
| `score=mfma` + `kw16` | - | 0.954-0.992 (W9/T1) | 0.900-0.977 (W10) |
| `score=mfma` | - | 0.976-1.009 (W5/T4/L1) | 0.967-0.986 (W10) |
| `waves=8` (+ `pf=1`, `qloc`, `qlds`) | 0.983-1.008 (T3/L2, 5 cells compiled clean) | 0.975-1.164 (W2/T2/L4) | 0.873-1.054 (W7/T1/L1) |
| `kvcp=ntsc1` | 0.997-1.002 (T10) | 0.978-1.102 (W3/T3/L4) | 0.984-1.004 (W5/T5) |
| `sela` | - | 0.976-1.003 (W3/T7) | 0.982-1.004 (W1/T9) |
| `spf` | - | 0.989-1.048 (W2/T2/L6) | 0.905-1.026 (W4/T4/L2) |
| `scan_its=8` | - | 0.982-1.052 (W2/T3/L5) | 1.003-1.123 (T2/L8) |
| `scan_its=4` | - | 0.961-1.151 (W2/T1/L7) | 1.028-1.419 (L10) |
| `pf=1` | 1.003-1.051 (T2/L8) | 0.968-1.086 (W4/L6) | 0.974-1.049 (W3/T1/L6) |
| `gearly` | 1.005-1.034 (T2/L8) | 1.002-1.030 (T3/L7) | 1.003-1.017 (T3/L7) |
| `gang = 2 x FlyTape's` | 0.944-1.780 (W8 = rows 8-64 / L2 = rows 128) | 1.096-1.984 (L) | 1.138-1.937 (L) |
| `gang = FlyTape's / 2` | 1.076-1.139 (L10) | 0.981-1.528 (W1/L) | 1.051-1.593 (L) |
| `sel=nib` | - | 1.008-1.051 (L10) | 0.997-1.008 (T8/L2) |
| `sel=bits2` | - | 1.006-1.037 (T1/L9) | 1.009-1.039 (L10) |
| `sel=hist` | - | 1.099-1.360 (L10) | 1.043-1.181 (L10) |

Every losing lever stays a default-off knob, measured on the FlyTape base only; the climb re-tested every one of them at
every step on the new base (PLAN decisions 13 / 15) and several came back in combinations: `pf=1` (a solo loser) is in
57 of the 67 picks, `gearly` (a solo loser) in 27, `kvcp=ntsc1` (mixed alone) in 32, `mpad` in 65.

### 5.2 Per-cell picks (climb ends, both starts; lookback)

| cells | pick (knobs deviating from FlyTape's) |
|---|---|
| p128 TP1 rows 8-32 | `gang=4 pf=1 mpad qloc` (+`gearly` in `pick`; the lookback dropped it: `pick3`, 0.974-0.975 vs `pick` on its base) |
| p128 TP1 rows 64-128, TP2 rows 8-128 | `waves=8 pf=1 mpad qloc qlds` (+`gearly`); TP2 rows 16: `gang=4 pf=1 mpad gearly qloc` |
| 8K rows 8 | `sel=v2 pf=1 scan_its=4 mpad` (+`kvcp=ntsc1` TP1; + `gearly sela spf qloc` TP2); lookback TP1: + `sela spf` |
| 8K rows 16-32 | `kvcp=ntsc1 sel=v2 pf=1 score=mfma kw16 mpad gearly sela spf qloc` (TP1 rows 32: `waves=8 kvcp=ntsc1 pf=1 mpad qloc qlds`, lookback + `scan_its=4 sela`) |
| 8K rows 64-128 | `waves=8 kvcp=ntsc1 pf=1 score=mfma kw16 mpad qloc qlds` |
| 32K rows 8-16 | `sel=v2 pf=1 scan_its=8 score=mfma kw16 mpad (gearly) sela spf qloc (kvcp=ntsc1 TP1)` |
| 32K rows 32-128 | TP1: `waves=8 kvcp=ntsc1 sel=v1/v2 pf=1 (scan_its=8) score=mfma kw16 mpad (sela spf) qloc qlds`; TP2 rows 32-64: `sel=v2 score=mfma kw16 mpad gearly sela spf qloc`; TP2 rows 128: `waves=8 kvcp=ntsc1 sel=v1 pf=1 scan_its=8 score=mfma kw16 mpad spf qloc qlds` |

Exact knob sets: `hackathon-prep/qwen38/truemono-lb/p-l5/picks.json`; full ledgers (every candidate, its base, median
and verdict at every step): private `p-l5-raw/cl1b`, `cl2b`. Open (decision 15): the climbs that the lookback improved
(`pick3`) were not climbed again from `pick3`.

## 6. Budget update (DESIGN-LB 15)

- 15.2 row "QSA (proj, core, out)": **core measured, both TPs, both nodes, same-job FlyTape brackets.**
  - **Conservative figure (carry this until the DESIGN 16 IR diff explains `ft / FT` < 1): best LB / FlyTape structure
    in rocKE (`best / ft`, same harness, in-job win rule) = 0.72-0.91 at TP1, 0.72-0.88 at TP2** (p128 0.72-0.84, 8K
    0.82-0.91, 32K 0.72-0.84).
  - Against FlyTape in situ (`best / FT`, phase metric): TP1 p128 0.72-0.79, 8K 0.70-0.82, 32K 0.67-0.77; TP2 p128
    0.79-0.84, 8K 0.66-0.79, 32K 0.60-0.77. This includes the unexplained port gap (up to 0.25 at 8K / 32K).
  - TP1 rows 128 (no FlyTape bar): 0.78-0.91 of the FlyTape structure in rocKE.
- FlyTape's `qsa_core` share of its traced step (same jobs): 0.015-0.043 at p128, 0.064-0.117 at 8K / 32K (largest at
  TP2 rows 128 32K). The pick saves, of FlyTape's step, **0.002-0.012 at p128 and 0.007-0.033 at 8K / 32K on the
  conservative figure**, or 0.003-0.012 and 0.012-0.047 against FlyTape in situ.
- The QSA family line of 15.3 (taken at 0.6-1.2 with the core unknown) can move to 0.6-0.85 once the q-proj / out-proj
  numbers of P-L3 (`qsa_in` 0.48-0.95) are combined; this probe does not re-measure the projections.
- The `stream` metric (no barrier between layers) is 0.06-0.47 of FlyTape's phase at p128 and 0.65-0.83 at 8K / 32K:
  what a dataflow pass gains if neighbouring stages fill the QSA core's latency chain, an upper bound that WP-L3 must
  measure in situ.

## 7. Lessons

1. **A faithful port first pays.** The rocKE twin of FlyTape's `qsa_core` was at parity with FlyTape's own phase at p128,
   so every later ratio is against a known-equivalent structure, and the levers can be attributed.
2. **Latency chains, not bytes**: at p128 the best kernel is still 4-66x the bytes floor; at 32K rows 128 it is 1.6-1.7x.
   Removing round trips (q via LDS, no release handshake, merges without bank conflicts) and serial single-CTA work (the
   select) is where the time is.
3. **Count on the VALU**: the select's ballot + `s_bcnt1` per key per round is SALU-latency bound; per-lane
   compare-adds and one DPP butterfly per round win (`v1` / `v2`), while fewer rounds with more ballots (`bits2`,
   `nib`) lose.
4. **LDS bank conflicts hide in "FlyTape's layout"**: the `[wave][16][256]` f32 merge puts the 16 heads of a lane group
   in one bank; a 4-word pad wins in every cell.
5. **Two COMGRs, two kernels**: the container COMGR's 512-thread builds were wrong with and without spills (the
   first round wrongly blamed the spills and trusted a scratch == 0 gate); never time a container-COMGR build, load the
   exact venv-built code objects by name in strict mode, and check every timed object (stream and phase twins) on
   random data in the timing job itself. A zero-data check proves nothing.
6. **Solo losers matter** (decision 13): `pf=1` and `gearly` lose alone and appear in most winning combinations.

## 8. Open items

- Continue the climbs from the lookback winners (`pick3`) and re-run the lookback (decision 15).
- `waves=8` failed the spill gate in 6 of 30 cells on the venv COMGR (p128 TP1 rows 8-32, some 8K rows): a register-lean
  512-thread variant (V through LDS by DMA, `kvdma`) is the next lever; not built.
- Paged KV (ATOM's block-16 layout, `kvl=page16`): not built; the port uses FlyTape's row-contiguous cache. WP-L2e needs
  it (one extra table load per token).
- DESIGN 16 reverse step: diff FlyTape's `qsa_core` LLVM IR / ISA against the port to explain `ft/FT` < 1 at 8K / 32K
  (0.76-0.94). Until it does, the budget carries `best / ft` (section 6).
- Accuracy: `score=mfma` (bf16 indexer query) is a tolerance-class change of the selection (N10); RULER gating before it
  can be a default.

## Verification

Independent verifier (W7), 2026-11-02. Inputs: this file, `qsa_lb.py` / `qsa_ref.py`, the harness
`hackathon-prep/qwen38/truemono-lb/p-l5/`, every raw `pl5.json` in the private `p-l5-raw/` and the job logs. The verifier's
own harness is `hackathon-prep/qwen38/truemono-lb/vP-L5/`. It reads the existing code objects and edits none of the
builder's files.

**Verdict: PARTIAL.** The TP1 result stands. The TP2-vs-FlyTape result does not stand as stated: its gate was not
exercised on both nodes, and its timed binaries come from a job whose own numerics failed.

### V1. What holds (re-checked)

1. **CPU tests:** `test_qwen38_lb_qsa.py` re-run locally: 22 pass.
2. **Hazard scan is a working gate:** compiled locally, `tp2 batch=8 ctxcap=8256` with `hz=0` gives 10 hazard hits, and
   with `hz` on (the default) gives 0. This is the must-fail control for the scan; the report did not have one.
3. **Numerics phase on MI355X:** re-read from the raw records of c1n01p (n01-09), c2n07v (n07-17) and c2n01 (n01-09).
   - The 8 source mutants plus their 2 path variants (`scan_mf`, `merge_2lvl`) are each caught in 3-6 of 6 cases.
     `tie`, `iqrope` and `scan` miss only where they cannot act: the dense case, which has no selection.
   - The stale-read twins (`lb_stale`, `stale_plain`) are caught in 5 of 6 cases, and their `sc1` controls are bitwise
     equal.
   - The timeout control sets the launch-fatal word in every sparse case.
   - `numerics_pass` is True for c1n01p and c2n07v and **False for c2n01** (see V2.2). "10 mutants" means 8 distinct
     source mutations plus 2 re-runs on other paths.
4. **TP1 timing:**
   - c1n01p (n01-09, bus 101) and c1n07p (n07-17, bus 101) used the same prebuilt code objects.
   - FlyTape was traced before and after in the same job; the bracket spread was 1.001-1.022.
   - 12 rounds, with arms and metrics shuffled every round.
   - A-vs-A controls (`ft2`, recompiled `ft_rc`): medians 0.979-1.020.
   - The pick wins against the port in 30 of 30 cell-jobs on each node, by DESIGN 9.1's rule.
   - The 4.2 table reproduces from the raw JSON with `pl5_summ.py`.
5. **TP2 against the port, both nodes:** c2n01 and c2n07v each give a WIN in 30 of 30 cells. Their controls were
   0.974-1.033 and 0.976-1.003.

### V2. Refuted or not exercised

1. **"Every timed binary passed its own check" is false as built.**
   - `pl5_bench.time_cell` checks only the `walk=stream` build of each arm (`if a.knobs.walk == "stream"`).
   - The `walk=phase` twin is a separate code object, and it carries the primary metric behind every `best / FT` ratio.
     No verdict job checked it.
   - Its codegen does differ: `w8q@ph` spills (8-16 B of scratch) where `w8q` does not.
   - **The verifier's check closes this for the venv binaries.** Markham MI350X, jobs 68244451, 68244452 and 68244494
     (`vP-L5/vpl5_check.py`, stages `w7vpl5-*`), triage-class hardware:
     - Every confirmation arm (`ft`, `lb`, `lbmf`, `w8q`, `pick`, `pick2`, `pick3`) was run from the builder's
       prebuilt stage (`w7-pl5-hsaco`, loaded by name, 0 of 354 rebuilt), in both twins.
     - That is 354 code objects over the 30 cells, TP1 and TP2.
     - Each object passed every stage check against `qsa_ref` on random data (2 layers, 5 rows), was bitwise
       deterministic over 5 launches, and had stream and phase twins bitwise equal: 177 of 177 pairs.
     - The same run on MI355X (q38 container job `w7-vP-L5-x1`) waited about 1.5 h on the q38 budget (9-11 of 8 GPUs
       held) and was withdrawn. It never ran.
2. **The TP2 FlyTape table (4.3, job c2n01) rests on unchecked container binaries, and the stated explanation is
   contradicted by the job's own data.**
   - c2n01's numerics phase FAILED (`numerics_pass` False). Its container `waves=8` builds gave core errors of 0.36-1.6
     and were nondeterministic in the `tp2_dense`, `tp2_bound` and `tp2_32k` cases.
   - The failures do **not** follow spills:

     | arm | scratch | result |
     |---|---|---|
     | `w8` (pf1) | 0 | wrong |
     | `w8g64` | 0 | wrong |
     | `w8mf` | 0 | wrong |
     | `w8_hist` (32K) | 0 | wrong |
     | `w8_pf2` | 132 | correct |
     | `w8g8` | 56 | correct |
     | `w8g16` | 48 | correct |
     | `w8` at rows 40 | 24 | correct |

   - So the scratch == 0 gate does not protect the container binaries that were timed. Section 2.1 and alola rule 209
     say "only the container's spilled builds were wrong"; that claim is wrong.
   - The venv `waves=8` builds pass with and without spills: c1n01p, c2n07v, and the verifier's 5-launch run above.
   - c2n01 had no per-cell check of its timed binaries. In 14 of its 30 cell-jobs the best arm is a container `waves=8`
     build (classified with the current `picks.json`): p128 rows 8 / 32 / 64 / 128, 8K rows 64 / 128, 32K rows 128.
   - Excluding those arms, the best TP2 arm over FlyTape in situ is 0.75-0.94 (p128), 0.67-0.80 (8K) and 0.66-0.76
     (32K). The section 0 item 5 / section 6 figure of 0.61-0.80 is not supported.
   - "TP2 beats FlyTape" survives on the arms that were not implicated, but on one node and on binaries never checked
     per cell.
3. **TP2 vs FlyTape on n07-17 was never exercised.** By DESIGN 9.1 (both nodes) and DESIGN-LB 15.6 (same-job FlyTape
   bracket), the TP2 side of the gate is open, not met. The report itself lists this as a gap, yet still says "gate met".
4. **The zero-data per-cell checks are vacuous.** Every error is exactly 0, because all-zero inputs give all-zero
   outputs. Only the random-data checks carry information. They passed: 15 cells per job, max core error 5.4e-3.
5. **`ft / FT` < 1 is unexplained.** The port measures 0.75-0.95 of FlyTape's own phase at 8K / 32K.
   - `best / FT` therefore credits LB with up to 25 % that is not a lever.
   - Until DESIGN 16's IR diff explains the gap, the conservative QSA-core budget figure is the same-harness `best / ft`.
     That is 0.72-0.90 at TP1 on both nodes, and 0.70-0.89 at TP2 on both nodes.
   - The 15.2 / 15.3 budget update should carry that range next to 0.61-0.81.

### V3. What the builder must do before "gate met" at TP2

- Run the TP2 FlyTape bracket on both nodes with the prebuilt venv code objects (`PL5_HSACO_DIR`), as the TP1 jobs did.
- Add the phase twin to the timed-binary check in `time_cell`: check every arm, not only `walk=stream`.
- Correct section 2.1 and rule 209: wrong container `waves=8` outputs occur with and without spills. Container-COMGR
  `waves=8` builds should not be timed at all.
- Report the TP2 `best / FT` from that bracket. Until then, the TP2 entry is port-relative only: 0.70-0.89, a WIN in
  30 of 30 cells on both nodes.

## Fix round

Builder (W7), 2026-11-02. Every Verification issue, what was done, and the evidence. Harness changes are in
`hackathon-prep/qwen38/truemono-lb/p-l5/` only (`pl5_bench.py`: `check_cell`, `--check-cells`, `PL5_HSACO_STRICT`;
`mk_job.sh`: `MK_H` override, default unchanged; new `fixjobs.sh`). No kernel source changed (`qsa_lb.py` /
`qsa_ref.py` / `lbf_hazard.py` sha256 identical to the climbed and prebuilt tree, checked against both cluster stages).

| # | issue (V2 / V3) | fix | evidence |
|---|---|---|---|
| 1 | `time_cell` checked only the stream build; the phase twin (primary metric) was never checked | `check_cell` checks every timed arm's stream build AND `walk=phase` twin, on random data at the cell's exact shape, layers 0-1 (the phase twin's grid sync between layers is crossed), rows 0 / B/2 / B-1, every stage vs `qsa_ref`, fatal word 0, 2 launches bitwise, stream / phase twins bitwise equal; an arm failing any of it is dropped from timing. `--check-cells` runs it on cells not timed in the job | MI355X, both nodes: 207 / 207 arm pairs over 30 cells PASS per job (TP2: the 15 timed cells; TP1: the 15 cells' code objects of c1n01p / c1n07p), 207 / 207 twin pairs bitwise equal, max core error 4.9e-3 (jobs 68244896, 68244900). Closes the MI355X side the verifier could not run |
| 2 | TP2 FlyTape table from c2n01 (numerics FAILED, container binaries never checked per cell) | Table withdrawn. New TP2 brackets on both nodes with the prebuilt venv code objects only | 4.3; numerics overall PASS in both jobs (all mutants caught, all non-mutant arms PASS and deterministic, groups bitwise equal) |
| 3 | 2.1 / rule 209 wrong: container `waves=8` wrong with and without spills | 2.1 rewritten; `PL5_HSACO_STRICT=1` makes a missing prebuilt object a refused build instead of a container compile; rule 209 marked corrected, rule 219 added | strict mode must-fail: with an empty `PL5_HSACO_DIR` the build is refused (`not prebuilt (PL5_HSACO_STRICT)`); both fix jobs: 1,056 / 1,056 builds prebuilt, 0 container compiles |
| 4 | 14 of 30 TP2 best arms were suspect container `waves=8` builds | Re-measured with verified venv builds | TP2 best / FT 0.79-0.84 (p128), 0.66-0.79 (8K), 0.60-0.77 (32K), both nodes. The 32K rows-128 point (0.60) is again a `waves=8` pick, now venv-built and checked per cell. The verifier's range without the suspect arms (0.75-0.94 / 0.67-0.80 / 0.66-0.76) still came from container builds; on verified binaries the p128 values are higher than first claimed (0.79-0.84 vs 0.75-0.79), so the first round's p128 TP2 claim was optimistic |
| 5 | TP2 vs FlyTape on n07-17 never ran; "gate met" overstated | Both 2-GPU brackets ran (each started about 35 min after submission) | f2n07: FlyTape before / after spread 1.000-1.021, pick WIN vs port in 30 / 30 cell-jobs, controls 0.987-1.022; f2n01 the same (spread 1.002-1.041, controls 0.990-1.022). Gate now met at TP2 on both nodes |
| 6 | zero-data per-cell checks vacuous | The per-cell check is always on random data (seed 13), independent of the timing data; checker must-fail controls per cell: the restored buffers without a launch, and the `ft` output with one checked row scaled by 1.05, must both be caught, else every arm of the cell counts as failed | controls caught in 30 / 30 cells per job and in the Markham smoke (68244803, 5 cells, 35 / 35 arms) |
| 7 | `best / FT` credits up to 0.25 of an unexplained port gap | Section 6 and summary item 7 carry the port-relative `best / ft` as the conservative budget figure: 0.72-0.91 (TP1), 0.72-0.88 (TP2), savings 0.002-0.012 of FlyTape's step at p128 and 0.007-0.033 at 8K / 32K; 4.1 no longer explains the gap, it lists untested candidates | section 6; DESIGN 16 IR diff stays open (section 8) |
| 8 | "10 must-fail mutants" | Worded as 8 distinct source mutants plus 2 re-runs on other code paths | summary item 2, section 2 table |

Jobs (MI355X, `rocke` partition, q38 container, 2 GPUs each, about 40 min of wall time, 35 min of it FlyTape's cold
traced pass): 68244896 `w7-pl5-f2n01` (n01-09, bench GPU bus 0x75, FlyTape TP2 on 0x75 + 0x15) and 68244900
`w7-pl5-f2n07` (n07-17, bench GPU bus 0x05, FlyTape TP2 on 0x05 + 0x15; the faulty bus e5 was not allocated, health
check passed on both GPUs). Arms and metrics shuffled every round, 12 rounds, zero and random data. Markham MI350X smoke
of the new check before submission: 68244757 (argument typo, no GPU work) and 68244803 (PASS). Absolute times:
private `P-L5.md` "Fix round", raw `p-l5-raw/f2n01`, `f2n07`, `fix_tp1.json`, `fix_tp2.json`.

Still open (unchanged by this round, section 8): climbs from `pick3`, the register-lean 512-thread variant, paged KV,
the DESIGN 16 IR diff, RULER gating of `score=mfma`.

## Verification 2 (W10, of the fix round)

Independent verifier (W10 reverify-LB), 2026-11-02. Inputs: the fix round above, `pl5_bench.py` (stage `w7-pl5-hf`,
sha `acac6e3c...`, the same as the local file), the raw `p-l5-raw/f2n01`, `f2n07`, `c1n01p` and `c1n07p`, and the job logs.
Hardware reruns: MI355X, q38 container, on frozen copies of the builder's stages (`w10-pl5-rocke` = `w7-pl5-rockee`,
`w10-pl5-h` = `w7-pl5-hf`, `w10-pl5-hsaco` = `w7-pl5-hsaco`). Harness: `hackathon-prep/qwen38/truemono-lb/w10-reverify/`
(`job.sh`, `w10_pl5.py`, `sens_picks.json`). The verifier edited none of the builder's files. Absolute times are in
`private-perf/.../large-batch/w10-raw/`.

**Verdict: CONFIRMED.** Every fix-round claim checked holds. The per-cell check also catches the real container
miscompile: it is a working gate. Two process gaps remain (V2.3).

### V2.1 Re-derived from the raw data

1. **Table 4.3.** `pl5_tables.py` on `f2n01` + `f2n07` gives every row of 4.3: ft / FT, best / ft, the verdicts (WIN in
   60 / 60 cell-jobs), best / FT for the phase and launch metrics, and the floor columns.
   - The `ft2` / `ft_rc` phase controls are 0.987-1.022 (`ft_rc` 0.990-1.018, `ft2` 0.987-1.022).
   - The FlyTape bracket spread is 1.000-1.041.
   - Summary item 5 / section 6: TP2 best / ft 0.72-0.88, best / FT 0.60-0.84, launch 0.61-0.99. All match.
2. **Strict mode held.** Both jobs loaded 1,056 / 1,056 builds from `w7-pl5-hsaco`, with 0 compile errors.
   - The only gate refusals are `w8q@ph` at TP2 rows 16, 8K / 32K (16 B scratch). That arm then drops out of those cells.
   - `cellcheck_pass` and `numerics_pass` are True in both jobs. All 12 must-fail entries were caught, `m_tie` in 3 cases.
3. **TP1 coverage is complete.** The TP1 timing jobs `c1n01p` / `c1n07p` predate strict mode. Every one of their 804 / 268
   builds came from the same `w7-pl5-hsaco/hsaco` directory (`prebuilt` field). Their 246 timed TP1 cell objects (no
   mutants or stress builds) are the same set, by build key, as the 246 that the fix jobs checked with `--check-cells`. So
   every timed TP1 object has a per-cell check on MI355X.
4. **No kernel source changed.** `qsa_lb.py` / `qsa_ref.py` / `lbf_hazard.py` have the same sha in the local tree, in
   `w7-pl5-rocked` / `rockee`, and in the verifier's job log.

### V2.2 Hardware reruns (jobs 68245314 `w10-v5-a` and 68245397 `w10-v4b-n01`, both n01-09)

1. **Independent check of every timed object (strict, prebuilt, MI355X).**
   - `check_cell` was re-run on all 30 cells. Its source was patched by `w10_pl5.py`: each anchor is asserted to occur
     exactly once, and the rest runs verbatim.
   - The patch changes three things: the seed (29 instead of 13); the rows (8-10 per cell, 0 and B-1 plus random ones,
     instead of 0 / B/2 / B-1); and the positions, which differ per row (pos - (37 r mod min(512, pos)); at the
     tp2:8:2048 sensitivity cell this mixes dense and sparse rows) instead of being one position for all rows.
   - Result: 207 / 207 arm pairs PASS, all twins bitwise equal, max core error 5.3e-3, checker controls caught in 30 / 30
     cells.
2. **Port-relative timing reproduces** (same job, 12 shuffled rounds, zero and random data). At TP2 32K rows 128 (the
   `waves=8` 0.60 point), 8K rows 128 and p128 rows 8:
   - best / ft = 0.719-0.722 / 0.866-0.879 / 0.822-0.827, WIN in 6 / 6 cell-jobs, controls 0.997-1.023;
   - the fix round on the same node: 0.720-0.721 / 0.868-0.870 / 0.815-0.820.
3. **Must-fail test of the fix round's per-cell check against the real container miscompile.** Container-COMGR builds
   (no prebuilt directory):
   - The numerics phase FAILS again, as in c2n01: `w8`, `w8g64`, `w8lb`, `w8mf` (and `w8_hist` at 32K) give core errors
     of about 1 and are nondeterministic in `tp2_dense`, `tp2_bound` and `tp2_32k`.
   - The builder's unmodified `check_cell` at the same code objects (cells tp2:8:2048, tp2:8:8192 and tp2:4:32768,
     `sens_picks.json` carrying those arms' exact knobs) FAILS every one of them: 13 of 13 (cell, arm) pairs dropped, max
     core error 0.85-1.07. The patched check also fails 13 of 13 (max 1.17-1.54).
   - `w8q` and the A-vs-A arms pass in every cell. **The fix round's gate is sensitive to the failure it was built
     for.**
4. **A narrower bad class than V2.2 of the first verification assumed.** At the 7 TP2 timing cells where c2n01's best
   arm was a container `waves=8` build, the container builds of every timed arm pass `check_cell`: 45 / 45, including
   the `waves=8` picks (`pf=1 qloc qlds`, with or without `mpad`). In this container the wrong class is `waves=8` with
   `qloc=0`, or with `sela + spf` (`w8lb`).
   - Withdrawing c2n01 was still right (rules 217 / 219: never time a container build).
   - But the first-round TP2 numbers were most likely slower code, not wrong outputs: p128 0.75-0.79 then, 0.79-0.84 on
     venv binaries now.

### V2.3 Open (not fixed here: builder scope)

1. **DESIGN 8.1 neighbours were not recorded in the fix round.** `sacct -a` over the 1-minute rocKE window of each job:
   - f2n01 (03:53-03:54 UTC) shared n01-09 with 4 jobs of ours: `w9-n4a-r2`, `lb2-u6`, `lb2-u8` and `w8-wp2b-n2n01`. 8.1
     allows one.
   - f2n07 (03:55-03:56 UTC) had 1 (`lb2-u7`).
   - The two nodes agree within 0.03 on best / ft, and the n07-17 job is 8.1-compliant, so the verdicts stand. Record the
     neighbours in the report.
   - The verifier's own job 68245314 also ran with 5-6 jobs of ours on n01-09.
2. Unchanged from section 8: the unexplained `ft / FT` < 1 at 8K / 32K (DESIGN 16 IR diff), climbs from `pick3`, paged
   KV and RULER gating of `score=mfma`.
