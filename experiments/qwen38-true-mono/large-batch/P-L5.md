# P-L5: the large-batch QSA core (sparse attention) at rows 8-128, TP1 and TP2, contexts 128 / 8K / 32K (gfx950)

Probe P-L5 of `../DESIGN-LB.md` 15.4 / 15.6: "QSA core at LB (row-gang, MFMA QK / PV, indexer, select) at p128 and 8K
vs FlyTape's QSA phases; the last unmeasured family". Workflow W7, 2026-11-01, alola `rocke` partition (MI355X, 256 CUs)
for every verdict, Markham MI350X for triage only (numerics, phase stamps, WaveScope). This file has ratios only;
absolute times are in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/P-L5.md`.

Status: **done. Gate met.** Numerics passed on MI355X on every timed binary (every stage against a float64 host reference,
must-fail mutants all caught, stale-read twin caught, bounded-wait timeout control fires). The FlyTape-structure port
reproduces FlyTape's in-situ `qsa_core` phase, and the per-cell LB pick beats both, with the win rule, in every cell on
both nodes. TP2 on n07-17 with the prebuilt binaries: see 4.3 (state at the end of the file).

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
   2050-2055, 8K, 32K, block-completing and ring steps mixed in one launch, 2 layers), about 50 arms each, plus a per-cell
   check of every timed binary in every timing job (layer 0, 3 rows): all PASS on MI355X; 10 must-fail mutants caught;
   the 2 stale-read twins (`acq=plain` + pretouch / delayed publish) caught in 3-5 cases per job; order-keeping knob groups
   bitwise equal; 2 launches bitwise equal.
3. **The port is faithful, and slightly faster than FlyTape in situ** (phase metric, same job, FlyTape bracketed before
   and after): FlyTape structure in rocKE / FlyTape's traced `qsa_core` = **0.97-1.02 at p128, 0.82-0.94 at 8K,
   0.86-0.92 at 32K** (TP1, both nodes; TP2 n01-09, rows 8-128: 0.91-0.99 p128, 0.75-0.95 8K, 0.84-0.92 32K). The gap grows with
   bandwidth demand (rows x context), so it is mostly not structure; DESIGN 16's IR diff of FlyTape's own build would
   tell (open item).
4. **The QSA core is a latency chain, not a bandwidth stream** (stamps, WaveScope; section 4.4): one unit takes almost
   the whole phase. At p128 a unit is heads prep (0.16-0.23) + attention of 2-3 stages and the gang merge (0.77-0.83);
   at 8K / 32K the single-CTA select dominates at small rows (0.46-0.59 of a unit at rows 8) and the scan at rows 128
   (0.35-0.47). Even the best builds sit 1.5-1.8x the bytes floor at rows 128 long context and 4-63x at p128.
5. **Per-cell LB picks** (a hill climb with lookback over every knob, then a same-job FlyTape-bracketed confirmation on
   both nodes, section 4): best / FlyTape-structure-in-rocKE = **0.72-0.74 (p128, rows 8-32), 0.79-0.84 (p128, rows
   64-128), 0.82-0.91 (8K), 0.77-0.84 (32K)**, WIN in every TP1 cell-job; best / FlyTape in situ = **0.72-0.79 (p128),
   0.70-0.81 (8K), 0.67-0.77 (32K)** at TP1, both nodes, zero and random data. TP2 (n01-09): 0.75-0.79 (p128),
   0.67-0.80 (8K), 0.61-0.76 (32K). The launch metric (one launch per layer) gives 0.79-0.88 / 0.72-0.85 / 0.68-0.80
   of FlyTape in situ at TP1.
6. **The levers that carry it** (single-lever ledger on the FlyTape base, 30 cells, section 5): `mpad` (the waves' LDS
   merge with a padded row: FlyTape's 256-word rows put 16 head rows of a lane group in one bank) WIN in 30 / 30, 0.88-0.99;
   `qloc` (q kept in LDS on the dense path) WIN 10 / 10 at p128, 0.94-0.98; `sel=v1 / v2` (FlyTape's bit search with the
   counts done on the VALU instead of a ballot + `s_bcnt1` per key, the same selection) WIN 39 / 40 long-context cells,
   0.91-0.99; `score=mfma` + `kw16` (indexer scores on MFMA, bf16 query: tolerance class) WIN 19 / 20, 0.90-0.99;
   `waves=8` with `qlds` (512-thread CTA, 128-token stage) WIN 9 of 20 long-context cells, up to 0.87; `gang=4` at p128
   rows 8-64 (twice FlyTape's 2) WIN 8 / 8 (it loses at rows 128, where it needs two unit rounds). `pf=1` loses alone
   (0.97-1.09) but is in 57 of the 67 picks (PLAN decision 13).
7. **Budget consequence** (section 6): FlyTape's `qsa_core` is 0.015-0.043 of its step at p128 and 0.065-0.115 at 8K /
   32K. At the measured pick ratios the QSA core costs 0.67-0.80 of FlyTape's phase, i.e. it saves 0.004-0.012 of
   FlyTape's step at p128 and 0.014-0.038 at 8K / 32K. DESIGN-LB 15.3 took the QSA family at 0.6-1.2 with the core
   unmeasured; the core is now measured at 0.61-0.81 (all cells, both TPs).

## 1. The kernel (`qsa_lb.py`)

| piece | FlyTape (`bodies_qsa.py` / `bodies.py`) | rocKE port (`ft`, every knob at its default) | LB levers (knobs) |
|---|---|---|---|
| unit | (row, member), gang = `gang_size` (power of two <= 32 dividing the grid, at most the 64-token rounds of the context) | identical; units (layer, row, member), layers runtime | `gang` (any power of two <= 64) |
| CTA | 256 threads, 64-token stage (16 per wave) | identical | `waves=8`: 512 threads, 128-token stage (needs `pf=1` + `qloc` + `qlds` to fit 256 VGPRs) |
| heads prep | every member preps every head within the budget (identical stores); past it each member preps `member + i * gang` | identical | `qloc`: q to LDS (dense rows skip the global round trip); `qlds`: S MFMAs read q from LDS each stage |
| indexer | wave 0 appends to the ring or completes a block (mean, norm, RoPE at block start, E4M3 + scale); the block completed past the budget is scored by member 0 | identical | - |
| scan | 16-lane rows, 8 dims per lane, f32 FMAs, row reduce, ReLU sum over 4 heads; chunks of 256 blocks dealt over the gang | identical | `scan_its` (chunk 2-16 x CTA/16 blocks), `spf` (next chunk in flight), `score=mfma` (16x16x32 bf16, keys E4M3 -> bf16 exact, query bf16), `kw16` (32-byte key loads, K order permuted in both operands) |
| select | last arriver: keys = f32 bits 30..15 in LDS, 16 single-bit rounds (ballot + popcount), runs + block scan, ascending list; release flag | identical | `sel`: `v1` / `v2` (VALU counts, 1 / 2 bits per round), `bits2` (2 bits, ballots), `nib` (4-bit digits, packed counters), `hist` (byte histograms): all give the same boundary; `sela` (every member selects, list in LDS, no release handshake) |
| attention | 8 x 16x32x32 S MFMAs, wave-private online softmax (exp2 domain, scale log2e/16), 16 x 16x16x16 PV MFMAs, V transposed through LDS (`ds_read_b64_tr_b16`), next stage prefetched | identical (unrolled by 2 so the prefetch never copies at the back edge) | `pf=1` (load per stage), `kvcp=ntsc1` (nt on the KV loads), `gearly` (gate loads before the stages) |
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
| must-fail mutants | each caught in >= 1 case | all 10 caught (tie in 2-3 cases, the others in 3-6) |
| stale-read twin (`acq=plain`, member 0 delays its publish, the others pretouch) | reported | caught in 3-5 cases per job; `stress` with `sc1` stays bitwise equal |
| timeout control (stress 200 + 1 poll iteration) | the launch-fatal word must be set | set in every sparse case |
| timed binaries in the timing jobs | every arm, layer 0, rows 0 / B/2 / B-1, every cell, zero and random data | 30 / 30 cells PASS per job (TP1 both nodes) |

### 2.1 The container COMGR finding

The FlyTape brackets run in the q38 container (ROCm 10 COMGR); the climbs ran in the venv (torch 2.12 / ROCm 7.2
COMGR). The container COMGR spilled the 512-thread builds (`waves=8`) that the venv COMGR compiled spill-free, and those
spilled container binaries produced **wrong, nondeterministic outputs** (core error about 1.0, two launches different;
`numerics` caught it in job c2n01). They are refused by the compile gate in timing anyway (scratch must be 0). The
confirmation jobs therefore run the venv-built code objects inside the container (`prebuild.sh` in a GPU-less venv job,
shipped as a q38 stage, loaded by kernel name: `PL5_HSACO_DIR`), so the confirmed binaries are exactly the climbed ones,
and their numerics are checked again in the same job. Recorded as alola rule 209.

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
  **68242876 (TP1, n07-17 bus 101, prebuilt)**, TP2 prebuilt n07-17 / n01-09: section 4.3. Node + bus per row in the
  private tables; the bad n07-17 GPU (bus e5) was skipped by the health check. MI350X triage: 68242054, 68242147,
  68242062, 68242105, 68242162, 68242203 (numerics + stamps; 68242054 failed on an argument typo), 68242252 / 68242418
  (WaveScope).

## 4. Results

### 4.1 Port fidelity: the FlyTape structure in rocKE vs FlyTape in situ (phase metric)

| cells | ft / FT (TP1 rows 8-64, n01-09 + n07-17, zero + random) | ft / FT (TP2 rows 8-128, n01-09, 2 jobs) |
|---|---|---|
| p128 | 0.974-1.016 | 0.912-0.989 |
| 8K | 0.818-0.942 | 0.753-0.946 |
| 32K | 0.860-0.923 | 0.844-0.919 |

- At p128 (latency chain, little data) the port is at parity, which says the structure was reproduced faithfully: same
  units, same waits, same merges. The gap grows with rows and context (more bytes per unit), so the difference is mostly
  in how the bytes move (code generation, cache bits), not in the structure; FlyTape's in-situ phase also starts right
  after its `qsa_proj` phase, with other phases' traffic just finished.
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

### 4.3 TP2: per-cell LB pick (n01-09; n07-17 prebuilt)

Job 68242343 (n01-09, bus 133): container-COMGR binaries (the picks that spilled under that COMGR were refused, so a
cell's best is among the arms that compiled clean there). This job's own numerics (same container binaries) passed for
every build that did not spill; its spilled `waves=8` numerics arms failed (2.1) and spilled builds are never timed. It
predates the per-cell timed-binary check. Each cell: min-max over zero / random data.

| prompt | rows | ft/FT phase | best/ft phase | verdicts vs ft | best/FT phase | best/FT launch | best/floor | best arm(s) |
|---|---|---|---|---|---|---|---|---|
| 128 | 8 | 0.961-0.985 | 0.782-0.815 | WIN 2 | 0.771-0.784 | 0.876-0.877 | 62.00-62.98 | pick |
| 128 | 16 | 0.981-0.983 | 0.761-0.767 | WIN 2 | 0.749-0.753 | 0.851-0.868 | 29.83-29.94 | pick |
| 128 | 32 | 0.978-0.989 | 0.791-0.803 | WIN 2 | 0.782-0.785 | 0.886-0.903 | 15.75-15.83 | pick, pick2 |
| 128 | 64 | 0.944-0.946 | 0.816-0.821 | WIN 2 | 0.770-0.776 | 0.850-0.869 | 8.15-8.24 | pick2 |
| 128 | 128 | 0.912-0.934 | 0.807-0.824 | WIN 2 | 0.752-0.754 | 0.823-0.826 | 4.43-4.47 | pick2 |
| 8192 | 8 | 0.946 | 0.827-0.831 | WIN 2 | 0.783-0.786 | 0.823-0.828 | 11.39-11.50 | pick |
| 8192 | 16 | 0.918-0.921 | 0.871 | WIN 2 | 0.800-0.803 | 0.817-0.824 | 6.11-6.13 | pick |
| 8192 | 32 | 0.862-0.893 | 0.859-0.866 | WIN 2 | 0.741-0.773 | 0.758-0.798 | 3.53-3.62 | pick |
| 8192 | 64 | 0.822-0.859 | 0.876-0.879 | WIN 2 | 0.723-0.752 | 0.740-0.769 | 2.33-2.46 | pick |
| 8192 | 128 | 0.775-0.776 | 0.865-0.868 | WIN 2 | 0.670-0.674 | 0.685-0.686 | 1.76-1.84 | pick |
| 32768 | 8 | 0.916-0.919 | 0.828-0.830 | WIN 2 | 0.758-0.762 | 0.790-0.793 | 11.16-11.32 | pick |
| 32768 | 16 | 0.901-0.905 | 0.826-0.830 | WIN 2 | 0.748 | 0.785-0.792 | 6.02-6.09 | pick2 |
| 32768 | 32 | 0.901-0.902 | 0.821-0.823 | WIN 2 | 0.740-0.743 | 0.769-0.771 | 3.67-3.68 | pick |
| 32768 | 64 | 0.848-0.856 | 0.770-0.774 | WIN 2 | 0.656-0.659 | 0.675-0.678 | 2.36-2.37 | pick3 |
| 32768 | 128 | 0.858-0.871 | 0.704-0.707 | WIN 2 | 0.606-0.614 | 0.615-0.621 | 1.67-1.71 | pick |


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

- 15.2 row "QSA (proj, core, out)": **core measured**. Best LB / FlyTape in-situ `qsa_core` (phase metric, both nodes at
  TP1, n01-09 at TP2): p128 0.72-0.79, 8K 0.67-0.81, 32K 0.61-0.77; TP1 rows 128 (no FlyTape bar) 0.78-0.91 of the
  FlyTape structure in rocKE.
- FlyTape's `qsa_core` share of its traced step (same jobs): 0.015-0.043 at p128, 0.065-0.097 at 8K / 32K TP1 and TP2
  rows <= 64, 0.084 / 0.115 at TP2 rows 128 8K / 32K. The pick saves 0.004-0.012 of FlyTape's step at p128 and
  0.014-0.038 at 8K / 32K (largest at TP2 rows 128 32K).
- The QSA family line of 15.3 (taken at 0.6-1.2 with the core unknown) can move to 0.6-0.85 once the q-proj / out-proj
  numbers of P-L3 (`qsa_in` 0.48-0.95) are combined; this probe does not re-measure the projections.
- The `stream` metric (no barrier between layers) is 0.06-0.47 of FlyTape's phase at p128 and 0.65-0.83 at 8K / 32K:
  what a dataflow pass gains if neighbouring stages fill the QSA core's latency chain, an upper bound that WP-L3 must
  measure in situ.

## 7. Lessons

1. **A faithful port first pays.** The rocKE twin of FlyTape's `qsa_core` was at parity with FlyTape's own phase at p128,
   so every later ratio is against a known-equivalent structure, and the levers can be attributed.
2. **Latency chains, not bytes**: at p128 the best kernel is still 4-63x the bytes floor; at 32K rows 128 it is 1.6-1.7x.
   Removing round trips (q via LDS, no release handshake, merges without bank conflicts) and serial single-CTA work (the
   select) is where the time is.
3. **Count on the VALU**: the select's ballot + `s_bcnt1` per key per round is SALU-latency bound; per-lane
   compare-adds and one DPP butterfly per round win (`v1` / `v2`), while fewer rounds with more ballots (`bits2`,
   `nib`) lose.
4. **LDS bank conflicts hide in "FlyTape's layout"**: the `[wave][16][256]` f32 merge puts the 16 heads of a lane group
   in one bank; a 4-word pad wins in every cell.
5. **Two COMGRs, two kernels**: the container COMGR spilled the 512-thread builds and the spilled binaries were wrong;
   time and check the exact code objects (prebuilt, loaded by name).
6. **Solo losers matter** (decision 13): `pf=1` and `gearly` lose alone and appear in most winning combinations.

## 8. Open items

- TP2 confirmation with the prebuilt binaries on n07-17 and n01-09 (section 4.3) if not in the table above.
- Continue the climbs from the lookback winners (`pick3`) and re-run the lookback (decision 15).
- `waves=8` failed the spill gate in 6 of 30 cells on the venv COMGR (p128 TP1 rows 8-32, some 8K rows): a register-lean
  512-thread variant (V through LDS by DMA, `kvdma`) is the next lever; not built.
- Paged KV (ATOM's block-16 layout, `kvl=page16`): not built; the port uses FlyTape's row-contiguous cache. WP-L2e needs
  it (one extra table load per token).
- DESIGN 16 reverse step: diff FlyTape's `qsa_core` LLVM IR / ISA against the port to explain `ft/FT` < 1 at 8K / 32K.
- Accuracy: `score=mfma` (bf16 indexer query) is a tolerance-class change of the selection (N10); RULER gating before it
  can be a default.
