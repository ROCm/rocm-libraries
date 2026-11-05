# P3x: N1, the tolerance-class HC down-projection forms (workflow W11, 2026-11-02)

DESIGN 12.3 lever N1 (probe P3x variants iv-vi; D0.5 lever order: N1 second), on the consolidated-push-2 served stack
`c2s3b` (recipe `gen_c2Ab` of `truemono-quick/arms.d/c2.sh` with the served WP2b picks: rocKE fused MoE, B3, WP2a HC,
WP2b GEMV, WP2c QSA, the round-boundary device path). PLAN decisions 1 (accuracy-gated levers), 13 / 15 (losers stay
knobs; rejections are "on base X"), 16 (find the bubble first: bounds before forms), 17 (nothing deleted). Ratios,
shares and counts only; absolute layer times and tok/s are in `hackathon-prep/private-perf/qwen38-true-mono/w11/P3x.md`.
Below 1 is faster.

STATUS: in progress (this file is updated as the remaining gates report; section 9 lists what is still running).

## 0. Verdict

**N1 is built (three forms plus two timing-only bounds), gated (T0 per stage, determinism, must-fail mutants; T2 rev 3
PASS for the tolerance form; T1 / T4 see section 6), and it splits by cell:**

- **K=0 (decode r1): a clear end-to-end win.** The served-stack K=0 arm with N1 on `delta.r1` and `qsa.r1` (attention
  HC: stream-split units with the norm after the GEMV; MLP HC: stream-split n-block groups) runs at **0.966 of
  c2s0b's per-token time** (DESIGN 9.1 E2E rule: WIN, 4 of 4 paired passes below 1, outside both A-vs-A bands;
  n01-09), **1.036x c2s0b's tok/s per request** and 1.18x the same-job hybrid's. The bit-preserving form (`ks`: the
  mono's MXFP8 codes, only the f32 order of the 4 stream partials changes; every in-situ output bitwise equal to the
  served layer) gives 0.972 (WIN, 4 of 4).
- **K=3 (verify r4s + the draft round): neutral.** The N1 config moves the K=3 verify round to 0.993 (`na`) / 0.997
  (`ks`) of c2s3b's on n07-17: 3-4 of 4 pairs below 1 but inside the E2E rule's band (NEUTRAL). N1 does not close the
  last ~1.5 % to 1000 tok/s at K=3. Why (section 4): the verify cells' A-side ceiling is 2.3-3.6 % of a layer even
  for an empty down task, the stream-split forms recover 0.5-1 % at qsa r4s and nothing at delta / delta_ple r4s,
  and every M-side form at r4s loses (the extra M1 tasks land on the MLP CTAs, P3's finding).
- **N1c (P3x variant vi, partial-down in the M5 / C1 epilogues) is not built.** Its upper bound was measured instead
  (an empty down task in the served layout, timing only): the A-side bound is 0.975-0.977 (delta r4s), 0.963-0.969
  (qsa r4s), 0.993-0.994 (delta_ple r4s); the M-side bound 0.96-0.97 at qsa r4s / r1, 0.92-0.93 at delta r1 /
  delta_ple r1 and none at delta r4s (1.004). N1c's producer-side GEMV would also lengthen C1 / M5 (WP17 measured
  exactly that for the analogous out_proj move: B3 1.05-1.51x). At K=3 the bound leaves at most about 0.5-1 % of the
  round for all of N1, so vi cannot reach the K=3 target either; at K=0 the M-side bound (up to 7-8 % of delta r1)
  is the open headroom (section 8).

## 1. What was built

| piece | where |
|---|---|
| N1 emitter (new file; imports WP2a's `stages2/hc.py` helpers, nothing existing edited): the stream-aligned down unit (`form=ks`: partial `s` = k_iters `20 s .. 20 s + 19`, the unit normalizes only its stream with FlyDSL's tree and f64 inverse RMS; `form=na`: packs `bf16(x w')` with no statistics), `g` n-blocks per task (1: 80 tasks per row, ATOM spread 1 / 10; 4: 20 per row, spread 2 / 3), the deduplicated unit (ATOM spread 4: one unit per (n-block, stream) for every row, one weight stream), `kh` in-CTA chain split (1 / 2 / 4 / 5 waves, fixed-order LDS sum), the weights' E8M0 scale bytes regathered for the contiguous k_iters (two dword loads per part, a select and a shift per step); the reducer A2 / M2 (`_n1_pack`: the 4 stream partials summed `((p0 + p1) + p2) + p3`, `na`: each scaled by its stream's inverse RMS that A2 / M2 already compute); `form=nb` (N1b in the SERVED layout: the WP2a task map, the un-normalized pack, each 20-step chain split into 4 per-stream 5-step accumulators, the row statistics issued after the chains, the served A2 / M2 kept); timing-only bounds `null` (stream units publish 0) and `nbz` (the served layout's down task does nothing); T0 must-fail mutants `drop` / `dup` / `rrms`; `nab` (na: pack the f32 product); writer with sidecars (`hc.spread` = the ATOM layout value, `hc.n1`, `timing_only`) | `copy-rocm-lib/.../qwen38_mono/stages2/hc_n1.py` |
| CPU tests (82, local, pass): every form / layout / `kh` lowers against the local `hc.py` (and, by a compat tree, against the served stack's older `hc.py`); refusals; spec round trip; mutants change the IR; sidecars; importing the module changes no WP2a build; `nb` issues the statistics after the first MFMA, WP2a before | `copy-rocm-lib/.../library/tests/test_qwen38_stages2_hc_n1.py` |
| in-situ rig (numerics T0 vs two fp32 goldens + determinism + mutants, then DESIGN 9.1 timing) on the served c2s3b layer exactly as `C2/c2_layer_gate.py` builds it; arm tokens `<A|M><form><g>.<kh>[.n0][.dd][.mut]` | `hackathon-prep/qwen38/truemono/W11/p3x/w11_insitu.py`, `w11_n1lib.py`, `job_insitu.sh`, `summarize_insitu.py` |
| T1 (Tier A with the N1 functions linked vs the FlyDSL HC) | `W11/p3x/job_tier.sh`, `w11_tier_cmp.py` |
| served-stack generator (puts the N1 functions of a config into a c2Ab function dir, sets the cell's spread lever, rewrites its expect build line) and the quick-track arms | `W11/p3x/w11_gen.py`; `truemono-quick/arms.d/w11.sh` (new: `w11na3`, `w11na0`, `w11ks3`, `w11ks0`, `_b` twins, T2 arms `w11nat0`, `w11kst0`, planted `w11xt0b2x`, configurable `w11x*`) |
| stages: `rocke-qk-w11-<x>` = `rocke-qk-c2-f` + `hc_n1.py` (login-node tar append), `atom-truemono-w11-<x>` / `hb-atom-w11-<x>` = byte copies of c2-f's, `h-qk-w11-<x>` = c2-f's harness build + `truemono/W11` (P3x files) + `tests/` | `W11/p3x/stage_p3x.sh` |

Every lever is default-off: no existing rocKE file changed (the served WP2a HC functions are byte-identical); no ATOM
change (every N1 layout uses an existing ATOM spread value); an arm without a `w11` recipe is the served stack.

## 2. The forms and what they change (P3x iv, v, vi)

| P3x | form | down task | reducer | numerics |
|---|---|---|---|---|
| iv (N1a) | `ks` | (n-block group, stream) units: one stream's statistics and pack (1/4 of the row), 20-step chain over that stream's k_iters | A2 / M2 sum 4 stream partials (80 units' pairs per row) | the mono's MXFP8 codes; only the f32 order of the 4 partials. **Bitwise equal output in every in-situ sample** (bf16 rounding of the low-rank sum hides the reorder) |
| v (N1a + N1b) | `na` | as `ks` without any statistics: packs `bf16(x w')` | as `ks`, each partial times its stream's inverse RMS | MXFP8 codes of the un-normalized stream (differ where `inv` is not a power of two); tolerance class |
| v (N1b alone) | `nb` | the served task map, z pack, 4 per-stream accumulators, statistics after the chains | served A2 / M2 | as `na` |
| vi (N1c) | not built | (bound: `nbz` / `null`, timing only) | - | - |

## 3. Numerics (DESIGN 12.4 T0 / T1 / T2 / T4)

**T0 (in situ, every arm before timing).** Per arm: two launches bitwise equal (deterministic), output nonzero, both
NUMA copies equal, every checked mailbox carries the launch tag. Over 6 random layer inputs with per-stream RMS 0.3 /
1 / 2.7 / 8.1 (a wrong-stream inverse RMS must show): the HC outputs (A2 / M2 mixed rows and injection weights) against
two fp32 goldens of the rig's own HC weights: `golden.hc_mono` (the mono's MXFP8 numerics: implementation fidelity)
and `golden.hc` (ATOM's bf16 regular path, no packs: the accuracy reference; the served HC sits about 1.1 % from it,
the re-quantization floor). Gate (DESIGN 12.4: max(1.25x the stage's own drift, the re-quantization floor)): for the
HCs the arm replaces, pooled relative L2 <= 1.25x the served layer's and max error <= max(1.25x, served + 2 bf16
ulps of the tensor max); the injection weights' L2 gets a floor of half a bf16 ulp (4 values per row).

Two corrections were made to the T0 rule during the work, both before any lever decision depended on them, and both
are recorded here: (1) the first job gated against `hc_mono` alone, where any code-moving lever shows 100-170x drift
because the served HC reproduces that golden to 1e-4; the gate moved to the regular-path golden (`hc_mono` ratios
stay reported). (2) An A-only arm "failed" on M2's injection weights, which are computed by the SERVED (unchanged) M1
code on the arm's slightly different C1 streams; the gate now covers the replaced HCs' outputs only and gives the
4-value injection tensor its half-ulp floor.

| arm family | T0 (regular-path golden, arm / served) | output vs the served layer |
|---|---|---|
| `ks` (all `g`, `kh`, dedupe) | 1.000 (bitwise) in every cell and job | bitwise in every sample |
| `na` (all) | 1.00-1.33 on the max error (within the 2-ulp floor), 0.99-1.02 on the L2 | differs (h_out L2 0.04-0.15 on the random-weight rig, where routing flips amplify) |
| `nb` | 1.00-1.47 max (floor), 0.99-1.14 L2 | differs |
| bounds (`null`, `nbz`) | 23-30x (timing only, never served) | - |
| must-fail mutants (`drop`, `dup`, `rrms` per form; 43 arms over 7 jobs) | 11-70x except one (below) | - |

Every must-fail mutant failed T0 except `nb` with a uniform 1 + 2^-5 inverse-RMS scale at delta_ple r4s (1.229, job
w11-is4): a uniform scale is not the DESIGN's "wrong per-stream rrms". The `nb` mutant became a stream rotation (stream
q scaled by stream q + 1's inverse RMS); it fails at 43-69x in every cell (w11-is5). The `ks` uniform-scale mutant is
caught at 1.36-1.43 everywhere it ran.

**T2 (rev 3, P11 section 5.5; job `w11-t2na`, n01-09).** K=0 teacher forcing (canonical protocol: 128 gsm8k + 64
humaneval items, 512 reply tokens), reference arm = c2's T2 arm of the served stack (`c2t0b`), one record run, k = 4
forced references, n = 5 runs of the N1 `na` config (`w11nat0`: `qsa.r1` and `delta.r1` with A `na` g1 kh4 + M `na`
g4 kh2; the K=0 cells are the ones teacher forcing exercises), n = 5 runs of the P11-calibrated must-fail (`b2x`:
all-layer planted noise at 0.0192 = 2.20x N1's share), seeded shuffled order. N1 share f = 0.2 (design k 4 / n 5).

| arm | gsm8k KL increment over the floor | as a share of f B | within limit | 2x-exclusion limit | verdict |
|---|---|---|---|---|---|
| `w11nat0` (N1 na) | +7.60e-4 | 0.49 | 2.12e-3 | 2.43e-3 | **PASS** (reply KL, flips, humaneval KL also inside) |
| `w11xt0b2x` (planted 2x) | +3.37e-3 | 2.16 | 2.12e-3 | 2.43e-3 | **FAIL** (as it must) |

The `ks` form needs no T2 run of its own on this evidence: its output equals the served layer's in every in-situ
sample (section 3 table); it is gated by T0 and determinism, and it is covered by the `na` pass (a strictly larger
numerical change in the same stages).

**T1 / T4: section 9.**

## 4. In situ (DESIGN 9.1; served c2s3b layer, 16 cold weight copies, 96 launches, 30 shuffled rounds, controls = a
second graph of the served layer + the served layer compiled under another kernel name)

Bounds first (decision 16). `nbz` = the served layout's down task does nothing (A1 or M1); `null` = stream units that
publish 0. Layer-time median / served layer:

| cell | A-side bound (`nbz`) | M-side bound (`nbz`) | both | jobs |
|---|---|---|---|---|
| delta r4s | 0.975-0.977 | 1.004 (no M headroom) | 0.987 | is4, is5 |
| qsa r4s | 0.963-0.969 | 0.960-0.968 | 0.928 | is4, is5, is6, is8 |
| delta_ple r4s | 0.993-0.994 | 1.008 | 0.999 | is4, is5 |
| qsa r4 (the K=3 draft round's 4-row layer) | 0.952 | - | - | is8 |
| qsa r1 | - | 0.961 | - | is6 |
| delta r1 | - | 0.922 | - | is6 |
| delta_ple r1 | 1.015 (the empty task loses: the layout is P3's dedupe value 3) | 0.934 | - | is6 |

The units' own reduce costs: `null` stream units at g = 4 (A2 sums 80 units' partials) bound delta r4s at only 0.994
against `nbz`'s 0.977 in the served layout: re-reading 4 partials per column in all 80 A2 tasks costs about 1.7 % of
the layer at r4s. This is the price every stream-split form pays at r4s.

Measured forms (served-pick base jobs is4-is8; WIN / LOSS per DESIGN 9.1; n07 = n07-17, n01 = n01-09):

| cell | best forms (median, verdict, node) | rejected on base c2s3b |
|---|---|---|
| qsa r1 | A `na` g1 kh4 + M `na` g4 kh2: 0.952 (n01), 0.953 (n07) WIN; A `ks` g1 kh4/kh5 + M `ks` g4 kh2: 0.963-0.964 (both), 0.957 (n07); A alone `ks` g1 kh4: 0.972-0.975, `na` kh4: 0.966; M alone `ks` g4 kh2: 0.989-0.991 | g = 4 A units (1.010-1.014 LOSS), `nb` M (1.005) |
| delta r1 | A `na` g1 kh4 + M `na` g4 kh2: 0.956 (n01) WIN; `ks` + `ks`: 0.967 (n01), 0.970 / 0.9695 (n07) WIN; A `ks` kh4: 0.979-0.982; M `ks` g4 kh2: 0.986-0.987 | g = 4 A units (LOSS), `nb` M (1.010) |
| delta_ple r1 | none | every form LOSS or neutral (1.005-1.036): the polled PLE A1 already runs P3's dedupe layout |
| qsa r4 (draft) | A `ks` g1 kh5 (deduplicated): 0.975, A `na` g4 kh2: 0.975, `ks` g4 kh2: 0.980 (n07) WIN | M forms (1.006), `ks` dedupe kh1 (1.008) |
| qsa r4s (verify) | A `na` g4 kh2: 0.990 / 0.994 WIN (n07, two jobs), neutral 0.996 (n01); A `ks` g4 kh2: 0.990-0.993 WIN on n07 in 2 of 4 jobs, neutral elsewhere | every M form (1.005-1.044), deduplicated A (neutral) |
| delta r4s (verify) | none robust (`nb` + f32 pack 0.994 WIN once, 1.001 in the second job) | `ks` g4 kh1 (1.007-1.010 LOSS), M forms (1.006-1.058) |
| delta_ple r4s (verify) | none | every A / M form neutral or LOSS |

Full per-job table (every arm, both nodes, tokens, T0, controls): `hackathon-prep/qwen38/truemono/W11/p3x/results/
insitu_table.md`. Jobs is2 and the smoke job ran on the c2Ab recipe WITHOUT the served WP2b picks (alola rule 236):
their timings are on base "c2Ab with default WP2b GEMVs" and are listed as such, not used for decisions.

## 5. End to end (quick track, CONC 1, 32 gsm8k prompts, greedy, one server at a time, seeded rotated order, 4 passes;
DESIGN 9.1 E2E rule over the A-vs-A bands of hybrid / hybrid_b and c2s3b / c2s3b_b, or c2s0b / c2s0b_b at K=0)

Configs (`arms.d/w11.sh`): **na** = `qsa.r4s=Ana4.2; qsa.r4=Ana4.2; qsa.r1=Ana1.4+Mna4.2; delta.r1=Ana1.4+Mna4.2`;
**ks** = `qsa.r4s=Aks4.2; qsa.r4=Aks1.5; qsa.r1=Aks1.5+Mks4.2; delta.r1=Aks1.5+Mks4.2`.

| job (node) | arm / ref | metric | median | pairs below 1 | A-vs-A band | verdict | vs the same-job hybrid |
|---|---|---|---|---|---|---|---|
| w11-e3k0-n01 (n01-09) | w11na0 / c2s0b | per-token time | **0.966** | 4/4 | 0.996-1.004 | **WIN** | 0.848 |
| w11-e3k0-n01 | w11ks0 / c2s0b | per-token time | 0.974 | 4/4 | 0.996-1.004 | **WIN** | 0.855 |
| w11-e3k3-n07 (n07-17) | w11na3 / c2s3b | ms per round | 0.993 | 3/4 | 0.995-1.005 | NEUTRAL | 0.886 |
| w11-e3k3-n07 | w11ks3 / c2s3b | ms per round | 0.997 | 4/4 | 0.995-1.005 | NEUTRAL | 0.887 |
| w11-e1-n07 (n07-17) | qsa.r4s=Aks4.2 alone / c2s3b | ms per round | 0.998 | 3/4 | 0.998-1.006 | NEUTRAL | 0.886 |

Per-request tok/s, ratios to c2s3b / c2s0b of the same job: K=0 na 1.036, ks 1.029; K=3 na 1.004 (n07-17).
Acceptance length unchanged (K=3 medians within 0.01 of c2s3b's). Absolute tok/s: private-perf.

## 6. (reserved: T1 / T4 results)

## 7. Lookback, losers, "rejected on base X"

All verdicts are on base c2s3b / c2s0b (the c2-f stages). Every form, layout and knob above stays a default-off value
of `hc_n1.py` (decision 13): the losing g = 4 r1 units, the M forms at r4s, the deduplicated units at r4s, `nb`, the
f32 pack (`n0`), every `kh`. Decision-15 re-tests to run when the base changes: `nb` (it attacks the statistics,
which a future base may expose more), the r4s M forms (their loss is a CTA-placement clash with M2 / M3: a placement
or `prio` re-climb with the N1 layout may turn it), and the r4s A forms on n01-09.

## 8. Open items

- **N1c at K=0 (M side).** The M-side bound at delta r1 is 0.922 and the best M form reaches 0.986: the rest of M1
  (the stream poll from C1, the pack, the chain) is the K=0 headroom. N1c moves M1's GEMV into C1's epilogue
  (C1 owns 16 columns x 4 streams; 324 f32 partials per task, 160 producers): not built here.
- **K=3 needs another structural piece**: the verify layers' HC has at most 2-4 % of a layer even with an empty down
  task, and the stream-split forms pay their reduce at r4s.
- T1, T4 (K=3 final arm and K=0 arm) and the second-node e2e (K=3 on n01-09, K=0 on n07-17): section 9.

## 9. Still running / to be added

## 10. Process note

The P3x agent created `truemono/W11/stage_w11.sh` with the Write tool while the WP17 agent's file of the same name
existed (09:22): the WP17 agent's script was overwritten (alola rules 236 / 238). The WP17 agent moved to
`W11/wp17/`; P3x's files now live in `W11/p3x/` (the W11 root copies are left as they were, nothing deleted).
