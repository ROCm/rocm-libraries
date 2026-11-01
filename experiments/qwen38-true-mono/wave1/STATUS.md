# Wave 1 status (W5, 2026-11-01)

Wave-1 report for the true rocKE monokernel (Path B, `D0.md`). Ratios and hybrid units (hu: the pinned hybrid's round at
K>=1, its step at K=0, = 1) only; absolute numbers stay in `hackathon-prep/private-perf/qwen38-true-mono/`. Below 1 is
faster. Inputs: the six wave-1 reports in this directory (each with its Verification and, where present, Fix round
section), the D0 adversarial check (`D0.md` section 7) and `quick/README.md` (consolidated stack, 2026-11-01). Nothing
was run for this file. `D0.md` section 5 carries the matching `[W1: ...]` status markers and a short wave-1 paragraph.

## 1. Verdict

Wave 1 is done. All six packages met their gates on hardware. Three were independently confirmed (WP4h, P13, P15).
Three had a PARTIAL verification whose issues were all closed in a fix round that no separate agent has re-checked
(WP1, P11, WP-R3). Every existing build stayed byte-identical with the new knobs off, and no git state changed.

Wave 2 (WP2a/b/c/d/e/f and P3x) is unblocked; W8 launched it on 2026-11-01 (`wave2/`). The plan of record (Path B)
stands, but two wave-1 findings move the budget against the targets: bitwise N3b loses (P15), and the tail's device
savings do not reach the round while the round boundary goes through the host (`D0.md` section 6 item 7). The largest
unbooked lever is now that round boundary (0.07-0.08 of the round as GPU idle), and nothing in wave 2 owns it.

## 2. Package by package

| WP | gate | independent verification | what D0 / wave 2 gets | still open |
|---|---|---|---|---|
| **WP1** platform | met | PARTIAL (the ptoy mutant verdict did not reproduce on a third GPU), then a fix round closing all 7 issues; not re-verified | inline + R1-R5, `args=const`, D0.8 defaults, per-CTA arrivals, deadline polls + launch-fatal word; `sync.check_launch` / `knobs.require_launch_fatal`; `tagged_poll` reload fix as a new op variant; ISA (`.text`) added to the identity census (87 / 87 ISA, 196 / 196 IR identical); a latent bug fixed: `cta_arrive` never reset its LDS counter, and LDS persists across launches on MI355X; loop driver (makes WP3a and WP-L4 buildable). **D0.8 amendment:** a timed-out deadline poll returns the stale payload, which in Path B can be the previous launch's correct value, so a broken tag can be numerically invisible: every build with a deadline edge must check the fatal word after every launch or graph replay | emitter-level edge knobs (`protocol` / `placement` / `queue` values) are validated, not implemented (WP2 / WP3); per-variant run-ahead hold and A1 streams-first not built; exec-join lint report-only; 21 equivalent / harmless GPU-path mutant survivors; M11 picks withdrawn (NOISE under 9.1); tuner confirm rig has a baseline-arm bias (an identical copy of the baseline reads 1.06-1.10) |
| **WP4h** ATOM call-path contract | met | CONFIRMED (re-ran tests, 22 / 22 unit mutants, served bitwise checks; 3 new must-fail mutants caught; strict mode served) | `ATOM_Q38_TRUE_MONO=tail`: per-width tail provider (`owns_logits`, verify, sample, propose), hybrid pass-through for non-greedy / logprobs / refused widths with logits recomputed, `CHECK=n` bitwise compare; P9's `input_ids` alias rule as `StepIds`. A provider already exists at the default path (`qwen38_mono/truemono_tail.py`, consolidated stack) | P11's `_Q38_TF` hunk rides in the same `model_runner.py`: with `ATOM_Q38_TF=1` and the tail on, tail-owned K=0 steps skip teacher forcing; one unexplained graph-vs-eager logits DIFF in a mutant arm (not reproduced in 2208 probed steps); served transitions only shown between requests; ATOM's OpenAI server drops logprobs and MTP refuses them (WP8 must lift both) |
| **P11** accuracy-gate calibration | met | PARTIAL on rule rev 2 (a noisier lever could pass more easily; N2 bracket not 2x / 0.5x; pre-registration wording); fix round introduced rule rev 3, held 4 / 4 pre-registered predictions, recomputed by a separate script in the same lane | canonical RULER mix v1 (7 tasks, 4096-token cap); T4 tolerances per suite / length (K=3 and partly K=0); B = T2 drift of all:2^-5; T2 rule rev 3 with Monte-Carlo designs: f 0.2 / 0.15 need k 4 / n 5 (about 1.6 G per lever), f 0.1 k 10 / n 8 (about 3 G), f 0.05 (N6 or N7 alone) has no design; the served path is not deterministic inside one process (1-ulp prefill, amplified), so T2 is statistical | verify-mode teacher forcing not built (r4s-only levers need T4 + a T1 rig check); the 0.5x side of the N2 bracket is not demonstrated (0.87x arm); RULER 128K / max at K=0 and the non-greedy cells not measured |
| **P13** QSA under-budget order (N4a / N4b) | met for N4a | CONFIRMED (new seed, third GPU, ISA identity, split-KV emulation reproduced) | N4a (`qsa:qdense=1/2/3`, ATOM-truemono, default off): tolerance class, near-bitwise, bitwise-inert above the index budget; per qsa layer 0.918-0.925 vs `best`, 0.923-0.944 on the tuned base; quick e2e K=3 round 0.985, K=0 tok/s 1.025 (about 0.015 / 0.020 hu). It is a FlyDSL-side change, so the hybrid adopts it (already in `finalIqd`). N4b (fused split-KV softmax) passes T0 in emulation only | T2 / T4 for N4a owed (K=0 quick arm: 27-29 / 32 GSM8K against its base's 30-31; not an accuracy test, not ruled out); N4b needs a real kernel (WP14); the `delta_ple_r4s` build compiles to one of two binaries, so a single identity comparison is flaky for it (use `keys_cmp.py --variants`) |
| **P15** N3b router in the fused MoE | met (verdict: LOSS) | CONFIRMED (re-timed on n07-17, recomputed all 56 arm verdicts) | bitwise N3b loses everywhere: delta r4s 1.066-1.081, r1 1.023-1.028 per layer, 0 / 24 rounds faster; both halves lose alone; phase stamps show no hop removed. **Book N3b at 0; rescope WP16 to N3a (router K-split fused into M2's epilogue) or drop it.** `m3_ept=2`, `route=wave`, `m3_rows` kept as rejected default-off catalog knobs; router must-fail mutants added (P10 gap closed) | the n01-09 repeat that meets the 8.1 rule (about 3 GPU-minutes); qsa / delta_ple not timed; N3a and the MFMA / tolerance N3b forms unmeasured; another lane edited `moe_fused.py` and `core/ir.py` after P15's identity proof |
| **WP-R3** Rust direct backend into the tree | met (standalone kernels) | PARTIAL (two rows lacked a compliant n01-09 run; framework `auto` unmeasured; control bias; identity set missed the W6 builds), then a fix round closing all 7; not re-verified | direct backend in the tree, default `comgr`; 100 / 100 existing builds byte-identical (IR, serialized IR, `.text`, whole code object; mutants on both halves caught); B3 r1 0.975-0.984, r4s tuned 0.978-0.980, r4s mono 0.940-0.943, fused MoE r1 0.968-0.974 direct / COMGR, WIN on both nodes under 8.1; per-op lever scoping consumes WP1's hook; framework kernels stay COMGR (`RUST_FRAMEWORK_DIRECT_FAMILIES = ()`) | compliant n01-09 runs of framework-hosted B3 r4s and MoE r1, then D0.6's in-situ `layers=1` test; framework-hosted Q2 direct loses about 8 %; Q2, probe, CCL and lookback verdicts rest on n07-17 alone |
| WP0b-acc | not run as a package | - | covered elsewhere: the hybrid's A-vs-A per suite and length (P11 section 4); the quick-stack RULER 8K question did not reproduce (finalF and finalIqd inside the hybrid's spread, mixes A and B at 8K / 32K), so D0 section 6 item 2 and 7.3 M8 are resolved in substance | the two non-greedy cells, K=1 / K=2 accuracy, RULER 128K / max at K=0, and the quick stack on P11's canonical mix with the T4 rule (estimate about 10 G) |

## 3. D0 check: what was acted on

- Arithmetic fixes are applied in `D0.md` (N4 increment 0.022, sums 0.1185 / 0.081, cell 1 0.89, cell 13 0.89 / 0.89,
  D0.1's P1 citation, the 170 G note).
- Done since the check, by other lanes: 7.3 M7(a) (the lookback picks served end to end on both nodes: `finalG` round
  0.961-0.962), M7(b) (P9's tail on the hybrid through WP4h: neutral end to end, which produced section 6 item 7), M8
  (RULER, above). M2 (spread as a WP3 lookback knob) is written into WP2a's status.
- Still open: B1 (no K=0 lever program; P16 hop-cut probe not scheduled; WP3a not pulled forward), M1 (book measured
  combinations at face), M3 (K=0 G3 0.006), M4 (draft layers inside the tail launch unmeasured; P9's verify build at
  256 VGPRs with 11 spills), M5 (K=1 idle unmeasured), M6 (served-metric column, refill-prefill lever), M9 (no
  independent verification of the quick-track inputs; the quick track now has both nodes for its main cells).

## 4. Budget consequences (proposals; `D0.md` section 4 keeps D0's values)

| line | change | cell 1 (K=3 CONC 1) | cell 6 (K=0 CONC 1, with WP9) |
|---|---|---|---|
| N3 (P15) | 0.0125 -> 0 | +0.009 | +0.009 |
| G3 served (section 6 item 7) | K=3 0.023 x0.8 -> about 0 until the round boundary leaves the host; K=0 small (the fused-argmax tail served at tok/s 1.003, about 0.003 hu) | +0.018 | about 0 |
| N4a (P13) | measured on the hybrid at about 0.015 / 0.020 hu; tolerance class; hybrid-adoptable | inside N4's booking | inside N4's booking |

Starting from 7.6's re-issued centrals (K=3 accuracy-gated 0.888, 0.882 with N8 at face; K=0 0.877 with WP9):
**K=3 about 0.91-0.915, K=0 about 0.886.** Against the targets (0.875 / 0.78) the gaps are about 0.035-0.04 hu at K=3
and about 0.11 hu at K=0. Against the measured best hybrid (consolidated stack, both nodes: K=3 round 0.945-0.946, K=0
per-token about 0.914) the projected margin is about 0.03-0.04 hu, and most of it is accuracy-gated levers that the
hybrid could also take at FlyDSL effort (N1, N4, N6, N7). What remains structure-only: N2, N3a (unprobed), WP9, the wide
/ grouped layers (L unmeasured until WP2d), the Rust backend in situ, Path A, and a device-side round boundary.

## 5. What wave 2 can start

| item | deps | can start | conditions from wave 1 |
|---|---|---|---|
| **WP2a** HC | WP1, P3 | yes (W8 launched) | D0.3's spread values are provisional; every spread value (dedupe, M1 placements) is a knob for WP3's lookback (M2); N1 hooks reserved |
| **WP2b** GEMV B1 / C1 / Q1 / P1 | WP1 | yes (W8) | - |
| **WP2c** QSA Q3-Q7 | WP1 | yes (W8) | parity at served L and 32K / 128K / max; carry `qdense` (N4a) as a default-off knob in the rocKE Q3 / Q4 port |
| **WP2d** MoE grouped / wide, B3 wide, Q2 | WP1, P7 | yes (W8) | re-prove identity on the current `moe_fused.py` / `core/ir.py` (edited after P15); direct only through `rust_family` |
| **WP2e** tail round stages | WP1, P9 | yes (W8) | start from `truemono_tail.py`; first the K=0 tail trace and the round-boundary measurement, since tail savings book about 0 served until the boundary moves |
| **WP2f** PLE | WP1 | yes (W8) | - |
| **P3x** tolerance HC variants | WP0a, P11 | yes (W8) | T2 rev 3 at f 0.2: k 4 / n 5 per variant; r4s-only variants need T4 + T1 (no verify-mode teacher forcing) |
| WP3 `layers=1` kernels | WP2a/b/c/f, WP2d narrow | no: after wave 2 (W8) | every Path B validation build checks the launch-fatal word per replay (D0.8 amendment); holds re-climbed per D0.9 |
| WP3a Path A switch test | WP3 | not formally; a minimal delta r1 loop test on WP1's loop driver is buildable now (7.2 B1(b)) | D0.7 rule, both nodes, 8.1 |
| WP13 N8 | WP2e + RULER | after WP2e | RULER resolved in substance; the canonical-mix T4 run is owed |
| WP14 N4 | WP2c, P13, P11 | after WP2c | structure value is N4b; T2 / T4 for N4a owed |
| WP15 N1 | WP2a, P11, P3x | after WP2a and P3x | - |
| WP16 | WP2d, P15, P11 | **rescope first** | N3a or drop (decision pending) |
| WP4B tail kernel | WP3, WP2e, WP4h | after WP3 / WP2e | WP4h met; draft layers inside the tail stay a knob (M4) |

Small owed items that fit beside wave 2 (each a few GPU-minutes to about 3 G): P15's compliant n01-09 repeat; WP-R3's
two compliant n01-09 framework runs; N4a's T2 (f 0.2, about 1.6 G) and T4; WP0b-acc's remainder; an independent
re-verification of the WP1, P11 and WP-R3 fix rounds (WP1 first: every wave-2 package builds on it).

## 6. Open risks (largest first)

1. **The round boundary binds and nothing owns it.** The served round carries 0.07-0.08 of GPU idle at the boundary
   (host `hipGraphLaunch`, prologue copies); every tail saving reappears there. Until a whole round is launched without
   the host in between (WP5c / WP12) or ATOM prepares round t+1 while round t runs, G3, WP5a and WP5b book about 0, and
   the K=3 target gap is about 0.035-0.04 hu instead of 7.6's 0.01. Proposal: an ATOM-side prepare-ahead arm on the
   quick track now (hybrid-adoptable) and WP12 moved out of wave 6.
2. **K=0 target (<= 0.78) has no lever program** (7.2 B1): gap about 0.11 hu after wave 1. P16 (delta hop cuts, XCD
   levers 1 / 4 / 5) is not scheduled, and the early Path A loop test is not pulled forward.
3. **The structure-only margin over the best hybrid is thin** (about 0.03-0.04 hu projected, mostly levers the hybrid
   can adopt). N3b is dead and N4a is already in the hybrid; the wide cells rest on L, which only WP2d measures.
4. **Three fix rounds are not independently re-verified** (WP1, P11, WP-R3). WP1's is the most important: it sets the
   D0.8 launch-fatal rule and fixed a real LDS-persistence bug that wave-2 stages could repeat.
5. **Accuracy-gate resolution:** T2 is statistical because the served path is not deterministic inside one process; f
   0.05 levers (N6, N7 alone) cannot be gated alone; r4s-only levers have no T2; tail-owned K=0 steps skip teacher
   forcing when both hooks are on (WP4h verifier). A deterministic decode (D0 open item 6, Tier G) would shrink every
   T2 design about tenfold and is still not owned.
6. **Logprobs cell blocked in ATOM** (the server drops logprobs; MTP refuses them): WP8 must change ATOM.
7. **Two-node evidence lags:** n01-09 is often occupied by our own lanes, so 8.1-compliant n01-09 runs are missing for
   P15, two WP-R3 framework rows and P13's job d; with three concurrent workflows this gets worse.
8. **Rust in-situ upside may be zero for framework kernels:** framework-hosted Q2 direct loses 6-13 %; the B3 / MoE r1
   framework rows are unconfirmed on n01-09.
9. **Build hazards for wave 2:** `delta_ple_r4s` compiles to one of two binaries (identity gate needs `--variants`);
   P9's verify build sits at 256 VGPRs with 11 spills; duplicate `iv_name` loops in one kernel fail to compile (alola
   rule 181); concurrent lanes edit `moe_fused.py` and `core/ir.py`.

## 7. Files

- Reports: `wave1/WP1.md`, `wave1/WP4h.md`, `wave1/P11.md`, `wave1/P13.md`, `wave1/P15.md`, `wave1/WP-R3.md`.
- Plan of record: `D0.md` (section 5 `[W1: ...]` markers and the wave-1 paragraph; section 7 the adversarial check).
- Edit script for the D0 markers: `hackathon-prep/qwen38/truemono/w5_status/d0_wave1_edit.py`.
