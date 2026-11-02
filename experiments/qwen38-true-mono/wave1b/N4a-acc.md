# N4a-acc: accuracy gates of N4a (`qsa:qdense=3`) on the current best stack (W9, wave 1b)

P11's T2 (rev 3 rule) and T4 (canonical RULER mix v1 at 8K / 32K / 128K, GSM8K, HumanEval) for lever N4a (P13:
QSA rows under the index budget, `L + 1 <= 2048`, attend their blocks in position order; narrow ungrouped QSA builds
only), run on the current best stack with and without the lever. Accuracy deltas, spreads and KL only; absolute
accuracies are in `hackathon-prep/private-perf/qwen38-true-mono/wave1b/N4a-acc.md`. Harness:
`hackathon-prep/qwen38/truemono-w9n4a/` (new files only: `arms_w9.sh`, `t2job.sh`, `submit.sh`, `sync.sh`,
`prereg_n4a.json`, `cross_disc.py`, `dry_arms.sh`). Every job: `rocke` partition, n01-09, q38 container, one GPU.

## 1. Verdict

**Gate met. N4a passes T2 and T4; the T2 gate and the T4 gate both fail their must-fail mutants on this stack.**

- **T2 (rev 3, share f 0.2 for N4, design k 4 / n 5 met):** N4a's mean kl_gsm8k increment over the same-job forced
  references is **+0.004e-3 = 0.002 f B** (limit 2.16e-3, 2x-exclusion limit 2.27e-3, arm sd 1.3 S0): **PASS**, the
  lever is indistinguishable from the reference arm at the protocol's resolution. Must-fails: the `qdense=8` kernel
  mutant (block 0 dropped under the budget) **FAILS** at 47x f B; the calibrated planted arm (`qdense=3` + P11's `b2x`)
  **FAILS** at 2.07x f B. All three pre-registered predictions held (`prereg_n4a.json`, sha256 `443febfa...` printed
  at the job start before the record run).
- **T4:** the qdense stack passes every suite and length against the pinned hybrid's pool (P11 tolerances), against
  the matched-concurrency hybrid pool (K=3 CONC 1), and against the same stack without qdense in the same job. GSM8K
  -0.0006 / +0.0002 (flexible / strict, S 0.0121 / 0.0136), HumanEval -0.014 (S 0.061), RULER 8K -0.0001 (S 0.012),
  32K +0.0011 (S 0.014), 128K -0.0033 (S 0.031). Must-fail: planted all:2^-2 **FAILS** GSM8K (both filters),
  HumanEval and RULER 8K (-0.23 / -0.24 / -0.49 / -0.24).
- **What T4 can see of N4a at all (finding):** qdense is dropped on grouped / wide builds, so at P11's canonical served
  concurrency (K=3 CONC 4: the `r16g4` verify build) it is off on most steps, and it is bitwise inert on every RULER
  position (all have `L + 1 > 2048`). The earlier stack accuracy jobs at CONC 4 (`qk-acc3`, `qk-acc4`) therefore did
  not exercise N4a on most steps. This round ran GSM8K / HumanEval at **K=3 CONC 1** (the `r4s` verify build carries
  `qdense=3` on every verify round; a 5-shot prompt of roughly 0.8-0.9K tokens plus about 450 generated tokens per
question on average keeps most rounds inside the budget, and a question whose reasoning runs past 2048 leaves it) and
  ran RULER as the canonical mix as asked (a structural no-op for N4a: an A-vs-A of the stack, which passed).
- **T4 is far coarser than T2 for this lever class:** the `qdense=8` mutant (47x f B in T2) stays inside every T4
  tolerance (GSM8K -0.011, S 0.012; HumanEval -0.024; RULER 8K -0.008), while the per-sample discordance against the
  qdense stack rises above the within-arm rate (GSM8K 0.032 vs 0.026, HumanEval 0.172 vs 0.133). T2 decides
  tolerance-class levers; T4 is the coarse safety net (as P11 designed it).
- **Booking:** N4a (measured end to end at about 0.015 / 0.020 hybrid units, P13) is no longer "T2 / T4 owed": book it
  at its measured value with p 1.0 on this stack. It stays a default-off knob (`ATOM_Q38_SCHED` kind group
  `qsa:qdense=3`; WP2c carries it in the rocKE Q3-Q7 port). Not changed here: D0 / STATUS (other owners); the next
  editor of D0 section 5 / wave-1 status can replace "T2 / T4 owed" by this file.

## 2. Arms and stages (`arms_w9.sh`)

Stages are frozen login-node copies of the consolidated-stack lane's `cs11` stages (the trees its last gated full-stack
jobs ran): `atom-truemono-w9-n4a`, `rocke-qk-w9-n4a`, `hb-atom-w9-n4a`, plus the harness stage `w9-n4a-h` and
`rocke-tmbase`. No ATOM-truemono or rocKE file was edited for this work.

| arm | what | gate |
|---|---|---|
| `w9b3` | the current best stack at K=3: `picksI` (incl. the grouped K=3 picks) + `qsa:qdense=3` + N8 96K + subset-head pick + the verify tail (`hdelay` 0) | T4 |
| `w9b3nq` | the same without qdense (`picksI` only) | T4 |
| `w9b3q8` | the same with the `qdense=8` mutant (under the budget list slot 0 takes block 1: block 0 dropped, block 1 twice) | T4 must-fail candidate |
| `w9f3p2` | `finalIqd` (no tail) + planted `all:2^-2` (P11's every-suite must-fail) | T4 must-fail |
| `w9t0` | K=0 `picksI` stack, no qdense: the T2 record and the 4 forced references | T2 |
| `w9t0qd` | the same + `qsa:qdense=3` (5 forced runs) | T2 lever |
| `w9t0q8` | the same with `qdense=8` (3 runs) | T2 must-fail |
| `w9t0qb` | `qdense=3` + planted `all:0.0192` (P11's `b2x`, measured 2.20x of a 0.2 share) (3 runs) | T2 calibrated must-fail |

T2 runs without the K=0 tail: `ATOM_Q38_TF` forces tokens in the plain sampler branch of `postprocess`, which the K=0
tail's fused-argmax route bypasses (the hook would silently do nothing). The tail's sample is bitwise the hybrid's
(cs gate, 64 / 64), so the K=0 layer kernels (where qdense acts) are what T2 measures. No N8 at K=0 (draft-side only).

Server-log checks, every arm-run (quick.sh `check_log` and the extended T2 copy): 0 tracebacks, mono on, rocKE head
on, no `numa off`, the recipe's exact build lines (`expect_k0.txt` 3 lines at K=0, `expect_k3.txt` at K=3; exact
strings, so they assert `qdense=3` / `=8` present in the lever arms and absent in the base: e.g. `build qsa r4s: ...
prio.q2=2 qdense=3` vs `... prio.q2=2`), `tm tail r4s: bound` and no tail reject line at K=3, the perturb line exactly
on the planted arms. **All 30 arm-runs (16 T2, 14 T4): checks ok=1, 0 missing build lines, 0 reject lines.** 21 accuracy-hook tests passed in the T2 job before any run.

## 3. T2 (job 68243152 `w9-n4a-t2a`)

Canonical protocol (P11 5.1): K=0, `tf_client --prefill-tokens 1 --sets "gsm8k:128:512 humaneval:64:512"`,
`--max-model-len 65536`, one record run of `w9t0`, then 15 forced runs in a seeded shuffled order (seed 91). Every
forced gsm8k item (prompt + 512 reply tokens) stays under the budget, so qdense is on at every gated position.

| arm | share f | n | kl_gsm8k increment per run (e-3) | mean d (multiple of f B) | limit / 2x-exclusion limit | rev 3 |
|---|---|---|---|---|---|---|
| `w9t0qd` (N4a) | 0.2 | 5 | +0.69, -0.51, -0.30, +0.49, -0.36 | **+0.004e-3 (0.002x)** | 2.16e-3 / 2.27e-3 | **PASS** |
| `w9t0q8` (mutant) | 0.2 | 3 | +72.6, +73.8, +73.8 | +73.4e-3 (47x) | 2.23e-3 / 1.95e-3 | **FAIL** (over limit) |
| `w9t0qb` (lever + b2x) | 0.2 | 3 | +3.14, +3.44, +3.12 | +3.23e-3 (2.07x) | 2.23e-3 / 2.29e-3 | **FAIL** (over limit) |

Floor (4 references): kl_gsm8k sd 4.8e-4 = 1.2 S0 (P11 jobs: 2.25-2.34e-2 floors; this job's sits in that range).
Reported metrics for N4a (not gated): reply KL +0.008e-3, reply flips -0.02 pp, humaneval-set KL -0.05e-3, all far
inside their limits. `w9t0qb`'s 2.07x multiple is the planted 2.2x of P11 plus N4a's ~0: consistent with N4a adding
nothing measurable. n 3 for the must-fail arms is below the f 0.2 design, which only bars a PASS (rev 3 FAILs below
the design).

## 4. T4 (`t4_gate.py`; pools recomputed from the P11 jobs' raw results)

Pools: GSM8K / HumanEval = P11's hybrid runs (5 / 9 runs, K=3 CONC 4: tolerances 0.0121 / 0.0136 / 0.0610 exactly as
P11 section 4); RULER = P11's canonical pool (`w5-p11-ra` 3 x 350 + `w5-p11-s2` 2 x 175; at 350 samples S 0.0117 /
0.0136, at 175 for 128K S 0.0309; P11's table quotes 0.0121 / 0.0149 / 0.0309 for the same pool, the verdicts are the
same under either); matched concurrency: the hybrid's K=3 CONC 1 runs of `qk-cs-accA` (3 runs).

### 4.1 GSM8K 1,319 5-shot chat and HumanEval 164 (low), K=3 CONC 1, `--max-model-len 262141` (job 68243250, 4 shuffled passes)

| arm | GSM8K flexible / strict: delta vs P11 pool (S) | vs CONC 1 hybrid (S) | HumanEval delta vs P11 pool (S) | vs CONC 1 hybrid (S) | verdict |
|---|---|---|---|---|---|
| `w9b3` (qdense) | -0.0006 / +0.0002 (0.0121 / 0.0136) | -0.0003 / +0.0004 (0.0084) | -0.0137 (0.0610) | -0.0097 (0.0545) | **PASS** |
| `w9b3nq` | -0.0021 / -0.0015 | -0.0018 / -0.0013 | -0.0061 | -0.0020 | PASS |
| `w9b3` vs `w9b3nq` (same job) | +0.0015 / +0.0017 (0.0099 / 0.0084) | - | -0.0076 (0.0541) | - | PASS |
| `w9b3q8` (1 run, job 68244177) | -0.0112 / -0.0092 | - | -0.0244 | - | PASS (not caught) |
| `w9f3p2` (1 run) | **-0.2311 / -0.2375** | - | **-0.4878** | - | **FAIL** (must-fail) |

Within-arm A-vs-A (4 runs): GSM8K flexible spread 0.0038 (qdense) / 0.0099 (base), HumanEval 0.031 / 0.037.
Per-sample discordance (`cross_disc.py`, permutation test over run labels): GSM8K within-arm 0.0245, cross-arm 0.0246
(cross - within +0.0001, p 0.49); HumanEval 0.1265 vs 0.1280 (+0.0015, p 0.49): qdense changes no more answers than a
re-run does. The same statistic against the mutants (single runs, so p >= 0.2 by construction): `qdense=8` +0.0064
(GSM8K) / +0.039 (HumanEval), all:2^-2 +0.216 / +0.396. MTP acceptance over the GSM8K cells: qdense / base 1.000
(4 runs each; generated tokens per question within a 6 % band over all 8 runs, no arm effect).

### 4.2 RULER canonical mix v1, K=3, `--max-model-len 262141` (jobs 68244123 `r1`, 68244721 `r2`; 1 pass each, opposite orders)

| arm | 8K delta (S) | 32K delta (S) | 128K delta (S) | verdict |
|---|---|---|---|---|
| `w9b3` (qdense), 2 runs | -0.0001 (0.0117) | +0.0011 (0.0136) | -0.0033 (0.0309) | **PASS** |
| `w9b3nq`, 2 runs | -0.0069 | -0.0050 | -0.0057 | PASS |
| `w9b3` vs `w9b3nq` | +0.0068 (0.0171) | +0.0061 (0.0171) | +0.0024 (0.0343) | PASS (and the reverse) |
| `w9b3q8` (1 run, 8K) | -0.0083 | - | - | PASS (qdense inert above the budget) |
| `w9f3p2` (1 run, 8K) | **-0.2381** | - | - | **FAIL** (must-fail; 78 of 350 replies empty) |

Fixed task set: identical prompt hashes in every RULER run of every arm. Cross-arm minus within-arm discordance:
8K -0.0014, 32K +0.0029, 128K +0.0014 (p 1.0 / 0.67 / 0.67): at RULER lengths the two arms are the same numerics
(qdense's per-row branch takes the mono path for every `L + 1 > 2048` row), and the deltas between them are the served
path's own run-to-run noise (P11 5.2).

## 5. Must-fail summary

| gate | mutant | expected | result |
|---|---|---|---|
| T2 rev 3 (f 0.2) | `qdense=8` (lever-specific kernel mutant) | FAIL | FAIL, 47x f B, 3 of 3 runs |
| T2 rev 3 (f 0.2) | `qdense=3` + planted `b2x` (2.2x share) | FAIL | FAIL, 2.07x f B, each run over the limit |
| T4 | planted all:2^-2 | FAIL GSM8K, HumanEval, RULER 8K | FAIL all four |
| T4 | `qdense=8` | (informational, pre-registered without prediction) | inside every tolerance: T4 cannot see a 47x-f-B kernel defect of this class |
| T4 RULER | `qdense=8` at 8K | inside (structural: above the budget) | inside, as pre-registered |

## 6. Discipline

- Pre-registration (`prereg_n4a.json`: rule, designs, arms, predictions for T2 and T4) staged before submission; its
  sha256 `443febfa03be...` is in the T2 job log before the record run. Every prediction held.
- Same-job references for T2 (the rule never uses a stored floor); same-job with / without arms for T4, shuffled pass
  orders (seeds 91-95), a fresh server per arm-run.
- These are accuracy jobs, not timing: DESIGN 8.1's neighbour rule does not apply (n01-09 carried other lanes' jobs).
- Decision 13 / 15: N4a is a lever verdict on base `picksI` + N8 + head + tail (K=3) / `picksI` (K=0). The rejected-lever
  pool is untouched by an accuracy gate; the lever stays a default-off knob.
- Tools: P11's `t2_gate.py` (16 tests, 27 / 27 mutants), `t4_gate.py`, `acc_stats.py`, `t2_stats.py` unchanged; the
  T2 job script is a copy of `p11.sh` whose only changes are the candidate rocKE stage, the harness stage name, a free
  port and the extra K=0 log checks (which fired on every run: `file-lines=3, file-missing=0`). `cross_disc.py` is a
  new diagnostic; its must-fail behaviour is the mutant rows of 4.1.

## 7. Open items (nothing silently narrowed)

1. **K=1 / K=2 (`r2s` / `r3s` builds with qdense):** not exercised by T2 (K=0 only) or this T4 (K=3 CONC 1 serves
   `r4s`). The lever's code is the same per-row branch at every narrow width; a K=1 / K=2 T4 cell at CONC 1 would cover
   them (about 3 h of GPU for 4 + 4 runs).
2. **K=0 T4 (GSM8K / HumanEval served at K=0):** not run; T2 is the stronger gate there and passed at 0.002 f B.
3. **RULER max (257024):** not in the requested set; qdense is structurally inert there as at 8K-128K.
4. **Verify-mode T2** (P11 open item 1): still not built; the `r4s` builds are covered by T4 at CONC 1 only, which is
   coarse (it missed `qdense=8`).
5. **CONC 2-4 accuracy of the stack** keeps qdense off on the grouped builds by construction; nothing to gate there for
   N4a.
6. **N4b** (fused split-KV softmax) is not built in a kernel (P13 / WP14) and not gated here.

## Verification (independent verifier, W9, 2026-11-02)

**Verdict: CONFIRMED (scoped).** I tried to refute the report and could not. Both gates were exercised where N4a acts
on the claimed configurations (K=0 r1 for T2, K=3 CONC 1 r4s for T4). The scope limits are the ones section 7 already
states: the r2s / r3s builds are ungated, and the r4s path has only T4, which is coarse, plus P13's stage-level ladder.

What I checked:

1. **Provenance.** The `t2_stats.json` in private-perf is byte-equal (sha256) to the cluster's
   `results/w9-n4a-t2a/p11/t2_stats.json`. The pre-registration's sha256 (`443febfa...`) matches the local
   `prereg_n4a.json`. That file's mtime (21:43 UTC) is 4 minutes before the T2 job's start (21:47 UTC). The frozen stages
   `rocke-qk-w9-n4a`, `atom-truemono-w9-n4a` and `hb-atom-w9-n4a` have the same sha256 as `rocke-qk-cs11`,
   `atom-truemono-cs11` and `hb-atom-cs11`.
2. **Arm isolation.** `tunedIqd` is `gen_tuned picksI.json` plus `GT_EXTRA=qsa:qdense=3`, and `w9I` is the same picks
   without it. The resolved `ATOM_Q38_SCHED` strings of `w9t0` and `w9t0qd` differ only by the trailing `qsa:qdense=3`.
   The per-run `qsa_builds.txt` show `qdense=3` / `=8` on `qsa r1`-`r4` in the lever and mutant arms and no qdense in the
   base. At T4, `qsa r4s ... qdense=3` appears in `w9b3` and not in `w9b3nq`. The grouped `r8g4` / `r16g4` builds carry
   no qdense in either arm, as `sched.resolve` intends.
3. **Engagement.** In ATOM-truemono `stages/qsa_dense.py`, `qdense=8` is level 3 (`DIRECT`) plus the slot-0 mutant on
   the same under-budget per-row branch. The mutant's 47x-share failure therefore shows that the `qdense=3` branch runs
   at the forced T2 positions.
4. **Gate recomputation (local, P11 tools unchanged).** `t2_gate.py --rule mean` on the job's stats reproduces every T2
   line: N4a PASS at +3.75e-06; `qdense=8` FAIL at +7.34e-02; `b2x` FAIL at +3.23e-03. `--rule mean2` and `--rule run`
   also pass N4a. `t4_gate.py` reproduces every T4 delta in sections 4.1 and 4.2:
   - against the P11 hybrid pool;
   - against the CONC 1 hybrid pool;
   - qdense vs no-qdense, both directions;
   - RULER 8K / 32K / 128K;
   - the planted all:2^-2 arm FAILS GSM8K (both filters), HumanEval and RULER 8K.

   The planted-arm deltas differ from the table by at most 0.003, depending on which pool files are combined. All
   verdicts are unchanged.
5. **Hardware replication** (job 68245667 `w9-vN4a-acc-t2r`, n01-09, a fresh job, seed 97). Stages
   `rocke-qk-w9-vN4a-acc`, `atom-truemono-w9-vN4a-acc`, `hb-atom-w9-vN4a-acc` and `w9-vN4a-acc-h` are login-node copies
   of the builder's frozen stages, sha-verified at unpack. Design: one record run of `w9t0`, then forced runs of `w9t0`
   x2, `w9t0qd` x2 and `w9t0q8` x1, all on the canonical sets. Every run passed the log checks (3 / 3 K=0 build lines).
   Results (kl_gsm8k increment over the same-job mean of the 2 references):
   - `qdense=3`: +0.22e-3 and +0.43e-3, mean **+0.32e-3 = 0.21 f B (0.79 S0)**. This is below the original job's limit
     of 2.16e-3 and inside its reference sd of 4.8e-4.
   - `qdense=8`: **+72.2e-3 = 46x f B**. The original job measured 47x.

   Two references are below the rule's minimum of 4, so this is a descriptive replication, not a second gate verdict.
   It reproduces both the null effect of the lever and the size of the mutant's failure. Raw files:
   `private-perf/.../wave1b/N4a-acc-raw/verify-t2r/`.

Refutation attempts that did not land, and residual caveats:

- **RULER cannot test N4a at all.** Every RULER decode row has `L + 1 > 2048`, and the 8K / 32K cells run at CONC 4
  (grouped builds). The report says this itself. The RULER PASS is an A-vs-A of the stack, not evidence about N4a.
- **The K=3 verify path (`r4s`, `c.seq` code in `qsa_dense`) has no T2 gate.** T4 at CONC 1 is the only model-level
  gate there, and it let the 47x `qdense=8` mutant through. The real protection for r4s is P13's per-stage tolerance
  ladder, which covers r4s cells at L 639 / 1021 / 2044 / 2048 with mutants that fail. The model-level evidence (MTP
  acceptance ratio 1.000; cross-arm discordance equal to within-arm) is supporting but weak. Verify-mode T2 (open item
  4) remains the missing sensitive gate for the headline K=3 path.
- **`r2s` / `r3s` (K=1 / K=2) are not exercised by either gate** (open item 1). Do not deploy qdense at K=1 / K=2 on the
  strength of this file.
- **Wording.** `prereg_n4a.json` names the K=0 base `tunedI`, while this file calls it the `picksI` stack. Both are
  `gen_tuned picksI.json` without qdense (`gen_w9I`), so this is a naming difference, not an arm mismatch.

Nothing was edited beyond appending this section. No local file was deleted. No new alola fact was found: the scp
brace-expansion catch I hit is already in the skill.
