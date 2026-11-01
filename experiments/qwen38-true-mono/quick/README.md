# Quick end-to-end tok/s check (PLAN decision 14)

The standing progress metric of the true-mono project: ONE job, one GPU, the pinned hybrid ALL stack and candidate
arms back to back, K=3 and K=0, 32 gsm8k prompts. Every work package and every deployable lever ends with a number
from this harness in the same job as the hybrid. Ratios only in this tree; absolute tok/s live in
`hackathon-prep/private-perf/qwen38-true-mono/quick/`.

Harness: `hackathon-prep/qwen38/truemono-quick/` (`quick.sh`, `arms.sh`, `arms.d/`, `summarize.py`, `sync.sh`,
`submit.sh`; the 3a in-situ lookback in `lookback/`; the N8 prep in `n8/`).

## Current best stack (reuse these definitions; updated as levers are gated)

All in `hackathon-prep/qwen38/truemono-quick/` (h stage `h-qk-<sfx>`; ATOM `atom-truemono-<sfx>`; rocKE `rocke-qk-<sfx>`).

| arm (`arms.d/tuned.sh`) | what | status |
|---|---|---|
| `finalG` (K >= 1) / `tunedG` (K = 0) | `lookback/picksG.json` (in-situ lookback picks of delta / qsa / delta_ple r4s and r1, incl. the P3 spread and P10's holds where they won) + N8 96K draft vocab (`n8/ranking.pt`) + the subset-head pick | **gated, both nodes** (current best) |
| `finalH` / `tunedH` | `picksH.json` = picksG + the K=1 / K=2 verify cells (r2s, r3s) | gated (n01-09): K=1 round 0.972, K=2 0.976 |
| `finalGqd` / `tunedGqd`, `finalHqd` / `tunedHqd` | the same + P13 N4a `qsa:qdense=3` (kind group appended by `gen_tuned.py --kind-extra`; expected build lines include it) | e2e gated (n01-09); superseded by finalIqd |
| **`finalIqd` / `tunedIqd`** | `picksI.json` = picksH + the grouped K=3 builds (r8g4, r16g4 schedule picks) + qdense + N8 96K + subset head | **current best, gated**: both nodes at K=0..3 CONC 1 (CONC 2 / 4: a gain on n01-09, inside the noise on n07-17); GSM8K, HumanEval, RULER mixes A and B at 8K / 32K inside the hybrid's spread |

Serving recipe of a `tuned*` arm: `gen_tuned.py --base <e2ecomp2 functions> --picks <picks file> --out <dir>
[--kind-extra qsa:qdense=3]` writes per-kind function dirs, `sched.txt` and `expect_k<K>.txt`; the server env is
`TUNED_ENV` in `arms.d/tuned.sh` (ATOM-truemono + copy-rocm-lib rocKE + rocKE head). `final*` adds
`ATOM_Q38_DRAFT_VOCAB*`. Every lever is default-off: unset, the server is the hybrid.

## How it runs

- **One job, one GPU** (q38 container, `rocke` partition, pinned to n01-09: n07-17's bus e5 GPU is bad). Per K: one
  untimed prime server (fills the job-local compile caches), then `PASSES` passes over the arms; the arm order of a
  pass is a seeded permutation rotated by the pass index. One fresh server per arm-run, one server at a time.
- **Per arm-run**: 2 untimed gsm8k requests (they also JIT the hybrid's extern-linked builds at the served shapes,
  which are never disk-cached), then `single_request_bench`: gsm8k, 32 prompts (the DFlash seed-42 selection, the same
  as `quick_tps.sh`), CONC 1, greedy, reasoning xhigh, 2048-token cap.
- **Arms are env-var sets + stage names** (bash functions `arm_<name>` in `arms.sh` / `arms.d/*.sh`): ATOM tree
  (`$HB` = the pinned hybrid `hb-atom-<sfx>`, byte copy of `atom-rkrepro`; `$TM` = `atom-truemono-<sfx>` from
  `dev/ATOM-truemono`), rocKE stage (`rocke-tmbase` = the pinned hybrid rocKE; `rocke-qk-<sfx>` = copy-rocm-lib),
  a functions recipe (`gen_<recipe>`: `e2ecomp2` = the hybrid's MoE / B3 functions, `tuned*` = lookback picks), the
  server env, and server-log checks (`ARM_HEAD`, `ARM_EXPECT[_K0|_K3]`, `ARM_REJECT`, a recipe's
  `expect_k<K>.txt` build lines). Built in: `mono` (plain FlyDSL mono), `hybrid` (the bar: pinned tree + rocKE +
  `ATOM_Q38_SCHED=best` + rocKE head + fused MoE / B3 functions), `hyb_tm` (the same env on ATOM-truemono +
  copy-rocm-lib rocKE: the base of every candidate arm). Add an arm: drop `arms.d/<x>.sh`.
- **Gates in the job**: a 16 GiB randn health fill first; `PRE_TESTS` (ATOM-truemono test files, GPU tests included)
  must pass before any arm; every server log: mono decode on, no `mono ... off`, 0 tracebacks, rocKE head on iff the
  arm says so, no `numa off for`, the arm's expected / rejected lines (e.g. the exact `[ATOM_Q38_SCHED] build ...`
  lever line of each tuned cell), the extern fallback count (`EXTERN_STAGES] no`) printed; GSM8K correct per run.
- **Metrics** (`summarize.py`, per arm and K, ratios to `hybrid` paired by pass): per-request mean and **median**
  decode tok/s (`req`, `med`), token-weighted and end-to-end tok/s, ms per verify round (`round`; **`rnd_med`** = the
  median over requests of each request's ms per round), ms per token, acceptance, gsm8k correct, and the share of
  requests slowed by a neighbour burst (`slow`).
- **Use the medians.** Other lanes' jobs on the node slow single requests to 0.5-0.8 of the steady rate; one hybrid
  K=0 run had 38 % of its requests below 0.85 x its median: its mean per-request tok/s moved 13 %, its median 0.2 %.
  At K=3 the per-request tok/s also carries the acceptance length, which varies 2-4 % between runs of the same arm
  (the served path is not token-deterministic, WP0a); `rnd_med` resolves about 0.3 % (hybrid vs `hyb_tm`, A-vs-A:
  0.999-1.000) and is the decision metric for lever arms at K=3; tok/s at K=0.
- **Cost**: 3 arms x 2 K x 2 passes = 45 min; 3 arms x 2 K x 4 passes = 70 min; an arm-run is 2-3.5 min.

```bash
cd hackathon-prep/qwen38/truemono-quick
bash sync.sh <sfx> all                     # h-qk-<sfx>, atom-truemono-<sfx>, hb-atom-<sfx>, rocke-qk-<sfx>
bash submit.sh qk-<tag> <sfx> --env "ARMS=mono hybrid hyb_tm"                       # K=3 and K=0, 2 passes
bash submit.sh qk-<tag> <sfx> --env "ARMS_K3=hybrid x y" --env "ARMS_K0=hybrid z" --env PASSES=3 \
     --env "PRE_TESTS=tests/test_qwen4_exp_mono_draft_vocab.py"
# lookback (3a): bash submit.sh qk-<tag> <sfx> --cmd truemono-quick/lookback/job.sh --env "CELLS=delta.r4s@256 qsa.r1@160"
```

Watch `/home_aus/AMD/vanantha/rocke-sdpa-claude/q38/logs/<tag>-<id>.out`; results in `.../q38/results/<tag>/q/`.

## Today's bar (job 68227261 `qk-bar1`, n01-09, 2 passes)

| arm / hybrid ALL | K=3 tok/s per request (median) | K=3 ms per round (median) | K=3 acceptance | K=0 tok/s per request (median) |
|---|---|---|---|---|
| mono (plain FlyDSL) | 0.863 | 1.157 | 1.001 | 0.870 |
| hyb_tm (candidate base) | 1.008 | 0.999 | 1.016 | 0.997 |

The hybrid served at the user's bar level in this job (private store). The hybrid's gain over plain mono matches
`baseline/MATRIX.md` (K=3 round 0.865, K=0 1.142x): here round 0.864 (1 / 1.157), K=0 1.149x.

## Early deployable wins (each gated, each measured against the hybrid in the same job)

### P10 hold drops (W2 probe P10) - job 68228347 `qk-p10`, 4 passes

`ATOM_Q38_SCHED` = `sched.BEST` as a literal string minus holds that sit on stages the fused MoE emptied (server-log
build lines differ from the hybrid's in exactly those levers):

| arm | change | K | metric vs hybrid, mean [min-max] over 4 passes |
|---|---|---|---|
| p10a | - `hold.m4` at delta.r1, delta_ple.r1 | 0 | tok/s per request (median) **1.021 [1.015-1.026]**, 4/4 faster |
| p10b | p10a - `hold.m3` at delta.r4s, delta_ple.r4s | 3 | ms per round (median) 0.996 [0.993-0.998], 4/4 |
| p10c | p10b - `hold.m3` at qsa.r4s | 3 | ms per round (median) 0.995 [0.992-0.997], 4/4 |

### N8: the drafter's argmax over a frequency-ranked vocab subset - jobs 68227697, 68228046, 68228765

ATOM-truemono `atom/models/qwen4_exp_mono/draft_vocab.py` (`ATOM_Q38_DRAFT_VOCAB=<ranking>`,
`ATOM_Q38_DRAFT_VOCAB_N`, default off): only DRAFT forward contexts route `compute_argmax_token` to a re-packed copy of
the first N ranked rows of the MXFP4 head (rows kept in ascending id order, so ties pick the lowest id as the full head
does) and one gather back to token ids; the target's heads are untouched, so every verified token is the target's
own. Gates: 9 CPU + GPU tests (`tests/test_qwen4_exp_mono_draft_vocab.py`: off by default, file formats, subset rules,
only draft argmax routes, subset argmax == subset logits' argmax and == the full head's where its winner is in the
subset, rows 1-16, the served shape K=2560 with the rocKE fused head), 6 of 6 unit mutants killed (map off by one, no
sort, no gather, target routed, no duplicate check, wrong rows). Ranking (`n8/rank.py`): token counts of the hybrid's
replies to 368 prompts DISJOINT from the timed gsm8k test prompts (gsm8k train, MATH-500, MBPP, MT-Bench), special
tokens first, uncounted ids by id. Held-out coverage of reply tokens (gsm8k test 32 / HumanEval 32): 32K 0.987 /
0.985, 64K 0.995 / 0.996, 96K 1.000 / 0.9997. Served audit (`ATOM_Q38_DRAFT_VOCAB_AUDIT`, 64K subset, 11,000 draft
rows): the full head's draft token was in the subset for 0.9958 of rows, and the subset token equalled it for 0.9958.

| arm (vs hybrid, same job) | ms per round (median) | acceptance | tok/s per request (median) |
|---|---|---|---|
| n8 16K (2 passes) | 0.972 | 0.946 | 0.995 |
| n8 32K (2 passes) | 0.970 | 0.970 | 1.019 |
| n8 64K (2 + 3 passes, two jobs) | 0.971 / 0.974 | 0.965 / 0.996 | 1.016 / 1.017 |
| n8 96K (3 passes) | 0.975 | 0.977 | 1.017 |

Acceptance per run varies 2-4 % between runs of one arm; the audit bounds N8's own acceptance cost at 64K to the
0.42 % of draft tokens it changes. The stacks use 96K (the same round time, fewer changed draft tokens).

### 3a: combination lookback over the hybrid's existing knobs (in situ, layer time) - jobs 68228033, 68228647, 68229129, 68229419

`lookback/`: per layer cell, the mono tuner's `phased` strategy (`SignedLookback` with the combination lookback over
EVERY knob and level, solo losers included, then TPE) over one space joining the fused-MoE `FusedKnobs`, the
`B3Knobs` (delta kinds) and ATOM's `ATOM_Q38_SCHED` levers of the cell (`tune.catalog` spaces rebased on the served
sets, so level 0 of every knob is the hybrid). Fixed, not searched: `numa` (the hybrid's structural base) and the
tolerance-class `b3.exp` / head `act`. Objective: the cell's layer time over the hybrid build's, same batch, per round
(`sched_rig.LayerRig`, 16 cold weight copies, 96 launches per graph). Gate before timing: layer output and every
written state bitwise equal to the plain mono build, deterministic; self-test mutants (ATOM `mutate.<stage>`, the MoE's
`combine_swap`) must differ. A device fault (two knob combinations raised HSA aperture violations) blocks the open
configs and resumes the search from its checkpoint (`lookback/guard.py`). Decision: 12-16 shuffled rounds with an
A-vs-A and a recompiled control, the DESIGN 9.1 win rule. The pick's functions + `sched` group are served by
`lookback/gen_tuned.py` (per-kind function dirs, `sched.BEST` with the picked groups, the expected build lines).

| cell | evaluations | pick / hybrid layer | solo-winners-only / hybrid | pick holds solo losers | pick |
|---|---|---|---|---|---|
| delta.r4s | 256 | **0.979** (12/12 WIN) | 0.983 | none | moe `sh_base=160`; sched `hold.m3=64` (was 128), `hold.a2=96`, `b3pf=1` |
| qsa.r4s | 256 | **0.975** (12/12 WIN) | 0.981 | **9** | moe `dn_issue=lds`, `router_knobs=fused`, `line_order=slot`, `sh_join=2`, `mailbox.pause=none`; sched `dyn.q6`, `dyn.q7`, `hold.q2=192`, `hold.q5=96`, `prio.m1`, `prio.q2` |
| delta.r1 | 256 | **0.955** (12/12 WIN) | 0.956 | none | moe `sh_ctas=20`, `m3_ept=4`; sched `hold.m4=0` (P10's lever, found independently), `hold.b1=16`, `prio.m5`, `b3pf=1` |
| qsa.r1 | 160 | **0.989** (12/12 WIN) | = pick | none | moe `sh_ctas=16`, `sh_base=152`, `route_pause=s_nop`, `tw_at=epi`, `line_order=slot`; sched `hold.q7=192`, `hold.m3=128` |
| delta_ple.r4s | 128 + confirm | **0.990** (16/16, pick + P10 `hold.m3=0`) | - | - | b3 `state_issue=after_ba`; sched `hold.m1=16`, `hold.m3=0` (pick alone 0.994, P10 alone 0.995) |
| delta_ple.r1 | 128 + confirm | **0.958** (16/16, pick + P10 `hold.m4=0`) | - | - | moe `map=groups`, `sh_ctas=12`, `dn_issue=after_gu`; b3 `hoist=none`; sched `hold.a2=128`, `hold.m1=96`, `hold.m4=0` (pick alone 0.990, P10 alone 0.971) |
| head (subset 96K, draft argmax) | 64 | **0.984** (16/16 WIN) | 0.985 | - | `korder=lane`, `xpad=64`, `fsync=counter` (draft-side: the gate compares argmax indices; acceptance-gated e2e) |
| head (full, draft argmax) | 64 | 0.991 (16/16 WIN) | - | - | `korder=lane`, `dg=1`, `cpol=sc0`, `waves_per_eu=2` (not served: N8 replaces this head) |
| head (target logits, m4 / m1) | 64 each | NEUTRAL | - | - | the verify head stays as served |

- PLAN decision 13 in action: on qsa.r4s the pick carries 9 levels that each lost alone and beats the best
  combination of solo winners by 0.6 % of the layer, every round.
- The 128-evaluation delta_ple searches never reached the hold levers (the climb spent its budget on MoE / B3 levels);
  measured combinations (`lookback/extra/`: pick, pick + the P10 hold drop, the drop alone) closed that: the
  combination beats both parts on both cells. Budget 256+ per cell from now on.
- 40 / 17 / 18 compile failures (illegal combinations: LDS, plan) and 4 / 21 bitwise failures (delta.r4s / delta.r1)
  were refused by the gate, never timed.

### Stacked, end to end

| stack (vs hybrid, same job) | K=3 ms per round (median) | K=3 tok/s per request (median) | K=0 tok/s per request (median) | job |
|---|---|---|---|---|
| tuned (delta.r4s pick) | 0.991 | 1.012 | - | 68228765 |
| tuned + N8 96K | 0.967 | 1.034 | - | 68228765 |
| tuned + N8 96K (repeat) | 0.967 | 1.029 | - | 68229209 |
| p10a | - | - | 1.021 | 68229209 |
| tunedB (r4s + r1 picks, delta_ple P10 drops) | 0.988 | 0.999 | **1.029** | 68229209 |
| **tunedB + N8 96K** | **0.961** | **1.052** | - | 68229209 |
| tunedB + N8 96K (repeat) | 0.963 | 1.046 | 1.031 (tunedB) | 68229491 |
| tunedD (+ delta_ple pick + P10 combinations) + N8 96K | 0.963 | 1.041 | 1.030 (tunedD) | 68229491 |
| **tunedD + N8 96K + subset-head pick** (final stack) | **0.962** | **1.054** (mean 1.047) | **1.030** | 68229491 |

Final stack = `arm_tunedD_n8_96k_hd` at K=3 and `arm_tunedD` at K=0 (`arms.d/tuned.sh`, `lookback/picksD.json`):
serve it with the hybrid ALL env on ATOM-truemono + copy-rocm-lib rocKE, `ATOM_Q38_EXTERN_STAGES` pointing at the
per-kind dirs `gen_tuned.py` writes, `ATOM_Q38_SCHED=$(cat <dir>/sched.txt)`, `ATOM_Q38_DRAFT_VOCAB=n8/ranking.pt`,
`ATOM_Q38_DRAFT_VOCAB_N=98304`, `ATOM_Q38_DRAFT_VOCAB_SPEC` = the subset-head pick. Every piece is default-off: unset,
the server is the hybrid. The delta_ple picks (1 of 48 layers) are inside the end-to-end noise (tunedB vs tunedD);
the subset-head pick moves the round by 0.001 (inside the noise) - both are kept for their layer / kernel wins.

Against the headline targets (decision 7): the final stack is a verify round of about 0.96 of the hybrid's (target
<= 0.875) and about 0.97 of its per-token time at K=0 (target <= 0.78). These are the hybrid's own knobs plus N8:
the gap is the true-mono work (D0 onwards).

GSM8K correct stayed within the hybrid's own run-to-run spread in every arm (28-32 of 32; private store per run).

## Caveats

- 32 prompts x 2-4 passes resolve about 0.3 % of round time and about 1 % of K=0 tok/s (medians); per-request tok/s
  at K=3 carries 2-4 % acceptance noise per run, so a K=3 tok/s claim below about 2 % rests on the round time.
- One node (n01-09), other lanes' jobs on the other GPUs; never compare across jobs (each table row is one job).
- The head picks marked draft-side change the draft logits' summation order (not bitwise to the mono head); they
  are judged on acceptance + round time end to end, like N8 (DESIGN 12.1 rule 3).
- Follow-up results (second node, accuracy, CONC / K cells, the 256-budget delta_ple searches, the P3 spread)
  are in the next section.

## Follow-up: second node, accuracy, other cells, restack, P3 spread (2026-10-31 afternoon)

All against the pinned hybrid ALL in the same job; medians as above.

### Second rocke node (job 68229998 `qk-n7conf`, n07-17, GRES idx 0 / bus 5, not the bad e5; 4 passes)

| arm | K=3 ms per round (median) | K=3 tok/s per request (median) | K=0 tok/s per request (median) |
|---|---|---|---|
| final (tunedD + N8 96K + subset-head pick) / tunedD at K=0 | **0.963** [0.960-0.966], 4/4 | 1.054 [1.036-1.077] | **1.028** [1.026-1.030], 4/4 |

The n01-09 result (0.962 / 1.030) holds on the other node. The spread stack (finalF / tunedF, below) on n07-17
(job 68233533 `qk-n7F`, GRES idx 5, 4 passes): K=3 round **0.959** [0.956-0.962], tok/s per request 1.032 (median);
K=0 tok/s per request **1.062** [1.059-1.070], 4/4 - the same as on n01-09 (0.959 / 1.061).

### Accuracy (PLAN decision 1), K=3 CONC 4, final vs hybrid (jobs 68230565 `qk-acc`, 68231490 `qk-acc2`)

| suite | runs (hybrid / final) | hybrid range | final - hybrid (mean) | verdict |
|---|---|---|---|---|
| lm_eval GSM8K 1,319, 5-shot chat, flexible-extract | 3 / 3 | spread 0.006 | +0.0010 | inside |
| same, strict-match | 3 / 3 | spread 0.004 | +0.0028 | inside |
| HumanEval 164, pass@1 (low reasoning) | 7 / 7 | spread 0.055 | -0.004 | inside (per-run stderr about 0.033) |
| RULER 8K, 5 tasks, 256-token cap (job qk-acc2) | 4 / 4 | spread 0.026 | -0.027 | all 4 final runs below all 4 hybrid runs: see below |
| RULER 8K, 4 tasks, 1536-token cap (job 68231933 `qk-ruler`) | 3 / 3 | - | +0.007 | inside |

The qk-acc2 RULER gap sits entirely in `niah_multiquery`. With a 256-token cap the reasoning model's answer often never
starts (the reply content is empty in most samples, reasoning consumed the cap), so that task scores "did the
reasoning end within 256 tokens": a run-to-run lottery (hybrid vs hybrid: 4-5 of 25 replies identical). With a 1536-token
cap every arm scores about 1.0 on it (`qk-ruler`: hybrid, final, tunedE = picks only, n8hd_96k = N8 + subset head
only, 3 passes each: 0.975-1.000 overall). 32K was never evaluated in qk-acc2 (with `--limit`, lm_eval takes the first
N docs of a task, i.e. only the first length; `quick.sh` now runs one lm_eval per length). Both mixes at 8K AND 32K, 4
passes each, are in the RULER section below.

Why outputs can move at all: every lookback pick is bitwise-gated against the mono layer, so the target's numerics do
not change; N8 and the subset head change only the drafts. Greedy MTP is output-exact only if the verify pass is
batch-invariant; the hybrid itself is not run-to-run deterministic (WP0a: 3 of 64 identical texts hybrid vs hybrid),
so other drafts can move near-tie tokens like a re-run does.

### Other cells (job 68230373 `qk-conc`, 3 passes; the grouped / wide builds keep the FlyDSL MoE and B3)

| cell | what of the stack is active there | ms per round (median) | tok/s per request (median) | output tok/s (e2e) |
|---|---|---|---|---|
| K=1 CONC 1 (`r2s`) | N8 + subset head (1 draft head per round), qsa.r1 drafter pick | 1.003 | 1.007 | 0.998 |
| K=2 CONC 1 (`r3s`) | same (2 draft heads) | 0.999 | 1.009 | 1.003 |
| K=3 CONC 1 | everything | 0.962 | 1.039 | 1.030 |
| K=3 CONC 2 | r4s picks while one request is in flight, N8 (r16 subset build) | 0.995 (all rounds) | 1.104 (noisy) | 1.028 |
| K=3 CONC 4 | as CONC 2 | 0.978 (all rounds) | 1.046 | 1.033 |
| K=0 CONC 4 (`r1..r4`) | r1 picks only | - | 1.007 | 0.973 [0.933-1.036] |

At K=1 / K=2 nothing measurable transfers: their verify builds (`r2s`, `r3s`) have no picks yet and N8's one or two
draft heads per round are inside the noise. K=0 CONC 4 is neutral within its noise. Next lookback cells for these:
`r2s`, `r3s`, `r2..r4` (and the wide builds once rocKE owns their MoE / B3).

The same cells for the spread stack (finalF / tunedF; job 68233567 `qk-concF`, 3 passes):

| cell | ms per round (median) | tok/s per request (median) | output tok/s (e2e) |
|---|---|---|---|
| K=1 CONC 1 | 1.005 [1.002-1.010] | 0.996 | 0.951 |
| K=2 CONC 1 | 1.004 [1.000-1.006] | 0.996 | 1.003 |
| K=3 CONC 1 | **0.962** [0.960-0.963] | **1.050** | 1.031 |
| K=3 CONC 2 | 0.987 (all rounds) | 1.058 (noisy) | 1.033 |
| K=3 CONC 4 | 0.996 (all rounds) | 1.063 (noisy) | **1.070** [1.033-1.139] |
| K=0 CONC 1 | - | **1.060** [1.057-1.061] | 1.060 |
| K=0 CONC 4 | - | 0.985 | 1.039 [0.959-1.127] |

K=1 / K=2 read slightly slower than the hybrid in this job (3/3 passes above 1 at K=1). Their verify builds carry no
pick; what differs is N8 + the subset head (one / two draft heads per round) and the candidate base (ATOM-truemono,
the copy-rocm-lib rocKE snapshot of the stage for the unpicked variants' functions). Ablation: job `qk-k12abl`
(below).

K=1 / K=2 ablation (job 68234270 `qk-k12abl`, 3 passes, ms per round median vs hybrid): K=1 finalF 1.002, tunedF
(picks only) 1.000, N8 + subset head only 0.997, hyb_tm 0.998; K=2 0.998 / 1.000 / 1.001 / 1.002. Every arm is at
parity: qk-concF's K=1 1.005 was noise, and nothing of the stack transfers to `r2s` / `r3s` yet (no picks there; N8's
saving on one or two draft heads per round is below 0.3 %).

### delta_ple at budget 256, restack (jobs 68230032 `qk-lbE`, 68231050 `qk-restack`)

The 256-evaluation searches beat the "pick + P10 drop" combinations in the same confirmation rounds: delta_ple.r4s
0.983 (was 0.992), delta_ple.r1 0.951 (was 0.957); qsa.r1 at 256 (seed 1) tied the earlier pick (0.9905 vs 0.9908),
which stays. End to end the restack (`finalE` / `tunedE`) equals the previous stack within the noise (round 0.962 /
0.962; K=0 1.030 / 1.029): one layer of 48.

### P3 HC spread (W2 probe P3) on top of the picks (jobs 68231967 `qk-lbF`, 68232031 `qk-stackF`)

`spread.a1=10` built extern-linked for the first time (bitwise-gated against the mono layer in the lookback rig, the
`mutate` mutants caught), confirmed in situ on top of each pick (16 rounds, both controls):

| cell | pick | pick + spread | spread alone | kept |
|---|---|---|---|---|
| delta.r1 | 0.958 | **0.929** | 0.970 | pick + `spread.a1=10` |
| qsa.r1 | 0.994 | **0.964** | 0.975 | pick + `spread.a1=10` |
| delta_ple.r1 | 0.955 | **0.943** | 0.977 | pick + `spread.a1=10` |
| qsa.r4s (`spread=2`) | **0.971** | 0.973 | 0.995 | pick |
| delta.r4s (`spread.a1=2`) | **0.980** | 0.993 | 0.991 | pick |

The spread knobs (`spread.a1`, `spread.m1`: 0, 1, 2, 3, 4, 10) are now in the lookback pool of every narrow cell.

| stack (job 68232031, 4 passes) | K=3 ms per round (median) | K=3 tok/s per request (median) | K=0 tok/s per request (median) |
|---|---|---|---|
| finalE / tunedE | 0.962 | 1.037 | 1.028 [1.027-1.030] |
| **finalF / tunedF (+ spread)** | **0.959** [0.957-0.961] | **1.043** | **1.061** [1.058-1.063], 4/4 |

### RULER, both mixes, 8K and 32K, finalF vs hybrid (jobs 68232435 `qk-rulA`, 68232436 `qk-rulB`; 4 passes each)

One lm_eval per length (25 samples per task and length), K=3 CONC 4, `--max-model-len 40960`; score = mean over tasks.

| mix | length | final - hybrid (mean over 4 passes) | hybrid A-vs-A spread | verdict |
|---|---|---|---|---|
| A = qk-acc2's (single_1, single_2, multikey_1, multiquery, vt), 256-token cap | 8K | **0.000** | 0.022 | inside |
| A | 32K | -0.022 | 0.063 | inside |
| B = niah-heavy (multiquery, multivalue, vt, single_1), 1536-token cap | 8K | +0.005 | 0.008 | inside |
| B | 32K | -0.005 (one final pass 0.97, three 1.00; hybrid 0.99-1.00) | 0.010 | inside the per-sample noise (3 of 100 samples) |

- The qk-acc2 gap (mix A, 8K, -0.027, 4/4 passes) did **not** reproduce on the full stack with 4 new passes per arm
  (delta 0.000), so no lever ablation was run; the bisection at 8K / 1536 tokens (`qk-ruler`: picks only, N8 + subset
  head only, both, hybrid) is flat too.
- Mix A measures mostly the 256-token cap: `niah_multiquery` (0.28-0.39) and `ruler_vt` (0.00-0.07) are about "did the
  reasoning finish within the cap"; under mix B (1536 tokens) the same tasks score 0.99-1.00 in every arm. A RULER
  gate for this reasoning model needs a cap that lets the answer start; P11 (W5) defines the canonical mix.

### Re-tests on the best stack and picksG (PLAN decision 15; jobs 68234933 n01-09 and 68234935 n07-17, 3 passes each)

Every lever not taken earlier is a verdict ON A BASE; here the base is finalF (K=3) / tunedF (K=0), both nodes in
parallel. K=3 = ms per round (median over requests) vs the hybrid, n01-09 / n07-17; K=0 = tok/s per request (median).

| arm on base finalF / tunedF | earlier verdict (base) | K=3 round | K=0 tok/s | verdict on base finalF / tunedF |
|---|---|---|---|---|
| finalF / tunedF (the base) | - | 0.960 / 0.961 | 1.059 / 1.061 | - |
| finalG / tunedG: picksG (delta.r1, delta_ple.r1 re-searched at 256 with the spread knobs in the pool, `qk-lbG`: 0.928 / 0.932 vs 0.932 / 0.941 of the layer) | - | 0.963 / 0.962 | **1.060 / 1.064** | layer WINs, end to end inside the noise (+0.1-0.3 % at K=0); kept as the new base |
| N8 64K (served subset spec) | not chosen (base hyb_tm, 96K preferred) | 0.957 / 0.960 | - | ties 96K (within 0.3 %); 96K stays (fewer changed drafts) |
| N8 64K + subset-head pick | - | 0.957 / 0.959 | - | ties |
| N8 32K | not chosen (base hyb_tm) | 0.959 / 0.958 | - | same round, acceptance 0.984-0.986: tok/s lower; rejected on base finalF |
| verify-head pick logits m4 `cpol=sc0` | NEUTRAL (base: the hybrid head, layer rig) | 0.965 / 0.961 | - | no gain; rejected on base finalF |
| verify-head pick logits m1 `sched=q` | NEUTRAL (base: the hybrid head) | - | 1.060 / 1.062 | neutral; rejected on base tunedF |

Still open per decision 15 (rejected on an older base, not yet re-tested on finalF): the qsa.r4s `spread=2` and
delta.r4s `spread.a1=2` (lost on top of their picks in situ, `qk-lbF`; their base has not changed), the P10 r4s
hold drops (superseded by the r4s picks), N8 16K (base hyb_tm).

### K=1 / K=2 verify cells, decision-15 re-tests on base picksG, P13 qdense (2026-11-01; n01-09)

Lookbacks (updated tuner: base-relative re-tests, revisit pass, 2 extra lever orders; budget 256, delta_ple 128),
jobs 68235802 `qk-lbH`, 68235808 `qk-lbI`, layer time vs the hybrid build, 16 confirmation rounds:

| cell | pick / hybrid layer | pick |
|---|---|---|
| delta.r2s | 0.966 (WIN) | sched holds a2 / b1 / m1-m3, `pipe=1`, `prio.b3`, `spread.a1=2` |
| qsa.r2s | **0.935** (WIN) | `dyn.q6`, `head_sleep=128`, `hold.m3=96`, `spread.a1=10` |
| delta_ple.r2s | 0.983 (WIN) | `hold.b1=16` |
| delta.r3s | 0.975 (WIN) | holds m1-m3, `spread.a1=2` |
| qsa.r3s | 0.959 (WIN) | `dyn.q7`, holds q1 / q6, `prio.m5`, `spread=2` |
| delta_ple.r3s | 0.9985 (NEUTRAL) | the served set stays |

End to end (job 68236246 `qk-k12H`, 3 passes; ms per round median, tok/s per request median, vs the hybrid):
**K=1 finalH 0.972 / 1.030**, **K=2 finalH 0.976 / 1.021** (finalG, without r2s / r3s picks: 1.002 / 0.995 and 0.997 /
1.002). The K=2 server-log check flagged the arm because `gen_tuned.py` wrote every expected build line into the K=2
file (a loop-variable bug, fixed); the K=1 / K=2 / K=3 build lines in `builds.txt` match the picks exactly.

Decision-15 re-tests on base picksG, in situ (job 68236222 `qk-lbJ`, all subsets in the same rounds):
delta.r4s pick 0.980 vs + `spread.a1=2` 0.994, + P10 `hold.m3=0` 0.987, + both 0.999; qsa.r4s 0.970 vs + `spread=2`
0.979, + `hold.m3=0` 0.974, + both 0.977; delta_ple.r4s 0.982 vs + `hold.m3=0` 0.987, + `spread.a1=2` 0.990, + both
0.990: every add-on loses on base picksG, alone and together. End to end, N8 16K on finalG: round 0.954 but acceptance
0.966, tok/s per request 1.019 vs finalG's 1.063: rejected on base finalG.

P13 N4a `qsa:qdense=3` (W5) stacked (job 68236742 `qk-qd1`, 4 passes): **K=3 finalHqd round 0.948 [0.945-0.953], tok/s
per request 1.045 (median; mean 1.060)**; **K=0 tunedHqd 1.089 [1.082-1.092]** (finalG / tunedG in the same job: 0.961
/ 1.060). Tolerance class: the accuracy suites for finalHqd are queued.

### Grouped / wide K=3 builds (CONC 2-4): schedule-lever lookbacks, picksI (jobs 68236222 `qk-lbJ`, 68237321 `qk-lbK`)

`lookback/grp_rig.py`: the in-situ rig for grouped verifies (`MonoRig` with `max_groups`, the group runner's tables and
wide scratch), bitwise against the plain grouped mono build; the space is the schedule levers a grouped build takes
(holds, head sleep, pause, pipe, prio; ATOM drops numa / dyn / place / spread there; the MoE / B3 stay FlyDSL).

| cell | evaluations | pick / hybrid layer |
|---|---|---|
| delta.r8g4 (CONC 2, and CONC 3-4 while 2 requests are in flight) | 160 | 0.978 (WIN) |
| qsa.r8g4 | 160 | 0.987 (WIN) |
| delta.r16g4 (CONC 4) | 128 | 0.990 (WIN) |
| qsa.r16g4 | 128 | 0.998 (NEUTRAL) |

(The first confirmation crashed in `run.py`'s pick record for a space without MoE; fixed, and the checkpoints were
resumed in `qk-lbK` - `lookback/ckpt/`, restored by `job.sh`.)

### Stack with qdense and the grouped picks (job 68237407 `qk-concI`, n01-09, 3 passes, vs the hybrid)

| cell | finalHqd / tunedHqd | **finalIqd** (+ grouped picks) |
|---|---|---|
| K=3 CONC 1: round (median), tok/s per request (median) | 0.949, 1.057 | **0.947, 1.066** |
| K=3 CONC 2: round (all rounds), output tok/s | 0.997, 1.034 | **0.964**, 1.031 |
| K=3 CONC 4: output tok/s | 1.066 | **1.088** [1.026-1.174] |
| K=0 CONC 1: tok/s per request (median) | **1.092** | (= tunedHqd) |
| K=0 CONC 4: output tok/s | **1.094** [0.999-1.204] | (= tunedHqd) |

### finalIqd on n07-17 and its accuracy (jobs 68238041 `qk-qd3n7` n07-17 GRES idx 1, 68238294 `qk-acc4` n01-09; 3 passes)

| cell (n07-17) | finalIqd / tunedIqd vs hybrid | finalH (no qdense) |
|---|---|---|
| K=3 CONC 1: round (median), tok/s per request (median) | **0.947, 1.040** (mean 1.057) | - |
| K=0 CONC 1: tok/s per request | **1.091** [1.090-1.093] | - |
| K=1 CONC 1: round, tok/s | **0.951, 1.041** | 0.973, 1.024 |
| K=2 CONC 1: round, tok/s | **0.960, 1.049** | 0.980, 1.018 |

The n01-09 values (K=3 0.947 / K=0 1.092) hold on n07-17; qdense adds about 0.02 of the round at K=1 / K=2 too.

Accuracy, finalIqd vs hybrid (K=3 CONC 4, 3 runs each): GSM8K 1,319 flexible +0.0020 (hybrid spread 0.0045), strict
+0.0033 (0.0061): inside; HumanEval -0.004 (spread 0.061): inside; RULER mix A (256-token cap) 8K -0.007, 32K +0.016:
both inside mix A's measured A-vs-A spread (0.022 / 0.063 over 4 hybrid passes in `qk-rulA`; this job's 3 hybrid passes
happened to agree within 0.004 at 8K). RULER mix B (1536-token cap, the one that tests retrieval) is running
(`qk-rulBq`).

## Consolidated stack (2026-11-01, agent "cs": everything deployable, one served stack)

The best of every verified deployable win in one server, measured increment by increment against the same-job
hybrid on both rocke nodes, then the full-stack accuracy gates and the PLAN decision-15 re-tests on it. Ratios only;
absolute tok/s in `hackathon-prep/private-perf/qwen38-true-mono/quick/CONSOLIDATED.md`. Arms: `arms.d/cs.sh`; gates:
`hackathon-prep/qwen38/truemono/CS/`; coordination with the quick track: `truemono-quick/COORD-consolidated.md`.

### The tail provider (Path B tail, D0 WP2e / WP4B direction): `qwen38_mono/truemono_tail.py`

WP4h's hook (`ATOM_Q38_TRUE_MONO=tail`, ATOM-truemono unchanged) loads the default provider module
`kernels.gfx950.qwen38_mono.truemono_tail:provider` (copy-rocm-lib, new file; P9's code reused, not forked):

- **K=3, one request (`r4s`)**: P9's verify chain `best_spec("verify")` as ONE launch: final HC mix, MXFP4 head with
  the fused greedy argmax, the accept (sampled tokens, verdict, the drafter's `input_ids` scatter), the draft front of
  the 4 rows into the draft layer's input and `write_draft_meta`. The decode graph holds no head (`owns_logits`);
  `propose` runs ATOM's drafter, its step-0 forward skipping the front + meta the launch wrote (one override of the
  draft runner's `forward`, restored).
- **K=0, one request (`r1`)**: the rocKE fused-argmax head (P2's lever) instead of the graph's logits head + `topk_select`.
- **Draft chain** (`ATOM_Q38_TM_TAIL_DRAFT=p9`, default off): every draft step's tail as a P9 draft launch (anchor
  select, the drafter's mix, the N8 subset head + id map, `qsa_draft_decode_metadata`, next front + meta; last step
  head only) inside a restricted copy of `EagleProposer.propose`. Needed three default-off P9 knobs (`p9_tail.py`):
  `vsub` (subset head + `vmap` parameter), `pbump` (mRoPE's three position rows), `tokonly` (no FRONT, task 0 stores
  the token). The default P9 builds lower to byte-identical LLVM IR with the knobs off (6 builds, `CS/p9_ident.py`).
- Every other width (CONC 2-4 grouped verifies, `r2`-`r4`) is refused per width: the hybrid serves it.
- Linked levers: `ATOM_Q38_TM_TAIL_P9` / `_P9D` (P9 knob overrides), `_K3` / `_K0` (off switches).

**Gates (served inputs, both K, `CS/cs_gate.sh`; jobs cs-gate3a/3b/4a/5a, n07-17):**

| check | result |
|---|---|
| ATOM check mode: tail verify == hybrid verify, bitwise (64 steps) | 64 / 64 (verify-only and with the draft chain) |
| K=0 tail sample == hybrid sample, bitwise | 64 / 64 |
| deep check: the launch's front, counter, pos, QSA table vs ATOM's `draft_front` / `write_draft_meta` | 64 / 64 each |
| draft chain deep check, steps 0-1: token, `qsa_draft_decode_metadata` (6 outputs), front, counter, pos, table; step 2 token | 64 / 64 each |
| served N8 subset head (tuned pick) vs P9's head, draft token per step | equal 64 / 64 at every step (informational) |
| transitions in one server (greedy -> sampled -> CONC 2 -> greedy), P9 alias geometry | offset 1 / overlap, 12-13 transitions, 0 tracebacks, gsm8k 8 / 8 |
| must-fail mutants | caught: accept off-by-one (15/16 DIFF), alias race (13/16), front ids (`fids`, 16/16), front lost (16/16), meta word (16/16), K=0 token (16/16), draft MFMA order (12/16), draft id map (16/16), draft position bump (16/16). P9's `stale_ids` build faults (verdict never written) instead of failing the data gate: replaced by `fids`. |

### Increments, same job as the hybrid, 4 rotated passes per node (jobs 68237522 n01-09, 68237911 n07-17)

K=3 = ms per round (median over requests), K=0 = tok/s per request (median); n01-09 / n07-17; per-pass mean, passes
the right way.

| increment | K=3 round | K=0 tok/s | verdict |
|---|---|---|---|
| hybrid -> quick best stack (finalG / tunedG) | 0.961 / 0.962 (4/4, 4/4) | 1.061 / 1.063 (4/4) | WIN |
| + qdense (finalHqd / tunedHqd) | 0.984 / 0.983 (4/4) | 1.026 / 1.027 (4/4) | WIN |
| + P9 verify tail (K=0: fused-argmax tail) | 0.999 (3/4) / 1.000 (2/4) | 1.003 (3/4) / 1.003 (4/4) | K=3 NEUTRAL on base finalHqd (0.996, 4/4 on base finalG, job 68236978); K=0 small WIN |
| + P9 draft chain | 1.003 (0/4) / 0.999 (2/4) | - | LOSS / NEUTRAL: stays default-off (`full3_d`) |
| **full stack vs hybrid** | **0.945 / 0.946** | **1.092 / 1.095** | tok/s per request (median) at K=3 1.058 / 1.071 |

Full stack = finalHqd + the verify tail (K=3), tunedHqd + the K=0 tail (K=0): arms `full3` / `full0`.

### Accuracy, full stack vs hybrid at K=3 CONC 1 (the tail engaged; job 68238265)

| suite | full - hybrid (mean of 3 runs each) | hybrid A-vs-A spread | verdict |
|---|---|---|---|
| GSM8K 1,319 5-shot chat, flexible / strict | -0.0010 / -0.0015 | 0.0061 / 0.0053 | inside |
| HumanEval 164 pass@1 | +0.022 | 0.006 (3 runs; 7-run spread of the quick track 0.055) | no regression |

K=1 / K=2 on n01-09 for finalIqd (job 68240268 `qk-k12q`, 3 passes, every server-log check passing with the fixed
expected-line files): **K=1 round 0.949, tok/s per request 1.058**; **K=2 round 0.955, 1.041** (finalH: 0.971 / 1.030,
0.978 / 1.020). Same as n07-17.

RULER mix B (1536-token cap) for finalIqd (job 68240263 `qk-rulBq`, 3 passes): 8K +0.005 (passes 0.98 / 1.00 / 1.00 vs
hybrid 0.988-0.990; the low pass is 2 of 100 samples), 32K -0.004 (0.990-0.998 vs 0.990-1.000): no regression. With
GSM8K, HumanEval and mix A inside, the qdense stack passes the accuracy suites of PLAN decision 1 at this sample size.
| RULER mix A (5 tasks, 256-token cap) 8K / 32K | +0.008 / -0.017 | 0.001 / 0.062 | 8K above the (3 nearly identical) hybrid runs, upward; 32K inside |
| RULER mix B (4 tasks, 1536-token cap) 8K / 32K | -0.003 / +0.008 | 0.020 / 0.020 | inside |

RULER: job 68240261 (n07-17), 25 samples per task and length, one lm_eval per length, 3 passes per arm. Every accuracy
arm ran at CONC 1 so the tail served its steps (at CONC 4 the grouped widths are the hybrid's). Verdict: the full
stack passes the accuracy gates (no suite below the hybrid's A-vs-A spread).

### Decision-15 re-tests on the full stack (base `full3`; job 68240346 n01-09, 3 passes; ms per round median, arm / base)

| lever (earlier verdict and base) | round | verdict on base full3 |
|---|---|---|
| N8 64K (tied 96K on base finalF) | 1.001 (1/3) | NEUTRAL |
| N8 32K / 16K (lost on finalF / hyb_tm: acceptance) | 0.998 / 0.999, acceptance 0.988 / 0.970, tok/s 0.966 / 0.953 | LOSS (acceptance) |
| P10 r4s hold drops (superseded by the r4s picks) | 1.0035 (0/3) | LOSS |
| r4s spread variants (lost on their picks, qk-lbF) | 1.0073 (0/3) | LOSS |
| P15 N3b router form (lost in situ, P15) | 1.0432 (0/3) | LOSS |
| verify-head picks: replaced at CONC 1 by P9's head; its levers `hdelay` 0 / 16, `head_dg` 3 | 0.998 (2/3) / 0.999 / 1.001 | NEUTRAL |

CONC cells on n07-17 for finalIqd (job 68240805 `qk-concIn7`, GRES idx 3, 3 passes): K=3 CONC 1 round 0.946, tok/s
1.054 (as n01-09); **K=3 CONC 2 output tok/s 0.986 [0.932-1.047], CONC 4 1.018 [0.959-1.108], K=0 CONC 4 1.003
[0.988-1.033]**: the CONC 2 / 4 gains of n01-09 (1.031 / 1.088 / 1.094) are NOT confirmed on the second node; both
sit inside a per-pass spread of about +-7 %. The CONC > 1 cells need more prompts / passes (or the batch harness) to
decide; the grouped picks stay as layer-level WINs (bitwise) pending that.

Second round, both nodes (jobs 68240983 n01-09 / 68241175 n07-17, 3 passes each; arm / full3 or full0):

| lever | n01-09 | n07-17 | verdict |
|---|---|---|---|
| P9 verify `hdelay` 0 (P9's isolated best was 8) | 0.9979 (2/3; 0.9980 2/3 in rt3) | 0.9986 (3/3) | lean WIN on both nodes (7 of 9 passes lower): **adopted** as the served default (`truemono_tail.SERVED_VERIFY`; a sleep before the head's first panels, numerics unchanged) |
| `hdelay` 0 + N8 64K | 0.9984 (2/3) | 1.0002 (2/3) | NEUTRAL (no gain over hdelay 0 alone) |
| `hdelay` 0 + `head_dg` 3 | 1.0002 (1/3) | 1.0017 (1/3) | NEUTRAL |
| draft chain (`ATOM_Q38_TM_TAIL_DRAFT=p9`) | 0.9979 (2/3) | 1.0024 (1/3) | NEUTRAL (and 1.0027 / 0.9992 in the increments jobs): stays off |
| draft chain + draft `hdelay` 8 | 0.9989 (2/3) | 1.0015 (1/3) | NEUTRAL |
| K=0: P15 N3b at delta r1 | 0.988 (0/3) | 0.986 (0/3) | LOSS |
| K=0: argmax head `sched=q` / `cpol=sc0` (the verify-head pick levels, bitwise) | 1.0007 / 1.0014 | 0.9990 / 0.9992 | NEUTRAL |

Every lever above stays a default-off knob (arms `full3_*` / `full0_*`, `picksCS_*.json`); verdicts are "on base full3 /
full0" (PLAN decision 15). Same-job full stack vs hybrid in these jobs: K=3 round 0.948 / 0.947, K=0 tok/s 1.093 /
1.095.

### Where the consolidated stack stands against the targets (decision 7 / 12)

K=3 CONC 1: verify round about 0.946 of the hybrid's (target <= 0.875 at equal acceptance for 1000 tok/s); K=0:
about 0.914 of its per-token time (target <= 0.78). The layer kernels (79 % of the round) are still the hybrid's
FlyDSL mono with tuned rocKE MoE / B3 functions; the tail work this round moved the non-layer share only slightly
(served, the verify tail is neutral-to-0.2 % on the qdense base and P9's draft chain is neutral against the tuned N8
draft head). The gap is the true-mono layer work (WP3 / WP4B), not more lever stacking: none of the re-tested levers
pays on the full stack.

### Why the tail does not pay end to end, and the restructure tried (2026-11-01 evening)

P9 measured the verify chain at 0.795 and the draft chains at 0.848 / 0.890 of ATOM's kernel sequence in isolation;
served, the verify tail is 0.999-1.000 and the draft chain 0.999-1.003 of the round. Measured, in order:

1. **The served segments are what P9 assumed, and the tail does shorten them on the device.** Profiled round traces
   (offline driver, rocprofv3, `CS/cs_trace.sh` + `cs_round.py`; jobs 68241753 / 68241890): the verify segment is
   0.045 of the round with the hybrid (P9's G3 table: 0.041) and 0.038-0.040 with the tail (the P9 kernel itself 0.029);
   the draft chain cuts the two draft tails from 0.024 + 0.019 to 0.017 + 0.013. Device round stamps without a
   profiler (`CS/cs_stamps.sh`, job 68242174, 2 passes): post-target 1.000 -> 0.85, between-drafts (lean round, below)
   1.000 -> 0.70; target layers unchanged (no contention with the layer kernels); device round wall 0.993 (verify
   tail) and 0.986 (lean round) of the base's.
2. **The savings do not reach the token rate: the round's binding path is the host's round boundary.** In the profiled
   traces every tail saving reappears as GPU idle (per round: busy -38 / -83 / -86 us, idle +83 / +57 / -11 us); the
   boundary (last draft step + next round's prologue) carries 0.07-0.08 of the round as GPU idle in every arm, about
   100 us of it while the host is inside the next round's `hipGraphLaunch`. Host-stall test (`ATOM_Q38_TM_TAIL_HOSTUS`,
   jobs 68242291 / 68242298, 3 passes): 100 / 300 us of host busy-wait right after the verify launch cost 1.000-1.001
   / 0.999-1.005 of the round, so the host has slack there; added GPU time on the verify chain (`hdelay` 0 -> 16)
   shows at about 20 %. The tail sits off the critical path: the round is set by the host's work after it reads the
   verify result, through to the next graph launch.
3. **Restructure tried: the lean draft round** (`ATOM_Q38_TM_TAIL_DRAFT=lean`): the draft round as GPU launches only
   (positions, the 4-row draft layer, per step a P9 draft launch and the 1-row layer, the last head-only launch), no
   host-side ATOM drafter code (no `_publish_draft_shape`, metadata objects, FlyDSL plan refresh: no draft kernel
   reads them). Gate (job cs-gate7): ATOM check mode 64 / 64, every draft-chain deep check 64 / 64 (token, the six
   `qsa_draft_decode_metadata` outputs, front, counter, pos, table, last token), must-fail position mutant caught
   16 / 16, gsm8k 8 / 8. Device: between-drafts 0.70, device round 0.986. End to end (jobs 68241943 / 68241946, 4
   passes, arm / finalHqd ms per round median): **1.0007 / 1.0003**; in the host-stall jobs 1.0010 / 0.9970.
   NEUTRAL: stays a default-off knob.
4. **hdelay 0 default re-gated and confirmed:** gate job cs-gate6 (ATOM check mode 64 / 64, K=0 64 / 64, deep checks
   64 / 64, must-fail accept / front-ids / meta mutants caught) PASS; end to end full3 (hdelay 0) / finalHqd 0.9998 /
   0.9992 (lean jobs), unchanged from the hdelay 8 build (0.999 / 1.000).

**What this means for the plan** (D0.md section 6 item 7): G3 (P9's 0.018 hu) and WP5a / WP5b book about 0 while the
round boundary goes through the host. The tail's device wins (about 0.6 % verify, 1.4 % with the lean round) become
round time only when the next round no longer waits for the host: a whole round launched without the host in between
(DESIGN `round` / `loop` modes, WP5c / WP12; an in-graph next-round prologue) or an ATOM-side prepare-ahead of round
t+1 while round t runs. The boundary idle is 0.07-0.08 of the round, so that work is worth far more than DESIGN's
"about 0.01, optional" for WP12.
