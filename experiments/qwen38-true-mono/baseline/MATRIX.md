# Serving-matrix baseline: plain mono vs the hybrid ALL stack (gfx950, TP1, MXFP4)

What a true rocKE monokernel has to beat, cell by cell. Arm **A** is plain FlyDSL mono (ATOM `352229d6` mono
path, no knobs). Arm **ALL** is the hybrid of `Qwen3.8-Flash-Next-rocKE.md`, exactly as `e2ecomp2` / `quick_tps.sh`
served it: `ATOM_Q38_SCHED=best`, `ATOM_Q38_ROCKE_HEAD=1`, fused MoE (m3/m4/m5) and B3 / B3-prologue functions
from `stage_moe_fused --knobs best` / `stage_b3 --knobs tuned_numa`, `ATOM_Q38_EXTERN_MISSING=mono`. Ratios only
here; absolute tok/s live in `hackathon-prep/private-perf/qwen38-true-mono/baseline-matrix.md`.

Measured 2026-10-30 on the alola `rocke` partition (both gfx950 nodes, other users' and other lanes' jobs on the
other GPUs), q38 ROCm 10.0 container, one server per GPU, one server at a time per job.

## What was run

- **rocKE**: `copy-rocm-lib` at `89209ac7665` (`dnn-providers/hip-kernel-provider/rocke`, stage `rocke-tmbase`).
  Its file manifest equals the `rocke-rkrepro` stage of the published repro kit (`rk-repro-all3`).
- **ATOM**: stage `atom-rkrepro` = `dev/ATOM`'s hybrid working tree (`352229d6` plus the rocKE / sched / numa / head
  files); nothing but the recipe text changed in `dev/ATOM` since that stage was packed.
- **Generated functions**: 33 sidecars, all `"mailboxes": "numa1"` (12 B3: bodies r1 r2 r2s r3 r3s r4 r4s r4g2,
  prologues r1 r2 r2s r3s; 21 MoE: m3/m4/m5 x r1 r2 r3 r4 r2s r3s r4s), as published.
- **Harness**: `hackathon-prep/qwen38/truemono-baseline/` (`matrix.sh`, `summarize.py`, `sync.sh`). Per pair a
  seeded permutation of (A, ALL) rotated by the pair index (each arm takes each position); a fresh server per arm
  run; every cell of the job runs on that server, in order. Same prompts for both arms of a pair, new prompts
  per pair. `PRIME=1` (the CONC > 1 reruns) adds one untimed plain-mono run first so the job-local disk caches
  hold every shape.
- **Per-run accuracy check**: every server starts with a GSM8K `single_request_bench` cell (correct / checked
  per run); every server log is checked (mono decode on, no `mono ... off`, 0 tracebacks, rocKE head on iff
  ALL, no `numa off for`); the fallback lines `[ATOM_Q38_EXTERN_STAGES] no <stage> function for build variant
  <v>` are collected. All 60 arm-run server logs (56 timed, 4 prime runs) passed.

## The matrix

Mono serves: K = 0 (no MTP) at rows 1..4 (`MAX_ROWS`), K = 1..3 (`SPEC_TOKENS`), CONC 1..4 with MTP
(`dispatch.GROUPS` = 4, `G*(K+1) <= WIDE_ROWS` = 16), contexts up to 262,144 (`qsa_fast.MAX_CONTEXT`).
Build variants by cell: K = 0 decode `r1..r4`; MTP CONC 1 verify `r{K+1}s` (`r2s` / `r3s` / `r4s`); MTP
CONC G > 1 verify `r{G(K+1)}g{K+1}`: `r4g2` (K=1 G=2, narrow grouped), and the wide builds `r6g2 r8g2`,
`r6g3 r9g3 r12g3`, `r8g4 r12g4 r16g4`; the drafter's QSA layer runs `r{K+1}s` at step 0 and `r1..r4`
(one row per request) at later steps. A CONC c cell exercises G = 1..c (requests finish at different times).
Every ALL server builds `r8g4 r12g4 r16g4` (K = 3) or the K's wide family at startup whatever the CONC.

| # | K | CONC | workload (ISL band) | target builds exercised | what ALL runs in rocKE there | metric |
|---|---|---|---|---|---|---|
| 1 | 3 | 1 | GSM8K speed, 64 prompts, <= 2048 out | `r4s` (+ drafter `r4s`, `r1`) | MoE + B3 + levers + head | per-request tok/s, verify round |
| 2 | 3 | 2 | GSM8K speed, 64 prompts | `r8g4`, `r4s` | head + drafter MoE only | per-request tok/s |
| 3 | 3 | 4 | GSM8K speed, 64 prompts; random 1k / 1k x 16 | `r8g4 r12g4 r16g4` | head + drafter MoE only | per-request tok/s; output tok/s |
| 4 | 3 | 1 | random 8k / 1k x 8 | `r4s` | MoE + B3 + levers + head | output tok/s, TPOT x acceptance |
| 5 | 3 | 4 | random 8k / 1k x 16 | `r8g4 r12g4 r16g4` | head + drafter MoE only | output tok/s |
| 6 | 0 | 1 | GSM8K speed, 64 prompts | `r1` | MoE + B3 + levers + head | per-request tok/s |
| 7 | 0 | 4 | GSM8K speed, 64 prompts; random 1k / 1k x 16 | `r1..r4` | MoE + B3 + levers + head | per-request tok/s; output tok/s |
| 8 | 0 | 1 | random 8k / 1k x 8 | `r1` | MoE + B3 + levers + head | output tok/s, TPOT |
| 9 | 1 | 1 | GSM8K speed, 96 prompts | `r2s` | MoE + B3 + levers + head | per-request tok/s, verify round |
| 10 | 1 | 2 | GSM8K speed, 64 prompts; random 1k / 1k x 16 | `r4g2`, `r2s` | B3 + numa + head (FlyDSL MoE) | per-request tok/s; output tok/s |
| 11 | 2 | 1 | GSM8K speed, 96 prompts | `r3s` | MoE + B3 + levers + head | per-request tok/s, verify round |
| 12 | 2 | 4 | GSM8K speed, 64 prompts; random 1k / 1k x 16 | `r6g3 r9g3 r12g3` | head + drafter MoE only | per-request tok/s; output tok/s |
| 13 | 3 | 1 | random 64k / 1k x 4 and 248k / 1k x 3 (`--max-model-len 262141`) | `r4s` at long context | MoE + B3 + levers + head | TPOT x acceptance, output tok/s |
| 14 | 3 | 4 | accuracy suite: lm_eval GSM8K 1,319 5-shot, HumanEval 164 | grouped + narrow | as 1-3 | exact match / pass@1 |

Why these cells: every build family the mono serves is exercised at least once (narrow decode, seq verify for
each K, the narrow grouped `r4g2`, the three wide-grouped families), at both ends of the ISL range the roadmap
gates (1k, 8k) plus the supported maximum, and at the concurrency where each family dominates. Not included,
by design: K = 0 CONC > 4 and MTP CONC > 4 (rows above `MAX_ROWS` / groups above `GROUPS` run ATOM's native
path in both arms, so the hybrid's levers do not apply; a true rocKE mono would have to add those widths),
K = 0 CONC 2-3 and MTP CONC 3 (same build families as cells 7 / 3 / 12, exercised inside the CONC 4 cells),
the other speed sets (HumanEval / MT-Bench: same builds as GSM8K), and AgentX (deferred by ROADMAP 4.5).

## Results: ALL / A per cell

Mean [min - max] over pairs. Speed ratios above 1 and time ratios below 1 mean ALL is faster. "round" =
ms per verify round (`single_request_bench`, server counters); "rnd" = TPOT x acceptance (time per round from
`benchmark_serving`, random prompts). CONC > 1 GSM8K and 1k-bench cells are from the primed reruns (`c31b`,
`c20b`); see "What broke" for the first wave.

| # | cell | pairs | primary | secondary | acceptance | GSM8K right / run, A vs ALL |
|---|---|---|---|---|---|---|
| 1 | K3 C1 GSM8K | 3 | tok/s 1.150 [1.139-1.156] | round 0.865 [0.863-0.865] | 0.994 | 60 59 58 vs 58 58 58 (of 64) |
| 1' | K3 C1 GSM8K (long job, 16 prompts) | 2 | tok/s 1.163 [1.163-1.164] | round 0.854 [0.851-0.858] | 0.991 | 16 16 vs 16 15 |
| 2 | K3 C2 GSM8K | 3 | tok/s 1.011 [0.984-1.031] | round 0.984 [0.972-0.996] | 1.000 | 60 60 59 vs 59 61 59 |
| 3 | K3 C4 GSM8K | 3 | tok/s 0.996 [0.971-1.013] | aggregate e2e 1.013 [0.981-1.062] | 1.002 | 60 58 58 vs 60 57 59 |
| 3b | K3 C4 random 1k/1k | 3 | output 0.938 [0.893-0.980] | rnd 1.031 [0.973-1.131] (pair-1 ALL outlier; median 0.989) | 0.970 | - |
| 4 | K3 C1 random 8k/1k | 3 | output 1.099 [1.077-1.113] | rnd 0.877 [0.874-0.882], TPOT 0.868 | 1.011 | - |
| 5 | K3 C4 random 8k/1k | 3 | output 1.008 [0.988-1.023] | rnd 0.993 [0.984-1.008] | 1.001 | - |
| 6 | K0 C1 GSM8K | 3 | tok/s 1.142 [1.139-1.147] | per token 0.874 (token-weighted 1.144) | - | 59 56 60 vs 57 61 62 |
| 7 | K0 C4 GSM8K | 3 | tok/s 1.105 [1.038-1.143] | aggregate e2e 1.058 [1.000-1.099] | - | 57 57 60 vs 61 62 60 |
| 7b | K0 C4 random 1k/1k | 3 | output 1.117 [1.095-1.134] | TPOT 0.880 [0.876-0.887] | - | - |
| 8 | K0 C1 random 8k/1k | 3 | output 1.106 [1.103-1.111] | TPOT 0.881 [0.876-0.883] | - | - |
| 9 | K1 C1 GSM8K | 3 | tok/s 1.145 [1.142-1.150] | round 0.874 [0.872-0.876] | 0.996 | 90 88 89 vs 89 89 87 (of 96) |
| 10 | K1 C2 GSM8K | 3 | tok/s 1.077 [1.071-1.089] | round 0.921 [0.911-0.929] | 0.993 | 60 59 60 vs 59 58 59 |
| 10b | K1 C2 random 1k/1k | 3 | output 1.103 [1.051-1.190] | rnd 0.930 [0.925-0.932] | 1.027 | - |
| 11 | K2 C1 GSM8K | 3 | tok/s 1.137 [1.133-1.143] | round 0.880 [0.874-0.884] | 0.996 | 89 89 91 vs 87 85 87 (of 96) |
| 12 | K2 C4 GSM8K | 3 | tok/s 1.051 [1.032-1.066] | round 0.990 [0.940-1.042] | 1.006 | 59 58 58 vs 59 58 59 |
| 12b | K2 C4 random 1k/1k | 3 | output 1.027 [0.981-1.079] | rnd 0.999 [0.995-1.002] | 1.005 | - |
| 13a | K3 C1 random 64k/1k | 2 | rnd 0.866 [0.853-0.880] | TPOT 0.799, output 1.058 (prefill-bound) | 1.092 (4 prompts) | - |
| 13b | K3 C1 random 248k/1k | 2 | rnd 0.844 [0.788-0.899] | TPOT 0.832, output 1.016 (prefill-bound) | 1.013 | - |
| 14 | K3 C4 accuracy suite | 2 | see below | | | |

Cell 14 (accuracy, K = 3, CONC 4, 2 pairs; ALL - A per pair, in absolute accuracy points):
- lm_eval GSM8K 1,319 5-shot (chat template, <= 4096 tokens): flexible-extract +0.0016 / -0.0015 (mean 0.0000),
  strict-match +0.0038 / 0.0000 (mean +0.0019); per-run stderr about 0.0056, so both are well inside 1 sigma.
- HumanEval pass@1 (164, greedy, 768 tokens): +0.0183 / -0.0061 (mean +0.0061); per-run stderr 0.033-0.034.
- The same servers' K3 C1 GSM8K cell (16 prompts): tok/s 1.171 [1.167-1.175], round 0.855 [0.854-0.857],
  16 / 16 right in all four runs.

Reading the matrix:
- **CONC 1, every K**: the hybrid's gain holds: verify round 0.854-0.880 (K = 1..3), per token 0.874 at K = 0
  (published: round 0.851-0.861 at K = 3, 0.868 at K = 0, from 6 pairs of 96 prompts). It also holds at long
  context: time per round 0.87 at 8k, 0.87 at 64k, 0.84 at 248k.
- **Narrow multi-row builds** (K = 0 rows 2-4, K = 1 `r4g2`): about 1.08-1.12x.
- **Wide grouped builds** (K = 2 / 3 at CONC 2-4): parity, 0.99-1.05x per request, round 0.98-1.03. These
  builds keep the FlyDSL MoE and B3 and drop every lever; only the head (rows 8-16) and the drafter's `r1..r4`
  QSA layers are rocKE there. This is where a true rocKE monokernel can win outright, and where it must at
  least match plain mono.
- **Acceptance** stays within 0.97-1.03 everywhere (1.09 at 64k on 4 random prompts). GSM8K right per run is
  inside mono's own run-to-run spread in every cell.

## Where the hybrid falls back to FlyDSL (server-log build lines, every ALL run)

| build variant | where it runs | MoE (m3/m4/m5) | B3 (+ prologue) | levers | head |
|---|---|---|---|---|---|
| `r1` (decode / drafter) | K0 CONC 1; drafter later steps | rocKE fused | rocKE (+ b3pre) | numa + holds (`hold.b1/m3/m4`, qsa `hold.q1`) | rocKE |
| `r2`, `r3`, `r4` | K0 CONC 2-4; drafter n-row steps | rocKE fused | rocKE (r2 + b3pre) | numa | rocKE |
| `r2s`, `r3s` | K1 / K2 CONC 1 verify, drafter step 0 | rocKE fused | rocKE + b3pre | numa | rocKE |
| `r4s` | K3 CONC 1 verify, drafter step 0 | rocKE fused | rocKE (no prologue) | numa + holds + `head_sleep` | rocKE |
| `r4g2` | K1 CONC 2 | **FlyDSL** | rocKE (no prologue) | numa only | rocKE |
| `r6g2 r8g2` | K1 CONC 3-4 | **FlyDSL** | **FlyDSL** | none | rocKE |
| `r6g3 r9g3 r12g3` | K2 CONC 2-4 | **FlyDSL** | **FlyDSL** | none | rocKE |
| `r8g4 r12g4 r16g4` | K3 CONC 2-4 | **FlyDSL** | **FlyDSL** | none | rocKE |

So cells 2, 3, 3b, 5, 12, 12b run the target layers entirely on FlyDSL in ALL, cell 10 / 10b keeps the FlyDSL
MoE; ATOM logs `[ATOM_Q38_EXTERN_STAGES] no <stage> function for build variant <v>` for each (34 lines per K = 1
server, 42 per K = 2 or K = 3 server at `--max-model-len 10240`; none at K = 0).

## What broke, and caveats

- **First-wave CONC > 1 GSM8K cells at K = 2 / 3 were invalid** (jobs `tmb-k2`, `tmb-k3`): the first steps at a
  new batch shape JIT-compile, and the hybrid's extern-linked builds are never written to FlyDSL's disk cache,
  so every ALL server stalled for seconds several times inside the timed run (its first requests ran at a
  small fraction of their normal speed), while plain mono paid it only in its first run (cold cache). Those cells read 0.90-0.97 for ALL.
  Fixed in the harness (an untimed mt-bench warmup at the cell's concurrency before each CONC > 1 speed cell,
  plus `PRIME=1`) and rerun (`tmb-c31b`, `tmb-c20b`); the table uses the reruns. K = 1 CONC 2 was already clean
  in the first wave (1.080 [1.046-1.107]) and agrees with the rerun. This JIT cost is real for the hybrid in
  serving (every server start), not just a harness artefact.
- **Cold first arm**: before `PRIME`, the first arm of a job paid every cold compile (A always ran first in
  pair 1); bench cells with their own warmup were mostly immune, but K2 C4 1k first wave read output 1.330
  [0.961-1.976] for that reason (rnd 1.001). The primed rerun gives 1.027 / rnd 0.999.
- **Cell 3b** has one outlier pair (ALL pair 1, round time +13 %); the other two pairs give 0.989 and 0.973.
- **Noise sources**: the nodes carried other users' and other lanes' GPU jobs during every run; paired, shuffled,
  same-GPU arms cancel slow drift, not bursts. Random-prompt acceptance is noisy with few prompts; use "rnd" or
  TPOT, not output tok/s, for decode at 64k / 248k (output tok/s there is prefill-bound).
- No cell failed to run (every cell of every run: `ok`), no server log check failed, the 248k-token cell
  prefilled and served in both arms. The local `q38_job.sh` pollers never saw their jobs finish (`sacct` could not
  reach the Slurm database from login-01); results were read from the job logs and result directories.

## How to rerun

From `hackathon-prep` on the laptop (Git Bash), with `CL_HOST=vanantha@ctr2-alola-login-01`:

```bash
bash qwen38/truemono-baseline/sync.sh all     # h-tmbase + rocke-tmbase (copy-rocm-lib rocKE); ATOM = existing atom-rkrepro
J="bash tools/q38/q38_job.sh --mode container --stage h-tmbase --stage rocke-tmbase --stage atom-rkrepro"
M='bash /job/stages/h-tmbase/truemono-baseline/matrix.sh'
$J --tag tmb-k3 --time 03:30:00 --env K=3 --env PAIRS=3 --env PRIME=1 \
   --env "CELLS=speed:gsm8k:1:64 speed:gsm8k:2:64 speed:gsm8k:4:64 bench:8192:1024:1:8 bench:8192:1024:4:16 bench:1024:1024:4:16" -- "$M"
$J --tag tmb-k0 --time 03:30:00 --env K=0 --env PAIRS=3 --env PRIME=1 \
   --env "CELLS=speed:gsm8k:1:64 speed:gsm8k:4:64 bench:8192:1024:1:8 bench:1024:1024:4:16" -- "$M"
$J --tag tmb-k1 --time 03:00:00 --env K=1 --env PAIRS=3 --env PRIME=1 \
   --env "CELLS=speed:gsm8k:1:96 speed:gsm8k:2:64 bench:1024:1024:2:16" -- "$M"
$J --tag tmb-k2 --time 03:00:00 --env K=2 --env PAIRS=3 --env PRIME=1 \
   --env "CELLS=speed:gsm8k:1:96 speed:gsm8k:4:64 bench:1024:1024:4:16" -- "$M"
$J --tag tmb-long --time 04:00:00 --env K=3 --env PAIRS=2 --env Q38_MML=262141 \
   --env "CELLS=speed:gsm8k:1:16 bench:65536:1024:1:4 bench:253952:1024:1:3" -- "$M"
$J --tag tmb-acc --time 07:00:00 --env K=3 --env PAIRS=2 \
   --env "CELLS=speed:gsm8k:1:16 lmeval:gsm8k:4 acc:humaneval:4:low" -- "$M"
```

Each job writes `m/abs.json` (absolute, private) and `m/ratios.json` into `~/rocke-sdpa-claude/q38/results/<tag>/`
and prints both tables; `summarize.py <dir> <K> <pairs> "A ALL" "<cells>"` re-summarizes fetched results. A
true-mono arm is one more entry in `matrix.sh`'s `ENVS` (or a separate ATOM stage) against the same cells.

Jobs of this baseline: `68214613` (tmb-k3), `68214624` (tmb-k0), `68214744` (tmb-k1), `68214765` (tmb-k2),
`68214873` (tmb-long), `68214990` (tmb-c31b: K3 + K1 CONC > 1, primed), `68215008` (tmb-c20b: K2 + K0 CONC 4,
primed), `68215019` (tmb-acc); `68214976` / `68214978` were cancelled after a minute (resubmitted primed).
