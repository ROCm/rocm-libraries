# W17 B1: DeepSeek-V4.1-Flash at TP8 with K1's GEMM: bring-up, accuracy, baseline

W17, 2026-11-05. MI355X x 8 (`smci355-ccs-aus-n07-17`), atom-dev image `nightly_202610171506` (aiter `2cc8c750`),
ATOM tree `dev/ATOM-dsv41` (stage `w17-atom-a1`, ATOM base `df86c482`), K1's K = 288 kernels (`ATOM_DSV41_RK_K288=1`,
picks from K1's search job `k1s3`). Ratios only. Absolutes: `hackathon-prep/private-perf/dsv41-mono/w17/B1-tp8-abs.md`.
Harness: `hackathon-prep/dsv41/w17/b1/` (section 8). Raw: `/scratch/users/vanantha/w17/results/b1t1/`.

Names: "ATOM native" = ATOM's regular path; "ATOM mono" = ATOM's FlyDSL per-layer V4.1 mono kernels
(`ATOM_MONO_ENABLE=1`). Every server: bf16 KV, fp8 index, no EP, `--level 0`, FULL graphs, DSpark 5 (B0's flags).

## 0. Verdict

| question | answer | evidence |
|---|---|---|
| Does ATOM start at TP8? | **Yes, native and mono**, with `ATOM_DSV41_RK_K288=1`. The knob-off server (the must-fail control) dies in warm-up on all 8 ranks at the K = 288 shared-expert GEMM, as K1 predicted | job b1t1, section 1 |
| Does ATOM's mono accept TP8? | **Yes.** `V4.1 mono decode on` + `V4.1 mono draft on` on all 8 ranks. One build is declined: the 42-row (CONC 7) kernel needs 44 B of scratch a lane at TP8, so CONC-7 steps take the native path. CONC 1 / 2 / 4 / 8 (6 / 12 / 24 / 48 rows) serve on the mono | section 1 |
| Any other TP8 blocker? | **No software blocker found.** The long-prompt failures seen in the job are a faulty GPU, not TP8 (section 3) | sections 1, 3 |
| Accuracy at TP8 | **b1t1: FAIL under the rule pre-registered in `b1_acc.py`** (TP8 mono exceeds the twins' discordance on 1 flexible + 2 strict pairs, TP8 native on 1 + 1). The earlier "PASS" was judged on a looser rule chosen after the data and is **withdrawn**. A v2 rule (`h/b1_acc2.py`, sha256 `a43cf7fd...131c`, frozen 2026-11-05 23:26 UTC) is pre-registered for the replicate job b1t7 only (fix-round status in section 9). **Passkey at TP8: not measured** | sections 2, 9 |
| Decode step time (CONC 1) | TP8 mono = **0.91** of TP4 mono, **0.74** of TP2 mono; TP8 native = **1.95** TP8 mono. At CONC 8: TP8 mono 0.80 of TP4 mono, 0.52 of TP2 mono | section 4 |
| Measured acceptance | 0.59 of the speculated tokens (AL about 3.95 of 6) in every arm, TP8 included; LithosAI credits 0.75 | section 4 |
| vs LithosAI | TP8 mono's CONC-1 step is **1.48x** LithosAI's published 8-GPU step. Re-derived with LithosAI's own credit (4.75 tokens a step) a TP8-mono request runs at **0.54** of their 2,116 tok/s; at the 5.9-6.0 tokens a step their own numbers imply, 0.68. With our measured acceptance it is 0.45 | section 5 |
| Served per-request matrix (C1 / C2 / C4 / C8, 2 passes, TP8 twins, 1024-token answers) and long context | **Pending a node with 8 healthy free GPUs**: job `b1t5` is built and a login-node watcher submits it the moment n07-17 has 8 free GPUs. Long context at TP8 (32K / 128K) cannot run on n07-17 at all (faulty GPU); the 1K step is the CONC-1 step above | sections 6, 7 |
| Hardware | **n07-17's GPU at PCI bus 0x05 has a memory-error storm** (1,941 correctable + 734 deferred ECC errors; the other 7 GPUs 0). Every server that held it failed on the 16K-token passkey prefill; the same arms on the other GPUs passed. Recorded as alola rule 268 | section 3 |

## 1. Bring-up (job b1t1, 68290309, 8 GPUs)

| arm | knob | result |
|---|---|---|
| `tp8n0` (control) | `ATOM_DSV41_RK_K288` off | **dies in warm-up**, all 8 ranks: `ValueError: [FlyDSL gfx950 bmm] no kernel config runs B=1 N=5120 K=288` |
| `tp8n` native | `=1` | starts. `[K288]` on every rank: 43 weights registered (40 target + 3 draft), 36 kernels prepared, aiter's refusal confirmed at M = 1 and 4096, install check bitwise equal to ATOM's local FP8 GEMM. 8 / 8 text probes, 0 errors |
| `tp8m` mono | `=1` + `ATOM_MONO_ENABLE=1` | starts. Mono widths 5-48 built on every rank (the 42-row build declined, above), mono decode and draft on. 8 / 8 text probes |

- The mono path at TP8 was never run on GPU before (ATOM's own `config.py`: "TP8's builds compile and fit, but the
  original path does not start there"; K1's simulated TP8 could not host it). Its first real TP8 run serves, and its
  GSM8K is in the baseline band (section 2).
- Start time: TP8 mono about 2.4x TP8 native (the per-width mono builds), about the same as a TP4 or TP2 mono start.
- **TP8 open item:** the 42-row build spills (44 B a lane) only at TP8. CONC 7 is not one of the requested cells; a
  rocKE replacement is not needed for this baseline, but it is the one TP8-only refusal in the mono.

## 2. Accuracy (GSM8K 1319, 3-shot chat, 8 concurrent so every step is mono-eligible, greedy, max 4096 tokens)

| arm (GPUs) | flexible | strict | passkey 8K / 24K |
|---|---|---|---|
| TP8 native | 0.9052 | 0.1160 | not measurable here (section 3) |
| TP8 mono | 0.9121 | 0.1190 | not measurable here |
| TP4 mono (0-3, holds the faulty GPU) | 0.8999 | 0.1259 | server lost (section 3) |
| TP4 mono twin (4-7) | 0.9121 | 0.1221 | 20 / 20, 20 / 20 |
| TP2 mono (0,1, holds the faulty GPU) | 0.9067 | 0.1221 | stopped (section 3) |
| TP2 mono twin (2,3) | 0.9105 | 0.1175 | 20 / 20, 20 / 20 |

Paired per-question discordance (share of the 1,319 questions where the two runs disagree) and exact McNemar p:

| pair | flexible disc. / p | strict disc. / p |
|---|---|---|
| TP2 A-vs-A | 0.042 / 0.59 | 0.082 / 0.63 |
| TP4 A-vs-A | 0.047 / 0.06 | 0.083 / 0.70 |
| TP2 vs TP4 (4 pairs) | 0.041-0.060 / 0.10-0.89 | 0.078-0.093 / 0.32-1.00 |
| TP8 native vs TP2 / TP4 (4 pairs) | 0.042-0.052 / 0.28-0.90 | 0.073-0.086 / 0.23-0.92 |
| TP8 mono vs TP2 / TP4 (4 pairs) | 0.043-0.048 / 0.05-1.00 | 0.078-0.088 / 0.43-0.92 |
| TP8 mono vs TP8 native | 0.045 / 0.30 | 0.080 / 0.77 |

- **Gate (b1t1): FAIL under the pre-registered rule; the earlier PASS is withdrawn.** The rule written into `b1_acc.py`
  before the run: every TP8-vs-baseline pair discords no more than the baselines' twin pairs (flexible 0.047, strict
  0.083). TP8 mono exceeds it on 1 of 4 flexible pairs (by 1 question) and 2 of 4 strict pairs (by 1-7 questions), TP8
  native on 1 flexible and 1 strict pair. The first version of these notes then judged the result on a looser
  cross-pair rule chosen after seeing the data; that is not a pass under the pre-registered rule.
  - Why v1 is a poor rule (stated, not used to rescue b1t1): it compares the maximum of 4 TP8 pairs with the maximum of
    2 twin pairs drawn from the same noise, and the baselines fail it among themselves (3 of 4 TP2-vs-TP4 pairs on
    flexible, 1 of 4 on strict). The lowest TP8 McNemar p is 0.052 (TP8 mono vs the TP4 run on the faulty GPU),
    borderline.
  - A v2 rule (median-vs-maximum discordance, accuracy difference within the baselines' range, Bonferroni McNemar,
    passkey, plus must-fail 2.5 % flip mutants) was written after b1t1 and pre-registered for the replicate job only
    (section 9).
- The served V4.1 path is not run-to-run deterministic (alola rules 248 / 251), so the twins' discordance is the floor.
- `strict-match` is low for every arm (thinking-mode answers do not end in `#### N`, as in B0); `flexible-extract` is
  the score.
- The K288 knob adds no numerics of its own: its routed output is bitwise ATOM's row-major GEMM in every gated cell (K1).
- **Passkey at TP8 is the one accuracy item still open**: it needs 8 healthy GPUs (section 3).

## 3. The long-prompt failures are a faulty GPU, not TP8

The passkey client sends 4 concurrent 8K-token prompts: one 8,068-token prefill, then a 16,264-token batch of three.
Every server that held HIP device 1 of this job (PCI bus 0x05) failed right there; every server without it passed:

| server | GPUs | what happened at the first 16K-token prefill |
|---|---|---|
| TP8 native | 0-7 | 7 ranks waited in the sampled-token broadcast (`BROADCAST`, 3 elements) until the 600 s NCCL watchdog; the 8th rank (bus 05) never got there |
| TP8 mono | 0-7 | the same, the same rank (`py-spy`: 7 ranks in `Event.synchronize`, the bus-05 rank still inside the forward) |
| TP4 mono | 0-3 | the bus-05 rank aborted: `HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Queue Aborting` |
| TP4 mono twin | 4-7 | passkey 40 / 40 in seconds |
| TP2 mono | 0,1 | no fault, but each 16K-token prefill took 1.5-3 minutes |
| TP2 mono twin | 2,3 | passkey 40 / 40 in seconds |

`amd-smi metric -e` on the node then showed, since the morning's reboot, 1,941 correctable and 734 deferred ECC errors on
that GPU and none on the other seven. The 2 GiB health fill every job runs passes on it.

- **Decode with short prompts is not affected:** the CONC-1 step on (0,1) vs (2,3) is 0.997, on (0-3) vs (4-7) 0.995,
  and the CONC-8 step 1.001 / 1.002 (section 4). GSM8K on the faulty pair is inside the band.
- Consequences: TP8 passkey and TP8 long-context runs cannot be done on n07-17 until the GPU is fixed; short-prompt TP8
  timings there stand (the TP8 step is a collective, and the faulty GPU does not slow short-prompt decode). The
  faulty GPU may also explain W15's n07-17-only bitwise noise (alola rule 266).

## 4. Decode step time (no-profiler host timeline, same job, fresh server per arm)

The step is the whole DSpark runner forward (target verify + draft + sampling). Measured with P1's host timeline
(`ATOM_DSV41_RK_HOSTTL`: host CLOCK_MONOTONIC marks + GPU events, no profiler, no added sync) over each arm-run's whole
life (lm_eval GSM8K at 8 concurrent + a CONC-1 cell): pure-decode steps by batch size, the slowest rank's median.
"Period" = GPU graph span + the GPU gap to the next step; graph span and host period agree with it within 0.5-1 %.

Step period, ratios (TP4 / TP2 references = the mean of the arm and its twin; the twins agree within 0.6 % at bs 1 and
0.2 % at bs 8):

| bs (requests a step) | TP8 mono / TP4 mono | TP8 mono / TP2 mono | TP4 mono / TP2 mono | TP8 native / TP8 mono |
|---|---|---|---|---|
| 1 | 0.909 | 0.743 | 0.818 | 1.949 |
| 2 | 0.913 | 0.703 | 0.770 | 1.594 |
| 4 | 0.856 | 0.590 | 0.689 | 1.422 |
| 8 | 0.803 | 0.521 | 0.648 | 1.279 |

- bs 1 and 8: 590-8,900 steps each; bs 2 and 4: 44-228 steps each (the tail of the 8-concurrent run).
- TP8's gain over TP4 grows with the batch: the per-GPU bytes halve while the latency chain does not (ANALYSIS 4.2).
  At CONC 1 TP8 mono saves only 9 % of TP4 mono's step.
- TP8 native is about 2x TP8 mono at CONC 1 (B0 found 1.8x at TP4).

**Acceptance (measured):** AL 3.92-3.98 tokens a step over each GSM8K run, i.e. 0.585-0.596 of the 5 speculated tokens
accepted, in all six arms. TP size and mono do not change acceptance (as B0 found at TP2 / TP4).

## 5. Against LithosAI's published TP8 result (2,116 tok/s a user, 2.8 ms a step at 1K, credited 75 %)

| quantity (CONC 1, ~1K context) | ratio |
|---|---|
| TP8 mono step / LithosAI 8-GPU step | 1.48 |
| TP4 mono step / LithosAI 4-GPU step (3.75 ms) | 1.22 |
| TP8 mono tok/s a request, credited 4.75 tokens a step (1 + 0.75 x 5) / 2,116 | 0.54 |
| the same, credited 6.0 tokens a step / 2,116 | 0.68 |
| TP8 mono tok/s with the measured AL / 2,116 | 0.45 |
| credited 4.75 / measured AL (same step) | 1.20 |

- **LithosAI's own numbers are not internally consistent with 4.75 tokens a step:** 2,116 tok/s x 2.8 ms = 5.92 tokens
  a step on 8 GPUs, and 1,598 x 3.75 ms = 5.99 on 4. Their credit is about 6 tokens a step (for example 75 % of 8
  speculated tokens), so the like-for-like per-step comparison is the step ratio (1.48 at 8 GPUs, 1.22 at 4) or the
  6.0 row, not the 4.75 row. Both are reported; the summary script prints both columns.
- Their numbers are self-reported, with assumed (not measured) acceptance; ours are measured on the served path.
- LithosAI's step grows 1.18x from 1K to 1M context. Ours at long context is not measured yet (section 7).

## 6. The served per-request matrix (pending a healthy 8-GPU window)

The step times above are from one TP8 server per path (no TP8 twin yet) and the client rates of b1t1 are 8-prompt
CONC-1 cells only. The full matrix the task asks for is job `b1t5`, built and staged
(`/scratch/users/vanantha/w17/jobs/b1_job.b1t5.sh`, harness `b1-h6`):

1. `memtest`: device-memory pattern test of every GPU + per-GPU speed probe + ECC counters before / after;
2. `perf`: TP8 native, TP8 mono, TP4 mono (GPUs 4-7), TP2 mono (2,3), TP2 mono on the faulty pair (0,1: the fault's
   effect on served decode, measured), and the A-vs-A twin of each, 2 rotated passes, fresh server per arm-run, cells
   C1 / C2 / C4 / C8 with 16 / 16 / 32 / 32 GSM8K prompts at 1024-token answers, host timeline on: per-request decode
   tok/s, AL, step period / span / host period at bs = CONC, the credited 4.75 and 6.0 columns (`b1_summary.py`);
3. `long`: 1K at TP8 (mono and native), 1K / 32K / 128K at TP4 / TP2 on healthy GPUs, 512 decode tokens a request.

A login-node watcher (`b1_watch.sh`, `/scratch/users/vanantha/w17/logs/b1_watch.t5.log`) submits it only when n07-17 has
all 8 GPUs free (never parks an 8-GPU request; cancels and keeps watching if it loses the race; no start from 03:00 to
07:00 UTC, n07-17's drop-out window). From 21:36 UTC W15 / W16 job chains (3-4 h limits, dependency chains) and
another user's 1-GPU perf array kept the node partly busy, so at hand-off (23:16 UTC) b1t5 had not started.

- The watcher is login-01 PID 1758818 (`bash .../jobs/b1_watch.t5.sh`), deadline 2026-11-07 03:39 UTC; stop it with
  `kill 1758818` if the window should go to another job.
- Once b1t5 has run: fetch `/scratch/users/vanantha/w17/results/b1t5/out.tgz`, unpack, then
  `python hackathon-prep/dsv41/w17/b1/b1_summary.py <out> --ref tp4mg --abs <private>.md --ratio <ratio>.md` (per-request
  tok/s, AL, step period / span / host period, credited 4.75 and 6.0 columns, long-context table) and
  `b1_htl_bs.py <out>`; memtest / ECC lines are in the job log.

## 7. Long context (1K / 32K / 128K)

- **1K:** the CONC-1 step of section 4 (GSM8K prompts are 0.4-0.8K tokens).
- **32K / 128K at TP8: blocked on n07-17** by the faulty GPU (a 16K-token prefill already fails there). Needs n01-09
  back from its drain, or another MI355X node with 8 GPUs and container GPU access (m10-09 and m07-13 have none for us;
  n14-05 and m13-25 hold perpetual Jenkins jobs). TP4 / TP2 32K / 128K run in `b1t5` on healthy GPUs.

## 8. Harness and files

- `hackathon-prep/dsv41/w17/b1/` (new):
  - `b1_job.sh`: raw-sbatch body (stage sha check, K1 picks, earlier jobs' arm markers, periodic result saves, checks
    >= 200 GB free `/tmp`, removes its own node-local job dir: PLAN decision 17);
  - `h/b1_run.sh`: in-container driver, phases `bring`, `acc`, `accpar` (accuracy arms side by side on disjoint GPUs),
    `perf`, `nohtl`, `memtest`, `hangprobe`, `long`; clients run under a guard that stops them when the model runner
    dies or the server log stalls (`STALL_S`), and dumps every rank (py-spy, rocgdb) on a stall;
  - `h/b1_bench.py` (B0 client + CLOCK_MONOTONIC stamps per request and cell, long-prompt mode), `h/niah.py`,
    `h/textprobe.py`, `h/gpu_memtest.py`, `h/gpu_bw.py`, `h/arms.def`;
  - `b1_htl.py` / `b1_htl_bs.py` (step time per cell window / per batch size), `b1_acc.py` (scores + exact McNemar),
    `b1_summary.py` (tables, ratios, credited columns), `b1_watch.sh` (8-free-GPU watcher), `up_b1.sh`,
    `hangdiag.sh` (host-side stall catcher used in b1t1).
- Jobs: b1t1 = 68290309 (bring-up + accuracy, rc 0). b1t2 (68290310), b1t3 (68290768), b1t4 (68290897) were queued
  behind b1t1 and cancelled before starting (harness fixes); b1t5 (68291031) lost the node race and was cancelled; the
  watcher resubmits b1t5.
- No ATOM or rocKE source changed in B1 (the K288 knob is K1's; stage `w17-atom-a1`).
- Harness lessons, fixed for b1t5: a `timeout -s KILL` wrapper killed on server death orphans its client (the pipe then
  waits for the client's own timeout): stop it with TERM (forwarded) first; a 3-element `BROADCAST` NCCL timeout in
  V4.1 is the sampled-token broadcast (`model_runner.postprocess`), so the rank missing from the watchdog list is the
  one behind.

## 9. Fix round (verifier's problems) - status

In progress (2026-11-05 23:30 UTC). Pre-registration: `hackathon-prep/dsv41/w17/b1/h/b1_acc2.py`, sha256
`a43cf7fddaff1252b68d72d4bed2d9174b793c7de017d5eb96e017f859af131c`, staged as `b1-h7` (its sha is printed in the job log)
and passed in the job's submit line (`W17_PREREG`). Jobs b1t6 (perf matrix) and b1t7 (accuracy replicate, passkey, long
context) are staged; watcher `b1_watch.t6` (login-01) submits them when a node has 8 free GPUs.
