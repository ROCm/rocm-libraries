# WP2: device-side step metadata, eager tail into the graph, persistent tail, multi-step replay

W12, 2026-11-03. MI355X (gfx950), TP2 no-EP, DSpark 5, bf16 KV, fp8 index plane, `--level 0` + FULL graphs (the B0
recipe flags). Ratios only; absolute numbers are in `hackathon-prep/private-perf/dsv41-mono/WP2-abs.md`.

Revision 2 (fix round) answers the verifier's findings:
- no absolute times here any more: every GPU segment is a fraction of the step;
- the parity claim is now backed at every width the served path uses, including padded replays. The check found a
  real difference there, so the default changed (§2.1);
- `Tail.verify()` and `nobcast` now have their own in-place checks and must-fail mutants (§2.1, §2.2);
- the GSM8K strict-match result is recorded as a documented exception, not a pass (§2.4);
- the in-situ timing has 3 rotated passes per node, every lever is timed alone and in pairs, and the served C8 / C16
  cells have 3 passes on both nodes (§3, §5).

## Verdict

| knob | status | gate | in-situ GPU step, both nodes, 3 passes | served e2e, both nodes |
|---|---|---|---|---|
| `ATOM_DSV41_RK_TAIL` = `all` (argmax + dkv + nobcast; exact widths) | **built**, default off | **PASS** on the parity, verify and cross-rank checks at every width (§2.1), 7 of 7 mutants caught (§2.2); accuracy: flexible-extract and passkey pass, strict-match is a **documented exception** (§2.4) | **0.985-0.995** (mono) and **0.982-0.996** (native) of base at every CONC; A-vs-A twin medians 0.994-1.007 | **tie** at every cell, signs inconsistent across nodes (§3.3) |
| `ATOM_DSV41_RK_TAIL` lever `pad` (also use the levers on padded replays) | built, default off, **rejected on base B0** | FAIL: draft-KV not bitwise at padded widths, 11 planes in 3 runs (§2.1) | not timed separately (bs = CONC cells are exact widths) | - |
| `ATOM_DSV41_RK_DMETA` | **not built** | - | measured headroom: 0 (§4) | - |
| `ATOM_DSV41_RK_MSTEP=<k>` | **not built** (needs DMETA) | - | measured headroom: 0 (§4) | - |

**The D0 expectation (C1 1.08, C2 1.06 ... C16 1.03) is refuted by measurement.** What WP2 can remove today is
0.005-0.018 of the GPU step, not 0.03-0.08:

- The eager tail between the two graphs is GPU work, not launch gaps. Captured into the graph, the same kernels still
  run: the target segment grows by most of what the eager segments lose.
- The host is already off the critical path. The inter-step GPU gap is 0.0003-0.0011 of the step at every cell on
  both nodes. Device-side metadata and multi-step replay can only remove host time that the GPU waits for, and today
  it waits for almost none.

What WP2 did deliver is the second half of its goal: **the host's own work per step fell from 0.24-0.25 to 0.09-0.10
of the mono C1 step** (native: 0.24-0.29 to 0.15-0.19). At a 3x faster step that is about 0.3 of the step instead of
0.74, so the host stays hidden without DMETA / MSTEP up to a step about 10x faster than today's (mono).

3x: not reachable from WP2. Its share of the plan was always small (D0 §4 rank 2). The measured gain is about 1.01 on
the GPU step and a tie end to end.

## 1. What was built (`dev/ATOM-dsv41`, additive, default off)

- `atom/models/deepseek_v41/rk/tail.py` (new): `ATOM_DSV41_RK_TAIL=<levers>`, levers joined by `,` `+` or `:`;
  `1` / `all` turns on the three levers below (not `pad`):
  - **`argmax`**. The target FULL graph also computes, for every row, the greedy argmax of the target head over the
    TP-sharded vocabulary, in pair form. It uses ATOM's own `ParallelLMHead.compute_argmax_token`: a per-rank
    `(max, global id)` pair packed in fp32, an all-gather of `[rows, 2]`, and the lowest index on ties. For an
    all-greedy, logprob-free DSpark verify step, the runner then skips:
    - the eager `compute_logits` (final norm, bf16 head GEMM, full-vocab all-gather);
    - the sampler's and rejection sampler's `topk_select` passes.

    Verify / accept runs on the ids, through ATOM's greedy rejection kernel (`rejection_sample(target_token_ids=...)`).
    Any other step takes ATOM's path unchanged: sampling, logprobs, synthetic or relaxed acceptance, PCP, eager or
    PIECEWISE.
  - **`dkv`**. `DSparkProposer.compute_draft_kv` (the draft context-KV write) is captured at the end of the target graph.
    It no longer runs eagerly after each replay.
  - **`nobcast`**. When `argmax` is taken, the post-verify `sampled_tokens` TP broadcast is dropped. Every rank derives
    the same ids from the same all-gathered pairs.
  - **Exact widths only (new in revision 2).** The levers act only on a replay whose graph width M equals the scheduled
    rows. That holds at every capture size (bs 1-8, 16, 32, 48, 64). A padded replay (bs 9-15, 17-31, 33-47, 49-63: the
    graph runs at the next capture width) takes ATOM's eager tail: eager logits, and ATOM's eager draft-KV write over
    the graph's. The reason is the padded-width finding in §2.1.
  - **`pad`** (default off, not part of `all`): use the levers on padded replays too. Kept as a knob, recorded
    "rejected on base B0: not bitwise at padded widths".
  - Gate-only env: `ATOM_DSV41_RK_TAIL_CHECK=<dir>` (in-place check, §2.1), `..._CHECK_TWIN=1` (eager-vs-eager twin of
    the draft-KV write), and `ATOM_DSV41_RK_TAIL_MUTANT=rowshift|stale|skipdkv|posshift|vtgt|vbonus|rankdiv`.
- `atom/model_engine/model_runner.py`: hooks guarded by `_RK_TAIL is not None`. They sit in:
  - `capture_cudagraph` (warm-up and capture body);
  - `run_model` (after the replay);
  - `postprocess` (verify on ids, which now also gets the runner so the check can rerun ATOM's eager verify;
    broadcast skip);
  - `forward` (`pre_step`; eager draft-KV skip).
  With the knob unset, `_RK_TAIL` is None and the original code runs.
- Harness (new): `hackathon-prep/dsv41/wp2/` (round 1) and `hackathon-prep/dsv41/wp2/r2/` (round 2).
  - `wp2_gate2.py`: offline engine, closed groups or a closed loop with refill (`<c>:<n>:r`), greedy.
  - `wp2_run2.sh` (profiles A-E), `wp2_job2.sh`, `chk_sum.py` (check totals per graph width), `r2_sum.py` (per-pass
    paired ratios), `overlay_stage.py` (stage = the round-1 stage with only `tail.py` and `model_runner.py` replaced,
    so other agents' edits of the shared tree cannot leak in).
  - `mock_tail_test.py`: CPU mock of the new check and mutant paths, run before every stage.
- Stages: `atom-dsv41-wp2b` (round 1), `wp2d` (round-2 checks), `wp2e` (+ draft-KV diagnostics), `wp2f` (+ exact-width
  default; production paths of `wp2d`/`wp2e` are identical to it at every exact width).

Not reused: Qwen's P9 verify tail and TRIM+ADV metadata. Both belong to the persistent-tail / device-meta design, and
§4 shows that design has no headroom on this step today. The K=0 fused-argmax idea is reused in ATOM's own pair-form
head.

## 2. Gates

**The baseline is not bitwise reproducible, so a token-stream gate is impossible.** In the offline engine with fixed
batch composition, greedy, GSM8K-64, 512 tokens, two fresh engines of the same B0 arm returned identical token streams
for only 7 / 64 (mono C1), 8 / 64 (mono C4), 4 / 64 (native C1) and 5 / 64 (native C4) prompts (job wp2g1). Divergence
starts anywhere from token 7 to token 180. The nondeterminism is in ATOM's baseline itself, not in WP2. So the parity
gate is an in-place check.

### 2.1 In-place check (the parity gate)

With `ATOM_DSV41_RK_TAIL_CHECK`, every graph step is checked in the same process, on the same hidden states, against
ATOM's own eager path. There are four halves:

1. **argmax**: `topk_select(compute_logits(h), tie="low")` against the in-graph ids, on every scheduled row.
2. **draft-KV**: the whole window plane is snapshotted after the replay, ATOM's eager `compute_draft_kv` is rerun, and
   the plane must be unchanged.
3. **verify** (new): ATOM's eager verify branch is rerun on the reference logits: `index_select` of the bonus and target
   rows, `Sampler` greedy, `RejectionSampler.forward`. Its sampled tokens and its verdict (accepted, rejected, anchor
   row) must equal what `Tail.verify()` returned, for every request.
4. **cross-rank** (new): every rank's sampled tokens and verdict are all-gathered and must equal rank 0's. This is the
   assumption `nobcast` rests on. The check runs before the (dropped) broadcast.

Each record carries the graph width M next to the scheduled rows, so padded replays are counted apart.

**Round 1 (exact widths only).** Offline engine, GSM8K-32 at C1 + C4, 384 tokens (job wp2a7). Every batch size there is
a capture size. argmax 0 of 20,712 rows (mono) and 0 of 21,120 (native); draft-KV 0 of 2,426 / 2,499 planes.

**Round 2 (every width the served path and the GSM8K run use).** Plan `8:16 16:32 8:32:r 16:48:r 32:64:r 48:96:r`
(closed groups at C8 / C16 whose tails drain through every smaller width, then closed loops with refill at C8, C16, C32
and C48), 384 tokens. Padded replays seen: bs 9-15 (M = 16 x 6), 17-31 (M = 32 x 6) and 33-47 (M = 48 x 6), 34 of
the 37 padded batch sizes per run.

Checked steps per rank (both ranks report identical totals). Stage `wp2d` / `wp2e`, i.e. levers also on padded replays:

| job, node | arm | steps (padded) | argmax rows bad (padded) | verify requests bad (padded) | cross-rank bad | draft-KV planes changed (padded) | eager-vs-eager twin bad |
|---|---|---|---|---|---|---|---|
| wp2r2a, n07-17 | mono + all | 1,336 (230) | 0 / 90,384 (0 / 30,006) | 0 / 15,064 (0 / 5,001) | 0 / 1,336 | **1** / 1,336 (**1** / 230) | - |
| wp2r2a, n07-17 | native + all | 1,348 (225) | 0 / 89,346 (0 / 29,406) | 0 / 14,891 (0 / 4,901) | 0 / 1,348 | **1** / 1,348 (**1** / 225) | - |
| wp2r2d, n01-09 | mono + all | 1,377 (216) | 0 / 90,270 (0 / 28,602) | 0 / 15,045 (0 / 4,767) | 0 / 1,377 | **2** / 1,377 (**2** / 216) | 0 / 1,377 |
| wp2r2d, n01-09 | native + all | 1,336 (220) | 0 / 91,164 (0 / 29,670) | 0 / 15,194 (0 / 4,945) | 0 / 1,336 | 0 / 1,336 (0 / 220) | 0 / 1,336 |
| wp2r2d, n01-09 | mono + all, C48 / C40 refill only | 732 (363) | 0 / 100,542 (0 / 64,734) | 0 / 16,757 (0 / 10,789) | 0 / 732 | **4** / 732 (**4** / 363) | 0 / 732 |

**Finding: at padded widths the in-graph draft-KV write is not bitwise ATOM's.** All 8 differing planes are padded
replays at M = 48 x 6 (33-42 requests). The diagnostics (stage `wp2e`):
- the differences sit in the three draft layers' window rows of one request each, 1-36 bytes of a 512-byte packed row;
- ATOM's eager write is self-consistent: rerunning it from the same plane gives the same bytes in 3,445 of 3,445 steps;
- no step metadata is unusual (decode, no replay plan, no recompute floor, no cached-until).

The cause: ATOM's eager `compute_draft_kv` projects the aux taps at M = scheduled rows (`aux_for` slices the capture
buffers to `hidden_states`), while the graph projects them at M = capture width. The FP8 / quantized projection then
rounds a few elements differently, the GEMM M-dependence the verifier predicted for the head. On the argmax half the
same M-dependence produced no id flip in 0.12 M padded rows, but nothing guarantees that.

**Fix (stage `wp2f`, now the default): exact widths only.** A padded replay takes ATOM's eager tail, so parity holds by
construction there (ATOM's own eager write runs last and overwrites every row it owns; padding rows are zero-length
requests that neither path writes). Using the levers on padded replays is the default-off `pad` lever. Re-gate of the
default with the same plan (job wp2r2e, n01-09, stage `wp2f`, eager twin on):

| arm | steps (padded, all taking the eager tail) | argmax rows bad | verify requests bad | cross-rank bad | draft-KV planes changed | eager twin bad |
|---|---|---|---|---|---|---|
| mono + all (default) | 1,379 (218) | 0 / 61,668 | 0 / 10,278 | 0 / 1,161 | 0 / 1,161 | 0 / 1,161 |
| native + all (default) | 1,376 (210) | 0 / 62,970 | 0 / 10,495 | 0 / 1,166 | 0 / 1,166 | 0 / 1,166 |
| mono + all + `pad`, C48 / C40 refill | 726 (356) | 0 / 100,908 | 0 / 16,818 | 0 / 726 | **3** / 726 (all 3 padded) | 0 / 726 |

On the default every padded replay ran ATOM's eager tail (`pad_eager` = padded steps), and every exact-width step
matched ATOM bitwise on all four halves. The `pad` arm reproduces the padded draft-KV difference on a third run
(11 differing planes in 3 runs, all at M = 48 x 6), so the knob stays rejected.

The parity claim is therefore: **with the default (`all`), every decode step is either bitwise ATOM's eager tail (in
place, all four halves checked) or is ATOM's eager tail.** Checked over C1-C48 (rounds 1 and 2: about 1.4 x 10^4
checked steps per rank). The default itself was re-gated on n01-09; its exact-width code path is the one wp2r2a checked
on n07-17.

### 2.2 Must-fail mutants

Each mutant must fail, and on the half of the check it targets. Round 2, plan `4:8 16:16:r` (exact and padded widths),
job wp2r2a; per rank, both ranks identical:

| mutant (what it breaks) | argmax rows bad | verify requests bad | cross-rank bad | draft-KV planes changed |
|---|---|---|---|---|
| `vtgt`: verify maps draft r to row r + 1 (wrong `target_logits_indices` mapping inside `verify()`) | 0 / 18,120 | **2,028 / 3,020** | 0 / 791 | 0 / 791 |
| `vbonus`: bonus id read from the last draft row (wrong `bonus_logits_indices` mapping) | 0 / 10,500 | **485 / 1,750** | 0 / 459 | 0 / 459 |
| `rankdiv`: ranks > 0 flip one sampled id after verify (the divergence `nobcast` must never produce) | 0 / 8,358 | 0 / 1,393 | **405 / 405** | 0 / 405 |
| `rowshift`: verify reads the neighbouring row (off-by-one row) | **8,675 / 19,644** | 1,574 / 3,274 | 0 | 0 / 528 |
| `stale`: verify reads the previous step's ids (stale verify width) | **25,217 / 35,040** | 4,255 / 5,840 | 0 | 0 / 1,067 |
| `skipdkv`: draft-KV write dropped (skipped KV commit) | 0 | 0 | 0 | **417 / 417** |
| `posshift`: draft-KV write at positions + 1 (off-by-one position) | 0 | 0 | 0 | **425 / 425** |

`vtgt`, `vbonus` and `rankdiv` leave `step_ids` intact, so they exercise `verify()` and the broadcast drop, which the
round-1 mutants (which corrupt `step_ids`) did not.

Re-run on the exact-width default (job wp2r2e, n01-09, same plan), each again caught on its own half only:
`vtgt` verify 2,143 / 2,946; `vbonus` verify 439 / 1,498; `rankdiv` cross-rank 414 / 414; `rowshift` argmax
13,259 / 23,718; `stale` argmax 15,872 / 22,956; `skipdkv` draft-KV 419 / 419; `posshift` draft-KV 411 / 411. No mutant
moved an eager twin (0 bad in every arm), so the twin cannot mask a real draft-KV difference.

### 2.3 Token-stream screen (statistical, round 1)

Identity between runs was 3-14 of 64 prompts for every pair: B0 against its own twin, and the tail arms (`all`,
`argmax` only, `dkv` only, mono and native) against B0. Acceptance length was 3.87-4.00 for every non-mutant arm, and
3.81-4.01 in the round-2 check runs. Mutants against the same screen: `rowshift` and `stale` 0 of 64 identical, AL
1.5-3.2; `skipdkv` AL 1.15; `posshift` AL 3.81-3.85, just under the twin range, so **the token screen alone would miss
it**. Round 2: `vtgt` AL 1.7-2.4; `vbonus` 3.51-3.62 and `rankdiv` 3.72-3.83, both inside or near the twin range. The
in-place check is the gate; the screen is not.

### 2.4 Accuracy: documented exception

Served, job wp2a7 (round-1 stage, levers also on padded replays). GSM8K 1319, 3-shot chat, `max_gen_toks` 4096; passkey
20 per length:

| arm | flexible | strict | passkey 8K | passkey 24K |
|---|---|---|---|---|
| B0 mono | 0.9083 | 0.1160 | 20/20 | 20/20 |
| B0 mono twin | 0.9113 | 0.1221 | 20/20 | 20/20 |
| TAIL=all | 0.9105 | 0.1289 | 20/20 | 20/20 |

- Flexible-extract is inside the twin range. Passkey passes at both lengths.
- **Strict-match misses the gate as W12 defines it** ("within the A-vs-A spread of the baseline"): 0.1289 is 0.0068
  above the upper twin (0.1221). This is recorded as an exception, not as a pass. Why it is accepted:
  - the difference is in the better direction and about 0.75 binomial stderr of one arm (n = 1319);
  - the A-vs-A pair itself differs by 0.0061, about the same size;
  - the in-place check shows the tail computes ATOM's ids, verdicts and (at exact widths) draft-KV bitwise on every
    checked step, so a systematic accuracy effect would need a path the check does not cover;
  - the one known difference of that stage is the padded-width draft-KV rounding (§2.1: a few bytes of one draft
    window row on about 1-2 % of padded steps). The GSM8K run (num_concurrent 16) did hit padded widths. That changes
    draft proposals only (the draft layers' window rows), which verify then accepts or rejects against the target
    argmax. It does not enter the target's own computation; it can move acceptance length, and through the step shapes
    the same run-to-run variation the baseline already shows. The default no longer has it.
- What would turn it into a pass: a wider baseline spread (more B0 twins) or a re-run that lands inside it. Not done
  in this round; the exception stays open for the integration gate.

### 2.5 Launch-fatal word per replay

Not applicable. TAIL adds no persistent kernel and no mailbox: every added op is an ATOM / aiter / Triton kernel already
used elsewhere in the step. The word belongs with a persistent tail; see §4.

## 3. Timing

Every comparison is same-job, with a fresh engine or server per arm-run, an A-vs-A twin of the comparator, and the arm
order rotated per pass. Values are the median over passes of the per-pass paired ratio arm / mean(base, twin), with the
range over passes in brackets. Round 2 timed the round-2 stages (`wp2d` on n07-17, `wp2e` on n01-09). Their production
path differs from the default (`wp2f`) only on padded replays, and the in-situ cells below are all at bs = CONC, which
is a capture size, so they time the default too.

### 3.1 In situ, no profiler

P1's host timeline (`ATOM_DSV41_RK_HOSTTL`), offline engine, closed loop with refill, per-bs median GPU step on the
decode steps at bs = CONC. 3 passes per node (jobs wp2r2a on n07-17, wp2r2d on n01-09). Every lever alone and in
the pairs that make sense (`nobcast` does nothing without `argmax`):

| GPU step, arm / mean(base, twin) | node | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|---|
| mono, all | n07-17 | 0.991 [0.990, 0.999] | 0.993 [0.987, 0.998] | 0.990 [0.988, 0.993] | 0.990 [0.990, 0.993] | 0.985 [0.983, 0.987] |
| mono, all | n01-09 | 0.993 [0.990, 0.995] | 0.991 [0.989, 0.995] | 0.991 [0.984, 0.997] | 0.995 [0.989, 0.996] | 0.987 [0.986, 0.988] |
| mono, argmax | n07-17 | 1.000 [0.997, 1.000] | 0.993 [0.992, 0.997] | 0.995 [0.993, 0.997] | 0.990 [0.989, 0.995] | 0.985 [0.983, 0.990] |
| mono, argmax | n01-09 | 0.999 [0.995, 1.002] | 0.993 [0.986, 1.002] | 0.995 [0.992, 0.996] | 0.993 [0.989, 0.996] | 0.984 [0.983, 0.986] |
| mono, dkv | n07-17 | 1.003 [0.999, 1.008] | 1.000 [0.998, 1.009] | 0.997 [0.996, 1.003] | 0.997 [0.996, 0.999] | 0.998 [0.998, 1.002] |
| mono, dkv | n01-09 | 1.003 [0.998, 1.004] | 1.000 [0.999, 1.007] | 1.001 [1.000, 1.003] | 0.999 [0.994, 1.000] | 0.997 [0.995, 0.999] |
| mono, argmax + nobcast | n07-17 | 0.992 [0.989, 0.993] | 0.993 [0.990, 0.993] | 0.991 [0.985, 0.992] | 0.987 [0.986, 0.988] | 0.985 [0.983, 0.986] |
| mono, argmax + nobcast | n01-09 | 0.992 [0.990, 0.994] | 0.992 [0.987, 0.995] | 0.991 [0.988, 1.000] | 0.984 [0.981, 0.991] | 0.984 [0.984, 0.986] |
| mono, argmax + dkv | n07-17 | 0.999 [0.996, 0.999] | 0.995 [0.992, 0.996] | 0.995 [0.991, 0.997] | 0.988 [0.988, 0.994] | 0.989 [0.985, 0.990] |
| mono, argmax + dkv | n01-09 | 0.998 [0.996, 0.999] | 0.998 [0.994, 0.999] | 0.994 [0.992, 0.994] | 0.990 [0.990, 0.996] | 0.985 [0.983, 0.989] |
| mono A-vs-A twin (base b / base) | n07-17 | 1.002 [1.001, 1.007] | 0.999 [0.997, 1.011] | 1.000 [0.998, 1.001] | 1.000 [0.996, 1.006] | 0.998 [0.996, 0.998] |
| mono A-vs-A twin (base b / base) | n01-09 | 0.998 [0.996, 1.002] | 1.000 [0.998, 1.001] | 1.002 [0.996, 1.010] | 1.000 [0.999, 1.003] | 0.995 [0.992, 0.996] |
| native, all | n07-17 | 0.993 [0.993, 0.996] | 0.995 [0.991, 0.996] | 0.989 [0.988, 0.989] | 0.991 [0.985, 0.992] | 0.991 [0.990, 0.993] |
| native, all | n01-09 | 0.996 [0.995, 0.997] | 0.993 [0.991, 0.995] | 0.992 [0.992, 0.993] | 0.990 [0.988, 0.991] | 0.982 [0.982, 0.986] |
| native A-vs-A twin | n07-17 | 1.001 [0.997, 1.003] | 0.994 [0.994, 1.005] | 1.000 [0.996, 1.003] | 1.007 [0.999, 1.012] | 1.001 [0.992, 1.004] |
| native A-vs-A twin | n01-09 | 1.002 [1.001, 1.003] | 0.998 [0.997, 1.008] | 1.001 [1.000, 1.005] | 1.001 [1.000, 1.005] | 1.002 [1.000, 1.004] |

C16 decodes natively on both arms: 96 rows is above mono's MAX_ROWS 48. Round 1 (1 pass per node, jobs wp2n1 /
wp2a7) gave the same picture for `all`: 0.984-0.997.

What the lever split shows, the same on both nodes:
- **`argmax` carries the GPU gain at C2-C16** (0.984-0.995). At C1 it is a tie alone (0.999-1.000): the head GEMM,
  argmax pack and pair all-gather it moves into the target graph cost about what the eager head and samplers cost.
- **`nobcast` is the C1 gain**: argmax + nobcast is 0.992 at C1 on both nodes, against 0.999-1.000 for argmax alone.
  Dropping one small TP broadcast per step is worth about 0.007 of the C1 step, 0-0.009 above.
- **`dkv` is a tie on the GPU step** (0.997-1.003, inside the twins) at every CONC, alone and next to argmax. Its value
  is on the host (§3.2).
- All three together are 0.985-0.995 (mono) and 0.982-0.996 (native). In 19 of 20 cells the `all` median is below the
  same cell's twin median, by 0.005-0.020; the exception is native C2 on n07-17 (0.995 against 0.994). The per-pass
  ranges of `all` and twin are disjoint in 16 of 20 cells; they overlap at mono C2 (n07-17), mono C4 (n01-09) and
  native C2 / C16 (n07-17).

Segment view at mono C1, as fractions of the base step (round 1, both nodes):
- the eager draft-KV segment shrinks from 0.0086 to 0.0011 of the step, the sampler segment from 0.012 to 0.0042;
- the target segment grows by 0.005-0.010 of the step: the head GEMM, argmax pack, pair all-gather and draft-KV
  kernels now run inside it;
- net about -0.01.

### 3.2 Host's own work per step

Host time after the start-of-step wait (P1's metric: every host segment but `prep`), as a fraction of the GPU step;
median over 3 passes (same jobs):

| host own work / GPU step | node | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|---|
| mono base | n07-17 | 0.242 | 0.190 | 0.136 | 0.094 | 0.111 |
| mono base | n01-09 | 0.249 | 0.195 | 0.140 | 0.097 | 0.132 |
| mono base twin | n07-17 | 0.249 | 0.195 | 0.141 | 0.094 | 0.112 |
| mono base twin | n01-09 | 0.246 | 0.195 | 0.140 | 0.097 | 0.129 |
| argmax | n07-17 | 0.206 | 0.163 | 0.118 | 0.082 | 0.126 |
| argmax | n01-09 | 0.200 | 0.156 | 0.114 | 0.079 | 0.103 |
| dkv | n07-17 | 0.146 | 0.113 | 0.081 | 0.059 | 0.085 |
| dkv | n01-09 | 0.152 | 0.119 | 0.085 | 0.058 | 0.084 |
| argmax + nobcast | n07-17 | 0.195 | 0.156 | 0.114 | 0.080 | 0.119 |
| argmax + nobcast | n01-09 | 0.198 | 0.155 | 0.109 | 0.079 | 0.118 |
| argmax + dkv | n07-17 | 0.107 | 0.087 | 0.062 | 0.043 | 0.089 |
| argmax + dkv | n01-09 | 0.109 | 0.086 | 0.061 | 0.043 | 0.091 |
| all | n07-17 | 0.096 | 0.076 | 0.055 | 0.038 | 0.086 |
| all | n01-09 | 0.097 | 0.076 | 0.055 | 0.038 | 0.087 |
| native base | n07-17 | 0.242 | 0.224 | 0.186 | 0.150 | 0.113 |
| native base | n01-09 | 0.279 | 0.258 | 0.216 | 0.173 | 0.131 |
| native base twin | n07-17 | 0.284 | 0.262 | 0.218 | 0.173 | 0.132 |
| native base twin | n01-09 | 0.284 | 0.262 | 0.217 | 0.174 | 0.117 |
| native all | n07-17 | 0.190 | 0.173 | 0.147 | 0.118 | 0.089 |
| native all | n01-09 | 0.146 | 0.133 | 0.112 | 0.090 | 0.069 |

- The eager draft-KV write is the largest host cost the tail removes (`dkv`: 0.25 -> 0.15 at mono C1), then the eager
  head and samplers (`argmax`: 0.25 -> 0.20). Together (`argmax + dkv`) 0.11; `nobcast` takes the last 0.01.
- The native base's own work moves by up to 0.04 between its twin arms and passes (its graph launch is about 4x
  mono's, P1); compare native cells within a row pair only.

### 3.3 Served e2e

B0 client: per-request median decode tok/s, GSM8K prompts, 1024 tokens, mono arms unless marked; per-pass ratio of
TAIL (`all`) to the mean of base and twin, median [range] over 3 rotated passes, fresh server per arm-run. C1-C4 are
round-1 jobs (wp2s1 on n01-09, wp2s7 on n07-17), C8 / C16 round-2 jobs (wp2r2d on n01-09, wp2r2a on n07-17).

| served, TAIL / mean(base, twin) | node | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|---|
| mono, all | n01-09 | 1.019 [1.004, 1.027] | 0.999 [0.942, 1.067] | 0.996 [0.950, 1.014] | 1.008 [0.938, 1.038] | 0.973 [0.939, 0.994] |
| mono, all | n07-17 | 0.992 [0.991, 1.020] | 1.046 [1.011, 1.047] | 1.032 [1.010, 1.051] | 1.051 [0.986, 1.098] | 1.038 [1.021, 1.040] |
| mono A-vs-A twin | n01-09 | 1.017 [0.986, 1.024] | 0.989 [0.901, 1.001] | 0.958 [0.952, 0.995] | 1.006 [0.927, 1.078] | 0.964 [0.954, 1.063] |
| mono A-vs-A twin | n07-17 | 1.020 [0.952, 1.050] | 0.924 [0.870, 1.092] | 0.994 [0.968, 1.052] | 1.004 [1.002, 1.046] | 0.990 [0.959, 0.998] |
| native, all | n01-09 | - | - | - | 1.040 [1.015, 1.169] | 1.006 [0.949, 1.040] |
| native, all | n07-17 | - | - | - | 1.020 [0.989, 1.039] | 1.035 [0.961, 1.071] |
| native A-vs-A twin | n01-09 | - | - | - | 0.994 [0.980, 1.141] | 0.989 [0.910, 1.004] |
| native A-vs-A twin | n07-17 | - | - | - | 0.995 [0.953, 1.013] | 1.008 [0.970, 1.064] |

- The A-vs-A twins alone span 0.87-1.14 per pass across these cells; the TAIL passes span 0.94-1.17. Per cell the
  TAIL range is not always inside the twin's range of the same cell (n07-17 C16: TAIL 1.021-1.040 against twin
  0.959-0.998; n01-09 C16: TAIL 0.939-0.994 against twin 0.954-1.063), but the sign is not consistent across nodes:
  C16 is 1.038 on n07-17 and 0.973 on n01-09, C2 1.046 and 0.999, C1 0.992 and 1.019. The single-pass outliers of
  round 1 (n07-17 C2 0.830, C8 1.093; n01-09 C16 1.096) do not recur with 3 passes. **Tie at every cell; no e2e win
  is claimed, and none is excluded below about 5 %.**
- The spread comes from prefill stalls landing in per-request medians at C2-C16 (P1 §3). A 1 % GPU-step gain is below
  what the served path resolves with 3 passes.
- These served runs used stages that also run the levers on padded replays; the default now takes ATOM's eager tail
  there. C1-C8 cells never pad (every bs <= 8 is a capture size). C16 pads only while requests drain or are admitted;
  job wp2r2e re-times C16 on the default (§3.4).

### 3.4 Exact-width default: re-gate and C16 re-time (job wp2r2e)

The gate half is in §2.1 / §2.2. Served C16 on the default (stage `wp2f`, n01-09, 3 rotated passes, fresh server per
arm-run):

| served C16, TAIL / mean(base, twin) | ratio | A-vs-A twin |
|---|---|---|
| mono, all | 1.016 [0.970, 1.022] | 0.985 [0.963, 1.000] |
| native, all | 0.948 [0.926, 1.018] | 1.071 [1.056, 1.097] |

Same verdict as §3.3: a tie. The native twin's own ratio (1.07) shows the C16 served noise floor on that node.

## 4. DMETA and MSTEP: why not built, and what would change that

- **Headroom measured, not assumed.** On both nodes, at every CONC, with or without TAIL (rounds 1 and 2, every
  host-timeline arm-run):
  - the inter-step GPU gap is 0.0003-0.0011 of the step;
  - the start-of-step GPU-drained flag (`q`) is set on 0-0.19 of the steps.

  The GPU almost never waits for the host. Metadata built on the device, or k steps per host round trip, would move
  work the GPU does not wait for.
- **The GPU part of "prepare / meta" does not disappear either.** The `prep` segment is 0.003-0.008 of the GPU step
  (0.007-0.008 at C1). It is H2D copies plus `fill_deferred_decode_ids` and the metadata kernels. Computing the same
  metadata with device kernels keeps a comparable GPU segment; only the copies could go.
- **When it becomes necessary.** DMETA becomes necessary when host own work exceeds the GPU step. With TAIL that is
  about 0.10 of today's mono C1 step, so at roughly 10x today's step speed (native: about 5x). No plan in D0 gets there.
  Without TAIL the threshold is about 4x.
- **What a build would need, recorded for whoever picks it up.** V4.1's per-step metadata is far richer than Qwen's
  TRIM+ADV:
  - per-request `RequestSpan`s and block rows;
  - two compressor plans with `extra_write` slack;
  - Engram rows and the tentative cursor;
  - FP4 walk plans;
  - the `num_rejected` shift in `prepare_decode` (`backend.py::_prepare_step`, `cache.begin_step`,
    `prepare_model_inputs`).

  MSTEP additionally needs a scheduler that can run k verify rounds without the host. That means device-side
  finish / admit detection, with fallback to the host.
- **Consequence for the WPs that listed WP2 as their enabler.**
  - WP3c (ragged verify width): the width has to come from the host, with ATOM's own q-bucket graphs.
  - WP6 (device-side build choice): it uses the host's graph key (one capture per width bucket, as today).
  - WP7 (TP1): it rides on TAIL. TAIL's `argmax` lever is a no-op at TP1, where ATOM already captures the logits in
    the graph; `dkv` applies there.

  None of the three needs DMETA to run.
- **Verdict.** Both stay unbuilt. They are not losers to keep as knobs: there is nothing to time until the step is
  several times faster. Re-open them after WP4 / WP5 / WP6, using the same host-timeline metric (host own work per
  GPU step), when that ratio passes about 0.7.

## 5. Rejected-on-base / lookback record

Base: B0 (atom main df86c482 + W12 default-off knobs), mono and native. Timed alone, in pairs and together on both
nodes with 3 rotated passes (§3.1, §3.2); every lever stays a knob.

| lever | GPU step vs base (both nodes) | host own work | status on base B0 |
|---|---|---|---|
| `argmax` | 0.999-1.000 at C1, 0.984-0.995 at C2-C16 | -0.04 to -0.05 at C1 | win at C2-C16, tie at C1; kept in `all` |
| `dkv` | 0.997-1.003 (tie at every CONC) | -0.10 at C1 | GPU tie, host win; kept in `all` |
| `nobcast` (with argmax) | -0.007 at C1, 0 to -0.009 above | -0.005 to -0.01 | win at C1 and C4-C8; kept in `all` |
| `all` | 0.985-0.995 mono, 0.982-0.996 native | 0.25 -> 0.10 at mono C1 | default-off candidate for the integrated stack |
| `pad` | not timed separately | - | **rejected on base B0**: not bitwise at padded widths (§2.1) |

- Order is not a knob here: the levers act at different points of the step (graph body, eager draft-KV skip, broadcast
  skip) and do not interact except through `argmax` gating `nobcast`.
- Status: "tie on base B0 e2e, about 1.01 in situ". Keep default off and re-test on the integrated base: once WP4 / WP5
  shrink the target graph, the same saving is a larger share of the step. Re-test `pad` with any change of the
  draft-KV projection kernel or of the capture sizes (with a capture size per batch size, nothing pads).

## 6. Jobs and artefacts

- Round 1:
  - 68256800 `wp2g1`: gate and screen, n07-17;
  - 68257117 `wp2a7`: check gate, timing, accuracy, n07-17;
  - 68257118 `wp2n1`: timing, n01-09;
  - 68257355 `wp2s7` and 68257356 `wp2s1`: served, 3 passes each, n07-17 and n01-09.
- Round 2 (`hackathon-prep/dsv41/wp2/r2/`):
  - 68259109 `wp2r2a` (profile A, stage `wp2d`, n07-17): padded-width check, all 7 mutants, in-situ 3 passes x 10 arms,
    served C8 / C16 3 passes x 6 arms;
  - 68260442 `wp2r2d` (profile C, stage `wp2e`, n01-09): padded-width check with the draft-KV diagnostics and the
    eager twin, in-situ and served as wp2r2a;
  - 68260762 `wp2r2e` (profile E, stage `wp2f`): re-gate of the exact-width default, `pad` arm, 7 mutants, served C16.
  - 68259510 (`wp2r2c`) was cancelled at start because it would have taken the workflow over its 8-GPU budget;
    68259110 (`wp2r2b`) was replaced by it before starting.
- Results: `~/rocke-sdpa-claude/dsv41/results/<tag>` (Austin home). Copies with absolutes are in
  `hackathon-prep/private-perf/dsv41-mono/raw/<tag>`.
- Finding for the baseline owners: ATOM V4.1 TP2 decode is not bitwise reproducible run to run, even in the offline
  engine at C1 with fixed batch composition, mono and native alike. Every WP that claims parity needs an in-place check
  like §2.1, not a token-stream diff. And every WP that moves work into the FULL graph must check padded replays apart:
  the graph runs every op at the capture width, ATOM's eager tail at the scheduled rows.
