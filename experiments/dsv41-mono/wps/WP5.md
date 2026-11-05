# WP5: in-kernel CCL for the TP2 reductions and merges

W13, 2026-11-03. MI355X (gfx950), TP2 no-EP, DSpark 5, bf16 KV, fp8 index plane, `--level 0` + FULL graphs (the B0
recipe flags), ATOM df86c482 export (`dev/ATOM-dsv41`) with the W12 / W13 default-off knobs. Ratios only; absolute
numbers are in `hackathon-prep/private-perf/dsv41-mono/WP5-abs.md`.

## Verdict

**The D0 expectation (C1 1.08, C2 1.06, C4 1.03, C8 1.04, C16 1.05) is refuted by measurement.** The mono's TP2
exchange is latency-bound, not byte-bound, and its latency is the xGMI hop that no CCL form removes. What WP5 delivers is
about 0.5-0.7 % of the C1 GPU step (1.2 % with W12's TAIL), nothing resolvable at C2-C16, and 3-10 % of the decode step
at C32 / C64 from quantized wide all-reduces (tolerance class, accuracy-gated).

| knob `ATOM_DSV41_RK_CCL=` | what | gate | in-situ GPU step, both nodes | served e2e |
|---|---|---|---|---|
| `epoch` | epoch-tagged mailboxes, no per-step peer zero, no step fence | **PASS** bitwise, run as served (`epoch` alone: 8648 launches, 0 differ; `epoch+argmax`: 8820, 0); stale-epoch mutants `noepoch` / `noepochd` NOT caught in this form (equivalent here, section 2); delay control passes | C1 **0.994-0.996** (4 runs); C2-C8 tie | see `ea1` |
| `argmax` | Markov-step (and TAIL head) cross-rank merge in one kernel | **PASS** bitwise (Markov 0 / 1585 rows, 0 / 1630 as `epoch+argmax`, 0 / 1565 as `1`; head 0 / 7062 rows, 0 / 6888 as `1`); `argswap` caught | C1 **0.993-0.998**; draft segment **0.958-0.983** at C1-C8 | see `ea1` |
| `epoch+argmax` (`ea1`) | both | PASS (above) + passkey 40 / 40; GSM8K strict inside the mono twin spread, flexible 0.0015 ABOVE the upper mono twin (0.2 stderr): outside the two-run spread, not distinguishable from it | C1 **0.993-0.998**; with W12 `TAIL=argmax` **0.987-0.988** vs its TAIL twins | C1: 4 passes of 32 prompts (n01-09) **1.007 [0.989, 1.015]**, inside the twin spread (0.976-1.025); an earlier 16-prompt run read 1.007-1.045 on both nodes (noise); C2-C16 inside the twin spread: **no e2e claim** |
| `lamport` | untagged partial words (half the bytes) | **PASS** bitwise, run as `lamport` alone (8648, 0 differ); `droptile` / `peer2` caught in `lamport` alone and in `epoch+lamport` | tie (0.990-1.005) | not run (no step gain) |
| `dup` (probe) | every partial pushed twice | numerics unchanged | tie (0.990-0.999): **bytes on the link cost nothing** at 6-48 rows | - |
| `lat` (probe) | a fixed `s_sleep` before every push | numerics unchanged | C1 1.014, C4 1.023, C8 1.025: **the exchange latency is exposed** | - |
| `arfp8` (+ `AITER_QUICK_REDUCE_QUANTIZATION`) | native decode all-reduces of >= 1500 KiB through aiter's quick all-reduce, quantized | tolerance: passkey 40 / 40 for every codec. GSM8K NOT inside the native twin spread for every codec: FP8 strict 0.0053 above its same-job upper twin (0.6 stderr), INT6 flexible 0.0008 below its same-job lower twin (0.1 stderr); FP8 flexible and INT6 strict are inside; INT4's only same-job comparator is one native run (+0.0015 / +0.0023), so it has no same-job spread to be inside. Every excursion is under one binomial stderr: no codec is distinguishable from native, but "inside the twin spread for all codecs" is not what was measured | C32: FP8 0.971-0.972, INT6 0.963, INT4 0.947-0.948; C64: 0.935-0.936 / 0.915-0.922 / 0.896-0.900; C1-C16 tie (path not taken); **both nodes** | C16-C64: inside the native twin spread (0.85-1.02 at C32 / C64), not resolvable: **not claimed** |
| CCL-LM `lm_select` for the wide all-reduces | - | - | **not built**: bounded below 0.01 of the step (section 5) | - |

3x: not reachable from WP5. Its measured share is about 1.005-1.012 at C1 and nothing at C2-C16; at C32 / C64 the
quantized all-reduce is worth 1.03-1.11 on the decode step.

## 1. What was built (`dev/ATOM-dsv41`, additive, default off)

All forms are behind `ATOM_DSV41_RK_CCL=<forms>` (joined by `+`, `,` or `:`; `1` / `all` = `epoch+lamport+argmax`).
The host hooks and the kernel side parse it the same way since the verifier round: the hooks (`embed_head`,
`dspark`, `communication_op`) did a raw substring test, so `=1` / `=all` built the `epoch+lamport` kernels but never
engaged the argmax merges (gate `c_1old` on stage b3: no merge logged, no Markov records). They now call
`rk/ccl_env.host_on` (new, `os` / `re` only, the grammar of `rk_ccl.env_forms`); `hackathon-prep/dsv41/wp5/
cpu_checks_env.py` holds the two parsers equal on 27 probe strings, checks each hook goes through `host_on`, and its
must-fail control (the old substring hook) disagrees on `1`, `all` and others; it FAILs on the old b3 tree. On stage
b4 `=1` engages the Markov merge (gate `c_1`: 1565 rows, 0 differ) and the head merge under `TAIL=argmax` (tailgate
`t_1`: 654 steps a rank, 6888 id rows, 0 bad; the merge kernel is logged on, but that log line does not name the
site, so the head hook's switch rests on `cpu_checks_env.py`).

Unset, nothing changes. Knob-off identity (local trace of the traced MLIR, `hackathon-prep/dsv41/wp5/
local_trace_ccl2.py`, a copy of `local_trace_ccl.py` that runs under the current local torch): K1 at 6 / 12 rows and
K2 at 6 / 12 rows (12 token-parallel) are byte-identical to a fresh `git archive` of df86c482 for the stages b3 and b6
(`lt_k{1,2}_pristine_s{6,12}.txt` vs `lt_k{1,2}_{b3off,b6off}_*`; the earlier K2 claim had no pristine K2 trace, it
has one now, and it also equals the earlier `lt_k2_off_*`); control: the `epoch` K2 build differs from pristine. Unset,
each host hook is an env test that imports nothing and returns ATOM's path. The served kernel forms trace identically
on every stage they were timed or gated on: `epoch` and `lamport` K1 / K2 at 6 rows are byte-equal on b1 (wp5b), b2
(wp5c), b3 (wp5d / wp5f / gate g1) and b6 (gate g2); `epoch+lamport` K2 on b6 equals the b1-era trace.

- `mono/kernels/rk_ccl.py` (new): the kernel forms and the epoch step begin.
  - **`epoch`.** Every mailbox tag becomes `layer + 1 + (epoch mod 2^24) << 8` (K1, K2a, K2b, draft). The epoch is an
    int64 past the last peer base of the `peers` table (entry `tp`); kernels read it at entry. A new step-begin kernel
    (one launch, as ATOM's) advances it and zeroes only this rank's scratch: no zero of the peer receive region and no
    cross-GPU fence, except on the first step of every 2^24 (the tag wrap, and the very first step), which runs
    ATOM's zero + fence. Word-a-tag counters (`moe.take_ug_task`, `take_down_rank`) index by the tag's layer bits.
  - **`lamport`.** The `wo_b` and MoE-down bf16 partial pairs go to the peers as bare 32-bit words: the bit complement
    of the packed pair (never 0; the all-ones NaN pair is sent as a canonical NaN pair, the only output change). The
    consumer polls the words, sums the ranks in rank order 0 .. tp - 1 in fp32 exactly as ATOM's `sum_partials`, and
    writes 0 back; K2a drains the resets before its next hand-off (K2b's sit before the kernel end). The words live in
    their own area past every width's peer layout (`lam_base`, 64 token rows): see the bug in section 2.
  - **`dup` / `lat`** probes (section 3).
  - Build keys and kernel names carry the form (`_ccl<letters>`), so ATOM's build and a CCL build coexist in one
    process (the gate builds both).
- `rk/ccl.py` (new): the gate (`ATOM_DSV41_RK_CCL_CHECK=<dir>`, `..._CHECK_EVERY=<n>`), refusal of
  `ATOM_MONO_CHECK` with a kernel form.
- `rk/ccl_argmax.py` (new): **`argmax`**. One 16-CTA x 64-thread FlyDSL kernel per merge replaces aiter's custom
  all-gather of the `[N, 2]` (max, id) pairs plus the pick (`argmax_gather`, or torch `argmax` + `gather` + copy for the
  head). Each rank pushes its pairs as tagged words (system scope) into every rank's slot and polls its own; the tag is a
  per-64-row call counter the kernel advances itself (graph-replay safe), and two slots alternate by its parity. The
  pick is `argmax_gather`'s: strictly larger wins, equal keeps the lower rank, the first NaN wins. The per-rank pairs
  are ATOM's own (`dspark_markov_sample._launch`, `lm_head_argmax_pack`), so the ids are bitwise ATOM's.
  Hooks: `models/deepseek_v41/dspark.py` (`_head_and_sample_sharded`'s Markov loop) and
  `model_ops/embed_head.py` (`compute_argmax_token`, used by W12's `TAIL=argmax`).
- `rk/ccl_arfp8.py` (new): **`arfp8`**. `model_ops/communication_op.tensor_model_parallel_all_reduce` hands a bf16
  decode all-reduce between `ATOM_DSV41_RK_CCL_ARFP8_MIN_KB` and `..._MAX_MB` to aiter's `QuickAllReduce`, whose codec is
  `AITER_QUICK_REDUCE_QUANTIZATION` (FP8 / INT6 / INT4). aiter's own quick-reduce size window is switched off, so only
  these calls quantize; prefill keeps ATOM's all-reduce.
- Edits to existing files (all marked `W13 WP5`, each a call that returns ATOM's object when off):
  `mono/kernels/{attn_pre,layer_post,attn_post,moe}.py`, `mono/runner.py`, `models/deepseek_v41/dspark.py`,
  `model_ops/{embed_head,communication_op}.py`.
- No rocKE kernel was needed: the TP2 exchange lives inside ATOM's FlyDSL K1 / K2, so the forms are FlyDSL edits of
  those kernels; nothing was added to `rocke/.../dsv41_mono/`.
- Harness (new): `hackathon-prep/dsv41/wp5/` (`wp5_job.sh`, `h/wp5_run.sh` with phases gate / tailgate / htl / served /
  acc / cclmb / contract, `h/ccl_chk.py`, `h/wp5_armb.py`, `wp5_sum.py`, `local_trace_ccl.py`, `mkstage.py`), plus
  copies of W12's `p1_driver.py`, `p1_htl.py`, `wp2_gate.py`, `niah.py`.

## 2. Gates

**Parity gate (bitwise, in place).** `ATOM_DSV41_RK_CCL_CHECK` in an eager offline engine (GSM8K, groups of 1 / 4 / 8,
every 2nd step): each mono K1 and K2 launch first runs ATOM's own build of the width on the same inputs, every buffer
the launch can write is restored (scratch, this rank's peer region, every row buffer, the selections), the CCL build
runs, and every byte is compared. The mailbox scratch and the peer region hold the forms' own encodings, so they are
reported and not gated. Collective (both ranks, barriers between the phases).

| run (job) | form | launches compared (both ranks) | widths | launches with any differing output byte | verdict |
|---|---|---|---|---|---|
| `c_ela` (wp5b, stage b1) | epoch + lamport + argmax | 8648 (target K1 + K2, draft K1 + K2) | target 6-48 rows (12-48 token-parallel), draft 5-40 | 0 | **PASS** |
| `c_ela` Markov merges (wp5b) | argmax | 1585 rows | - | 0 ids differ | **PASS** |
| `c_e` (g1, stage b3) | **epoch alone (the form `e1` / `ea1` time and serve)** | 8648 | target 6-48, draft 5-40 | 0 | **PASS** |
| `c_l` (g1, b3) | **lamport alone** | 8648 | target 6-48, draft 5-40 | 0 | **PASS** |
| `c_ea` (g1, b3) | **epoch + argmax (the served `ea1`)** | 8820; Markov 1630 rows a rank | target 6-48, draft 5-40 | 0; 0 ids differ | **PASS** |
| `c_1` (g1, stage b4 = b3 + the hook fix) | `1` (= epoch + lamport + argmax) | 8488; Markov 1565 rows a rank | target 6-48, draft 5-40 | 0; 0 ids differ | **PASS** |
| `t_1` head merge in the FULL graph (g1, b4) | `1` + `TAIL=argmax` | 654 steps a rank, 6888 id rows | C1 / C4 | 0 | **PASS** |
| `c_1old` (g1, b3: the old substring hooks) | `1` | 8304; no Markov records | - | 0 | control: kernels `epoch+lamport`, argmax merge NOT engaged (the doc / impl mismatch, fixed in b4) |
| `t_a` head merge in the FULL graph (wp5c, W12's `TAIL_CHECK`: in-graph ids vs eager full-logits argmax) | argmax + `TAIL=argmax` | 631 checked steps a rank, 7062 id rows | target widths of C1 / C4 | 0 | **PASS** |
| mutant `noepoch` (epoch step begin, ATOM's tags) | epoch + lamport | 8304 | 6-48 | 17 | must fail: **caught** |
| mutant `noepoch` (g1, b3) | **epoch alone** | 8992 | 6-48 | 0 | **NOT caught**: equivalent in this gate (below) |
| mutant `noepochd` (g2, stage b6: `noepoch` + the last rank's pushes delayed by a memrealtime spin) | epoch alone | 8476 | 6-48 | 0 | **NOT caught**: equivalent in this gate (below) |
| control `delay1` (g2, b6: the same delay, real epoch tags) | epoch alone | 8316 | 6-48 | 0 | must pass: **PASS** |
| mutant `peer2` (g1, b3) | **lamport alone** | 22076 | 6-48 | 11038 | **caught** |
| mutant `droptile` (g1, b3) | **lamport alone** | 184 (engine died: illegal address, as in wp5b) | 5-24 | 90 | **caught** |
| mutant `droptile` (one rank's word read without waiting) | epoch + lamport | 184 (then the engine died: illegal address downstream of the garbage) | 10-24 | 90 | **caught** |
| mutant `peer2` (the peer's partial summed twice) | epoch + lamport | 21892 | 6-48 | 10946 | **caught** |
| mutant `order` (rank order reversed) | epoch + lamport | 8660 | 6-48 | 0 | **equivalent at TP2** (a + b == b + a in IEEE fp32), as predicted: the spec's "swapped rank order" cannot fail at TP2; `peer2` is the discriminating replacement |
| mutant `argswap` (winner's value, other rank's id) | argmax | 5795 Markov rows; head 28104 rows | - | all | **caught** |
| mutant `argtie` (equal values keep the higher rank) | argmax | 1565 Markov rows | - | 0 | report-only: no cross-rank tie in the sample |

- **The stale-epoch mutant does not discriminate the pair form (`epoch` alone).** In the in-place gate the reference
  is ATOM's own kernel run under the same (epoch) step begin from the same restored state. `noepoch` keeps ATOM's
  tags, so in the pair form its kernel is the reference's kernel: its mailbox state equalled the reference's byte for
  byte (`mailbox_diff` scratch 0, peers 0, in every record), and only a race could make them differ. Delaying the
  last rank's pushes (`noepochd`, so the other rank polls first) did not create one: no stale pair with the polled
  tag was present, most likely because later layers of the same step rewrite every pair slot with other layer tags.
  The mutant fails where slots are not rewritten each step (`epoch+lamport`: 17 / 8304). So, measured: the `epoch`
  form is bitwise equal to ATOM's kernel on 8648 launches and stays equal with one rank delayed (`delay1`); the claim
  "the epoch tag is what keeps a stale pair out" is shown only for `epoch+lamport`, not for the pair form, where ATOM's
  per-layer tags and slot reuse already keep stale pairs out in every launch the gate saw. A deterministic stale-pair
  mutant for the pair form (one that forces a same-tag pair to survive a step) was not built.
- The wp5a gate runs (`gate_c_e`, `gate_c_l`, `gate_c_ela` on stage a1) compared nothing (verdict NO_DATA in
  `raw/wp5a/*/chk_sum.txt`: the frozen-dataclass bug below); every gate verdict above is from wp5b, wp5c, g1 or g2.
- **CPU invariants** (`hackathon-prep/dsv41/wp5/cpu_checks.py`, local, no GPU; re-run on stage b6 through
  `cpu_checks_run2.py`, all pass): the lamport area is disjoint from
  every width's peer layout (1-64 rows) at TP2 / 4 / 8 and fits the peer buffer; the attn / ffn word ranges are
  disjoint; the layer tags stay below the epoch bits; env parsing, refusals, symbol mangling and `forced()` scoping.
  Its must-fail controls (the first build's placement at offset 0; a 32-row area for 48-row steps) are flagged as
  violations. All pass.
- **Bitwise vs the aiter all-reduce at 48-384 rows: not applicable.** No bitwise-class wide form was built (CCL-LM is
  bounded below 0.01 of the step, section 5); the wide form that was built (`arfp8`) changes the numerics on purpose and
  is gated by accuracy below.
- **Hand-off contract** (`mono.contract`, traced, every region's producer / consumer stage): holds at S = 1, 6, 12, 48
  for off, `epoch+lamport`, `dup` and `lat`.
- **Accuracy** (served, GSM8K 1319 3-shot chat flexible / strict, passkey 20 x 8K / 24K; mono arms at lm_eval
  concurrency 8 so the mono serves, native arms at 64 so `arfp8` quantizes):

  | arm | flexible | strict | passkey | reading |
  |---|---|---|---|---|
  | mono base / twin | 0.9151 / 0.9022 | 0.1221 / 0.1168 | 20 / 20, 20 / 20 | twin spread |
  | `epoch+argmax` | 0.9166 | 0.1221 | 20 / 20 | flexible 0.0015 above the upper twin (0.2 stderr), strict inside |
  | native base / twin (wp5b: the same-job comparators of FP8 / INT6) | 0.9090 / 0.9045 | 0.1259 / 0.1160 | 20 / 20 each | twin spread |
  | native run (wp5d: the only same-job comparator of INT4) | 0.9052 | 0.1213 | 20 / 20 | one run, no spread |
  | `arfp8` FP8 (wp5b) | 0.9052 | 0.1312 | 20 / 20 | flexible inside; strict 0.0053 ABOVE the upper same-job twin (0.6 stderr) |
  | `arfp8` INT6 (wp5b) | 0.9037 | 0.1228 | 20 / 20 | strict inside; flexible 0.0008 BELOW the lower same-job twin (0.1 stderr) |
  | `arfp8` INT4 (wp5d) | 0.9067 | 0.1236 | 20 / 20 | +0.0015 / +0.0023 vs its one same-job native run; inside the cross-job three-run range |

  Reading: two of the six codec x metric cells fall just outside their same-job two-run spread (FP8 strict above,
  INT6 flexible below), both by less than one binomial stderr (about 0.008 flexible, 0.009 strict at n = 1319). The
  accurate claim is "no codec differs from native by more than the run-to-run noise", not "inside the spread". The
  three-run native range quoted earlier mixed jobs (wp5d's run with wp5b's two); the same-job comparators are above.

  The quantized path was live in these runs: each server logged its codec and a first quantized all-reduce of 3.75 MiB
  (the 64-request graph) on both ranks, and lm_eval kept up to 64 requests in flight, so most decode steps replayed the
  32- / 48- / 64-request graphs whose all-reduces are 1.9-3.75 MiB. GSM8K and passkey do not resolve INT4's larger
  per-call error (0.12 relative L2 against 0.03 for FP8 / INT6): it passes this gate, but P3 found end-to-end accuracy
  insensitive where a layer-level gate was not, so INT6 is the safer default candidate and INT4 needs a layer-level
  numerics gate before it serves.

  Numerics of one all-reduce (microbench, N(0, 1) bf16 data, relative L2 vs the fp32 sum): aiter's custom all-reduce
  1.8e-3 (the bf16 output rounding), quick reduce FP8 3.4e-2, INT6 3.0e-2, INT4 1.2e-1.

**A bug the timing run caught and the first gate missed.** The first `lamport` build reused the mailbox regions'
offsets, which are per width (`peer_layout(s)`): one width's partial words overlapped another width's tagged regions
and words. ATOM's per-step zero hid it (`lamport` alone was fine), but with `epoch` the leftovers survived across
steps, and `epoch+lamport` read stale words at C4 / C8 (acceptance 2.79 / 1.73 against about 3.8). The words now live
in a dedicated area past every width's layout, zero at rest; the fixed build passed the gate at every width and runs at
normal acceptance. The first gate run (wp5a) never engaged the mono: the gate's reference build was assigned to a
frozen dataclass and every width was declined, so it compared nothing. Fixed (`dataclasses.replace`); wp5b's
`epoch+lamport+argmax` run was the only valid gate until the verifier round, which ran `epoch` alone, `lamport` alone,
`epoch+argmax` and `1` (jobs g1 / g2 above).

## 3. Where the exchange cost is: the two probes

Same job, fresh engine per arm, the host-timeline GPU step (P1 metric), ratio to the mean of the base and its A-vs-A
twin (twin spread 0.995-1.006 in these runs):

| probe | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| `dup` (twice the partial bytes on the link) | 0.998 | 0.990 | 0.999 | 0.999 |
| `lat` (one `s_sleep 37` before every push; CTAs with two `wo_b` tasks sleep twice) | 1.014 | 1.006 | 1.023 | 1.025 |

- Doubling the bytes of every `wo_b` / MoE-down exchange changes nothing measurable at 6-48 rows. **No byte-saving
  form can help here**: `lamport` (half the bytes), an fp8 `ar_dtype` for the mono exchange, and NUMA dual copies
  (twice the bytes) are all bounded by this result.
- At C1 the injected sleep comes back as about 0.9 of its own length per exchange in the step: the exchange sits on
  the critical path almost one for one (the arithmetic, with absolutes, is in WP5-abs.md). That latency is the
  system-scope push over xGMI plus the consumer's poll; the forms here keep the same one-way hop. What removes it is fewer exchanges (a different
  parallelization of attention or the MoE), not a CCL form.
- `epoch` removes the step begin's peer zero and its cross-GPU fence (one per runner per step): the measured
  0.4-0.6 % of the C1 target segment. `argmax` removes, per Markov step, aiter's all-gather launch and the pick launch
  (5 steps a draft pass): 3.3-4.2 % of the draft segment, about 0.3 % of the step.

## 4. Timing

In situ: P1's no-profiler host timeline (`ATOM_DSV41_RK_HOSTTL`), offline engine, closed loop with refill, median GPU
step at bs = CONC, fresh engine per arm-run, base and A-vs-A twin in the same job. Ratio = arm / mean(base, twin),
rank 0 (rank 1 identical to 3 decimals). Every ratio was re-derived locally from the raw per-step jsonl (identical).

| arm | node, job, pass | C1 | C2 | C4 | C8 |
|---|---|---|---|---|---|
| A-vs-A twin (twin / base) | n07-17 wp5a / wp5b; n01-09 wp5c p1 / p2 | 1.001 / 1.000; 1.002 / 0.955* | 1.006 / 1.010; 1.000 / 0.995 | 0.999 / 1.002; 1.006 / 1.000 | 0.999 / 1.000; 0.988 / 1.003 |
| `epoch` | n07-17 wp5a, wp5b | 0.995, 0.994 | 0.992, 1.001 | 0.993, 0.999 | 1.001, 0.999 |
| `epoch` | n01-09 wp5c p1, p2 | 0.996, 0.995* | 0.998 | 1.008 | 0.990 |
| `argmax` | n07-17 wp5a, wp5b | 0.993, 0.998 | 0.990, 0.998 | 0.993, 0.993 | 1.002, 0.999 |
| `argmax` | n01-09 wp5c p1, p2 | 0.994, 0.994* | 0.999 | 1.001 | 0.991 |
| `epoch+argmax` | n07-17 wp5b | 0.998 | 0.994 | 0.992 | 0.996 |
| `epoch+argmax` | n01-09 wp5c p1, p2 | 0.993, 0.994* | 0.993 | 1.007 | 0.997 |
| `TAIL=argmax` + `epoch+argmax` vs `TAIL=argmax` twins | n01-09 wp5c p1, p2 | 0.987, 0.988 | 0.996 | 1.005 | 0.995 |
| `TAIL=argmax` + `argmax` vs `TAIL=argmax` twins | n01-09 wp5c p2 (p1: a +9 % one-run stall, excluded) | 0.992 | 1.001 | 0.999 | 0.993 |
| `TAIL=argmax` twins (twin / base) | n01-09 wp5c p1, p2 | 1.007, 1.001 | 0.998, 0.994 | 0.995, 0.998 | 0.987, 0.992 |
| `lamport` (fixed) | n07-17 wp5b | 1.001 | 1.003 | 0.990 | 1.005 |
| `epoch+lamport` (fixed) | n07-17 wp5b | 1.002 | 1.004 | 0.996 | 1.003 |

\* wp5c pass 2's base C1 step is an outlier (+4.7 % against its own twin); those C1 values are against the twin alone.
C2-C8 cells of wp5c are the two-pass medians against mean(base, twin).

Post-hoc exclusions (decided after seeing the data, so weaker evidence): (1) the wp5c pass-2 C1 comparator above;
(2) the `TAIL=argmax` + `argmax` pass-1 run, a one-run stall (+9 %) that is excluded, so that row's C1 0.992 rests on
ONE clean pass (wp5c p2) and is provisional; the `TAIL=argmax` + `epoch+argmax` row (0.987 / 0.988) has both passes.

- **C1: a consistent 0.4-0.7 % from `epoch` and from `argmax`, on both nodes, in every run; they do not add up
  (0.993-0.998 together).** With W12's `TAIL=argmax` (whose head merge then also runs in one kernel) the C1 step is
  0.987-0.988 of TAIL's own.
- **C2-C8: inside the twin spread**, both signs.
- C16-C64 (wp5b, n07-17): the mono forms do not apply (native path); ratios 0.993-1.007.

Served e2e (B0 client, per-request median decode tok/s, 2 passes per node, arm / mean(base, twin) per pass):

| | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| `epoch+argmax`, n07-17 (wp5b) | 1.042, 1.018 | 0.885, 0.979 | 0.951, 1.028 | 0.994, 1.059 | 0.921, 1.000 |
| `epoch+argmax`, n01-09 (wp5c) | 1.045, 1.007 | 0.956, 0.921 | 1.024, 1.099 | 0.994, 1.008 | 1.014, 1.084 |
| twin / base, n07-17 | 1.002, 0.976 | 1.025, 0.926 | 1.011, 1.154 | 0.936, 0.999 | 0.963, 1.035 |
| twin / base, n01-09 | 0.984, 0.963 | 0.979, 0.828 | 1.036, 0.947 | 1.015, 0.992 | 0.859, 1.093 |

- With 16 prompts a run, C1 sat above the twin spread in all 4 passes on both nodes (1.007-1.045 against 0.963-1.002),
  which the 0.2-0.7 % GPU step could not explain. An attribution job with 32 prompts a run and 4 rotated passes per
  arm (wp5f, n01-09) settles it (median of the 4 per-pass ratios [min, max]): `epoch` 1.007 [0.995, 1.015],
  `argmax` 1.010 [1.000, 1.020], `epoch+argmax` 1.007 [0.989, 1.015], against a twin spread of 0.976-1.025. (An
  earlier version of this note said 1.005 for `epoch`; the per-pass values 1.015, 0.995, 1.005, 1.010 give a median of
  1.007 and a mean of 1.006. `srv_p4_e1__c1.json` is now in the private raw copy.) The forms move served C1
  by about what the step predicts, inside the served noise: **no e2e win is claimed**; the booked C1 gain is the
  step's, about 1.005. Its n07-17 twin (wp5e) was cancelled unrun after 1.5 h in the queue (the node was held for 2 h
  more); with no e2e claim, the both-node rule for claimed wins does not depend on it.
- C2-C16: inside the twin spread (the prefill stalls dominate the per-request medians, as in WP2).

Microbench (same job, 2 GPUs, captured graph of 64 back-to-back all-reduces, 20 shuffled rounds, paired per-round ratio
to aiter's custom all-reduce, ATOM's path; its separately captured twin 0.998-1.002):

| per-rank payload (rows x 5120 bf16) | 60 KiB (6) | 120 KiB (12) | 240 KiB (24) | 480 KiB (48) | 960 KiB (96) | 1.9 MiB (192) | 3.75 MiB (384) |
|---|---|---|---|---|---|---|---|
| quick reduce FP8 | 3.27 | 2.64 | 2.06 | 1.44 | 1.07 | 0.81 | 0.65 |
| quick reduce INT6 | 3.83 | 3.11 | 2.38 | 1.58 | 1.09 | 0.78 | 0.57 |
| quick reduce INT4 | 3.48 | 2.87 | 2.21 | 1.45 | 0.96 | 0.65 | 0.46 |

Hence the 1500 KiB threshold: only C32 (192 rows) and C64 (384 rows) quantize. In situ (wp5b, n07-17, native path,
GPU step against the native base and its twin, twin spread 0.993-1.007):

| arm | C1-C8 | C16 | C32 | C64 |
|---|---|---|---|---|
| `arfp8` FP8 | 0.998-1.004 | 0.995 | **0.971** | **0.936** |
| `arfp8` INT6 | 0.999-1.000 | 0.999 | **0.963** | **0.915** |
| `arfp8` INT4 | 0.995-1.002 | 0.999 | 0.947 | 0.896 |

Second node (wp5d, n01-09, 2 passes; C64 from pass 1, where both twins completed the cell; twin spread 0.998-1.005):

| arm | C1 | C16 | C32 | C64 |
|---|---|---|---|---|
| `arfp8` FP8 | 1.001 | 1.006 | **0.972** | **0.935** |
| `arfp8` INT6 | 0.999 | 1.002 | **0.963** | **0.922** |
| `arfp8` INT4 | 1.001 | 1.002 | 0.948 | 0.900 |

This matches the all-reduce share P4 measured (0.16 at C32, 0.22 at C64) times the microbench saving.

Served (wp5d, n01-09, 2 passes, per-request median decode tok/s, arm / mean(native base, twin) per pass):

| | C16 | C32 | C64 |
|---|---|---|---|
| `arfp8` FP8 | 0.976, 0.962 | 0.978, 1.166 | 1.060, 0.926 |
| `arfp8` INT6 | 0.948, 1.026 | 0.952, 1.196 | 0.943, 1.061 |
| native twin / base | 1.017, 1.018 | 0.928, 0.986 | 0.851, 0.979 |

The closed-loop served cells at C32 / C64 swing by up to 15 % between twins (GSM8K replies end early, the prefill
stalls dominate: P4 §2); a 3-8 % decode-step gain cannot be resolved there. Not claimed end to end.

## 5. Not built, with the measured reason

- **CCL-LM (`qwen38_mono/ccl` `lm_select`) for the wide all-reduces (480 KiB-3.75 MiB).** Its measured margin against
  aiter's captured all-reduce at these byte sizes is 0.90 (640 KiB), 0.94 (1.25 MiB), 0.976 (2.5 MiB) and 0.99 (5 MiB)
  (Qwen `large-batch/CCL-LM.md`, same hardware). Times the all-reduce share of the decode step (P4: 0.09 at C8, 0.11 at
  C16, 0.16 at C32, 0.22 at C64) that is at most 0.01 of the step at any cell, before the cost of running a second
  IPC runtime and rocKE launches inside ATOM's graphs. The quantized quick reduce above is worth 3-10x more at C32 / C64.
- **Tile-flag overlap.** ATOM's mono already polls every partial pair on its own tag: a consumer CTA starts on the
  pairs of its 32-column task as they land (pair granularity, finer than a tile flag). Nothing to add.
- **NUMA-homed receive buffers / dual copies.** The `dup` probe shows twice the bytes cost nothing, so a dual copy
  would not lose, but the gain is only the IOD hop of the local poll: 0.95-0.96 of the poll round trip on the same IOD
  versus the other (Qwen P16's sc1 ping-pong), i.e. about 0.05 of the exposed exchange latency: under 0.001 of the C1
  step (WP5-abs.md), below what the timeline resolves.
- **`ar_dtype=fp8` for the mono's in-kernel exchange.** Bounded at zero by the `dup` probe (bytes do not matter at
  6-48 rows). The native wide path's quantized all-reduce is built (`arfp8`).
- **A pipelined poll** (two poll loads in flight, so a landed word is seen half a poll round trip earlier): the `lat`
  probe says cutting 0.3-0.5 of the exposed per-exchange latency would be worth 0.5-0.8 % at C1 (WP5-abs.md). Not built: the poll is ATOM's shared
  `atom.mono.device.sync.Mailbox`, and LLVM drains loop-carried in-flight loads at the back edge (alola rule 185), so it
  needs an asm poll; recorded for WP4's GEMV / stage ports, which own those stages.

## 6. Versus D0

D0 booked WP5 at C1 1.08 (1.03-1.15) on a "CCL bucket 0.19 +/- 0.1 of the step". That bucket was the K2 `slice` /
`reduce` stages' share from eager stamps, which includes the rank skew and the stage's own work. The probes measure the
exchange directly: its bytes are free and its latency, though exposed, is the hop the forms cannot remove. The
achievable part was the per-step reset and fence (`epoch`) and the Markov merges' launches (`argmax`): 0.4-0.7 % each
at C1. Wide (C16): replacement alone is under 0.01 (P4 said the same); the quantized all-reduce does not engage below
about 1 MiB. C32 / C64: 1.03-1.11 on the decode step with FP8 / INT6, accuracy-gated.

## 7. Rejected-on-base / lookback record

- `lamport`: tie on base B0 (C1 1.001-1.004, C4 0.990-0.998, C8 1.003-1.005); kept default off. Re-test only if a later
  WP makes the exchange byte-bound (the `dup` probe is the cheap check).
- `epoch+lamport`: tie on base B0 (after the fix).
- `dup`, `lat`: probes, kept as knobs for re-measuring exposure after WP4 / WP6 change the stages.
- `arfp8`: FP8 / INT6 / INT4 all win on the decode step at C32 / C64 on both nodes; passkey 40 / 40 and GSM8K within
  one binomial stderr of native (two cells just outside the same-job two-run spread, section 2); default
  off (tolerance class; C32 / C64 are information cells). INT6 is the candidate for the integrated stack (best error
  per byte saved); INT4 the fastest, pending a layer-level numerics gate.
- `epoch`, `argmax`, `epoch+argmax`: small C1 wins (step 0.993-0.996), default off; candidates for the integrated
  stack (with W12's `TAIL=argmax`, 0.987-0.988; the `TAIL=argmax` + `argmax` 0.992 is one clean pass). Each is now
  gated bitwise in the form it is served (section 2); the stale-epoch mutant is not discriminating for `epoch` alone.
- Order as a knob: `epoch` then `argmax` were timed alone and together; they do not add (shared C1 headroom in the
  draft / step tail).
- Integration notes: the kernel forms apply to every V4.1 K1 / K2 build, WP4.1's non-indexer K2 build included (it
  goes through the same `build_layer_post`), but the WP5 gate is not wired into WP4's K2SPLIT launch path (check them
  one at a time). W12 WP3's `DVOCAB` replaces the draft's `head_and_sample`, so `argmax` does not reach the Markov steps
  under it. `ATOM_MONO_CHECK` (split K2a / K2b) refuses a kernel form. `ATOM_DSV41_RK_CCL=1` / `all` engage the
  argmax merges only from stage b4 on (`rk/ccl_env.py`); with an older tree name the forms explicitly
  (`epoch+lamport+argmax`).

## 8. Jobs and artefacts

- Jobs (all `rocke`, 2 GPUs): 68259653 `wp5a` (n07-17: microbench, contract, probes, first forms), 68260681 `wp5b`
  (n07-17: gate + mutants, forms, wide arms, served, accuracy), 68260776 `wp5c` (n01-09: forms incl. TAIL arms 2
  passes, served, TAIL head-merge gate), 68261908 `wp5d` (n01-09: wide arms 2 passes, served C16-C64, INT4
  accuracy), 68262576 `wp5f` (n01-09: served C1 attribution of `epoch` / `argmax`; its n07-17 twin 68262575 `wp5e`
  was cancelled before it started).
- Results: `~/rocke-sdpa-claude/dsv41/results/<tag>` (Austin home); copies with absolutes and the summaries in
  `hackathon-prep/private-perf/dsv41-mono/raw/wp5{a,b,c,d,f}`.
- Verifier-round jobs (`rocke`, 2 GPUs, n07-17, harness `hackathon-prep/dsv41/wp5/wp5g_job.sh` + `g/env_b*.sh`, the
  same `wp5_run.sh` / `ccl_chk.py` / `wp2_gate.py` as wp5b by sha): 68263703 `g1` (stage b3: `c_e`, `c_l`, `c_ea`,
  `noepoch` / `peer2` / `droptile` on the single forms, `c_1old`; stage b4: `c_1`, tailgate `t_1`), 68264891 `g2`
  (stage b6: `noepochd`, `delay1`). Results with absolutes: `hackathon-prep/private-perf/dsv41-mono/raw/wp5g1`, `wp5g2`.
- Stages (dev/ATOM-dsv41 snapshots, sha256 in WP5-abs.md): `wp5-atom-a1`..`a3`, `b1`..`b3` under `hackathon-prep/dsv41/wp5/stages`;
  verifier round (`mkstage_patch.py`: a base stage with named files replaced, since the shared tree carries other WPs'
  edits): `b4` = b3 + the three hook files + `rk/ccl_env.py`; `b6` = b4 + `rk_ccl.py` with the `noepochd` / `delay1`
  mutants (`b5` was an untraceable first try, never run). `dev/ATOM-dsv41`'s `rk_ccl.py` differs from b6 only in its
  docstring (the measured mutant outcome).
