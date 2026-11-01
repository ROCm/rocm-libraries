# WP4h: the ATOM call-path contract for Path B (ATOM-truemono)

Status: built, gate met on hardware (2026-10-31, W5 wave 1). Independent verification: not yet run (the workflow's
verifier step; everything below is reproducible from the scripts and raw files named). Ratios only here; absolute
numbers are in `hackathon-prep/private-perf/qwen38-true-mono/wave1/WP4h.md` (raw: `WP4h-raw/`). Below 1 is faster.

Plan of record: D0.1 (Path B = graph-launched `layers=1` rocKE layer kernels + one persistent tail launch), D0 section 5
row WP4h ("Path B hook = `layer` mode + tail launch; carries P9's `input_ids` alias rule"), DESIGN 7.3, 7.4, 7.6.
WP0a built the `layer` hook; WP4h adds the tail hook beside it and fixes the contract a WP4B tail kernel plugs into.

## 1. What was built (ATOM-truemono, host side only; nothing under `kernels/`)

| file | change |
|---|---|
| `atom/models/qwen4_exp_mono/truemono/tail.py` (new) | the tail hook: `TailRequest` (one step width, named as the layer builds name it: `r4s`, `r8g4` .. `r16g4`, `r1` .. `r4`), `StepIds` (P9's alias), `TailStep` (what a tail gets per step), `HybridTail` (the hybrid's own tail: the pass-through and the fallback), `TailHook` (provider loading once per width, per-step routing, transitions, check mode, stats) |
| `truemono/hook.py` | `tail` is a built mode now and keeps the layer hook on (`LAYER_MODES = ("layer", "tail")`); `pass` / `round` / `loop` stay not built |
| `true_mono.py` | `tail_hook(model, atom_config)`: the hook in `tail` mode, None (nothing imported) in every other mode |
| `dispatch.py` | `MonoDecodeModel.q38_tail = None`; `install_mono_decode` sets it in `tail` mode |
| `atom/model_engine/model_runner.py` | `ModelRunner.q38_tail = None` (class attribute), bound after the mono installer; the hybrid's sampling moved verbatim into `_sample_tokens` / `_verify_tokens` (same ops, same order) so the pass-through runs exactly the hybrid's code; `postprocess` asks the hook per step; `propose_draft_token_ids(..., q38_route=)`; graph capture and replay ask the hook whether a width's decode graph captures `compute_logits` |
| `tests/test_qwen4_exp_mono_truemono_tail.py` (new, 58 tests) | modes / imports, widths, alias geometry, provider served / refused / missing / strict / malformed, routing reasons, transitions, owned logits, fallback logits, check mode with must-fail mutants, output signatures, the `ModelRunner` call sites (AST) and behaviour (`postprocess`, `propose_draft_token_ids`, with fakes) |
| `tests/test_qwen4_exp_mono_truemono.py` | one parametrize case removed: `test_unbuilt_modes_keep_the_hybrid[tail]` (tail is built; its replacement is `test_tail_mode_keeps_the_layer_hook`) |

With `ATOM_Q38_TRUE_MONO` unset (or `off` / `layer`), every new branch is `self.q38_tail is None` and the code below it
is the hybrid's. The live `dev/ATOM-truemono` also carries P11's later, independent edits (`runner.py`,
`group_runner.py`, `q38_accuracy.py`, its test); they touch none of WP4h's files, and WP4h's gates ran without them
(3, "Trees"). Probe code: `hackathon-prep/qwen38/truemono/WP4h/` (`job_gate.sh`, `job_e2e.sh`, `wp4h_env.sh`,
`wp4h_providers.py`, `mut_unit.py`, `keys_gate.py`, `serve_sum.py`, `serve_gate.py`, `trans_gate.py`, `e2e_sum.py`,
`e2e_med.py`, `sync.sh`, `overlay.txt`, `local_shim.py`).

## 2. The contract

### 2.1 Switches

| env | meaning |
|---|---|
| `ATOM_Q38_TRUE_MONO=tail` | Path B: layers from the layer provider (`ATOM_Q38_TRUE_MONO_LAYER`, WP0a), the round's non-layer work from the tail provider |
| `ATOM_Q38_TRUE_MONO_TAIL=<module>:<callable>` | the tail provider; default `kernels.gfx950.qwen38_mono.truemono_tail:provider` (the rocKE module WP4B writes; absent today, so the hybrid serves, logged) |
| `ATOM_Q38_TRUE_MONO_STRICT=1` | a width the provider refuses (or a provider that does not import) raises `TrueMonoRefused` at capture instead of falling back; a check mismatch raises |
| `ATOM_Q38_TRUE_MONO_CHECK=<n>` | gates only: for the first n tail steps of each width, the hybrid runs first on the same inputs and the tail's outputs must equal it bit for bit; where the graph captured the logits they are also compared with an eager `compute_logits` of the same hidden states (capture / replay check). Synchronises: never in timed runs |
| `ATOM_Q38_TRUE_MONO_STATS=<n>` | a `tail stats` log line every n routing decisions (default 4096) |

### 2.2 The provider and the tail object

`provider(request: TailRequest, hybrid: HybridTail) -> tail | None`, called once per width (at graph capture for
captured widths). A tail has:
- `owns_logits` (bool). False: the width's decode graph captures `compute_logits` as the hybrid does. True: the graph
  holds no head and `step.logits` is None on its replays (eager steps still have them).
- `verify(step) -> (sampled_tokens, verdict)` (MTP): `sampled_tokens` int32 `[bs, k + 1]`; `verdict` int32 `[3, bs]`
  = accepted drafts (`num_bonus`), rejected (`num_reject`), anchor rows. (The verdict is ONE tensor; WP4h's first
  mutant assumed a 3-tuple and the signature check stopped it on hardware: see 4.3.)
- `sample(step) -> sampled_tokens` (K=0): int32 `[bs]`.
- `propose(step, **kw) -> draft_token`: what `drafter.propose(**kw)` returns; `kw` unchanged (target ids view,
  positions, hidden states, rejected counts, next tokens, anchor rows).

`TailStep` carries `request`, `batch`, `spec` (`SpecDecodeMetadata`), `logits` (or None), `hidden` (MTP: the unmixed HC
streams `[rows, hc, hidden]`; K=0: the mixed hidden state), `ids` (P9's alias, MTP only), `mono` (the mono decode
runner: its `counter` is the one step word all paths share, DESIGN 7.4), and the sampling parameters. The first output
of each kind per width is checked against the hybrid's signature (shape, dtype, device); a mismatch raises.

`postprocess` then runs its own bookkeeping on the tail's tensors exactly as on the hybrid's (`prepare_sampled_ids`,
the deferred-output path, `send_mtp_status_to_cpu_async`, `prepare_draft_ids`): one writer per word (DESIGN 7.3).

### 2.3 Which steps go to the tail (`TailHook.route`); everything else is a transition to the hybrid's code

A step goes to the tail only if all hold, else the hybrid's code runs it and the reason is counted:
real (not dummy) step; no prefill sequence and not a prefill forward; no mid-prompt sequence beside the decodes; a
decode / verify step of the configured K; every request greedy (`not_greedy`); no request asks for logprobs
(`logprobs`; DESIGN 7.6: hybrid until WP8); the graph's width (running batch x (k + 1)) is served by the mono runner
(`mono_not_serving`); the provider has a tail for it (`refused`).

- **Prefill hand-off.** Prefill and mixed steps are ATOM's (DESIGN 7.4); drafts for the first verify come from ATOM's
  proposer over the prefill chunk. The first tail round reads the pools, KV and draft ids that path wrote. Measured:
  every prompt enters the tail from a prefill step (`transition.to_tail` = prompts), see 3.4.
- **Transitions** (CONC change, a non-greedy or logprobs request, a refused width): per step, no state of its own;
  both paths read and write the same pools and per-step buffers and the mono runner's step counter. After a graph
  replay of a width whose tail owns the logits, a step the hybrid takes gets them from `TailHook.hybrid_logits`
  (one eager head) first, so it is a plain hybrid step.
- **Per-width refusal** falls back to the hybrid for that width only, logged once, never "mono off for good".
- **Logprobs** today: ATOM's OpenAI server builds `SamplingParams` without `logprobs` (a served request cannot ask for
  them), and with MTP ATOM refuses a logprobs request at `add_request`. So the `logprobs` route is reachable only
  offline at K=0; it is gated there (3.4).
- **Graph capture**: capture asks `runner_logits(bs, q)` per key after the warm-up forward (so the mono widths are
  prepared); a width whose tail owns the logits captures no head (`graph.no_logits.<variant>`); replay hands
  `logits=None` for exactly those keys. The tail itself runs in `postprocess` (eager, after the replay); a WP4B tail
  that wants its own graph captures it inside the provider.

### 2.4 P9's `input_ids` alias rule (carried as data and as a check)

The verify reads draft r at `spec.input_ids[target_logits_indices[r] + 1]`; the drafter scatters the next token into
`tokenID_processor.input_ids.gpu[1 : total + 1]` at the anchor rows. `StepIds` gives the tail both views, the element
offset and whether they overlap. Measured on the served K=3 path: **offset 1, overlap True** (every gate run, both
providers), exactly P9's geometry. Rule for WP4B: a tail whose verify writes the drafter's ids (a fused accept +
front) must read every draft id of the step first, in every CTA / wave that uses them (P9 finding 1). The check mode
runs the hybrid first on the untouched ids, so a tail that breaks the rule fails it (mutant `mutant_alias`, 4.2).

### 2.5 What WP4B's tail must still do itself (obligations the hook cannot take)

- `propose_draft_token_ids` still calls `commit_speculative_state(attn_metadata, anchors)` before `tail.propose`; a
  tail that drafted inside `verify` must have drafted from exactly that state (or move the commit into the kernel,
  DESIGN 7.3 meta stages, and keep the host call idempotent).
- `propose` cannot be shadow-checked (the drafter writes its KV and state); its gate is end to end (acceptance and
  texts) plus WP5's Tier G chains.
- K=0: the tail gets the mixed hidden state; owning the final mix needs a `[tokens, hc, hidden]` output buffer in
  ATOM's decode graph (today `[tokens, hidden]`). Not built: P9's K=0 tail is head + argmax + meta, and WP2e's K=0
  trace decides whether the mix is worth moving; the hook refuses nothing here, it simply does not offer it.
- Eager decode steps (no graph) compute the logits even when the tail owns them (one extra head on eager steps only;
  served K=3 / K=0 decode steps are graph replays).

## 3. Gate results (hardware: smci355-ccs-aus-n07-17 / n01-09, q38 container, `rocke` partition)

Trees: P = `dev/ATOM-truemono` before WP4h (stage `atom-truemono-w5-wp4h-pre`, 1,994 files), T = P + the 7 WP4h files
(built on the node from `overlay/`, so another agent's later edits of the tree, P11's `runner.py` / `group_runner.py`
/ `q38_accuracy.py`, are in neither), H = the pinned hybrid (`hb-atom-w5-wp4h` = `atom-rkrepro`). Final bytes: job
68234093 (`gate5`: diff, unit, keys, serve) and 68234712 (`gate7`: diff, transitions); the same gates on earlier
bytes (formatting / test-only differences): 68233497 (`gate3`), 68233789 (`gate4`); 68233050 (`gate2`, partial, see
4.3) and 68234552 (`gate6`, cancelled: its offline arm did not catch ATOM's MTP logprobs refusal and hung).

### 3.1 Tree diff and ATOM CPU tests

| gate | result |
|---|---|
| P -> T | exactly the 7 overlay files (5 changed, 2 added), 0 removed, 0 under `kernels/` |
| ATOM CPU tests (52 files: every `test_qwen4_exp_mono_*`, `test_mono_*`, `test_qwen4_exp_mtp`, and every test that names `model_runner`), P and T in parallel | P 1,458 pass / 184 skip / 8 fail; T 1,515 pass / 184 skip / 8 fail. 0 outcomes changed; 58 new tests, all pass; 1 removed (the intended `[tail]` case). The 8 failures are the same in P and T and pre-existing (`test_gdn_state_relocation` x7, `test_h2d_v4_publication` collection error: both files are byte-identical to `dev/ATOM`'s) |
| unit mutation check (`mut_unit.py`, 22 mutants of tail.py, hook.py, true_mono.py, dispatch.py, model_runner.py) | 22 of 22 killed, unmutated passes (node: gate3, gate4, gate5; local Windows run with a shim: same) |

### 3.2 Build keys with the hook unset (host-only edit: the key gate is decisive, ISA a sanity check)

All 48 layer builds (3 kinds x r1-r4, r2s-r4s, r4g2-r16g4), P vs T, plain env and the hybrid ALL env:

| env | keyed / FlyDSL key / memory keys | disk-cache entries | sources digest | ISA sanity (.text + resources) |
|---|---|---|---|---|
| plain | 48 / 48 identical | equal (48) | equal | identical except 0-2 of the known two-binary builds per run (`delta_ple` r3s, `qsa` r9g3: skill rule 137) |
| ALL | 48 / 48 identical | equal (25) | equal | identical except `qsa` r9g3 in 2 of 4 runs (rule 137) |

The key-only mutant (a comment in `kernels/layout.py`) FAILs the identity gate as it must (every run).

### 3.3 Served functional checks (T, ALL env, one server each, strict unless stated; `serve_gate.py`)

| arm | result |
|---|---|
| K=3, tail pass-through, check 64 | the tail served all 4 grouped widths (`r4s`, `r8g4`, `r12g4`, `r16g4`); 64 / 64 verify steps bitwise equal to the hybrid; graph logits == eager logits on 64 / 64 steps; layer hook on 15 builds; alias offset 1 / overlap; every prompt entered the tail from its prefill; 0 tracebacks |
| K=3, tail owning the logits, check 64 | decode graphs captured without `compute_logits` at all 4 widths; 64 / 64 verify steps bitwise equal; 0 tracebacks |
| K=0, pass-through, check 64 | widths `r1`-`r4`; 64 / 64 sample steps bitwise equal; graph == eager logits 64 / 64 |
| K=0, owning the logits, check 64 | graphs without the head at `r1`-`r4`; 64 / 64 bitwise equal |
| K=3, refuse every width (per-width fallback) | 4 widths refused, logged once each; every decode step on the hybrid (`route.refused` only); served normally |
| must-fail mutant: accepted count moved by one | caught: 16 / 16 checked steps DIFF |
| must-fail mutant: P9's race (front scatter before the accept reads) | caught: DIFF on 5-10 of 16 checked steps per run (the steps with a rejection; a fully accepted round leaves the drafts unchanged, so it cannot differ) |

GSM8K in these small arms (8 prompts) ranged 7-8 / 8 for every non-mutant arm across runs (the mutants: 0 / 2 and 1 / 2).

### 3.4 Transitions (`trans`, final bytes, job 68234712 `gate7`; the served half also in `gate5` / `gate6`)

Served (T, ALL env, K=3, check 32 per width, strict), one server through: 2 greedy prompts -> 2 sampled prompts
(T 0.7, top-p 0.95) -> 4 greedy prompts at CONC 2 -> 2 greedy prompts. Offline (ATOM's `LLMEngine` under the same CLI,
`offline_trans.py`), K=3 and K=0: greedy -> a logprobs request -> ONE batch of a greedy and a sampled request -> greedy.

| arm | result |
|---|---|
| served, tail owns the logits | tail at both widths (`r4s` 357 steps, `r8g4` 144: the CONC 1 <-> 2 width change runs on the tail); 143 non-greedy steps on the hybrid, each after a graph replay without logits, so each got them back from the hook (`hybrid_logits` 143); 64 checked verify steps bitwise equal (both widths); 0 tracebacks; the last greedy client served normally |
| served, pass-through | `r4s` 385 / `r8g4` 131 tail steps; 208 non-greedy steps on the hybrid with the graph's own logits (`hybrid_logits` 0); 64 + 64 checked steps equal (verify, graph-vs-eager logits) |
| offline K=3, owned / pass-through | the logprobs request is refused by ATOM itself before it reaches the runner ("Token logprobs are not supported with speculative decoding"), as at the API; the greedy + sampled batch runs on the hybrid (4 steps), then the greedy request continues ALONE on the tail: `not_greedy -> tail` inside a running request; 64 / 42 checked verify steps equal |
| offline K=0, owned / pass-through | the logprobs requests run on the hybrid (64 steps) and return logprobs for every token; the mixed batch on the hybrid (8 steps), then back to the tail mid-request; 64 checked sample steps equal; owned: `hybrid_logits` 72 (= 64 logprobs + 8 sampled steps) |

TRANS_GATE PASS (30 of 30 checks); SERVE_GATE PASS (37 of 37 checks, gate5). Side observation (not a gate): the same greedy prompt decoded alone and decoded beside a
sampled request gave different texts in all four offline runs: greedy output depends on the batch width (a different
layer build, `r4s` vs `r8g4`); this is the hybrid's behaviour, unchanged by the hook.

### 3.5 Served smoke, paired (`job_e2e.sh`): pass-through serves gsm8k like the hybrid

Hybrid ALL env everywhere; one server per arm-run; per pair a seeded permutation rotated by the pair index; 16 gsm8k
prompts x 2,048 tokens, CONC 1. Arms: H (pinned hybrid), H2 (H again, A-vs-A), T (the copy, hook unset), TP (`tail`,
layer + tail pass-through, strict), TO (`tail`, tail owning the logits, strict).

**K=3** (job 68233494, n07-17 bus 05, 5 pairs; T lost pair 3 to a port-bind failure of its server start, an
infrastructure flake, 4.4):

| arm | ms / round, per-request median vs H: median [range] | acceptance vs H | gsm8k right (H in the same pairs) | texts identical to H's | verdict |
|---|---|---|---|---|---|
| H2 (A-vs-A) | 0.9981 [0.9967-1.0008] | 0.972-1.008 | 79 / 80 (77) | 4 / 80 | control |
| T | 0.9980 [0.9966-1.0010] | 0.988-1.031 | 60 / 64 (63) | 1 / 64 | NEUTRAL |
| TP | 0.9989 [0.9973-1.0013] | 0.974-1.018 | 79 / 80 (77) | 3 / 80 | NEUTRAL |
| TO | 0.9980 [0.9961-1.0004] | 0.982-1.022 | 78 / 80 (77) | 5 / 80 | NEUTRAL |

(The run-mean metric of `e2e_sum.py` gives the same verdicts; its A-vs-A band is wide, 0.923-1.010, because pair 4's
H run was slow for every arm, 0.92-0.93: a neighbour burst on H, rule 163's reason to decide on per-request medians.)

**K=0** (job 68233724, n01-09 bus 75, 4 pairs, arms H H2 TP TO):

| arm | per-request median tok/s vs H: median [range] | run-mean per-request tok/s vs H | gsm8k right (H) | texts identical to H's | verdict |
|---|---|---|---|---|---|
| H2 | 1.0021 [0.9980-1.0030] | 1.0003 [0.9949-1.0045] | 61 / 64 (62) | 3 / 64 | control |
| TP | 0.9995 [0.9986-1.0024] | 1.0011 [0.9983-1.0030] | 63 / 64 (62) | 2 / 64 | NEUTRAL |
| TO | 0.9994 [0.9970-1.0006] | 1.0003 [0.9994-1.0008] | 63 / 64 (62) | 6 / 64 | NEUTRAL |

Reading: with the hook in pass-through (and with the tail owning the logits) the served path is inside the hybrid's
own A-vs-A behaviour on every measured quantity: round time, acceptance, gsm8k, and the share of texts identical to a
hybrid run (served greedy decoding is not token-deterministic run to run, WP0a; H vs H2 shares 3-4 of 64-80 texts).
In-server, the check arms show the tail's outputs bitwise equal to the hybrid's on every checked step. **Timing
caveat:** both e2e jobs ran next to 1-3 busy jobs of ours on the same node (n07-17: `qk-n7F`, `gate3`, short WP1 jobs;
n01-09: P11 jobs, `qk-concF`, `gate4`), so they do not meet the 8.1 neighbour rule; the ratios are a parity smoke, not
a timing claim. The hook costs nothing measurable here by construction (one Python attribute test and, in `tail`
mode, one routing call per step).

## 4. Findings

1. **The pass-through hook is exactly the hybrid**, measured three ways: keys (48 / 48, both envs), in-server
   bitwise per step (verify, sample, graph-vs-eager logits), and served parity inside the A-vs-A band at K=3 and K=0.
2. **P9's alias geometry holds on the served path** (offset 1, overlapping views), and the check mode catches a tail
   that writes the drafter's ids before the accept reads them, on exactly the steps where it matters (rejections).
3. **ATOM's verdict is one int32 `[3, bs]` tensor**, not a tuple. WP4h's first served mutant run (gate2) returned a
   tuple, the output-signature check raised in the model runner, and the API server stayed up with the client
   waiting forever. Fixed: mutants and fakes use the tensor; `wp4h_bench` stops the client when the runner dies
   (rule for every served gate: a dying model runner does not stop the server).
4. **Infrastructure:** one server start failed with "address already in use" on the job's port right after the
   previous arm's server stopped (K=3 pair 3, T). Not WP4h; recorded in the skill (rule 179).
6. **Logprobs and MTP do not mix in ATOM** (API drops `logprobs`; `add_request` refuses them with MTP). DESIGN 7.6's
   "requests that need logprobs route to the hybrid" therefore only bites at K=0 offline today; WP8 (logprobs in the
   kernel) must add the API plumbing and lift ATOM's MTP refusal, or the logprobs cell of the matrix cannot be served.
5. **The graph's logits equal an eager head's** on all 512 checked served steps (K=3 and K=0, gate runs 2-5): the head is not
   a source of the served path's run-to-run divergence for identical hidden states. D0 open item 6 (where the served
   divergence comes from) is NOT closed by this; WP4h's check mode is the tool a Tier G driver can use for it.

## 5. What this hands on

- **WP4B**: a provider at `kernels.gfx950.qwen38_mono.truemono_tail:provider` returning a tail per width (2.2) with
  `owns_logits=True`; gate it with `ATOM_Q38_TRUE_MONO_CHECK` + strict (bitwise verify / sample against the hybrid
  on served inputs) and the two must-fail mutants of `wp4h_providers.py`, then the same paired smoke.
- **D0 M7 (b)** (P9's tail on the hybrid's layer graph as an early quick arm): exactly `tail` mode with the hybrid's
  layers (`ATOM_Q38_TRUE_MONO_LAYER=wp4h_providers:layer_passthrough`) and P9's tail as the provider.
- **WP3** layer kernels enter through the unchanged `layer` provider (WP0a); `tail` mode keeps it.

## Verification (independent verifier, W5 vWP4h, 2026-10-31)

Verdict: **CONFIRMED** (gate met), with two report corrections and one unreported observation (below). Scripts:
`hackathon-prep/qwen38/truemono/vWP4h/` (`vwp4h_providers.py`: the verifier's own providers and mutants; `job_v.sh`;
`v_gate.py`; `sync_v.sh`, stage `w5-vWP4h-h`). Jobs: 68234930 `w5-vWP4h-a` (n07-17, GRES IDX 1, bus 15: diff + unit +
mut_unit; its serve half died on a script bug of mine, `WORK` unset) and 68235074 `w5-vWP4h-b` (n01-09, bus 05:
diff + served arms). Not a timing run; no timing claims checked beyond reading the builder's caveat.

What was read: this report, `tail.py`, every WP4h hunk of the 4 edited ATOM files (diffed against the pre-WP4h stage
`atom-truemono-w5-wp4h-pre`), the builder's gate scripts and the logs of gate2-gate7. The overlay the builder's stage
shipped (and that gate7 ran) has the same 7 sha256s as the current `dev/ATOM-truemono` bytes (LF), and the verifier
stage was built from those same bytes.

Re-run on hardware (T rebuilt on the node from P + overlay):

| check | result |
|---|---|
| tree diff P -> T (builder's step) | exactly the 7 files, 0 under `kernels/` |
| ATOM CPU tests P vs T (builder's step) | P 1,458 pass / 8 fail / 184 skip, T 1,515 / 8 / 184; 58 new all pass; 0 outcomes changed; the same 8 failures in both; 1 intended case gone |
| unit mutation check | 22 / 22 killed, unmutated passes (each mutant's pattern is required to match exactly once) |
| K=3 pass-through, check 64, strict (verifier provider: hybrid tail + a probe on EVERY tail step) | verify 64 / 64 bitwise equal; hook graph-vs-eager logits 64 / 64; probe: graph logits == eager head and eager head deterministic on 704 / 704 steps, all finite; 15 layer builds on the hook; alias offset 1 / overlap; gsm8k 8 / 8 |
| K=0 pass-through, same | sample 64 / 64; logits 64 / 64; probe 1,280 / 1,280 |
| NEW must-fail: K=0 sample, first token moved by one | caught 16 / 16 (the builder had no K=0 hardware mutant) |
| NEW must-fail: K=3 sampled tokens moved, verdict untouched | caught 16 / 16 (the builder's mutants all changed the verdict) |
| builder's alias mutant (P9's race), re-run | caught 9 of 16 (the rest: fully accepted rounds, as the report says) |
| builder's accept mutant + the verifier's probe | caught 64 / 64; probe graph == eager on 224 / 224, all finite |
| NEW: strict + refuse-all | `TrueMonoRefused` raised at graph capture; the server never became ready (the runner died and the engine shut down) |

Corrections to the report:
1. **P11's edits are NOT disjoint from WP4h's files.** Section 1 says P11's edits "touch none of WP4h's files" and the
   summary says the node build excludes other agents' edits. The overlay's `model_runner.py` (sha `b3a22474`, the
   bytes every gate ran) carries P11's `_Q38_TF` hunk (module flag + a 4-line `teacher_forcing().step(...)` call in the
   K=0 branch; `wave1/P11.md` names it). So T = P + WP4h + that hunk. It is inert in every gate (`ATOM_Q38_TF` unset;
   its import of `q38_accuracy`, absent from T, is lazy), so no gate result changes, but "every P -> T difference is
   WP4h's" is false. Side effect for later users: with `ATOM_Q38_TF=1` and `ATOM_Q38_TRUE_MONO=tail`, K=0 steps the
   tail takes skip teacher forcing silently (the tail branch sits before the TF call).
2. **Unreported observation:** gate5's `k3_mut_accept` arm logged one graph-vs-eager logits DIFF
   (`check.logits.DIFF 1`, 15 equal). Finding 5 counts only non-mutant arms, so its "512 checked steps" is literally
   true, but a head mismatch on identical hidden states is exactly what D0 open item 6 asks about. It did NOT reproduce:
   0 mismatches in 224 probed steps of the same mutant configuration and in 1,984 probed pass-through steps
   (bitwise, finite, deterministic eager head). Open item 6 stays open; the single event is unexplained.

Minor notes (no change of verdict): the served-transition checks for a logprobs request and an in-request
`not_greedy -> tail` failed in gate5's first version of `trans_gate.py` and were moved to the offline arms in gate7.
That is justified (the API cannot carry logprobs, and served sampled prompts are separate requests), but the served
path shows only between-request transitions. The key gate is close to trivial here (nothing under `kernels/`
changed); the hook-unset identity rests on the hunk review (the moved sampler / rejection-sampler code is the same ops
in the same order, `q38_tail is None` guards every new branch), the unchanged unit outcomes and the T arm of the e2e
smoke. The paired e2e is a parity smoke under neighbours (8.1 not met), as the report says.
