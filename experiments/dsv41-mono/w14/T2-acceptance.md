# W14 T2: more accepted tokens per step (acceptance length, verify structure), lossless

W14, 2026-11-03/04, revision 2 (the verifier's ten problems, section 1). DeepSeek-V4.1-Flash on MI355X (gfx950),
TP2 no-EP, DSpark, the B0 recipe (bf16 KV, fp8 index plane, `--level 0` + FULL graphs), ATOM `df86c482` + W12's
`atom-wp3-a7` stage (read-only base) + W14 T2's own files. Ratios only. Absolutes and raw logs:
`hackathon-prep/private-perf/dsv41-mono/w14/T2-abs.md` and `w14/raw/`. Harness: `hackathon-prep/dsv41/w14/t2/`.
ATOM code (default-off, knob-off identical to df86c482, section 10): `dev/ATOM-dsv41/atom/models/deepseek_v41/rk/`
`t2.py` (VRAGGED), `t2tree.py` (TREE), `t2depth.py` (DRAFTW), one env-gated block each in
`model_engine/model_runner.py` and one in `models/deepseek_v41/config.py` (DRAFTW).

Names: "ATOM mono" = ATOM's FlyDSL per-layer fused kernels (serve C1-C8 at TP2), "ATOM native" = the regular path
(serves C9+), "rocKE hybrid" = our work inside ATOM. "Graph-target tree" = the tree prototype with ATOM's captured
target graphs and an eager draft (`ATOM_DRAFT_CUDAGRAPH=0`).

PENDING_RESULTS (filled in below when the jobs end): ds1 / xds / dp2 (draft depth knob), tc2b (single-forward tree
cost), xb2 (replay bisect, first divergent stage), x7 (ng7 replicate), xdt / tg3 (device tree builder), id2 (final
knob-off identity).

## 0. Verdict (read first)

1. **Acceptance today** (step 1, unchanged method, re-measured): AL 3.9 of 6 on GSM8K at every CONC, 2.7-2.8 on a
   non-GSM8K mix, 3.6-3.8 on passkey; 35-50 % of the verify rows are wasted.
2. **Extra rows are not free at C1** (step 2). Newly measured: ONE request verifying more rows costs 1.028 / 1.050 /
   1.116 / 1.189 / 1.206 of the 6-row step at 7 / 8 / 9 / 11 / 12 rows (job cw1), cheaper than two requests of 6
   (1.27-1.30). Trace diagnosis (job tr1): the growth is per-row work inside the mono's K2 (x1.37 from 6 to 12 rows),
   not expert bytes (identical-prompt pairs: +0.6 %).
3. **Tree verify: BUILT** (`ATOM_DSV41_RK_TREE=<policy>`), exact (section 4): a tree walk that verifies the tree's
   primary path and re-verifies a sibling path when the target's token at the first rejected depth is a tree sibling;
   CPU-proven equal to a single-forward tree's acceptance (3000/3000), token-exact gate passed (section 5), three
   tree mutants (wrong tree mask, wrong commit path, off-by-one accept) fail it by 3-4 orders of magnitude. Measured
   served AL at the SAME 6 rows: **+3.0 % GSM8K, +7.5 % non-GSM8K** (512-prompt run, C16), +2.4-2.9 % / +5-8 % at
   C1, matching the simulator. The single-forward tree's bound at C1 is **1.03-1.08** at 6-8 rows (measured AL x
   measured cost of the same rows); 12-row trees and forks stay losses after their restructure (0.93-0.98).
   The served prototype itself (re-forwards + a Python tree builder) runs at 0.37-0.53 of ATOM: diagnosed (host tree
   builder 4-12 ms of GPU idle per step, 1.3-1.8 forwards per step) and restructured (device tree builder,
   section 4.6). The single-forward tree (per-node CSA2 storage in the native path and in the mono kernels) is the
   remaining engineering, with that measured bound.
4. **Token-exact gate** (section 5): every output token, teacher-forced through a DSpark-OFF engine, is the
   target's greedy token or a near tie. On this path NO two greedy runs are token-identical (two free DSpark-off
   passes of the same engine: 1-10 % of streams identical), so the gate is per token: 0 violations (gap > 4 nats) in
   the levers' 0.9 M tokens except 4 rare events also found in pure-ATOM arms; mutants 2-29 % of tokens. Root
   cause of the rare events: ATOM's own router non-determinism (replay probe + bisect: 4-10 % of rows change their
   routed expert set at layer 0-9 between two forwards of identical inputs, at bf16 router-score ties).
5. **Ragged verify** (step 4): `ATOM_DSV41_RK_VRAGGED` (C9+) is a measured **tie** over 8 rotated passes on both
   nodes (0.995-1.003, p_twin >= 0.19); restructured after a timeline diagnosis (no ell work on mono-served steps).
   At C4-C8 (mono) the **measured bound is 0.94-1.00**: the mono's step moves 0.04-0.07 % per routed expert
   (identical-prompt duplication cuts the union by a third for -1.3 % of the step), so a mask inside the mono saves
   0.2-1.6 % for an AL cost of 0.4-31 %.
6. **Draft depth** (step 5): diagnosed (the width-8 block's first five drafts lose 5.5 % because positions past the
   trained width are extrapolated) and restructured: clamping the block positions at +5 removes that loss; a
   width-6 clamped block verifies 7 rows at **+7.2 % AL** (probe), C1 bound 1.04 (measured 7-row cost 1.028).
   Built as `ATOM_DSV41_RK_DRAFTW=6` (default clamp); served / exact / accuracy results: section 7 (PENDING ds1, xds).
7. **Served TP2 C1-C16** (both nodes, 8 rotated passes, twin in every pass): WP1 +0-41 % (C1-C16, p <= 0.003);
   WP1 + VRAGGED = WP1 (contrast 0.987-1.000, p >= 0.15); T2 adds nothing measurable to WP1 with the levers that
   can serve today.

## 1. Revision 2: the verifier's problems

| problem | revision 2 | where |
|---|---|---|
| step 3 skipped on a scope argument | tree verify BUILT (exact walk, eager and graph-target), gates, served AL, cost probe, bound, restructures | 4 |
| no wrong-tree-mask mutant; commit / accept mutants only in VRAGGED form | three TREE mutants (`mask`, `commit`, `accept`), each failing the token-exact gate; VRAGGED mutants kept | 4.2, 5 |
| no token-exact end-to-end comparison | teacher-forced token gate (every token of every stream, DSpark-off engine) + direct stream comparison vs DSpark-off greedy; 4 jobs, 0.9 M lever tokens | 5 |
| accuracy not inside the literal twin range | 10 base twins over both nodes and 4 jobs; per-run ranges, run-level and per-question permutation tests, McNemar | 8 |
| passkey re-run evidence missing; OOM unexplained | the pk job's results fetched (`raw/pk_fetch`); 18 more GSM8K-then-passkey servers on both nodes, all 20/20 + 20/20; OOM explained by WP1's memory (device memory per server measured) | 8.2 |
| n07-17 matrix had no twin in pass 1 | that pass's twin server DIED (illegal address in aiter `fused_moe` stage 1, knob-off base); new matrix: twin in every pass, 4 passes per node | 9, 12 |
| twins not tight | 8 passes pooled over both nodes: permutation test against the twin and bootstrap CI | 9 |
| knob-off identity only a manifest diff | traced IR / `.text` of every mono K1 / K2 build (rows 1-48) equal to df86c482 + a must-differ mutant; AST identity of every edited ATOM file with W14's blocks removed | 10 |
| step 4 not armed at C4-C8 on a paper argument | measured bound at C4-C8 (native masking, mono union sensitivity, eager union counts) | 6.2 |
| losses without diagnosis / restructure | depth: positional diagnosis + clamp restructure (section 7); wider trees / forks: trace diagnosis + measured cost curve + gated / short-fork restructures (4.5); served tree prototype: timeline diagnosis + device builder (4.6); VRAGGED C16 cells: timeline budget + kineto diagnosis + lazy-ell restructure (6.3) | 4.5-7 |

## 2. Step 1: today's acceptance, and what is wasted

Unchanged from revision 1 (job w14t2-a1; ATOM's own `/debug/mtp_stats`), confirmed in the exact-gate jobs:

| prompt set, CONC | AL (of 6) | survival j = 1 / 2 / 3 / 4 / 5 | wasted rows per request-step |
|---|---|---|---|
| GSM8K, C1 | 3.68-3.94 | 0.83 / 0.67-0.68 / 0.52-0.53 / 0.39-0.40 / 0.28-0.30 | 2.1-2.3 |
| GSM8K, C16 (native) | 3.88-3.94 | 0.85-0.86 / 0.70-0.71 / 0.56 / 0.44 / 0.33 | 2.1 |
| non-GSM8K mix, C1-C16 | 2.73-3.02 | 0.74 / 0.51 / 0.35 / 0.24-0.25 / 0.17 | 3.0 |
| passkey 8K / 24K | 3.6-3.8 | | 2.2-2.4 |

AL does not depend on CONC beyond noise; 35-38 % (GSM8K) to 50 % (non-GSM8K) of the verify rows produce no token.
The exact per-position probe (91,499 draft calls, the tree logged at every anchor, `t2_sim.py`) reproduces the served
AL (GSM8K 3.871).

## 3. Step 2: the cost curve, and what the extra rows cost

### 3.1 Step GPU time vs verify rows (graph mode, HTL, slower rank, ratio to the C1 6-row step)

| rows of ONE request (C1) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| step | 0.80 | 0.85 | 0.92 | 0.98 | 1 | 1.028 | 1.050 | 1.116 | 1.189 | 1.206 |

Rows 2-5: W12's DLEN (job a1); rows 7-12: job cw1 (a w-draft chain at C1, w = 6..11, forced acceptance so the run is
lossless whatever the drafts; mono on; A-vs-A twin 1.001). Two requests of 6 rows: 1.30 (independent), 1.28
(identical prompts). bs3 1.67, bs4 1.87, bs8 2.74, bs16 (native) 3.6-4.0.

### 3.2 Trace diagnosis (kineto decode windows, job tr1, per verify step)

| comparison | step span | what grows |
|---|---|---|
| C1 -> C2 identical prompts (6 -> 12 rows, same experts) | 1.267 | mono K2 x1.37 (+25 % of the C1 step), K1 +1.7 %; K2 switches to its token-parallel build at 12 rows |
| C2 identical -> C2 independent (same rows, more experts) | 1.006 | nothing beyond noise |
| C8 identical pairs -> C8 independent | 1.011 | K2 +1.1 % |

The cost of extra rows at C1-C8 is per-row work inside the mono's K2 (its GEMVs, MoE row tiles and seams), not
routed-expert bytes. One request's wider verify stays below the token-parallel regime up to 11 rows, which is why
it is cheaper than a second request.

## 4. Step 3: tree verify (BUILT: `ATOM_DSV41_RK_TREE=<policy>`)

### 4.1 Design

- **Tree**: at every draft call the DSpark block's per-position logits are expanded into a beam tree (beam 16 per
  depth, top-4 children per node, each node's Markov bias from its OWN parent token, the confidence head per node;
  the probe's tree, whose top-1 path is ATOM's chain on every one of 91,499 calls). A policy picks <= 5 nodes
  (the 6-row budget): `conf:<N>:<th>` (chain cut where the confidence survival drops below th, then best-first
  extras), `best:<N>`, `gate:<lam>:<N>` (variable width: nodes whose draft path probability >= lam), `chain:<L>`.
- **Tree attention mask, as path re-verification** (the walk): the step verifies the tree's primary path (best
  selected child at each depth) as a chain in the 6 rows; when the target's token at the first rejected depth equals
  another selected child of the last accepted node, the rows are rewritten with that sibling's path (accepted
  prefix + sibling + its best selected descendants), the step's token-derived inputs are rebuilt (Engram n-gram
  embeddings and tentative staging; the step's uncommitted tentative state is dropped first) and the target forward
  runs again on the same positions; at most 6 forwards. The accepted path is the longest root path of the tree whose
  tokens equal the target's greedy tokens: what a single-forward tree with an ancestor mask accepts (CPU unit
  `unit/t2_tree_cpu.py`: 3000 / 3000 random trees, streams and policies equal to `t2_sim.accept`).
- **Exact by construction**: the LAST forward is the committed path's; it writes every store of every accepted
  position (window ring, compressor, PAGE entries, index planes, Engram tentative staging, draft context KV), ATOM's
  rejection sampler verifies its logits (verify length = path length, bonus from the last path row), and the drafter
  reads its hidden states. Every TP rank runs rank 0's paths (one broadcast per walk iteration).
- Runs eager (`--enforce-eager`) or with ATOM's captured target graphs and an eager draft (`ATOM_DRAFT_CUDAGRAPH=0`,
  "graph-target"); the mono serves C1-C8 there unchanged. Not composable with VRAGGED / TAIL / DLEN / RAGGED.

### 4.2 Gates

| gate | result |
|---|---|
| CPU unit (walk vs single-forward tree acceptance) | PASS 3000/3000; walk mutants `nosib` / `wrongsib` / `offby1` caught in 1173 / 1161 / 1325 of 3000 |
| CPU unit (device tree builder vs reference builder, `unit/t2_devtree_cpu.py`) | PASS 6 / 6 cases (structure, tokens, log-probs, chain flags exact; confidence to 1e-6); mutant (no chain child) caught 9 / 20 |
| token-exact gate (section 5), tree arms | 0 violations in 0.57 M tokens except 2 events shared with pure-ATOM behaviour (5.3) |
| must-fail TREE mutants in the token-exact gate | `mask` (a sibling's subtree verified against the previous forward's rows: wrong ancestors) 86 violations / 3,359 tokens; `commit` (the primary path's state written last, the committed path emitted) 57 / 3,600; `accept` (draft j verified against row j + 1: off-by-one) 880 / 3,060; vs <= 1 per 30k in correct arms |

### 4.3 Measured acceptance (served, exact; AL ratio to the chain control of the same job)

| run (job) | GSM8K | non-GSM8K | passkey |
|---|---|---|---|
| `conf:5:0.3`, graph-target, C16, 512 + 40 + 8 prompts (x3) | **1.030** | **1.075** | 1.013 |
| `conf:5:0.3`, eager, C4 (x1) | 1.027 | 1.068 | 1.027 |
| `conf:5:0.3`, graph-target, C1 (x1 / xg2) | 1.024 / 1.029 | 1.078 / 1.052 | 0.98 / 0.97 (8 prompts) |
| `gate:0.1:5`, C16 (x3) / C1 (xg2) | 1.023 / 1.029 | 1.074 / 1.062 | 1.026 / 1.017 |
| `gate:0.2:11` (up to 11 nodes), C1 (xg2) | 1.048 | 1.039 | 1.00 |
| `best:11` (12-row tree), C1 (xg2) | 1.148 | 1.195 | 1.082 |
| `best:5`, eager C4 (x1) | 1.009 | 1.048 | 1.08 |
| simulator, same policies (probe log) | conf:5:0.3 1.032 / gate:0.1:5 1.021 / best:11 1.143 | 1.068 / 1.056 / 1.184 | 1.043 / 1.034 / 1.119 |

The served exact AL reproduces the simulator within about 1 % (GSM8K) and 1-3 % (non-GSM8K). Walk statistics (tree
logs of job xd1): 14 % of request-steps switch to a sibling path, +0.165 accepted tokens per request-step;
1.34 (eager, C4) and 1.78 (graph-target, C16) target forwards per step (a step re-forwards when any of its
requests switches).

### 4.4 Cost of a single-forward tree step and its bound

A single-forward tree verifies its nodes in the same rows a chain would, so at C1-C8 its cost is the chain's cost at
the same row count (3.1: per-row work; identical vs different tokens in the rows: +0.6 %). At C16 (native) the
tree's off-chain tokens can change the routed-expert union: measured by the cost probe `ATOM_DSV41_RK_TREE_COST=1`
(the tree's selected nodes in the 6 rows, one forward, forced acceptance so the run stays lossless; the target
forward's GPU time from a timeline mark right before it): PENDING tc2b (job tc1's step-level numbers are dominated
by the probe's own Python tree builder and are not used).

C1 bound = measured AL ratio (served where measured, simulator otherwise) / measured C1 cost of the rows:

| tree | rows | AL GSM8K / non-GSM8K / passkey | C1 cost | C1 bound |
|---|---|---|---|---|
| `conf:5:0.3` | 6 | 1.030 / 1.075 / 1.013 (served) | 1 | **1.03 / 1.075 / 1.01** |
| `conf:6:0.3` | 7 | 1.069 / 1.102 / 1.086 (sim) | 1.028 | **1.040 / 1.072 / 1.056** |
| `conf:7:0.3` | 8 | 1.099 / 1.133 / 1.097 (sim) | 1.050 | **1.047 / 1.079 / 1.044** |
| `gate:0.15:11` | 6.9 mean | 1.066 / 1.075 / 1.090 (sim) | per step | 1.037 / 1.083 / 1.052 |
| `best:8` | 9 | 1.099 / 1.140 / 1.107 | 1.116 | 0.984 / 1.022 / 0.992 |
| `best:11` | 12 | 1.148 / 1.195 / 1.082 (served) | 1.206 | 0.95 / 0.99 / 0.90 |
| `fork:2` (two request copies) | 12 | 1.110 / 1.116 / 1.090 | >= 1.206 (1.30 measured for 2 requests) | <= 0.92 |

### 4.5 Restructures of the C1 losses (wider trees, forks)

Diagnosis (3.2): the loss of 12-row trees and forks is the per-row work of the mono's K2 (and its token-parallel
regime from 12 rows), measured, not expert bytes. Restructures, each scored with the measured cost (`t2_sim2.py`):

- **one request, 7-8 rows** (instead of 12 or a second request): `conf:6:0.3` / `conf:7:0.3` bound 1.04-1.08 (above
  the 6-row tree); `best:8` at 9 rows is the turn-over (0.98-1.02).
- **gated variable width** (`gate:<lam>:<N>`: rows only where the draft path probability is high): `gate:0.15:11`
  1.04-1.08 at 6.9 rows on average; `gate:0.1:5` 1.02-1.07 at <= 6 rows (served exact AL in 4.3).
- **short forks** (a fork holds only the alternative branch and its chain ancestors, not 6 rows): 0.93-0.95 at C1:
  still a loss (a second request's fixed cost).
- 12-row trees: 0.93-0.99 with the measured single-request cost (0.87-0.92 under revision 1's linear model): a loss.

Status: the 7-8-row trees are the best tree shapes at C1; they need the single-forward tree (4.7).

### 4.6 The served prototype (graph-target walk): diagnosis and restructure

Served vs ATOM mono + DSpark (job tg2, 2 passes, per-request decode tok/s, medians): 0.37 (C1), 0.40, 0.48, 0.53,
0.50 (C16); its chain control with the same eager draft (`treegc`) 0.96-1.04. Timeline diagnosis (job tc1, HTL):
the Python tree builder holds the GPU idle 4-12 ms per draft step (a host loop over requests per depth, one sync
per depth), plus 1.3-1.8 target forwards per step (the walk). Restructure: `ATOM_DSV41_RK_TREE_DEVTREE=1` builds the
same tree on the device (padded [requests, frontier] tensors, one all-gather and one host copy per depth; CPU-proven
identical, 4.2): PENDING xdt (exactness) / tg3 (served). The walk's extra forwards remain by design: a prototype
of the acceptance, not of a tree's cost (4.4).

### 4.7 What the single-forward tree still needs (not built in W14)

Every per-token store of ATOM's V4.1 cache is addressed by POSITION and siblings share positions; an exact
single-forward tree needs per-NODE storage and ancestor addressing in: the window ring (slot = position mod ring) and
its rope, the window index lists, the compressed PAGE entries / compressor ring / index planes at compression ratios
1 and 2, the Engram history per path, the draft context KV, the tree verdict, and the same in W13's mono kernels
(K1 attention / index plans), which serve C1-C8. Its measured bound is 4.4; its cost side at C16 is PENDING tc2b.

## 5. Token-exact gate

### 5.1 Method

`t2_exact.py` + `site_tf/sitecustomize.py` + `t2_exact_cmp.py`, ATOM's offline engine, closed groups (the batch
composition of every step is a function of the prompts). Each arm generates GSM8K (64-512), non-GSM8K (32-40) and
passkey 8K / 24K (8) prompts greedily. One DSpark-OFF engine then (a) generates the same prompts freely (the
reference stream; twice), (b) TEACHER-FORCES every arm stream through decode (each output position: the forced token
is the next input, the hook records the target's argmax, its logit, the forced token's logit). Per token
gap = logit[argmax] - logit[forced] (0: the token IS the target's greedy token). Violation: gap > 4 nats
(pre-registered after the smoke job: the emitted token is < 1.8 % as likely as the greedy one; reference-class arms
reach 1.5-4 nats at near ties). Also reported: tokens with gap > 1 / > 2, identical streams vs the reference, the
common prefix, and whether each first divergence is a near tie or a violation.

### 5.2 Results (jobs x1, xd1, x3, xg2; 0.5-1 M tokens per class)

| class (arms) | tokens | gap > 1 per 10k (arm range) | gap > 2 per 10k (arm range) | violations (gap > 4) | streams identical to the DSpark-off reference |
|---|---|---|---|---|---|
| DSpark-off fresh twin (ref0b) | 30k | 13.3 | 1.7 | 0 | 0 / 104 |
| pure ATOM DSpark, 9 arm-runs (base1, base16, basee4, basegc1, basee4b, basegc16; x3 base16, basegc16; xg2 basegc1) | 475k | 9.0 (4.7-16.5) | 1.3 (0.3-3.0) | **1** (5.6 nats) | 0 |
| TREE, 12 arm-runs (tree4, treeb4, treeg1, treeg16, td4, tdg16, tgc1, tgg1, tgg2, tgb11; x3 treeg16, tgg16) | 566k | 9.5 (5.7-17.9) | 1.2 (0.3-2.6) | **2** (29.5, 41.0) | 0 |
| VRAGGED, 5 arm-runs (vr16, vf2_16, wp1vr16; x3 vr16, wp1vr16) | 324k | 5.9 (2.0-11.4) | 0.6 (0-0.7) | **2** (4.5, 32.5) | 0 |
| replay probe (pure ATOM forward run twice per step: rf4, rfg1, rfg16) | 91k | 12.8 (8.2-15.2) | 1.4 (1.0-2.3) | 2 (5.0, 5.25) | 0 |
| TREE mutants mask / commit / accept | 3.0-3.6k each | | | **86 / 57 / 880** (max 27-39 nats) | 0 |
| VRAGGED mutants mask / accept (fixed:2) | 7.4-8.3k each | | | **245 / 134** (max 34-39 nats) | 0 |

- Direct comparison with DSpark-off greedy: no arm, the DSpark-off twin included, reproduces the reference stream
  token for token (the same engine's second free pass matches 1-10 % of streams); every first divergence of every
  correct arm (95-512 per arm) is a near tie (div_viol 0), every mutant has first divergences at violations.
- The lever classes' violation counts do not differ from the pure-ATOM class (binomial on the event counts given the
  token shares: TREE p = 0.57, VRAGGED p = 0.36); the mutants' rates are 2e-2 to 3e-1 per token, 10^4 x higher.
- **The rare large gaps are prompt-specific and not lever-specific**: the TREE event of x1 (29.5 nats) and the
  WP1 + VRAGGED event of x3 (32.5 nats) are the same context of the same prompt (non-GSM8K #7, position 215-216,
  the same token 80009 where the DSpark-off engine strongly prefers token 20), in two different arms of two
  different jobs. PENDING x7 (that prompt x 64 in pure ATOM DSpark, TREE, WP1 + VRAGGED, DSpark-off).

### 5.3 ATOM's own non-determinism (why no stream is reproducible, and where the rare events come from)

- **Replay probe** (`ATOM_DSV41_RK_TREE=chain:5 _REFWD=1 _CHECK=1`: the step's target forward run a second time on
  identical rows, no tree logic): the argmax of 3.5-3.7 % of rows moves; by more than 4 nats on 2-11 of 55-85k rows
  (eager 1-2e-4, graph 2-9e-5); max logit change 27-65 nats. A device-wide synchronize between the two forwards
  (`_SYNC=1`) does not change it (5-7 of 57k): not a cross-forward race.
- **Bisect** (`_BISECT`, forward hooks per layer, eager): 3.7-4.8 % of rows route to a different expert SET between
  the two forwards already at layer 0 (up to 7.5-19 % at later layers); every row whose final argmax moved by more
  than 4 nats had its first routed-set flip at layer 0-9 with a router-score margin of 0-0.16 (bf16 ties). PENDING xb2
  (attention / gate-logit stage of the first difference).
- So ATOM's V4.1 decode is not run-to-run deterministic at the router (consistent with W12 WP1's finding on prefill:
  the MoE gate's GEMM and the routed `fused_moe` accumulate in a run-dependent order); its greedy output is defined
  only up to near ties of the router. The token-exact gate therefore compares every emitted token with the target's
  own greedy choice in its forced context (5.1), not streams with streams.

## 6. Step 4: ragged (confidence-scheduled) verify

### 6.1 `ATOM_DSV41_RK_VRAGGED=surv:<th>` at C9+ (native): built, tie

Mechanism unchanged (per-request verify width from the DSpark confidence head; dead rows leave the routed-MoE expert
union inside the captured graph; exact remap of the verify). Revision 2: the token-exact gate (5.2: 0 / 1 violations
in 324k tokens, equal to pure ATOM), its two mutants fail it 10^4 x more often, and a restructure (6.3).
Served, 8 rotated passes on both nodes, twin in every pass (section 9): 0.995-1.003 at C1-C16, p_twin 0.19-0.89;
old form (`_ALWAYS=1`) 0.981-1.011; lazy / always 0.993-1.022 (p >= 0.32): ties.

### 6.2 C4-C8 (the mono-served widths): measured bound (job d2b)

| measurement | bs4 (C4) | bs8 (C8) |
|---|---|---|
| native step / mono step (no mask) | 1.177 | 1.010 |
| native + `fixed:2` / `fixed:4` / `surv:0.1` mask, vs native | 0.906 / 0.991 / 1.000 | 0.886 / 0.980 / 0.996 |
| the same, vs the MONO step | 1.067 / 1.166 / 1.177 | 0.894 / 0.990 / 1.006 |
| mono step: identical-prompt pairs / independent (union 40 -> 63 / 63 -> 93 experts per layer) | 0.983 | 0.987 |
| mono step per distinct routed expert (slope) | 0.075 % | 0.043 % |
| union cut by the mask, distinct experts per layer (eager counts, real -> live rows): `surv:0.1` / `fixed:4` / `fixed:2` | 4.4 / 7.0 / 21.9 | 5.1 / 8.9 / 28.6 |
| bound of a mask INSIDE the mono (step saving at that slope; AL from the native arms) | **0.998 / 0.936 / 0.699** | **0.999 / 0.937 / 0.707** |

The mono's step depends on rows, not on the expert union (3.2), so a dead-row mask inside it saves <= 1.6 % for an
AL cost of 0.4-31 %: no rule wins at C4-C8. On the native path a strong mask beats the mono at C8 (0.894 of its step)
only at 0.70 of its AL.

### 6.3 The worse C16 cells: diagnosis and restructure

- Timeline budget of the served C16 cells (job d2b, `t2_budget.py`): 40 % of a GSM8K C16 cell's GPU wall is steps
  with prefill rows (17-21 % in decode-only windows); the twin's per-token prefill wall varies 0.92-1.18 between
  passes, which is the served ratios' noise (the 0.911 cell of revision 1 is inside it). The armed decode steps run
  3-5 % faster with `surv:0.2` (0.974 bs16; 0.967 decode-only).
- Trace (job tr1, C16 window): the MoE family -4.3 % of the step, paid back by VRAGGED's own launches:
  `_dead_rows_kernel` +0.9 % (one per routed layer), elementwise remap / arm ops +0.7 %, sorting +0.3 %; and on every
  draft step at every CONC the ell computation + TP broadcast (~50 us of GPU, also on mono-served steps where it masks
  nothing).
- Restructure (default since revision 2): the ell work and its broadcast are skipped on draft steps whose next
  verify the mono serves (`ATOM_DSV41_RK_VRAGGED_ALWAYS=1` keeps the first form). Served: a tie with the old form and
  with the base (6.1); the remaining launches (per-layer dead-row kernel, remap) are its cost floor.

## 7. Step 5: draft depth

### 7.1 Diagnosis (exact per-position acceptance, forced-AL probe, GSM8K 64 prompts, jobs dp1 / dp2)

| draft block | chain AL at 5 drafts (6 rows) | at 6 / 7 / 8 drafts |
|---|---|---|
| width 5 (checkpoint) | 1 (ref) | - |
| width 6 | 0.996 | 1.051 (7 rows) |
| width 7 | 0.991 | 1.042 / 1.066 |
| width 8 | 0.945 | 0.988 / 1.010 / 1.019 |
| width 8, positions clamped at +5 (R1) | 1.003 | 1.044 / 1.054 / 1.055 |
| width 6, positions clamped at +5 (R1) | 1.031 | **1.072** (7 rows) |

The width-8 block's loss on its first five drafts is the positions past the trained width (RoPE extrapolated in the
bidirectional block): with the positions clamped at +5 the first five drafts accept as the width-5 block's do.
PENDING dp2 (twins of width 5 and width 6 clamped, width 7 clamped).

### 7.2 The restructure as a knob: `ATOM_DSV41_RK_DRAFTW=<w>` (+ `_CLAMP`, default 5)

A w-wide DSpark block with clamped positions; the target verifies w drafts. At C1 a width-6 clamped block's bound is
1.072 / 1.028 = 1.043 (rows 7). ATOM's mono DRAFT declines its 49- and 56-row builds (bs 7-8); the mono decode
serves.

- **Token-exact gate** (job xds; 64-256 GSM8K + 40 non-GSM8K + 8 passkey prompts): `DRAFTW=6` at C1 and C16, `=7` at
  C4: 0 violations in 145k tokens (max gap 3.0 nats), first divergences from DSpark-off greedy all at near ties, as
  the pure-ATOM controls of the same job (0 in 111k).
- **Served AL** (job ds1, ratio to ATOM in the same pass): width 6 clamped +6.0 / +6.6 / +5.0 / +4.3 / +3.0 %
  (C1 / C2 / C4 / C8 / C16), unclamped +6.0 / +7.4 / +5.3 / +4.3 / +5.5 %, width 7 clamped +5.9 / +6.9 / +5.0 / +3.7 /
  +4.4 % (the twin: +1.9 % to -1.0 %); the served AL gain of the clamp is within run-to-run noise at width 6.
- **Served tok/s screen** (job ds1, n07-17, 2 passes): width 6 clamped 1.060 / 1.029 / 1.020 / 1.027 / 0.976, width 6
  unclamped 1.060 / 1.002 / 1.022 / 1.040 / 0.987, width 7 1.004 / 0.933 / 0.998 / 0.997 / 0.971; the twin 1.024 /
  1.001 / 1.008 / 1.016 / 1.022. Candidate win at C1-C8 within a noisy screen; the 8-pass two-node matrix with WP1
  stacking and the accuracy gate: PENDING f4n01 / f4n07 / acc4n01 / acc4n07.

## 8. Accuracy

### 8.1 GSM8K 1319 (3-shot chat, flexible / strict), pooled

10 base twins (jobs f-n01-09, f-n07-17, acc3n01, acc3n07; fresh server each, both nodes). Run-level permutation of
the class mean vs the twins and a per-question sign-flip permutation (`t2_acc2.py`):

| arm (runs) | flexible: runs below the 10-twin range (questions) | p_run / p_question | strict: below range | p_run / p_question |
|---|---|---|---|---|
| twins (10): range flexible [0.9037, 0.9128], strict [0.1160, 0.1296] | - | twin-vs-twin McNemar min p 0.14 | - | 0.11 |
| VRAGGED `surv:0.1` (6) | 0 of 6 | 0.82 / 0.28 | 1 of 6 (3 q) | 0.11 / 0.98 |
| WP1 + VRAGGED (6) | 1 of 6 (2 q) | 0.60 / 0.39 | 0 of 6 | 0.89 / 0.81 |
| WP1 (2) | 0 | 0.44 / 0.34 | 0 (both above the mean) | 0.04 (higher) / 0.58 |
| TREE graph-target (2) | 0 | 0.97 / 0.65 | 0 (one above the range) | 0.20 / 0.16 |
| VRAGGED mutants mask / accept (200 q) | strict 0.30-0.52 vs 0.12 | | | p_question 5e-5, McNemar 1e-7..1e-20 |

With 10 references a same-distribution run falls below their minimum with probability 1/11 per metric; 2 of 28
(arm, metric) runs did (expected about 2.5). No arm differs from the twins (all p >= 0.1; the only p < 0.05 is WP1's
strict, higher).

### 8.2 Passkey and the OOM

Passkey 20 x 8K + 20 x 24K: 20/20 + 20/20 in every one of the 18 servers of jobs acc3n01 / acc3n07 (base x3, VRAGGED
x2, WP1, WP1 + VRAGGED x2, TREE, per node), each right after a full GSM8K run in the same server (the sequence that
OOMed once in revision 1), and in the pk job's three servers (results now fetched: `raw/pk_fetch`). Device memory in
use after the passkey phase (`memq`, per server): base / VRAGGED / TREE 257.1-258.4 GiB, WP1 and WP1 + VRAGGED
258.9-259.5 GiB, against 259.2 GiB = the 0.9 `--gpu-memory-utilization` of 287.98: WP1's prefill graphs (W12) add
about 1.7 GiB and put a server at its cap after a 24K prefill; VRAGGED adds nothing. The revision-1 OOM (a 640 MiB
allocation in Engram's post-WKV during a 24K passkey prefill, 244.7 GiB held by torch) is that headroom, not
reproduced in 8 WP1 servers since.

## 9. Served TP2 C1-C16 vs ATOM mono + DSpark (jobs f3n01, f3n07)

Fresh server per arm-run, 4 rotated passes per node, the A-vs-A twin `base_b` in EVERY pass, per-request decode tok/s
(median per arm-run, ratio to the same pass's base), cells C1:24, C2:48, C4:64, C8:128, C16:128 GSM8K prompts.
Pooled 8 passes; CI95 bootstrap; p = permutation test of the arm's per-pass ratios against the twin's.

| arm | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| twin `base_b` | 0.991 [0.983, 1.020] | 1.004 [0.966, 1.011] | 1.005 [0.975, 1.011] | 0.990 [0.981, 1.000] | 1.002 [0.993, 1.033] |
| VRAGGED `surv:0.1` | 0.996 (p 0.50) | 0.998 (0.89) | 0.995 (0.60) | 1.003 (0.19) | 0.997 (0.74) |
| VRAGGED first form (`_ALWAYS`) | 0.998 | 1.011 | 0.981 | 0.998 | 0.984 |
| WP1 (W12) | 1.000 (0.43) | 1.038 (0.013) | **1.201** (0.003) | **1.336** (0.003) | **1.410** (0.003) |
| WP1 + VRAGGED | 1.009 | 1.027 | **1.202** | **1.338** | **1.408** |
| contrast WP1 + VRAGGED / WP1 | 0.997 (0.62) | 0.987 (0.37) | 1.000 (0.55) | 0.998 (0.15) | 1.000 (0.93) |

Per node (medians of 4 passes): WP1 1.201 / 1.200 (C4), 1.348 / 1.317 (C8), 1.441 / 1.401 (C16) on n01-09 /
n07-17; the twin's per-pass range over both nodes 0.92-1.05. TREE prototype and DRAFTW: sections 4.6 / 7.2.

## 10. Knob-off identity (vs df86c482)

- **Kernels** (jobs id1 for stage v3 and id2 for the final stage v13, GPU-less, every ATOM_DSV41_RK_* unset): every V4.1 mono K1
  (`attn_pre`) and K2 (`layer_post`) build at rows 1, 6, 12, 18, 24, 30, 36, 42, 48 compiled from the W14 stage and
  from pristine df86c482 with `FLYDSL_DUMP_IR=1`: 23 / 23 files bytewise equal per build (MLIR stages, LLVM IR, final
  ISA). Must-differ twin: the same stage with one constant changed (`attn_pre.HC_EPS`) differs in 23 / 23 files of
  every K1 and K2 build, and K1 rows 6 vs 12 differ (the comparator works).
- **Python** (`t2_ast_ident.py`): the W14 stage's `model_runner.py` and `deepseek_v41/config.py` with W14's
  env-gated blocks removed are AST-identical to the base's; every W14 block is
  `if <environ>.get("ATOM_DSV41_RK_<VRAGGED|TREE|DRAFTW>", "") not in ("", "0"):` + an import of a `rk/t2*` module +
  `install(self)` / `return validate(...)`, no else; mutants (an extra statement in a block; a call instead of the
  return) fail. Job id2 ran this check inside the job on the v13 stage: PASS. The stage differs from W12's base in exactly those two files and the three new `rk/t2*.py`.

## 11. Lookback / knobs

| knob | status |
|---|---|
| `ATOM_DSV41_RK_TREE=<policy>` (+ `_DEVTREE`, `_COST`, `_CHECK`, `_LOG`, `_REFWD`, `_SYNC`, `_BISECT`, `_MUTANT`) | **built, exact; served prototype 0.37-0.53** (walk + eager draft); measured single-forward bound 1.03-1.08 at C1 (6-8 rows); kept for the single-forward tree |
| `ATOM_DSV41_RK_VRAGGED=surv:<th>` / `fixed:<L>` / `pad` (+ `_ALWAYS`, `_UNION`, `_CHECK`, `_MUTANT`) | built, exact, **tie** (8 passes, both nodes, with and without WP1); C4-C8 bound 0.94-1.00 |
| `ATOM_DSV41_RK_DRAFTW=<w>` (+ `_CLAMP`) | built; PENDING served / exact (ds1, xds) |
| `t2_sim2.py` policies (gated, short forks, measured cost) | simulator; best shapes recorded in 4.4-4.5 |

## 12. Incidents

- **Austin home quota** (2026-11-04 ~14:30 UTC): writes to `/home_aus/AMD/vanantha` fail with "Disk quota exceeded"
  (`quota` shows none: a server-side limit). W14 T2 moved its stages / harness / results / logs to
  `/scratch/users/vanantha/w14t2` (`t2_job2.sh`, `t2_submit2.sh`, `t2_stage2.sh`, `t2_stage_atom2.sh`); job xb1's
  results were copied from the node's /tmp (`t2_rescue.sh`). Other W12 / W13 / W14 jobs that save to the Austin home
  are affected; nothing was deleted.
- n07-17 went `down*` at 05:45 UTC again (third day) and lost its dsv41 seed; reseeded (job 68265736).
- Revision 1's n07-17 pass-1 twin died with `hipErrorIllegalAddress` in aiter's `fused_moe` FlyDSL stage 1 on the
  knob-off base (GPUs bus 5 / 245, not the bad e5).

## 13. Reproduce

`hackathon-prep/dsv41/w14/t2/`: stage `t2_mkstage.py <base tar> <name>` + `t2_stage_atom2.sh <name>`, harness
`t2_stage2.sh <h>`, submit `t2_submit2.sh <h> <tag> <gpus> <time> <stage> <step> VAR=val ...` (held; release with
`t2_release_q.sh`, the W14 4-GPU cap), steps `t2_arms.sh` (served / accuracy / probes), `t2_xarms.sh` (token-exact
gate), `t2_trace.sh` (kineto windows), `t2_ident.sh` (identity). Analysis: `t2_exact_cmp.py`, `t2_matrix2.py`,
`t2_acc2.py`, `t2_htl_diag.py`, `t2_budget.py`, `t2_fwdtime.py`, `t2_step4.py`, `t2_trcmp.py`, `t2_sim.py`,
`t2_sim2.py`. CPU units: `unit/t2_tree_cpu.py`, `unit/t2_devtree_cpu.py`, `unit/t2_unit_cpu.py`. Jobs (revision 2):
w14t2-xs3, x1, xd1, xd2, x3, xg2, xb1, xb2, x7, xdt (token-exact / diagnosis); d2b, tr1, tc1, tc2b, cw1 (cost,
diagnosis); dp1, dp2, ds1, xds (depth); f3n01, f3n07 (matrix); acc3n01, acc3n07 (accuracy); tg2, tg3 (served tree);
id1, id2 (identity).
