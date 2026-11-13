# gfx950 dense attention: tile selection heuristic — case study

How the hipDNN `hipkernel:Gfx950AttentionDense` engine chooses a tile, why the rule
is what it is, and what it does not cover. Every number here is a ratio against a
measured baseline on one gfx950 part, at the configurations described in §5. They
are outcomes at those configurations, not a general claim.

## Contents

- [1. Why the decision moved](#1-why-the-decision-moved)
- [2. The rule](#2-the-rule)
- [3. Outcomes](#3-outcomes)
- [4. Tried and rejected](#4-tried-and-rejected)
- [5. Method and correctness gate](#5-method-and-correctness-gate)
- [6. Caveats and honest losses](#6-caveats-and-honest-losses)
- [7. The trained ranker](#7-the-trained-ranker)
- [8. Reproducing](#8-reproducing)
- [9. Open items](#9-open-items)

## 1. Why the decision moved

The gfx950 dense kernel reads batch, both sequence lengths and both head counts as
kernel arguments. Once the catalog collapsed from 840 shape-baked entries to 52, the
shape was no longer encoded anywhere in the catalog, so nothing chose a tile per
problem: the scorer preferred `256x64` for every shape, which was the right default
while the catalog carried the shape and is not once it does not.

The selection space is small. The 52 kernels partition into eight groups of
(head_size, dtype, causal), with seven tiles at D64 and six at D128. Choosing among
six or seven tiles is the entire decision.

## 2. The rule

In `Gfx950AttentionDenseNative.cpp`, `preferredTile(head_size, causal, seqlen_q)`:

| cohort | block_m 256 from seqlen_q | block_n |
| --- | --- | --- |
| D64 causal | 8192 | 128 |
| D64 non-causal | 1024 | 128 |
| D128 causal | 4096 | 64 |
| D128 non-causal | always | 64 |

`tilePreference` then scores each candidate by closeness to that tile: a bonus per
matching axis with `block_m` weighted above `block_n`, then the historical ascending
`(block_m, block_n)` order as a remainder strictly below one.

The remainder is not decoration. The preferred tile is frequently absent for a given
shape, because `kernelMatches` drops every tile that does not divide both lengths, so
the ranking has to degrade rather than name one winner. It also guarantees two
distinct tiles never score equal, which keeps the descriptor-id tie-break out of the
decision.

Thresholds sit on plateaus, not edges. For D128 causal every value in (2048, 4096]
scores identically, so 4096 is a round number inside a wide band.

Non-causal wants the fat tile sooner because it does the full `seqlen_q x seqlen_kv`
work with no triangular saving, so a given length is a larger problem than the causal
case at the same length.

## 3. Outcomes

Geomean of the rule's pick over the fastest tile per shape; 1.0 would be picking the
fastest everywhere. "Regressions" counts shapes more than 2% slower than the shipped
scorer's pick.

| shape set | shapes | geomean | worst shape | regressions |
| --- | --- | --- | --- | --- |
| LLM suite (servable subset) | 101 | 1.0061 | 1.099 | 0 |
| coverage grid | 192 | 1.0345 | 1.300 | 0 |
| combined (union, 11 shapes in both) | 282 | 1.0256 | 1.300 | 0 |

The shipped scorer measures 1.1369 on the combined set, so the rule closes roughly
eleven of the fourteen available points.

By cohort on the combined set: D64 causal 1.0041, D128 causal 1.0109, D64 non-causal
1.0281, D128 non-causal 1.0894.

### What the change actually selects differently

Tile selection is a pure function of head size, mask and the two sequence lengths, so
it can be evaluated over its whole domain without a device.
`tile_selection_diff.py` does that for three selectors: what `develop` shipped
(256/64 wherever it serves, ascending otherwise), what this change ships, and rocKE's
own `library/dispatch/attention/gfx950_dense.py::_base_spec`, which it calls rather
than restates. Over the 180 servable shapes in a 9-length walk:

| cohort | shapes | selection changed | dispatcher differs |
| --- | --- | --- | --- |
| D64 causal | 9 | 9 | 9 |
| D64 non-causal | 81 | 81 | 81 |
| D128 causal | 9 | 6 | 6 |
| D128 non-causal | 81 | 81 | 81 |
| total | 180 | 177 | 177 |

The three shapes where nothing changes are the same three where all three selectors
agree: D128 causal at 4096, 8192 and 16384, which is the one cohort and length range
whose preferred tile is still 256/64. Everywhere else the old constant was wrong in at
least one axis.

The dispatcher column is the consistency gap in §9, measured rather than asserted: it
returns 256/64 for every shape, because the shape-aware branches in `_base_spec` are
all gated on a persistent body, a sliding window or a segment axis, and the catalog
ships none of those. It is not a second opinion that disagrees, it is the old constant.

`tile_selection_diff.py --check` re-reads the four thresholds out of
`Gfx950AttentionDenseNative.cpp` and fails if the mirror has drifted, so the table
above cannot quietly describe a rule the engine no longer implements.

## 4. Tried and rejected

**Grid fill against CU count.** The lead hypothesis, and wrong. Preferring the
largest tile that still fills the device scored 1.0943 against the shipped 1.1078 on
the suite: it separates the underfilled case cleanly and explains little else. The
CU count is not read by the shipped rule.

**Largest legal tile.** 1.2490, worse than shipping nothing.

**Splitting on dtype.** No effect at all. Fitting per dtype gives exactly the same
geomean as not doing so.

**A single `seqlen_q` threshold everywhere.** This is the instructive one. Fitted on
the suite alone it measured 1.0073, better than the rule that shipped. On the grid,
which it had never seen, it degraded to 1.0571 and regressed 16 shapes. The cause was
a `block_n = 64` preference generalised from D128-causal data, where `block_n = 128`
in fact wins in the other three cohorts, and D64 has nearly double the tile spread so
the mistake cost most where there was least data.

## 5. Method and correctness gate

Every legal `(shape, tile)` pair timed with `tile_sweep.py`, three rounds alternating
over the whole configuration list rather than repeating each config back to back, so
clock or thermal drift appears as spread between rounds instead of favouring whatever
ran while the device was cool. Median across rounds, not mean.

Correctness first: every timed configuration is checked against torch SDPA by the
same `run` the prefill harness uses, and each row records its own error. A row that
fails its check is excluded from the fit. The sweeps behind §3 had zero failures
across 5,574 timed rows.

Shape sources do different jobs. The LLM suite is real model shapes and carries the
geomean claim. The synthetic grid exists for catalog coverage: the suite alone reaches
only 19 of the 52 kernels, because every shape in it this engine can serve is causal.

Of the suite's 251 entries the engine serves 102. The other 149 are decode
(`seqlen_q = 1`, which no prefill tile divides), head sizes 192 and 256, sliding
window, sinks, fp8 KV and varlen, each declined by the engine rather than ranked badly.

Those 102 entries are 101 distinct shapes: Llama-3-8B and MiMo-V2 share
`b1_sq4096_sk4096_hq32_kv8_d128_bf16_causal`, and a shape is swept once however many
models ask for it. 11 of the grid's 192 shapes are also in the suite, so the combined
set is 101 + 192 - 11 = 282.

## 6. Caveats and honest losses

**The rule is fitted on all 282 shapes.** No held-out set remains. The honest claim is
that it describes the measured data at 1.0256 with no regressions, not that it was
validated on unseen shapes. What gives confidence in its shape rather than its exact
constants is §4's single-threshold experiment, which did have a held-out set and
failed it.

**D128 non-causal is the weak cohort at 1.0894.** Using `block_n = 128` there measures
1.0480, four points better, at the cost of two shapes crossing the 2% regression line.
The strict reading of the acceptance criterion was taken. If that criterion is
relaxed, this is the first thing to revisit.

**Batch and GQA ratio carry signal that is not used.** Splitting on them measures
1.0132 and 1.0175. They were left out because the rule already clears its target and
there is no held-out set left to separate a real effect from a fitted one.

**Nothing here covers sliding window, LSE, or the persistent body.** None are in the
catalog, and the engine declines the first two at graph match. They are not tiles the
rule declines to pick; they are not present.

## 7. The trained ranker

The rule above is the engine's native scorer. A trained UHD catalog ranker was also
built for the same decision, through the house pipeline (`hipdnn_corpus_gen` ->
`hipdnn_bench` -> `uhd_gen`), so the two can be compared rather than argued about.

**It ranks the same two knobs.** Of the ten kernel fields the KMD declares, five are
single-valued across the whole 52-kernel catalog: `batch`, `seqlen_q`, `seqlen_kv`,
`ragged` and `sliding_window` are the runtime-shape sentinels the catalog collapse left
behind. Only `block_m`, `block_n`, `causal`, `dtype` and `head_size` vary. The model
therefore chooses a tile within a cohort, exactly as `preferredTile` does.

**The shape set needed a different source.** corpus_gen's search walks around the
operation's declared archetypes, and a tile must divide both sequence lengths, so almost
every proposal is declined and the search saturates. Recorded shapes are passed in
directly with `--model-shapes` instead. The tool's own suggestion of `--kdp-root` is
wrong for this engine: it reads shapes back out of the pack's descriptors, and this
catalog deliberately carries none.

**Two tiers, and the first one was not shippable.** A smoke set of 200 graphs trained a
model whose mean top-1 regret was 0.0205. It passed every Stage V gate, but over its 22
held-out graphs it was faster on 9, within one percent on 6 and slower on 7, with a
median paired time ratio of 0.9957. The mean was carried by two shapes. That is a tie,
and reporting it as a win would have rested a claim on a sample too small to carry one.

**The standard tier settles it.** 2,200 graphs across two sets, 12,870 measurements, 221
held out. Mean top-1 regret falls to 0.0017 with a median of 0.0000 and a p95 of 0.0110.
One of 219 scored problems sits above 5% regret. Top-1 recall is 0.8721 strict and
0.9406 tie-aware, and top-3 recall is 1.0000, so when the model misses it misses by one
position. Declared order measures 0.1943 on the same held-out problems.

| | smoke (22 held out) | standard (219 scored) |
| --- | --- | --- |
| mean top-1 regret | 0.0205 | 0.0017 |
| p95 | 0.0818 | 0.0110 |
| above 5% regret | 3 | 1 |
| top-1 recall (strict) | 0.5455 | 0.8721 |
| top-3 recall | 0.9545 | 1.0000 |

**One model per ranking metric.** The same dataset trains a `tflops` ranker and a `time`
ranker, since the role binds a model per metric. They land within a thousandth of each
other at 0.0017 and 0.0018 mean top-1 regret, both with top-3 recall 1.0000. Declared
order measures 0.1943 and 0.2633 against them respectively.

**On device, against this rule.** Offline regret says how well a model ranks a stored
sweep. Stage V says what the engine actually does, running each held-out graph twice:
once under the build's tree and once under a tree with the candidate installed. Both
metrics pass all four gates (`available`, `read`, `winner_records`, `regret`) over 200
paired graphs with no engine failures.

| | tflops | time |
| --- | --- | --- |
| mean time ratio vs this rule | 0.9601 | 0.9629 |
| faster / within 1% / slower | 103 / 82 / 15 | 96 / 87 / 17 |
| median ratio | 0.9883 | 0.9919 |
| best, worst | 0.679, 1.033 | 0.688, 1.068 |

So the ranker is about 4% faster than the rule of §2 on average, improving roughly half
the graphs, leaving most of the rest indistinguishable, and losing on 15 to 17. Its best
cases are near 0.68, which is where this rule picks badly.

**Read the spread, not the mean.** The smoke model's mean ratio was 0.9828, which looks
like a win, but its 22 graphs split 9 faster, 6 within a percent and 7 slower with a
median of 0.9957, and two shapes carried the mean. That is a tie. The standard model's
median sits on the winning side, which is the difference between the two.

**Why the tier mattered.** The two differ only in how many graphs were measured, so the
smoke model was not a worse idea, it was the same idea fitted on too little. Running the
cheap tier first is what exposed that, and it cost about half an hour of device time to
learn it rather than shipping a tie.

**What the corpus could not reach.** The shape pool caps out near 800 from the LLM suite
and the search together, so a denser aligned grid was generated to give the standard tier
2,000 graphs to cut from. That grid exists for the corpus only and is deliberately not
the coverage grid of §3, which is the one these measurements are reported against.

## 8. Reproducing

```bash
cd dnn-providers/hip-kernel-provider/rocke
export PYTHONPATH=$PWD/platform/python:$PWD/library

# Host-only: build and compile every (shape, tile) the sweep will time.
# The servable pairs must resolve to exactly the 52 shipped kernels.
python library/builders/gfx950/attention/prefill/tile_sweep_validate.py

# Host-only: the §3 differential of develop, this change and the rocKE dispatcher.
# --check alone verifies the mirrored thresholds still match the engine.
python library/builders/gfx950/attention/prefill/tile_selection_diff.py

# On gfx950. --dry-run prints the plan; --limit and --offset make it resumable.
python library/builders/gfx950/attention/prefill/tile_sweep.py \
  --suite <attention_shapes_llm.txt> --grid --rounds 3 \
  --output-jsonl sweep.jsonl

# Fit and report. --source suite restricts to the geomean set.
python library/builders/gfx950/attention/prefill/tile_sweep_fit.py sweep.jsonl
```

`tile_sweep_fit.py` reports the shipped scorer's headroom first. If that is already
near 1.0 on some future catalog, there is nothing for a heuristic to win and that is
the finding.

## 9. Open items

- The rocKE dispatcher's `_base_spec` does not yet mirror this rule: it picks 256/64 on
  177 of the 180 shapes §3's differential walks, agreeing only where the new rule also
  wants 256/64. That path is not load-bearing for production, so it is a consistency fix
  rather than a perf change, and `tile_selection_diff.py` is the check that closes it.
- No C++ twin of the rocKE dispatcher exists. That work is tracked separately and is
  not part of this change.
- The trained rankers of §7 are installed. They ship under
  `heuristics/<ued-id>/sort_kernel_catalog/gfx950/<metric>/`, one per metric, bound in
  the config's `engine.heuristics` beside `trained_catalog_sha256`, which is also the
  retrain trigger: the fingerprint is of the KDP the rankers were trained on, so any
  catalog add, remove or rename invalidates it and the generator's gate fails.
  `tools/catalog_fingerprint.py` prints the value. Stage V ran under
  `--arch-key default`, which decides where a model installs rather than whether it
  works, so its verdict carries to `gfx950`.
- Promotion is additive in the role map but not in behaviour. All three ids appear, the
  native one still under `default`, but on gfx950 the trained models are what rank the
  catalog: the device suite caught two cold cases where a model picks a different tile
  from `preferredTile`, both D128 non-causal and neither in the sweep, so which is
  faster there is unmeasured. Both picks are correct, and the native scorer still ranks
  every other arch. The rule of §2 therefore describes what ships off gfx950, and the
  floor under gfx950 when no model is bound.
- 26 of the 12,870 corpus rows carry no decided correctness verdict, which RFC 0019
  Open Question 19 leaves open per operation. Their timings trained the model without
  having been shown correct.
- `ground_truth.py compare --strict` is attempted and blocked on tooling, not on this
  engine. The sweep behind it does run: 282 shapes end to end through hipDNN, warmup
  plus three timing rounds plus the correctness phase, every gate green, every served
  graph carrying a genuine tolerance match against a pytorch reference. That is a
  correctness result at the hipDNN level, which §3 cannot give because it compares tiles
  at the rocKE level. Three tooling defects stop the comparison itself:
  - `tools/sweep.py` reads `doc["metadata"]`, which dnn-benchmarking dropped at
    `schema_version: 2`. Current HEAD fails silently: the benchmark times all 282 graphs
    and the driver reports every one `missing`. Only `73fff8a` (2026-09-30) works.
  - `configs/templates/attention_sdpa_fwd.graph.json` emits no `attn_scale_value`.
    hipDNN binds an absent scale to 1.0 and torch applies `1/sqrt(d)`, so the two
    compute different functions and every graph fails correctness. Patching the graphs
    afterwards fixes the run but breaks `compare`'s semantic-identity check, correctly.
    A proper fix needs a scale field on `AttentionRequest` and a `${}` placeholder in
    the template: `value_map` cannot patch a field that is already a placeholder
    (`ground_truth.py:339`), and every cell field except `mask_type` is one. The
    template is shared by all three gfx950 attention engines.
  - The owner's times and hipDNN's are not directly comparable. Per-tile ratios between
    the two harnesses span 0.72 to 1.31 on identical shapes, which no overhead model
    explains, so the performance half of the comparison would be unreliable even once
    it runs. This sweep also autotunes rather than using the heuristic, so it measures
    what hipDNN can do, not what §2 picks.
