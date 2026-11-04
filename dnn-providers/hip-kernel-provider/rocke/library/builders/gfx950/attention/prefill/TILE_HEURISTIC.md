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
- [7. Reproducing](#7-reproducing)
- [8. Open items](#8-open-items)

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
| combined | 282 | 1.0256 | 1.300 | 0 |

The shipped scorer measures 1.1369 on the combined set, so the rule closes roughly
eleven of the fourteen available points.

By cohort on the combined set: D64 causal 1.0041, D128 causal 1.0109, D64 non-causal
1.0281, D128 non-causal 1.0894.

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

Of the suite's 251 shapes the engine serves 102. The rest are decode (`seqlen_q = 1`,
which no prefill tile divides), head sizes 192 and 256, sliding window, sinks, fp8 KV
and varlen — each declined by the engine rather than ranked badly.

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

## 7. Reproducing

```bash
cd dnn-providers/hip-kernel-provider/rocke
export PYTHONPATH=$PWD/platform/python:$PWD/library

# Host-only: build and compile every (shape, tile) the sweep will time.
# The servable pairs must resolve to exactly the 52 shipped kernels.
python library/builders/gfx950/attention/prefill/tile_sweep_validate.py

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

## 8. Open items

- The rocKE dispatcher's `_base_spec` does not yet mirror this rule. That path is not
  load-bearing for production, so it is a consistency fix rather than a perf change.
- No C++ twin of the rocKE dispatcher exists; that is AICK-2711.
- The engine ranks natively with no trained model, as every engine in this repository
  does. If UHD ranker data is wanted later, the sweep output is the training data.
