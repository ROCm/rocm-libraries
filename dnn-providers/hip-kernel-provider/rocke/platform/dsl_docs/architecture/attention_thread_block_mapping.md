# Attention dense — thread-block → work mapping

Which workgroup processes which (query-block, query-head, batch) tuple, and what that
costs. Scope: `library/kernels/{gfx942,gfx950}/attention_dense.py`, causal self-attention,
D64/D128, bf16/fp16.

Per `platform/AGENTS.md` §Compliance this file carries **relative** magnitudes only
(ratios between two code paths); absolute throughput stays out of the repo.

## The two mapping surfaces

The hardware assigns `xcd = linear_wgid % num_xcds`, and the DSL linearizes the grid
x-fastest. So **`grid.x` is what the XCD map reads** — that single fact drives everything
below.

| | non-persistent | persistent |
|---|---|---|
| knob | `default_grid_order` | `persist_decode` |
| grid | `(nqb, Hq, B)` shipped; `(Hq, nqb, B)` for every other order | `(num_persistent,)` |
| XCD map reads | `grid.x` | low bits of the work index `wi` (exact iff `num_persistent % num_xcds == 0`) |
| legality guards | **none** — every order is a bijection for any `Hq = Hkv*gqa` | per-decode (divisibility, causal, alignment, exact CTA count) |
| shape baking (gfx950) | `runtime_shape`: one kernel serves every seqlen | shape is baked |

They cannot co-occur: a non-default `default_grid_order` with `persistent=True` is
rejected. But they are **not different mechanisms** — see the next section.

## The design space: three independent factors

Every mapping on either path is a **mixed-radix decomposition** of a linear index over
four logical digits:

| digit | radix | meaning |
|---|---|---|
| `blk` | `NQB` | query block |
| `bt` | `B` | batch element |
| `hkv` | `Hkv` | kv head |
| `hql` | `gqa` | query head within its kv group (`hq = hkv*gqa + hql`) |

A mapping is then three independent choices:

1. **digit order** — the permutation fastest→slowest: `4! = 24`
2. **query-block traversal** — `asc` / `rev` / `fold`: `3`
3. **assignment policy** — *pinned* (persistent: CTA `c` owns residue class `c` for its
   whole life) or *queued* (one workgroup per item; the next free CU takes the next): `2`

`24 x 3 x 2 = 144` combinations. **Eight have been measured.**

**Both paths can express all 24 orders.** A grid axis may carry several digits — fuse on
the host, split with `%` and `//` in the kernel — which is exactly what the shipped
non-persistent swizzle `hq = gqa*(bx % Hkv) + bx // Hkv` already does. In the limit a 1-D
grid of `W` workgroups realises any order directly. The 3-axis grid is an encoding, not a
constraint. Verified: the shipped `np/hkv_minor` is bit-for-bit the mixed-radix order
`(hkv, hql, blk, bt)` — *the same digit order as the removed persistent `batch_outer`*,
which was measured-negative there. Identical order, opposite verdict: the order alone does
not determine performance.

So persistent vs non-persistent is **factor 3, not factors 1–2**. The two paths differ in
who assigns work, not in what they can express.

### Factor 1 sets locality, and the two fastest digits dominate

> **Measured correction (see "In flight" below).** The stronger reading of this heading —
> that *only* the two fastest digits matter, so the 12 classes are interchangeable inside
> — is **false**. Sweeping all 24 orders shows the within-class spread exceeds the
> measurement noise band in 10/12 classes (non-persistent) and 12/12 (persistent) on
> gfx942, and in the best class the slow-digit order is worth several percent on the
> persistent path. The model below predicts the *ranking between* classes well; it does
> not license collapsing 24 orders to 12.

`xcd = linear_id % num_xcds` reads the low end, so the 24 orders fall into 12 classes
of 2. At `Hq=32, Hkv=8`:

| two fastest | KV duplication B=1 | B=8 | batch slabs/XCD at B=8 | |
|---|---:|---:|---:|---|
| **`bt hkv`** | **1.0x** | **1.0x** | **1** | **the only optimal class** |
| `hkv bt` / `hkv blk` / `hkv hql` | 1.0x | 1.0x | 8 | KV-perfect, scatters batches |
| `bt hql` / `bt blk` | 8.0x | 1.0x | 1 | duplicates KV at low B |
| `blk *` (4 classes) | 8.0x | 8.0x | 8 | worst on every axis |

The XCD index must absorb `bt` first (pinning each XCD to a batch, so no scatter) and
`hkv` second (giving each XCD its own kv head, so no duplication). Every shipped variant
picks one and sacrifices the other: `qb_major` = `bt hql hkv blk` (no scatter, 8x
duplication at B=1); `hkv_minor` = `hkv bt hql blk` (no duplication, scatter grows with B).

### Factor 2 sets load balance, and interacts with factor 3

The right traversal depends on the assignment policy — see H5 and the simulation in
"Measurement notes". Pinned wants the **fold**; queued wants **reverse** (LPT).

## Hypotheses and verdicts

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | The shipped non-persistent order (`qb_major`) is a poor XCD map for causal attention | **Right.** It is the slowest non-persistent order on both arches, by a wide margin, in the large majority of shapes |
| H2 | The gain comes from XCD↔kv-head **locality** (shrinking each XCD's KV working set) | **Wrong.** Decomposing the chain: axis swap `qb_major→hq_major` ≈1.32x; reversal `→hq_major_rev` ≈1.12–1.18x; head swizzle `→hkv_minor_rev` ≈**1.01–1.02x**. The win is grid-axis **load balance**; the swizzle is marginal |
| H3 | The gain tracks `gcd(Hkv, num_xcds)` — more partitioning, more gain | **Wrong.** Flat across `gcd ∈ {1,2,4,8}`: `gcd=1` (no partition at all) gains as much as `gcd=8` |
| H4 | `hkv_minor` requires power-of-2 `num_kv_heads` | **Wrong — over-strict.** The real condition is divisibility (`Hkv % num_xcds == 0` or `num_xcds % Hkv == 0`). pow2 was sufficient, never necessary; it wrongly rejected 24/40/48/56/96, which are correct *and* faster. Now `xcd_partitionable()` |
| H5 | `reverse_qb` (strictly descending query blocks, LPT) beats the causal fold | **Partly — knob since removed.** Positive only for the `qb_major` decode and only at long sequences, where it reaches roughly 1%; clearly and reproducibly negative for the folded decodes. The regime it applies to (persistent `qb_major`, selected only where `hkv_minor` is illegal) is itself well behind the non-persistent axis swap there, so improving it cannot change a dispatch choice |
| H6 | `batch_outer` (batch as slowest work-index field) fixes `hkv_minor`'s B>1 dilution | **Wrong as a general fix, but the deletion was the wrong response — see H14.** Helps when `Hkv >= num_xcds`, regresses monotonically from B=2 when `Hkv < num_xcds`. The B>1 dilution itself is real and unsolved |
| H7 | `hkv_major` (kv-head in the MSB) is a useful decode | **Superseded.** `hkv_minor` dominates it wherever legal; every row where `hkv_major` led was at the smallest seqlen with run-to-run spread far exceeding the margin |
| H8 | The persistent grid is the dominant lever | **Wrong** (an early conclusion, corrected). The non-persistent axis swap is the larger effect, and on gfx942 the reordered non-persistent path beats the persistent path outright |
| H9 | The shipped `auto` policy picks a good variant | **Wrong.** `auto` is never the fastest column on either arch, and is among the slowest in a non-trivial number of configs |
| H10 | The slow variants can simply be deleted | **Partly.** No variant is ever the *sole* legal option in its family — the six non-persistent orders have identical legality in every mode. But `qb_major` is the only persistent decode available wherever `hkv_minor` is illegal, so demote it as a default, keep it as an implementation |
| H11 | `hkv_minor`'s benefit is K/V duplication across the per-XCD L2s, so it should fade once other decodes stop duplicating | **Right, and the crossover is exact.** `qb_major`/`hkv_major` put `bt` in the FASTEST digit, so `xcd = wi % num_xcds` selects the batch element and they reach 1.0x duplication on their own at `B = num_xcds`. `hkv_minor` holds 1.0x at every B, so its *advantage* is `num_xcds / min(B, num_xcds)`: 8x at B=1, 2x at B=4, **none at B>=8** |
| H12 | Past `B = num_xcds` the remaining gap is B-independent | **Wrong.** `hkv_minor` keeps degrading to roughly −9% by B=32. A second, B-scaling term dominates once duplication is spent |
| H13 | That second term is batch-slab scatter (`hkv_minor` spans all B slabs; the others span `B/num_xcds`) | **Partly.** A `bt`-fastest variant was implemented and measured: exactly neutral at B=1 as predicted, then +1–4% rising with B. So scatter is a real contributor — but it does not close the gap, and `hkv_minor` still trails at B>=8. A third term remains unidentified |
| H14 | A variant that is not a safe *default* is not worth keeping (the reasoning that deleted `batch_outer` under H6) | **Wrong, and it deleted a useful variant.** `batch_outer` is the digit order `VGQB` — bit-for-bit the same order as `np/hkv_minor`, which ships. The full sweep puts it in the **best shared persistent covering set on both arches**, paired with `BGVQ`. H6's own finding was the selection rule, not a defect: `BGVQ.fold` wins if and only if `Hkv >= num_xcds` (36/36 on gfx942, 34/34 on gfx950, no exceptions) and `VGQB.fold` takes the rest. A sign flip on a shape parameter disqualifies a *default*; it qualifies a *covering-set member*, because that is exactly what the heuristic keys on. Judge a variant against the policy it is meant to serve |
| H15 | Causal masking breaks one-ACC-per-XCD mappings: blocks of unequal cost desynchronise an XCD's CUs, so it ends up straddling several K/V tensors | **Wrong in effect.** A model of one XCD predicted the property holds exactly without a mask and degrades to ~2 K/V tensors in flight with one. Measured, causal is exactly where Swizzled Head-first *wins* and non-causal where it loses. The desynchronisation is real; whatever it costs is outweighed |
| H16 | A pair-interleaved traversal (`N-1, 0, N-2, 1, …`, consecutive items summing to `N-1`) balances better than `fold` | **Wrong, by analysis — not measured.** Persistent: grid-stride hands a CTA items `c, c+NP, c+2NP, …`, and `num_persistent` (the CU count) is even on both arches, so each CTA sees one parity only — all-expensive or all-cheap blocks, per-CTA imbalance 1.5–1.7x where `fold` gives 1.0. Balanced only for odd `NP`. Non-persistent: its tail is mid-cost where `rev`'s is the cheapest block. The pairing is right; it needs *adjacent* items, and grid-stride never delivers them |

## How the two paths differ in behaviour

- **Different winners per architecture — but not for a mapping reason.** Best
  non-persistent (`hkv_minor_rev`) against best persistent (`hkv_minor`), flat across
  sequence length and GQA ratio, two passes each: gfx942 **1.055 / 1.056** (non-persistent
  ahead), gfx950 **0.949 / 0.951** (persistent ahead). The ~10-point swing is *not* the
  work mapping — see the next section.
- **Guards are asymmetric.** Non-persistent orders need no legality check at all. Persistent
  decodes do, and `gqa_pair`/`gqa_pair_2phase` additionally only resolve when
  `num_persistent` *exactly* equals a derived CTA count — under the shipped default that
  match is accidental rather than intentional.
- **`varlen` is non-persistent-only** on gfx950: no persistent decode is legal there, so the
  non-persistent family as a whole is load-bearing (though any order within it works).
- **Grid fill is not a prerequisite.** Forcing the persistent path on held up even at
  well under one full wave of work items; no work threshold was needed.
- **Small shapes separate nothing.** At roughly two query-block tiles neither reordering
  produces a reproducible difference, and on gfx950 that regime is noise-dominated.
- **`num_persistent` defaults to the CU count**, but the correct value is
  `CUs × occupancy(block_m)`. At a halved `block_m` occupancy is 2, so the default
  silently under-subscribes.

## Why persistent wins on gfx950: `wide_lds_dma`, not the mapping

`Gfx950AttentionDenseSpec.wide_lds_dma` (128-bit buffer→LDS slab DMA plus `iglp_opt`)
is rejected unless `persistent=True`, and `dispatch/attention/gfx950.py` enables it
automatically for aligned causal D128. **So comparing the shipped non-persistent and
persistent arms changes two things at once.** gfx942 has no such field at all, which is
why its non-persistent preference was never confounded.

The restriction is an **implementation boundary, not an algorithmic or hardware one**.
All the wide-DMA code sits inside `_build_attention_dense_persistent`; the loaders read
only `lane`, `wave`, `tile_key0` and the K/V base/stride, and touch neither the
grid-stride loop nor the work-item index. Its real prerequisites — `head_size=128`,
`block_n=64`, slab padding 8/32, contiguous K/V — are all independent of `persistent`.
It is not a flag flip either: the wide path changes the K/V LDS *allocation shape*, so
the read sites are layout-aware too, and it replaces manual scheduling barriers with
`iglp_opt`.

Running the persistent builder with `num_persistent == W` (the total work count) gives
each CTA exactly **one** work item — the dynamic, one-item-per-CTA model of the
non-persistent grid — while keeping wide DMA. That separates the two factors without
porting anything:

| gfx950, D128 | narrow DMA | wide DMA |
|---|---:|---:|
| static assignment (`NP = CUs`) | 0.941 | **1.000** (shipped persistent) |
| dynamic assignment (`NP = W`) | 0.943 | **1.001** |
| real non-persistent grid | *0.959* | not implemented |

geometric mean over 36 configs, relative to the shipped persistent arm.

Reading, in order of size:

1. **DMA width is the whole story: ~+6%.** 1.000 vs 0.941 static, 1.001 vs 0.943 dynamic.
2. **Work assignment is worth nothing.** Static and dynamic are indistinguishable at
   equal DMA width (1.000 vs 1.001; 0.941 vs 0.943). Persistence per se buys nothing
   here — its value on this kernel is entirely the optimization it gates.
3. **The mapping is worth ~+1.7%** in favour of non-persistent (0.959 vs 0.943): the
   3-D grid with `hkv_minor_rev` beats the 1-D persistent grid with `hkv_minor`, the
   same direction gfx942 shows.

So gfx950 does not prefer the persistent *mapping*; it prefers the memory path that only
the persistent *builder* implements. Porting `wide_lds_dma` to the non-persistent body
projects to roughly 0.959 × (1.000 / 0.941) ≈ **1.02** — modestly ahead of the shipped
arm, and it would also reach gfx950 `varlen`, which is non-persistent-only and therefore
cannot use wide DMA today.

## Measurement notes that changed conclusions

- **Median, not mean, across repetitions.** A single bad repetition inside an otherwise
  tight set moved a mean far enough to invent two regressions that did not exist.
- **The `gqa == 1` identity is a free noise probe.** At `gqa == 1` the `hkv_minor` grid map
  degenerates to the identity, so `hkv_minor_rev` and `hq_major_rev` are provably the *same*
  map. Any gap between those two columns is pure measurement error — which is how two
  apparent regressions and one apparent 25%-scale outlier were each identified as artifacts.
- **Check kernel-name distinctness per arm.** The launcher cache is name-keyed and its
  assertion *passes* on a collision, serving a stale binary and reporting ~1.000x — i.e. a
  working knob reads as inert. This kernel has now shipped that bug three times
  (`batch`, `waves_per_eu`, and `bt_hkv_minor` on gfx950 only); the third is what made a
  winning decode read as a 1.5% loss for months. See "Resolved" below.
- **Re-baseline when the default moves.** A harness that hard-codes its reference keeps
  quoting the old one. The same variant measured ~5% against `qb_major` and ~1% against
  the `auto` that replaced it — the first number was not wrong, it was against a decode
  nothing selects.
- **Correctness checking is not the cost.** Measured at 2–3% of benchmark wall time
  (~10% excluding compile), because the SDPA reference is computed once per config and
  shared across arms. Compilation is 70–75%. Disabling the check to go faster trades the
  guard that catches name collisions for nothing.
- **Two independent passes.** Aggregate direction reproduced; per-row winners did not,
  wherever competing variants were within the noise floor.

## Removed levers

Deleted rather than kept as default-off sweep knobs, because nothing set them and a
dormant knob still costs a spec field, a name tag, per-arch decode branches, and
test-matrix entries. Each verdict is preserved as a comment at the site it was removed
from, so it is not rediscovered.

| lever | what it did | why removed |
|---|---|---|
| `reverse_qb` | strictly descending query blocks in place of the causal fold, persistent path | H5 — sub-1% on one decode, negative on the rest, in a regime that is not the dispatch choice anyway |
| `batch_outer` | batch as the slowest work-index field, for `hkv_minor` | H6 — sign flips on `num_kv_heads`, so no safe default exists. ~~**Re-add this one.**~~ SUPERSEDED: it is the digit order `VGQB`, which the persistent `qb_major` reference column later showed to be the WORST of the measured orders despite perfect K/V-per-XCD locality — it pins each CTA to one query block. See "The kv-phase split" below. H14 still stands as a lesson; the specific recommendation does not |

## In flight: the generalized-ordering sweep

> Its conclusions are amended by "Resolved: the shipped decode was never in the sweep"
> below, which identifies what this sweep could not see: `asc` was pruned, so the shipped
> persistent `qb_major` was outside the measured set, and no reference column caught it.

The eight measured points above were hand-written decodes, each chosen before the space
was understood. Four experimental spec fields now make the space itself reachable
(`digit_order`, `qb_traversal`, `qb_phase_rotate`, `kv_split_bt_minor`), plus
`force_baked_shape` to hold `runtime_shape` equal across a generic-vs-named comparison:

```
digit_order: str = ""    # permutation of "QBVG", FASTEST DIGIT FIRST
qb_traversal: str = ""   # asc | rev | fold
```

`Q`=`blk`, `B`=`bt`, `V`=`hkv`, `G`=`hql`. Empty string is OFF and the shipped decode
chain is emitted byte-for-byte, so the byte-identity gate stays green with no re-bless.
The shipped decodes in this notation are `hkv_minor`=`VBGQ`, `hkv_major`=`BGQV`,
`qb_major`=`BGVQ` — note the last two differ *only* in their two slowest digits, which
makes them a built-in test of whether those digits matter.

Scope: **both paths on gfx942, persistent only on gfx950.** The gfx950 non-persistent body
has `runtime_shape`, so `NQB` and `batch` are kernargs and a linear-index decode would
divide by a runtime radix — per work item, worst at B=1. Turning `runtime_shape` off is
not the fix: it also switches that body to a baked k-tile trip count, which would confound
an ordering comparison with unrelated codegen.

**Settled so far** (screen: 26 configs spanning all eight eligible production `Hq/Hkv`
geometries plus two non-pow2-`Hkv` cases; two independent passes agreeing to a p90 of
~1.2%):

- **No legality guard is needed.** A mixed-radix decode is a bijection for *every*
  permutation and radix set — asserted over all 24 orders on 11 shapes, then confirmed in
  the builder on 10 geometries × 2 paths. `xcd_partitionable()` therefore stops being a
  gate and becomes a *predictor* for the heuristic, which is what finally makes
  `gcd(Hkv, num_xcds) = 2` geometries measurable at all.
- **The space collapses per shape.** Radix-1 digits vanish, so the 24 order labels are
  only 6 distinct mappings at `B=1` and 2 for MHA; all 24 are distinct at `B>=2`. Labels
  that collide are the *same binary*, which gives a zero-cost noise-floor control: any gap
  between them is pure measurement error.
- **`asc` is pruned on both paths and both arches.** It is within noise of the per-config
  best on at most 2/26 configs, and — the test that actually matters — dropping it costs
  **0.00% median and 0.00% worst case**, with no config losing more than the noise band.
  It never wins and is never needed as a fallback. This prunes sweep variants, not kernel
  capability: at `NQB <= 1` all three traversals are already the identity.
- **Traversal preference is path-dependent, as H5's mechanism predicts.** Queued wants
  `rev` (LPT); pinned wants `fold`. On gfx950 persistent `fold` is at/near best on every
  config measured.
- **`hkv` is the best fastest digit** on both arches (~22/26 configs); `blk` fastest is
  never best anywhere on gfx950.
- **The `wide_lds_dma` ablation says the mapping ranking is DMA-independent.** Kendall
  tau between the DMA-on and DMA-off order rankings is 0.86–0.94, but that shortfall is
  entirely near-ties: the *winner is identical* with the DMA on and off on every ablation
  config, and **zero** discordant pairs are separated by more than the noise band in both
  states. Tuning in the shipped configuration is therefore safe — established rather than
  assumed.

**Method note.** Regret is normalized against the best variant *of the same path*, never
the global best. On gfx950 the persistent body has `wide_lds_dma` and `iglp_opt(1)` and
the non-persistent body cannot (they are separate builders), so a global normalization
would charge that implementation gap to the mapping and reject the whole non-persistent
path. A constant path offset cancels exactly in log space under per-path normalization.
Nothing is pooled across path, batch class or geometry — the design is unbalanced by
construction, since the order set depends on `B` and `gqa`.

### Results from the full grid

97 shapes x both paths x both arches, two independent passes each, agreeing: every
per-config winner in one pass is within 1.5% of the other pass's winner, on 96/96 configs
and both paths. Per-cell statistic is the median over runs, then the geometric mean over
passes; regret is always against the best variant *of the same path* on that config.

**The shipped non-persistent default is the worst order measured.** `QGVB` (`np
qb_major`) wins on 7/60 configs at B=1 and **0 at every batch size >= 1** on gfx942, and
deleting it entirely costs 0.00%.

**`Q` first is dead, and so is `Q` last.** All six orders with the query block as the
fastest digit are redundant on every arch and path; the six with `Q` second likewise. The
query-block digit wants the *middle* of the order. Only three orders on gfx942
non-persistent are not redundant within the noise band.

**MHA is the easiest case, not a special one.** At `gqa == 1` the `G` digit vanishes, so
the 24 labels are **2 distinct kernels at B=1 and 6 at B>=2** (against 6 and 24 for GQA).
`V` first, then `B`, then `Q` is optimal on *every* MHA config on gfx950 (`VQ.fold`,
mean gap −0.14%), and `Q` first costs **25–36%** — a far larger penalty than anything in
the GQA data, because with two mappings there is nowhere to hide.

### Stage 3: covering sets and what actually generalises

**Score policies held out, not in-sample.** A covering set scored on the configs it was
chosen from reports the per-shape oracle and means nothing. Under leave-one-geometry-out
cross-validation (hold out a whole `Hq/Hkv`, fit on the rest) the in-sample figures do not
survive: a 3-member set with a fitted rule reads −0.4% in-sample and **−1.4% to −1.6%**
held out. A lookup table over all 48 variants is the **worst** policy on both arches
(−2.7% / −3.6%, worse than shipping one fixed variant) — more selection freedom made
things worse, because ten geometries is too few to fit a table that transfers.

**Recommendation per path**, all numbers geomean / worst against a per-shape oracle:

| path | policy | geomean | worst |
|---|---|---:|---:|
| gfx942 non-persistent | `VGQB.rev` alone — **no rule** | −1.16% | −5.78% |
| gfx942 persistent | `VGBQ.fold`, `BGVQ.fold` when `B > 1` | −1.40% | −13.94% |
| gfx950 persistent | same pair | −2.16% | −20.08% |

**Superseded on the non-persistent path — `BVGQ` is better, and it is ONE order for
everything.** The table above ranks orders reachable on the *shipped* grid, which is
`(.., .., batch)` on both arches: batch is permanently the z-axis, the slowest digit, so
only labels ending in `B` were expressible. Lifting that restriction takes one grid
change, and then a single order dominates:

| order (traversal per path) | gfx942/P | gfx942/np | gfx950/P | **worst** |
|---|---:|---:|---:|---:|
| **`BVGQ`** | −1.60% | −1.61% | −1.55% | **−1.61%** |
| `VGBQ` | −1.62% | −2.01% | −1.60% | −2.01% |
| `VGQB` | −2.08% | −1.26% | −2.92% | −2.92% |

`BVGQ` with `rev` on the queued path and `fold` on the pinned one is never worse than
−1.61% on any arch × path, and B=1 is its *strongest* slice. Measured against the old
`qb_major` default: **+28.9% (gfx942) / +33.4% (gfx950)** geomean, never behind it on any
of 96 shapes, zero correctness failures. It ships as the named order **`bt_hkv_minor_rev`**
— grid `(B, Hq, nqb)`.

Two details make it cheap. **Fuse `(hkv, hql)`, not `(bt, hq)`:** `hq = hkv*gqa + hql` is
already a fused quantity the body needs for addressing, and *both* its radices are baked
on both arches, so the unpack is a div/mod by a compile-time constant. Fusing batch onto
an axis instead would divide by `batch`, which gfx950 takes as a kernarg — that would
force `runtime_shape` off and silently switch the body to a baked k-tile trip count.
And **the traversal must stay per-path**: forcing one traversal on both costs 4–6%
worst-case, far more than the order choice, because queued wants LPT and pinned wants the
causal pairing.

**At B=1 no heuristic is needed at all.** One fixed variant is within **0.88% (gfx942) /
0.49% (gfx950)** of the oracle, and every fitted policy improves on that by less than the
noise band. B=1 is also where the label choice is free: `VGQB` / `VGBQ` / `VBGQ` / `BVGQ`
are one kernel there — the batch digit is elided. That freedom is worth spending
deliberately, because **the four diverge sharply at B>=2**, and picking the wrong synonym
costs 2–3 points of B>=2 geomean for nothing.

**The only rule that generalises is `B > 1`.** Fitted predicates over `Hkv`, `gqa`, `NQB`,
`W/CU` and their pairs do not transfer — on gfx950 a fitted rule is *worse* than no rule
at all. `B > 1` is structural, not learned: it is the condition under which the `bt` digit
exists, so there is nothing to overfit. An earlier claim here that `BGVQ.fold` wins *iff*
`Hkv >= num_xcds` was wrong — that condition is necessary, not sufficient, and as a rule
it scores 29/63.

**On gfx942 the non-persistent policy beats the persistent one outright**: +3.69% geomean,
ahead on 72/96 shapes, and ahead at *every* batch size. The distribution is lopsided —
persistent's best case is +3.5%, non-persistent's is +26% — so this is not a close call.
Valid to state only on gfx942, where one body serves both paths.

### `gqa_pair` / `gqa_pair_2phase`: outside the space, and measured separately

**They are not digit orders.** Both add a *phase*: `cta = wi % NP`, `phase = wi / NP`,
with two CTAs covering one query-block pair and the phase selecting complementary blocks
(`gfx950/attention_dense.py`, the `gqa_pair` arms). That is a two-level assignment, so
none of the 144 mixed-radix combinations reaches it and **the sweep says nothing about
them either way**. They are gfx950-only.

**They fire rarely, and by coincidence.** `num_persistent` must *equal* `NQB*Hkv*B`
(`gqa_pair`) or `W/2` (`2phase`). The shipped dispatch sets it to the CU count, so the two
coincide only accidentally. Over the measured shapes: eligible on 68/100 and 80/100 by the
parity conditions, but actually selected on **7/100 and 5/100**; at non-power-of-2 seqlens
**0/76 and 0/76**, because odd `NQB` fails the `nqb % 2 == 0` gate on half that grid. MHA
and odd `gqa` are the other blockers.

**Head to head on exactly those 12 shapes, at matched `num_persistent`** — so the decode
is the only variable, not the occupancy — they are **behind the best available variant on
12 of 12**, by 0.6% to 6.2% (median ≈ 3%). Never a winner: a swept digit order takes 7 of
the 12, a shipped decode the other 5. On one shape (`128/8 S=1024 B=1`) the phase decode
edges every swept order by 0.4% but still loses to `hkv_minor`. Zero correctness failures.

The consequence is concrete: on the shapes where the shipped policy *auto-selects* these
decodes, it is selecting a 1–6% loss. The three shapes asserted in
`library/tests/dispatch/attention/test_gfx950_dense_wiring.py` are all inside that set.

**Untested regime: non-power-of-2 seqlens.** Every measured `NQB` was a power of two
(2, 4, 8, 16, 32, 64), and this biases one arch's numbers specifically. The causal fold
balances a persistent CTA's grid-stride sequence *perfectly* only when the work count is a
multiple of the CTA count; simulation puts the residual per-CTA cost spread at 0% when
`NP | W` but **46–72%** at `NQB` = 12 or 20. In the measured grid `W` divided `NP` on
**70/97** gfx950 configs (`NP` is a power of two there) versus **0/97** on gfx942, whose
persistent path was therefore always measured in the unfavourable regime. So gfx942's
conclusions should be robust to non-pow2 seqlens while **gfx950's persistent margin may
partly be a divisibility artifact**. `seqlen_q % block_m == 0` is enforced, so the cheap
test is `S` a multiple of 256 that is not a power of two (`NQB` = 3, 5, 6, 20); genuinely
arbitrary lengths need `ragged`, which disables `runtime_shape` on gfx950 and must be
measured as its own family. What should *not* move: the `V`-first locality class
(`xcd` depends on `Hkv`, not `NQB`) and the `B > 1` rule.

Still open: a heuristic term for `gcd(Hkv, num_xcds) = 2`, where the only substantive
`Q`-first win lives (`40/10` at low `W/CU`, +2.7–4.3% over the best non-`Q`, reproducible
across passes), and the gfx950 non-persistent path, which this sweep does not cover.

## Resolved: the shipped decode was never in the sweep

Three findings that together changed the persistent default on both architectures.

**`qb_major` is `BGVQ` with the ASCENDING traversal**, not the fold — its decode sets
`qb_v = qb0` directly. `asc` was pruned after the screen, so the shipped decode's exact
configuration was absent from the full sweep, and no persistent `qb_major` reference
column was carried to catch the omission. Every "X beats qb_major" claim before this was
therefore unmeasured. The prune itself was sound for the generic decode path it was
measured on; it did not transfer to a hand-written decode that skips the fold entirely.

**The recorded "bt_hkv_minor loses on gfx950" was a symbol collision.** gfx950 overrode
`_persist_decode_name_part` by RESTATING the shared tag map instead of extending it, so a
decode added to the base emitted no tag, shared `qb_major`'s symbol, and — the launcher
cache being name-keyed — was timed as the same binary twice. Re-measured with distinct
symbols, that decode wins on gfx950 by a larger margin than on gfx942. The override now
defers to `super()` first, and `test_persist_decodes_have_distinct_kernel_names_on_both_arches`
enumerates the decode set so the next shared decode cannot go missing the same way. This
is the third name-collision bug in this kernel (`batch`, `waves_per_eu`, now this); the
earlier guards were per-field, this one is per-decode.

**The persistent default is now B-conditional**, shared by both arches through
`_batch_conditional_auto_decode`:

| batch | decode | digit order |
|---|---|---|
| `< num_xcds` | `bt_hkv_minor` | `BVGQ` |
| `>= num_xcds` | `qb_major_fold` | `BGVQ` + causal fold |

The split is at `num_xcds` because that is where the MECHANISM changes, not where a curve
crossed. Both candidates put `bt` fastest and differ only in the second digit, and with
`wi = bt + B*X` the influence of that second digit on `xcd = wi % num_xcds` is decided
entirely by `gcd(B, num_xcds)`: 3 bits at `B=1`, 1 bit at `B=4`, **zero** once
`num_xcds | B`. So the two are furthest apart at `B=1` — where `BVGQ` gives each chiplet
exactly one kv head's K/V and `BGVQ` gives it four — and place work identically above it.
Past that point they still differ in WHICH kv head each item carries, and the ranking
REVERSES: `BGVQ` then holds fewer distinct kv heads per chiplet and measured ahead on
9/10 gfx942 and 5/10 gfx950 configs at `B >= 16`, by up to ~10%. `qb_major_fold` is the
new named decode for it — `qb_major`'s mapping with the causal fold, which was never
tried because `qb_major` was described as "the only decode with no fold to replace".

Also resolved: the generic ordering now reaches the gfx950 NON-persistent body. The old
objection — that a linear decode there would divide by a runtime radix — had a hole:
`digit_order` already forces `runtime_shape` off, so every radix is baked. The second
half of that objection was real and is handled by `force_baked_shape`, which holds the
flag equal on both arms of a generic-vs-named comparison instead of leaving half the
mapping space unreachable.

## End to end: the shipped path on the real LLM shape list

The sweeps above compare mappings against each other. This one compares the BRANCH
against `develop` through the dispatcher, with no overrides and `force_baked_shape`
OFF — the shipped non-persistent decode deliberately runs with `runtime_shape` ON, so
forcing the shape baked would benchmark a configuration that never ships. Same harness
against two checkouts; two passes; 91 distinct shapes; zero correctness failures. The
dispatched kernel differs from `develop` on 91/91, so this measures the policy change
end to end rather than a subset of it.

| | overall | worst | best | wins |
|---|---|---|---|---|
| gfx942 | **+8.5%** | -2.9% | +34.9% | 86/91 |
| gfx950 | **+6.3%** | -1.7% | +23.6% | 74/91 |

By sequence length — the gain tracks how much work there is to balance, and vanishes
when a single wave covers the grid:

| `Sq` | gfx942 | gfx950 |
|---:|---|---|
| 4096 | +16.4% (16/17) | +11.3% (17/17) |
| 2048 | +10.5% (22/22) | +8.1% (21/22) |
| 8192 | +8.9% (16/17) | +6.7% (16/17) |
| 1024 | +3.6% (14/16) | +5.8% (15/16) |
| 512 | +3.7% (17/18) | **+0.2% (5/18)** |

By geometry:

| `Hq/Hkv` | gfx942 | gfx950 |
|---|---|---|
| 64/8 | +14.9% (15/15) | +6.4% (12/15) |
| 40/8 | +9.0% (12/12) | +8.6% (10/12) |
| 64/4 | +8.9% (8/9) | +4.3% (8/9) |
| 32/32 | +7.3% (10/10) | +7.9% (8/10) |
| 28/4 | +7.1% (11/11) | +7.6% (7/11) |
| 40/40 | +6.8% (8/10) | +6.6% (9/10) |
| 32/8 | +6.7% (14/14) | +4.4% (10/14) |
| 128/8 | +5.8% (8/10) | +4.7% (10/10) |

gfx950 at `Sq == 512` is the one flat region (5/18 wins): that is the `W/CU < 1` regime,
where the grid does not fill the machine once and there is nothing for a reordering to
balance. It is a no-op, not a regression — worst case -1.7%.

**Two scope limits, both structural rather than incidental.**

*Every eligible row in that file is `B == 1`.* So this exercises only the small-batch arm
(`bt_hkv_minor`) of the B-conditional rule. `qb_major_fold` is NEVER selected here, and
gfx942 never takes the persistent path at all on this list, since it requires
`batch >= 16`. The large-batch half of the change rests on the separate sweeps, not on
this number — do not quote this as validating the whole rule.

*91 of 181 rows dispatch.* The remainder are outside this kernel's scope and are skipped
identically by both versions: D=192/256, and `full`-mask rows whose `Sq` is not a
multiple of `block_m`. Not a coverage regression; both checkouts skip the same set.

## The kv-phase split: a negative result worth keeping

At `B > 1` the distinct K/V data is indexed by the PAIR `(bt, hkv)`, so the ideal
schedule gives each XCD one of the `B*Hkv` tensors at a time and sweeps
`B*Hkv/num_xcds` phases. Expressing that needs the fused identity `F` in two positions at
once — low `log2(num_xcds)` bits fastest, high bits slowest — which a permutation of four
digits cannot do. Implemented as a separate FAMILY (`x` + perm of `y`,`G`,`Q`, six
members) plus an orthogonal **phase rotation** axis, `qb_phase_rotate`.

The two split digits, written out, with `M = num_xcds`:

```
    x  =  F % M          the LOW part -- fastest digit, so xcd = wi % M == x
    y  =  F // M         the HIGH part -- the phase index, radix B*Hkv/M
    F  =  x + M*y        reconstructed in the decode, then split back to (bt, hkv)
```

and `F` itself is one of two fusions, which is NOT a cosmetic choice:

| fusion | `F` | `(bt, hkv)` recovered as | collapses to a permutation when |
|---|---|---|---|
| V-minor (default) | `bt*Hkv + hkv` | `bt = F // Hkv`, `hkv = F % Hkv` | `Hkv == num_xcds` |
| B-minor | `hkv*B + bt` | `bt = F % B`, `hkv = F // B` | `B == num_xcds` |

The collapse happens exactly when the fusion lines up with a digit boundary. At
`Hkv == M` the V-minor split has `x == hkv` and `y == bt` identically, so it IS the
digit pair in those slots; at `B == M` the B-minor split likewise has `x == bt`,
`y == hkv`. Note `B % M == 0` is NOT sufficient: at `B = 16` or `32` the batch index is
*split* across both halves (`x = bt % M`, `y = hkv*(B/M) + bt // M`), which is a new
mapping.

`x` adjacent to `y` is always degenerate whichever fusion is used -- adjacent halves
just reassemble `F` contiguously, which is the `(V,B)` or `(B,V)` digit pair. That leaves
four non-degenerate members of the six.

So the common prod geometries, all `Hkv == 8 == num_xcds`, are reachable only with the
B-minor fusion; V-minor there re-measures permutations the sweep already covered
(`xGQy == VGQB`, `xQGy == VQGB`, `xGyQ == VGBQ`, `xQyG == VQBG`).

Equal mapping is not equal kernel. The split decode reconstructs `F` and divides it
back out, which the permutation never does -- measured at **+12 integer ops** in the
emitted body. On the persistent path that is paid once per work item under grid-stride,
so a V-minor split at `Hkv == 8` is the same MAPPING as its permutation twin but a
strictly more expensive kernel. Compare them by measurement, not by the mapping proof.

**The locality premise did not hold.** Ranked by K/V tensors per XCD, the best order
(`VGQB`, a perfect 1.00) measured WORST, and the shipped-class `BVGQ` at 8–32× poorer
locality won. Per-CTA causal balance is a hard gate; locality only separates variants
once balance is equal. The phase rotation IS the balance fix and behaves exactly as
predicted — it swung one variant from about -24% to +6% on gfx950 persistent — but it
rescues bad variants to parity, not past it, and its sign is arch-dependent.

Measured against the decode the dispatcher actually selects, on `Hkv == 8`:

| path | best split variant | gfx942 | gfx950 |
|---|---|---|---|
| persistent | `xGyQ.fold` B-minor | **+1.3%**, worst +0.2%, 9/9 | +0.4%, worst -0.5%, 6/9 |
| non-persistent | `xGyQ.rev` B-minor | +0.5%, worst -0.8%, 4/9 | -0.5%, worst -1.9%, 1/9 |

Only gfx942 persistent clears that arch's noise floor. It is also the narrowest slice:
gfx942 takes the persistent path only at `batch >= 16`. The family stays experimental —
kept, not deleted, because the rotation result explains WHY the high-locality orders lose,
which is a fact about the space rather than about these variants.

A methodology note that follows from it: quoting a variant against `qb_major` once the
auto has moved overstates it several-fold. Re-baseline against `auto`, not against
whatever the harness hard-coded.

## Further findings from the generalized sweeps

Results that are not a recommendation on their own but constrain any future one.
All figures are relative to `auto` on the same path unless stated, two passes, both
architectures.

### Deep persistent sweep: all 72 variants

17 configurations (a batch sweep at 32/8, long sequences to 64K, non-power-of-2
sequence lengths, and one each of `gqa = 8`, `gqa = 16`, `gcd(Hkv, 8) = 4`, `= 2`,
MHA) × every permutation and every non-degenerate split variant, both traversals,
with a harness assertion that no knob other than the ordering fields differs between
arms.

- **Best and most predictable: `xGyQ.fold` with the B-minor fusion** — +0.5%
  (gfx942) / −0.1% (gfx950) geomean, the only variant ahead of `auto` on gfx942.
  Its real distinction is stability: its spread across conditions (batch class,
  sequence length, geometry) is **3.0 / 1.1 pp**, against **4–8.5 pp** for every
  permutation. `VBGQ.fold` and `VGBQ.fold` are the least predictable.
- **`B = 32` is where the headroom is.** More variants beat `auto` there than in any
  other condition (26 on gfx942), by up to ~9% on gfx950 (`BGQV.fold`). The
  shipped rule's `B ≥ num_xcds` arm is therefore the one most worth revisiting.
- **Long sequences flatten everything.** At `S ≥ 16K` the best variant is within
  ±0.5% of `auto`: once each work item is long, ordering stops mattering.
- **Non-power-of-2 sequence lengths need no special case** — they behave like their
  power-of-2 neighbours.
- **All twelve `Q`-fastest orders lose badly**, −22% to −31% geomean, 0 wins. The
  next subsection shows why no traversal can fix them.

### Why `Q`-fastest orders cannot be rescued by any traversal

With `Q` as the fastest digit, `wi = blk + NQB·(rest)`, so every run of `NQB`
consecutive work items contains the block digit `0 … NQB-1` exactly once. A
traversal is a bijection on that range, so **every run still contains exactly one
block of each cost**, whatever the traversal. It can only permute costs *within* a
run, and once many runs are in flight that changes nothing. In a list-scheduling
simulation the makespan of every `Q`-fastest order is identical under `asc`, `rev`,
`fold` and pair-interleave, at every shape tried. This is exact, not a heuristic:
the traversal factor has leverage only when `Q` is slow enough to shape the cost
profile *across* the dispatch sequence. It is also why the expensive blocks recur
up to the very end of a `Q`-fastest dispatch, which is the −22…−31%.

### `BVGQ` vs `VBGQ` on the non-persistent path

The two differ only in which of `bt` / `hkv` is fastest, so at `B = 1` they are the
same kernel — and they measure the same. Above that, `BVGQ.rev` holds flat (within
~0.8% of `auto` at every batch size) while `VBGQ.rev` degrades steadily, to a gap of
**~14 pp at `B = 32`** on gfx942 (~5 pp on gfx950). `VBGQ` walks the kv head fastest,
so consecutive CTAs read *adjacent* K/V addresses — the better locality — and it is
the worse order. That is the third time in this document locality has predicted the
wrong ranking.

### `named_xgyq`: the kv-phase split without the generic path

`named_xgyq[_rev]` reproduces `xGyQ` with the V-minor fusion **at every shape**, as a
3-D grid — `x` fuses `(F % M, hql)` onto one axis, `y` is the phase `F // M`, `z` the
query block — whose only divisors are the constants `M`, `gqa`, `Hkv`. The batch
extent never appears, so `runtime_shape` stays on, which the generic `digit_order`
path cannot do.

- Against `BVGQ.rev` (the shipped class) it **ties on gfx950 and is ~1% behind on
  gfx942**, winning only at the batch extremes (`B = 1`, `B = 32`) and `Hkv = 4`.
  It is a correct and cheap handle on the split family, not a faster order.
- **Without `rev` it is −8% (gfx942) / −4% (gfx950).** On this path the traversal
  matters more than the mapping.
- An earlier version of this named order was the permutation `VGBQ`, which equals the
  split only when `Hkv == num_xcds`. Elsewhere it was measured **up to ~5% slower**
  than the split, with the gap tracking `gcd(Hkv, num_xcds)`: ~0 at `Hkv = 8`,
  −0.2% at `Hkv = 4`, −1.4 to −2.9% at `Hkv = 2` / `40`, −4.2% at `Hkv = 10`. That is
  the cleanest evidence that the split is a genuinely different mapping — and a
  better one — when `Hkv` does not divide `num_xcds`.

### `runtime_shape` is nearly free

The same mapping measured baked and with `runtime_shape` differs by **~0.5% on
gfx942 and ~0 on gfx950**; on gfx942 causal shapes the difference shrinks as `S`
grows. That is
the case for keeping `runtime_shape` on (one binary per shape family): it costs
almost nothing at run time. It also means any comparison that puts a baked arm next
to a runtime-shape arm of the *same* mapping will see them trade "wins" on noise.

### The XCD-fill gate

Any scheme that confines one K/V tensor's work to one XCD needs enough of that work
to occupy the XCD: `gqa × NQB` work items per `(bt, hkv)` against **38 CUs per XCD
on gfx942 and 32 on gfx950**.

| shape | `gqa × NQB` | gfx942 (38) | gfx950 (32) |
|---|---|---|---|
| 32/8, S=2048 | 32 | **under-fills (0.8x)** | 1.0x |
| 32/8, S=8192 | 128 | 3.4x | 4.0x |
| 128/8, S=2048 | 128 | 3.4x | 4.0x |
| MHA 32/32, S=2048 | 8 | **0.2x** | **0.2x** |

The mainstream shape under-fills a gfx942 XCD, and MHA never fills one. That is a
plausible reason gfx942 responded worse than gfx950 to every per-XCD confinement
measured here. It is also why Swizzled Head-first, which confines by *query head*
rather than by K/V tensor, is the scheme that works for MHA. A second gate is L2
capacity (see "Where a KV-direction reversal could pay off"): whole-tensor
residency in one XCD's L2 is only possible up to `S ≈ 8K`.

## Prior art: what other libraries and papers do

A survey, read from source rather than documentation wherever source was
available, so each row states what the code does. The versions read are pinned:
AITER `21ae719`, FlyDSL `v0.2.3` (`90a2427`), AOTriton `0.14b` (`b5e8cfb`), and
FlashAttention `main` (`e9cf2c1`). Mappings are restated in this document's digit
notation for `Hkv == num_xcds`, where most of them coincide with one of the 24
permutations.

### Libraries

| library / kernel | assignment | digit order | XCD-aware | query-block traversal |
|---|---|---|---|---|
| AITER `unified_attention` (Triton) | one CTA per item, `grid=(Hkv, q_blocks)` | `V` fastest, then `Q`, `B`; the GQA group is packed **inside the tile** | no | ascending |
| AITER `mha.py`, `flash_attn_triton_amd` | one CTA per item | `VGQB` via `remap_xcd(head, Hq)` | yes | ascending |
| AOTriton `attn_fwd`, **causal** | **persistent, dynamic**: tiles claimed from an atomic counter, two workgroups per CU | `QGVB` | no | ascending |
| AOTriton `attn_fwd`, non-causal | one CTA per item, `grid=(Hq, nqb, B)` | `VGQB` via `remap_xcd(head, Hq)` | yes | ascending |
| FlyDSL (generic and gfx950 paths) | one CTA per item | `GVQB` (`hq_major`) | no | ascending |
| HipKittens attention forward | one CTA per item, `head = (bx % Hkv)*G + bx / Hkv` | `VGQB` | head swizzle only | ascending |
| **this kernel** | non-persistent: one CTA per item; persistent: static grid-stride | non-persistent `BVGQ`; persistent B-conditional | through the digit order | **`rev`** (non-persistent), **`fold`** (persistent) |

Three observations follow.

**`VGQB` is the de-facto AMD convention.** AITER's FlashAttention paths, AOTriton's
non-causal path and HipKittens all reach it, through the same idiom: `remap_xcd`
applied to the head index, so each XCD owns a contiguous band of heads. It is this
kernel's `hkv_minor` non-persistent order, and what the NUMA paper below calls
"Swizzled Block-first".

**No other AMD library reverses the query block for causal attention.** Every
traversal in the table is ascending. AOTriton handles causal imbalance with a
*dynamic* work queue instead of by ordering; the `rev` / `fold` traversals here
have no counterpart elsewhere on this hardware.

**Two ideas this document has not tested.**
- *GQA packing.* AITER `unified_attention` (and FlashAttention-3's `PackGQA`) puts
  all `gqa` query heads of one KV head into a single tile, so a K/V tile is reused
  inside one CTA by construction. That removes the reuse from the cache's hands
  entirely, which no ordering can do. It changes the tile shape, not the mapping.
- *Dynamic persistent scheduling.* An atomic work queue is a third assignment
  policy, between the static grid-stride and the hardware dispatcher measured here.
  Combined with a longest-first order it would give both the tail behaviour of
  `rev` and the balance of a queue.

### Papers

- **FlashAttention-3 / -4 tile scheduler** — the closest external analogue to this
  work. Read in `hopper/tile_scheduler.hpp`: batch outermost; heads split into
  **sections sized to fit L2** (the section is the number of KV heads whose K/V fits
  the L2 budget, rounded to a power of two, times `qhead_per_khead`); heads inside a
  section fastest; the query block **reversed** (longest-processing-time-first);
  tiles claimed dynamically. In this notation that is `G, V_lo, Q(rev), V_hi, B` —
  the same fast-half / slow-half cut as the kv-phase split above, but with the cut
  placed by **L2 capacity** rather than by XCD count. The FA4 paper attributes its
  larger causal gains to the LPT order, and ablates it against naive ordering.
  [scheduler source](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/tile_scheduler.hpp),
  [FA4, arXiv 2603.05451](https://arxiv.org/html/2603.05451v1)
- **Swizzled Head-first** (arXiv 2511.02132) — confine each "Attention Compute
  Cluster" (the work sharing one K/V tensor) to one XCD. Reproduced exactly here as
  the named order `swz_head_first`; for `Hkv == num_xcds` it reduces to the
  permutation `VQGB`, so it is genuinely new only for MHA. The paper does not state
  whether its benchmarks are causal. Results are in "Swizzled Head-first,
  reproduced" below. [arXiv 2511.02132](https://www.alphaxiv.org/overview/2511.02132)
- **Sawtooth wavefront reordering** (arXiv 2601.16032) — a new axis relative to
  everything above: not *which* work item runs where, but the **direction of the KV
  loop inside a work item**, alternated between a CTA's consecutive grid-stride
  items so the tail it just loaded is reused first. It removes only *capacity*
  misses, so it cannot help while the swept K/V fits the cache serving it. See the
  note below on where that threshold sits on this hardware.
  [arXiv 2601.16032](https://arxiv.org/abs/2601.16032)
- **HipKittens** (arXiv 2511.08083) — on an 8-XCD part, **L2 and the shared LLC
  trade off**: maximising per-XCD L2 hits makes the XCDs fetch disjoint data, which
  duplicates traffic at the LLC. Its chiplet-aware scheduling targets GEMM; its
  attention kernels use a plain head swizzle. This is a candidate explanation for
  why the high-locality orders measured here kept losing, and is untested.
  [arXiv 2511.08083](https://arxiv.org/html/2511.08083v1)
- **Load balance by splitting the KV dimension** — Stream-K and its attention
  descendants LeanAttention and FlashInfer (deterministic, host-planned), plus
  POD-Attention, which co-schedules prefill with decode. These are decode-oriented;
  for prefill the equivalent lever is the tile scheduler.
  [Stream-K tutorial](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/),
  [LeanAttention](https://arxiv.org/abs/2405.10480),
  [FlashInfer](https://arxiv.org/abs/2501.01005),
  [POD-Attention](https://arxiv.org/abs/2410.18038)
- **GEMM ancestors** — Triton's `GROUP_SIZE_M` grouped ordering and CUTLASS's
  threadblock swizzle / raster order; `remap_xcd` above is the chiplet-era
  descendant of both.

### Where a KV-direction reversal could pay off

Sawtooth needs the swept K/V to exceed the cache that serves it. K plus V for one
`(bt, hkv)` is `4·S·D` bytes at 16-bit precision, i.e. `512·S` bytes at `D = 128`.
There are two levels to exceed:

- **per-XCD L2 (4 MB)**: exceeded above `S ≈ 8K` per KV head. Below that a cyclic
  sweep is already all hits after its first pass, so reversal buys nothing — which
  covers every row of the production shape list. A null result there says nothing
  about the idea.
- **shared LLC**: the L2 capacity misses of the previous regime land here, so between
  roughly `S = 16K` and `32K` a reversal only turns LLC hits into L2 hits, a much
  smaller saving than on a GPU whose L2 is its last cache level. The regime the paper
  measured — misses reaching DRAM — starts only once **all** the K/V in flight exceeds
  the LLC, which grows with `B·Hkv` and is reached near `S = 64K` at `B = 1`,
  `Hkv = 8`. The sweeps in this document barely sample it.

It also needs the **persistent** path, where a CTA runs consecutive items, and those
items must share a `(bt, hkv)`. That holds for `BVGQ` at `B = 1` and less as batch
grows. The recommended first step is diagnostic rather than a kernel change: count L2
and LLC misses per tile against the compulsory count while sweeping `S` from `8K`
upward. If misses do not outgrow the compulsory count, there is nothing for a
reversal to remove.

## Swizzled Head-first, reproduced

`swz_head_first` implements Figure 11 of arXiv 2511.02132 exactly, as a 3-D grid:

```
grid = (M·nqb, Hq/M, B)          M = num_xcds
hq   = (bx % M)·(Hq/M) + by      qb = bx // M      bt = bz
```

The only divisor is the constant `M`, so `runtime_shape` stays on. It was checked
against the published formula work item by work item on every shape of the
experiment grid below, on both architectures, with no mismatch. Two notes on the
source: Figure 11 is internally inconsistent about batch (one line makes it the
fastest digit, another the slowest); the slowest reading is the only one compatible
with the paper's own intent and is what is implemented. And the paper specifies no
query-block traversal, so `swz_head_first_rev` adds this kernel's `rev` on top.

**Relation to existing orders.** For `Hkv == num_xcds` it *is* the permutation
`VQGB`, and also the kv-phase split `xQGy` with the V-minor fusion. It is new only
where those coincidences fail, i.e. MHA: there it cuts the **query-head index by
contiguous band**, where the split family cuts the fused `(batch, kv-head)` identity
by residue, and the two agree on only a small fraction of work items.

**Measured**, non-persistent path forced (the paper's kernel is one CTA per work
item), 70 shapes per mask — MHA with 8 to 128 heads and GQA with `Hkv = 8`,
`S ∈ {8K, 32K, 64K}`, `B ∈ {1, 2, 4, 8}`, `D = 128` — two passes, no correctness
failures. Relative to `auto`:

| | gfx942 | gfx950 |
|---|---|---|
| causal, `swz_head_first` | −5.7% geomean | −1.1% |
| causal, **`swz_head_first_rev`** | **+0.8%** (33/70 wins) | **+3.4%** (45/70 wins) |
| causal, `swz_head_first_rev`, best cell | +15% | +21% |
| non-causal, either variant | −2.7% | −2.3% |

**Effect of `_rev`.** It is worth **+6.9% (gfx942) and +4.5% (gfx950)** over the
paper's own ordering on causal cells, and roughly halves the worst case. On
non-causal cells the two variants are the same kernel: `rev` is gated on `causal`
(it is a longest-first heuristic for triangular cost and has nothing to reorder
when every block costs the same), and they measure within 0.6% of each other,
i.e. noise. A comparison against the paper's mapping *without* `rev` would
attribute most of the gap to the mapping when it belongs to the traversal.

**Where it wins.** The gain is monotone in the number of Attention Compute Clusters,
`B·Hq` for MHA, identically ordered on both architectures: negative at 8–16 ACCs,
around zero at 32–64, and rising steadily to the best cells at 256–512. For MHA
causal, selecting it when `B·Hq ≥ 128` improves the mean over all MHA causal cells
by +2.4% (gfx942) / +3.8% (gfx950), and it is the lowest threshold at which **no**
affected cell regresses on either architecture. For GQA with `Hkv = 8` the sign
differs between the architectures (−2.2% / +0.8%): there every XCD already holds
exactly one kv-head, the ACC count does not grow with `Hq`, and there is nothing to
spread.

**Pros.**
- Real, large wins in its regime: causal MHA with many ACCs.
- Cheap: three integer ops per CTA, a constant divisor, `runtime_shape` preserved.
- A clean selection rule, `MHA && causal && B·Hq ≥ 128`, with one constant that
  holds on both architectures.

**Cons.**
- Needs `Hq % num_xcds == 0`, so some production geometries (`Hq = 28`) cannot use it.
- Loses on non-causal and on GQA.
- **Does nothing for the production shape list**, which is all `B = 1` and
  `S ≤ 8K`. There `B·Hq` is 28–128, the rule never fires, and the nearest measured
  cells put it 2–4% behind `auto`. It is a serving-shape optimisation — large
  batch, many heads, long context — not a win for the shapes shipped against.
- Not a like-for-like reproduction of the paper's numbers. `BLOCK_M` is 256 here
  against their 128, and it sets `blocks_per_head`, a first-order term of their
  mapping. Their headline configuration (128 heads at 128K) also exceeds this
  kernel's 32-bit extents at every batch size, so it was measured up to 64K instead.

It is opt-in; `auto` does not select it. Still open: whether the nearest split order
(`xQGy`, V-minor) captures the same MHA benefit, which would separate "confine an
ACC to an XCD" from "cut the head index by contiguous band" specifically. That
comparison has not yet been measured on both passes.

### On the persistent path: port as-is loses, a pair bundle fixes it

The same mapping is available as a persistent work-index decode:
`persist_decode="swz_head_first"` (`_rev`, `_fold`). It is explicit-only, never
selected by `auto`. It has the same digits, fastest first `a(M), blk(NQB),
c(Hq/M), bt(B)`, and puts each head band on the same XCD. Measured on the
paper's shape grid (50 shapes, `S ∈ {2K, 8K}`, `B ∈ {1, 2, 8}`, two passes), it
is well behind `auto` on causal shapes, and neither traversal rescues it:

| causal, vs `auto` | gfx942 | gfx950 |
|---|---|---|
| `swz_head_first` | −10.5% | −27.1% |
| `swz_head_first_rev` | −9.0% | −27.7% |
| `swz_head_first_fold` | −6.2% | −27.6% |

**Why.** This is the failure of "Why `Q`-fastest orders cannot be rescued",
appearing in a new place. On the grid-stride loop CTA `p` runs
`wi = p, p+NP, …`, so its block digit advances by `NP/M` per step: 32 on gfx950
(256 CTAs), 38 on gfx942 (304). Where `NQB` divides `NP/M` (every gfx950
shape here), **each CTA keeps one query block for the whole kernel**. Causal
cost per CTA then varies up to `NQB`-fold, and a per-CTA load model gives a
makespan of 1.8–1.9× ideal. `rev` and `fold` are functions of `blk` alone, so
they only relabel which CTAs are heavy. On gfx942 the block does move (38 is
not a multiple of `NQB`), so the loss is smaller and `fold` recovers part of it.
Non-causal shapes are unaffected (within about 1–2% of `auto`).

**The fix: bundle the fold pair onto one CTA.** Pair the causal fold
`{blk, NQB-1-blk}` *inside* one work unit rather than across grid-stride
steps. A CTA then runs both halves on consecutive steps (unit
`u = wi % NP + (wi // 2NP)·NP`, half `t = (wi // NP) % 2`). Every unit
has the same cost, so it no longer matters which unit a CTA gets. The head bands
stay on their XCDs, because `u % M = wi % M` whenever `M` divides `NP`. This is the
scheduler of the Triton-TLX persistent AMD flash-attention tutorial
([`amd_fa_persistent.py`](https://github.com/facebookexperimental/triton/blob/main/third_party/tlx/tutorials/amd_fa_persistent.py):
per-XCD head-batch ownership, two-tile `{light, heavy}` bundles). AITER's
LeanAttention with `XCD_REMAP` also balances per XCD, by equal tile counts.
Two details matter:
- **The last round** (`W mod 2·NP` items). CTAs that still get two items must get
  both halves of one pair. The naive split hands a CTA halves of two different
  pairs, and was worse than `fold` on gfx942.
- **`W ≤ NP`** (one item per CTA). There is nothing to balance, so emit the plain
  decode and keep the XCD alignment. Without this the pair decode lost up to about
  11% on those shapes.

Prototyped as `swz_head_first_pair` (not merged; it needs even `NQB`). On the same
grid, two fresh passes, no correctness or bit-identity failures:

| causal, vs `auto` | gfx942 | gfx950 |
|---|---|---|
| `swz_head_first_pair`, all shapes | +0.6% | **+3.3%** (32/47 wins) |
| MHA, `B·Hq ≥ 128` | **+6.8%** (10/13 wins, worst −3.2%) | **+10.5%** (12/13 wins, worst +0.1%) |
| MHA, `B·Hq < 128` | −1.7% | −0.4% |
| GQA, `Hkv = 8` | −1.6% | +1.8% |
| best cell | +18% | +26% |

The load model agrees: once there are at least two items per CTA, the gfx950
makespan drops to 1.00× ideal. So the paper's head-banded L2 locality *does* pay
on the persistent path, but only once load balance no longer depends on the
traversal. Its regime is the one seen on the non-persistent path, causal MHA with
many ACCs, and there it gains more than any other persistent order measured.
Open before shipping:
- the gfx942 losses on small MHA and GQA shapes;
- whether an `auto` rule of the form `causal && MHA && B·Hq ≥ 128` holds on the
  production shape list (by the non-persistent result above, it would rarely fire
  there);
- the golden re-bless any `auto` change requires.

## Further investigation

1. **Port `wide_lds_dma` to the non-persistent builder** (gfx950). The measured ceiling
   is ~2% over the shipped persistent arm, for a change that touches LDS layout, read
   addressing and scheduling in the largest function in the file — so it is worth doing
   behind the existing flag, with the persistent path untouched. The stronger argument
   is coverage, not peak: `varlen` is non-persistent-only and cannot use wide DMA at all
   today.
2. **Attribute the axis swap.** It is ~30x the size of the swizzle but is still only
   explained as "load balance". Measure the per-CTA cost distribution and end-of-kernel
   tail directly, rather than inferring from cache counters. Note that the assignment
   policy on its own (factor 3, holding order and traversal fixed) has been measured at
   **zero**, so the explanation has to be about which items land together, not about
   scheduling freedom.
3. **Fix `num_persistent`** to `CUs × occupancy(block_m)` and re-test the persistent path at
   a halved `block_m`; today's persistent results may be occupancy-limited artifacts.
4. **Compose the fold with the axis swap.** `hq_major_fold` exists and underperforms
   `hq_major_rev`; the pairing argument that makes the fold work on the persistent path
   should transfer, and does not.
5. **Sweep the rest of the optimal class `bt hkv *`** — the single highest-value gap.
   Per the design-space section it is the only locality class that is optimal at every
   B, it contains 6 of the 144 combinations, and exactly **one** has been measured
   (`bt hkv hql blk` + fold, built as `hkv_minor_btfast`, then dropped: a real but
   partial fix, neutral at B=1 as predicted and +1–4% rising with B):

   | digit order | traversal | status |
   |---|---|---|
   | `bt hkv hql blk` | `fold` | measured, dropped — `wi = ((blk*gqa + hql)*Hkv + hkv)*B + bt` |
   | `bt hkv hql blk` | `asc`, `rev` | **untested** |
   | `bt hkv blk hql` | `asc`, `rev`, `fold` | **untested** |

   `bt hkv blk hql` has *identical locality* to the measured one (same two fastest
   digits) and differs only in whether a CTA's stride walks `blk` or `hql` first — which
   is precisely the suspected residual in H12/H13. Both orders should be tried under
   **both** assignment policies (factor 3), not persistent only.
6. **Non-causal is unmeasured.** Every result here is causal; the production shape list
   contains no non-causal self-attention at all. The `*_rev`/`*_fold` orders silently
   degrade to forward order when `causal` is false.
7. **Thin coverage at B>1 and beyond ~16K sequence length.** The `hkv_minor` B>1 dilution
   (H6) is understood but unsolved, and the `reverse_qb` trend was still rising at the
   longest sequence measured.
8. ~~**Decide `gqa_pair`'s fate.**~~ **DONE — gated off.** Controlled head-to-head at
   matched `num_persistent` put it behind the best variant on 12/12 of the shapes where
   dispatch selected it, by 0.6–6.2%; see the section above. `resolved_persist_decode` no
   longer selects either phase decode under `auto`. The builder arms are untouched and an
   explicit `persist_decode="gqa_pair"` still works, so this gates the POLICY, not the
   capability — deleting the arms remains available but is a larger change.
9. **Revisit the XCD modulus.** `xcd_partitionable()` hardcodes 8. A part with a different
   XCD count silently turns the decode into an arbitrary permutation.
10. ~~**RECOMMENDED: demote both `hkv_minor` guards.**~~ **DONE.**
    `AttentionDenseSpec.__post_init__` rejects `hkv_minor` unless
    `xcd_partitionable(num_kv_heads)` *and* `num_persistent % num_xcds == 0`. **Neither is
    a correctness condition.** A mixed-radix decode is a bijection for any radix set, and
    both conditions only decide whether `xcd = wi % num_xcds` happens to coincide with the
    kv-head index — a *speed* property.

    Measured: the digit order `VBGQ` **is** `hkv_minor`, and the sweep ran it on
    `Hkv=10` (`gcd=2`), exactly the geometry the guard rejects — **161 cells, zero
    correctness failures, max abs error ~3e-4**. It is simply slower there: median −4.1%
    against the best variant on the same path, and it ties the best on some configs. The
    `num_persistent` guard has the same character: the shipped dispatch always passes a CU
    count (256 / 304, both multiples of 8), so other values are legal but pointless, and
    when one is used the consequence is a worse mapping, not a wrong answer.

    The cost of leaving them as errors is real and recurring: this is the second time the
    guard has excluded shapes that work (H4 relaxed pow2 → divisibility for the same
    reason), and it is why `40/10` appears as `skip` in every table above. Proposal: delete
    both `raise`s, **keep `xcd_partitionable()`** as the named predicate the `auto`
    heuristic keys on — `Hkv` vs `num_xcds` is the one shape variable that moves the
    policy's regret — and let an explicit `persist_decode="hkv_minor"` build for any shape.
    Both `raise`s are gone; `xcd_partitionable()` is retained and is now consulted by
    `AttentionDenseSpec.resolved_persist_decode` as the predictor it was always meant to
    be. `Hkv=10` and a non-multiple-of-`num_xcds` `num_persistent` both construct and
    build again.
11. **Merge the persistent `swz_head_first_pair`** (prototyped, not in the tree). It is
    the one persistent order found here that beats `auto` clearly in its regime, causal MHA
    with many ACCs; see "On the persistent path" above for the decode, the tail rule and
    what is still open before `auto` could select it.

## Where the code lives

| concern | file |
|---|---|
| `persist_decode` legality, `xcd_partitionable()` | `library/kernels/common/attention_dense_spec.py` |
| `default_grid_order`, persistent decode chain (gfx942) | `library/kernels/gfx942/attention_dense.py` |
| same + `gqa_pair*`, `runtime_shape` (gfx950) | `library/kernels/gfx950/attention_dense.py` |
| `auto` policy: persistent on/off, decode resolution | `library/dispatch/attention/{gfx942,gfx950}.py` |
