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
  working knob reads as inert. This kernel has shipped that bug twice.
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
| `batch_outer` | batch as the slowest work-index field, for `hkv_minor` | H6 — sign flips on `num_kv_heads`, so no safe default exists. **Re-add this one.** As the digit order `VGQB` it is half of the best shared persistent covering set on both arches; the sign flip is the heuristic's selection rule, not a defect. See H14 |

## In flight: the generalized-ordering sweep

The eight measured points above were hand-written decodes, each chosen before the space
was understood. Two experimental spec fields now make the space itself reachable:

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

`VGQB.rev` is `np/hkv_minor` + reverse, which already ships — the change on that path is
to make it the *default* in place of `qb_major`.

**At B=1 no heuristic is needed at all.** One fixed variant is within **0.88% (gfx942) /
0.49% (gfx950)** of the oracle, and every fitted policy improves on that by less than the
noise band. B=1 is also where the label choice is free: `VGQB` / `VGBQ` / `VBGQ` / `BVGQ`
are one kernel there. That freedom is worth spending deliberately, because **the four
diverge sharply at B>=2 and the best label differs by path** — non-persistent wants the
batch digit *last* (`VGQB`), persistent wants it *third* (`VGBQ`), and each is among the
worst choices on the other path. Picking the wrong synonym costs 2–3 points of B>=2
geomean for nothing.

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
8. **Decide `gqa_pair`'s fate — now with a measurement, not a suspicion.** Controlled
   head-to-head at matched `num_persistent` puts it behind the best variant on 12/12 of
   the shapes where dispatch actually selects it, by 0.6–6.2%; see the section above. What
   is left is a decision, not an experiment: gate it off, or delete both decodes. Deleting
   is the larger change — they are separate builder arms, not a knob — so gating the
   dispatch is the cheaper first step.
9. **Revisit the XCD modulus.** `xcd_partitionable()` hardcodes 8. A part with a different
   XCD count silently turns the decode into an arbitrary permutation.
10. **RECOMMENDED: demote both `hkv_minor` guards from errors to heuristic conditions.**
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
    No test asserts the rejection, so this is a dispatch-policy change rather than a
    contract change; still worth its own commit.

## Where the code lives

| concern | file |
|---|---|
| `persist_decode` legality, `xcd_partitionable()` | `library/kernels/common/attention_dense_spec.py` |
| `default_grid_order`, persistent decode chain (gfx942) | `library/kernels/gfx942/attention_dense.py` |
| same + `gqa_pair*`, `runtime_shape` (gfx950) | `library/kernels/gfx950/attention_dense.py` |
| `auto` policy: persistent on/off, decode resolution | `library/dispatch/attention/{gfx942,gfx950}.py` |
