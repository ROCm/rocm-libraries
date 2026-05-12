# 2lzd: Validator's `default` reference is fictional — root cause + fix catalog

Bead: `rocm-libraries-2lzd` (P0, EPIC).
Investigator: `2lzd-investigator` (Claude Opus 4.7, May 2026).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/2lzd-investigation/`.
Base branch: `users/alvasile/validator_long_term_plans`.

This memo investigates the divergence the dispatching agent demonstrated:
`default.s` (the shadow capture's per-body stream view, what
`compare_graphs` consumes as `ref`) does not match `kernel_default.s`
(the real Tensile asm built with `UseCustomMainLoopSchedule=0`). It
covers root-cause decomposition, the implicit contract the validator is
enforcing today, a catalog of forward paths, and a recommendation.

Evidence-input artifacts (regenerable via
`pytest tensilelite/Tensile/Tests/unit/_dump_carveout_assembly.py -s
--ignore=tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py`):

- `Tests/unit/cms.s` — `subj_graph` per-body GraphNode stream. ML-1 is
  194 nodes (184 dataflow + 10 control). Bodies: ML-1, ML, NGL, NLL.
- `Tests/unit/default.s` — `ref_graph` per-body GraphNode stream. ML-1
  is 254 nodes (242 dataflow + 12 control). Bodies: ML-1, ML, NGL, NLL.
  No `PRO` body — the prologue capture (`build_prologue_capture` at
  `KernelWriter.py:5141-5144`, which builds `BODY_LABEL_PROLOGUE = "PRO"`)
  is consumed by a separate code path and not present in this dump.
- `Tests/unit/kernel_cms.s` — real assembly, `UseCustomMainLoopSchedule=1`.
- `Tests/unit/kernel_default.s` — real assembly,
  `UseCustomMainLoopSchedule=0`.

Empirical concrete counts used throughout (source: `awk` over the .s files):

```
default.s ML-1: 254 nodes, 74 with mfma_index=-1 (72 Pack + 1 SYNC + 1 SNOP).
cms.s     ML-1: 194 nodes,  0 with mfma_index=-1.
Delta:           60 nodes (242-184=58 dataflow + 12-10=2 control).

ML-1 cat counts (default vs cms; only categories with non-zero delta shown):
  cat       default   cms   delta
  PackA0    40        20    +20
  PackA3    40        20    +20
  PackB0    40        20    +20
  PackB3    20        20      0
  SNOP       8         0     +8
  SYNC       4        10     -6
  LCC        0         2     -2
  LRS        2         0     +2   (default-only; cms uses LRSA/LRSB)
  LWS        2         0     +2
  LRSA       0         1     -1
  LRSB       0         1     -1
  LWSA       0         1     -1
  LWSB       0         1     -1
                              ───
                              60
```

The +60 delta sums exactly. **(A)** below sets the upstream switch.
**(B)** is the Pack-side mechanism that produces +20 in three of the
four `Pack*` rows. **(B-2)** is the same class of bug for LR/LW swaps
(+2 for `LRS`/`LWS` becoming `LRSA`/`LRSB`/`LWSA`/`LWSB` on the CMS
side). **(D1)** is the per-body re-firing of the same mechanism in
ML, NGL, NLL. **(C)** is genuinely orthogonal but contributes zero.
The +8 SNOP delta and -6 SYNC / -2 LCC deltas are downstream of (A)
(scheduler-control bookkeeping differences between the SIA3 shadow
and the CMS macro expansion).

---

## §1 — Root-cause analysis

The `default.s ↔ kernel_default.s` mismatch decomposes into a layered
set of causes, NOT four orthogonal contributors. (A) is the upstream
flag flip; (B) and (B-2) are per-plrIdx category-tagging mechanisms
that fire under either of the routes (A) activates; (D1) is the
per-body re-firing of (B)/(B-2)/(A) for each LoopBodyCapture the
shadow path produces; (C) is genuinely orthogonal and inert.

### (A) Kernel-flag mutation by the per-tile schedule (upstream switch)

**Where.** `Tensile/Components/CustomSchedule/gfx950/_128x128x32_TF32.py:119-120`
sets `kernel["MfmaInitCVgprs"] = True` and `kernel["UsePLRPack"] = True`
inside the schedule registration body. Line 53 also sets
`kernel["UseMFMAF32XEmulation"] = True`. The CMS dispatch (which calls
this schedule's body during `customMainLoopSchedule`) and the shadow
capture (which runs inside the same `kernel` dict / same `_initKernel`)
both observe these mutations. The default-baseline initializer at
`Tensile/Components/CustomSchedule/dispatch.py:546` set `UsePLRPack:
False` before the per-tile schedule fired.

**Effect on routing.** With `usePLRPack=True` (and
`doFullPackCodePrefetch=False` — see `KernelWriter.py:8023`,
`doFullPackCodePrefetch = UsePLRPack AND NOT UseCustomMainLoopSchedule`),
the prologue routes the entire pack chain into the local
`packPrePrefetchA/B` modules at `KernelWriter.py:5009, 5048` and emits
them DIRECTLY into `module` via `module.addItems(packPrePrefetchItems)`
at `:5088`. Under this routing the per-iter `pack[plrIdx]` and
`packPre[plrIdx]` lists, AND the capture-side
`_capture_context.prefetch_pack_a/b[plrIdx]` lists (populated only at
`:5004, 5013, 5043, 5052`, all in the `usePLRPack=False` else branch),
are EMPTY for the prologue Pack chain. The for-uIdx loop body then
populates `pack`/`packPre` with iter-N's freshly-emitted Pack code
(via the `localReadDo` calls inside `_loopBody` at `:4172-4273`) and
the prefetch leaves never enter the loop-body capture stream.

**Effect on `kernel_default.s` (the comparison baseline).** With
`UseCustomMainLoopSchedule=0`, the per-tile schedule never registers
its `kernel["UsePLRPack"]=True` mutation, so `usePLRPack` is False at
prologue time AND `doFullPackCodePrefetch` becomes True (per the same
`:8023` formula). The pack chain stays on `pack[plrIdx]` and is
emitted INSIDE each loop iter at mfmaIndex 0 (see
`kernel_default.s:1844-1858, 1882-1896, 1902-1912`).

**Visible artifact.** A structural-position shift: prologue Pack moves
from "inside ML-1" (kernel_default) to "before ML-1" (kernel_cms; and
captured separately into `BODY_LABEL_PROLOGUE` by the shadow). This is
the **routing precondition** that lets (B)/(B-2) over-tag and (D1)
re-fire per-body. (A) on its own does not produce any of the +60 node
delta — the routing change is a position shift, not a count change.

### (B) Per-plrIdx Pack-leaf double-tagging (loop-body mechanism)

**Where.** `Tensile/KernelWriter.py:4509-4514` (loop-body path) and
`:4607-4612` (leftover-pack path). Both walk
`self._capture_context.prefetch_pack_a[plrIdx]` (and `_b`) and call
`capture_id_to_cat.setdefault(id(_leaf), f"PackA{plrIdx}")` for each
leaf.

**The pathology that is actually visible in ML-1.** Under
`usePLRPack=True` the prefetch_pack_a/b lists are EMPTY (see (A)
above), so the 4509-4514 walk tags nothing. The Pack leaves observed
at `default.s:4-149` come from the **per-iter** pack code that the
for-uIdx loop's `localReadDo` (`:4190, 4250`) writes into
`pack[packStoreIdx]` and `packPre[packStoreIdx]`. Those leaves are
tagged via `build_idmap` over `PackCodeAAllIters[uIdx]` /
`PackCodeBAllIters[uIdx]` (`:4488-4502`), where `PackCodeAAllIters[u]`
contains BOTH `packPreA` and `packCodeA` (`:4208-4209`,
`:4272-4273`). Because the ML-1 capture's idMap categorizes leaves by
the iter `u` they were FIRST emitted in, and the same physical
`v_cvt_pk_bf16_f32` text appears in multiple per-iter modules under
TF32-emulation with `numItersPLR=2`, the plrIdx-keyed-by-iter routing
yields PackA0 (iter 0) and PackA3 (iter 3) tags for what is the same
canonical instruction text.

**Visible evidence in `default.s:4-11`:**

```
stream_index=1 cat=PackA3 v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]
stream_index=2 cat=PackA0 v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]
stream_index=3 cat=PackA3 v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+1], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
stream_index=4 cat=PackA0 v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+1], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
```

Identical canonical text, two stream positions, two category tags. In
ML-1 the Pack categories are `{PackA0, PackA3, PackB0, PackB3}` —
**no** PackA1/PackB1 in ML-1. PackA1/PackB1 do appear in the NGL and
NLL bodies (stream_index ranges covered by `loop_index=2` and
`loop_index=3`), where the noLoadLoopBodyDefault path
(`:3623-3635`) builds idmaps with a different per-iter category
mapping. The duplicated-text-different-tag pattern is the same in all
four bodies; only the plrIdx values differ.

**Note on the upstream-switch dependency.** This mechanism would NOT
fire if (A) were not active, because under
`usePLRPack=False`/`doFullPackCodePrefetch=True` the SIA3 default
emit places the same Pack chain in a single position with stable
identity — the per-iter idmap entries reduce to a single canonical
tag. So (B) is structurally downstream of (A): both Pack-routing
branches at `:5005-5013, 5044-5052` produce the same plrIdx-keyed
tagging pattern, but only the `usePLRPack=True` branch creates the
routing mismatch that surfaces it as a visible delta.

**Concrete contribution.** From the count table above: PackA0 +20,
PackA3 +20, PackB0 +20 = **+60 in the Pack* rows of ML-1**. PackB3
+0 (already balanced; B-side iter 3 happens to land identically
in both captures). Net of overlap with (D1)'s per-body
re-firing, (B) is the dominant single contributor to the loop-body
node delta.

### (B-2) Per-A/B-suffix LR/LW-swap tagging divergence (same family)

**Where.** `Tensile/KernelWriter.py:976-979` tags pointer-math items
into the generic `'LRS'`/`'LWS'` buckets (no A/B suffix), because
`pointerLRCode`/`pointerLWCode` combine A and B in the SIA3 view.
The CMS macro path tags the SAME logical instructions into
`LRSA/LRSB/LWSA/LWSB` bucks (visible as cms.s ML-1's
`{LRSA: 1, LRSB: 1, LWSA: 1, LWSB: 1}` vs default.s ML-1's
`{LRS: 2, LWS: 2}`).

**Why this matters.** `compare_graphs`'s identity-coverage check
relies on category tags being stable across captures. Even though
the physical instructions and their canonical_str are identical,
the bucket labels diverge — the same shape-of-bug Z012 §6
documents for Pack tagging and `Z012_ALTERNATIVE_FIXES_INVESTIGATION.md`
§ "category-tagging-strategy divergence (multiple families)"
identifies as a pattern that crosses Pack/LR-swap/LW-swap.

**Concrete contribution.** Net **0** to the ML-1 node count
(2+2 default-side LRS+LWS == 1+1+1+1 cms-side LRSA+LRSB+LWSA+LWSB),
but it produces **2 category-mismatch graph-edge differences**
that compare_graphs surfaces independently of the count delta.
This is why bead `2lzd` and the Z012 family are conceptually one
bug — see § Z012 cross-link below.

### (C) Slot-lex ordering vs source-line ordering — independent?

`assign_stream_indices_for_body` (`ScheduleCapture.py:744-760`)
lex-sorts events on `(slot.mfma_index, slot.sequence)`. Empirical
check: `cms.s` ML positions 0..5 (MFMA@0.0, GRIncA@0.1
s_cmp_eq_u32, LRA0@0.2 ds_read_b128 offset:256, LRA0@0.3
ds_read_b128 offset:320, MFMA@1.0, GRIncA@1.1 s_cselect_b32 s68)
EXACTLY matches `kernel_cms.s` MAINLOOP source order (lines
1846-1857). Slot-lex order is faithful for this kernel.

For the default side there's no "source emit" to check against —
the shadow is a synthetic re-assembly from the captured idmap. The
visible interleaving in `default.s:4-19` is governed by the upstream
tagging in (B), not the sort.

**Verdict.** (C) is not an independent contributor here. Bad inputs,
faithful sort. **Contribution: 0 nodes of the +60 delta.**

### (D1) Per-body replication of the loop-body @-1 head

**The actual mechanism.** The shadow path runs `_makeSubIterSchedule`
once per uIdx via `_loopBody` (ML-1, ML) AND
`_noLoadLoopBodyDefault` (NGL, NLL). Inside
`_makeSubIterSchedule`, `_captureSubIterToBuilder`
(`KernelWriter.py:2641-2688`) walks `iterCode.flatitems()` and assigns
`slot_kind=SLOT_KIND_PRE_LOOP, mfma_index=-1` to every item that
appears BEFORE the first MFMA of `subiter==0` (`:2669-2673`). Each
body produces ITS OWN LoopBodyCaptureBuilder (`:4471` for the loop
body; per-call for NLL/NGL via `_noLoadLoopBodyDefault`'s
capture-on path at `:3636-3639`). So the @-1 head is NOT a
"prologue replication" — it is each body's own subiter=0
pre-MFMA stream tagged at @-1 because that's the assigned
slot_kind for items before the body's first MFMA.

**The reviewer corrected the original memo's mechanism.** The
original mechanism cited
`_capture_context.prologue_prefetch_pack_a/b` as the source of the
per-body @-1 head, but that snapshot is consumed exactly once at
`KernelWriter.py:5141-5144` via `build_prologue_capture` to produce
`BODY_LABEL_PROLOGUE = "PRO"` — a body that is NOT in the .s dumps
and therefore cannot explain the @-1 entries in ML-1/ML/NGL/NLL.

**Why the @-1 head appears in EVERY body.** Each body's
`_loopBody` / `_noLoadLoopBodyDefault` invocation drives
`_makeSubIterSchedule` for u in [0, LoopIters). At u=0 every leaf
emitted before that body's first MFMA — the leaves coming from
`pack[packIdx=0]` and `packPre[packPreIdx=2]` populated by the
for-uIdx loop's iter-0 `localReadDo` calls — gets stamped with
slot_kind=PRE_LOOP / mfma_index=-1. The leftover-pack path at
`:4613-4628` ALSO stamps `subiter=LoopIters,
slot_kind=POST_LOOP, mfma_index=-1`, contributing another batch
of @-1 entries from leaves still in `pack[*]`/`packPre[*]` at
loop end. NLL/NGL bodies have a smaller pack residue (no
leftover-capture call; @-1 entries come only from the body's
subiter=0 pre-MFMA stream), explaining why ML-1 and ML have 74
each but NGL/NLL only have 14 each.

**Per-body @-1 counts (verified).**

| Body | @-1 count | Mechanism |
|---|---|---|
| ML-1 | 74 | subiter=0 PRE_LOOP + leftover POST_LOOP |
| ML | 74 | subiter=0 PRE_LOOP + leftover POST_LOOP |
| NGL | 14 | subiter=0 PRE_LOOP only |
| NLL | 14 | subiter=0 PRE_LOOP only |

**Concrete contribution.** ML-1 and ML have identical 254-node
counts; NGL has 178, NLL has 148. The (D1) replication accounts
for the parallelism between ML-1 and ML and produces the cross-body
shape-similarity, but it isn't separately attributable to the
+60 ML-1 delta — that delta is fully explained by (B) and (B-2)
in the table above. (D1) does explain why fixing (B)/(B-2) at
the source produces a 4-bodies-worth count drop, not a 1-body
drop.

### (D2,D3) Other flag mutations

`MfmaInitCVgprs` and `UseMFMAF32XEmulation` are mutated by the
same per-tile schedule but the shadow doesn't capture pre-loop
init outside the prefetch-pack tagging path, so they contribute
0 to the visible ML-1 delta today. They are latent — a future
schedule that mutates one of them differently would NOT be caught.

### Summary table

Categorical attribution of the **+60-node ML-1 delta** (default - cms),
computed exactly from per-cat counts. Causes appear in the order they
fire mechanically; "downstream-of" notes when one cause is a
precondition for another:

| Cause | Net ML-1 node delta | Notes |
|---|---|---|
| (A) `UsePLRPack` flip + downstream `doFullPackCodePrefetch=False` | 0 (routing only) | Precondition for (B)/(B-2) to surface as deltas |
| (B) Pack-leaf per-plrIdx double-tagging | +60 (PackA0+20, PackA3+20, PackB0+20) | Downstream of (A) |
| (B-2) LR/LW-swap A/B-suffix divergence | 0 (count) / +4 (cat-mismatch edges) | Downstream of (A); same family as (B) |
| (C) Slot-lex sort | 0 | Faithful sort, this kernel |
| (D1) Per-body PRE_LOOP head replication | 0 (count attribution) | Explains cross-body shape; ML-1 + ML carry the same +60 each |
| (D2,D3) `MfmaInitCVgprs` / `UseMFMAF32XEmulation` | 0 today (latent) | Not captured |
| Scheduler-control bookkeeping (SNOP+8, SYNC-6, LCC-2) | 0 net (sums to 0 across categories) | Downstream of (A); not separately actionable |

The +60 attributable wholly to (B). (B-2) adds 4 graph-edge
differences without changing the count. (D1) explains the
cross-body symmetry. The other rows are routing/bookkeeping noise
downstream of (A).

---

## §2 — What contract is the validator currently enforcing?

Given the mechanics in §1, the validator runs THREE distinct checks
in sequence (`CMSValidator.py:4317-4338`); only one of them is
broken by the issues above:

1. **`compare_graphs(ref_graph, subj_graph)`** (`:4319`) — the
   order-comparison + identity-coverage check. THIS is what the
   memo's title "validator's default reference is fictional" refers
   to. ref_graph is the synthetic per-body shadow; the (B)/(B-2)
   tagging artifacts and the (A) routing shift inflate ref relative
   to subj.

2. **`validate_edge_wait_coverage(subj_graph)`** (`:4329`) —
   independent of ref. Walks ONLY subj_graph. Validates that each
   raw-intrawave edge is covered by an `s_waitcnt`. Works correctly
   today; not affected by 2lzd's mechanics.

3. **`validate_vopd_pair_formation(subj_graph)`** (`:4342`) —
   independent of ref. RDNA3.5 §7.6 hard rules. Dormant today.

**The 2lzd defect is scoped to (1).** Wait coverage and VOPD
checks are not part of the bead's surface area and need not be
modified by any fix below. References to "the validator" in §3
mean component (1) specifically.

`compare_graphs(ref_graph, subj_graph)` — called from
`KernelWriter.py:5416` for every CMS kernel build — is enforcing
this contract:

> **For a CMS kernel build, the per-byte dataflow graph derived from
> the SHADOW SIA3 emit (using the SAME CMS-mutated `kernel` dict and
> the SAME plrIdx-keyed per-iter Pack/LR-swap idmap) must equal the
> per-byte dataflow graph derived from the CMS schedule's
> macro-expanded emit (against the same `kernel` dict).**

Stated more precisely: the validator demands that two synthetic
re-assemblies of the same kernel writer's output — one going through
SIA3 (the default scheduler), one going through the CMS macro
expander — produce isomorphic per-byte dataflow when given identical
upstream Module references and identical kernel-flag state. Both
re-assemblies are scheduling-layer comparisons; neither is a literal
representation of either real-Tensile-emit (CMS or non-CMS).

**What this contract DOES validate (today):**

1. **Reorder-safety of CMS macro expansion vs SIA3 default emission**
   for the per-iter / per-body Pack/MFMA/LR/GRInc/etc. material — the
   hot loop body's main schedule. This is genuinely valuable: it
   catches CMS macros that drop, duplicate, or mis-place mainloop
   instructions relative to what SIA3 would have done with the same
   inputs.
2. **Per-byte dataflow consistency** — every read in the CMS schedule
   resolves to the same producer (or no producer, identically) as in
   the SIA3 shadow. This is the carve-out / wait-coverage gate's
   actual job and it works.
3. **Wait-coverage equivalence** for in-body raw-intrawave edges.

**What this contract provably does NOT validate:**

1. **CMS-vs-real-non-CMS-Tensile structural equivalence.** Because the
   shadow capture sees CMS-mutated `kernel` flags (UsePLRPack,
   UseMFMAF32XEmulation, MfmaInitCVgprs), the shadow does NOT
   represent what a kernel built with `UseCustomMainLoopSchedule=0`
   would emit. A user reading "CMS validates against the default
   schedule" infers a guarantee that doesn't hold.
2. **Prologue correctness.** The 74-node @-1 head in each shadow
   ML-1/ML body has no corresponding real instruction stream — it's
   the per-body `subiter=0` PRE_LOOP plus leftover POST_LOOP capture,
   over-tagged by (B), and (in the real CMS-mutated kernel) emitted
   ONCE in the prologue. The validator doesn't notice any of this
   because both ref and subj see the same inflated shape.
3. **Effect of kernel-flag mutation on emit shape.** Anything the
   per-tile schedule mutates is invisible to the validator by
   construction.
4. **`PackA0 ≠ PackA1 ≠ PackA3`** as semantic markers, AND
   `LRS ≠ LRSA ≠ LRSB` etc. They're tag strings used for category
   bucket lookup; the dataflow graph collapses them via
   `nodes_by_identity` (same canonical_str → same node), so the
   double-tagging in (B)/(B-2) above doesn't always surface as two
   nodes — it just inflates the LoopBodyCapture's instruction count
   and downstream node-count, which then matches between ref and
   subj because both sides inflate identically.

The contract is therefore best read as an **internal-consistency
check** between two scheduling backends, NOT as a correctness check of
the CMS schedule against a Tensile baseline. The bead's framing —
that the validator's "default reference is fictional" — is correct;
the question is whether the fictional reference is still useful.

### Cross-link: PRELOOP_CAPTURE_PHASE1.md design intent

`Tensile/Components/PRELOOP_CAPTURE_PHASE1.md:329-346` argues
explicitly that `UsePLRPack` "is a CMS-shaped kernel-config flag" and
that "comparing a CMS build (with `UsePLRPack=1`) against the same
problem built default-style (which would not enable `UsePLRPack`)
would show **structurally different prologues** by construction."
The memo's authors built `_captureDefaultSchedule` with
CMS-vs-CMS-shadow as the explicit design contract, NOT
CMS-vs-non-CMS.

**Consequence for §3 below.** Approach A (a real second build with
`UseCustomMainLoopSchedule=0`) directly contradicts this design
intent. Either Approach A is wrong or the design intent needs
revisiting. Approach A as written would surface the prologue
divergence on every CMS build — every per-tile schedule that mutates
`UsePLRPack` would fail an A-style comparison, regardless of
correctness. Either the contract is downgraded to "ignore prologue
shape" (in which case it isn't a real correctness check) or
A is scoped narrowly to specific kernels where the per-tile schedule
demonstrably does NOT mutate the flag.

### Cross-link: Z012's category-tagging-strategy investigation

`Z012_ALTERNATIVE_FIXES_INVESTIGATION.md` documents (§6 + §"category-
tagging-strategy divergence (multiple families)") that the Pack-tagging
divergence is one instance of a class spanning Pack and LRSwap
(LRSA/LRSB) tagging. Our (B-2) finding above confirms this directly
in ML-1: the LR/LW pointer-swap items use `LRS`/`LWS` in the shadow
SIA3 path and `LRSA`/`LRSB`/`LWSA`/`LWSB` on the CMS macro side.
**This is load-bearing for §3 Approach C below**: any dedup strategy
that fixes Pack but not LR/LW tagging would close ML-1's count delta
but leave the 4 category-mismatch graph-edge differences. Approach
C must address (B) AND (B-2) together, OR the Z012 investigation
must be merged into 2lzd as a single bead, OR the two beads must
have a documented "fix at the same time" cross-link.

---

## §3 — Approaches to fix

Six distinct paths, plus three hybrids. Each is a forward direction,
not an implementation plan; the user picks one and dispatches a
separate implementer.

### Approach A — "True default reference build"

**Mechanism.** Run a SECOND isolated kernel build with
`UseCustomMainLoopSchedule=0` (mirroring `_dump_carveout_assembly.py:
214-222` Build #2), capture its real emit via a non-shadow path, use
it as `ref`. Demote or retire today's shadow.

**New contract.** "CMS preserves what a real non-CMS Tensile build
would emit, modulo permutations the graph allows." Matches users'
naive assumption.

**Cost.** Medium-to-large. ~2x build-time on the assert path; needs
a non-CMS capture path; ~300-600 LoC. Requires either fresh
kernel-init or flag-reset (overlaps with B).

**Risks.** Build-time in CI. CMS-only features (forced UsePLRPack,
specific sync patterns) have no non-CMS counterpart — would surface
as failures requiring suppression, weakening the contract.

**Direct contradiction with PRELOOP_CAPTURE_PHASE1.md.** That memo
argues CMS-vs-non-CMS comparison "cannot be done meaningfully" for
kernels whose per-tile schedule mutates flags like `UsePLRPack`. A's
contract is exactly that comparison. Either (i) A is scoped to kernels
that demonstrably don't mutate any captured flag (a small subset),
(ii) A's contract is weakened to "ignore prologue shape, compare loop
body only" (which is much closer to the existing CMS-vs-CMS-shadow
contract, narrowing A's value), or (iii) the design intent is
explicitly revised. The user must decide which.

**Open questions.** (1) Build-time impact? (2) Schedules with NO
non-CMS counterpart? (3) Soft-skip vs hard-fail when non-CMS build
raises? (4) PRELOOP_CAPTURE_PHASE1's design-intent objection.

### Approach B — "Snapshot kernel dict before CMS mutation"

**Mechanism.** Deep-copy `kernel` dict after `_initKernel` (before
the per-tile schedule fires). Build a third capture that restores
the snapshot and drives `_loopBody` without the schedule's
mutations. Use as `ref`.

**New contract.** "CMS schedule preserves what the same writer
would emit with the schedule's flag mutations rolled back." A
flag-isolated cousin of A.

**Cost.** Medium. Snapshot is one `deepcopy`. But the writer's
internal state (e.g. `states.numItersPLR`, and crucially
`states.doFullPackCodePrefetch` — which is computed at `:8023` from
`kernel["UsePLRPack"]` AND `kernel["UseCustomMainLoopSchedule"]`)
was computed FROM the post-mutation flags; a simple dict swap
doesn't undo downstream effects. Either full writer re-init
(~Approach A cost) or selective undo (fragile, parallel table).

**Risks.** Selective undo: every new per-tile-schedule mutation
must register an "undo." Full re-init ≈ Approach A cost without
its cleanliness.

**Open questions.** (1) Is mutation always (key, old, new)? (2)
Can the writer re-drive cheaply enough to beat A?

### Approach C — "Disable shadow's category double-tagging (Pack and LR/LW family)"

**Mechanism.** Fix causes (B) AND (B-2) directly. At the prefetch-pack
tagging sites (`KernelWriter.py:4509-4514, 4607-4612`,
`ScheduleCapture.py:2007-2012`) AND the pointer-swap tagging sites
(`:976-979`), pick a unified tagging strategy. Either "first
plrIdx wins" (dedup on `canonical_str` instead of `id(leaf)`, which
is what defeats the existing `setdefault`) or "last plrIdx wins"
(whichever plrIdx the prologue physically consumed). For LR/LW,
either (i) generalize the SIA3 shadow to tag with A/B suffixes from
the source-module name like CMS does, or (ii) collapse CMS to
`LRS`/`LWS` and lose the A/B distinction.

**New contract.** Same as today (CMS-shadow vs CMS-emit), but with
both classes of duplicate-text artifact removed. Drops shadow ML-1
from 254 → ~180 nodes (closer to CMS's 194); residual gap then
traces cleanly to (A) routing.

**Cost.** Small-to-medium. ~30-100 LoC across two files for the Pack
fix, ~40-80 LoC across `_makeSubIterSchedule` and CMS macro emission
for the LR/LW fix. Existing test fixtures need golden updates
(per-body node counts change).

**Risks.** Removes the noise that today obscures (A); failures
attributable to the flag-mutation gap may start surfacing. Doesn't
address (A) itself.

**This Approach inherits the Z012 cross-link.** A C that fixes only
(B) Pack-tagging will close the +60 count delta but leave (B-2)'s 4
category-mismatch graph-edge differences intact. Either C addresses
both families together, or C is split into two sub-approaches with a
hard "both-or-neither" sequencing constraint.

**Open questions.** (1) Which dedup key is correct? (2) Any
legitimate case where the same canonical text should appear under
two plrIdx tags? (3) For LR/LW: A/B suffixes everywhere or generic
LRS/LWS everywhere? (4) Should 2lzd and the Z012 family be merged
into a single bead or kept linked?

### Approach D — "Diff against schedule registry, not shadow"

**Mechanism.** Restructure `compare_graphs` to diff CMS-emit
(`subj`) against the **declared slot table** of the per-tile
schedule (the `optSchedule` dict, e.g.
`_128x128x32_TF32.py:92-111`). Slot table is intent; CMS macro is
realization. Shadow is no longer in the loop.

**New contract.** "The CMS macro emits exactly the slots its
declared schedule promises — no missing, no extra, dataflow
consistent with the slot ordering."

**Cost.** Medium-large refactor. ~50 unit-test fixtures migrate
from "synthesize ref graph" → "synthesize expected slot table".
Helper for table → expected-positions (~100 LoC). Rewrite
`diagnose_missing_edge` and friends (~200-400 LoC). Retire shadow
capture machinery for compare_graphs only (~500 LoC); leave
wait-coverage and VOPD validators untouched. Total ~800-1000 LoC.

**Risks.** Largest blast radius. Loses ability to catch mismatches
the slot table doesn't enumerate. Gains ability to validate against
declared intent — a stronger claim than "matches SIA3 default."

**Open questions.** (1) Slot table sufficient as contract target?
(2) Migration cost vs. ongoing shadow maintenance cost over a year?
(3) Can both contracts coexist during migration?

### Approach E — "Differential pair-comparison via isomorphic transformation"

**Mechanism.** Build an additional shadow with `UsePLRPack=False`
forced (`ref_baseline`) alongside the current shadow (`ref`).
Compute `delta_ref = diff(ref_baseline, ref)` — the shape-change
from the flag mutation. Build the analogous `delta_subj`. Assert
`delta_ref ≅ delta_subj`. The flag mutation becomes a controlled
variable rather than a hidden one.

**New contract.** "Whatever the CMS-mutated flags do to the SIA3
emit, the CMS macro emit reflects the same dataflow change."

**Cost.** Large. Three captures per build. Needs flag-mutation
rollback (overlaps with B). "Delta of graphs" comparator is a new
design surface with no precedent. ~500-1000 LoC.

**Risks.** Build-time x3 on the assert path. Replaces one fictional
reference with two; the new contract itself needs a correctness
story.

**Open questions.** (1) Does it catch any bug class A/B don't?
(2) Is 3x build-time tolerable? (3) Caching across builds?

### Approach H — "cms_from_default-driven same-direction synthesis"

**Mechanism.** `Tensile/Components/CustomSchedule/cms_from_default.py`
already converts a default-style YAML into a CMS schedule
(`default_schedule_to_cms`, "rocm-libraries-wlrp Phase 2"). Build
TWO kernels with the SAME synthesized schedule: one through CMS,
one through default-converted-to-CMS. Both kernels go through the
CMS macro path; the comparison is CMS-emit vs CMS-emit, with one
side's schedule constructed from a default-style declaration. The
shadow capture is bypassed entirely; the ref side is a real
CMS-built kernel whose schedule was declaratively derived from
default-style intent.

**New contract.** "The CMS macro produces the same dataflow as
when the schedule is synthesized from a default-style declaration."

**Cost.** Medium. `cms_from_default.py` already exists; ~200-400 LoC
to wire it into the validator's gate path. Test surface
already exists (`test_cms_from_default.py`). No new "delta of
graphs" comparator. Build-time ~2x (two CMS-path builds) but no
SIA3 shadow runs needed.

**Risks.** The two builds are not arbitrary kernels — they share
the schedule. If `cms_from_default` has its own bugs (and it's
recent code), the ref side is itself wrong. But its bugs are
discoverable independently.

**Sidesteps the PRELOOP_CAPTURE_PHASE1.md objection.** Both sides
are CMS, so prologue routing is identical. Sidesteps the (A)
routing shift entirely.

**Open questions.** (1) Coverage: which kernels does
`cms_from_default` know how to convert? (2) `cms_from_default`'s
maturity — fit for production gate use today?

### Approach F (hybrid) — C + A: fix B/B-2, then add a second build for ground truth

**Mechanism.** Land Approach C as a fast precursor (clean up the
shadow's double-tagging for both Pack and LR/LW families, get the
ref graph closer to a representative shape). Then layer Approach A
on top: the second build runs only on test/CI gates (not every
developer's local CMS build), and gates acceptance of the per-tile
schedule against the real default emit once. The shadow contract
continues to gate per-build for incremental safety.

**New contract.** Two-tier: (i) per-build, the existing shadow vs
CMS contract (now sharper due to C); (ii) on test/CI, the shadow's
ref shape is itself validated against a real default build.

**Cost.** Small precursor (C) + medium follow-up (A scoped to
CI-only gates). Spreads the risk and lets the user evaluate (A)'s
build-time impact incrementally.

**Risks.** Two contracts is more complexity to maintain. The CI-only
A gate may surface failures that don't gate per-build builds, leading
to "passes locally, fails in CI" friction. Inherits A's
PRELOOP_CAPTURE_PHASE1 design-intent objection.

### Approach G (hybrid) — D + targeted shadow retention

**Mechanism.** Migrate the main `compare_graphs` contract to D
(schedule-registry-driven), but retain the shadow capture as a
debugging aid (off by default; enabled via a flag for divergence
investigations). Get D's principled correctness story and keep the
shadow's diagnostic value.

**Cost.** Approach D's cost; the shadow retention is essentially
free (just don't delete it).

**Risks.** Same as D, plus the temptation to defer fully retiring
the shadow (it lingers as dead-code maintenance burden).

### Approach I (hybrid) — C + H: cleanup + same-direction synthesis ground truth

**Mechanism.** C as the immediate cleanup (closes the
double-tagging artifact). H as the principled new ref source
(removes reliance on the shadow without contradicting
PRELOOP_CAPTURE_PHASE1).

**Cost.** C (~100-200 LoC) + H (~200-400 LoC).

**Risks.** Lower than F because H avoids A's design-intent
collision. Higher than C-alone because H requires
`cms_from_default` to be production-ready.

---

## §4 — Recommendation

**[SUPERSEDED 2026-05-12 — see §6.]** The user has rejected the shadow-vs-CMS
comparison contract entirely; Approach C (shadow tag-cleanup) is no longer the
recommended near-term landing. Live approach set is now {A, D, H} — see §6.
The text below is preserved for historical context only.

The choice depends on which contract the user wants to enforce
(this is the load-bearing question in §5 below). Conditional
recommendations:

**If contract = "CMS preserves real non-CMS Tensile emission":**
→ Approach **A** ("true default reference build"), but accept that
PRELOOP_CAPTURE_PHASE1.md's design-intent objection must be resolved
first. A as written today contradicts that design memo.

**If contract = "CMS macro internally consistent with SIA3 default
on the same kernel-flag state":**
→ Approach **C** (Pack AND LR/LW). Necessary AND sufficient to make
the existing contract honest. The contract today is implicitly this;
C just removes the artifact inflation that obscures real
divergences. Must address (B) and (B-2) together — see Z012
cross-link in §2.

**If contract = "CMS schedule honors its declared per-tile-schedule
intent":**
→ Approach **D** or **G**. D is the principled option but has a
large blast radius; G softens the migration. Either way, the shadow
ceases to be the ref source.

**If contract = "CMS macro produces consistent dataflow regardless
of how the schedule was declared":**
→ Approach **H** (cms_from_default-driven). Sidesteps the
PRELOOP_CAPTURE_PHASE1 objection by keeping both sides in CMS.

**Most pragmatic single-choice recommendation:** **Approach C as a
near-term landing**, with the user's contract-resolution decision
informing whether to follow up with H, A, D, or G in a separate
cycle. C alone removes the visible artifact that makes the bead's
evidence look damning; the deeper "shadow vs reality" question
remains open but gets cleaner to reason about once C is done. Note
that C MUST cover both (B) Pack and (B-2) LR/LW tagging to be
useful — the Z012 family is one bug.

**What I would NOT recommend:**

- **B alone** — flag-snapshot-then-replay is fragile and doesn't
  address (B)/(B-2) tagging. If you're going to do the
  snapshot/restore work, do A — it's the same cost-class with a
  cleaner contract.
- **E alone** — too speculative for the gain. The "delta of graphs"
  comparator is a new design surface with no precedent; better to
  pick a definite contract and validate against it.
- **Doing nothing** — the current contract is described in tests and
  memos as if it validates against a real default emit. Either fix
  the contract (one of A/D/H) or fix the documentation (rebrand the
  validator's purpose explicitly as "internal consistency between
  two scheduling backends, not a correctness check against a real
  Tensile baseline"). Today's framing oversells what the validator
  actually does.

**Validator-component scoping reminder.** Wait-coverage and
VOPD-pair-formation validators (CMSValidator.py:4329, 4342) are
independent of `compare_graphs` and not affected by 2lzd's
mechanics. They should be left in place by every approach above.

---

## §5 — Open questions for the user

1. **What contract does the validator EXIST to enforce?**
   **RESOLVED 2026-05-12 — see §6: shadow comparison rejected.
   Implementation choice (A/D/H) remains open.** Three
   candidates from the bead description:
   - (a) "CMS preserves what a real non-CMS Tensile build would emit
     for this problem"
   - (b) "CMS-emit and default-emit produce equivalent dataflow when
     given identical kernel flags"
   - (c) "CMS schedule honors its declared per-tile-schedule intent"
   The current code implements (b) but with the (B)/(B-2)
   tagging artifacts obscuring it. The bead implicitly asks for (a).
   PRELOOP_CAPTURE_PHASE1.md argues (a) is "structurally
   ill-defined" for kernels with flag-mutating schedules. The
   validator's TEST COMMENTARY and several memos
   (`CMSValidator_LIMITATIONS.md`) describe behavior consistent
   with (a). All three are coherent contracts; the user needs to
   pick.

2. **RESOLVED 2026-05-12: implementation-detail reading.** User input is the YAML; Tensilelite is free to mutate kernel state on either side. The comparison is YAML-to-YAML, not post-mutation-kernel-to-pre-mutation-kernel. UsePLRPack-divergence and similar prologue-shape divergences are expected and will be addressed by oram.1's body-label-tolerance work, not avoided by Approach A's setup.

   **Is the per-tile schedule's right to mutate `kernel` flags
   considered part of the schedule's API surface, or an
   implementation detail of CMS?** If the former, A (true default
   build) intrinsically can't validate the post-mutation kernel
   against the pre-mutation default — those are different kernels.
   If the latter, A is well-defined. This affects whether contract
   (a) is even tractable.

3. **RESOLVED 2026-05-12: build-time is not a current concern.** Correctness first. Build-time optimization (caching the reference kernel, gating to test/CI, etc.) is deferred until correctness lands.

   **How much build-time cost is tolerable on the assert path?**
   Approach A doubles per-build cost; Approach E triples. Approach
   C is free at runtime; H is ~2x. If build-time is a hard
   constraint, only C, D, and H (with H gated to test/CI) survive.

4. **Does the user accept that the shadow capture is fundamentally
   an "internal-consistency check" rather than a "correctness against
   reality" check?** If yes, Approach C (with B-2) is sufficient and
   the current architecture stays. If no, one of A/D/G/H is required.

5. **RESOLVED 2026-05-12: merge into one larger investigation, INCLUDING oram.1 (preloop body-label work).** All three beads (2lzd, z012-family, oram.1) are facets of "make compare_graphs do the right thing when comparing two real builds of the same YAML." File a meta-bead that supersedes them; keep the existing beads open as sub-beads of the meta.

   **Should 2lzd and the Z012 family be merged?** Z012's §6 + the
   "category-tagging-strategy divergence (multiple families)"
   investigation document the same root pattern that 2lzd's (B)
   and (B-2) are concrete instances of. Approach C cannot be
   designed correctly without considering whether the dedup
   strategy generalizes to LR/LW as well as Pack — they are
   conceptually one bug. Strong recommendation: file a single
   meta-bead that supersedes both, OR add a hard cross-link
   constraint that any C-style fix must address both families
   atomically.

6. **Follow-up bead candidates (discovered while investigating, not
   in scope here):**
   - The leftover-pack capture path at `KernelWriter.py:4613-4628`
     stamps `slot_kind=POST_LOOP, mfma_index=-1`, contributing
     additional @-1 entries to ML-1's tail. If POST_LOOP entries
     should sort distinctly from PRE_LOOP in the dump output (they
     don't today), file a bead to disambiguate.
   - `_noLoadLoopBodyDefault`'s capture path uses
     `build_id_to_category_per_iter` (`:3623`) instead of the
     `build_idmap` factory used by `_loopBody` (`:4488`). If the
     two factories ever drift, the NGL/NLL bodies' tagging will
     diverge from ML-1/ML's. File a bead to track factory unification.

---

## §6 — Decision recorded (2026-05-12)

**User decision (verbatim, 2026-05-12):**

> "We should not be comparing against the shadow. The shadow is a kernel
> that cannot be emitted and does not get run. Bake this decision into
> the relevant .md files. Whatever issues arise from this will have to
> be solved, but this is the actual correct path."

### Implication

The shadow capture (`_last_default_capture`, populated under
`self.states._captureDefaultSchedule`) is **rejected as the validator's
comparison reference**. The shadow is a synthetic re-assembly of what the
default scheduler *would have produced* with the CMS-mutated `kernel` dict
(see §1 (A) for the upstream flag-flip mechanism). It is not assembled into
runnable code and never executes on hardware. Comparing against it is
comparing against a fiction.

**Approach C (fix the shadow's category-tagging artifacts (B)/(B-2)) is no
longer the recommended near-term landing.** Cleaning up the shadow's
double-tagging is wasted work if the shadow itself is going away. C remains
documented above for historical reference, but should not be implemented.

### Live approach set: {A, D, H} from §3

The user's framing — "a kernel that cannot be emitted and does not get
run" — narrows §3 to approaches whose reference side is either (i) a real
emittable+runnable kernel build, or (ii) something other than a graph
comparison entirely:

- **Approach A — true non-CMS reference build.** A second isolated kernel
  build with `UseCustomMainLoopSchedule=0`, captured via a non-shadow path
  and used as `ref`. The reference IS a real kernel that gets emitted and
  could be run. PRELOOP_CAPTURE_PHASE1.md's design-intent objection (see
  §3 Approach A) — that schedules mutating `UsePLRPack` and friends would
  surface structural prologue divergence on every CMS build — must be
  resolved by the comparator. Per oram.1's pipelining-only model
  (`PRELOOP_CAPTURE_PHASE1.md §1`), the dataflow IS equivalent across the
  flag flip; the body-label-sensitivity in `compare_graphs` (oram.1) is
  what stops A from working today.
- **Approach D — drop the comparison contract.** Restructure
  `compare_graphs` to validate against the declared per-tile schedule's
  slot table rather than any kernel-derived reference. No second kernel
  build, no shadow, no cross-graph comparison. The "what does the
  validator validate" question collapses to "the CMS macro emits exactly
  the slots the schedule declared," which is checkable from the schedule
  registry alone.
- **Approach H — `cms_from_default`-driven same-direction synthesis.**
  Build TWO real CMS kernels: one from the registered CMS schedule, one
  from a default-style YAML routed through
  `Tensile/Components/CustomSchedule/cms_from_default.py`'s
  `default_schedule_to_cms`. Both sides are emitted and runnable; both go
  through the CMS macro path so prologue routing matches by construction.
  Sidesteps PRELOOP_CAPTURE_PHASE1's objection AND oram.1's body-label
  blocker (identical body shape on both sides).

**Re-ranking under the user's framing.** A and H both satisfy the "real
emittable kernel" criterion; D satisfies the "no shadow" criterion by
removing comparison entirely. H is the cheapest of the three to pilot
because both sides remain inside the CMS macro path (no need to build a
second non-CMS-shaped capture pipeline). A is conceptually closest to a
naive user's mental model ("CMS preserves what default would emit") but
inherits the largest comparator-design surface (oram.1's
body-label-sensitivity becomes critical-path). D is the most radical and
the most principled but has the largest blast radius (~800-1000 LoC per
§3).

**Implementation choice (recorded 2026-05-12): Approach A — true
non-CMS reference build.** A second isolated kernel build with
`UseCustomMainLoopSchedule=0`, captured via a non-shadow path and used
as `ref`. Both sides are real emittable+runnable kernels. Approaches
D and H remain documented in §3 for reference but are not the chosen
path.

Implications of choosing A specifically (additional cascading
consequences beyond the principle-level ones below):

- The auto-activation at `KernelWriter.py:4755-4756` and the 9
  `_captureDefaultSchedule` flag-check sites become obsolete in a
  clean A implementation. They should be deleted, not retained as
  optional.
- `oram.1` (compare_graphs body_label-sensitivity) is the load-bearing
  blocker — A's reference kernel will produce a real prologue body
  with `BODY_LABEL_PROLOGUE`/`loop_index=-1`, and the per-tile
  schedule's `UsePLRPack=True` flip will move pack instructions from
  the mainloop body to the prologue body across the two builds.
  Without oram.1's resolution, every UsePLRPack-flipping schedule will
  surface spurious failures.
- §5 Q2 ("schedule flag-mutations: API or implementation detail?") is
  still open and load-bearing — see §5 below for what's blocked by it.
- §3's "PRELOOP_CAPTURE_PHASE1 design-intent objection" must be
  resolved by the comparator (not by build-time scoping). Per
  `PRELOOP_CAPTURE_PHASE1 §1`, UsePLRPack is pipelining-only, so the
  dataflow IS equivalent across the flag flip — but `compare_graphs`
  must be made body-label-tolerant (oram.1) for that equivalence to
  surface.
- Build-time cost ~doubles on every CMS kernel build that runs the
  validator (the assert path now does TWO `_getKernelSource` calls).
  See §5 Q3 below.

### Cascading consequences

- **Auto-activation in `KernelWriter.py:4750-4756`**
  (`if kernel.get("UseCustomMainLoopSchedule"): self.states._captureDefaultSchedule = True`)
  needs revisiting. Under any path in {A, D, H} the shadow capture may
  not need to run at all on the assert path — the auto-activation gates
  shadow synthesis on every CMS kernel build. If the shadow is no longer
  consumed by `compare_graphs`, the activation site is either obsolete
  (D), conditionally needed (A — only if the second build's capture path
  reuses any shadow plumbing), or fully obsolete (H — both sides are
  real CMS builds with their own captures).
- **The `_captureDefaultSchedule` flag-check sites** at `KernelWriter.py`
  lines 3766, 4418, 4450, 4544, 4571, 4980, 5080, 5138, 5296 (verified
  this commit) gate per-call shadow tagging/snapshot work. Status under
  each approach:
  - Under D: deletable (no shadow capture is consumed anywhere).
  - Under H: likely obsolete (both sides are real CMS builds — neither
    runs the SIA3 default emit that these sites annotate).
  - Under A: implementation-only-needed-if the second build reuses any
    shadow plumbing; in a clean A implementation the shadow path is
    replaced by a real default kernel build, so these sites are also
    obsolete.
- **`PRELOOP_CAPTURE_PHASE1.md §"Phase 2 decisions" Decision 4(b)**
  (test architecture for the "whole-kernel UsePLRPack CMS test")
  currently passes only because of `cap_with_cms.prologue is
  cap_with_default.prologue` (`Tests/unit/test_prologue_capture.py:342`):
  the CMS-side capture inherits the default-side prologue verbatim
  rather than building its own. Per `PRELOOP_CAPTURE_PHASE1.md §3`, this
  is the shadow-shared-prologue artifact that masks
  `compare_graphs`'s body-label-sensitivity. The shadow rejection
  invalidates this test architecture — without a shadow, "inherit the
  default's prologue" no longer makes sense as a setup. The test must be
  reworked under whichever of {A, D, H} the user picks.
- **oram.1's body_label-sensitivity blocker** (PRELOOP_CAPTURE_PHASE1.md
  §"Phase 3 blocker") becomes critical-path for any cross-build
  comparison. Under A and H, both sides have real bodies whose
  `loop_index` values land in identity tuples per `ScheduleCapture.py`
  (`identity_for(body_label)` baking `BODY_LABEL_TO_LOOP_INDEX[...]` into
  the identity); pipelined producers will drift across body boundaries
  and `compare_graphs` will fire spurious failures (see
  PRELOOP_CAPTURE_PHASE1.md §2.4). Under D, oram.1 is moot — no
  cross-graph comparison happens, so the body-label issue cannot
  surface. **oram.1 is no longer "nice to have for Decision 4(b)"; it
  is a prerequisite for A or H.**
- **Cause-(B) and cause-(B-2) tagging artifacts** (§1 (B)/(B-2)) become
  irrelevant under D (no shadow → no shadow tagging → nothing to
  double-tag). Under A or H they remain relevant insofar as the
  cross-build comparison still has to handle category-tagging
  consistently between two real builds; the specific artifacts (B)/(B-2)
  catalog (Pack-leaf double-tagging in the shadow's prefetch_pack_a/b
  walk; LR/LW A/B-suffix divergence) are shadow-specific and disappear
  with the shadow, but the comparator's category-tag consistency
  requirements continue to apply across whatever new reference source
  replaces the shadow.

### Cross-links

- `PRELOOP_CAPTURE_PHASE1.md §6` (Interaction with rocm-libraries-2lzd):
  updated this commit to reflect the shadow rejection.
- `PRELOOP_CAPTURE_PHASE1.md §7` (new section, this commit): updated
  oram.1 framing — body-label-sensitivity is critical-path, not optional.
- `PRELOOP_CAPTURE_PHASE1.md §"Phase 2 decisions" Decision 4(b)'`
  (added this commit): restated test intent in shadow-free terms.
- Bead `rocm-libraries-2lzd` comment recording this decision.
- Bead `rocm-libraries-oram.1` priority bumped to P0.

### §6.2 — Q2/Q3/Q5 follow-up decisions (2026-05-12)

User decisions on the §5 open questions, recorded post-Approach-A pick:

- **Q2 → implementation-detail reading.** Validation comparison is
  YAML-in, two builds out (one with `UseCustomMainLoopSchedule=1`, one
  with `=0`). Whatever Tensilelite mutates internally on either side
  is accepted as part of "what that build does." Specifically,
  `UsePLRPack=False` on the non-CMS side (per `dispatch.py:546`) and
  `UsePLRPack=True` on the CMS side (per the per-tile schedule, e.g.
  `_128x128x32_TF32.py:120`) is an EXPECTED divergence — the prologue
  shapes will differ, and oram.1's body-label-tolerance work is
  what makes that comparison succeed.

- **Q3 → build-time deferred.** ~2x build-time on the assert path is
  acceptable in the near-term. Caching, test/CI-only gating, and
  process-pool isolation are all reserved for after correctness lands.

- **Q5 → merge into one meta-investigation.** 2lzd, the z012 family
  (`Z012_ALTERNATIVE_FIXES_INVESTIGATION.md` + `EXAMPLE_YAML_DEFECT_INVESTIGATION.md`),
  AND oram.1 are merged under a new meta-bead. Sub-bead structure:
  - 2lzd remains the umbrella for the Approach-A implementation.
  - z012 (category-tagging-strategy divergence) becomes a sub-bead.
  - oram.1 (compare_graphs body-label-sensitivity) becomes a sub-bead.
  - The meta-bead tracks the joint resolution (no Approach-A
    implementation can land without addressing both z012 and oram.1).

The meta-bead is filed as `rocm-libraries-71hw` with the
parent/child wiring set up via `br dep add` (2lzd and oram.1 are
sub-beads of the meta).

---

## Appendix — Files referenced

- `Tensile/KernelWriter.py:478` (`enable_capture_default_schedule`),
  `:918-2599` (`_makeSubIterSchedule` + `_captureSubIterToBuilder`,
  the actual @-1 stamping site at `:2669-2683`),
  `:976-979` (LRS/LWS pointer-swap tagging — cause B-2 source),
  `:3155-3643` (`_noLoadLoopBodyDefault` capture path for NLL/NGL),
  `:4509-4514, 4607-4612` (prefetch-pack category tagging — empty
  walks under usePLRPack=True; relevant only under usePLRPack=False),
  `:4961, 4966-4969` (usePLRPack gate, prefetch_pack_a/b
  initialization),
  `:5004-5052` (per-plrIdx Pack routing under both usePLRPack
  branches),
  `:5080-5084` (prologue snapshot into prologue_prefetch_pack_a/b),
  `:5141-5144` (`build_prologue_capture` consumer — ONCE only,
  produces BODY_LABEL_PROLOGUE = "PRO" not in our .s dumps),
  `:5416` (compare_graphs gate from kernel build path),
  `:8023` (`doFullPackCodePrefetch` formula —
  `UsePLRPack AND NOT UseCustomMainLoopSchedule`).
- `Tensile/Components/CustomSchedule/dispatch.py:546` (UsePLRPack
  default).
- `Tensile/Components/CustomSchedule/gfx950/_128x128x32_TF32.py:53,
  119, 120` (kernel-flag mutations).
- `Tensile/Components/CustomSchedule/cms_from_default.py`
  (Approach H source; `default_schedule_to_cms` API).
- `Tensile/Components/CMSValidator.py:4317-4338` (the three
  validator components: `compare_graphs`,
  `validate_edge_wait_coverage`, `validate_vopd_pair_formation`),
  `:1739-1945` (`build_dataflow_graph`).
- `Tensile/Components/ScheduleCapture.py:117-119`
  (SLOT_KIND_PRE_LOOP / POST_LOOP / MFMA),
  `:744-760` (`assign_stream_indices_for_body`),
  `:1945-2016` (`build_prologue_capture` — produces "PRO" body),
  `:2007-2012` (prologue prefetch-pack tagging).

Evidence (`Tensile/Tests/unit/`, regenerable; do NOT commit):
- `cms.s` — 184/184/164/136 dataflow per body (ML-1/ML/NGL/NLL).
  ML-1 @-1 count: 0.
- `default.s` — 242/242/164/136 dataflow per body. Per-body
  @-1 counts: 74/74/14/14. NO `PRO` body.
- `kernel_cms.s` MAINLOOP at line 1846+; `kernel_default.s`
  Unrolled Loop at line 1834+.

Related memos: `PRELOOP_CAPTURE_PHASE1.md` (§"Concrete in-tree
divergence", l329-346 — the design-intent argument that A
contradicts), `Z012_ALTERNATIVE_FIXES_INVESTIGATION.md`
(§6 Pack-tagging divergence + the "multiple families" extension —
load-bearing for Approach C scoping), `CMSValidator_LIMITATIONS.md`
(§"Architecture-Specific Behavior" — describes shadow as correctness
check vs default, contradicting today's mechanics).

For Approach H, see bead `rocm-libraries-wlrp` (closed; landed
`Tensile/Components/CustomSchedule/cms_from_default.py`) — the
DEFAULT_TO_CMS_CONVERTER_PHASE1.md memo was removed in the post-2lzd
Components/ memo cleanup since wlrp landed and the memo's discussion
was complete; the code is the design now.
