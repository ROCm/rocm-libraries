# MMA Is Machinery

> **Audience: any agent (or human) reasoning about MMA/MFMA/WMMA operand or accumulator layouts.**
> Read this BEFORE you argue that a layout "won't work" with mma hardware or you reach for default "canonical" layouts in designing mma workflows. If you find
> yourself saying *"the hardware needs the canonical layout"* or *"data layouts for mma are fixed"* — stop. Yes, mma hardware is mechanically fixed but the *logical* data we flow into it *can* depart from canonical ordering giving us access to more complicated layouts.

## Evolution of MMA
Matrix multiplication is a classical branch of mathematics:

$$C_{ij} = \sum_{k} A_{ik}\,B_{kj}$$

This basically states that: each scalar output element $C_{ij}$ is a dot-product result of vectors $A_i$ and $B_j$.

A few important properties to remember:
* The $[i, j]$ matrix coordinates of the $C$ elements are logically tied to $[i]$ and $[j]$ coordinates of respective $A$ and $B$ operand vectors as the "free" dimensions.
* The $[k]$ matrix coordinate of the $A$ and $B$ operand vectors are the "fixed" or "contraction" dimensions, which *must* match in $A$ and $B$, as per the dot-product formula.
* The equation is *separable* and *parallelizable* meaning that we can calculate each $C_{ij}$ from dot-products *independently*.
* The dot product's "fixed" dimension $[k]$ strictly *localizes* which element of $A$ must be multiplied with which element of $B$ (dot product), but does *not* constrain in which order they are accumulated (commutative):

$$(A_{i0}\,B_{0j}) + (A_{i1}\,B_{1j}) + \dots \;=\; (A_{i1}\,B_{1j}) + (A_{i0}\,B_{0j}) + \dots$$

In classical computing, hardware has evolved from doing thread-wise parallelism of manual dot-products to vectorized dot-product intrinsics and now to tiled mma intrinsics optimizations. The properties above *still apply*.

## The one thing to internalize

**An MMA instruction drives fixed machinery: it parallelizes dot products in individual threads.** 
In general per mma instruction call, each thread will accumulate one partial dot product of $C_{ij} = \sum_{k} A_{ik}\,B_{kj} + \text{accum}$, for some vector size $K$, (which varies between instructions).
What this means:
* Mma instructions have *physical* routing in thread/register coordinates. For example in physical coordinates $[\text{thread}, \text{register}]$, the dot product machinery implements $C[0,0] = A[0,0]\,B[0,0] + A[0,1]\,B[0,1] + \dots + \text{accum}$. This is *fixed*, and is physical hardware. It is a tool we use to implement the separate higher-level logical equation.
* As a matter of *logical* data assignments to each thread, we get to pick (free coords) which $C_{ij}$ gets assigned to each thread because we saw before $C_{ij}$ is parallelizable. For example, we might assign logical coords $[i,j] = [5,2]$ to thread 0. Thus follows logically that an mma instruction will calculate a partial dot-product $C[5,2] = A_{5,k_0}\,B_{k_0,2} + A_{5,k_1}\,B_{k_1,2} + \dots + \text{accum}[5,2]$ in thread 0.
* It also stands that logical $[k]$ coordinate pairs in $A$ and $B$ vectors (and accum) *must* match for correctness, as we saw before.
* Logical $[k]$ summation order in the dot-product accumulation is *not* fixed. For example, we can adjust the summation order of $k$ as we wish, such as $[k] = 0, 3, 1, 2, \dots$ as long as $A$ and $B$ $[k]$ share the same ordering (fixed) e.g., $C[i,j] = A_{i,0}\,B_{0,j} + A_{i,3}\,B_{3,j} + A_{i,1}\,B_{1,j} + A_{i,2}\,B_{2,j} + \text{accum}[i,j]$
* As a teaching tool, diagrams demonstrating the mma instruction data flows generally apply "canonical" logical data layouts to mma registers such that they map 1:1 to the hardware routing. This facilitates teaching and is easier to understand, however it often imparts an erroneous assumption that this is a requirement. This document's purpose asserts that it is *not* a requirement to have canonical logical layouts when using mma hardware. Remember we can leverage the properties of separable parallelism, which enables flexible logical assignments of $[i,j]$ to any thread, and order $[k]$ in any way we like - as long as $[k]$ matches in $A$ and $B$ multiplications (as the equation states).
* The challenging part is adjusting the mapping of *logical* coordinates to the *physical* routing in hardware to have meaningful layouts. These in turn can have an effect on memory ordering and coalescing which have a broader impact on performance.
* Logical labels on physical routing defines the *flow* of data through an mma instruction, and can track meaningful localization of data. Mechanical "canonical" routing of data may be fixed in hardware but we may permute logical data assignment to optimize our layouts (e.g., interleaved mma wave-tiles).

## The invariant: a sound multiply-accumulate

Start from what is *physical*. The atom hardwires **which slots accumulate into which output**: for each output register it pairs **A-slot-`s` with B-slot-`s`**, multiplies, and sums over the paired slots. That wiring is silicon — you cannot change which products land in one accumulator. The machine reads slot `s` and nothing more; it has no opinion about layout and does not know what $M$, $N$, or $K$ a register holds.

Everything else follows from that one physical fact. Because the atom *physically* sums a fixed set of slots into each output, the labels you place on those slots must be coherent or the result is not a dot product at all. A **sound** multiply-accumulate is three conditions:

1. **Atom-fixed accumulation.** Which physical slots sum into each output is fixed by the atom — not a choice.
2. **One $M$ per output on A, one $N$ per output on B.** Every A-slot feeding a given output carries a single fixed $M$; every B-slot a single fixed $N$. If $M$ drifts across those slots, the accumulator holds a sum whose row wanders term to term — not $C_{ij}$ for any $(i,j)$.
3. **$K$-match, position for position.** $A.K\text{-dist} = B.K\text{-dist}$: $A$ and $B$ agree on which contraction index $K$ sits in each paired slot, and $K$ is well-formed (each index present once).

> The slots feeding one accumulator are fixed by silicon; a sound MAC labels them with **one $M$, one $N$, and a matched $K$** — nothing else is constrained.

This is where "$M$ and $N$ are free" is easy to over-read. Free means free to **choose the constant** — which $M$ this A-fragment carries, which $N$ this B-fragment carries — *never* free to vary $M$ or $N$ across the slots the atom ties into one accumulation. You pick the label; the physical wiring dictates that it be held constant.

And note what stays *un*constrained: canonical $M$ order, canonical $N$ order, a particular register grouping — none are requirements, they are choices. Conditions 1–3 constrain the *labels*, never the *ordering*. The incompatible case is a violated MAC — a wandering $M$/$N$, or a $K$-mismatch — never "interleaved vs canonical."

## Following the inputs — a worked derivation

The physical coupling is fixed, but the **label** on each slot is whatever you loaded there. So reading a result off the machine means running the fixed coupling and carrying the *supplied* labels through it:

```
canonical inputs                    ->  fixed coupling  ->  canonical outputs
interleaved (non-canonical) inputs  ->  SAME coupling   ->  non-canonical outputs
```

Worked example (single $K$-pair):

```
phys A(reg0,tid0) = M5·K0      (your interleaved A placed M5 here)
phys B(reg0,tid0) = N11·K0     (your interleaved B placed N11 here)
        -> fixed coupling ->
phys C(reg0,tid0) = M5 · N11  =  C[M5, N11]
```

That physical slot is the one the canonical machine calls `M0N0`; physically nothing moved — the label is derived from what flowed in.

> **Run the canonical machine on physical slots, but carry the SUPPLIED logical labels through it.**
> For each physical C slot whose canonical identity is $(M_c, N_c)$, relabel it with *(the $M$ your A put on that slot's A-row, the $N$ your B put on its B-column)*.

Two equivalent readings, both useful when visualizing:

- **Fixed-physical:** the C layout *stays* the canonical machine layout; only the $(M,N)$ labels move onto it.
- **Fixed-logical:** a result like $C[M5,N11]$ lands wherever the machine canonically calls $(M_c,N_c)$ — it moved *off* canonical $(M5,N11)$.

This is a clean relabel because a register shuffle is a **permutation** of the free dimension — pure encoding arithmetic, no hardware re-derivation. It earns its keep in the interleaved case: you cannot eyeball where an input lands, so you trace it through the fixed coupling.

## C's layout is DERIVED, never dictated

Because A's $M$-placement and B's $N$-placement are free, *they* fix C's native layout. You do not pick C independently and you do not "make C canonical" — you choose A/B and C falls out. A wide, coalesced C store is *produced* by placing $M/N$ in A/B, not by fighting a canonical accumulator into shape (that direction is often cross-lane and expensive).

## MMA-acceptability: per-operand soundness + a pairwise $K$-match

Two checks hide under "is this layout OK?", and conflating them is what sends people reaching for canonical:

- **Per-operand soundness** (conditions 1–2) is checkable on *one* fragment: does every slot feeding an output carry a single $M$ (A) or $N$ (B)? It asks nothing about the partner, and nothing about canonical.
- **Pairwise acceptability** (condition 3) is a property of the **pair**, never of one fragment: do A and B agree on $K$, slot for slot? Neither operand answers it alone — and neither is checked against a canonical reference.

Consequences:

- "Make this fragment acceptable" always means "reorder it until its $K$-row matches its **partner's**" — never "until it looks canonical."
- `validate_operands(A, B)` compares the two fragments **to each other**; it never checks either against a canonical reference.

## The traps agents fall into (name them, then don't)

Wrong-turns that come from pattern-matching on training data instead of reasoning from the machinery:

1. **"The MFMA has a fixed layout, so the fragment must be canonical."** — No. The hardware reads slots and multiply-accumulates; the layout is bookkeeping *we* maintain. $K$-pairing is the only **cross-operand** constraint — not canonical resemblance.
2. **"A lane can only own these $N$ strided elements, so we're capped at dwordx2."** — Reasoning from *one* canonical distribution. $M/N$ placement is free; choose a distribution where the lane owns a contiguous run and the wide store exists. "Canonical caps us" is never a valid conclusion.
3. **"Just shuffle the canonical C into the store order."** — C's ownership is derived from A/B. The store-friendly C is *produced* by choosing A/B placement, not recovered by post-hoc shuffling a canonical accumulator (often cross-lane and expensive).
4. **Treating `validate_operands` passing as "canonical-valid."** — It only means A and B agree on $K$. A hand-rolled distribution's correctness is confirmed by **bit-exact**, not by resemblance to canonical.
5. **Fighting the author on layout direction.** — If the author says "use this interleaved layout," the question is never "but canonical…". It is only: *does $K$ match, and is it bit-exact?*

## The decision procedure (use this instead of defaulting to canonical)

For any layout/direction question about an MMA operand or accumulator:

1. **Is this a correctness question?** Correctness = the sound MAC: per-operand $M$/$N$ fixed (1–2) **and** $A.K\text{-dist} = B.K\text{-dist}$ (3). If those hold, the MMA is correct — full stop.
2. **Place $M/N$ for the goal, not for canon.** Choose A's $M$-placement and B's $N$-placement for the memory behavior you want (coalesced wide load, store-friendly C, feed a downstream MMA).
3. **Let C be derived.** C emerges at the $M/N$ you chose; shape the *store* around that (or a cheap register reorder *within* the already-derived ownership), not by reshaping a canonical C.
4. **Verify by bit-exact.** The GPU is the arbiter. "It doesn't look canonical" is not evidence of anything.

## One-line summary

**MMA multiplies $A \times B$ and sums over $K$. Match $K$ between A and B; place $M/N$ however you like (one constant per output); C is derived; verify with bit-exact. Canonical is a baseline, never a requirement — stop defaulting to it.**

**POSITION ≠ LABEL here is the atom-local case of a pipeline-wide law:** a datum's label is its identity and
flows INVARIANT across every space (global → LDS → registers → MMA → epilogue → store); a transform changes
*where* a datum lives, never *what* it is; the only label change is an explicit **relabel** (e.g. AB-swap:
swap A↔B / M↔N on the inputs, C DERIVES). See **`label_flow_and_transforms.md`** for the full model + the four
edge kinds (reposition / reorder / cross_lane / relabel) and their enforcement.

---

See also: `tiling_interleaving_design.md` (§2 hardware contract, §3 fragments-vs-acceptability, §7
derived-C) for the full interleaved-layout spec and the closed-form transforms.
