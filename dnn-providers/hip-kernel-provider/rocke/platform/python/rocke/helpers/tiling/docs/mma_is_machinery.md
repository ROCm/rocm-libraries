# MMA Is Machinery — the mental model agents keep getting wrong

> **Audience: any agent (or human) reasoning about MMA/MFMA/WMMA operand or accumulator layouts.**
> Read this BEFORE you argue that a layout "won't work" or reach for "canonical." If you find
> yourself saying *"the hardware needs the canonical layout"* or *"a lane can only own X, so we're
> capped at dwordxN"* — stop. You are making the mistake this document exists to kill.

## The one thing to internalize

**An MMA instruction is dumb machinery. It multiply-accumulates.** For each output it pairs
A-slot-`s` with B-slot-`s`, forms the product, and sums over the paired slots. That is *all* it does.
It has **no opinion about layout**. It does not "require canonical." It does not know what M or N a
register holds — it reads slot `s`, multiplies, accumulates.

Because the sum is order-independent, the hardware imposes exactly **one** constraint:

> **`A.K-dist == B.K-dist`, position for position.** A and B must agree on which contraction index K
> sits in each paired slot. Nothing else.

Everything else people treat as a "hardware requirement" — canonical M order, canonical N order, a
particular register grouping — is **not**. It is a choice.

## M and N are FREE. They route the output; they do not constrain correctness.

The M coordinate that A carries in a slot, and the N coordinate B carries in a slot, only decide
**where the product lands in C**:

```
(A) M5·K0   ×   (B) K0·N90   →   C[M5, N90]
```

M5 and N90 are "funky," and that is completely fine — they just select the output cell. Put M and N
wherever it serves your goal (coalesced wide loads, a store-friendly C, a downstream consumer). As
long as **K matches between A and B**, the math is correct and C simply appears at the M/N you chose.

**Corollary — C's layout is DERIVED, never dictated.** Because A's M-placement and B's N-placement are
free, *they* determine C's native layout. You do not get to pick C independently, and you do not
"make C canonical." You choose A/B, and C falls out. If you want a wide, coalesced C store, you get
it by **placing M/N in A/B**, not by fighting a canonical accumulator into shape.

## The machine mapping is a fixed PHYSICAL coupling — "following" the inputs

The multiply-accumulate is a **physical** wiring, fixed by the atom and identical for every problem:
physical `A(reg,tid)` and physical `B(reg,tid)` are paired and summed into physical `C(reg,tid)`. That
coupling **never changes** — it *is* the machinery. It is written down in *canonical* terms only
because that is the convenient way to describe it, not because it privileges canonical data.

What changes between problems is **which logical label you loaded into each physical slot**:

```
canonical inputs                    ->  fixed machine coupling  ->  canonical outputs
non-canonical (interleaved) inputs  ->  SAME coupling           ->  non-canonical outputs
```

Worked example (single K-pair, for clarity):

```
phys A(reg0,tid0) = M5·K0          (your interleaved A placed M5 here)
phys B(reg0,tid0) = N11·K0         (your interleaved B placed N11 here)
        -> fixed coupling ->
phys C(reg0,tid0) = M5 · N11  =  C[M5, N11]
```

That physical slot is the one the canonical machine calls `M0N0`. Physically nothing moved; the
**label** on the slot is derived from what flowed in. This is exactly what "derivation" / "following
the inputs" means:

> **Run the canonical machine on physical slots, but carry the SUPPLIED logical labels through it.**
> For each physical C slot whose canonical identity is `(Mc,Nc)`, relabel it with *(the M your actual A
> put on the A-row coupled to that slot, the N your actual B put on the B-column coupled to it)*.

Two equivalent readings of the same fact — both useful when visualizing:

- **Fixed-physical:** the C physical layout *stays* the canonical machine layout; only the `(M,N)`
  labels move onto it.
- **Fixed-logical:** a given result like `C[M5,N11]` lands wherever the machine canonically calls
  `(Mc,Nc)` — i.e. it moved *off* canonical `(M5,N11)`.

This is a clean relabel because a register shuffle is a **permutation** of the free dimension (each
canonical row maps to exactly one actual row), so it composes as pure encoding arithmetic — no
hardware re-derivation. And it is precisely the interleaved (non-canonical A/B) case where "following"
earns its keep: you cannot eyeball where an input lands, so you trace it through the fixed coupling.

## "Canonical" is not special

Canonical is just **one** matched `(A, B)` K-pair — a convenient teaching baseline for the K
invariant. It is not privileged, not required, and not a target. Interleaved-A × interleaved-B (with
matched K) is exactly as valid. The incompatible case is a **K mismatch** between the two operands —
never "interleaved vs canonical."

## MMA-acceptability is a RELATIONSHIP, not a property of one fragment

There is no such thing as "this fragment is MMA-ready" in isolation. Acceptability is a property of
the **pair**: do A and B agree on K, slot for slot? So:

- "Make this fragment acceptable" always means "reorder it until its K-row matches its **partner's**"
  — never "reorder it until it looks canonical."
- `validate_operands(A, B)` compares the two fragments to each other. It never checks either against a
  canonical reference.

## The traps agents fall into (name them, then don't)

These are the specific wrong-turns that come from pattern-matching on training data instead of
reasoning from the machinery:

1. **"The MFMA has a fixed layout, so the fragment must be canonical."** — No. The hardware reads
   slots and multiply-accumulates. The layout is bookkeeping *we* maintain; the hardware is
   layout-blind. Only K-pairing matters.
2. **"A lane can only own these N strided elements, so we're capped at dwordx2."** — That is reasoning
   from *one canonical distribution*. M/N placement is free; choose a distribution where the lane owns
   a contiguous run, and the wide store exists. "Canonical caps us" is never a valid conclusion.
3. **"Just shuffle the canonical C into the store order."** — C's ownership is derived from A/B. The
   store-friendly C is *produced* by choosing A/B placement, not recovered by post-hoc shuffling a
   canonical accumulator (that direction is often cross-lane and expensive).
4. **Treating `validate_operands` passing as "canonical-valid."** — It only means A and B agree on K.
   Correctness of a *hand-rolled* distribution is confirmed by **bit-exact**, not by resemblance to
   canonical.
5. **Fighting the author on layout direction.** — If the author says "use this interleaved layout,"
   the question is never "but canonical…". The question is only: *does K match, and is it bit-exact?*

## The decision procedure (use this instead of defaulting to canonical)

When facing any layout/direction question about an MMA operand or accumulator:

1. **Is this a correctness question?** Correctness depends on ONE thing: `A.K-dist == B.K-dist`.
   If K matches, the MMA is correct — full stop.
2. **Place M/N for the goal, not for canon.** Choose A's M-placement and B's N-placement to get the
   memory behavior you want (coalesced wide load, store-friendly C, feed a downstream MMA).
3. **Let C be derived.** C emerges at the M/N you chose; shape the *store* around that (or a cheap
   Step-4 register reorder *within* the already-derived ownership), not by reshaping a canonical C.
4. **Verify by bit-exact.** The GPU is the arbiter. "It doesn't look canonical" is not evidence of
   anything.

## One-line summary

**MMA multiplies A×B and sums over K. Match K between A and B; place M/N however you like; C is
derived; verify with bit-exact. Canonical is a baseline, never a requirement — stop defaulting to it.**

---

See also: `tiling_interleaving_design.md` (§2 hardware contract, §3 fragments-vs-acceptability, §7
derived-C) for the full interleaved-layout spec and the closed-form transforms.
