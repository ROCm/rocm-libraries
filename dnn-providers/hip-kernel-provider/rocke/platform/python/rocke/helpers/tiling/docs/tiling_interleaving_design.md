# Interleaved Layouts — Design (SOT)

## TL;DR — the point (read this FIRST)

**The whole exercise is the full interleaved pipeline:** interleaved (free-dim-contiguous) A/B reads →
the MMA **produces** a store-friendly interleaved C → **wide coalesced store**. Canonical is a teaching
baseline for the K-invariant (§2), **NOT** a path to wide stores. If you are reasoning about a layout
and reaching for "canonical," stop — the target state is always interleaved.

**Load-bearing consequence (see §7):** the interleaved C ownership is *cross-lane* from canonical, so
you **cannot** reach the wide-store layout by shuffling a canonical accumulator. You get interleaved C
**only by feeding interleaved A/B**. "Just add a C-shuffle to the canonical kernel" is the classic
wrong turn — the interleaved A/B reads come first and are what make the wide store exist.

**If you are reasoning from "the MFMA needs canonical," STOP and read
[`mma_is_machinery.md`](./mma_is_machinery.md) first.** MMA is dumb multiply-accumulate machinery; correctness is the
**sound MAC** — per-operand M/N-fixed *plus* `A.K-dist == B.K-dist` positionally — and M/N placement is
free (you choose the constant, which merely routes the output). That doc is the **SOT for layout
correctness**; it exists specifically to kill the canonical-default mental model that repeatedly derails
this work.

Source-of-truth specification for interleaved MMA operand/accumulator layouts. Formulas are verified
against the interleave reference tables (MFMA/WMMA, CDNA + RDNA).

**Scope — what this doc is self-sufficient for:** the **MmaDim-16 A/B and C `reorder` tier** — intra-lane
register permutations (§8). Everything needed to implement that tier is here. The **cross-lane
extensions** (`gather>1` grouped interleaves, the 32×32-accumulator `flip & zip`, f64 `UnpackLoHi`, RDNA
`replicate`) are **out of scope**: they are named where they arise, but their mechanics live in the
reference tables, not this doc.

**Naming.** This layer uses two descriptive labels for the register order, each naming exactly which
axis the vector width runs along — no "structure" to interpret, no other layout vocabulary needed:
- **K-contiguous** — the vector width runs along K (K varies fastest within a lane's registers). The
  MMA-native input order.
- **free-dim-contiguous** — the vector width runs along the operand's non-K dim (M for A, N for B).
  A coalesced load-staging order.

## Vocabulary

Defined once here so the rest reads cold:

- **wave / lane** — a *wave* is the 64 threads that execute one MFMA together (wave64 on CDNA); a *lane*
  is one thread. Layouts are wave-wide.
- **atom** — the smallest hardware MMA instruction shape, e.g. a `16×16×16` MFMA. A wave *tile* is a
  grid of atoms.
- **fragment** — the data a wave holds in registers for one operand (A, B, or C), together with its
  layout.
- **MmaDim** — the atom's M and N size (16 here; "MmaDim-16" = 16×16 atoms).
- **VW (vector width)** — elements per contiguous memory access; also the accumulator vector width
  `ACC_VW` (4 for 16-bit C).
- **m_iter / n_iter / k_iter** — how many atoms the wave tile spans in M / N / K =
  `wave_tile_dim ÷ atom_dim`. (These are the `MmaA` / `MmaB` / K-atom counts in the reference tables.)
- **k_ab_per_lane** — K elements a lane holds per atom for A/B (4 for the f16 16×16×16 atom).
- **DPT / KPT** — per-lane free-dim / K element counts (§6): `DPT = m_iter`(A)/`n_iter`(B),
  `KPT = k_ab_per_lane × k_iter`.
- **K-dist / M-dist** — the map `(lane, register-slot) → K coordinate` / `→ M coordinate`: how an axis
  is spread across the wave's lanes *and* registers (§6, "Across the wave"). Two operands are K-aligned
  iff their K-dists are equal position-for-position.
- **WarpDistributionEncoding** — the library value that records a fragment's layout (which
  `(lane, register)` holds which logical coordinate). A `transform` is the *delta* between two of these (§9).
  The `reorder`/`cross_lane` **tiers priced here** are two of the four pipeline edge kinds; the general law
  (a label is INVARIANT identity; a transform moves destination coords, not the label; the only label change
  is an explicit **relabel**) lives in **`label_flow_and_transforms.md`** — interleaving is a *consumer* of it.
- **vec_extract / vec_insert** — read / write one element of a vector register; the ops a compile-time
  register reorder emits.
- **DPP / ds_bpermute** — hardware cross-lane data-movement ops; used only by the deferred `cross_lane`
  tier (§8).
- **flip & zip / unpackLoHi** — specific cross-lane register ops used by the 32×32 / f64 accumulator
  extensions (out of scope, §7).

## 1. What interleaving is

**Definition.** An **interleaved MMA** is a choice of which logical datum — A's `(M,K)`, B's `(N,K)`, C's
*derived* `(M,N)` — rides each physical `(lane, register)` slot of the fixed atom machine, such that **each
lane owns one or more contiguous rectangular patches that share the same re-arrangement**. It trades the
machine's native (**canonical**) placement for one whose per-lane register order serves **coalesced wide
loads/stores** and **cross-stage symmetry** (A=Bᵀ M↔N, col↔row). The silicon is unchanged — POSITION ≠ LABEL
(`mma_is_machinery.md`); only the label→slot assignment differs. It is a *valid* MMA exactly when the **sound
MAC** holds (per-operand M/N fixed + `A.K-dist == B.K-dist`) — interleaving never touches correctness.

- **Multiple patches, one rule.** A lane may own several patches, and they need **not** be one-per-atom nor
  logically adjacent — patch placement is **independent of atom proximity**. What ties them together is that
  each carries the *same* re-arrangement: a **block-structured** transpose applied identically to every patch
  (the `interleave_idx` Count = one patch, repeated across the lane's patches). A single patch is the simple
  case.
- **The degenerate case is canonical.** A lone `1×K` patch (`DPT==1`) has nothing to transpose → the interleave
  is the identity → the layout **is** canonical (label == native position).
- **Three facets of the one choice.** The per-lane *register order* (K-contiguous vs free-dim-contiguous, below)
  is the **A/B-input facet**; **derived-C ownership + the C-shuffle** (§7) is the **C facet**; a cross-atom
  lane-ownership bridge (§8) is the **free-symmetry facet** (reposition + routing). All three are the same "which label rides which
  slot" choice at different stages.

The rest of this section develops the A/B-input facet (the per-lane register order). An MMA operand fragment is
the data a wave holds in registers. Each lane's share is a small 2D tile of `(free-dim × K)` elements, and
**this facet of interleaving is just the order those elements sit in the lane's registers** — i.e. which axis
the vector width walks first. The *same* tile, read two ways:

```
lane's 2D tile          K-contiguous                   free-dim-contiguous
(rows = free dim,       (VW along K — walk each row)    (VW along free — walk each column)
 cols = K)              a b c  d e f  g h i             a d g  b e h  c f i
[[a b c]
 [d e f]
 [g h i]]
```

Both orders are the **same elements owned by the same lane** — only the register slot order differs. So
they cost the **same wide, coalesced load**; nothing about *which* data a lane holds changes.

**Why that matters — the whole point.** Because the two layouts differ only in register order, moving
between them is never a re-load and never cross-lane data movement — it is a single **in-register
re-order** (a `transform`), given in closed form by `interleave_idx` (§5). So you read whichever memory
order is naturally coalesced for the operand, then re-order registers in place into the order the next
consumer wants — wide loads either way, and the switch is nearly free.

Two clarifications carried by later sections:
- **Row/col-major is operand-relative.** K-contiguous is row-major for A but col-major for B (and
  free-dim-contiguous is the reverse). The register-order names here are the operand-neutral truth;
  row/col-major is only how each maps onto a given operand's memory.
- **A register spans the wave.** The tile above is one lane's share; read a register *across* the lanes
  and it is a structured **slice** of an axis (§6, "Across the wave"). §1 stays single-lane for clarity.

### Hallmarks of an interleaved layout (and the precondition that enables them)

**Precondition — the lane owns a set of congruent, evenly-spaced patches** (A `M×K`, B `N×K`, C `M×N`). This
is what makes interleaving possible; the two hallmarks below are its consequences. **One patch is the common
case, not the requirement** — a multi-patch accumulator qualifies (§9 → *Accumulators*). If a lane's owned
elements are not congruent and evenly spaced, neither hallmark holds (the reorder would go cross-lane).

1. **Interleaved register order ≠ canonical.** The thread-tile's elements sit across the lane's registers in a
   permuted (interleaved) order versus the canonical layout — same elements the lane owns, different register
   slots. The order is **not arbitrary**: it is the structured **transpose to the *other* vectorization axis**
   (K-contiguous ↔ free-dim-contiguous).
2. **The transpose is cheap — a pure register re-order.** Switching the patch's vectorization direction is the
   closed-form `interleave_idx<1, KPT, DPT·KPT>` permutation — the in-register **`reorder`** tier (§8), never
   cross-lane. Hallmarks 1 and 2 are the **same permutation seen two ways**.

**Checkable, not vibes.** #1: the register→coord order differs from the canonical encoding. #2:
`classify_transform(dir_a, dir_b)` returns `reorder` (not `cross_lane`) and matches the `interleave_idx`
oracle above.

**Caveat — interleaved ≠ automatically valid.** A cheap transpose does not exempt a layout from the two
validity gates: it must still pass MMA soundness (§2) AND tile-vectorization validity (§2b). "Cheaply
transposable" and "correct to load/use" are separate checks. (This cheap **in-thread** transpose is distinct
from bridging *canonical ↔ interleaved lane ownership*, which changes which lane owns which elements and can be
cross-lane — see §8.)

## 2. Hardware contract — the pairwise K-match half of the sound MAC

**Correctness SOT:** [`mma_is_machinery.md`](./mma_is_machinery.md) owns the full correctness model, the
**sound MAC** = per-operand soundness (one M per output on A, one N per output on B) *plus* the pairwise
K-match. This section derives that **pairwise K-match half**; per-operand soundness holds by construction
here (fragments are atom-derived — see the closing note).

The MFMA/WMMA hardware multiply-accumulates: it pairs A-slot-`s` with B-slot-`s`, forms the product,
and sums over K. The sum is order-independent, so **the K-slot ordering is free** — any bijection `σ`
is valid provided A holds `A[m, σ(s)]` and B holds `B[σ(s), n]` in the same slot.

- **The cross-operand constraint is `A.K-dist == B.K-dist` positionally** (the sound MAC's K-match half).
  A and B must agree on which logical K sits in each paired slot. A single MMA-native order is not required.
  (Per-operand soundness — M/N fixed per output — is the other half; see the SOT.)
- **Any A/B pair whose K-dists match positionally is a valid direct MMA** — canonical is not required.
  The incompatible case is a K-distribution **mismatch** between operands, not the interleaving itself.
- A's M-register order and B's N-register order are free; together they **determine the C output
  layout** (§7).

`validate_operands(a_layout, b_layout)` enforces exactly this: `k_distribution(a) == k_distribution(b)`
positionally. Lane ownership is guaranteed valid by construction (fragments come from atom-derived
descs), so a positional K-match is sufficient.

### Example — one contraction, 4 K-slots

Focus on the slots that produce a single output element. The hardware multiplies A-slot-`s` by
B-slot-`s` and sums; the result lands at `C[<A's M>, <B's N>]`.

**Acceptable — canonical.** K aligned, M/N plain:
```
slot:      0        1        2        3
A holds:  M0K0     M0K1     M0K2     M0K3      A.K-dist = [0,1,2,3]
B holds:  K0N0     K1N0     K2N0     K3N0      B.K-dist = [0,1,2,3]   == A.K-dist  ✓
sum = M0K0·K0N0 + M0K1·K1N0 + M0K2·K2N0 + M0K3·K3N0 = Σ_k A[M0,k]B[k,N0] = C[M0,N0]
```

**Acceptable — K reordered, but IDENTICALLY in A and B.** Order-independent sum, same result:
```
slot:      0        1        2        3
A holds:  M0K3     M0K0     M0K2     M0K1      A.K-dist = [3,0,2,1]
B holds:  K3N0     K0N0     K2N0     K1N0      B.K-dist = [3,0,2,1]   == A.K-dist  ✓
sum = M0K3·K3N0 + M0K0·K0N0 + ... = same C[M0,N0], just summed in a different order
```

**Acceptable — funky M/N.** The M on A and N on B can be any coordinates (and, for DPT>1, in any
register order); they don't have to be "canonical" — they only route where the result lands. K still
matches:
```
slot:      0        1        2        3
A holds:  M5K0     M5K1     M5K2     M5K3      A.K-dist = [0,1,2,3]
B holds:  K0N9    K1N9    K2N9    K3N9         B.K-dist = [0,1,2,3]   == A.K-dist  ✓
=> produces C[M5,N9]. Perfectly valid; the "funky" M5/N9 just select the output cell (see §7).
```

**NOT acceptable — K mismatch.** Slot pairing multiplies the wrong contraction indices:
```
slot:      0        1        2        3
A holds:  M0K0     M0K1     M0K2     M0K3      A.K-dist = [0,1,2,3]
B holds:  K1N0     K0N0     K3N0     K2N0      B.K-dist = [1,0,3,2]   != A.K-dist  ✗
sum = M0K0·K1N0 + M0K1·K0N0 + ...  -> slot 0 pairs A's K0 with B's K1. Garbage; rejected.
```

The rule in one line: **M/N may be anything (they route the output); K must match position-for-position
between A and B.**

## 2b. Tile validity — the vectorization contract (independent of MMA soundness)

A layout must pass **two INDEPENDENT validity gates** — both must hold, and they check different things:

1. **MMA soundness (§2)** — `A.K-dist == B.K-dist` positionally. A relationship *between* A and B.
2. **Tile-vectorization validity (this section)** — the chosen logical vectorization direction must not
   **contradict** the physical memory stride. A property of *one* tile against *its own* memory layout —
   it has nothing to do with the MMA.

**The rule.** A physical vector of width `VW > 1` is a single **contiguous** memory access. So widening `VW`
is only **correct** along an axis that is **both** the chosen logical vector direction **and** physically
**stride-1**. If that axis is physically **strided**, `VW` is forced to **1** (still correct, but a
bandwidth hit) — a wider access would pull the wrong (strided) elements. This is a **correctness** gate, not
merely a cost, and it is **orthogonal** to the MMA K-match gate above.

**Per-tile summary** — which logical vectorization is valid-and-wide, given which axis is physically stride-1:

| tile | axes (role) | physical stride-1 axis | vectorize logically along | valid VW | vectorize the *strided* axis instead |
|------|-------------|------------------------|---------------------------|----------|--------------------------------------|
| **A** | M free, K contract | **K** | K (contraction) | 1,2,4,… ≤ K patch extent | forced **VW=1** |
| **A** | " | **M** | M (free) | 1,2,4,… ≤ M patch extent | forced **VW=1** |
| **B** | N free, K contract | **K** | K (contraction) | 1,2,4,… ≤ K patch extent | forced **VW=1** |
| **B** | " | **N** | N (free) | 1,2,4,… ≤ N patch extent | forced **VW=1** |
| **C** | M free, N free (output) | **N** | N | 1,2,4,… ≤ N patch extent | forced **VW=1** |
| **C** | " | **M** | M | 1,2,4,… ≤ M patch extent | forced **VW=1** |

**Strategy.** Choose the logical vectorization that lands on the physically-contiguous axis (wide + correct),
then pay the cheap interleaved in-register shuffle (§6) to reach the order the consumer needs. **Never**
vectorize wide along a strided axis, and **never** propose a layout whose logical and physical vectorization
contradict.

### Computing the width — the mechanical form of the rule

The table above is the GEMM instance. The general procedure, for any tile in any kernel: take the
**innermost register bucket** (the last entry of the descriptor's register-major list) and ask whether the
axis it maps to is **stride-1 in THAT STAGE'S view**. If yes, the width is that bucket's extent in elements,
capped by the registered ceiling; if no, the width is one element.

**UNITS — state them at every occurrence.** This conversion has been written wrong repeatedly, always by
comparing two quantities in different units:

```
ceiling_elems = max_vector_load_dwords(dtype) * 4 // sizeof_bytes(dtype)
width         = min(bucket_extent_elems, ceiling_elems)
```

The bucket extent is in **ELEMENTS**. The registered ceiling is in **DWORDS**. The dtype appears **exactly
once**, in that conversion — not zero times (capping an element count directly by a dword count halves a
16-bit type) and not twice (applying it again divides by the pack factor a second time). Report the result in
**elements AND bytes**, never as a bare number.

> Today that registered call **accepts a dtype argument and ignores it** — the ceiling is expressed in dwords,
> and a dword ceiling is dtype-independent by construction. Do not read the signature as evidence that the
> ceiling varies by dtype; execute it. (Tracked as an API defect — the parameter should not exist.)

- **Keyed per `(stage, view, dtype)`.** A pure **re-tag** — same registers, new view — changes the answer with
  zero data movement. Width is not a property of a fragment alone.
- **Degenerate case:** a fragment with no register buckets (`register_count == 1`) has width 1.
- **Never substitute a literal** when the ceiling is not registered for that target — report it as
  unavailable and say what is missing.

> **The ceiling check.** A sweep ranks the space that was authored; it cannot see a cap that every candidate
> shares. Check the width once, by hand, against the alternative operand style before trusting a ranking.
> *(Measured on one kernel: two operand styles against one tensor's stride set give 1 vs 8 elements per lane —
> a factor the sweep could never surface, because no config in it differed on that axis.)*

### Worked example — a logical↔physical contradiction (A, row-major)

A stored row-major, shape (M,K): `addr(m,k) = m·lda + k` → **K is stride-1, M is strided (stride = lda)**.
A thread owns the logical patch M{0-3} × K{0-3}.

- **Vectorize along K (the stride-1 axis) — VALID.** Fix M=0, take K0-3. Their addresses are `0, 1, 2, 3`
  (contiguous). One `VW=4` load returns exactly `[M0K0, M0K1, M0K2, M0K3]` = the intended K-vector. Wide
  **and** correct.
- **Vectorize along M (the strided axis) — CONTRADICTION.** Fix K=0, intend the M-vector
  `[M0K0, M1K0, M2K0, M3K0]`. Those elements live at `0, lda, 2·lda, 3·lda` — **not** contiguous. But a
  hardware `VW=4` load reads the 4 *contiguous* addresses `0, 1, 2, 3` = `[M0K0, M0K1, M0K2, M0K3]` — the
  **wrong** elements (M0's K-run, not the M-run). So a wide load is incorrect; the only correct way to gather
  the M-vector is **`VW=1`** (four separate loads at `0, lda, 2·lda, 3·lda`).

The logical request ("vector along M") contradicts the physical layout (M strided) → wide vectorization is
invalid → forced `VW=1`. (Mirror case: A col-major makes M stride-1 → vectorizing M is valid/wide, and
vectorizing K becomes the contradiction → `VW=1`.)

## 3. Fragments vs MMA-acceptability

This is the distinction that most often trips people up, so state it plainly.

### A fragment is *state*: registers + a layout tag

A fragment is the set of values a wave holds in registers, plus a **layout** (a
`WarpDistributionEncoding`) recording which `(lane, register-slot)` holds which logical element. That
layout is bookkeeping **we** maintain — **the hardware is layout-blind.** The MFMA reads register slot
`s` of whatever operand it is handed; it has no idea what we *think* sits in slot `s`. So a fragment
can legitimately be in *any* order. Same 4 elements a lane owns (a 2×2 tile `{M0,M1}×{K0,K1}`), three
different fragment states:

```
 K-contiguous  :  [ M0K0  M0K1  M1K0  M1K1 ]     (each M's K together)
 free-contig   :  [ M0K0  M1K0  M0K1  M1K1 ]     (each K's M together — what a coalesced load gave)
 arbitrary     :  [ M1K1  M0K0  M1K0  M0K1 ]     (some other permutation)
```

"Fragment state" = which layout it is in **right now**. `transform_fragment` changes the state
(register order) without changing which elements the lane owns (§8 tiers). The single row above is one
lane; the *full* state also spans the wave — a register read across lanes is a structured **slice** of
an axis (the M=64-into-16-lanes picture in §6, "Across the wave").

### MMA-acceptability is a *relationship between A and B*, not a property of one fragment

There is no "this fragment is MMA-ready" in isolation. The hardware's only requirement is that the
**pair** agree on K, position-for-position (§2):

```
   A fragment, K per slot:   K0  K1  K2  K3
   B fragment, K per slot:   K0  K1  K2  K3
                             ══  ══  ══  ══     MFMA pairs slot s of A with slot s of B
   ACCEPTABLE  ⇔  the two K rows are identical.    (M on A, N on B are not shown — they are free)

   mismatch (REJECTED):
   A K per slot:   K0  K1  K2  K3
   B K per slot:   K1  K0  K3  K2      ← slot 0 pairs A.K0 with B.K1  →  wrong product
```

So "make this fragment acceptable" always means "reorder it until its K-row matches its **partner's**".
`validate_operands(a, b)` compares the two fragments; it never checks either against a canonical.

### A, B, C have different roles

| fragment | role | free dim | carries | constraint at MMA time | its layout is… |
|---|---|---|---|---|---|
| **A** | input operand | M | M (output row) + K (contract) | its K-row must equal B's | free except K (tied to B) |
| **B** | input operand | N | N (output col) + K (contract) | its K-row must equal A's | free except K (tied to A) |
| **C** | accumulator / output | — | receives `C[m,n]` | **none** — C is produced, not fed | **derived** from A's M-order, B's N-order, and the issue order (§7) |

What falls out of the table:
- **"Canonical" is not special.** It is just one matched `(A,B)` K-pair. Any matched pair — including
  free-dim-contiguous × free-dim-contiguous — is equally valid.
- **A's M-order (and B's N-order) is free.** It only decides *where in C* that lane's product lands
  (§7 derived-C); it never affects acceptability.
- **C is never "made acceptable."** It is *produced* by the MMA in a derived native order, then
  `C-shuffle`d (another `transform_fragment`) into whatever the store or a downstream MMA wants.
- **The two transform sites are different problems.** The A/B shuffle makes an *input pair* agree on K
  (§6); the C-shuffle re-expresses an *output* for its consumer (§7).

## 4. The end-to-end flow

```
   GLOBAL MEMORY            A: (M×K)                    B: (K×N)
        │  wide / coalesced load (read whichever memory order is contiguous)
        ▼                                                    ▼
  ┌───────────────────┐                            ┌───────────────────┐
  │   A FRAGMENT       │                            │   B FRAGMENT       │   state = the layout the
  │   (some layout)    │                            │   (some layout)    │   load produced (often
  └─────────┬─────────┘                            └─────────┬─────────┘   free-dim-contiguous)
            │  transform_fragment                            │
            │  (register reorder, §8) — ONLY if the pair's   │
            │  K-rows don't already match; else skipped      │
            ▼                                                ▼
  ╔══════════════════════ MMA-ACCEPTABLE PAIR (gate) ══════════════════════╗
  ║            A.K-dist == B.K-dist   (position-for-position, §2/§3)         ║
  ╚═══════════════════════════════════┬════════════════════════════════════╝
                                      ▼
                    MFMA — one issue per atom, over the M×N×K grid
                                      ▼
                          ┌────────────────────────┐
                          │   C FRAGMENT            │   state = native accumulator order,
                          │   (native accum order)  │   FIXED by the issue order (§7)
                          └───────────┬────────────┘
                                      │  C-shuffle = transform_fragment (§7)
                                      ▼
              storable order ─────►  STORE (wide)   or   feed a downstream MMA as its A/B
```

Read it as: **load → (maybe) reorder A/B so the pair agrees on K → MFMA → C in a derived order →
reorder C for its consumer.** The two reorders are the only transform sites; everything else is data
flowing through. As five steps:

1. **Load A/B** from global memory (row/col-major) → wide loads, identical lane ownership.
2. **A/B → MMA form.** If the pair's K-rows already match, nothing happens; otherwise reorder one (or
   both) so `A.K-dist == B.K-dist`.
3. **MFMA** over the M×N×K grid of atoms. The accumulator emerges in one fixed register order, set by
   the MMA issue order (§7).
4. **C-shuffle.** Re-order the native accumulator to a storable/consumable order (row/col-major, or an
   interleaved layout to feed a downstream MMA).
5. **Store C** in the chosen order (wide store).

The A/B shuffle (step 2) and the C-shuffle (step 4) are distinct transform sites with distinct formula
families (§6 vs §7).

## 5. `interleave_idx` primitive

`interleave_idx<gather, stride, count>` is the closed-form register-index permutation applied to a
fragment's register slots. **`gather`** is the group size — how many adjacent slots move as one unit;
**`gather == 1`** (the only case this doc needs) is element-granular. Within each `count`-sized block
the local index is transposed as a `(stride, count//stride)` grid:

```
gather == 1:  block = (i // count) * count
              local = i % count
              target[i] = block + (local % (count//stride)) * stride + (local // (count//stride))
```

Worked example — `interleave_idx<1, 2, 8>` (count 8, stride 2 → a `2 × 4` grid read out `4 × 2`):
```
 i:          0  1  2  3  4  5  6  7
 target[i]:  0  2  4  6  1  3  5  7      (slot i's contents move to register target[i])
```

- **NOP when `stride ∈ {1, count}`** (a 1×count or count×1 grid is not transposed).
- `stride` and `count` for each use are supplied by §6 (A/B) and §7 (C).
- **`gather > 1` is OUT OF SCOPE** — the grouped form used by a **canonically-laid-out** 32×32 accumulator
  (`interleave<4,8,16>`); its mechanics live in the reference tables, not here. ✗ This is **not** a verdict on
  a **constructed** interleaved multi-patch accumulator — see §9 → *Accumulators*.

## 6. A/B input interleave, MmaDim 16

Per lane:
- **DPT (DimPerThread)** = free-dim atoms per lane = `m_iter` (A) / `n_iter` (B).
- **KPT (KPerThread)** = total K per lane = `k_ab_per_lane × k_iter` (the k_iter atoms fold into K).

The transform is a transpose of the `DPT × KPT` per-lane register grid:

| direction | permutation |
|-----------|-------------|
| K-contiguous → free-dim-contiguous | `interleave_idx<1, KPT, DPT*KPT>`  (stride = KPT) |
| free-dim-contiguous → K-contiguous | `interleave_idx<1, DPT, DPT*KPT>`  (stride = DPT) |

The two are inverses. **NOP when DPT == 1** (a single atom in the free dim — nothing to interleave
against K).

Reference rows (MmaDim 16): 32×16 (DPT 2, KPT 4) → `<1,4,8>` / `<1,2,8>`; 64×16 (DPT 4, KPT 4) →
`<1,4,16>`; 32×32 (DPT 2, KPT 8) → `<1,8,16>` / `<1,2,16>`; 16×16 (DPT 1) → NOP.

### Worked example — A operand, 64×16 tile (DPT 4, KPT 4), one lane

**One thread owns ONE tile.** In the 64×16 A block, a single lane owns a 4×4 patch — 4 free-dim rows
(M0–M3) × 4 K columns (K0–K3). This same tile — the same 16 elements — is what the thread holds in
**both** layouts:

```
                       K-axis (contraction)  ───►
                    K0     K1     K2     K3
   free   M0  ┃   M0K0   M0K1   M0K2   M0K3   ┃
    │      M1  ┃   M1K0   M1K1   M1K2   M1K3   ┃   ◄─ one thread's tile
    │      M2  ┃   M2K0   M2K1   M2K2   M2K3   ┃      (identical in both
    ▼      M3  ┃   M3K0   M3K1   M3K2   M3K3   ┃       layouts below)
```

Only the **order those 16 elements sit in the thread's vector registers** changes with the read
direction (each `[...]` is a 4-register run):

**K-contiguous** — vector runs along K (walk each row):
```
 regs 0→15:  [M0K0 M0K1 M0K2 M0K3]  [M1K0 M1K1 M1K2 M1K3]  [M2K0 M2K1 M2K2 M2K3]  [M3K0 M3K1 M3K2 M3K3]
```

**free-dim-contiguous** — vector runs along the free dim / M (walk each column):
```
 regs 0→15:  [M0K0 M1K0 M2K0 M3K0]  [M0K1 M1K1 M2K1 M3K1]  [M0K2 M1K2 M2K2 M3K2]  [M0K3 M1K3 M2K3 M3K3]
```

Same 16 elements, same lane — only the register slot order differs. Transposing the 4×4 grid is
`interleave_idx<1, KPT=4, 16>`:
```
 K-contiguous reg:   0  1  2  3   4  5  6  7   8  9 10 11  12 13 14 15
      lands at reg:   0  4  8 12   1  5  9 13   2  6 10 14   3  7 11 15   (free-dim-contiguous)
```

### Across the wave — a register is an axis *slice*

The view above is one lane. Across the whole wave, an axis is **distributed over lanes × registers as
structured slices** — that is what "compress the free/K dims" means. free-dim-contiguous load
(A col-major, VW=4): lane `L` reads the 4 memory-contiguous elements `M{4L..4L+3}` at one K. So read
**one register across the 16 lanes of a K-group** and you get a stride-4 slice of the M=64 axis
(`M = 4·lane + reg`):

```
              lane:   0    1    2    3    4   ...   15
   reg0 (M-slice 0)   M0   M4   M8   M12  M16  ...  M60
   reg1 (M-slice 1)   M1   M5   M9   M13  M17  ...  M61
   reg2 (M-slice 2)   M2   M6   M10  M14  M18  ...  M62
   reg3 (M-slice 3)   M3   M7   M11  M15  M19  ...  M63      (all at K0)
```

The M=64 axis is **compressed into 16 lanes** (each a contiguous group of 4) × **4 registers** (four
stride-4 slices). A register is not a canonical single value — across the wave it holds a whole
structured slice. K is split the same way across the four lane-groups: lanes 0–15 → K0–3, 16–31 →
K4–7, 32–47 → K8–11, 48–63 → K12–15. In the reference sheet those lower groups are drawn compressed
and labeled `M0/M1/M2/M3, K4-7` (etc.) — there the `M0..M3` name the four register **slices**, not
literal M indices. This lane×register slicing is the "M-dist" the K/M-distributions encode (§2/§3);
the single-lane reorder above is just how one lane's share of these slices sits in its registers.

**B is identical with N in place of M** (a lane owns an `N × K` tile); only the row/col-major sense of
each order flips (K-contiguous is col-major for B, row-major for A).

**DPT = 1 (16×16 tile) → NOP.** The lane owns a `1 × 4` tile — a single row — so "walk each row" and
"walk each column" read out identically; there is nothing to transpose:
```
                 K0    K1    K2    K3
   free  M0  ┃  M0K0  M0K1  M0K2  M0K3  ┃      K-contiguous == free-dim-contiguous  (identity)
```

## 7. C accumulator transforms

The accumulator comes out of the MFMA in one fixed register order, and **that order is set by the
order the MMAs are issued** across the M×N subtile grid — nothing else. Same products, same math; only
the register grouping changes with the loop nest. Two cases (this is exactly the `Tiling.order` knob):

- **M-outer issue order** (iterate all N for a fixed M block, then advance M): the accumulator is
  grouped M-outer. == `Tiling.order` M-outer.
- **N-outer issue order** (iterate all M for a fixed N block, then advance N): the accumulator is
  grouped N-outer. == `Tiling.order` N-outer.

So the "native accumulator order" is not a hardware format to memorize — it is just a restatement of
which subtile loop is outer.

**Derived-C.** Because A's M-register order and B's N-register order are free (§2/§3), *they* decide
which `C[m,n]` each lane's product lands on. C's native layout is therefore *derived* from A's M-order,
B's N-order, and the issue order — not chosen independently. To store or reuse C you re-order it.

**The machine coupling is fixed and PHYSICAL — "following" the inputs.** The MFMA is a fixed physical
wiring: physical `A(reg,tid)` and physical `B(reg,tid)` are paired and summed into physical
`C(reg,tid)`, identically for every problem. Canonical vs interleaved is *only* which logical label you
loaded into each physical slot — canonical inputs → canonical outputs; interleaved inputs → interleaved
outputs, through the **same** coupling. So deriving C = run the canonical machine on physical slots and
carry the **supplied** logical labels through it: for each physical C slot whose canonical identity is
`(Mc,Nc)`, give it the DERIVED label *(the M actual-A placed on its coupled A-row, the N actual-B placed on its
coupled B-column)*. Two equivalent readings — **fixed-physical**: the C physical layout *stays* the
canonical machine layout, only the `(M,N)` labels move onto it; **fixed-logical**: a result `C[m,n]`
lands wherever the machine canonically calls `(Mc,Nc)`, i.e. *off* canonical `(m,n)`. It is a clean
DERIVATION because a register shuffle is a permutation of the free dim (each canonical row → exactly one
actual row), so it composes as pure encoding arithmetic. See `../mma_is_machinery.md` (§"machine
mapping is a fixed PHYSICAL coupling") for the `reg0/tid0` worked example.

> **The canonical trap — do not miss this.** The store-friendly interleaved C ownership is a
> **different element set per lane** than the canonical accumulator, so canonical-C → interleaved-C is
> **cross-lane**, NOT an intra-lane `interleave_idx`. You therefore **cannot** get the wide coalesced
> store by shuffling a *canonical* accumulator — the interleaved C is **produced** by feeding
> interleaved A/B. The intra-lane C-shuffle below only re-orders *within* an already-interleaved
> ownership (the Step4 finalize). This is why "interleaved wave tile" means interleaved A/B **first**.
>
> **Concrete — our 64×64 wave tile** (reference sheet `64x16 (VW_4_4)`, "Step4: Shuffle accum to
> output format"). Interleaved C ownership — lane `L` holds:
> - **M rows** `(L//16)*16 + {0..15}` (16 rows, via registers); **N cols** `(L%16)*4 + {0..3}` (4 cols)
> - register order `reg = m_local*4 + n_local` (M-outer, N-inner; VW = 4 in N)
> - across lanes: lane0→N0-3, lane1→N4-7, … lane15→N60-63 = **a full contiguous 64-N row**, so the
>   store coalesces across the wave.
>
> Contrast the **canonical** accumulator, lane0: `M∈{0-3,16-19,32-35,48-51} × N∈{0,16,32,48}` — a
> scattered set. That mismatch is exactly why "shuffle the canonical C to a store order" is cross-lane
> and wrong; the interleaved A/B path avoids it by construction.

The **C-shuffle** re-orders that native accumulator to a storable order; it is a function of the issue
order (the starting register grouping) and the target store order. For a single-patch atom it is a plain
intra-lane `interleave_idx`. Let `VW = ACC_VW` (4 for 16-bit C) and `MmaA`, `MmaB` = the atom counts
along C's two axes (`m_iter`, `n_iter`), so the C fragment has `VW·MmaA·MmaB` registers. For the
reference table's native accumulator (the **M-outer / A-major** issue order):

| store order | C-shuffle |
|---|---|
| row-major (VW = `VW`)       | `interleave_idx<1, VW, VW·MmaA·MmaB>` |
| col-major (VW = `VW·MmaB`)  | `interleave_idx<1, VW·MmaB, VW·MmaA·MmaB>` |

Verified across the reference ACC rows (16×16, 32×16, 64×16, 32×32, 64×64, …). Worked instance — 64×64
C, 4×4 atom grid (`MmaA=MmaB=4`, `VW=4` → 64 registers): row-major = `interleave_idx<1, 4, 64>`;
col-major = `interleave_idx<1, 16, 64>`.

**N-outer issue order** transposes the native grouping: swap `MmaA ↔ MmaB` (equivalently swap the
row/col roles). *Derived by M↔N symmetry — confirm against the ACC reference table's rows before
relying on it.*

**Cross-lane extensions — OUT OF SCOPE** (named for orientation; mechanics in the reference tables). **These
are shuffles of a CANONICAL accumulator; a multi-patch accumulator built by the §9 construction does not take
this path.**
- **32×32-acc**: `interleave<4,8,16> + "flip & zip"` (`flip&zip = unpackLoHi32(extractLo, extractHi)`);
  also the `gather=4` grouped `interleave_idx`.
- **f64**: `UnpackLoHi16 + UnpackLoHi32`.
- **RDNA3**: `replicate` (DPP).

### C-store coalescing — the lane-major axis (store-transaction cost)

Coalescing is decided by how many distinct cache lines the wave's addresses touch **per store
instruction** — it is address/cache-line based, **not** lane-adjacency based (non-adjacent lanes whose
addresses land in the same line DO coalesce). What matters is which C axis varies across **consecutive
lanes**: that axis must be the output's **stride-1** axis for the wave's writes to fuse.

For the 16×16 atom, 64×64 wave (4×4 atoms, `ACC_VW=4`) the interleaved C ownership is
`lane = 16·(M//16) + (N//4)` — each lane owns a contiguous **16 M × 4 N** patch (verified in code via
`RegisterMapper`). So the lane axis splits asymmetrically:
- **N is lane-major:** 16 consecutive lanes span a contiguous **64-N** run (lane0→N0-3 … lane15→N60-63).
- **M is block-major:** only the 4 lanes `{0,16,32,48}` span a contiguous **64-M** run; consecutive
  lanes step N and jump 16 lanes to advance M.

Consequence — the store major that matches the lane-major axis coalesces; the other pays for it:

| C output major | vectorized store axis (§2b) | lanes spanning the 64-run | store transactions / 128 B line |
|---|---|---|---|
| **row-major (N stride-1)** | N — valid, wide | **16** (lanes 0-15) | **1** — the wave fills each line in one instruction |
| **col-major (M stride-1)** | M — valid, wide | **4** (lanes 0,16,32,48) | **~4** — each line written by 4 partial `dwordx4` stores |

Both stores are per-lane wide and §2b-valid (each vectorizes its own stride-1 axis) — the difference is
purely **cross-lane fusion**. The col-major store touches the **same total footprint** (128 lines for the
64×64 f32 tile, identical DRAM bytes) but issues **~4× the store transactions**: each cache line is
written by 4 partial stores instead of 1, because only 4 lanes — not 16 — are inline along M. The factor
is `lanes_along_N / lanes_along_M` (here 16/4 = 4); it generalizes to any wave/atom via the ownership
split.

The MFMA fixes this lane split, so the **output major decides the cost**: row-major C is *with* the grain,
col-major C is *against* it. To make a col-major C wave-coalesced, move M onto the lane-major axis — the
**crossed A↔B swap + `c_transpose`** (§8, free-symmetry tier): **route** A into the B-slot so the machine
emits `Cᵀ` (source-swap, labels invariant), then **reposition** the C coordinates `(N,M)→(M,N)`; M becomes
lane-consecutive and the col-major store fuses. Register-identity (reposition + routing), not cross-lane —
NOT a label change.

**Cost reality (empirical seam).** The ~4× is *exact address arithmetic* (verified), but it is a
**store-transaction** count, NOT automatically a TFLOPS hit — the C store is a one-time epilogue that is
often hidden in the MFMA shadow. Treat it like a bank conflict: real, modelled, but subordinate to the
binding stage. **Measure per case** (sweep the A↔B-swap knob); do not assume the 4× shows up end-to-end.

## 7a. Register states + the layout optimizer

Two things a designer needs per A/B/C-major combination: (1) what register **state** the data is in at each
stage, and (2) whether a requested layout **will work** — and if not, how to **make it work**. The states are
described here; the transforms are **DISCOVERED per context**, never tabulated — because the permutation
changes with atom / wave-size / #atoms / tile shape / dtype (a fixed formula is right for one config and
silently wrong for the next).

### The three zones (the model the table encodes)

- **Inputs (A, B)** — two orthogonal states: **coalescing** (which thread-tile axis is stride-1 — matches the
  tensor's stride-1, so the load is wide) and **MMA-readiness** (K-contiguous **and** `A.K-dist == B.K-dist`,
  §2 — the pairing is the second half). Bridge coalescing→MMA is the in-thread `reorder` (§6), or `identity`
  if the load is already K-contiguous. Row/col is operand-relative (§1): A row-major = K-contiguous (ready);
  A col-major = free-dim-contiguous (reorder). B is the mirror: B col-major = K-contiguous; B row-major = free.
- **Accumulator (C native)** — MMA-friendly by construction; each lane owns a set of congruent, evenly-spaced
  M×N patches (one patch is the common case, not the requirement — §9 → *Accumulators*); register
  order set by the **issue order** (§7), generally neither row- nor col-major. Label **derived** (§7).
- **Output (C)** — reorder the native accumulator to the store major (§7). Two properties: the *shuffle*
  (a register reorder within the lane's C registers) and the *ownership-derived* coalescing + vectorizable
  width (below). A↔B symmetry is `free` when both operands share a contiguity mode (§8): RCR, RCC, CRR, CRC.

### Transforms are discovered, not tabulated

The permutation between two states is the **delta between two constructed layouts**, computed by rocKE's own
solver (`classify_transform`) — never a formula copied from a table. Those change with every context and are
error-prone to transcribe; construct the two states and ask the solver. The optimizer
(`helpers/tiling/layout_optimizer.py`) wraps this into the two questions a designer actually asks:

- **`evaluate_transform(source, target, *, canon=…, k_partner=…, dtype_bits=…)`** — is there a VALID path
  (an MMA-sound target via `operand_soundness` + a transform via `classify_transform`), and what does the
  cheapest edge cost?
- **`optimize_layout(source, candidates: dict, …)`** — the MINIMIZATION: sweep candidate distributions
  (different striping / ownership / register order) and rank them. It tries a free symmetry, an LDS
  reposition, an intra-lane reorder, or a *different distribution* before falling to cross-lane.
  `recommend(ranked)` picks the winner; `enumerate_stripings(shape, wave_size)` builds the candidate set.

**Cost ladder (cheapest → last resort):** free-symmetry (0) < `reorder` dword-aligned (1) < `reorder`
sub-dword (~pack factor) < **LDS reposition** < **`cross_lane`** (DPP/`ds_bpermute`, last resort, grows with
the tile). The **LDS reposition** is the cheaper alternative to a register cross-lane *when the data already
transits LDS* — but it is **not free and not below a register reorder**: a full round-trip = store + read at
the throughput **floor** + a **barrier** (paid even conflict-free) + the **new access's bank conflicts** (TWO
patterns — store and read — each EMPIRICAL under its own port rule via `/bank-conflict`) + any LDS
**capacity/occupancy** cost; routing through LDS can even *introduce* conflicts the register path never had, so
measure the new pattern per case. **Both heavy movers scale with the number of registers moved** — cross-lane
moves each register individually (steeper), the LDS reposition is **bandwidth-bound** (store + read of every
register) — so both get worse as the tile grows, which is where LDS bandwidth becomes the binding resource.
This is the §10 cost model made runnable, self-standing for any context (the sheet in `reference_docs/` is a
human aid, incomplete and error-prone — rocKE derives, it does not consult it).

**A load/store distribution whose COALESCED order ≠ the consumer's order pays a §6 in-thread reorder at this
ladder** (dword-aligned = register renumber ~0 ops; sub-dword = ~pack-factor `v_perm`/lane). This is the
**price of the wide coalesced access, not a peer knob** — you buy this cheap intra-lane reorder to KEEP the
wide load + conflict-free LDS; avoiding it by narrowing the load (VW→1) is a bandwidth cliff, backwards.
It is **derived per case** (`transforms.reorder_between` → `classify_transform`/`name_permutation`, e.g. the
CRC A-read `interleave_idx(1,8,32)` sub-dword, the C-shuffle `interleave_idx(1,16,64)` dword) — never
tabulated — and is **subordinate to the binding stage** (often hidden in the MFMA shadow: report the
`v_perm`/lane count, do not imply a TFLOPS hit; measure per case). `/layout-viz` draws it as an explicit
reg→reg stage with the named arrow + this cost.

### Coalescing / width (ownership-derived — the layout's intent)

Store coalescing is the lane-major property of the interleaved-C ownership (§7 "C-store coalescing"): a
C-major that lands on the lane-major axis stores **with the grain** (1×); the other pays `~lanes_maj/lanes_min×`
transactions **against the grain**. Vectorizable width is the per-lane stride-1 run capped at 128-bit — the
layout's **intent**, distinct from the codegen's *achieved* store width (a separate efficiency question).

## 8. Transform tiers

- **`reorder`** — a lane-uniform register permutation: **same element set per lane, same permutation on
  every lane** (the element STAYS on its lane). This tier spans a cost range, NOT a single cost:
  - **dword-aligned** — the permutation moves whole packed VGPRs (dwords). ~free: a register MOV that
    regalloc often coalesces away. Dtype-independent.
  - **sub-dword** — the permutation reorders elements *within* a packed dword (f16 = 2/dword, f8 = 4,
    f4 = 8). Needs **unpack (extract lo/hi) → move → repack**, cost ∝ the pack factor (f16 ~2×, f8 ~4×).
    Still `reorder` — the element does NOT change lane, it is only re-packed in place.
  Compile-time `vec_extract`/`vec_insert`. All single-patch A/B and C-shuffle interleaves are this tier.
  ✗ The `cross_lane` entries below (32×32 flip&zip, f64 unpack, RDNA replicate) are shuffles of a **CANONICAL**
  accumulator — they are NOT a verdict on a multi-patch accumulator BUILT by the §9 construction, which stays
  in this tier. See §9 → *Accumulators*.
- **`cross_lane`** — an element moves **between lanes**, or the on-lane permutation is not lane-uniform.
  Needs DPP / `ds_bpermute` / LDS. 32×32 flip&zip, f64 UnpackLoHi, RDNA replicate live here. **THIS is
  the reject/defer seam** (see the D1 correction below).

**D1 reject-seam CORRECTION (locked).** D1 originally rejected *sub-dword* permutations. That is wrong:
sub-dword addressing does NOT change which lane owns an element (it re-packs within the lane), so it
stays a (costlier) `reorder`, not a reject. The **only** invasive boundary — the tier the framework
rejects/defers — is **`cross_lane`** (inter-lane movement). Move the reject seam in
`classify_transform` from sub-dword → `cross_lane` accordingly (code follow-up).
- **Free symmetry — reposition (coordinate transpose) + source-swap (routing)** (the FREE tier, *below*
  `reorder`, zero register movement, register-identity, emits **nothing**). This is the `free-symmetry`(0)
  cost tier used in the ladders throughout. Both are **LABEL-INVARIANT** — ✗ NOT a "relabel" (it moves a
  *position/wiring*, never a label; a true relabel changes a label and is reserved for a FINISHED tile
  reused as input). Label semantics SOT: `label_flow_and_transforms.md`. Two structural symmetries:
  1. **Coordinate transpose = col-major ↔ row-major (a REPOSITION).** `col_major(X) == row_major(Xᵀ)`: the
     *identical* registers, **coordinates** transposed for free via addressing (`_transpose_desc` swaps
     which X-dim is stride-1). The **label rides invariant** — ✗ never phrase it as `(M,K)→(K,M)` on the
     datum (that is the memref's positional axis order, not a relabel). This is how M-innermost LDS is
     indexed for free.
  2. **Source-swap = A/B routing (M-data→B-slot, N-data→A-slot).** A(M,K) and B(N,K) are structurally
     identical fragments; feeding an A-shaped fragment into the B-slot (and vice versa) is a **machine-input
     ROUTING** — labels **INVARIANT** (B-slot carries M,K; A-slot carries N,K), C **DERIVES**. Free because
     M/N placement only routes where C lands (§2). ✗ NOT an M↔N label swap.

**Consequence — the crossed interleaved↔canonical *operand* bridge is FREE (reposition + routing), NOT
`cross_lane`.** Across a *multi-atom* wave tile the interleaved and canonical operand encodings assign a
given element to *different lanes* (e.g. (M0,K4) → lane 0 vs lane 16). That *looks* like `cross_lane`, but
it is the **source-swap** of symmetry (2): the interleaved-A layout *is* what a canonical-B wants. So the
free bridge is **operand routing + C coordinate transpose** (`c_transpose`, Part D2):
  (i) route the interleaved A-read into the MMA **B-slot** and the B-read into the **A-slot** (source-swap,
      labels invariant);
  (ii) the accumulator **DERIVES** as **Cᵀ (N,M)**;
  (iii) `_transpose_desc` the C encoding's *coordinates* back to (M,N) (a reposition).
Do NOT reach for `transform_fragment`/`cross_lane` here — the whole crossed bridge is register-identity.
*(Confirmed bit-exact: base and ab_swap both compute `C=A·Bᵀ`, `max_abs_diff=0.0`.)*

**Coupling.** In automation the A/B register order is coupled, and the pair must share the same
K-distribution (§2). The interleaved layouts we build — custom static tile distributions (K-contiguous
register order, rectangular per-lane patch) — are **sound for any tile shape**, square or rectangular. A
per-operand override can force otherwise (author's responsibility; verify K-dists match).

**K-alignment is validated PER ATOM, not per whole fragment.** `validate_operands` (`transforms.py`) compares
A and B on their **per-atom** K signature, not the concatenated whole-fragment K-list — because the MMA is
issued per 16×16×16 atom, and a rectangular wave tile has `m_iter ≠ n_iter` (A tiles more M-atoms than B tiles
N-atoms), giving different whole-fragment register counts even when every issued atom pairs the same K. It
takes the free-dim atom counts (`a_free_atoms`/`b_free_atoms` = m_iter/n_iter) so rectangular tiles like
64×32×32 build and are bit-exact; the whole-fragment compare would falsely reject them (it only passed
square `m_iter == n_iter` by coincidence).

## 9. Encoding requirement — how to build and verify it

A `WarpDistributionEncoding` lays a fragment's registers out as an ordered list of **register axes**
(each axis = one factor of the per-lane register count); the *order* of those axes is the register-slot
order (outer axis = most-significant slot).

### Operands — the DPT/KPT significance swap

K-contiguous and free-dim-contiguous share the exact same
axes and lane mapping — they differ only in the order of two of those axes.

**Construction.** The interleaved (free-dim-contiguous) encoding is the K-contiguous encoding with the
free-dim-atom axis (length `DPT`) and the within-atom K axis (length `KPT`) **swapped in significance**:
- K-contiguous: free-dim axis is **outer/major**, K axis **inner/minor** (K varies fastest).
- free-dim-contiguous: **swap them** — K axis outer, free-dim axis inner (free dim varies fastest).

That single axis swap *is* the construction — same axes, same lane→coordinate map, reordered register
significance. **NOP when `DPT == 1`** (nothing to swap against).

**Delta / verification.** The transform between two encodings is computed structurally, not assumed:
1. forward-map each encoding to `{(lane, register) → logical coordinate}`;
2. for each element, look up its `(lane, register)` in both maps — the **lane must match** (if any element
   changes lane it is `cross_lane`, rejected in this tier, §8);
3. read off the per-lane `source_register → target_register` permutation and check it is **identical on
   every lane** (a `reorder`, §8).

The resulting permutation must equal the §6 closed form — `interleave_idx<1, KPT, DPT*KPT>`
(K-contiguous → free-dim-contiguous) or `<1, DPT, DPT*KPT>` (the inverse) — for the tile's `(DPT, KPT)`.
The closed form is both the fast path (emit it directly) and the oracle the structural delta is checked
against.

### Accumulators — the multi-patch construction

An accumulator's lane may own **several congruent, evenly-spaced patches**, not one. One patch is the
common case, not the requirement: the de-interleave stays **intra-lane** at any patch count, because what
makes it intra-lane is that the target preserves the lane's element ownership — not that the elements are
adjacent. ✗ Do not read a multi-patch accumulator as disqualified from an interleaved layout. Construct it.

**Precondition — assert it, though it has never bitten.** `R · atom.n == wave_size`, where `R` is the
free-axis lane level (below) and `atom.n` the atom's lane level on the other free axis. Measured: it holds
for **all 128 registered rows**, so it rejects nothing today — keep it as a defensive assert against a future
atom, and do not present it as a screening gate. (The gate that DOES bite is the operand one below.) The two must multiply to exactly one
wave; that arithmetic is the whole reason the bijection validator passes at rank 2. An atom that breaks it
builds a silently wrong lane map — a wrong ANSWER, not an exception.

**Construction.** Four steps. Every quantity is READ from the traits row and never hand-typed, so a new atom
or a new target needs no new table:

1. Read the accumulator's shape from the traits row:

   | quantity | traits field |
   |---|---|
   | patches per lane `P` | `c_m_num_access` |
   | lane-rows `R` | `m / c_m_per_lane` |
   | inner run `V` | `c_m_per_lane / c_m_num_access` |

2. Factor the free axis into the levels `(P, R, V, free_sub)`.
3. Send `R` to the **lane**; send `P`, `V` and `free_sub` to **registers**. Every level is then claimed
   exactly once, which is what the bijection validator checks — so **no unmerge is needed for the
   accumulator descriptor itself**. ✗ An epilogue that CHUNKS the accumulator still needs one: the chunk's
   rank-3 form is what carries the lane level's place value in a tensor stride.
4. Confirm slot-by-slot against `derive_c_distribution` (the ground truth), then `classify_transform` the
   C-shuffle against the **same lane's** store order (§8 — a verdict against a target that itself re-owns
   lanes is self-consistent and answers a different question).

Worked, from the registered traits (two wave64 rows, to show single- and multi-patch are one procedure):

```
mfma_f32_16x16x16f16:  m=16 c_m_per_lane=4  c_m_num_access=1  ->  P=1  R=4  V=4   R*n = 4*16 = 64 = wave ✓
mfma_f32_32x32x8f16:   m=32 c_m_per_lane=16 c_m_num_access=4  ->  P=4  R=2  V=4   R*n = 2*32 = 64 = wave ✓
```

**`P == 1` is the degenerate case of this same construction** — single-patch and multi-patch are one code
path, not two.

Where this has been run the chain came out sound with **zero `cross_lane`**. That is a property of the
encoding, so the *constructibility* should transfer — but re-derive it rather than assume it, especially at
a different wave size. **`f64` is untested by this construction.**

**Operand-side precondition — check it PER CATALOGUE ROW; the wave size does not imply it.**
`free_lanes · k_lanes == wave_size`, with `k_lanes = k / k_ab_per_lane` and `free_lanes = m` on the A side,
`n` on the B side. **Check it PER OPERAND, not per row** — on a non-square atom one side can pass while the
other fails.

Measured over the registered catalogue (128 rows), the A side fails on **9**:
- 8 are the **gfx11-era** `wmma_*_16x16x16_*_w32` rows (those WITHOUT a `_gfx12` suffix), where a lane holds
  the atom's whole K (`k_ab_per_lane == k`, so `k_lanes == 1`). ✗ The bare glob also matches 10 `_gfx12`
  rows that **PASS** — the suffix is load-bearing.
- 1 is `wmma_f32_32x16x128_f4`, the only non-square row, where `k_ab_per_lane (64) != k (128)`. ✗ So
  "fails exactly when a lane holds the whole K" is **false** — that is the common cause, not the only one.
  Its B side passes (`16·2 == 32`), which is why the check is per operand.

✗ "Wave32 doesn't work" is **false** — 55 of 64 dense wave32 rows pass, and **no** dense wave64 row fails.
Scan the catalogue; never infer from the wave size or from a row's name. The accumulator precondition above holds for
every dense square row, so the two preconditions do not fail together.

**Structural constraints that bind.** The wave tile is a whole number of atoms on **both** free axes, and
`tile_k` is a multiple of the atom's K.

**Sizing — the width arithmetic, which transfers.** Every free-dim-vectorised stage gives a lane a
contiguous run of `wave_free / atom_free` elements — **per free axis**: `wave_m / atom.m` on M, `wave_n /
atom.n` on N. (✗ There is no `atom.mn` field; the traits row carries `m`, `n`, `k` separately, and this
construction's own precondition above turns on m and n differing.) At a **fixed wave tile**, doubling the atom therefore
**halves the LDS-read width and the C-store width together**. Push it to one atom across a free axis and
that stage degenerates to **one element per lane** — an access whose width is just the dtype's size, on the
operand read AND the C traffic, not merely a narrower C. **Size the wave tile WITH the atom, not after it.**
✗ The resulting width is atom- and dtype-specific — never quote a byte count from one case as the general
result.

**Whether a bigger atom is FASTER is a separate, per-kernel question.** It is a KNOB: sweep it, never assume
it either way, and re-sweep the neighbouring knobs when you change it (a wave-split ordering can invert).
*For example, on one GEMM a bigger atom won (~+8%), tied, AND lost (~−5%) at different configs of that same
kernel.*

## mma_workflow — the layout-choice recipe (thread-tile transpose flow)

The `mma_workflow` component answers: **given the memory layouts + dtypes of A, B, and C, choose the
highest-bandwidth way to move the data through the MMA.** It is the recipe that §10's cost model optimizes
over. It rests on two facts (derived in §1/§6 and §2):
- **The thread-tile is a rectangle** (A `M×K`, B `N×K`, C `M×N`) → its elements lay into registers two ways
  (row/col-major), and **row↔col-major is a pure in-register reorder** (a transpose; §1/§6) — no re-load, no
  cross-lane. (row/col is operand-relative: K-contiguous = row-major for A, col-major for B.)
- **MMA-ready = K-vectorize both operands** → an A run is *fixed M, K running*, a B run *fixed N, K running*;
  pairing slot-for-slot is the sound MAC (§2, `mma_is_machinery.md`).

### The five-step recipe

1. **Know the memory order, load coalesced.** The thread-tile serves either row- or col-major data; you just
   need to know which so you vectorize along the **contiguous** axis (wide/coalesced), never strided.
2. **Reach MMA-ready by in-register transpose if needed.**
   - *Happy case* — A row_major `M×K` × B col_major `K×N`: the coalesced K-load is **already** MMA-ready →
     zero reorder.
   - *Mismatch* — e.g. A arrives col_major (M contiguous). Do **not** strided-load into K-order. Load
     coalesced in the memory-native order (vectorize wide), **then re-order registers** into K-contiguous
     (the transpose). Wide load + one cheap reorder beats a strided load.
3. **Issue the atoms** in a chosen order (M-outer / N-outer). C comes out carrying the **flowed logical
   labels** — derived from A's M-placement, B's N-placement, and the issue order — riding the fixed canonical
   C positions (§7; POSITION ≠ LABEL, see `mma_is_machinery.md`).
4. **C is ALSO interleaved → its thread-tile is a rectangle (`M×N`).** So making C store-compatible
   (row- or col-major, wide coalesced store) is the **same in-register transpose** used on A/B — not
   cross-lane (within the already-interleaved ownership; the canonical-C trap of §7 does not apply here).
5. **Store coalesced.** The loop closes: coalesced load → transpose to MMA-ready → issue → interleaved C →
   transpose to store order → wide coalesced store. **Every transpose is an in-register reorder; nothing goes
   cross-lane.**

### The cost — and the intelligence

Layout choice is a **minimization**: the cheapest *valid* chain `total = load + A/B transform + C-shuffle +
store` at max bandwidth, *valid* = BOTH gates (MMA soundness §2 + tile-vectorization §2b); costs are
coalesced-vs-strided + the dtype-graded reorder tiers (§8: free-symmetry < dword-`reorder` < sub-dword `reorder` ≪
`cross_lane`). Two levers beyond "reorder or not" (the MMA Expert owns the call; scored in §10):
- **Move the transpose to the cheaper SIDE** — C *derives* from A/B, so reorder inputs OR output; f16 inputs
  (sub-dword, pricey) vs an f32 C (whole-dword, cheap) often means reorder C. Price both sides.
- **Shortcut with a swizzle / free symmetry** — a free symmetry (§8), an LDS swizzle (`lds_banks.md §5`), or
  DPP reaches the target with no explicit shuffle.

C's rectangle **derives** from A's and B's, so the whole A/B→C flow is *calculable* — exactly what §10 scores.

## 10. Picker logic — choosing the optimal layout chain (cost model, SOT)

The author states logical intent (**per-axis roles + strides, dtype, wanted output layout**) + a bound
**arch**; the SELECTOR picks the cheapest **chain** = `prefetch-LDS-layout × mma-operand-layout ×
share-region × C-epilogue`. Score every *valid* chain and take the min:

    cost(chain | roles+strides, dtype, arch) = global_load + transform + lds_read + c_epilogue

### Operand description — the picker's inputs (axis ROLES + strides; row/col is DERIVED) — LOCKED
The picker consumes exactly two things per operand, and NEITHER is "row/col-major":
1. **Per-axis role** — a tag on each tensor axis: `contract` (the K axis), `free` (M for A, N for B),
   `batch`. This is the N-D-general form of "which axis is the contraction" (einsum shared-label /
   linalg `iterator_types` + `indexing_maps`). The picker reads roles to find K -> the coalescing
   direction and which axis LDS must serve. (For >=2 contraction axes this role becomes a *shared label*,
   not just a category -- see "Multi-axis contraction" below.)
2. **Strides** — the memory order (which axis is stride-1), already carried by `TensorDesc`. The picker
   reads strides for the `global_load` / `lds_read` cost terms.

**Row/col-major is NOT the primitive; it is a lossy 2D PROJECTION of (role, stride) and is DERIVED.**
Reasons (locked): (a) row/col names which of *two* axes is contiguous -> only 2 orderings, but an N-D
operand has N! -> it cannot express batched / conv / attention layouts = exactly rocWMMA limitation #3
that this layer exists to escape; (b) strides already give the memory order, so row/col adds nothing
there -- its only real content is the role convention, which is the part that doesn't generalize.
Therefore: **roles + strides are the internal source of truth; row/col-major is exposed ONLY at the 2D
edge** -- an ergonomic `make_gemm_operand(a_row_major=...)` constructor and for rocWMMA/BLAS interop --
computed from (role, stride), never stored. The free symmetries (§8) stay general in this
framing: transpose = swap two axes' (role, stride); A<->B = swap the `free`-role axis (M<->N). Under
row/col those look like 2D special cases; under roles+strides they are the general operation. A canonical
logical order (M,K)/(K,N)/(M,N) can still be *fixed* for the MMA because repositioning any physical layout
into it is free (coordinate/stride reasoning, label invariant) -- the caller hands over arbitrary strides, the picker reasons canonically.

**Multi-axis contraction — labels, not ordered roles (generalizes the above) — LOCKED.** A general
contraction has *many* free and *many* contraction axes (e.g. A = `[a b c d e | f g]` = 5 free + 2
contract). Two consequences:
- A plain `contract/free/batch` *category* is INSUFFICIENT once there are >=2 contraction axes: it
  cannot express the **pairing** (which of A's contract axes contracts with which of B's). The primitive
  is therefore an **einsum-style shared axis id**: a shared id between A and B *is* the contraction and
  encodes its pairing; an id surviving to the output is `free`; an id on A, B AND C is `batch`.
  Rank-agnostic and ORDER-FREE (== linalg `indexing_maps` + `iterator_types`; NumPy `einsum`). Example:
  A `[a b c d e f g]`, B `[f g h i j]`, C `[a b c d e h i j]` -> `f,g` pair by label, no positional convention.
- The ONLY place an order enters is lowering the N-D contraction onto the **2D MMA atom** (M×N×K): the
  free ids **merge -> logical M / N**, the contract ids **merge -> logical K**, via a place-value
  unmerge/merge. That merge ORDER is a **layout decision** (choose the innermost merge for coalescing),
  **derived from strides** -- NOT a semantic label the author writes. This is exactly CK's **compact
  unmerge-merge descriptor** (the master plan's chosen formalism: a layout is fully specified by the
  K-unmerge + M-unmerge sizes). row/col-major is the degenerate **1-free / 1-contract** projection of it.

So the split holds at any rank: **labels carry semantics (unordered, N-D); strides + merge-order carry
layout (derived)**; the picker merges N-D -> 2D-logical (M, N, K) and scores the cost model there.

### Cost terms
| term | free (0) | cheap | expensive |
|---|---|---|---|
| **global_load** | load-dir matches the contiguous input axis (coalesced) | — | strided (major mismatch) |
| **transform** | free-symmetry — transpose (reposition) / A↔B (source-swap routing), any dtype; label invariant | `reorder` dword-aligned (whole-VGPR MOV) | `reorder` sub-dword (unpack/move/repack, ∝ pack_factor: f16 2×, f8 4×); **`cross_lane`** = REJECT/DEFER |
| **lds_read** | read contiguous on the LDS innermost | — | N-way bank conflict (read stride vs 32 banks) |
| **c_epilogue** | native accum order == output (interleaved MMA) | in-register `reorder` (dtype-graded) | LDS round-trip (cross-lane C) |

**Tier ranking (cheapest → invasive):** free-symmetry(0) < `reorder`-dword-aligned < `reorder`-sub-dword
(∝ 1/dtype-size) ≪ `cross_lane` (reject/defer seam). Smaller dtype ⇒ sub-dword reorder gets pricier ⇒
the value of reaching a chain via a **free symmetry rises** (symmetry moves nothing at any dtype).

### Chains + findings (the MECHANISM; measured perf lives in the per-kernel design record, not the SOT)
| chain | transform | lds_read | c_epilogue | outcome |
|---|---|---|---|---|
| interleaved / interleaved | none (matched) | M-innermost free-dim (`ds_read2_b32`); low conflict WITHOUT a pad | in-register, native==store | **model winner** (ranking validated) |
| canonical / canonical | none (matched) | K-innermost (`ds_read2_b64`, wider/op); K-row aliases banks → fixable conflict (pad to fix) | none (RCC direct store) | baseline (honest = PADDED) |
| interleaved / canonical (crossed) | A↔B swap + `c_transpose` (free-symmetry: routing + reposition, §8, DERIVED) | — | round-trip only if output ≠ native | not yet built |
| canonical / interleaved (crossed) | ″ | — | in-register (interleaved C) | not yet built |

**Findings (the durable MECHANISM — measured perf/counters live in the per-kernel design record + `/bank-conflict`, never the SOT):**
- **Interleaved beats canonical (RCR):** interleaved's M-innermost free-dim LDS read has low conflict WITHOUT
  a pad; canonical's K-innermost read has a K-row stride that is a multiple of 32 dwords → K rows alias the
  same banks (a REAL, *fixable* K-aliasing conflict, NOT the throughput floor — model in `lds_banks.md`).
- **Use the HONEST baseline: a PADDED canonical.** Padding the canonical K-dim de-aliases the banks and closes
  most of the gap; the naive unpadded canonical is a strawman. Interleaving's real edge is **modest**, and its
  structural win is getting low conflict *without spending LDS on a pad*.
- **De-aliasing helps ONLY when the read is EXPOSED (single-buffer).** In the double-buffered winner the
  prefetch overlap HIDES the read conflict, so a pad only HURTS (it spends LDS the config can't afford). The
  winner is therefore NOT read-conflict-limited — its remaining MFMA idle is issue-arbitration (addressed by
  `mac_prio`) + barrier/latency.
- **MfmaUtil is the causal signal, not `ALUStalledByLDS`** (a VALU counter, blind to the matrix pipe). The pad
  intervention PROVES causation: conflict DOWN + MfmaUtil UP + perf UP together (not mere correlation).
- **Not latency- or HBM-bound:** the MAC block per K-trip (32 MFMA) ≫ every memory latency (MFMA/LDS/L2/HBM,
  `kernels/latency_probe.py`); MemUnitStalled ~1–2%.
- **Tooling caveat:** `rocprofv3` aborts here (dlopen "config period" bug) — use `rocprof` v1; all conflict
  claims go through `/bank-conflict` (measured + a validated simulator).

### Worked examples — exercising the model (RCR vs RRR, f16 vs f8)
TWO transform SITES: **global→LDS** (coop store) and **LDS→MMA** (wave read). Costs are qualitative.

**RCR (A K-contig, B K-contig — symmetric):**
- interleaved/interleaved — global_load coalesced (K); global→LDS = free-dim `reorder` on BOTH A,B
  (dword-cheap at f16, hidden in the MFMA shadow); LDS→MMA none (matched); lds_read low-conflict (free-dim);
  c_epilogue in-register. **The model's predicted winner** (ranking validated).
- canonical/canonical — global_load coalesced (K); global→LDS identity; LDS→MMA none; lds_read K-innermost
  (wide `ds_read2_b64`) with the K-aliasing conflict (pad to fix); c_epilogue none (RCC direct). Ranks below
  interleaved; the honest baseline is the PADDED canonical.

**RRR (A K-contig, B N-contig — asymmetric; kernel currently assumes B K-contig):**
Symmetry broken ⇒ A and B want *different* prefetch (the "treat A/B separately" case). B's coalesced load is
along N = its FREE dim ⇒ B is *natively* free-dim-contiguous ⇒ interleaved prefetch for B is **free** (no
global→LDS reorder), whereas a canonical B would need one. Model predicts interleaved/interleaved again, and
**cheaper on B's transform than RCR** (B's reorder vanishes). *Prediction to measure once RRR is wired.*

**f8 shift (any layout; NOT YET MEASURED — needs an f8 atom):** the `reorder` term goes sub-dword ∝
pack_factor (f8 ≈ 4× f16), so the interleaved global→LDS reorder that hides for free at f16 may NOT hide
at f8. Model predicts the optimum can **flip to canonical** (or to whichever layout the *natural* load
already matches, avoiding the reorder) once `reorder_cost(f8) > bank_conflict_cost`. This is the
falsifiable f8 prediction and exactly why the two TO-CALIBRATE coefficients below are load-bearing.

### LDS bank-conflict model

See **`lds_banks.md`** — the SOT for the LDS bank model (per-half-wave × per-dword-phase arbitration),
K-stride aliasing, the contiguity floor + width ladder (b128/b64/b32), the fixes (free symmetry / pad /
contiguity-preserving swizzle / narrow / redistribute), and the **binding-stage decision** (conflict
reduction is subordinate to wall-time — a conflict-free narrow-store variant can be *slower*). A and B are
separate LDS regions — isolate them (A-only / B-only, store-only / read-only) to attribute a conflict.

### TO CALIBRATE (evidence pending — do NOT trust the coefficients until measured)
- `reorder` cost per dtype, **dword-aligned vs sub-dword** → extend `latency_probe`.
- bank-conflict cost per LDS read stride → see `lds_banks.md` (validated per-half-wave/per-phase model +
  the binding-stage decision); a per-(width, stride) cost table is still owed for the layout cost model.
- crossed-combo TFLOPS once the A↔B-swap + `c_transpose` path is built (predicted ≈ the matched combo
  of the *mma* side, plus at most a dword-aligned reorder — per the free-symmetry derivation, §8).

## 11. References

- Interleave reference tables (MFMA/WMMA), CDNA + RDNA: `interleave_idx`, the transforms summary, the
  per-tile A/B grids, and the ACC transform tables.
- `helpers/tiling/transforms.py` — `interleave_idx`, `k_distribution`, `classify_transform`,
  `validate_operands`.
- `helpers/tiling/mma/warp_encoding.py` — canonical `a_/b_warp_encoding` (the `interleaved=` flag is BROKEN
  and raises; it does not produce a proper interleaved layout).
- `helpers/tiling/kernels/tiling_gemm_interleaved_demo.py` — `_wave_descs_interleaved`: the real interleaved
  layouts (custom `make_tile_desc` static tile distributions).
- `helpers/tiling/kernels/latency_probe.py` — instruction-latency microbench feeding §10's cost model.
