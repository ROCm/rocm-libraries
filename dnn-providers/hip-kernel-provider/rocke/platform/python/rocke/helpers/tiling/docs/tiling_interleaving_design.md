# Interleaved Layouts — Design (SOT)

## TL;DR — the point (read this FIRST)

The MMA is a fixed vector machine. ISA documentation often refers to **"canonical"** data layouts when using
MMA, as a teaching baseline for the K-invariant (§2). Canonical layouts are, in many cases, not
efficient in memory-access ordering. Fortunately, canonical is **not** the requirement — we have
flexibility to use more efficient layouts while still satisfying the physical constraints of the MMA
(see [`mma_is_machinery.md`](./mma_is_machinery.md)).

An **interleaved layout** is a non-default rule for **which thread (and register) owns which elements of
the matrices** in its wave-tile assignment. It computes the *same* result as canonical — it only rearranges
who-owns-what so the memory side is more efficient. Two properties are the whole point:

**1 — What each issued atom covers.** A wave tile is one or more atoms, and the MMA always issues **whole
atoms** — what differs between the layouts is which logical data each issued atom carries:
- **Canonical:** each issued atom **is** one whole canonical atom; iterate atoms to cover the wave tile.
- **Interleaved:** an issued (**interleaved**) atom computes a **small portion of all the canonical
  atoms** at once; iterate those portions to cover the wave tile. That spreading is the interleave — and
  it can make the loads/stores more **coalescing**-friendly than one canonical atom at a time.

(Either way the MMA issues whole atoms; the *issue order* — M-outer/N-outer — is a separate knob, §7.)

```
CANONICAL — each issued atom IS one whole canonical atom; iterate to cover the wave tile:
   step 0:  atom0 M[0-15] K[0-15]
   step 1:  atom1 M[16-31]K[0-15]
   step 2:  atom2 M[32-47]K[0-15]   
   ... 

INTERLEAVED — each issued atom computes a small portion of ALL canonical atoms; iterate to cover the wave tile:
   step 0:  a small portion of atom0, atom1, atom2, …
   step 1:  the next small portion of atom0, atom1, atom2, …
   …
```

**2 — Switching coalescing order is a cheap in-register re-order.** Because interleaved thread patches are regular,
moving between the coalesced-memory order and the order the next stage needs (e.g., free-dim or k-dim contiguous)
is a **pure register re-order inside each thread — no reload, no cross-lane exchange**. So one layout serves the wide coalesced
access **and** is at most one cheap re-order away from what the next stage wants, on **both** sides of the
MMA — input (memory → MMA) and output (MMA → store). C is store-ready **for free** when the issue order
already matches the store order; otherwise it pays **one cheap in-register re-order** on the way out
(intra-lane only because C came from interleaved A/B — the §7 canonical trap).

```
One thread's 4×4 patch (rows = M, cols = K) → two register orders, same 16 elements:

       K0 K1 K2 K3
  M0 [ a  b  c  d ]     K-contiguous    (row-walk): [a b c d][e f g h][i j k l][m n o p]
  M1 [ e  f  g  h ]     free-dim-contig (col-walk): [a e i m][b f j n][c g k o][d h l p]
  M2 [ i  j  k  l ]
  M3 [ m  n  o  p ]     switch = a transpose of the register slots — no reload, no cross-lane
```

**Canonical is the degenerate case, not a special one.** A tile with a single atom along the free dim
(`DPT==1`) has nothing to interleave, so its map *is* canonical.

**The physical constraint (why any of this is legal).** Correctness is the **sound MAC** — per-operand
M/N-fixed *plus* `A.K-dist == B.K-dist` positionally — and M/N placement is free (you choose the constant;
it merely routes the output). [`mma_is_machinery.md`](./mma_is_machinery.md) is the **SOT for MMA
soundness** — the full model lives there.

Source-of-truth specification for interleaved MMA operand/accumulator layouts. Formulas are verified
against the interleave reference tables (MFMA/WMMA, CDNA + RDNA).

**Scope.** One recipe (§6, §8, §9) builds the intra-lane `reorder` tier **cross-lane-free**, by adjusting
the **static distribution** read from the target's traits. It covers **every dense square atom** — any
AMDGPU arch (CDNA, and RDNA/gfx11 via operand **replication**, §9), any dtype **including f64** — with no
special-casing. The **one non-square** catalogue atom (`wmma_f32_32x16x128_f4`) over-subscribes the wave on
one operand and does **not** interleave under this construction; it is deferred (task #21). Which rows
interleave is enumerated and locked by the proof sweep `test_interleave_all_dense.py` — read the census
there, never copy a count into prose. What varies per instance is **validation depth**: the construction is
proven cross-lane-free across the whole square catalogue (incl. f64) by that CPU sweep; GPU **bit-exact**
perf is validated on MmaDim-16 f16 / gfx90a and is a per-instance measurement elsewhere, never assumed.

**Building it — the shipped API.** The recipe below is what `InterleavedStyle` (`mma/styles/interleaved.py`)
implements. Call `style.lds_bridge(traits, role=…, free_sub=…, k_sub=…)` for the A/B operand bridge and
`style.accumulator_desc(traits, m_sub=…, n_sub=…)` for C; the sections that follow are the **why**, not a
second thing to hand-build.

**Naming.** This layer uses two descriptive labels for the register order, each naming exactly which
axis the vector width runs along — no "structure" to interpret, no other layout vocabulary needed:
- **K-contiguous** — the vector width runs along K (K varies fastest within a lane's registers). The
  MMA-native input order.
- **free-dim-contiguous** — the vector width runs along the operand's non-K dim (M for A, N for B).
  A coalesced load-staging order.

## Vocabulary

Defined once here so the rest reads cold:

- **wave / lane** — a *wave* is the threads that execute one MMA together (64 on CDNA/wave64, 32 on
  RDNA/wave32); a *lane* is one thread. Layouts are wave-wide.
- **atom** — the smallest hardware MMA instruction shape, e.g. a `16×16×16` MMA. A wave *tile* is a
  grid of atoms.
- **fragment** — the data a wave holds in registers for one operand (A, B, or C), together with its
  layout.
- **MmaDim** — the atom's M and N size (16 in the worked examples; "MmaDim-16" = 16×16 atoms).
- **VW (vector width)** — elements per contiguous memory access.
- **ACC_VW** — the accumulator's per-lane register run (a distinct quantity from VW; 4 per lane per
  16×16 atom in the worked examples).
- **m_iter / n_iter / k_iter** — how many atoms the wave tile spans in M / N / K =
  `wave_tile_dim ÷ atom_dim`. (These are the `MmaA` / `MmaB` / K-atom counts in the reference tables.)
- **k_ab_per_lane** — K elements a lane holds per atom for A/B (read from the traits row for any atom;
  4 for the f16 16×16×16 atom).
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
- **DPP / ds_bpermute** — hardware cross-lane data-movement ops; used only by the `cross_lane` tier (§8),
  which the static-distribution recipe is built to avoid.

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
the vector width walks first — **K-contiguous** walks each row (VW along K), **free-dim-contiguous** walks
each column (VW along the free dim). Both orders hold the **same elements owned by the same lane** — only
the register slot order differs, so they cost the **same wide, coalesced load**; nothing about *which* data
a lane holds changes. (Worked in full, with real register indices, at §6.)

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
   closed-form `interleave_idx<1, DPT, DPT·KPT>` permutation (K→free; the inverse free→K uses stride `KPT`) — the in-register **`reorder`** tier (§8), never
   cross-lane. Hallmarks 1 and 2 are the **same permutation seen two ways**.

**Checkable, not vibes.** #1: the register→coord order differs from the canonical encoding. #2:
`classify_transform(dir_a, dir_b)` returns `reorder` (not `cross_lane`) and matches the `interleave_idx`
oracle above.

**Caveat — interleaved ≠ automatically valid.** A cheap transpose does not exempt a layout from the two
validity gates: it must still pass MMA soundness (§2) AND tile-vectorization validity (§2b). "Cheaply
transposable" and "correct to load/use" are separate checks. (This cheap **in-thread** transpose is distinct
from bridging *canonical ↔ interleaved lane ownership*, which changes which lane owns which elements and can be
cross-lane — see §8.)

**Three canonical↔interleaved relationships — do not conflate them:**

| relationship | tier | where |
|---|---|---|
| in-thread K-contiguous ↔ free-dim-contiguous (register order) | `reorder` (cheap) | §1 / §6 |
| canonical ↔ interleaved *lane ownership* across a multi-atom tile | `cross_lane` (avoided) | §7 canonical trap |
| crossed A↔B source-swap + `c_transpose` bridge | **free** (reposition + routing) | §8 |

## 2. Hardware contract — the pairwise K-match half of the sound MAC

**Correctness SOT:** [`mma_is_machinery.md`](./mma_is_machinery.md) owns the full correctness model, the
**sound MAC** = per-operand soundness (one M per output on A, one N per output on B) *plus* the pairwise
K-match. This section derives that **pairwise K-match half**; per-operand soundness holds by construction
here — the fragments are atom-derived, so each output gets exactly one M on A and one N on B.

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
positionally, **compared per atom** — not across the concatenated whole-fragment K-list (§8 for the
rectangular-tile rationale). Lane ownership
is guaranteed valid by construction (fragments come from atom-derived descs), so a positional per-atom
K-match is sufficient.

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
layout is bookkeeping **we** maintain — **the hardware is layout-blind.** The MMA reads register slot
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
an axis (the wave64 M=64-into-16-lanes example in §6, "Across the wave").

### MMA-acceptability is a *relationship between A and B*, not a property of one fragment

There is no "this fragment is MMA-ready" in isolation. The hardware's only requirement is that the
**pair** agree on K, position-for-position: acceptability ⇔ A's and B's K-rows are identical slot-for-slot
(M on A / N on B are free) — worked in full at §2's slot tables.

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
                    MMA — one issue per atom, over the M×N×K grid
                                      ▼
                          ┌────────────────────────┐
                          │   C FRAGMENT            │   state = native accumulator order,
                          │   (native accum order)  │   FIXED by the issue order (§7)
                          └───────────┬────────────┘
                                      │  C-shuffle = transform_fragment (§7)
                                      ▼
              storable order ─────►  STORE (wide)   or   feed a downstream MMA as its A/B
```

Read it as: **load → (maybe) reorder A/B so the pair agrees on K → MMA → C in a derived order →
reorder C for its consumer.** The two reorders are the only transform sites; everything else is data
flowing through. As five steps:

1. **Load A/B** from global memory (row/col-major) → wide loads, identical lane ownership.
2. **A/B → MMA form.** If the pair's K-rows already match, nothing happens; otherwise reorder one (or
   both) so `A.K-dist == B.K-dist`.
3. **MMA** over the M×N×K grid of atoms. The accumulator emerges in one fixed register order, set by
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

## 6. A/B input interleave (worked on MmaDim-16; general construction in §9)

*(Worked instance: MmaDim-16 f16, wave64, gfx90a — the numbers below (M=64, 16 lanes, DPT/KPT) are this
instance. The construction generalizes to any dense square atom / wave size / arch; see §9.)*

Per lane:
- **DPT (DimPerThread)** = free-dim atoms per lane = `m_iter` (A) / `n_iter` (B).
- **KPT (KPerThread)** = total K per lane = `k_ab_per_lane × k_iter` (the k_iter atoms fold into K).

The transform is a transpose of the `DPT × KPT` per-lane register grid:

| direction | permutation |
|-----------|-------------|
| K-contiguous → free-dim-contiguous | `interleave_idx<1, DPT, DPT*KPT>`  (stride = DPT) |
| free-dim-contiguous → K-contiguous | `interleave_idx<1, KPT, DPT*KPT>`  (stride = KPT) |

The two are inverses. **NOP when DPT == 1** (a single atom in the free dim — nothing to interleave
against K).

Reference rows (MmaDim-16 **tile** shapes, 16×16 atom; each pair is **K→free / free→K**): 32×16 (DPT 2,
KPT 4) → `<1,2,8>` / `<1,4,8>`; 64×16 (DPT 4, KPT 4) → `<1,4,16>` (DPT==KPT, one value); 32×32-**tile**
(DPT 2, KPT 8) → `<1,2,16>` / `<1,8,16>`; 16×16 (DPT 1) → NOP.

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

Same 16 elements, same lane — only the register slot order differs. Transposing the 4×4 grid (K→free) is
`interleave_idx<1, DPT=4, 16>` — **note DPT==KPT==4 here, so this example cannot distinguish the
direction; the general stride is `DPT` for K→free (table/rows above)**:
```
 K-contiguous reg:   0  1  2  3   4  5  6  7   8  9 10 11  12 13 14 15
      lands at reg:   0  4  8 12   1  5  9 13   2  6 10 14   3  7 11 15   (free-dim-contiguous)
```

### Across the wave — a register is an axis *slice* *(worked instance: wave64, 16×16 atom; the mechanism generalizes, the 16-lane / M=64 numbers do not)*

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

The accumulator comes out of the MMA in one fixed register order, and **that order is set by the
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

**The machine coupling is fixed and PHYSICAL — "following" the inputs.** The MMA is a fixed physical
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
order (the starting register grouping) and the target store order. For a **single-patch** accumulator
(`P=1` — every 16×16-atom wave tile) it is a plain intra-lane `interleave_idx`. Let `ACC_VW = c_m_per_lane` (= `P·V`; the per-lane accumulator run — 4 for the 16×16 atom, f32) and
`MmaA`, `MmaB` = the atom counts along C's two axes (`m_iter`, `n_iter`), so the C fragment has
`ACC_VW·MmaA·MmaB` registers. The C-shuffle stride is the **store vector width** `store_VW` (distinct from
`ACC_VW`). For the reference table's native accumulator (the **M-outer / A-major** issue order):

| store order | `store_VW` | C-shuffle |
|---|---|---|
| row-major | `ACC_VW`      | `interleave_idx<1, ACC_VW, ACC_VW·MmaA·MmaB>` |
| col-major | `ACC_VW·MmaB` | `interleave_idx<1, ACC_VW·MmaB, ACC_VW·MmaA·MmaB>` |

Verified across the reference ACC rows (16×16, 32×16, 64×16, 32×32, 64×64, … — all **16×16-atom** wave
tiles, so `P=1`; a multi-patch *atom* like the 32×32 atom is built via §9's construction, not this closed
form). Worked instance — 64×64
C, 4×4 atom grid (`MmaA=MmaB=4`, `ACC_VW=4` → 64 registers): row-major = `interleave_idx<1, 4, 64>`;
col-major = `interleave_idx<1, 16, 64>`.

**N-outer issue order** transposes the native grouping: swap `MmaA ↔ MmaB` (equivalently swap the
row/col roles). *Derived by M↔N symmetry — confirm against the ACC reference table's rows before
relying on it.*

**Accumulators are built the same way, cross-lane-free**, by the §9 construction — every dense **square**
atom, any dtype (incl. f64, proven cross-lane-free by the sweep), any arch (RDNA via replication). Only GPU
bit-exact validation varies per instance; see §9.

### C-store coalescing — the lane-major axis (store-transaction cost)

Coalescing is decided by how many distinct cache lines the wave's addresses touch **per store
instruction** — it is address/cache-line based, **not** lane-adjacency based (non-adjacent lanes whose
addresses land in the same line DO coalesce). Consecutive-lane stride is what *sets* those addresses,
so which C axis varies across **consecutive lanes** decides it: that axis must be the output's
**stride-1** axis for the wave's writes to fuse.

For the 16×16 atom, 64×64 wave (4×4 atoms, `ACC_VW=4`) the interleaved C ownership is
`lane = 16·(M//16) + (N//4)` — each lane owns a contiguous **16 M × 4 N** patch (verified in code via
`RegisterMapper`). So the lane axis splits asymmetrically:
- **N is lane-major:** 16 consecutive lanes span a contiguous **64-N** run (lane0→N0-3 … lane15→N60-63).
- **M is block-major:** only the 4 lanes `{0,16,32,48}` span a contiguous **64-M** run; consecutive
  lanes step N and jump 16 lanes to advance M.

Consequence — the store major that matches the lane-major axis coalesces; the other pays for it (16×16
atom / wave64 example; the factor generalizes below):

| C output major | vectorized store axis (§2b) | lanes spanning the 64-run | store transactions / 128 B line |
|---|---|---|---|
| **row-major (N stride-1)** | N — valid, wide | **16** (lanes 0-15) | **1** — the wave fills each line in one instruction |
| **col-major (M stride-1)** | M — valid, wide | **4** (lanes 0,16,32,48) | **~4** — each line written by 4 partial `dwordx4` stores |

Both stores are per-lane wide and §2b-valid (each vectorizes its own stride-1 axis) — the difference is
purely **cross-lane fusion**. The col-major store touches the **same total footprint** (128 lines for the
64×64 f32 tile, identical DRAM bytes) but issues **~4× the store transactions**: each cache line is
written by ~4× as many partial stores (the `16/4` lane ratio), because only 4 lanes — not 16 — are inline along M. The factor
is `lanes_along_N / lanes_along_M` (here 16/4 = 4); it generalizes to any wave/atom via the ownership
split.

The MMA fixes this lane split, so the **output major decides the cost**: row-major C is *with* the grain,
col-major C is *against* it. To make a col-major C wave-coalesced, move M onto the lane-major axis — the
**crossed A↔B swap + `c_transpose`** (§8, free-symmetry tier): **route** A into the B-slot so the machine
emits `Cᵀ` (source-swap, labels invariant), then **reposition** the C coordinates `(N,M)→(M,N)`; M becomes
lane-consecutive and the col-major store fuses. Register-identity (reposition + routing), not cross-lane —
NOT a label change.

**Cost reality (empirical seam).** The ~4× *ratio* is *exact address arithmetic* (verified), but it is a
**store-transaction** count, NOT automatically a TFLOPS hit — the C store is a one-time epilogue that is
often hidden in the MMA shadow. Treat it like a bank conflict: real, modelled, but subordinate to the
binding stage. **Measure per case** (sweep the A↔B-swap knob); do not assume the 4× shows up end-to-end.
The table's *absolute* per-line counts assume the intended `b128` (`dwordx4`) store; the emit realizes
≤ `b64` (see [`bug_report_c_store_vectorization.md`](./bug_report_c_store_vectorization.md)), so read those
two columns as **intent-side**. The 4× ratio is a cross-lane fusion property, **width-independent**, and
holds regardless.

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
solver (`classify_transform`) — never a formula copied from a table. (Closed forms exist for exactly two
cases: the single-patch operand transpose (§6) and the single-patch `P=1` accumulator C-shuffle (§7);
multi-patch (`P>1`) and novel-atom cases are discovered.) These change with every context and are
error-prone to transcribe; construct the two states and ask the solver. The optimizer
(`helpers/tiling/layout_optimizer.py`) wraps this into the two questions a designer actually asks:

- **`evaluate_transform(source, target, *, canon=…, k_partner=…, dtype_bits=…)`** — is there a VALID path
  (an MMA-sound target via `operand_soundness` + a transform via `classify_transform`), and what does the
  cheapest edge cost?
- **`optimize_layout(source, candidates: dict, …)`** — the MINIMIZATION: sweep candidate distributions
  (different striping / ownership / register order) and rank them. It tries a free symmetry, an LDS
  reposition, an intra-lane reorder, or a *different distribution* before falling to cross-lane.
  `recommend(ranked)` picks the winner; `enumerate_stripings(shape, wave_size)` builds the candidate set.

**Cost ladder (cheapest → last resort; register tiers defined in §8, the SOT; the LDS-reposition rung is defined below):** free-symmetry (0) < `reorder` dword-aligned (1) < `reorder`
sub-dword (~pack factor) < **LDS reposition** (only when the data already transits LDS; otherwise a full
round-trip that can exceed a register `cross_lane` on a small tile) < **`cross_lane`** (DPP/`ds_bpermute`, last resort, grows with
the tile). The **LDS reposition** is the cheaper alternative to a register cross-lane *when the data already
transits LDS* — but it is **not free and not below a register reorder**: a full round-trip = store + read at
the throughput **floor** + a **barrier** (paid even conflict-free) + the **new access's bank conflicts** (TWO
patterns — store and read — each EMPIRICAL under its own port rule (`lds_banks.md` §1.4 write / §1.5 read) via `/bank-conflict`) + any LDS
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
tabulated — and is **subordinate to the binding stage** (often hidden in the MMA shadow: report the
`v_perm`/lane count, do not imply a TFLOPS hit; measure per case). `/layout-viz` draws it as an explicit
reg→reg stage with the named arrow + this cost.

### Coalescing / width (ownership-derived — the layout's intent)

Store coalescing is the lane-major property of the interleaved-C ownership (§7 "C-store coalescing"): a
C-major that lands on the lane-major axis stores **with the grain** (1×); the other pays `~lanes_maj/lanes_min×`
transactions **against the grain**. Vectorizable width is the per-lane stride-1 run capped at the registered vector ceiling (128-bit on current AMDGPU targets) — the
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
  The §9 construction stays in THIS tier for any dense **square** atom (any dtype, any arch incl. RDNA via
  replication) — it adjusts the static distribution and never emits a cross-lane shuffle; the non-square
  case is deferred (§9). See §9 → *Accumulators*.
- **`cross_lane`** — an element moves **between lanes**, or the on-lane permutation is not lane-uniform.
  Needs DPP / `ds_bpermute` / LDS. **THIS is the reject/defer seam** (see the D1 correction below) — the
  static-distribution recipe is built to avoid it.

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
issued per atom, and a rectangular wave tile has `m_iter ≠ n_iter` (A tiles more M-atoms than B tiles
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

The resulting permutation must equal the §6 closed form — `interleave_idx<1, DPT, DPT*KPT>`
(K-contiguous → free-dim-contiguous) or `<1, KPT, DPT*KPT>` (the inverse) — for the tile's `(DPT, KPT)`.
The closed form is both the fast path (emit it directly) and the oracle the structural delta is checked
against.

### Accumulators — the multi-patch construction

An accumulator's lane may own **several congruent, evenly-spaced patches**, not one. One patch is the
common case, not the requirement: the de-interleave stays **intra-lane** at any patch count, because what
makes it intra-lane is that the target preserves the lane's element ownership — not that the elements are
adjacent. ✗ Do not read a multi-patch accumulator as disqualified from an interleaved layout. Construct it.

**Precondition — assert it, though it has never bitten.** `R · atom.n == wave_size`, where `R` is the
M-side lane level (below) and `atom.n` is the N-side lane count — it equals the size `atom.n` only because
each lane owns one N per atom-column (`c_n_per_lane ≡ 1`; the traits carry only the M-side per-lane
structure). The assert catches a future atom that breaks that one-N-per-column assumption. Measured: it holds
for **all 128 registered rows**, so it rejects nothing today — keep it as a defensive assert against a future
atom, and do not present it as a screening gate. (The gate that DOES bite is the operand one below.) The two must multiply to exactly one
wave; that arithmetic is the whole reason the bijection validator passes at rank 2. An atom that breaks it
builds a silently wrong lane map — a wrong ANSWER, not an exception.

**Construction.** Four steps. Every *atom* quantity is READ from the traits row (`traits/mma_traits.py`,
the 128-row catalogue) and never hand-typed; the *wave-tile* factors (`m_sub`, `n_sub`) come from the
chosen tile. A new atom or a new target needs no new table:

1. Read the atom's accumulator shape from the traits row (`traits/mma_traits.py`) — **both free axes**:

   | quantity | traits field / derivation |
   |---|---|
   | **M** patches per lane `P` | `c_m_num_access` |
   | **M** lane-rows `R` | `m / c_m_per_lane` |
   | **M** inner run `V` | `c_m_per_lane / c_m_num_access` (so `c_m_per_lane = P·V = ACC_VW`) |
   | **M** across-atom factor `m_sub` | `wave_m / atom.m` (= `m_iter`; **1** for a single atom) |
   | **N** lane count | `atom.n` (traits `n`) — one N per lane per atom-column (`c_n_per_lane ≡ 1`) |
   | **N** across-atom run `n_sub` | `wave_n / atom.n` (= `n_iter`; **1** for a single atom) |

2. Factor the free axes: **M** into `(P, R, V, m_sub)`, **N** into `(atom.n, n_sub)`.
3. **Lane** = `R` (M) × `atom.n` (N) — together `R·atom.n = wave` (the precondition). **Registers** =
   `P·V·m_sub` (M) × `n_sub` (N). Every level is claimed exactly once (what the bijection validator checks),
   so **no unmerge is needed for the accumulator descriptor itself**. ✗ An epilogue that CHUNKS the
   accumulator still needs one: the chunk's rank-3 form carries the lane level's place value in a tensor stride.
4. Confirm slot-by-slot against `derive_c_distribution` (the ground truth), then `classify_transform` the
   C-shuffle against the **same lane's** store order (§8 — a verdict against a target that itself re-owns
   lanes is self-consistent and answers a different question).

**General identities** (any atom / tile): `wave_m = P·R·V·m_sub` (= `atom.m·m_sub`), `wave_n = atom.n·n_sub`;
total C registers per lane = `(P·V·m_sub)·n_sub`. Register significance (major → minor) = `m_sub`, `n_sub`, `P`, `V`.

Worked from the registered traits — single-atom, then multi-atom, then the `P>1 ∧ free_sub>1` composition:

```
single atom (m_sub = n_sub = 1):
  16×16  mfma_f32_16x16x16f16:  P=1 R=4 V=4  ->  R·atom.n = 4·16 = 64 = wave ✓;   4 C regs/lane
  32×32  mfma_f32_32x32x8f16:   P=4 R=2 V=4  ->  R·atom.n = 2·32 = 64 = wave ✓;  16 C regs/lane

multi-atom (free_sub > 1):
  16×16 atom, 64×64 tile (m_sub = n_sub = 4):  P=1 R=4 V=4
     wave_m = P·R·V·m_sub = 1·4·4·4 = 64;   wave_n = atom.n·n_sub = 16·4 = 64;   regs/lane = (P·V·m_sub)·n_sub = 16·4 = 64
  32×32 atom, 64×64 tile (m_sub = n_sub = 2):  P=4 R=2 V=4     (P>1 AND free_sub>1)
     wave_m = P·R·V·m_sub = 4·2·4·2 = 64;   wave_n = atom.n·n_sub = 32·2 = 64;   regs/lane = (P·V·m_sub)·n_sub = 32·2 = 64
```

**`P == 1` is the degenerate case of this same construction** — single-patch and multi-patch are one code
path, not two.

Where this has been run the chain came out sound with **zero `cross_lane`** — a property of the encoding, so
*constructibility* transfers across dtype and wave size. `f64` is **constructible and proven cross-lane-free**
by the CPU proof sweep (the registered `mfma_f64_16x16x4f64` row); only GPU **bit-exact** validation is still
per-instance. Re-derive rather than assume, especially at a different wave size.

**Operand-side precondition — check it PER OPERAND (one side can pass while the other fails).**
`free_lanes · k_lanes` must **divide** `wave_size`, with `k_lanes = k / k_ab_per_lane`, `free_lanes = m` on
the A side and `n` on the B side. The quotient is the operand **replication** factor: **1** on CDNA / gfx12,
**2** on gfx11 WMMA — there a lane holds the atom's whole K (`k_lanes == 1`) and the operand is duplicated
across the two 16-lane halves (identical data, no cross-lane; the duplicate is a replication axis, §6). A
**square** atom satisfies this on both sides (and the accumulator precondition above, so the two never fail
together); a **non-square** atom can over-subscribe one side (`free_lanes · k_lanes > wave_size`, no integer
quotient) and then does not interleave — the deferred case (task #21).

**Which catalogue rows interleave is enumerated and asserted by the proof sweep** `test_interleave_all_dense.py`:
it drives the real construction over every dense row and locks the population, so it is the executable
census — trust it, don't copy a count here (that is what goes stale). The durable rule is the one above:
**every dense square row interleaves; the lone non-square row does not.**

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

## The layout-choice recipe (thread-tile transpose flow)

This recipe answers: **given the memory layouts + dtypes of A, B, and C, choose the highest-bandwidth way
to move the data through the MMA.** It is what §10's cost model optimizes over; its runnable form is
`layout_optimizer.py`. It rests on two facts (derived in §1/§6 and §2):
- **The thread-tile is a rectangle** (A `M×K`, B `N×K`, C `M×N`) → its elements lay into registers two ways
  (row/col-major), and **row↔col-major is a pure in-register reorder** (a transpose; §1/§6) — no re-load, no
  cross-lane. (row/col is operand-relative: K-contiguous = row-major for A, col-major for B.)
- **MMA-ready = K-vectorize both operands** → an A run is *fixed M, K running*, a B run *fixed N, K running*;
  pairing slot-for-slot is the sound MAC (§2, `mma_is_machinery.md`).

### The five-step recipe

It is exactly §4's end-to-end flow — **coalesced load → transpose to MMA-ready (if needed) → issue →
interleaved C → transpose to store order → wide coalesced store** (see §4 for the steps). Two things this
recipe adds on top of §4:
- **Step 2 has a happy case and a mismatch case.** *Happy:* A row-major `M×K` × B col-major `K×N` loads
  coalesced **and** is already MMA-ready → zero reorder. *Mismatch* (e.g. A col-major, M contiguous): load
  coalesced in the memory-native order, then **one cheap in-register transpose** to K-contiguous — never a
  strided load.
- **C is also a rectangle (`M×N`),** so making it store-compatible is the **same in-register transpose** as
  A/B (within the already-interleaved ownership; the §7 canonical trap does not apply here).

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
| **lds_read** | read contiguous on the LDS innermost | — | N-way bank conflict (read stride vs `NB` banks, 32 on gfx90a) |
| **c_epilogue** | native accum **order** already == store order (aligned issue-order) | in-register `reorder` (dtype-graded) | LDS reposition, else register `cross_lane` (round-trip for cross-lane C) |

**Tier ranking (cheapest → invasive; register tiers defined in §8, the SOT; the LDS-reposition rung is defined in §7a):** free-symmetry(0) < `reorder`-dword-aligned < `reorder`-sub-dword
(∝ 1/dtype-size) < **LDS reposition** (conditional — only if the data already transits LDS; else can exceed a register `cross_lane` on a small tile) ≪ `cross_lane` (reject/defer seam). Smaller dtype ⇒ sub-dword reorder gets pricier ⇒
the value of reaching a chain via a **free symmetry rises** (symmetry moves nothing at any dtype).

### Chains + findings (the MECHANISM; measured perf lives in the per-kernel design record, not the SOT)
| chain | transform | lds_read | c_epilogue | outcome |
|---|---|---|---|---|
| interleaved / interleaved | none (matched) | M-innermost free-dim (`ds_read2_b32`); low conflict WITHOUT a pad | free(0) if issue-order==store order, else one in-register `reorder` (native ownership==store, intra-lane) | **model winner** (ranking validated) |
| canonical / canonical | none (matched) | K-innermost (`ds_read2_b64`, wider/op); K-row aliases banks → fixable conflict (measured via `/bank-conflict`, outside the `lds_banks.md` §1.5 b32 read model; pad to fix) | none (RCC direct store) | baseline (honest = PADDED) |
| interleaved / canonical (crossed) | A↔B swap + `c_transpose` (free-symmetry: routing + reposition, §8, DERIVED) | — | round-trip only if output ≠ native | not yet built |
| canonical / interleaved (crossed) | ″ | — | in-register (interleaved C) | not yet built |

**Findings (the durable MECHANISM — measured perf/counters live in the per-kernel design record + `/bank-conflict`, never the SOT; measured on gfx90a/CDNA2, so `MFMA`/`MfmaUtil` below are the CDNA matrix op and its util counter — the RDNA analogue is the WMMA-pipe util):**
- **Interleaved beats canonical (RCR):** interleaved's M-innermost free-dim LDS read has low conflict WITHOUT
  a pad; canonical's K-innermost read has a K-row stride that is a multiple of `NB` dwords (`NB`=32 on gfx90a) → K rows alias the
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
  (dword-cheap at f16, hidden in the MMA shadow); LDS→MMA none (matched); lds_read low-conflict (free-dim);
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

See **`lds_banks.md`** — the SOT for the LDS bank model (per-served-group (`ArchLDS.HALF`, per-arch) × per-dword-phase arbitration),
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
- `helpers/tiling/traits/mma_traits.py` — the traits / catalogue registry (128 atom rows); every §9
  construction quantity is read from here.
- `helpers/tiling/transforms.py` — `interleave_idx`, `k_distribution`, `classify_transform`,
  `validate_operands`, `derive_c_distribution` (the §9 accumulator ground truth).
- `helpers/tiling/mma/styles/interleaved.py` — `InterleavedStyle`: the shipped interleaved construction
  (`lds_bridge` / `operand_desc` for A/B, `accumulator_desc` for C). The API this document describes.
- `helpers/tiling/mma/warp_encoding.py` — canonical `a_/b_warp_encoding`: the atom-native reference the
  soundness gate reconstructs from (the interleaved operand is `InterleavedStyle`, not a flag here).
- `helpers/tiling/kernels/latency_probe.py` — instruction-latency microbench feeding §10's cost model.
