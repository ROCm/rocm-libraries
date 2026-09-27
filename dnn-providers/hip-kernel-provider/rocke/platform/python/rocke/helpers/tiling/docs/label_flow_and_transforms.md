# Label Flow & Transforms — SOT

How a datum's **identity** moves through a kernel's dataflow: `global → LDS → registers → MMA → epilogue →
store`. General to ANY tiling kernel (GEMM, elementwise, conv, SDPA). The atom-local case (POSITION ≠ LABEL at
one MMA output) is in `mma_is_machinery.md`; this generalizes it across spaces.

## The law

> **A LABEL is a datum's identity. It flows INVARIANT. A transform changes WHERE a datum lives (its
> destination coordinates), never WHAT it is (its label).**

One exception: an **explicit relabel** (below). Nothing else changes a label.

**The trap:** never derive a label from a position. `A[m,k]` stored "transposed" into LDS keeps its label
`(M=m,K=k)`; only its bank/depth moved. Reading the label back off the transposed position (`(K,M)`) is *the*
recurring bug.

```
 datum A[m=2,k=1]      LABEL stays (M2,K1) everywhere      only the POSITION changes
 ───────────────      ───────────────────────────────      ─────────────────────────
 registers   (lane 5, reg 3)  ── reposition ──▶  LDS (row 1, bank 2)  ── read ──▶  registers (lane 9, reg 0)
   label:      M2,K1                                 M2,K1                              M2,K1     ← invariant
```

## Reposition: the transpose lives in the POSITION, not the label

A `4×2` A tile (M=0..3, K=0..1). Same labels in registers and in LDS — only the grid they sit in transposes.

```
 REGISTERS  (fwd_map: (lane,reg) → label)        LDS  (M along banks, K along rows)
        reg0    reg1                                     bank0  bank1  bank2  bank3
 lane0  M0K0    M0K1                              row0/K0  M0K0   M1K0   M2K0   M3K0
 lane1  M1K0    M1K1                              row1/K1  M0K1   M1K1   M2K1   M3K1
 lane2  M2K0    M2K1
 lane3  M3K0    M3K1        labels IDENTICAL ─────────────▲ same M,K cells, transposed placement

 → the store arrow is a REPOSITION (free, addressing) — NOT a relabel.
 ✗ BUG: labeling row0/bank1 as "K0,M1" — that is a label DERIVED FROM A POSITION.
```

## The edge kinds — how a datum MOVES between two spaces

All **preserve the label**; they differ only in *which destination coordinate changes* and *its cost*.

```
 kind         changes            cost / mechanism                 renders as
 ─────────    ───────────────    ─────────────────────────────    ─────────────────────────
 reposition   address (space)    FREE  — addressing (no op)        into the space (LDS); no new regfile
 reorder      register           PRICED — VGPR shuffle             a NEW register file
 cross_lane   lane               PRICED — DPP / ds_bpermute / LDS  a NEW register file
 relabel      the label VIEW     DECLARED — FlowStage(relabel=True) no data moves; a declared re-view
```

The fourth is the odd one out and is why this table used to list three: `relabel` moves no datum at all — it
re-views a FINISHED tile's axes into a downstream input frame. It is an edge kind because `describe_edge`
classifies it and because it is the ONE sanctioned label change (see below); it is not a movement.

A store/read is a **reposition**: it changes a datum's physical **storage-axis alignment / address**, never
its label. **Never express a placement change in LABEL vocabulary.** Writing `(M,K)→(K,M)` on a store arrow is
*the* recurring bug: the LDS memref's `(K, free)` axis order is a **positional** convention, not a relabel of
the datum. When a store's free axis is already innermost (CRC's free-contiguous gift) even the storage
alignment doesn't move — it is the identity reposition (`place into lds; free (label invariant)`). A genuine
register realignment (row-major kernels) is a *priced* `reorder`/`cross_lane`, classified by
`classify_transform` — still label-invariant, never a label transpose.

**The free symmetry is a coordinate transpose (reposition), NOT a "relabel."** `col_major(X) ≡ row_major(Xᵀ)`
is the same data with its **coordinates** transposed for free via addressing — which physical axis is stride-1
flips; the **label rides invariant**. This is a *reposition symmetry*, the cheapest transform tier. (Older
docs call it "free-relabel symmetry" — a misnomer; it moves a position, not a label. Only the C-reuse case
below is a relabel.)

## Two things that are NOT a datum's label changing

**Source-swap (AB-swap): machine-input ROUTING, not a relabel.** Feeding operand A into the machine's B-slot
and B into the A-slot is a **wiring** choice — it changes WHICH SLOT consumes a datum, not the datum's label
*or* distribution. Labels ride invariant:

```
 tensor A regs (label M,K) ─┐ routed to the  ┌─ machine coupling is FIXED ─┐
                            ├─ B-slot         │  A-slot(N,K)·B-slot(M,K)     │──▶ C emerges at a FIXED position,
 tensor B regs (label N,K) ─┘ routed to the  │  accumulate over shared K    │    (N,M)=Cᵀ, LABELED by
                               A-slot         └─────────────────────────────┘    derive_c_distribution — never
                                                                                  a hand M↔N swap
```

The A-slot carries **(N,K)** [tensor B's data], the B-slot carries **(M,K)** [tensor A's data]; the machine
contracts over the shared K and **C's (M,N) DERIVES** (position ≠ label). Signature of routing (vs a relabel):
the fragments are passed in **swapped arg order** with **no `transform_fragment`**, and C is stored transposed
(`c_td.permute([1,0])`, Cᵀ). *Proof: base and ab_swap both compute `C=A·Bᵀ` bit-exact (`max_abs_diff=0.0`); a
label mutation would give a wrong/transposed result.* AB-swap is therefore **not** an edge kind and **not** a
relabel — the labels never move.

**Relabel: the one DECLARED label re-view (a FINISHED tile reused as input).** The only sanctioned label
change is reinterpreting an **already-computed C tile's** `(M,N)` axes into a downstream input frame
`(M,K)`/`(N,K)` — a bijection re-view (axis permutation + rename) of a finished tile, declared explicitly
(`FlowStage(relabel=True)`), never a mutation of a datum mid-flow, and it never adds or drops a label. (Not
AB-swap — that is routing, above.)

## Enforcement (code, not convention)

- `transforms.describe_edge(src, tgt, *, src_dims, tgt_dims, to_space=None, relabel=False) → (kind, why)` —
  classifies the four kinds with a mandatory `why`. `transform_note` routes reg↔reg arrows through it.
- `Pipeline.check_label_invariance()` runs inside `render`/`render_panels` and raises **`LabelMutationError`**
  if a stage introduces a label absent upstream without `FlowStage(relabel=True)` (label-**set** preservation).
- Square-tile transpose (same set) is caught at the store *source* by the per-`(lane,reg)` assert
  `reg.fwd_map == lds.flow_map`.
- `FlowStage(relabel=True)` is the ONE declared escape; it carries a `why`.

## Stage classes — the general decomposition of a tiling kernel

A **stage class** is defined by **which space pair it moves a datum between**, never by what the kernel
computes. That is why the set is general: GEMM, reduction, scan, conv and attention differ in their COMPUTE
class and in how many axes they iterate, not in the space pairs available.

| # | class | space pair | GEMM instance | reduction instance | mandatory |
|---|---|---|---|---|---|
| 1 | STAGE | global → LDS | cooperative prefetch + LDS store | input tile stage | no |
| 2 | DISTRIBUTE | LDS → registers, + the in-register bridge | wave read + operand reorder | per-worker slice | no |
| 3 | COMPUTE | registers → registers, **or registers → LDS → registers** for a cross-worker step | MMA | tree / shuffle reduce | **yes** |
| 4 | ITERATE | loop + buffering + sync | K-loop, buffers, barriers | reduction-axis loop | no |
| 5 | COMBINE/WRITEBACK | registers → global (± read-modify-write) | C-shuffle + `beta*C` + store | cross-worker combine | **yes** |

Only COMPUTE and COMBINE/WRITEBACK are mandatory: a kernel must compute something and must write it out.
Everything else is an optimisation — a kernel that reads straight from global to registers has no STAGE class,
and one that fits its whole problem in registers has no ITERATE class.

**The classes are NOT equally observable, and a report must say which is which.** Only the decorated verbs
produce nodes (`load_fragment` / `store_fragment` / `fill_fragment` / `transform_fragment`, and `TileMma`):

| class | what records | what does NOT |
|---|---|---|
| STAGE | two transactions (global load, LDS store) | — |
| DISTRIBUTE | the LDS load (a transaction) + the bridge (a `PipelineOp`, **not** a transaction) | — |
| COMPUTE | an MMA; `fill_fragment` records *that* an accumulator was initialised | non-MMA arithmetic — invisible to the recording AND to `witness`; and `fill` carries no scalar field, so "initialised to what" is unobservable |
| ITERATE | the **buffer count**, as the round-trip halves returned per LDS space | sync placement and loop structure — barriers produce no node. ✗ `lds_spaces()` counts SPACES (one per operand), not buffers |
| COMBINE/WRITEBACK | its stores | its arithmetic |

✗ Do not let a five-class diagram read as five-class evidence. Mark asserted classes as asserted.

## Gating a bridge before it runs

A **bridge** is one edge: the delta between a producer's register frame and its consumer's. Three checks
decide whether a bridge is well-formed, and they are cheap enough to run on every candidate config. They are
the *label-flow* half of layout validity; the memory half is `tiling_interleaving_design.md` §2b, and the
compute half is the unit's own contract (`mma_is_machinery.md` → *MMA-acceptability: per-operand soundness +
a pairwise K-match*, for an MMA).

### Bridge-kind is not edge kind

Two taxonomies, both needed, routinely conflated:

| term | values | what it decides |
|---|---|---|
| **bridge-kind** | `permuting` · `element-count-changing` | *which checks run* |
| **edge kind** | reposition · reorder · cross_lane · relabel | *what verdict to expect* (above) |

The test is on the **element SET, not the count**: the two encodings' per-lane element sets must be equal.
A bridge that keeps the count but changes the set — a shifted window, a gather, a same-size broadcast — is
`element-count-changing`.

**What the API does today (verified, not assumed):** `classify_transform` **RAISES**
`ValueError: cannot transform between layouts that hold different elements` for exactly that case — the global
element-set guard is `_classify_maps`. So a mis-routed bridge does **not** silently return `cross_lane`; it
blows up. ✗ Do not write that it returns a spurious verdict — it does not. The gap is that there is **no
predicate to ASK with**, so the only way to test the kind is to catch an exception, which is why the kind
ends up hand-declared.

> **Derive the kind; do not declare it.** It is computable from the two encodings, so a hand-declared field is
> a second source of truth that can disagree with the data. Where the API today forces a declaration (it
> raises rather than reporting), that is a defect to fix, not a convention to keep.

### The three checks

1. **Producer/consumer order match** *(permuting bridges)*. A consumer either declares a required per-lane
   order as a descriptor, or declares it has none. **Look THROUGH order-agnostic consumers to the first
   downstream consumer that DOES declare one**, and name that consumer in the verdict.
   ✗ Testing only the immediate consumer hard-stops correct kernels: a transform that exists for a store two
   stages later looks gratuitous to an elementwise stage sitting in between.
   If nothing downstream declares an order, the check becomes *the bridge must be the identity* — which is
   what catches a genuinely pointless reorder.

2. **Ownership preservation** *(permuting bridges)* — **store the TARGET each verdict was classified
   against**. A `reorder` verdict against a target that itself re-owns lanes is self-consistent and tells you
   nothing.

   > Checks 1 and 2 are **ONE `classify_transform` call with two assertions**. Listing them as two checks
   > invites two implementations that drift apart.

3. **Bijection / coverage** — **per DESCRIPTOR, and it always runs**, on every descriptor in the recording,
   whatever any bridge's kind is. It is a property of one descriptor, not of a movement.
   ✗ Scoping it to permuting bridges leaves a reduction's or broadcast's output descriptor unchecked — and
   that is the descriptor most likely to be hand-built, precisely because no canonical model produces it.

**`cross_lane` is not a blanket failure.** For a reduction it is the mechanism. The expected edge kind is
predicted per bridge before recording, and the gate diffs against that prediction.

**Checks 1 and 2 have a domain.** In a kernel where every bridge happens to be a permutation the restriction
is invisible, which tempts everyone to call them universal. They are not: `classify_transform` on a reduce
raises `cannot transform between fragments of different dimensions`. A check advertised as "always runs" that
raises on the first reduction is worse than one that states its domain.

## Pointers (owned elsewhere — don't restate)

- `mma_is_machinery.md` — POSITION ≠ LABEL at one atom + the sound MAC (the atom-local case).
- `tiling_interleaving_design.md` — `reorder`/`cross_lane` tier **pricing**; interleaving is a consumer.
- `visualization_api_surface.md` — the viz API + physical-accuracy contract.
- `.claude/skills/layout-viz/SKILL.md` — the operational render guardrail.
