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
```

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

## Pointers (owned elsewhere — don't restate)

- `mma_is_machinery.md` — POSITION ≠ LABEL at one atom + the sound MAC (the atom-local case).
- `tiling_interleaving_design.md` — `reorder`/`cross_lane` tier **pricing**; interleaving is a consumer.
- `visualization_api_surface.md` — the viz API + physical-accuracy contract.
- `.claude/skills/layout-viz/SKILL.md` — the operational render guardrail.
