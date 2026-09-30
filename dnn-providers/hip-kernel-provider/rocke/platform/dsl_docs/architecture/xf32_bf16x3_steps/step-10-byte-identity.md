# Step 10 — byte-identity, both LLVM flavors

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
step 10). Ticket AICK-2208.

## Contents

- [What this step had to establish](#what-this-step-had-to-establish)
- [Results](#results)
- [The one DRIFT is pre-existing](#the-one-drift-is-pre-existing)
- [Why no golden needed re-blessing](#why-no-golden-needed-re-blessing)
- [The gate does not tolerate concurrent runs](#the-gate-does-not-tolerate-concurrent-runs)
- [Verification](#verification)

## What this step had to establish

Byte-identity is the repo's #1 invariant: the Python engine and the C++ engine
must emit the same LLVM-IR bytes for every kernel family.
[Steps 5+6](step-05-bf16x3-lowering.md) changed emission in **both** engines,
which is exactly the change shape the invariant exists to police. Two questions
followed, and the plan deliberately left them to the end:

1. Do the two engines still agree, at both LLVM flavors?
2. Does any golden need re-blessing — i.e. did an existing kernel's output
   move?

The second was an open question rather than an assumption. The expansion hooks
into `_op_tile_mma`, which every MMA-bearing kernel in the tree goes through,
so "only emulated atoms are affected" had to be measured, not asserted.

## Results

Four full runs of `tools/check_byte_identity.py`, every family, both flavors,
on the branch and on its merge base:

| tree | flavor | GREEN | DRIFT |
|---|---|---|---|
| `develop` (merge base) | llvm20 | 69 | 1 |
| `develop` (merge base) | llvm22 | 69 | 1 |
| this branch | llvm20 | 69 | 1 |
| this branch | llvm22 | 69 | 1 |

Identical on all four, down to the same single family. The engines agree, and
no family's emission moved.

## The one DRIFT is pre-existing

`gfx942_attention_tiled_3d` diverges between the engines on 1 of its 5
configs. It does so on `develop` too, at both flavors, so it predates this
work and is untouched by it. The gate therefore reports `RESULT: RED` on this
branch — and reports exactly the same RED on the merge base.

Stating this plainly rather than burying it: **the byte-identity gate is not
green in this tree, and this change does not make it green.** What the step
establishes is the narrower claim the change is responsible for — that the
branch is byte-identical to its merge base in what it emits, and adds no new
divergence.

## Why no golden needed re-blessing

No golden kernel selects an emulated atom, so none of them changed:

- nothing under `instances/`, `helpers/` or `library/` references `xf32` at
  all — the identifier appears only in `core/`, `cpp/core/`, the arch data and
  the tests;
- [step 7](step-07-selection-regression.md) pins that no query predating the
  emulated rows can return one, over every arch and every shipped atom;
- `_expand_emulated_mma` returns `false` for a native atom before touching
  anything, so a kernel that selects a native atom takes a path it already
  took.

The measurement above is what turns those three into a fact rather than an
argument.

## The gate does not tolerate concurrent runs

Worth recording, because it produced a false result that cost a round of
investigation. The llvm20 and llvm22 runs were first issued **in parallel**
against the same source tree. The llvm22 run then reported two extra DRIFTs —
`add_rmsnorm2d_bf16` and `add_rmsnorm2d_rdquant`.

Neither reproduced. Both families came back GREEN when scoped with `--only`,
and the full llvm22 run came back 69/1 when re-run alone. The gate builds its
engine archive from the tree it is pointed at, so two concurrent runs sharing
that tree interfere with each other.

Run the flavors **sequentially**. A DRIFT that appears only under concurrency
is an artifact, and the two families it named here had no MMA in them at all —
which is the signal that should prompt a re-run rather than a bisect.

## Verification

- `python tools/check_byte_identity.py` — 69 GREEN / 1 DRIFT.
- `ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py` — 69 GREEN /
  1 DRIFT.
- Both re-measured on the merge base and matching exactly.
- No golden re-blessed; none needed it.
