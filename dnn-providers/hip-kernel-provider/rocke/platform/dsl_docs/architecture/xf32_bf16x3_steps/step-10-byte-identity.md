# Step 10 — byte-identity, both LLVM flavors

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
step 10). Ticket AICK-2208.

## Contents

- [What this step had to establish](#what-this-step-had-to-establish)
- [Results](#results)
- [The one DRIFT is pre-existing](#the-one-drift-is-pre-existing)
- [Why no existing golden needed re-blessing](#why-no-existing-golden-needed-re-blessing)
  - [The parity family is blessed into the L5 anchor](#the-parity-family-is-blessed-into-the-l5-anchor)
    - [The `ll` entries are llvm20; the `ir_canonical` entries are flavor-free](#the-ll-entries-are-llvm20-the-ir_canonical-entries-are-flavor-free)
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

### The gap review found, and the family that closes it

"No family's emission moved" was true but beside the point: **no family in
that run selects an XF32 atom**, so it never compared the two engines on the
expansion at all. The only engine-parity check that did is the `rocke_engine`
pybind case in `test_xf32_bf16x3_lowering.py`, which skips wherever only the
archive is built.

The [`xf32_bf16x3_mma`](../../../tests/instances/parity/xf32_bf16x3_mma_emit.py)
parity pair, added after review, puts the expansion in the gate. It builds both
emulated atoms through each engine's builder from the bare op_id (load, emulated
`tile.mma`, store every lane), so `run_diff.py` byte-compares the BF16x3 text
with nothing but the archive. Re-run on the branch:

| flavor | GREEN | DRIFT |
|---|---|---|
| llvm20 | 70 | 1 |
| llvm22 | 70 | 1 |
| llvm23 | 70 | 1 |

The one extra GREEN is the new family, identical at every flavor and in the
`ir` and `verify` lanes too. The DRIFT is the same `gfx942_attention_tiled_3d`
cfg[3]; a pristine-HEAD archive reproduces it with byte-identical C and Python
output for all five of its configs. To check the family is not vacuous, two
BF16x3 terms were swapped in the C++ expansion alone: both configs went
`MISMATCH` and the gate failed.

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

## Why no existing golden needed re-blessing

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

### The parity family is blessed into the L5 anchor

The parity family added after review *is* also blessed, in both
`differential/golden/llvm_gfx_all.json` keys:

```json
"ll":           { "xf32_bf16x3_mma": { "0": "1b73e8bd...", "1": "223ae6ef..." } }
"ir_canonical": { "xf32_bf16x3_mma": { "0": "4826b0c2...", "1": "2098f5c3..." } }
```

The differential alone compares the two engines *against each other*, so a
future change that moved both the same way — a term reordered in
`_expand_emulated_mma` and mirrored into `cpp/core/`, which is exactly the
edit the invariant asks for — stays GREEN while the emitted bytes change.
`--check-golden` reports unblessed configs as informational, so without an
entry that drift is silent. The anchor is the absolute half of the pair:
the engines agree *and* they agree with what was reviewed.

The shas are the Python reference emission, re-checked against a full
`--check-golden` run rather than only the `--only` run they were read from —
`--record-golden` replaces an entire key, so blessing them with it would have
silently unblessed every other family.

#### The `ll` entries are llvm20; the `ir_canonical` entries are flavor-free

The anchor file carries no flavor axis, so each key holds exactly one sha per
config and the flavor it was taken at has to be stated rather than inferred.
These two are **llvm20**, the flavor `CLAUDE.md` runs the byte-identity gate
at first. All three differ:

| config | llvm20 (recorded) | llvm22 | llvm23 |
|---|---|---|---|
| `xf32_bf16x3_f32_16x16x32` | `1b73e8bd` | `c5ca733a` | `1d857887` |
| `xf32_bf16x3_f32_32x32x16` | `223ae6ef` | `7e8a5f48` | `2c49a350` |

So `--check-golden` reports this family as changed in a ROCm 7.2 image, where
the flavor auto-detects to llvm22 — `ROCKE_LLVM_FLAVOR=llvm20` is what makes
it match. (Auto-detection reads the comgr vintage: ROCm 7.1.1 gives llvm20,
7.2 gives llvm22. An explicit `ROCKE_LLVM_FLAVOR` produces the same bytes in
either image, so the flavor, not the container, is what the recorded sha is a
function of.)

`ir_canonical` has no such caveat: that mode serialises the kernel IR rather
than the lowered LLVM, so both configs hash the same under llvm20, llvm22 and
llvm23 — measured, not assumed.

An earlier revision of this document said llvm20 and llvm22 agreed here and
only llvm23 differed. That was wrong, and wrong in the direction that matters:
the recorded shas were in fact llvm22's, taken from an unflavored run in a
7.2 image and labelled llvm20.

**The rest of that file is stale, and was before this branch.** A full
`--check-golden` on this tree at llvm20 reports 653 of 673 `ll` configs and
579 of 643 `ir_canonical` configs changed, including families this branch does
not touch. It is not the flavor caveat above writ large: `add_rmsnorm2d_bf16[0]` emits
`b64c6fef...` at llvm20, `edad2c15...` at llvm22 and `8335125c...` at llvm23,
and the anchor holds `102bd96c...` — no flavor reproduces it. The same three
shas come out of the merge base (`320f7a89`), so the drift predates this work;
the file was last re-blessed in September. Re-blessing it is a tree-wide
decision for its owners, not something to fold into this branch; the two
entries above were merged in textually for that reason.

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
- After review, with `xf32_bf16x3_mma` added: 70 GREEN / 1 DRIFT at llvm20,
  llvm22 and llvm23, run sequentially in an archive-only environment (no
  `rocke_engine`).
- Re-measured on the gfx950 node: the same 70 / 1 at all three flavors, the
  same single DRIFT, `xf32_bf16x3_mma` GREEN on both configs. With
  `rocke_engine` built there,
  `test_xf32_bf16x3_lowering.py::TestEngineParity` also runs rather than
  skipping, so the expansion is compared across the engines by both routes —
  the parity family through the archive and the pybind module in-process.
- No existing golden re-blessed; none needed it. `xf32_bf16x3_mma` added to
  both anchor keys and confirmed by a full `ROCKE_LLVM_FLAVOR=llvm20
  --check-golden` run in each mode: the family's two configs match the
  recorded shas and are absent from the mismatch list (that run's remaining
  mismatches are the pre-existing staleness described above).
- The `ll` flavor spread measured by emitting the family directly under each
  flavor in the ROCm 7.1.1 and 7.2 images: the same three shas in both, and
  the auto-detected flavor following the image (llvm20 / llvm22). The
  `ir_canonical` sha is identical under all three.
