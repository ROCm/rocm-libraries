# Step 10 — byte-identity at every LLVM flavor

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
step 10).

## Contents

- [What this step had to establish](#what-this-step-had-to-establish)
- [Results](#results)
  - [The gap review found, and the family that closes it](#the-gap-review-found-and-the-family-that-closes-it)
- [The earlier DRIFT no longer reproduces](#the-earlier-drift-no-longer-reproduces)
- [Why no existing golden needed re-blessing](#why-no-existing-golden-needed-re-blessing)
  - [The parity family is blessed into the L5 anchor](#the-parity-family-is-blessed-into-the-l5-anchor)
    - [The `ll` entries are llvm20; the `ir_canonical` entries are flavor-free](#the-ll-entries-are-llvm20-the-ir_canonical-entries-are-flavor-free)
    - [The rest of that file](#the-rest-of-that-file)
- [The gate does not tolerate concurrent runs](#the-gate-does-not-tolerate-concurrent-runs)
- [Verification](#verification)

## What this step had to establish

Byte-identity is the repo's #1 invariant: the Python engine and the C++ engine
must emit the same LLVM-IR bytes for every kernel family.
[Steps 5+6](step-05-bf16x3-lowering.md) changed emission in **both** engines,
which is exactly the change shape the invariant exists to police, and
[step 11](step-11-tf32-operand-abi.md) changed what the expansion starts from. Two
questions follow, and the plan deliberately left them to the end:

1. Do the two engines still agree, at every LLVM flavor?
2. Does any golden need re-blessing — i.e. did an existing kernel's output
   move?

The second is an open question rather than an assumption. The expansion hooks
into `_op_tile_mma`, which every MMA-bearing kernel in the tree goes through,
so "only emulated atoms are affected" has to be measured, not asserted.

## Results

One full run of `tools/check_byte_identity.py` per flavor, run sequentially, on the
branch after it was merged with `develop`, the emulated atoms moved onto the TF32
operand type, and the BF16 split capped; and again after the latest merge of `develop`,
which is what the table records:

| flavor | families GREEN | DRIFT | supplementary gfx1250 pass (llvm23) |
|---|---|---|---|
| llvm20 | 78 | 0 | GREEN |
| llvm22 | 78 | 0 | GREEN |
| llvm23 | 78 | 0 | (is the main pass) |

The 78 include the base's native TF32 family (`tf32_mma`, 10 configs) and this work's
`xf32_bf16x3_mma` (2 configs); the one family more than the previous run (77) came in
with `develop`. The engines agree, and no family's emission moved.

When the work was first written, against an older base, the same runs were 69 GREEN /
1 DRIFT on both the base and the branch, down to the same single family.

### The gap review found, and the family that closes it

"No family's emission moved" is true but beside the point: **no other family in that
run selects an emulated atom**, so without more it never compares the two engines on
the expansion at all. The only engine-parity check that did is the `rocke_engine`
pybind case in `test_xf32_bf16x3_lowering.py`, which skips wherever only the archive is
built.

The [`xf32_bf16x3_mma`](../../../tests/instances/parity/xf32_bf16x3_mma_emit.py)
parity pair puts the expansion in the gate. It builds both emulated atoms through each
engine's builder from the bare op_id (load, wrap as TF32, emulated `tile.mma`, store
every lane), so `run_diff.py` byte-compares the BF16x3 text with nothing but the
archive. It is GREEN at llvm20, llvm22 and llvm23, in the `ll` lane and in canonical
`ir`.

To check the family is not vacuous, the first two BF16x3 terms were swapped in the C++
expansion **alone**, in a scratch copy of the tree. Against that archive the family
reports DRIFT on both configs at every flavor in the `ll` lane; the canonical `ir` lane
stays GREEN, correctly, since the IR handed to the lowerer did not change; and the real
archive stays GREEN.

## The earlier DRIFT no longer reproduces

The first version of this work found `gfx942_attention_tiled_3d` diverging between the
engines on 1 of its 5 configs, on `develop` too, and said so plainly: the gate was RED
on the branch and RED on its merge base, and the step could only establish that the
branch added no divergence of its own.

On the current base that family is GREEN on all five configs at every flavor, so the
gate is green outright. Nothing in this work touches it, and the divergence is simply
absent from the merged tree; the earlier finding is kept here as the record of why the
step's claim was once the narrower one.

## Why no existing golden needed re-blessing

No golden kernel selects an emulated atom, so none of them changed:

- the base's native TF32 probe selects native gfx942 atoms, which this work does not
  touch; nothing else under `instances/`, `helpers/` or `library/` selects a TF32 atom;
- [step 7](step-07-selection-regression.md) pins that no query predating the emulated
  rows can return one, over every arch and every shipped atom;
- `_expand_emulated_mma` returns `false` for a native atom before touching anything, so a
  kernel that selects a native atom takes a path it already took.

The measurement above is what turns those three into a fact rather than an argument,
and the representative-IR golden
(`tests/golden/rocke_representative_ir_sha256.json`) is not touched by this work. Its
check is clean for every case but one, `crosslane/gfx1201/warp_shuffle_xor_dpp_reduce_i32`,
which drifts at all three flavors on the merged tree and, with the identical old and new
hashes, on `develop` itself: the golden is stale against `develop`'s DPP lowering, which has
nothing to do with an emulated atom, and it is left for its owners to re-bless.

### The parity family is blessed into the L5 anchor

The parity family *is* also blessed, in both `differential/golden/llvm_gfx_all.json`
keys:

```json
"ll":           { "xf32_bf16x3_mma": { "0": "6eea0fa7...", "1": "2ea60615..." } }
"ir_canonical": { "xf32_bf16x3_mma": { "0": "24272fd3...", "1": "d1a1147a..." } }
```

The differential alone compares the two engines *against each other*, so a future
change that moved both the same way — a term reordered in `_expand_emulated_mma` and
mirrored into `cpp/core/`, which is exactly the edit the invariant asks for — stays
GREEN while the emitted bytes change. `--check-golden` reports unblessed configs as
informational, so without an entry that drift is silent. The anchor is the absolute
half of the pair: the engines agree *and* they agree with what was reviewed.

The shas are the Python reference emission. They were read from `--only` runs and
re-checked against full `--check-golden` runs — `--record-golden` replaces an entire
key, so blessing them with it would have silently unblessed every other family.

**The two `ll` entries have been re-recorded once, on purpose.** The expansion's
split gained a cap on its leading bf16
([step 5](step-05-bf16x3-lowering.md#saturation-the-top-of-the-finite-range)), which
changes what an emulated atom emits and so changes these shas — in both engines, in
the same change, GREEN against each other at every flavor. Before the entries were
updated, `--check-golden` reported exactly these two configs as changed and nothing
else; after, none. The `ir_canonical` entries did not move: that mode hashes the IR
handed to the lowerer, which the cap does not touch. No other family's sha moved,
which is the property the next section depends on.

#### The `ll` entries are llvm20; the `ir_canonical` entries are flavor-free

The anchor file carries no flavor axis, so each key holds exactly one sha per config and
the flavor it was taken at has to be stated rather than inferred. These are **llvm20**.
All three differ in the `ll` lane:

| config | llvm20 (recorded) | llvm22 | llvm23 |
|---|---|---|---|
| `mfma_f32_16x16x32_xf32` | `6eea0fa7` | `e2f3afcd` | `db1b5c8b` |
| `mfma_f32_32x32x16_xf32` | `2ea60615` | `8d916e45` | `f5fa7e2b` |

So `--check-golden` reports the family as changed in a ROCm 7.2 image, where the flavor
auto-detects to llvm22 — `ROCKE_LLVM_FLAVOR=llvm20` is what makes it match. (Auto-detection
reads the comgr vintage: ROCm 7.1.1 gives llvm20, 7.2 gives llvm22. An explicit
`ROCKE_LLVM_FLAVOR` produces the same bytes in either image, so the flavor, not the
container, is what the recorded sha is a function of.) llvm20 is also what the base's own
newest family is recorded at: all 10 configs of `tf32_mma` match at llvm20 in a full
`--check-golden` run.

`ir_canonical` has no such caveat: that mode serialises the kernel IR rather than the
lowered LLVM, so both configs hash the same under llvm20, llvm22 and llvm23 — measured,
not assumed.

#### The rest of that file

**Most of the older entries in that file do not match the tree, and did not before this
work.** A full `--check-golden` at llvm20 on this base (as of the latest merge of `develop`)
reports 633 of 888 blessed `ll` configs and 537 of 854 blessed `ir_canonical` configs
changed, none missing, and 236 and 314 configs present in the tree but not in the anchor.
By family, 12 of 73 `ll` families and 14 of 70 `ir_canonical` families match entirely;
`tf32_mma` and `xf32_bf16x3_mma` are among them. `ir_canonical` is flavor-free, so
this is not a flavor mismatch: the anchor is add-only in practice — a family gets its
entries when it lands, as `tf32_mma` and this one did, and older entries are not
refreshed. Re-blessing it is a tree-wide decision for its owners, not something to fold
into this change; the two entries above were merged in textually for that reason, and the
file's other bytes are untouched.

## The gate does not tolerate concurrent runs

Worth recording, because it produced a false result that cost a round of
investigation. The llvm20 and llvm22 runs were first issued **in parallel**
against the same source tree. The llvm22 run then reported two extra DRIFTs —
`add_rmsnorm2d_bf16` and `add_rmsnorm2d_rdquant`.

Neither reproduced. Both families came back GREEN when scoped with `--only`,
and the full llvm22 run came back clean when re-run alone. The gate builds its
engine archive from the tree it is pointed at, so two concurrent runs sharing
that tree interfere with each other.

Run the flavors **sequentially**. A DRIFT that appears only under concurrency
is an artifact, and the two families it named here had no MMA in them at all —
which is the signal that should prompt a re-run rather than a bisect. Each run also
makes the supplementary gfx1250 pass at llvm23; expect about twelve minutes per flavor
on a two-CPU host.

## Verification

- `python tools/check_byte_identity.py` at `ROCKE_LLVM_FLAVOR` llvm20, llvm22 and
  llvm23: GREEN, 78 families each, no DRIFT.
- The `xf32_bf16x3_mma` parity family, re-diffed against the final archive at every
  flavor in both `ll` and canonical `ir`: GREEN.
- The scratch-copy mutation described above: DRIFT on both configs in the `ll` lane at
  every flavor, GREEN on the real archive.
- `tests/golden/rocke_representative_ir_sha256.json` is unchanged. It checks clean at
  llvm20, llvm22 and llvm23 on every case except `crosslane/gfx1201/warp_shuffle_xor_dpp_reduce_i32`,
  which drifts identically on `develop` (same old and new hashes at all three flavors), so
  it is not this work's.
- A full `ROCKE_LLVM_FLAVOR=llvm20 --check-golden` run in each mode: the family's two
  configs match the recorded shas in both keys, as do the base's `tf32_mma` configs; the
  remaining mismatches are the older entries described above.
- With `rocke_engine` built, `test_xf32_bf16x3_lowering.py::TestEngineParity` also runs
  rather than skipping, so the expansion is compared across the engines by both routes —
  the parity family through the archive and the pybind module in-process.
- No existing golden re-blessed by this work; none of them moved because of it (the one drifting case above is `develop`'s own).
