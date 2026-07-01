# Exec-Mask Grouping

`collapseExecMaskedRegions()` / `expandExecMaskedGroups()` (in
`ExecMaskGrouping.hpp`/`.cpp`) let `StinkyDAGSchedulerPass` schedule around a
narrowed-`exec` span without reordering into or out of it.

## The problem

A narrowing write to `exec`/`exec_lo` followed later by a full-mask reset behaves
like `if (cond) { ... }` for vector lanes: every VALU/DS/global instruction in
between only affects the currently-active lanes. Ordinary vector instructions don't
carry an implicit `exec` read (see the "does not include VOPs where EXEC solely
selects active threads" note in `Gfx1250Instructions.def`), so the DAG scheduler has
no dependency edge tying them to that window. Nothing stops it from reordering work
into or out of the span, or from moving guarded instructions out of order relative
to each other.

Flagging every vector instruction as an implicit `exec` reader would fix this but
would also serialize nearly everything relative to `exec`, defeating the scheduler's
whole purpose in blocks that never narrow `exec` at all.

## The fix

Collapse the whole narrow-write..reset span into one opaque `ExecMaskGroup`
instruction before scheduling. Its `getSrcRegs()`/`getDestRegs()` are the union of
its children's operands (which already includes `exec` itself, since the narrowing
write and the reset both target it explicitly), so the *existing* RAW/WAW/WAR graph
builder in `StinkyDAGSchedulerPass.cpp` orders it correctly against its neighbors
with no changes there or in `CDNA5.hpp`. Nothing can be scheduled inside the span
because it's never decomposed into separate DAG nodes in the first place.
`expandExecMaskedGroups()` puts the original instructions back, in their original
relative order, once scheduling is done.

`ExecGroupData` (in `StinkyModifiers.hpp`) stores raw `StinkyInstruction*` pointers
to the original children, not copies. "Restore exactly" is then just re-linking the
same objects back into the `BasicBlock`'s list — nothing about a child (registers,
modifiers, cost, def-use `users`/`sources` links to instructions elsewhere in the
function) is ever touched, so there's no reconstruction step that could drift from
the original.

## Detection

`collapseExecMaskedRegions()` recognizes only the "narrow write, ..., full-mask
(`-1`) reset" idiom, matched purely by destination register identity (so it doesn't
care whether the write is `s_mov_b32` or something else). Nested spans close on
their own matching reset (depth-tracked) before the outer span closes.

**Single-BB only, by design.** A span is never matched across a `BasicBlock`
boundary. A BB is already the real control-flow partition — a lane-mask span
crossing that boundary would mean the mask has to survive a branch, a different
problem than intra-block lane predication. An unmatched narrow write (including one
whose reset would be in a successor BB) is left ungrouped rather than searched for
across the CFG — this silently falls back to today's unfixed scheduling behavior for
that span rather than failing loudly, so `ExecMaskGroupingTest.cpp` pins it down
explicitly (`CollapseExecMaskedRegions_LeavesUnmatchedNarrowUngrouped`).

**Known scope gaps**, not yet handled:
- The save/restore-from-sgpr idiom (`s_*_saveexec_*`/`s_*_wrexec_*`) isn't
  recognized — only a literal `-1` reset closes a span.
- Whether MFMA/WMMA inside a guarded span should get CDNA5's WMMA-aware scheduling
  (today it doesn't — it's swallowed into the opaque group like everything else).

## Side-effect inheritance

`ExecMaskGroup`'s own descriptor carries no `IF_HasSideEffect` — but if a guarded
span happens to swallow an instruction that `hasSideEffect()` would flag on its own
(a real store, a non-token barrier, a wait, a branch), the group must report that
too, or `StinkyDAGSchedulerPass.cpp`'s region-splitting scan would never know to pin
it and would treat the whole group as freely movable. `hasSideEffect()` handles this
by checking an `ExecMaskGroup`'s children and returning true if any of them would.

This makes the *entire* group non-movable relative to its neighbors, not just the
side-effecting child — there's no mechanism to partially decompose a group, so this
is the conservative, correct choice rather than an optimization gap. In practice
this case should be rare: a hard side effect inside a lane-masked span isn't the
expected shape.

## Testing

- `ExecMaskGroupingTest.cpp` — unit tests of `collapseExecMaskedRegions()`/
  `expandExecMaskedGroups()` in isolation: round-trip fidelity, nesting, the
  unmatched-narrow-write fallback, and multiple sibling spans.
- `DAGSchedulerPassTest.cpp` (`ExecMaskGroup_*` tests) — whether the scheduler
  treats a hand-built `ExecMaskGroup` as a single atomic node: it must land strictly
  between a real producer/consumer pair per its declared operands, and it must not
  be misclassified as WMMA/DS/barrier/VALU by `CDNA5ReadyQueue::push()`.

## See Also

- [Architecture Overview](architecture.md)
- `src/transforms/asm/StinkyDAGSchedulerPass.cpp` — the scheduler that calls these
- `src/transforms/asm/dag/CDNA5.hpp` — unmodified by this; groups fall into its
  generic `otherQueue` bucket
