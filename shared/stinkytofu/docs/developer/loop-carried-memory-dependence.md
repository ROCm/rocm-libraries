# Loop-Carried LDS Memory Dependence

Status: implemented end to end. Both dependence kinds are emitted by the
frontend and consumed by the backend; the SIA4 miscompile is fixed at PLR0 and
PLR2. Branch: `users/cycheng2/loop-carry-dep`.

One conclusion changed late and is worth reading before the rest: **the
`s_wait_tensorcnt 0` this produces in the main loop is the correct answer, not a
conservative fallback.** See §5, "The barrier drain is not over-draining".

## 1. The problem

A GEMM main loop rotates through N LDS buffers, advancing one step per trip.
Each memory op carries a `mod.memtoken`, but **a memtoken names a physical
buffer only for the trip it was emitted from**. The static tag never moves; the
buffer under it shifts every iteration.

Ring 3, reads at generation offset 0 and the TDM fill at offset 2:

```
                          iter 0   iter 1   iter 2   iter 3
  ds_load     (Read,  [0])  LDS0     LDS1     LDS2     LDS0
  tensor_load (Write, [2])  LDS2     LDS0     LDS1     LDS2
```

Both dependence directions are real and neither is visible to token overlap,
which compares `[0]` against `[2]` and finds nothing:

- **WAR**, `Write(k) = Read(k-d)` → `d = 1`. Iter 1 fills LDS0 while iter 0 may
  still be reading it. Needs `s_wait_dscnt`.
- **RAW**, `Read(k) = Write(k-d)` → `d = 2`. Iter 2 reads what iter 0 filled.
  Needs `s_wait_tensorcnt`.

In general `d * advance ≡ gdelta_earlier - gdelta_later (mod ring)`.

### Why it matters

`TDMPlusLdsBuf` (the 3rd LDS buffer) is force-disabled in
`Solution.py:assignDerivedParameters` with a TEMP kill switch, because of "an
unresolved cross-wave read-after-write race on the rotating LDS". Shipped
library logic already selects triple-buffer solutions and silently gets two.

There is also a **confirmed miscompile**. At `ScheduleIterAlg=4` +
`PrefetchGlobalRead=2` + `TDMPlusLdsBuf=1`, the kernel is wrong on hardware while
SIA0 is correct. Its main loop issues 32 `ds_load`s with **no `s_wait_tensorcnt`
anywhere in the body** — 3 in the whole kernel versus SIA0's 10. That is the RAW
above, unguarded.

`PrefetchLocalRead` is not part of the trigger. The same drop reproduces at PLR0
on the StreamK kernel (`examples/debug/sk-tdm3.yaml`: SK3 + TDMInst3 + PGR2 +
PLR0, 4 waves), whose loop has 8 `ds_load`s and the same empty tensorcnt. Two
very different bodies, one cause.

Root cause of the drop: SIA4 is the only path where StinkyTofu owns the waits.
`StinkyRemoveWaitCntPass` strips the incoming `s_wait_tensorcnt` because it is
classified `WaitReconstruction::WaitCntInsertion` ("I can rebuild this"), and
then `WaitDataflow` cannot — `restoreTensorState` overwrites `CK_Tensor` with the
sweep-0 snapshot, so the producing fill never survives the back edge. Strip plus
failed rebuild drops the wait.

## 2. Design

**The frontend emits the relation; the backend supplies the count.** TensileLite
already owns the rotation relabelling (`_ldsTokenBackEdgeMap`), so it names which
tag aliases and how many trips back. Only the backend knows the post-scheduling
issue order, so only it can turn that into a wait immediate. This keeps the ring
arithmetic in exactly one place, which is the failure mode that produced several
earlier bugs.

Rejected alternatives, and why:

- *Replace `MemTokenData`.* No. It answers "what storage does this instruction
  touch", and is the identity `StinkyBuildImplicitDependencyPass` turns into
  `LDS<n>` pseudo-regs. The new modifiers express a *relation to other tags* and
  need that base to be interpretable.
- *Reuse loopmodel's `OrderTokenData`.* No. Its contract is explicitly "orders
  but does not wait on"; `WaitDataflow` never reads it. It is a scheduling wall
  (`RegionDAG::orderWall`), which constrains *placement*. DS reads are
  asynchronous and `s_barrier` does not drain memory on gfx1250, so a wall does
  not make a read *complete*. The two are complementary and act at different
  pipeline stages (scheduler, then waitcnt).
- *Ship generations and derive hazards in the backend.* Most robust, but
  duplicates `FrameMap`/`FrameHazards`, which already exist in Python on the
  `loopmodel` branch.

## 3. What is implemented

### IR (`StinkyModifiers.hpp`)

```cpp
struct LoopCarriedWarData { std::vector<int> tokens; int distance = 1; };
struct LoopCarriedRawData { std::vector<int> tokens; int distance = 1; };
```

- **WAR** rides the *writer* (barrier or fill); `tokens` are the tags the
  aliasing **reads** carried `distance` trips ago. Scanned on `CK_DS`.
- **RAW** rides the *reader*; `tokens` are the tags the producing **fill**
  carried `distance` trips ago. Scanned on the producer's counter (`CK_Tensor`).

Serialized as `mod.loopcarriedwar` / `mod.loopcarriedraw`.

### Dataflow (`WaitDataflow.{hpp,cpp}`)

`PerPredQueue::ops` holds `QueuedOp{op, tripsBack}` rather than a bare pointer.
`mergeFromPredecessors` ages every entry by one when the edge does not advance in
RPO (`isBackEdge`), capped at `kMaxTripsBack = 16` so the lattice stays finite.

Both scans select entries at **`>= distance`**, which is exact and robust at
once: nearer entries do not alias and are skipped, and among the rest the one at
exactly the distance is the newest, so it sets the `min` — older ones only relax
it, and a capped counter can never cause a miss.

`LoopCarriedRawData` also opts its block out of the `CK_Tensor` freeze
(`needsLiveTensorState`, formerly `hasUntaggedTensorAnchor`). A declared
loop-carried tensor dependence is exactly the state `restoreTensorState`
discards, so the two cannot both apply. Every other block keeps the freeze, which
is why nothing else moved.

### Emitted-wait provenance (`StinkyWaitCntInsertionPass.cpp`)

`annotateLoopCarriedWar` tags the emitted `s_wait_dscnt` with
`// covers loop-carried WAR on LDS0 (1 trip back)`. Always accurate because
`computeRequiredWaits` takes the **min** over every dependency at the anchor, so
a wait emitted there is at least as strict as the WAR scan asked for.

### Frontend (TensileLite)

`rocisa` `MemTokenData` gained `warTokens` / `warDistance` and `rawTokens` /
`rawDistance`, lowered to `LoopCarriedWarData` / `LoopCarriedRawData` in
`ToStinkyTofuUtils.cpp`. Three emission sites:

1. `KernelWriter.py:~4834` — WAR on the manual `_syncThreads` barrier from
   #11779. One `_ldsTokenBackEdgeMap` lookup.
2. `KernelWriter.py:~11380` — WAR in `postMainLoopBarrierCheckAndReset` pass-2,
   needed because SIA4 **deletes** the manual barrier and rebuilds its own.
3. `KernelWriter.py:~11400` — RAW on the first read of each token in the loop
   body, from the same pass.

Site 3 needs `_carriedWriterOf`, which walks `_ldsTokenBackEdgeMap` up to a full
ring rather than taking one step. `buffer(t, k) == buffer(map[t], k-1)`, so after
`d` steps `map^d(token)` is the tag that named this buffer `d` trips ago; the
first such tag the body writes is the producing fill, at distance `d`.

**The single-step lookup is why the RAW was missed for so long.** For the ring-3
body that reads `[0]` and fills `[2]`, one step lands on tag `1`, which the body
never touches, so the phase comes back `standby` and nothing conflicts. The WAR
in the other direction resolves in exactly one step, which is what made the
single-step map look sufficient.

`_detectLoopHeadInfo` records a third field per token, `wroteInBody`, because the
walk asks "was this tag ever written" rather than "what was its last phase".

### Tests

`shared/stinkytofu/tests/filecheck/waitcnt_insertion_loop_carried_*`:

| test | pins |
|---|---|
| `war_annotated` | canonical case, `dscnt 2` |
| `war_already_drained` | annotation present, nothing carried in → no wait |
| `war_comment` | the `// covers loop-carried WAR` text, asm level |
| `war_single_wave` | no barrier; annotation on the fill; `NumWaves=1` |
| `raw_distance2` | RAW at d=2 → `tlcnt 1`, single wave, no barrier |
| `raw_distance1` | d=1 → `tlcnt 0`, and why that is exact, not a fallback |
| `raw_cross_wave` | the real shape: d=2 + barrier + `NumWaves=4` → `tlcnt 0` |

The last two exist to keep `tlcnt 0` from being read as a defect. Both are
correct answers arrived at by different routes, and neither should be "optimised"
into `tlcnt 1`.

## 4. What is NOT done

1. **WAW is unhandled**, and probably fine: the only WAW is at `d = ring`, both
   ends are on `CK_Tensor` which is `InOrder`, and it is transitively covered by
   RAW + WAR. The one real case would be cross-counter (`ds_write` vs
   `tensor_load` on one buffer), which no known config produces.
2. **`restoreTensorState` still exists.** Only opted out per-block. Removing it
   globally is the proper fix; blast radius across the suite is unmeasured.
3. **The ring buys no tensorcnt overlap in the multi-wave loop.** The wait is
   `tlcnt 0` once per trip, which is what SIA0 also emits, so the third buffer
   pays for itself in scheduling slack rather than in outstanding fills. Getting
   overlap back needs a *second* barrier in the body, not a looser immediate —
   see §5.
4. **The kill switch is still in `Solution.py`.** Lifting it needs a hardware
   run of both fixed configs, not just the codegen diff.

## 5. Traps

**The two halves are independent.** A barrier orders *waves*; `s_wait_*` retires
*this wave's* outstanding ops. `s_barrier` does not drain memory on gfx1250. Do
not assume one implies the other.

**The barrier drain is not over-draining.** `rawNeedsWait[CK_Tensor]` is
`isBarrier || numWaves == 1`, so every barrier drains the tensor counter. On a
rotating ring this looks obviously wrong — the barrier's own `CK_Tensor` edge
runs through the `LDS<n>` pseudo-reg to the previous trip's fill *on the same
tag*, which is a different physical buffer, the very confusion these modifiers
exist to correct. Narrowing it looks like free performance. It is not, and the
reasoning is worth following once:

`tensor_load_to_lds` is split across waves — even waves fill A, odd waves fill B
— while every wave reads both. So a fill must land **in the filling wave** before
the last barrier the **reading** wave crosses on its way to the consuming read.
`s_wait_tensorcnt` retires only the issuing wave's own ops, so a wait placed at
the reads cannot discharge that, at any immediate.

With one barrier per trip, the last rendezvous before trip *k*'s reads is the
barrier at the end of trip *k-1*, and `fill(k-2)` is the newest fill outstanding
there. "`fill(k-2)` has landed" is therefore exactly `tlcnt = 0`. A `tlcnt = 1`
would leave the other wave's `fill(k-2)` in flight and race.

This was tried. Suppressing the drain at write-phase barriers makes the annotated
loop emit `s_wait_tensorcnt 1` before the reads, which looks like the textbook
result and is cross-wave unsound. It also broke
`waitcnt_insertion_tensor_per_path_self_loop_anchor`, whose barrier-before-fill
is the same shape but on a *non-rotating* token, where the edge is real. Nothing
local to the barrier distinguishes the two.

Buying the overlap back means giving the body a second barrier — one before the
reads and one after, which is what SIA0 emits. A publishing barrier at the top of
the trip sees `[fill(k-2), fill(k-1)]` outstanding and can relax to `tlcnt = 1`.
That is a scheduling change, not a waitcnt change.

`raw_cross_wave` pins this. `raw_distance2` (`tlcnt 1`) is the same dependence
with `NumWaves=1` and no barrier, and the contrast between the two files is the
point.

**SIA4 deletes manual barriers.** `postMainLoopBarrierCheckAndReset` (gated on
`_StinkyTofuOptLevel == 3 && _ScheduleIterAlg == 0`, i.e. SIA4) strips every
workgroup `SBarrier` not containing `-3` and re-inserts its own from token phase
transitions. Comments and modifiers go with them. This is why
`"Waiting current LR finish for next GR(TDM)"` never appears at SIA4 — it is
emitted and then deleted. Any annotation on a barrier needs an emission site in
that pass too.

**Name only the nearest partner.** A fill can have several WAR partners at
different distances (see `war_single_wave`: `d=1` against one read group, `d=2`
against another). `CK_DS` is in-order, so a wait retiring the `d=1` reads retires
everything behind them. One `(tokens, distance)` pair suffices; adding a second
annotation is the wrong instinct.

The same holds on the RAW side, and it has a consequence that reads like a bug.
When two readers on one ring carry different distances, the nearer one binds and
the further one then finds nothing at `tripsBack >= distance` — correctly, since
`CK_Tensor` is also in-order. `raw_distance1` pins exactly that: a `d=1` reader
forces `tlcnt 0` and the `d=2` reader beside it emits nothing. Both the `0` and
the silence are right.

**"No wait emitted" is often correct.** Production loops frequently discharge
these hazards by accident — a `dscnt 0` for some WMMA's operands retires the
previous trip's reads before the fill issues. `war_already_drained` pins that.
Safe by accident of the schedule, not by construction: deepen register buffering
or hoist the fill and the drain vanishes while the hazard stays.

**Adding a wait can remove later ones.** `restoreTensorState` makes the pass
non-monotonic, and this is demonstrated, not theoretical. Preserving the StreamK
persistent-loop barrier (a `CK_Tensor` consumer, so it drains and trims the queue
in sweep 0) caused `restoreTensorState` to propagate that empty snapshot
downstream, and a `TDMPlusLdsBuf=0` StreamK kernel lost **three**
`s_wait_tensorcnt 0` — before the loop, *inside* the main loop body, and at
`toPGR1`. One added barrier, three guards deleted, on a kernel with no rotating
ring and no annotations anywhere near it.

The distinction that matters is **derived vs artifact**. `sk-tdm3` survives the
same barrier unharmed because its reads carry `mod.loopcarriedraw`, so
`needsLiveTensorState` is true for the loop block, the queue crosses the back
edge, and the in-loop wait is derived from the dependence. `bbs` has no
annotation (`TDMPlusLdsBuf=0`, tags do not rotate), so the freeze is active and
its in-loop wait was only ever a shadow of the sweep-0 snapshot. Perturb anything
upstream and it moves.

This is why preserving the StreamK barrier is gated on `TDMPlusLdsBuf == 1` — not
because triple buffering needs it more, but because that is the case where the
annotation has already made the block's tensor state live. Widening the gate
means first making that true everywhere, i.e. removing the freeze.

Two consequences. Any change that introduces a tensor-counter consumer needs the
whole suite re-diffed, not just the kernel it targets; and kernels that look
correct today may be relying on waits that exist only as an artifact of where the
sweep-0 snapshot happened to be taken. Removing the freeze (item 2 above) is no
longer a tidy-up — it is what makes results in this pass composable.

**`tripsBack` is lattice state.** It is part of `operator==`. Convergence was
verified (no cap-hit warnings across the suite), but changes here can stall the
fixed point and silently fall back to `s_wait_* 0` everywhere.

**Saturated ops lose their trip count.** Entries pushed past `kMaxInFlight` go to
`saturatedOps` (pointer-only) and are not scanned. Not a problem at current queue
depths, but it is an unsound edge if a partner ever falls out of the window.

## 6. Environment

Codegen runs in docker; host-side Python cannot import `rocisa`.

```bash
cd /data1/cycheng2/tickets/st-codegen-test
scripts/run-single-yaml.sh examples/test.cfg examples/debug/tdm3.yaml \
  --env TENSILE_ENABLE_TDMPLUSLDSBUF=1 \
  --env STINKYTOFU_DUMP_WAITCNT_ASM=/tmp/waitcnt
```

- `TENSILE_ENABLE_TDMPLUSLDSBUF=1` lifts the kill switch (**not committed** —
  the tree currently has the guard commented out entirely, which enables triple
  buffering for *every* kernel; restore before merging).
- `STINKYTOFU_DUMP_WAITCNT_ASM` enables a **TEMP DEBUG** dump pass in
  `Gfx1250Backend.cpp` (remove before merging). Output lands **inside the
  container**, not on the host.
- `Tensile.sh` rebuilds rocisa + stinkytofu each run, so C++ edits are picked up.
- Comments only reach the `.s` with `--preserve-comments`; the `.stir` serializer
  drops `CommentData` entirely.
- `--StinkyWaitCntInsertionPass=enableLoopCarriedTokenDeps` is the pre-existing
  conservative escape: it disables the `CK_Tensor` freeze globally. Still useful
  as a bisect oracle, though it now lands in the same place as the proper fix for
  this loop (`s_wait_tensorcnt 0` per trip) rather than being strictly worse.

Codegen diff from the fix, for reference. Both SIA4 kernels gain exactly one
instruction in the main loop and nothing else; SIA0 is byte-identical:

```
sk-tdm3.yaml  (SK3, PGR2, PLR0, 4 waves)   3 -> 4 s_wait_tensorcnt, 0 -> 1 in the loop
tdm3.yaml     (PGR2, PLR2)                 3 -> 4 s_wait_tensorcnt, 0 -> 1 in the loop
```

Backend-only iteration is much faster via `stinkytofu-opt` on hand-written
`.stir`, built at `/data1/cycheng2/tickets/build-opt`. Note that directory is
reached via a `/home/cycheng2` symlink; running `cmake .` from the `/home` path
re-resolves it and drops generated `intrinsics.st.bc`, breaking 8 unrelated
`IntrinsicFlowTest`/`IntrinsicExpansionPassTest` cases until a rebuild.

## 7. Related work

`loopmodel` branch (`/data1/cycheng2/loopmodel`, `user/chengchingwen/loopmodel`)
has a full Python GIR dependence framework under
`Tensile/Lowering/gir/analyses/`: `FrameMap` (forward dataflow over sets of phase
vectors), `FrameHazards` (RAW/WAR/WAW per frame with a `gap` and a `cross_agent`
flag), `FenceRegions`, `LdsBufferIds`. It computes everything described here and
more.

Its `TokensPass` still collapses the stamp to one frame, so what crosses into
StinkyTofu loses the generation — the same gap. Its `needing_fence()` returns
only `cross_agent=True` edges, leaving same-wave `gap >= 1` uncovered, which is
exactly the set these modifiers carry.

**If loopmodel lands, align to it**: derive both modifiers from the same
`FrameHazards` edge set rather than from `_ldsTokenBackEdgeMap`, and emit the
*whole* same-wave `gap >= 1` set rather than hand-picking. The PLR2 miscompile was
not a wrong annotation — it was a missing one, because a human chose which
hazards to report. Any curated design will keep failing that way, and quietly.

Also prefer loopmodel's carrier shape: it adds a separate `m_orderToken` slot on
`rocisa::Instruction` reusing the `MemTokenData` container, rather than extending
`MemTokenData` with extra fields as this branch does. That keeps `MemTokenData`'s
pickle format stable (ours changed `__getstate__` to a 5-tuple).

While extending it, the pure-Python adaptor's `__copy__` / `__deepcopy__` /
`__getstate__` turned out to be dropping `warTokens` and `warDistance` — they
round-tripped `tokens` only. SIA4 deepcopies whole instruction modules to build
the NoLoadLoop bodies, so that path could silently lose an annotation, which
presents as a missing wait rather than an error. Now fixed for all five fields;
if the carrier shape changes, keep the round-trip complete.
