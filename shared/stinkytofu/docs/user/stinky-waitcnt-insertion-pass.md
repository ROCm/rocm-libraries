# StinkyWaitCntInsertionPass — SSA Def-Use + Dataflow Wait Count Insertion

`StinkyWaitCntInsertionPass` inserts `s_wait_dscnt`, `s_wait_loadcnt`, `s_wait_kmcnt`, and `s_wait_tensorcnt` so that asynchronous memory operations complete before their results are consumed. The pass reads dependencies from an SSA def-use chain over memory-token pseudo-registers and computes wait placement via a forward dataflow solver (`WaitDataflow`), followed by optional plan optimizers and a finalize replay step.

### Key characteristics

- **SSA def-use dependencies** via `buildUseDefChain(includePseudo=true)` — memtoken pseudo-registers become first-class edges, so `inst->getSources()` lists the memops a consumer depends on (including through PHIs at CFG joins)
- **Four counter types**: DS (`dlcnt`), buffer/load (`vlcnt`), scalar memory (`kmcnt`), tensor (`tlcnt`), tracked as `CounterKind` in `WaitDataflow`
- **Per-predecessor queues** — each counter keeps separate in-flight FIFOs tagged by CFG predecessor edge, so join consumers see each path's depth instead of a collapsed union queue
- **Tensor loop policy** — by default, the `CK_Tensor` slice of dataflow state is frozen after the first solver sweep so tensor tokens do not propagate around loop back-edges; a conservative option restores normal tensor fixed-point iteration
- **Anti-dependency scans** for hazards the SSA RAW chain does not capture (WAR-on-LDS, barrier ordering, untagged conservative fallbacks)
- **Analyze → Optimize → Finalize** — dataflow solve, then `ShallowPredPromotion`, then `finalizePlan()` to align the plan with post-optimizer FIFO simulation
- **Selective IR mutation**: `buildUseDefChain` and `WaitDataflow::solve()` run over every basic block so skipped preds still contribute in-flight state; `PassContext::shouldProcessBasicBlock` gates `emitWaits` and `removePHIs` only

---

## Hardware background

Asynchronous memory ops (LDS `ds_*`, global/buffer loads and stores, `tensor_load_to_lds`) retire out of order. The hardware exposes one FIFO counter per kind of async memop. Before a consumer reads a result, the compiler must insert the matching `s_wait_*` to drain the producer.

| `CounterKind` | Covers | Wait instruction | Modifier field |
|---------------|--------|------------------|----------------|
| `CK_DS` | `ds_read` / `ds_write` / `ds_atomic` | `s_wait_dscnt N` | `SWaitCntData.dlcnt` |
| `CK_Buffer` | global/buffer load + store | `s_wait_loadcnt N` | `SWaitCntData.vlcnt` |
| `CK_KM` | scalar memory loads (`s_load_*`) | `s_wait_kmcnt N` | `SWaitCntData.kmcnt` |
| `CK_Tensor` | `tensor_load_to_lds` | `s_wait_tensorcnt N` | `SWaitTensorCntData.tlcnt` |

`s_wait_*cnt N` blocks until **at most `N`** ops of that kind remain outstanding (it keeps the `N` most-recently-issued in flight and drains everything older).

### Wait arithmetic

If producer `D` sits at index `i` (0 = oldest) in a counter FIFO of size `n`, the wait immediate that drains `D` is:

```
w(D) = n - i - 1
```

`PerPredQueue::countFrom(D)` returns the number of ops from `D` (inclusive) to the tail. The emitted wait is `countFrom(D) - 1`. When multiple deps constrain the same counter, the pass takes the **min** of their wait values (the most permissive value that still drains the oldest needed dep on every constrained path).

`classifyMemOp` maps an instruction to its counter; an instruction that is not a tracked memop returns `CK_Count` (no counter).

---

## Prerequisites: memtokens as SSA edges

The hard part of wait insertion is that dependencies do not flow through real registers — a `tensor_load_to_lds` writes an LDS region and a later `ds_read` of that region depends on it, but there is no vreg linking them.

`StinkyBuildImplicitDependencyPass` must run upstream. It attaches `MemTokenData` modifiers (integer token IDs) to tensor loads, DS ops, and barriers, materialising LDS ordering edges as pseudo-register defs/uses.

The wait pass then calls:

```cpp
buildUseDefChain(func, domInfo, /*clearExisting=*/true, /*includePseudo=*/true);
```

With `includePseudo=true`, memtoken pseudo-regs are treated like ordinary registers: PHIs are inserted at dominance-frontier join points and linked through `getSources()` / `getUsers()`. After this:

- A consumer's `inst->getSources()` directly contains the producer memops it depends on — a **RAW** edge — even across blocks and through joins.
- At a CFG merge, a memtoken value becomes a **PHI** whose incoming sources are the per-predecessor producers.

Token overlap (`MemTokenData`) is still consulted for **anti-dependencies** the SSA RAW chain does not capture (WAR-on-LDS, barrier ordering, untagged fallbacks).

---

## Pass flow

The orchestration in `StinkyWaitCntInsertionPass::run()`:

```cpp
buildUseDefChain(func, domInfo, /*clearExisting=*/true, /*includePseudo=*/true);

WaitDataflow df(func, domInfo, rpo);
df.setLoopCarriedTokenDepsEnabled(options.enableLoopCarriedTokenDeps);

const auto numWaves = passCtx.getGemmTileConfig().NumWaves;
df.setRawNeedsWait(CK_Tensor, [numWaves](const StinkyInstruction& i) {
    return isBarrier(i) || numWaves == 1;
});

df.solve();
WaitInsertionPlan plan = df.materializePlan();

ShallowPredPromotion shallowPred;
std::vector<WaitPlanOptimizer*> optimizers = {&shallowPred};
for (auto* opt : optimizers) opt->rewrite(plan, df.getResult(), func);

df.finalizePlan(plan);

emitWaits(func, passCtx, arch, plan);
removePHIs(passCtx, rpo);
```

```
+---------------------------------------------------------------+
|              StinkyWaitCntInsertionPass::run()                 |
+---------------------------------------------------------------+

                  Function Entry
                        |
                        v
            +-----------------------+
            |  buildUseDefChain()   |   Memtoken pseudo-regs as
            |  (includePseudo=true) |   SSA edges + PHI placement
            +-----------+-----------+
                        |
                        v
            +-----------------------+
            |  setRawNeedsWait()    |   Per-counter drain policy
            |  + tensor-loop option |   (optional overrides)
            +-----------+-----------+
                        |
                        v
            +-----------------------+
            |  WaitDataflow.solve() |   Forward fixed-point (full RPO)
            +-----------+-----------+
                        |
                        v
            +-----------------------+
            |  materializePlan()    |   Conservative WaitInsertionPlan
            +-----------+-----------+
                        |
                        v
            +-----------------------+
            |  WaitPlanOptimizer(s) |   e.g. ShallowPredPromotion
            +-----------+-----------+
                        |
                        v
            +-----------------------+
            |  finalizePlan()       |   Replay with final plan + tail
            |                       |   drains (+ tensor freeze)
            +-----------+-----------+
                        |
                        v
            +-----------------------+
            |  emitWaits()          |   Anchor waits + tail drains
            +-----------+-----------+
                        |
                        v
            +-----------------------+
            |  removePHIs()         |   Strip PHI pseudo-instructions
            +-----------+-----------+
                        |
                        v
                  Function Exit
```

### Why three stages (Analyze → Optimize → Finalize)

`WaitDataflow::transferBlock` does two jobs in one walk:

1. **Plan** — decide `(anchor → WaitCountSpec)` using required waits from the current dataflow state.
2. **Simulate** — `trimQueues()` so later instructions (and successor blocks via the fixed point) see a FIFO that matches the **planned** drain.

`WaitPlanOptimizer`s run **after** convergence and may change what will actually be emitted. For example, at a join the baseline solver may emit `tlcnt = 0` (min across paths), but `ShallowPredPromotion` may relax that to `tlcnt = 2` and add a tail drain on the shallow predecessor. The solver's queue simulation used `0` (full trim); the final plan uses `2` (partial trim).

When an earlier anchor's simulated drain is **stronger** than the final emitted drain, every later anchor in the same block — on **any** counter — can be wrong:

- **Missing wait** — FIFO cleared in simulation but still in flight after promotion.
- **Redundant wait** — FIFO kept alive in simulation but fully drained in the final plan.

`finalizePlan()` closes this gap by replaying affected blocks with the post-optimizer plan values. Optimizers are **not** run inside `transferBlock` during the fixed point: they are inter-block (`ShallowPredPromotion` reads predecessor exit state and writes tail drains), and embedding them in the solve would couple promotion to stale or oscillating analysis snapshots.

### Tensor counter policy

Built-in default (`CounterPolicy` in `WaitDataflow.cpp`): tensor RAW deps drain only at barriers. The pass overrides this from tile config:

```cpp
df.setRawNeedsWait(CK_Tensor, [numWaves](const StinkyInstruction& i) {
    return isBarrier(i) || numWaves == 1;
});
```

When `NumWaves == 1`, `rawNeedsWait` is true at **every** instruction, so tensor RAW deps drain at each consumer (not only at barriers). In multi-wave kernels, tensor counter drains are limited to barriers — cross-wave LDS visibility is handled by the barrier itself.

Loop-carried tensor-token dependencies are controlled separately:

```cpp
df.setLoopCarriedTokenDepsEnabled(options.enableLoopCarriedTokenDeps);
```

By default this option is **disabled**. In that mode, `WaitDataflow` computes `CK_Tensor` normally during solver sweep 0, then freezes only the tensor slice of `DataflowState` on later sweeps. This prevents tensor token state from propagating around loop back-edges and avoids loop-header waits such as an unnecessary first `s_wait_tensorcnt 0`.

When `enableLoopCarriedTokenDeps` is **enabled**, `CK_Tensor` participates in the normal fixed-point iteration just like the other counters. This is the conservative mode and can reintroduce loop-header tensor waits when a back-edge carries tensor token state.

Entry points for the conservative mode:

- `WaitDataflow::setLoopCarriedTokenDepsEnabled(true)`
- `WaitCntInsertionOptions::enableLoopCarriedTokenDeps = true`
- `ModuleOptions::EnableLoopCarriedTokenDeps = true`
- `stinkytofu-opt`: `--StinkyWaitCntInsertionPass=enableLoopCarriedTokenDeps`

DS, buffer, and KM counters use the default `rawNeedsWait` (drain at every consumer). Callers may override any counter via `WaitDataflow::setRawNeedsWait()` before `solve()`.

### Selective processing

The dataflow must see **every** block in RPO so a skipped predecessor still contributes its in-flight state to successors. `PassContext::shouldProcessBasicBlock` gates only `emitWaits` and `removePHIs` — disabled blocks participate in analysis but receive no IR mutations.

---

## Data structures

### PerPredQueue

One in-flight FIFO for a single counter, tagged with the CFG predecessor edge it arrived on.

```
+--------------------------------+
|         PerPredQueue           |
+--------------------------------+
| pred: BasicBlock*              |  CFG predecessor (nullptr = synthetic)
| ops:  deque<Instruction*>      |  Memops in issue order
+--------------------------------+
| countFrom(op) -> int           |  Ops from op to tail (0 if absent)
+--------------------------------+
```

Per-pred tagging lets `ShallowPredPromotion` attribute waits to specific predecessors (for tail drains) and lets join consumers see each path's FIFO depth independently. `pred == nullptr` is a synthetic queue used when a block has no seeded preds yet (entry block, or local ops before any predecessor queue exists).

### PhiSummary

Summary of what a memtoken PHI implies for a consumer, per counter. `WaitCountSpec::kUnused` (-1) means no constraint on that counter.

### DataflowState

```
+----------------------------------------------+
|              DataflowState                   |
+----------------------------------------------+
| queues[CK_Count]: vector<PerPredQueue>       |  Per-pred FIFOs per counter
| phiSummaries: map<PHI*, PhiSummary>          |  PHI wait bounds
+----------------------------------------------+
```

Per-pred queues are kept separate at block exit (not collapsed) so successors preserve per-path FIFO positions. Queues are capped at `kMaxInFlight` (64); exceeding the cap drops the oldest provably-complete op and emits a non-fatal diagnostic at convergence.

### DataflowResult

Converged per-block `entryState` and `exitState`. Optimizers read `exitState[pred]` for per-path queue depths; `finalizePlan` reads `entryState`. In default mode, the `CK_Tensor` slice of these states is the frozen post-sweep-0 tensor snapshot, while the other counters are fully converged.

### WaitCountSpec / WaitInsertionPlan

```
+--------------------------------+
|         WaitCountSpec          |
+--------------------------------+
| dsCount:     int  (dlcnt)      |  or kUnused (-1)
| bufferCount: int  (vlcnt)      |
| tensorCount: int  (tlcnt)      |
+--------------------------------+

+----------------------------------------------+
|           WaitInsertionPlan                  |
+----------------------------------------------+
| anchorWaits: map<Inst*, WaitCountSpec>       |
| tailDrains:  vector<TailDrain>               |  Waits before pred terminators
+----------------------------------------------+
```

- `anchorWaits[I]` — waits to emit immediately before instruction `I`.
- `tailDrains` — extra waits an optimizer requested before a predecessor's terminator.

### CounterEmitState

Per-counter bookkeeping during a block walk. Tracks `lastEmittedWait` and `opsSinceLastWait` for redundancy elision:

```
needsNewWait(required) =
    lastEmittedWait == kUnused
    OR lastEmittedWait + opsSinceLastWait > required
```

Intuition: after emitting `s_wait N`, at most `N` old ops remain; each new op issued since bumps the effective residual by 1. If `lastEmittedWait + opsSinceLastWait <= required`, the hardware is already at least as drained as required demands.

---

## WaitDataflow solver

`WaitDataflow::solve()` runs a forward fixed-point over the **full** RPO block list:

1. Seed every block with empty state (lattice bottom).
2. Each iteration: for each block, `mergeFromPredecessors` builds entry state, then `transferBlock` walks instructions and mutates state.
3. In default mode, after sweep 0, restore the `CK_Tensor` slice of each block's entry/exit state from the frozen sweep-0 snapshot. DS, buffer, and KM continue normal fixed-point iteration.
4. Convergence = full RPO sweep with no exit-state change.
5. On convergence, report any counter that exceeded `kMaxInFlight` as a non-fatal diagnostic.

**Iteration cap**: `min(256, max(kMaxInFlight + 8, 2 * n))` where `n = |rpo|`. On cap hit, `materializePlan()` forces wait 0 at every emitting anchor and logs a warning.

If `enableLoopCarriedTokenDeps` is true, the tensor freeze step is skipped and `CK_Tensor` iterates to the same fixed point as the other counters.

### mergeFromPredecessors

At block entry:

1. **Seed per-pred queues** — copy each predecessor's exit queues, retagging `pred`. Self-predecessors (back-edges) are included; at fixed point the back-edge's exit is the loop body's true exit. Identical `(pred, ops)` queues are deduplicated so loop bodies converge (without dedup, back-edges would re-copy the same queue each iteration and the state would never stabilize).
2. **Forward PHI summaries** from all preds; collisions keep the strictest (min) wait per counter.
3. **Build PHI summaries** — for each PHI at the top of the block, compute per-counter wait as the **min** of `countFrom(src) - 1` over constrained incoming paths. Nested PHIs chain through the predecessor's already-computed summary.

Queues are **copied, not merged** — a join holds one queue per (predecessor, original-queue) so per-path depth survives. The old "collapse to union queue at exit" design was deliberately removed; collapsing would lose per-pred position info and force over-deep waits.

### transferBlock

For each non-PHI instruction in program order:

1. **`computeRequiredWaits`** — determine `required[CK_Count]` (see below).
2. **Emit decision** — for each counter with a required wait, apply redundancy elision; record `(anchor, WaitCountSpec)` in `emitPlan`.
3. **`trimQueues`** — model the hardware drain on all per-pred queues for that counter.
4. **Record producer** — append the instruction to its counter queue *after* the wait decision (so the wait's snapshot excludes its own consumer).

Per-pred queues are **not** collapsed at block exit.

---

## Dependency resolution

### RAW from SSA sources

RAW dependencies come from `inst->getSources()` after `buildUseDefChain(includePseudo=true)`:

- Concrete memop sources: `classifyMemOp(*src)` maps to a counter; `countFrom(src) - 1` from each per-pred queue contributes via `tightenRequired` (min across hits).
- PHI sources: `phiCurrentQueueWait` recurses through PHI inputs and scans **live** per-pred queues (not stored PHI summaries alone), so intervening in-block ops are counted.

```
v_wmma uses v[0:3]
  -> getSources(): PHI
       -> incoming[0]: ds_read (preloop)
       -> incoming[1]: ds_read (loop body)
  -> both ds_read ops contribute DS-counter waits
```

There is **no same-pipeline filter for RAW**: a `ds_store` consuming a `ds_load`'s vreg still needs the DS wait. Same-pipeline skip applies only to anti-deps.

Each RAW contribution is gated by `rawNeedsWait[c](*inst)` — the per-counter predicate that decides whether this consumer is an anchor where a drain may be emitted.

### computeRequiredWaits

Per instruction, `required[c]` starts at `kUnused` and is tightened to the min wait across all contributing deps on counter `c`:

1. **RAW from SSA** — walk `getSources()` as above; gated by `rawNeedsWait[c](*inst)`.
2. **Anti-deps (DS)** — `scanDsAntiDeps` for LDS writers (`tensor_load_to_lds`, `ds_write`) and barriers with `MemTokenData` token overlap against per-pred DS queues; same-pipeline pairs (`ds_write` vs `ds_read`) skipped.
3. **Tensor untagged scan** — tensor anchors with tagged tokens still scan for in-flight tensor loads lacking `MemTokenData`.
4. **Conservative fallbacks** — force wait 0 when disjointness cannot be proved (see table below).

### Anti-dependencies (WAR-on-LDS and barrier ordering)

The SSA RAW chain captures "consumer reads producer's result" but not anti-dependencies:

- An **LDS writer** (`tensor_load_to_lds` or `ds_write`) must wait for prior LDS readers/atomics on a matching token (WAR): the write must not clobber LDS a reader has not consumed yet.
- A **barrier** must wait for any prior DS op on a matching token.
- Each in-flight DS op whose token set overlaps the anchor's becomes an extra DS dep.
- **Same-pipeline pairs are skipped** (`isOnSamePipeline`): two DS ops are ordered by the DS FIFO in hardware.
- A DS op with no `MemTokenData` is treated as overlapping — disjointness cannot be proved.

Anchor helpers:

- `isTensorAnchor` — barrier, `ds_read`, `ds_write`, `ds_atomic`
- `isLdsWriterAnchor` — `tensor_load_to_lds`, `ds_write`

### Conservative fallbacks

| Site | Condition | Effect |
|------|-----------|--------|
| Tensor anchor | Anchor lacks `MemTokenData` and any tensor load in flight | `required[CK_Tensor] = 0` |
| LDS writer | Writer lacks `MemTokenData`, any DS op in flight, writer is not `ds_write` | `required[CK_DS] = 0` |
| Barrier | Any in-flight DS op and (barrier untagged OR any pending DS op untagged) | `required[CK_DS] = 0` |

These widen waits, never narrow them.

---

## Plan materialization

`materializePlan()` flattens `emitPlan` into `WaitInsertionPlan.anchorWaits`. On normal convergence, values are copied as-is. On iteration-cap hit, every counter field at each anchor is forced to 0 — a fully-drained, always-safe plan.

---

## ShallowPredPromotion

After `materializePlan()`, `WaitPlanOptimizer::rewrite` may relax the plan. `ShallowPredPromotion` is the only shipped optimizer.

**Problem**: at a CFG join, the baseline solver emits a single anchor wait sized for the shallowest constrained path (`min` across paths), over-draining deeper paths.

**Solution**: for each anchor wait at a join (≥2 predecessors), per constrained counter:

1. **Collect deps** — flatten PHIs to concrete memops via `collectMemOpDeps` (RAW SSA only).
2. **Per-pred required wait** — from `DataflowResult.exitState[pred]`, compute the strictest `w` that path alone would need (`perPredRequiredWait`).
3. **Classify predecessors** — *correctable* if not a self-pred and the pred has only this block as successor (otherwise a tail drain would also fire on the pred's other successors); otherwise *uncorrectable*.
4. **Choose relaxed anchor wait**:
   - Only correctable preds → use the deepest (`maxCorrectable`); shallow preds get tail drains.
   - Any uncorrectable pred → cannot exceed its requirement (`minUncorrectable`).
   - Mixed → `min(maxCorrectable, minUncorrectable)`.
   - Skip if the new wait is not looser than the current one.
5. **Record tail drains** — for every correctable pred whose requirement is below the relaxed anchor wait, keep the strictest (smallest) immediate per (pred, counter).
6. **Materialise `tailDrains`** deterministically (preds sorted by pointer).

**Net effect** on a diamond with shallow and deep paths: the shallow predecessor gets a tail drain (e.g. `s_wait_tensorcnt 0` before its branch), and the merge anchor relaxes from the shallow path's depth to the deep path's depth, preserving in-flight overlap on the deep path.

---

## finalizePlan

After optimizers, `WaitDataflow::finalizePlan()` replays blocks where the solver's queue simulation may disagree with the final plan.

### When a block is replayed

`blockNeedsFinalize` returns true when the block has any entry in `plan.anchorWaits` **or** ≥2 wait-anchor candidates (any instruction where `rawNeedsWait` is true for any counter).

### Per-block replay algorithm

For each replayed block in RPO order:

1. **`adjustedEntry`** — start from converged `entryState[bb]`. For each predecessor with a `TailDrain` in the plan, virtually trim only that pred's per-pred queues by the tail-drain wait values. Tail drains execute on the predecessor **before** control enters the successor; converged `entryState` does not include them.
2. **Tensor freeze restore** — in default mode and after replay sweep 0, restore only the `CK_Tensor` slice of the block entry from the frozen replay entry snapshot.
3. **Program-order walk** — for each non-PHI instruction:
   - `computed = computeRequiredWaits(inst, state, rawNeedsWait)`
   - `applySpec = mergePlanAndComputed(plan, inst, computed, emitState)`
   - Trim queues with applied wait values (never with pre-optimizer solver values)
   - Update `plan.anchorWaits[inst]` — add, update, or erase redundant fields
   - Append producers to counter queues after the wait decision
4. **Tensor exit restore** — in default mode and after replay sweep 0, restore only the `CK_Tensor` slice of the block exit from the frozen replay exit snapshot before convergence comparison.

When `enableLoopCarriedTokenDeps` is true, both tensor restore steps are skipped and all counters are replayed to a fixed point.

### mergePlanAndComputed

| Planned wait | Computed need | `needsNewWait` | Action |
|--------------|---------------|----------------|--------|
| set | — | yes | Use **planned** value for trim and emit |
| set | — | no | Drop planned field (redundant) |
| unset | set | yes | **Add** computed wait to plan |
| unset | set | no | No wait |

Planned values win when redundancy allows — optimizers may intentionally relax below the replay recompute.

### Effects

- **Adds** missing later-anchor waits when promotion relaxed an earlier drain.
- **Removes** redundant waits when an earlier full drain already covered them.

---

## Multi-anchor diamond example

A diamond CFG with two barriers on disjoint tokens exercises per-path promotion and multi-anchor finalize together:

```
                         ^entry
                    tensor_load [11]
                        /     \
                     v           v
               ^p_shallow     ^p_deep
              (no extra ops)  tensor_load [12] × 2
                     \         /
                      v       v
                        ^merge
                   s_barrier [11]
                   s_barrier [12]
```

### Per-path FIFO at `^merge` entry

| Path | In-flight tensor ops | Wait for token `[11]` barrier |
|------|----------------------|-------------------------------|
| `^p_shallow` | `[t11]` | `tlcnt = 0` |
| `^p_deep` | `[t11, t12, t12]` | `tlcnt = 2` (drain `t11`, keep the two `t12`) |

The baseline solver takes `min(0, 2) = 0` — sound but over-drains the deep path.

`ShallowPredPromotion`:

- Inserts `s_wait_tensorcnt 0` at `^p_shallow`'s tail (pre-drain the shallow path).
- Relaxes the merge anchor before barrier `[11]` from `0` to `2`.

After barrier `[11]` with `tlcnt = 2`, the deep path still has `[t12, t12]` in flight. Barrier `[12]` must emit `s_wait_tensorcnt 0`. Without `finalizePlan`, the solver already simulated a full drain at barrier `[11]` (`tlcnt = 0`) and would not add the second wait.

### Expected output

**With `ShallowPredPromotion` enabled (production default):**

```asm
^p_shallow:
  s_wait_tensorcnt 0            ; tail drain

^merge:
  s_wait_tensorcnt 2            ; relaxed merge wait for token [11]
  s_barrier tokens = [11]
  s_wait_tensorcnt 0            ; added by finalizePlan — drain token [12]
  s_barrier tokens = [12]
```

**With optimizers disabled:**

The baseline solver emits a single `s_wait_tensorcnt 0` before barrier `[11]`, fully draining every path. `finalizePlan` replays the block, sees barrier `[12]` has no in-flight deps, and does not add a second wait:

```asm
^merge:
  s_wait_tensorcnt 0
  s_barrier tokens = [11]
  s_barrier tokens = [12]       ; no second wait
```

Approaches that were considered and rejected for this case:

| Approach | Problem |
|----------|---------|
| Selective trim on `wait 0` when a later anchor exists | Correct with optimizer on, wrong when optimizer is off (redundant second wait) |
| Run optimizers inside the fixed-point loop | Couples promotion to incomplete snapshots; hard to terminate |
| Always use `max` instead of `min` at joins in the solver | Breaks the baseline plan that promotion is designed to relax |

---

## Emit phase

### Anchor waits

Walk blocks and instructions in **program order** for deterministic placement when multiple anchors share a block. Only blocks passing `shouldProcessBasicBlock` are modified.

### Tail drains

For each `TailDrain`, insert waits before the predecessor's **branch terminator** (if the terminator is a branch; otherwise anchor is `nullptr`).

Each non-`kUnused` field in `WaitCountSpec` produces one instruction:

| WaitCountSpec field | Emitted instruction | Modifier |
|---------------------|---------------------|----------|
| `dsCount` | `s_wait_dscnt <N>` | `SWaitCntData.dlcnt` |
| `bufferCount` | `s_wait_loadcnt <N>` | `SWaitCntData.vlcnt` |
| `tensorCount` | `s_wait_tensorcnt <N>` | `SWaitTensorCntData.tlcnt` |

### removePHIs

Memtoken PHIs are pseudo-instructions used only to thread SSA edges. After emission they are erased from every processed block in RPO order.

---

## Worked examples

### DS reads consumed by WMMA

**Input:**

```assembly
label_LoopBeginL:
    ds_read_b128 v[0:3],  v8
    ds_read_b128 v[4:7],  v9
    ds_read_b128 v[8:11], v10
    ds_read_b128 v[12:15], v11
    v_wmma_f32 v[16:23], v[0:3], v[4:7], v[16:23]
    v_wmma_f32 v[24:31], v[8:11], v[12:15], v[24:31]
```

**Analysis:** After `buildUseDefChain`, each `v_wmma` lists its `ds_read` sources via `getSources()`. The dataflow records all four reads in the per-pred DS queue:

```
At first v_wmma (uses op0, op1):
  op0: countFrom = 4, requiredWait = 3
  op1: countFrom = 3, requiredWait = 2
  -> Emit s_wait_dscnt 2

At second v_wmma (uses op2, op3):
  op2: countFrom = 2, requiredWait = 1
  op3: countFrom = 1, requiredWait = 0
  needsNewWait(0): lastEmitted=2, opsSinceLastWait=0, 2+0 > 0 -> true
  -> Emit s_wait_dscnt 0
```

**Output:**

```assembly
label_LoopBeginL:
    ds_read_b128 v[0:3],  v8
    ds_read_b128 v[4:7],  v9
    ds_read_b128 v[8:11], v10
    ds_read_b128 v[12:15], v11
    s_wait_dscnt 2
    v_wmma_f32 v[16:23], v[0:3], v[4:7], v[16:23]
    s_wait_dscnt 0
    v_wmma_f32 v[24:31], v[8:11], v[12:15], v[24:31]
```

### WAR-on-LDS across a CFG edge

```text
^entry:
  v[0:1] = ds_load_b64  tokens=[0]      ; DS reader R0
  v[2:3] = ds_load_b64  tokens=[0]      ; DS reader R1
  -> ^body
^body:
  tensor_load_to_lds    tokens=[0]      ; LDS writer W
```

`R0`/`R1` → `W` is a **WAR** (writer after readers), which the SSA RAW chain does not carry. Solving `^entry` leaves `state.queues[CK_DS] = [{entry, [R0, R1]}]` at exit. At `W`, `scanDsAntiDeps` finds `R0` → wait 1, `R1` → wait 0; min = **0**. Emit `s_wait_dscnt 0` before the `tensor_load_to_lds`.

### Per-path tail compensation

```text
^entry: T0 (token 0)  -->  ^A (deep): X1, X2 (token 1)
                        \->  ^B (shallow): (empty)
                              both --> ^merge: ds_load_b64 (token 0)
```

- `exitState[A]` tensor depth 3 for `T0` → `w = 2`; `exitState[B]` depth 1 → `w = 0`.
- Conservative plan: `s_wait_tensorcnt 0` at merge (over-drains `A`).
- After promotion: `s_wait_tensorcnt 0` at `^B`'s tail; `s_wait_tensorcnt 2` at merge before `ds_load_b64`.
