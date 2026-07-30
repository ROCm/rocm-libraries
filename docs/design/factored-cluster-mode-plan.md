<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Design PLAN: Factored WG-Cluster Mode for gfx1250 StreamK (Cs × Ck)

> **STATUS: PLAN ONLY — NOT IMPLEMENTED, PENDING HUMAN APPROVAL.**
> This document is a local-only design plan (like `docs/design/pr-split-plan.md`).
> Do not `git add`/commit it. No source is modified by this document.

**Target arch:** gfx1250 (ISA `(12,5,0)`, wave32, `TDMInst=3`, `HasTDM`,
`HasClusterBarrier`, `HasNewBarrier`).

**Goal.** Let one StreamK kernel enable **both** DP cooperative B-multicast
(`StreamKMulticast`) **and** K-split cluster partial-reduction
(`StreamKClusterReduction`) — today they are **mutually exclusive** (rejected in
`_validateStreamKMulticast`, `Solution.py:352-360` @ `0eed89b48a`). We factor the
1-D HW cluster size `C = Cs · Ck`:

- **Cs** = spatial multicast axis. `Cs` peers process M-adjacent *distinct*
  output tiles that share the same B N-block over the same K-slice → B is
  TDM-multicast across the `Cs` peers.
- **Ck** = K-split reduction axis. `Ck` peers split one tile's K range and reduce
  partials through the cluster split-barrier.

Each cluster of `C = Cs·Ck` workgroups is a logical `(Cs × Ck)` grid; a
workgroup's within-cluster rank decodes to `(s, k)`, `s∈[0,Cs)`, `k∈[0,Ck)`.
Multicast runs along `s` (peers sharing `k`); reduction runs along `k` (peers
sharing `s`).

---

## 0. Where the two features live today (verified anchors)

Current working tree = branch `users/jolabega/streamk-cluster-multicast`
(`HEAD = d5d96923cf`). It contains the **multicast** feature. The **reduction**
feature is NOT on this branch; it is read from `0eed89b48a` (an ancestor commit
on branch `users/jaopaulolc/streamk-wg-clusters`; the reduction
design doc `docs/design/streamk-wg-clusters.md` is on tip `5ded69c72d`).

### Multicast (current tree)
- Mask helpers: `Tensile/Components/StreamK.py:2487-2604`
  (`streamKMulticastMaskPredicate`, `streamKMulticastBoundaryClear`,
  `streamKMulticastPrologueSignal`), call sites in `preLoop`
  (`StreamK.py:2631,2637`) and the DP→SK boundary clear in `graWorkGroup`
  (`StreamK.py:2950`).
- Mask value/attach: `Tensile/Components/ClusterLoad.py` — `computeMasks`
  (`:129-187`, the genuine 2-D maskA/maskB math), `applyToDescriptor`
  (`:191-206`), `usesCombinedMask`/`maskSgprName` forcing split A/B when
  `StreamKMulticast` (`:42-86`).
- Kernel cluster decode: `KernelWriterAssembly.py:2582-2644`
  (`WorkGroup0 = cluster_x·nwg_x + wg_x`, the `RemapWorkGroupDone` block) and the
  `computeMasks` call at `:2649-2654`.
- Solution: collapse `ClusterDim!=[1,1] && StreamK==3 ⇒ StreamKMulticast`
  (`Solution.py:1162-1163`), tri-state `Multicast` derivation (`:1167-1194`),
  `ClusterBarrier` enable (`:1201-1205`), `_validateStreamKMulticast`
  (`:233-325`).
- Host: `sizeMapping.streamKMulticast` (`ContractionSolution.hpp:146`, serialized
  `Serialization/ContractionSolution.hpp:109`, wired `Contractions.py:630,725`);
  `getSKGridImpl` multicast grid `= ceil(tiles/C)·C` (`ContractionSolution.cpp:4068-4072`);
  `solve()` re-round guard (`:3198-3202`); `ClusterDimCheck` (`Contractions.py:589-595`
  → `ContractionProblemPredicates.hpp:3029-3091`, checks `nWG_x % C0 == 0 &&
  nWG_y % C1 == 0`).

### Reduction (`git show 0eed89b48a:<path>`)
- Barrier helpers: `StreamK.py:1520-1611` — `_streamKClusterReductionEnabled`,
  `clusterReduceSignal` (`s_barrier_signal -3`, wave-0 elect),
  `clusterReduceWait` (`s_barrier_wait -3`), `clusterReduceIntraCheck`
  (`cluster_last = StreamKIdx | (C-1); cluster_last < skGrid`).
- Epilogue call sites: owner wait in `storeBranchesCommon` (`:958-1010`), non-owner
  signal in `writePartials` (`:1388-1406`).
- Solution: `_validateStreamKClusterReduction` (`:233-315`), the mutual-exclusion
  reject inside `_validateStreamKMulticast` (`:352-360`), the collapse's `and not
  state.get("StreamKClusterReduction", 0)` term (`:1256`),
  `_validateStreamKClusterReduction` invoked at `:1975`, seeded off at `:2039`.
- Predicate: `ClusterReductionIterCheck` = `ceil(K/DepthU) % C == 0`
  (`ContractionProblemPredicates.hpp:3100-3155`; emitted `Contractions.py:605-606`,
  `value=[DepthU, C]`).
- Host: `getSKGridImpl` reduction grid `= C·tiles`
  (`ContractionSolution.cpp:4097-4100`); two-tile cluster-reduction kernarg block
  (`:943-966`, `skSplit=C`, `SKItersPerWG=itersPerTile/C`, `skTiles=tiles`),
  guarded by `sk.grid == C·tiles`; workspace-fallback re-round (`:3218-3223`).

### The `-3` cluster split-barrier semantics (critical)
`rocisa` `SBarrier(separate, wait, clusterBarrier)` lowers to id `-3` when
`clusterBarrier=True` and `HasClusterBarrier`
(`rocisa/.../instruction/common.hpp:1618-1640`): `separate&&!wait ⇒
s_barrier_signal -3` (arrive), `separate&&wait ⇒ s_barrier_wait -3`. Id `-3`
scopes the **entire HW cluster of C WGs** (not a sub-group). One wave per WG
signals (wave-0 election); any/all waves may wait. A `wait` phase releases when
the arrival count reaches the full cluster membership `C`; **multiple waiters on
the same phase all release** (waits do not consume arrivals). This property is
the linchpin of §3.

---

## 1. Index factoring: `C = Cs · Ck`

### 1.1 The hard 1-D-grid constraint (why not a genuine 2-D HW `[Cs,Ck]`)

The StreamK launch grid is **1-D** `{skGrid,1,1}`
(`ContractionSolution.cpp:1758-1763` @ `0eed`; current `:1765` guards the
`enableCluster` branch). The HW clustered launch requires `gridDimY %
clusterDim.y == 0`; with `gridDimY == 1` this forces `clusterDim.y == 1`. Both
shipped features therefore hard-require `ClusterDim == [C, 1]`
(`_validateStreamKMulticast` `:291-293`; `_validateStreamKClusterReduction`
`:284-286`). **A genuine 2-D HW cluster `[Cs,Ck]` mapped to clusterDim.x/.y is
impossible** for StreamK. The factoring must therefore be a **logical**
decomposition of the *single* 1-D HW cluster rank `wg_x ∈ [0,C)`.

### 1.2 Recommended specification: keep HW `ClusterDim=[C,1]`, add a `Ck` factor

**Recommended (Option B).** Keep `ClusterDim = [C, 1]` as the HW-accurate 1-D
cluster (so launch, remap `KernelWriterAssembly.py:2582-2644`, mask-SGPR alloc,
`rv.clusterDim` `ContractionSolution.cpp:1776`, and `ClusterDimCheck` all keep
their current 1:1 HW meaning), and add ONE new solution parameter:

```python
# ValidParameters.py, near ClusterDim (:1117)
# Ck = K-split reduction factor within the [C,1] cluster. 1 => pure multicast
# (current). C => pure reduction. 1<Ck<C => factored. Cs = C // Ck.
"StreamKClusterKSplit": [1, 2, 4, 8, 16],
```

Then `Cs = ClusterDim[0] // StreamKClusterKSplit`, `Ck = StreamKClusterKSplit`.
Both derived-only booleans fall out:
- `StreamKMulticast` on ⟺ `Cs > 1`.
- `StreamKClusterReduction` on ⟺ `Ck > 1`.

Degenerate equivalences (the incremental path, §8):
- `Ck == 1` ⇒ `Cs == C` ⇒ **exactly today's multicast**.
- `Ck == C` ⇒ `Cs == 1` ⇒ **exactly today's reduction**.
- `1 < Ck < C` ⇒ factored (new).

**Why not overload `ClusterDim=[Cs,Ck]` (Option A).** Appealing because
`ClusterLoad.computeMasks` already emits the correct 2-D maskA/maskB for a
`[C0,C1]` cluster (`ClusterLoad.py:144-186`). But `ClusterDim` is consumed as the
**HW** cluster shape in ~6 places (launch `clusterDim.x/.y`, remap wg_x/wg_y
decode, `ClusterDimCheck` value[3]/[4], `cooperativeThreadPartition`
`ClusterLoad.py:88-95`, `rv.clusterDim`). Reusing `[Cs,Ck]` as a *logical* dim
would silently feed `clusterDim.y=Ck` into a `gridDimY=1` launch (invalid) and
make `computeMasks` shift by the raw HW `wg_y` (always 0 for a 1-D grid) →
wrong mask. Option A would require a parallel "logical vs HW ClusterDim"
split everywhere. Option B confines the new concept to one scalar + a factored
mask/decode helper. **Recommend Option B.** (If a future true 2-D grid appears,
Option A can subsume it; out of scope here.)

### 1.3 Kernel decode: within-cluster rank → `(s, k)` ("k fastest")

The base StreamK schedule already produces the right layout for free once the
host sets `skGrid = Ck·tiles` and `skSplit = Ck` (§1.4, §5):
`StreamKIter = StreamKIdx · SKItersPerWG` with `SKItersPerWG = itersPerTile/Ck`
(`preLoop`/`graWorkGroup` `StreamK.py:2645,2921`), so

```
tile     = StreamKIter // itersPerTile = StreamKIdx // Ck
kslice k = StreamKIdx % Ck                       # K-reduction axis rank
cluster  = StreamKIdx // C                        # C = Cs·Ck
wg_x     = StreamKIdx % C   = s·Ck + k
spatial s = wg_x // Ck      = (StreamKIdx % C) // Ck   # which of Cs tiles
```

This is **"k fastest"**: consecutive `StreamKIdx` sweep k first, then s. Cheap
bitwise decode (all powers of two, no division):
- `k          = StreamKIdx & (Ck-1)`
- `reduceBase = StreamKIdx & ~(Ck-1)`   (owner of this s-row's tile; `k==0`)
- `clusterBase= StreamKIdx & ~(C-1)`
- `s          = (StreamKIdx & (C-1)) >> log2(Ck)`

`StreamKIdx = WorkGroup0` is the cluster-remapped global index (the `ttmp9`
reread is skipped under clustering — `preLoop` `StreamK.py:2616`), so it equals
`cluster_x·C + wg_x`. **Anchor to reuse:** the reduction's bitwise
`cluster_last = StreamKIdx | (C-1)` idiom (`clusterReduceIntraCheck`
`StreamK.py:1590`).

### 1.4 Host grid so `Cs·Ck` WGs map to `(M-tile, K-split)` correctly

- **Grid:** `skGrid = Ck · tiles`, rounded UP to a multiple of `C = Cs·Ck`
  (§5). Every tile is split `Ck` ways (K reduction). `Cs` *adjacent tiles* are
  grouped per cluster — the grouping adds NO extra WGs (unlike Ck), it just
  co-locates existing spatial tiles. Full clusters require `tiles % Cs == 0`
  (⟺ `nWG0 % Cs == 0`, §4).
- **Adjacency:** `skIndexToWG` (`StreamK.py:487-503`) linearizes
  `tileID = WG2·(nWG0·nWG1) + WG1·nWG0 + WG0` (WG0/M fastest). Cluster `c` owns
  tiles `[c·Cs, c·Cs+Cs)`; these are M-adjacent (same N-block) iff `nWG0 % Cs ==
  0`. So the `Cs` s-peers share the same B N-block, and (same k) the same
  K-slice → identical B load → multicastable. Reduction k-peers share `(tile,
  N, M)` but disjoint K → zero A/B reuse (per `cluster-load-…md §1.8`), so they
  reduce but do not multicast.

---

## 2. Multicast along `Cs` (B-broadcast mask over the s-axis only)

### 2.1 The mask must cover only the `Cs` peers sharing `k`

Today's `[C,1]` multicast sets `maskB = (1<<C)-1` — **all C** peers share B
(`ClusterLoad.py:148`, shifted by `wg_y·nwg_x` which is 0). In factored mode B is
shared only by the `Cs` peers with the *same* `k` (same K-slice), i.e.
within-cluster ranks `{k, k+Ck, k+2Ck, …, k+(Cs-1)Ck}`. The correct B mask is:

```
maskB_base = OR over s' in [0,Cs) of (1 << (s'·Ck))     # Cs bits, stride Ck
MulticastMaskB = maskB_base << k                          # k = StreamKIdx & (Ck-1)
MulticastMaskA = 1 << wg_x                                # self-only (A per-WG)
```

Note `maskB_base << k` == the stock `computeMasks` **maskA** shape for a genuine
2-D `[Ck, Cs]` cluster (`ClusterLoad.py:144-146`: `OR_{i<C1}(1<<(i·C0))` with
`C0=Ck, C1=Cs`, shifted by the column index). This confirms Option A's elegance,
but the shift operand here is the **logical `k`** derived from `StreamKIdx`, not
the raw HW `wg_x`/`wg_y` — which is exactly why a factored mask helper is needed
instead of the stock 2-D compute.

### 2.2 Where the mask value changes

- **Compute:** the `computeMasks` call in `defineAndResources`
  (`KernelWriterAssembly.py:2649-2654`) runs at kernel init from HW coords and, for
  a `[C,1]` cluster, yields `maskB=(1<<C)-1`. For factored mode this init value is
  **overwritten** in `preLoop` once `StreamKIdx` (hence `k`, `s`) is known — the
  same place the current predicate runs (`StreamK.py:2631`). Add a
  `streamKFactoredMaskCompute(writer, kernel)` helper (sibling of
  `streamKMulticastMaskPredicate`) that:
  1. computes `k = StreamKIdx & (Ck-1)`,
  2. materializes `maskB_base` (a compile-time constant `OR_{s'}(1<<(s'·Ck))`)
     and emits `SLShiftLeftB32(MulticastMaskB, shiftHex=k, src=maskB_base)`,
  3. leaves `MulticastMaskA = 1<<wg_x` (self) — reuse the existing init compute or
     recompute from `StreamKIdx & (C-1)`.
  Gate: only when `Cs > 1` (multicast active). When `Cs == 1` (pure reduction) B
  is per-WG (self mask) and this helper is inert.
- **Runtime validity gate:** extend `streamKMulticastMaskPredicate`
  (`StreamK.py:2487-2544`) to the factored condition:
  `nWG0 % Cs == 0` (not `% C`) **and** `clusterBase + C <= totalTiles·Ck`
  (the launched grid unit is now `Ck·tiles`; a full cluster spans `C` StreamK
  indices). On failure, rewrite `MulticastMaskB → MulticastMaskA` (self-only),
  exactly as today (`:2541`). The A/B split-name forcing in
  `ClusterLoad.usesCombinedMask`/`maskSgprName` (`:57,80`) already keys on
  `StreamKMulticast` and needs no change.
- **No DP→SK boundary clear.** Factored mode uses the skSplit schedule (every WG
  is a fixed `(tile, k)` partial-tile WG, §5), so there is **no DP round and no
  DP→SK transition**; `streamKMulticastBoundaryClear`
  (`StreamK.py:2546-2570,2950`) is inert (guarded by the DP-only/DP-first path).
  This is a behavioral divergence from the current multicast MVP, which required
  `StreamKForceDPOnly=1`. See §3.4.

---

## 3. Reduction along `Ck` (the shared cluster-barrier problem)

### 3.1 The problem, precisely

The `-3` barrier syncs **all `C = Cs·Ck` WGs**, but K-reduction wants a
per-s-group rendezvous of the `Ck` arrivals sharing each `s`. There is no
sub-cluster barrier in HW. The reduction MVP sidestepped this by making
`cluster == one tile's peers` (`skSplit == C`, one tile per cluster,
`streamk-wg-clusters.md §2.3`). Factored mode has `Cs` tiles per cluster
(`skSplit == Ck < C`), so the cluster-wide barrier no longer equals one tile's
peer set.

### 3.2 Analysis: the full-cluster barrier IS usable (benign over-sync)

Because id `-3` releases a wait phase when arrivals reach the full membership
`C`, and multiple waiters share the phase:

- **Every** one of the `C` WGs arrives exactly once (`clusterReduceSignal`,
  wave-0 elect) at the epilogue reduction phase — both the `Cs` owners (`k==0`)
  and the `Cs·(Ck-1)` non-owners (`k!=0`).
- The `Cs` owners each `clusterReduceWait` once.
- Arrivals `= C` (one per WG, independent of how many wait) → the phase releases
  → **all `Cs` owners release together**, with the guarantee that **all `Ck`
  partials of every s-row are published**. This is a *stronger* guarantee than
  each owner needs (it also waits on other s-rows' peers), but it is **correct**
  and **deadlock-free**: counts stay balanced (C arrivals, any number of waits).

**Conclusion: use the existing cluster-wide barrier unchanged.** No per-s-group
barrier is required or possible. Over-synchronization across s-rows is harmless
(all s-rows finish their equal-length mainloop together anyway, §3.3). The
reduction then proceeds per owner over its **own** `Ck-1` k-peers.

### 3.3 Balanced signal/wait counts across the whole kernel

Factored mode is the first config where the `-3` barrier is used by BOTH the
mainloop multicast-lockstep path AND the epilogue reduction. Balance argument:

1. **Prologue:** `streamKMulticastPrologueSignal` (`StreamK.py:2572-2604`) — one
   arrive per WG, pairs the InsertClusterBarrierPass first-load wait. All `C`.
2. **Mainloop:** the `ClusterBarrier` path (enabled at `Solution.py:1201-1205`;
   stinkytofu `InsertClusterBarrierPass`) brackets each `tensor_load_to_lds` with
   signal/wait. Every WG runs **exactly `itersPerTile/Ck`** mainloop iterations
   (fixed even K-split), so all `C` WGs emit the identical number of barrier
   phases and stay in lockstep. **This is why `ceil(K/DepthU) % Ck == 0` is
   mandatory** (equal mainloop length across k-peers *and*, trivially, across
   s-peers). The B multicast among the `Cs` s-peers happens on these lockstepped
   loads via `MulticastMaskB`.
3. **Epilogue:** one arrive per WG (`clusterReduceSignal`) + one wait per owner
   (`clusterReduceWait`). All `C` arrive → balanced.

Every WG's total arrivals `= 1 (prologue) + N (mainloop, N=itersPerTile/Ck) + 1
(epilogue)`, identical across all `C` → no WG ever leaves the kernel with an
unmatched arrival, no waiter blocks forever. **This whole-kernel balance is the
top correctness risk and must be asserted by a codegen snapshot (§7).**

### 3.4 Owner election, fixup peer count, intra-cluster guard

- **Owner** of an s-row's tile = the `k==0` WG (`StreamKLocalStart == 0`), which
  falls out of the skSplit=Ck schedule (`skTileIndex` `StreamK.py:439-483`). There
  are now **`Cs` owners per cluster** (not 1).
- **Fixup peers:** the owner reduces its `Ck-1` k-peers `[StreamKIdx+1 …
  StreamKIdx+Ck-1]` = `[reduceBase+1 … reduceBase+Ck-1]`. The existing skSplit
  fixup loop already walks `skSplit-1 == Ck-1` peers per tile
  (`storeBranchesCommon` fixup bounds via `sFixupEnd`, `StreamK.py:955-1032` @
  `0eed`) — so with `skSplit=Ck` the accumulation math is **unchanged**. The
  non-owner publishes to its WS slot (`computeWorkspaceSrd`,
  `writePartials`) and signals (`clusterReduceSignal` `:1400`).
- **Intra-cluster guard:** `clusterReduceIntraCheck`
  (`StreamK.py:1573-1611`) keeps `cluster_last = StreamKIdx | (C-1)` compared to
  `skGrid` (validate the **full** `C` cluster is populated, since the barrier is
  cluster-wide). No change to the C used there; only the fixup peer count (Ck)
  and owner detection (per s-row) differ, and those are already driven by
  `StreamKLocalStart`/`skSplit`. When the guard fails (trailing partial cluster),
  the **entire cluster** falls back to the global-flag reduction (unchanged
  path). Because the fallback global-flag reduction walks the same `Ck` peers, it
  is correct for the factored split too.
- `_streamKClusterReductionEnabled` (`StreamK.py:1520-1537`) keeps its gate, but
  drop the `and not kernel["StreamKForceDPOnly"]` conflict only insofar as
  factored mode never sets DP-only (it uses the skSplit schedule).

---

## 4. Constraints / predicates and where enforced

| Constraint | Meaning | Enforced where |
|---|---|---|
| `Cs·Ck == C`, `C = ClusterDim[0]`, `ClusterDim[1] == 1` | 1-D HW cluster | `Solution.py` factored validator (new) |
| `Ck` power-of-two, `Ck ∈ [1,C]`, `Ck \| C` ⇒ `Cs` power-of-two | valid factoring | `Solution.py` validator |
| `Cs·Ck <= 16` | HW `maxWGsInCluster` (`ValidParameters.py:74`) | `Solution.py` validator (subsumed by `ClusterDim` validity) |
| `nWG0 % Cs == 0` | s-peers M-adjacent (share B) | build: — (runtime K-independent but nWG0 known at selection) → **`ClusterDimCheck`** with `value[3]=Cs` (§5); runtime kernel gate `streamKMulticastMaskPredicate` (`% Cs`) |
| `ceil(K/DepthU) % Ck == 0` | equal mainloop length across k-peers; split-barrier arrivals balanced | selection: **`ClusterReductionIterCheck`** `value=[DepthU, Ck]` (`Contractions.py:605-606`; predicate `ContractionProblemPredicates.hpp:3100-3155`) |
| gfx1250 `(12,5,0)`, `HasTDM`, `HasClusterBarrier`, `TDMInst==3` | multicast is TDM; barrier handshake | `Solution.py` validator (merge of both current validators) |
| `StreamK==3`, `!StreamKAtomic`, `StreamKXCCMapping==0` | SK3 DP schedule; XCC overflow | `Solution.py` validator |
| `!StreamKForceDPOnly` | factored needs partial tiles (K-split) | `Solution.py` validator |

Key change vs today: `ClusterDimCheck`'s divisibility must test **`Cs`** for the
multicast M-alignment (not `C`). Today `Contractions.py:593-594` feeds
`value[3]=ClusterDim[0]=C`, `value[4]=ClusterDim[1]=1`, and the predicate checks
`nWG_x % C == 0` (`ContractionProblemPredicates.hpp:3060`). For factored mode the
M-adjacency requirement is `nWG_x % Cs == 0`, so **feed `value[3]=Cs`** when
`Cs>1` (multicast active). The `nWG_y % C1` term stays (`C1=1` ⇒ no-op).
`ClusterReductionIterCheck` must use **`Ck`** (not `C`) as `value[1]`.

---

## 5. Host grid + kernarg for the factored layout

`getSKGridImpl` (`ContractionSolution.cpp:3909-4075`): replace the two mutually
exclusive overrides with one factored override (Ck-driven grid, Cs-driven
grouping):

```cpp
// Factored cluster mode (gfx1250): C = Cs*Ck.
//   Ck>1 : each tile split Ck ways  -> grid multiplies by Ck  (reduction axis)
//   Cs>1 : Cs adjacent tiles grouped per cluster -> NO extra WGs (spatial axis)
// => skGrid = Ck * tiles, rounded up to a multiple of C = Cs*Ck so every HW
//    cluster is full. tiles % Cs == 0 (== nWG0 % Cs) is required for full
//    clusters; a trailing partial cluster falls back (multicast self-mask +
//    global-flag reduction) via the kernel runtime guards.
if (streamKFactoredCluster && clusterDim.x > 1) {
    size_t C  = clusterDim.x;         // = Cs*Ck
    size_t Ck = streamKClusterKSplit; // 1 => pure multicast, C => pure reduction
    skGrid = ((Ck * tiles + C - 1) / C) * C;
}
```

Degenerate checks: `Ck==1 ⇒ skGrid = ceil(tiles/C)·C` (current multicast
`:4068-4072`); `Ck==C ⇒ skGrid = C·tiles` (current reduction `:4097-4100`). ✔

`solve()` re-round guard (`:3198-3202` multicast / `:3218-3223` reduction @
`0eed`): unify to `sk.grid = ceil(sk.grid / C)·C` whenever
`streamKFactoredCluster && clusterDim.x > 1`.

**Kernarg block** (`generateSingleCall`, `ContractionSolution.cpp:930-992` @
`0eed`): the two-tile cluster path must set `skSplit = Ck` (not `C`):

```cpp
if (streamKFactoredCluster && clusterDim.x > 1
    && sk.grid == roundUpToMultipleOfC(Ck * tiles)) {   // contract intact
    uint32_t skItersPerWG = itersPerTile / Ck;          // was / C
    args.append("SKItersPerWG", skItersPerWG);
    args.append("skGrid",       sk.grid);
    args.append("skTiles",      tiles);                  // every tile split Ck ways
}
```

This makes `StreamKIdx // Ck == tile`, `StreamKIdx % Ck == k` (§1.3). The
existing invariant guard ("disable path if a fallback re-rounded sk.grid",
`:934-944` @ `0eed`) carries over with the `Ck·tiles` contract. New host field:
`sizeMapping.streamKClusterKSplit` (mirror `streamKMulticast`:
`ContractionSolution.hpp:146`, `Serialization/ContractionSolution.hpp:109`,
`Contractions.py:630,725`). `sizeMapping.streamKMulticast` and
`streamKClusterReduction` remain as derived bools (`Cs>1`, `Ck>1`) for the
existing guards, or are subsumed by a single `streamKClusterKSplit` + `clusterDim`.
Launch (`ContractionSolution.cpp:1765-1794`) is unchanged: `clusterDim.x = C`,
`.y = 1`, grid multiple of `C`.

---

## 6. Relaxing the mutual exclusion

Today (`0eed`): the collapse gives reduction precedence
(`Solution.py:1256`: `and not StreamKClusterReduction`), and
`_validateStreamKMulticast` hard-rejects when reduction is on
(`Solution.py:352-360`). Change to allow both when a valid `(Cs,Ck)` factoring
exists.

1. **Collapse** (`Solution.py:1162-1163` current / `1255-1257` @ `0eed`): derive
   `Ck = state["StreamKClusterKSplit"]`, `Cs = C // Ck` on `StreamK==3 &&
   ClusterDim!=[1,1]`. Set derived keys:
   ```python
   C  = state["ClusterDim"][0]
   Ck = state.get("StreamKClusterKSplit", 1)
   Cs = C // Ck
   state["StreamKMulticast"]        = 1 if Cs > 1 else 0
   state["StreamKClusterReduction"] = 1 if Ck > 1 else 0
   ```
   Remove the `and not StreamKClusterReduction` exclusion term.
2. **`_validateStreamKMulticast`**: delete the mutual-exclusion reject
   (`:352-360` @ `0eed`). Keep all other checks. The `Multicast!=0` requirement
   stays (masks must be declared). Only gate the B-mask/predicate emitters on
   `Cs>1`.
3. **`_validateStreamKClusterReduction`**: keep all checks; it already permits
   `ClusterDim=[C,1]`. Ensure it does not reject when multicast is also on.
4. **New unified validator** (or extend both): assert `Ck | C`, `Ck` & `Cs`
   powers of two, `Ck ∈ [1,C]`. When `Ck==1` route to pure-multicast checks;
   `Ck==C` to pure-reduction; else both.
5. `ValidParameters.py`: add `StreamKClusterKSplit`; document that it factors the
   existing `ClusterDim[0]`.

Net: `_validateStreamKMulticast` and `_validateStreamKClusterReduction` become
**composable** rather than mutually exclusive; the collapse enables both derived
keys from one `(ClusterDim[0], StreamKClusterKSplit)` pair.

---

## 7. Task breakdown (ordered, file-by-file)

Sequencing preserves the two degenerate cases as green regression gates before
the general mode. `[P]` = parallelizable within a group.

### Group 0 — Prep / snapshots (no source change)
- **T0.1** Add char snapshots for the two degenerate factored configs
  (`Ck=1` == current multicast; `Ck=C` == current reduction) so the refactor is
  provably byte-exact for both. Anchor:
  `Tests/unit/characterization/_codegen/test_streamk_cluster_multicast_gfx1250_char.py`,
  `..._reduction_gfx1250_char.py`.

### Group 1 — Parameter + host plumbing
- **T1.1** `ValidParameters.py:~1117`: add `StreamKClusterKSplit`.
- **T1.2** `Solution.py`: collapse (§6.1) + validators (§6.2-6.4); derive
  `StreamKMulticast`/`StreamKClusterReduction` from `(C, Ck)`.
- **T1.3 [P]** Host field `sizeMapping.streamKClusterKSplit`
  (`ContractionSolution.hpp:146`, `Serialization/ContractionSolution.hpp`,
  `Contractions.py:630,725`).
- **T1.4 [P]** `getSKGridImpl` unified factored grid (`ContractionSolution.cpp`
  `:4059-4100` region) + `solve()` re-round (`:3198-3223`).
- **T1.5** Kernarg block `skSplit=Ck` (`ContractionSolution.cpp:930-992` @ `0eed`
  region).
- **T1.6** Predicates: `ClusterDimCheck value[3]=Cs` when `Cs>1`
  (`Contractions.py:589-595`); `ClusterReductionIterCheck value=[DepthU, Ck]`
  when `Ck>1` (`Contractions.py:605-606`).

### Group 2 — Kernel: factored mask (multicast along s)
- **T2.1** `ClusterLoad.py` (or a StreamK helper): `streamKFactoredMaskCompute`
  emitting `MulticastMaskB = maskB_base << k`, `MulticastMaskA = 1<<wg_x`
  (§2.1-2.2). Call from `preLoop` (`StreamK.py:2631` region) replacing/extending
  `streamKMulticastMaskPredicate` for `Cs>1`.
- **T2.2** Generalize `streamKMulticastMaskPredicate` (`StreamK.py:2487-2544`):
  gate `nWG0 % Cs` and `clusterBase + C <= Ck·totalTiles`.
- **T2.3** Confirm `usesCombinedMask`/`maskSgprName` split-A/B forcing
  (`ClusterLoad.py:57,80`) still holds (keys on `StreamKMulticast`). No change
  expected.

### Group 3 — Kernel: reduction along k (reuse existing barrier)
- **T3.1** Bring the reduction helpers onto the working branch: `clusterReduce*`
  (`StreamK.py:1520-1611` @ `0eed`), owner-wait (`:958-1010`), non-owner-signal
  (`:1388-1406`). No barrier-scope change (§3.2).
- **T3.2** Ensure the epilogue reduction phase co-exists with the mainloop
  `ClusterBarrier` lockstep phases (§3.3) — verify balance; adjust
  `_streamKClusterReductionEnabled` so it does not require DP-only.
- **T3.3** Owner detection per s-row (`k==0`) + fixup over `Ck-1` peers falls out
  of `skSplit=Ck`; verify `sFixupEnd`/`sCtaIdx` bounds.

### Group 4 — Enable both + tests (§ below)

Dependency order: G0 → G1 → (G2 ∥ G3) → G4.

---

## 8. Test plan

| Behavior | Type | Where |
|---|---|---|
| Degenerate `Ck=1` byte-identical to current multicast | char snapshot (gate) | `test_streamk_cluster_multicast_gfx1250_char.py` |
| Degenerate `Ck=C` byte-identical to current reduction | char snapshot (gate) | `test_streamk_cluster_reduction_gfx1250_char.py` |
| Factored `(Cs,Ck)` decode: `k=idx&(Ck-1)`, `s=(idx&(C-1))>>log2Ck` | CPU asm-string unit | new `Tests/unit/test_streamk_factored_cluster.py` |
| B-mask = `maskB_base<<k` (Cs bits, stride Ck); A-mask = self; predicate gates on `nWG0%Cs` | char snapshot | new `test_streamk_factored_cluster_gfx1250_char.py` + designed `_designed/gfx1250/streamk_factored_cluster.yaml` |
| **Whole-kernel `-3` balance**: prologue(1)+mainloop(N)+epilogue(1) arrivals per WG; Cs owner waits | char snapshot assert (signal/wait counts) | same char test |
| Validation matrix: accept `(Cs>1 && Ck>1)` on SK3+gfx1250+TDMInst3+XCC0+`!DPOnly`+`!Atomic`; `Ck∤C` reject; non-pow2 reject; both degenerate cases accept | CPU unit | `test_streamk_factored_cluster.py` (+ extend `test_streamk_multicast.py`, `test_streamk_cluster_sk45_reject.py`) |
| `ClusterDimCheck` uses Cs; `ClusterReductionIterCheck` uses Ck | CPU unit (predicate values) | `test_streamk_factored_cluster.py` |
| GPU roundtrip `(Cs,Ck)∈{(2,2),(2,4),(4,2)}`: reduces correctly, kernel completes cleanly | GPU roundtrip `@requires_gpu_gfx1250` | new `test_streamk_factored_cluster_gpu.py` (watchdog) |
| End-to-end multi-WG factored GEMM | C++ client FFM | new `Tests/common/streamk/gfx1250/core/sk_mxf8gemm_factored_cluster.yaml` (+ mxf4 sibling) |

FFM configs to add (mirror the shipped multicast/reduction siblings under
`Tests/common/streamk/gfx1250/core/`): `sk_mxf8gemm_factored_cluster.yaml` and
`sk_mxf4gemm_factored_cluster.yaml`, each pinning `ClusterDim: [4,1]` +
`StreamKClusterKSplit: 2` (⇒ Cs=2, Ck=2), sizes with `nWG0 % 2 == 0` and
`ceil(K/DepthU) % 2 == 0`.

---

## 9. Risks / open questions

| Risk | Why | Mitigation |
|---|---|---|
| **Whole-kernel barrier imbalance** | first config to use `-3` in BOTH mainloop lockstep AND epilogue reduction | §3.3 balance proof + snapshot asserting per-WG arrival counts; keep `ceil(K/DepthU) % Ck == 0` hard (equal mainloop length) |
| **Over-sync deadlock across s-rows** | owner waits on the full C cluster, not just its Ck | §3.2: benign — releases at C arrivals; all C reach the epilogue signal (all are partial-tile WGs) |
| **Wrong-B multicast (silent)** | s-peers not actually M-adjacent (nWG0 ∤ Cs, partial cluster) | `streamKMulticastMaskPredicate` self-mask fallback (`% Cs`, `clusterBase+C <= Ck·totalTiles`); `ClusterDimCheck` selection reject |
| **skSplit=Ck vs cluster=C mismatch in fixup bounds** | reduction MVP assumed skSplit==C | fixup walks `skSplit-1=Ck-1` peers (already skSplit-driven); intra-check still validates full C |
| **`MulticastMaskB` init (2-D compute) vs preLoop overwrite ordering** | `computeMasks` runs at init with HW coords, factored value set in preLoop | overwrite `MulticastMaskB` in preLoop after `StreamKIdx` known (like current predicate); snapshot-gate |
| **Multicast cooperative-load stability (gfx1250)** | factored mode uses the DP cooperative B-multicast path, which is currently supported only with single-buffered global prefetch (`PrefetchGlobalRead <= 1`) on gfx1250; and unlike the shipped multicast MVP it cannot use the `StreamKForceDPOnly` single-round escape (factored needs partial tiles, §3.4) | **GATING DEPENDENCY** — gfx1250 multicast cooperative-load stability: defer real-HW / MXFP4 enablement until the multicast cooperative-load path is validated at `PrefetchGlobalRead > 1`; land codegen + CPU/snapshot tests behind that gate (FFM MXFP4 configs deferred until cleared). |
| **SGPR pressure** | extra `k`/`s`/`maskB_base` scratch on the already-tight SK path | reuse `allocTmpSgpr` scopes + bitwise (pow2) decode; forbid `StreamKXCCMapping!=0` |

### Open questions (human decision at approval gate)
1. **Spec: factor param (Option B) vs logical `[Cs,Ck]` ClusterDim (Option A)?**
   Plan recommends B (§1.2). Confirm.
2. **Decode order: "k fastest" (recommended, §1.3) vs "s fastest".** k-fastest
   makes reduction peers contiguous (reuses skSplit fixup verbatim) and multicast
   peers strided (mask handles it). Confirm.
3. **HW behavior of a masked multicast target that is mid-reduction / idle** —
   same open question as the multicast MVP (`cluster-load-…md §9.1`); factored
   mode's conservative full-cluster predicate assumes the safe answer. Needs HW
   confirmation.
4. **Multicast cooperative-load stability at `PrefetchGlobalRead > 1`** — is the
   DP cooperative B-multicast path validated at `PrefetchGlobalRead > 1` on
   gfx1250? Determines whether the factored mode can enable on real HW for MX
   types at all (gating, §9 table).

---

## 10. Recommended incremental path

1. **Step 0 (degenerate multicast):** `Ck=1, Cs=C` — must reproduce the current
   `StreamKMulticast` byte-for-byte (T0.1 gate). Validates the param/host
   refactor with zero behavior change.
2. **Step 1 (degenerate reduction):** `Ck=C, Cs=1` — must reproduce the current
   `StreamKClusterReduction` byte-for-byte. Validates the unified grid/kernarg.
3. **Step 2 (smallest factored):** `C=4, Ck=2, Cs=2` — first genuine both-on
   kernel. Land codegen + CPU/snapshot tests (incl. the barrier-balance assert).
4. **Step 3 (general):** `Cs,Ck > 1` sweep `{(2,4),(4,2),(2,2),(4,4)}` within
   `Cs·Ck<=16`; GPU roundtrip once the multicast cooperative-load dependency (§9) clears.

---

## Appendix A — Worked example (C=4, Cs=2, Ck=2, one cluster c=0)

`skGrid = Ck·tiles = 2·tiles`, `skSplit=Ck=2`, `SKItersPerWG=itersPerTile/2`.
Cluster 0 = StreamKIdx `{0,1,2,3}`, tiles `{0,1}` (Cs=2 adjacent tiles):

| StreamKIdx | k=idx&1 | s=(idx&3)>>1 | tile=idx>>1 | role | B-mask peers (share k) | K-reduce peers (share s) |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | owner t0 | {0,2} (k=0) | {0,1} (s=0) |
| 1 | 1 | 0 | 0 | peer  t0 | {1,3} (k=1) | {0,1} (s=0) |
| 2 | 0 | 1 | 1 | owner t1 | {0,2} (k=0) | {2,3} (s=1) |
| 3 | 1 | 1 | 1 | peer  t1 | {1,3} (k=1) | {2,3} (s=1) |

- B multicast: WG0↔WG2 (both k=0, tiles 0&1 M-adjacent, same K-slice → same B);
  WG1↔WG3 (k=1). `MulticastMaskB(WG0)= (1<<0 | 1<<2)=0b0101`;
  `MulticastMaskB(WG1)=0b0101<<1=0b1010`.
- K reduction: owner WG0 waits, reduces peer WG1 (its `Ck-1=1` peer); owner WG2
  reduces peer WG3. Both owners release from ONE cluster-wide barrier once all 4
  arrive.
