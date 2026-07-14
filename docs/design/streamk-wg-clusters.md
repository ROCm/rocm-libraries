<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Design: WG Clusters for StreamK GEMM (gfx1250)

**Target arch:** gfx1250 (ISA `(12,5,0)`, wave32).
**Scope:** Use gfx1250 workgroup *clusters* to accelerate the StreamK partial-tile
reduction by replacing the cross-CU global-flag spin-wait with an intra-cluster
split barrier, while keeping a correct fallback for the general
(peers-span-multiple-clusters / DP / atomic) cases. The MVP implemented here is
StreamK variant 3 (two-tile), barrier-only, linear-reduction fast path, with a
fixed even split (one tile per cluster); the global-flag reduction is retained
as a runtime/compile fallback. Source-line anchors below reference the code as
implemented.

---

## 1. Background

### 1.1 Cluster hardware + existing (non-StreamK) support

- **Split barrier objects** are addressed by negative id: `-1` = workgroup,
  `-3` = cluster. The rocisa `SBarrier(separate, wait, clusterBarrier, comment)`
  emitter lowers to id `-3` when `clusterBarrier=True`. Only one wave per WG
  signals/arrives; any wave in the cluster may wait.
- **Solution derivation** (`Tensile/SolutionStructs/Solution.py:975-981`):

  ```975:981:projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py
      state["Multicast"] = False
      state["ClusterBarrier"] = False
      if state["ClusterDim"] != [1, 1]:
        state["Multicast"] = True
        # ClusterBarrier emits SCmp/branch on sgpr("WaveIdx"), which is only allocated when TDM is enabled.
        if state["TDMInst"] != 0 and isaInfoMap[state["ISA"]].asmCaps.get("HasClusterBarrier", False):
          state["ClusterBarrier"] = True
  ```

  Note: `ClusterDim != [1,1]` unconditionally forces `Multicast=True`. This
  coupling must be broken for a barrier-only StreamK path (see §5).
- **Constraints:** `maxWGsInCluster = 16`; `validClusterDimensions` is all
  `[i,j]` with `i*j <= 16` (`Common/ValidParameters.py:74-80`, exported at
  `:1107`). For `PrefetchGL2`/non-subtile, `ClusterDim` components must be
  powers of two and `!= [1,1]` (`Solution.py:5296-5302`).
- **Host launch is already cluster-aware:** `src/hip/HipSolutionAdapter.cpp:573-604`
  uses `hipDrvLaunchKernelEx` + `hipLaunchAttributeClusterDimension`, gated on
  `HIP_HAS_CLUSTER_LAUNCH`, with `kernel.clusterDim` from
  `ContractionSolution.hpp` / `Contractions.py`. The comment at `:586-589`
  states explicitly: *"The grid dimension should be a multiple of cluster
  size."* The launch sets `attribute[0].val.clusterDim.z = 1` — clusters are
  effectively 2-D `[x,y]`.
- **Kernel-side WG-id derivation under clustering**
  (`KernelWriterAssembly.py:2581-2644`). When `WorkGroupIdFromTTM` and
  `enableCluster = (ClusterDim[0]*ClusterDim[1]) != 1`, and `cluster_id != 0`,
  the WG ids are rebuilt from the cluster HW regs:

  ```2628:2632:projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py
              moduleRegInit.add(SMulI32(dst=sgpr("WorkGroup0"), src0="ttmp9", src1=sgpr(sTmp+3),\
                                 comment="cluster_x * nwg_x"))
              moduleRegInit.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), \
                                 src1=sgpr(sTmp+1), \
                                 comment="WorkGroup0 = (cluster_x * nwg_x) + wg_x"))
  ```

  i.e. **`WorkGroup0 = cluster_x * nwg_x + wg_x`**, where `nwg_x = ClusterDim[0]`
  is the number of WGs per cluster along x, `cluster_x` is the cluster index,
  `wg_x` is the WG's position within the cluster. **Key property: WGs of one
  cluster occupy a *contiguous* `WorkGroup0` range `[cluster_x*C, cluster_x*C + C)`
  with `C = ClusterDim[0]`.** (`ttmp9` here is the raw `wg_x`.)
- **WGM/XCC remap is bypassed under clustering:**
  `Components/WorkGroupMappingAlgos.py:99-101` short-circuits `wgmXCC` when
  `enableCluster`.

### 1.2 StreamK internals (verified)

- Variants dispatched by `kernel["StreamK"]` in
  `Tensile/Components/StreamK.py`: `0` Off; `3` `StreamKTwoTileDPFirst`
  (`:2479`); `4` `StreamKDynamic` (`:2896`, atomic work queue); `5`
  `StreamKHybrid` (`:3301`). Table at `:4087-4089`. Orthogonal flags:
  `StreamKAtomic`, `StreamKForceDPOnly`, `StreamKFixupTreeReduction`.
- **StreamKIdx == WorkGroup0** (`StreamK.py:2504`):

  ```2504:2505:projects/hipblaslt/tensilelite/Tensile/Components/StreamK.py
              module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"),
                                 comment="Save original StreamK index"))
  ```

  **Gotcha:** the SK3 `preLoop` re-reads a *raw* WG id from `ttmp9`
  immediately before this, overwriting the cluster-remapped value:

  ```2495:2498:projects/hipblaslt/tensilelite/Tensile/Components/StreamK.py
          if writer.states.archCaps["WorkGroupIdFromTTM"]:
              module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
              module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
              module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))
  ```

  Under clustering `ttmp9 == wg_x` (position *within* the cluster), **not** the
  global linear index. This workaround must be made cluster-aware (§2).
- `StreamKIter = StreamKIdx * SKItersPerWG (+ extras)` (tree-reduction init at
  `:2621-2626`; parallel/DP paths at `:2513`, `:2569-2587`). So **consecutive
  `StreamKIdx` map to consecutive global-iteration windows.**
- Tile mapping: `skTileIndex` (`:439`) magic-divides the global iter into a tile
  index and derives `StreamKLocalStart`/`StreamKLocalEnd`; `skIndexToWG` (`:487`)
  maps tile → `WorkGroup0/1/2`. **Owner** of a tile is the WG with
  `StreamKLocalStart == 0` (it runs the reduction + final store).
- **Reduction (producer/consumer over global workspace + flag)** lives in the
  epilogue, `storeBranchesCommon` (`:744`), in two topologies:
  - *linear* (`:911-1035`): owner sets `sCtaIdx = StreamKIdx+1` (`:941`) and
    walks **consecutive** peer indices `+1,+2,…`, per-peer spin-waiting on the
    global flag:

    ```961:965:projects/hipblaslt/tensilelite/Tensile/Components/StreamK.py
                  module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(sCtaIdx), shiftHex=log2(4), comment="flag offset based on CTA index"))
                  module.add(memOrder.readFlag(writer, dst=tmpSgpr+2, soffset=sgpr(tmpSgpr)))
                  if kernel["DebugStreamK"] & 2 == 0:
                      module.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=1, comment="check if ready"))
                      module.add(SCBranchSCC0(labelName=skFixupLabel.getLabelName(), comment="if flag not set, wait and check again"))
  ```

    then `acquireFence`, a **workgroup** `SBarrier` (`:968`), wave-0 flag reset
    (`:970-979`), and `fixupStep` (`:1010`, defined `:1680`) which loads peer
    partials from the workspace and `VAddF32`-accumulates.
  - *tree* (`StreamKFixupTreeReduction=1`, `:755-910`): same handshake with a
    log-tree peer schedule (`sIdxOffset *= 2`).
  - Non-owners write their partial to the global workspace slot via
    `computeWorkspaceSrd` (`:1108`, slot `= AddressWS + StreamKIdx*(MT0*MT1*bpeCinternal)`)
    and raise their flag via `setFlagValue` (`:1414`).
- **Memory ordering** (`StreamK.py:112-249`). gfx1250 uses
  `StreamKMemoryOrderingDevScopeFences` (cap `HasInvWbDevFences`): release =
  `s_waitcnt` + `global_wb scope:SCOPE_DEV`; acquire = `global_inv
  scope:SCOPE_DEV`; the flag is read via **VMEM** (`readFlag`, `:236-246`) with
  `s_wait_xcnt 0` before volatile VMEM (`preVolatileVmem`, `:144-153`,
  `RequiresXCntForVolatileVMEM`).
- **Flags buffer:** `library/.../hipblaslt.cpp` allocates + zeros `d_Synchronizer`
  and passes it as `AddressFlags`; `AddressFlags == 0` is the sentinel meaning
  "parallel / post-kernel reduction" (checked at `StreamK.py:579`, `:2533`).
- **Host grid** forces `{sk.grid,1,1}` for StreamK
  (`ContractionSolution.cpp:1758-1763`); the `enableCluster` branch immediately
  after keeps y/z unflattened for cluster kernels:

  ```1758:1774:projects/hipblaslt/tensilelite/src/ContractionSolution.cpp
          if(sizeMapping.streamK != 0)
          {
              rv.numWorkGroups.x = sk.grid;
              rv.numWorkGroups.y = 1;
              rv.numWorkGroups.z = 1;
          }

          bool enableCluster = (sizeMapping.clusterDim.x > 1 || sizeMapping.clusterDim.y > 1);
          if(!enableCluster)
          {
              if(internalArgsSupport.version >= 1)
              {
                  rv.numWorkGroups.x *= (rv.numWorkGroups.y * rv.numWorkGroups.z);
                  rv.numWorkGroups.y = 1;
                  rv.numWorkGroups.z = 1;
              }
          }
  ```

  `sk.grid` is chosen by `getSKGridImpl` (`:3895-4045`); the SK3 per-WG
  accounting (`skTiles`, `SKItersPerWG`, `extraIters`) is built at `:856-906`
  and `:945-973`. There is **no** rounding of `sk.grid` to a cluster multiple
  today.

### 1.3 Cluster-barrier codegen paths that already exist

- **Subtile Python path** `Tensile/Components/Subtile/ClusterBarrier.py`:
  `subtileClusterBarrierSignal` (wave-0 election: `s_cmp_eq_u32 WaveIdx==0`,
  `s_cbranch_scc0`, `SBarrier(True,False,True)` = `s_barrier_signal -3`),
  `subtileClusterBarrierWait` (`SBarrier(True,True,True)` = `s_barrier_wait -3`),
  and `insertClusterBarrier` which **splices around the mainloop's workgroup
  barrier** and asserts `asmCaps["HasClusterBarrier"]`.
- **stinkytofu C++ pass** `shared/stinkytofu/src/transforms/asm/InsertClusterBarrierPass.cpp`,
  run from `Gfx1250Backend.cpp` when `moduleOptions.ClusterBarrier`. Its anchors
  are all **mainloop / load / tail-loop** points: `tensor_load_to_lds` (Rule 4),
  `label_GSU_1` (Rule 1), `label_openLoopL` (Rule 3, currently disabled), and the
  `/* Tail Loop */` TEXTBLOCK marker (Rule 5). **None of these fire in the
  StreamK epilogue/store reduction path** — verified by reading the pass in full.

---

## 2. WG-id / indexing reconciliation

### 2.1 The core geometric fact

With `ClusterDim = [C, 1]` (1-D cluster along x), the kernel-side remap gives
`WorkGroup0 = cluster_x * C + wg_x`. StreamK uses `StreamKIdx = WorkGroup0` and
`StreamKIter = StreamKIdx * SKItersPerWG (+extra)`, and the reduction walks
**consecutive** `StreamKIdx` (`sCtaIdx = StreamKIdx+1, +2, …`). Therefore:

> **Cluster `c` owns exactly the contiguous StreamK index range
> `[c*C, c*C + C)`, which is also a contiguous block of global iteration
> windows.** A tile whose peer set is a contiguous run of `p` indices starting at
> its owner `k` is fully intra-cluster **iff** `k` and `k+p-1` fall in the same
> `[c*C, c*C+C)` block.

This is the property the whole design leans on: no reshuffling is needed, the HW
remap already clusters consecutive StreamK indices.

### 2.2 Chosen mapping

- **`ClusterDim = [C, 1]`**, `C` a power of two, `2 <= C <= 16`, `ClusterDim[1] == 1`.
  A 1-D cluster is required because the StreamK grid is 1-D `{skGrid,1,1}`; a
  `ClusterDim[1] > 1` would demand `gridDimY % ClusterDim[1] == 0` while
  `gridDimY == 1` (launch at `HipSolutionAdapter.cpp:589-600`), which cannot be
  satisfied.
- **Fix the `ttmp9` workaround** (`StreamK.py:2495-2498`): under
  `enableCluster`, `StreamKIdx` must be the *global* linear index
  `cluster_x*C + wg_x`, not raw `ttmp9 (= wg_x)`. Two options:
  1. *(preferred)* When `enableCluster`, **skip** the `ttmp9` workaround and use
     the already cluster-remapped `WorkGroup0` produced in
     `KernelWriterAssembly.py:2588-2644`.
  2. Recompute `cluster_x` from `HW_REG_IB_STS2[6:4]` locally and form
     `cluster_x*C + wg_x`. More SGPR traffic; only needed if the remapped
     `WorkGroup0` is not live at `preLoop`.
- **Derive cluster-local coordinates** for barrier owner election and peer
  bookkeeping (cheap, from values already computed):
  - `wg_in_cluster = StreamKIdx & (C-1)` (C power of two).
  - `cluster_base = StreamKIdx & ~(C-1)` (first StreamKIdx of this cluster).
  - `cluster_last = cluster_base + C - 1`.
- **Grid sizing (host).** `getSKGridImpl` (`ContractionSolution.cpp:3895-4045`)
  must round `sk.grid` **up to a multiple of `C`** when the StreamK-cluster path
  is active, and that rounded value must flow into both `rv.numWorkGroups.x`
  (`:1760`) and the `skGrid`/`SKItersPerWG`/`skTiles` kernel args (`:856-906`,
  `:945-973`). The tree-fixup `< 2^24 / 2^16` guards at `:4031-4043` still apply
  to the rounded grid. `HipSolutionAdapter.cpp:589` already passes
  `numWorkGroups.x` as `gridDimX`, so a multiple-of-`C` grid satisfies the HW
  requirement with no launch change.

### 2.3 One-tile-per-cluster alignment (MVP partition)

To make the intra-cluster fast path the *guaranteed* common case (and to sidestep
the deadlock hazard of §6), the MVP additionally **aligns each StreamK tile's
peer group to one cluster**:

- Choose the per-WG split so **peers-per-tile `== C`** and **tiles align to
  cluster boundaries**: `SKItersPerWG * C == itersPerTile` and
  `skGrid == C * skTiles` (after rounding). Then cluster `c` ⟷ StreamK tile `c`,
  and every WG in a cluster is a peer of the same tile.
- This is a constrained sub-mode of SK3 (fixed, even split — closest to the
  existing "parallel reduction" `skSplit` path, `StreamK.py:2536-2591`,
  `ContractionSolution.cpp:933-942`, where `skSplit = grid/tiles` and
  `SKItersPerWG = itersPerTile/skSplit`). The `skSplit` model already produces a
  **fixed, contiguous** peer group of size `skSplit` per tile — setting
  `skSplit == C` makes each tile's peers exactly one cluster.
- The residual imbalance cases (last cluster partially filled, `extraIters`
  giving big/little WGs) are handled by keeping the global-flag path as a runtime
  fallback (§3.4).

---

## 3. Reduction redesign

### 3.1 Which handshake is replaced

The expensive operation is the **cross-CU global-flag spin-wait** in
`storeBranchesCommon` (`:961-965` linear, `:872-877` tree) plus the device-scope
release/acquire fences and the wave-0 flag reset (`:970-979`). When the tile's
peers co-reside in a cluster, all of that is replaced by **one cluster split
barrier**:

- **Non-owner (peer WG)** — after computing its partial:
  1. write partial to its global workspace slot (`computeWorkspaceSrd`,
     `writePartials`) — *unchanged for v1*;
  2. `releaseFence` (relaxed scope, §3.3);
  3. wave-0 `s_barrier_signal -3`;
  4. exit (no flag store).
- **Owner WG** — after computing its own partial portion:
  1. `s_barrier_wait -3` **once** (not per-peer): the cluster split barrier
     guarantees *every* WG in the cluster has signalled, i.e. all `C-1` peers
     have published;
  2. `acquireFence` (relaxed scope, §3.3);
  3. run the existing `fixupStep`/`fixupBatch` loop over the `C-1` peer slots
     (`StreamKIdx+1 … cluster_last`) to `VAddF32`-accumulate — *unchanged
     accumulation math*;
  4. normal alpha/beta store.

Because the barrier is a single all-cluster synchronization, the owner waits once
for the whole cluster instead of `C-1` per-peer flag polls, and **no global flag
store/reset/spin is executed at all** on the fast path.

### 3.2 Where partials live — global WS for v1

- **v1:** keep partials in the **global workspace** (reuse `computeWorkspaceSrd`
  slot layout and `fixupStep` verbatim). Only the *synchronization* changes
  (barrier instead of flag). This minimizes the diff and reuses the audited
  accumulation path.
- **v2 (future):** stage cluster-local partials in **LDS** (cluster WGs co-reside
  on one shader engine and can share via LDS/TDM multicast), eliminating the WS
  round-trip. Deferred: it changes the `fixupStep` addressing and interacts with
  epilogue LDS usage/bias LDS barriers.

### 3.3 Fences — can device scope relax to cluster scope?

- The device-scope `global_wb`/`global_inv` (`StreamK.py:206-249`) exist because
  producer and consumer may sit on **different CUs / L2 partitions**. Intra-cluster
  peers co-schedule on **one shader engine**, so a narrower coherence scope is
  correct for the fast path *iff* the arch exposes it.
- **v1 (safe):** keep `SCOPE_DEV` `global_wb`/`global_inv` around the WS
  partials even on the fast path. This is strictly correct (superset ordering)
  and lets us validate the barrier mechanics independently of fence tuning. The
  win is removing the *spin-wait*, which dominates.
- **Future (opt):** if a shader-engine / cluster cache scope is available
  (extend the `CacheScope` enum + a new arch cap, mirroring `HasInvWbDevFences`),
  add a `StreamKMemoryOrderingClusterScopeFences` subclass selected only on the
  intra-cluster fast path. Gate behind a cap so non-gfx1250 and the fallback
  path are untouched. This depends on HW confirmation that a cluster/SE-scoped
  `global_wb/global_inv` is semantically sufficient given the cluster
  co-residency guarantee, so the MVP keeps `SCOPE_DEV`.

### 3.4 Owner selection + correct fallback

- **Owner within a cluster:** unchanged — the WG with `StreamKLocalStart == 0`.
  Under one-tile-per-cluster alignment this is `wg_in_cluster == 0`
  (`cluster_base`), which also naturally elects the flag-reset/store wave.
- **Fallback (peers span >1 cluster, DP tiles, atomic, tree-straddle):** keep the
  existing global-flag reduction **compiled in** and select it at runtime. The
  guard is computable in the epilogue:

  ```
  intra_cluster = (owner_idx == cluster_base) && (last_peer_idx <= cluster_last)
  ```

  where `last_peer_idx` is derived from the existing fixup bounds
  (`sFixupEnd`/`sCtaIdx` logic, `:955`, `:1020-1032`). If `intra_cluster`, take
  the cluster-barrier fast path; else fall through to the existing flag path
  verbatim. Under the MVP alignment (§2.3) `intra_cluster` is true for all SK
  tiles except the last partial cluster, so the fallback rarely runs but
  guarantees correctness.
- **DP tiles** (`StreamKLocalStart==0 && finished`) never enter the reduction and
  branch to the regular store (`:778`, `:932-933`), so they never *wait* on a
  cluster barrier — but they must still **signal** if they share a cluster with
  reducers (see §6 deadlock). MVP alignment avoids mixing DP and SK WGs in a
  cluster, so this does not arise in v1.

---

## 4. Where the cluster barrier is emitted

**Decision: emit the cluster barrier inline in `StreamK.py`** (new small helpers
`clusterReduceSignal(writer, kernel)` / `clusterReduceWait(writer, kernel)`)
using the rocisa `SBarrier(separate=True, wait=…, clusterBarrier=True)` emitter
directly, reusing the **wave-0 election pattern** from
`Subtile/ClusterBarrier.py:subtileClusterBarrierSignal` (copy the 3-instruction
`s_cmp_eq_u32 WaveIdx,0 / s_cbranch_scc0 skip / s_barrier_signal -3 / skip:`
shape). Assert `asmCaps["HasClusterBarrier"]` exactly like
`insertClusterBarrier` (`ClusterBarrier.py:74-75`).

**Rationale:**

- **The stinkytofu `InsertClusterBarrierPass` will not fire here.** Its anchors
  (`tensor_load_to_lds`, `label_GSU_1`, `label_openLoopL`, `Tail Loop`) are all
  mainloop/prologue/tail markers; the StreamK reduction is in the epilogue
  `storeBranchesCommon`. Adding a new anchor for the epilogue to that C++ pass
  would duplicate StreamK's tile/peer bookkeeping in a place that has no access
  to `StreamKLocalStart`/`sCtaIdx`. Rejected.
- **`Subtile/ClusterBarrier.py:insertClusterBarrier` is also mainloop-shaped**
  (it splices around the *mainloop* workgroup barrier and hides the election
  branch behind a WMMA). The StreamK reduction has no WMMA to hide behind and a
  very different control-flow (spin loop, wave-0 flag reset). We *reuse the
  primitive* (`SBarrier(...,clusterBarrier=True)` + wave-0 election) but not the
  splice. This keeps the barrier co-located with the `StreamKLocalStart==0` owner
  logic and the `intra_cluster` runtime guard, where the needed SGPRs are live.

Concretely the emit sites in `storeBranchesCommon`:
- Replace the per-peer flag poll block (linear `:960-979`; tree `:868-889`) with:
  `if intra_cluster: {owner: clusterReduceWait + acquireFence; skip flag reset}`.
- Add non-owner signal: at the end of `writePartials`
  (`:1042`/`:2462`/`:2876`), after the partial store + `releaseFence`, emit
  `clusterReduceSignal` on the fast path.

---

## 5. Enablement / plumbing

### 5.1 Solution parameter

Add an explicit boolean solution parameter **`StreamKClusterReduction`**
(default `0`) rather than overloading `ClusterDim` alone, because clustering is
already overloaded to mean "Multicast on" (§1.1). Enabling it requires:

- `StreamK == 3` (see §2.3; SK4/SK5 are out of MVP scope — dynamic/atomic peer
  sets cannot be statically clustered).
- `ClusterDim == [C, 1]` with `C` a power of two, `2 <= C <= 16`.
- gfx1250 (`asmCaps["HasClusterBarrier"]`) and `archCaps["HasNewBarrier"]`.
- `TDMInst != 0` (so `WaveIdx` is allocated — the wave-0 election needs it; same
  condition that gates `ClusterBarrier` at `Solution.py:980`).
- `not StreamKAtomic` and `not StreamKForceDPOnly` for the fast path (these skip
  the reduction entirely — `storeBranchesCommon:748-749`).

### 5.2 Decouple Multicast from ClusterDim

`Solution.py:977-978` forces `Multicast=True` for any `ClusterDim != [1,1]`.
For a **barrier-only** StreamK cluster we do *not* want cooperative TDM loads in
the MVP. Change: when `StreamKClusterReduction` is on (and Multicast
is not independently requested), keep `Multicast=False` while still setting
`ClusterBarrier`-style capability for the epilogue emit. `Multicast` is now an
independent tri-state opt-in and cluster-cooperative loads (StreamK DP
B-multicast) have shipped as a sibling feature — see
`cluster-load-component-and-streamk-multicast.md`.

### 5.3 Validation rules (SolutionStructs)

- Reject `StreamKClusterReduction` unless the §5.1 predicate holds
  (mirror the reject-with-reason pattern used throughout `Solution.py`).
- Reject `ClusterDim[1] != 1` when `StreamKClusterReduction` (1-D grid, §2.2).
- Interaction with existing gfx1250 StreamK constraints: gfx1250 StreamK today
  requires MX data (`TDMInst=3`, MX TDM loads); `isStreamKConstantsToVgprEnabled`
  is SK3-only; `StreamKXCCMapping=3` overflows SGPRs (use `0`). The new param
  inherits all of these (SK3 + MX + XCC=0).

### 5.4 Host plumbing

- `getSKGridImpl` rounding to multiple of `C` (§2.2) — behind the same
  size-mapping flag (thread `sizeMapping.streamKClusterReduction` +
  `sizeMapping.clusterDim` through, both already partially present:
  `clusterDim` at `ContractionSolution.cpp:1765,1776`).
- Kernel-arg accounting (`skGrid`, `SKItersPerWG`, `skTiles`) uses the rounded
  grid and the fixed even split (`skSplit == C`, reuse the parallel-reduction
  accounting at `:933-942`).

---

## 6. Risks / gotchas and de-risking

| Risk | Why | De-risk |
|---|---|---|
| **Deadlock if a cluster member never signals.** | `s_barrier_wait -3` blocks until *all* cluster WGs have `s_barrier_signal -3`'d. Any cluster WG that early-exits (e.g. `preLoop` `KernelEnd` branch `StreamK.py:2522`; a DP-only WG; a peer that took the "started & finished tile" store path `:778/:933`) never signals ⇒ owner hangs. | **MVP alignment (§2.3):** a cluster == one SK tile's peers, all of which run the reduction; DP WGs are in DP-only clusters that never wait. **Invariant to enforce in codegen:** *every* WG on the cluster fast path must signal on *every* exit path before leaving (including the "finished my slice" path). Add the signal at a single choke point in the epilogue guarded by `intra_cluster`, not scattered. Keep the global-flag fallback for any tile whose membership is not provably complete. |
| **Cluster size vs `fixup_peers` mismatch (big/little imbalance).** | SK3 `extraIters` gives the first `extraIters` WGs one extra iteration, so peers-per-tile can vary by ±1 and straddle a `C` boundary. | Use the **fixed even split** (`skSplit==C`, no extraIters) for the cluster mode; route the leftover/partial last cluster through the fallback via the `intra_cluster` runtime guard (§3.4). |
| **Peers span >1 cluster (general SK).** | General SK tiling does not align tiles to clusters. | Runtime `intra_cluster` guard selects the fast path only when provably intra-cluster; otherwise the existing global-flag path runs unchanged. |
| **SGPR pressure.** | gfx1250 SK SGPRs already tight; XCC=3 overflows. | Derive `wg_in_cluster`/`cluster_base` by cheap `AND`/`AND-NOT` on the existing `StreamKIdx` (C power of two) — no division, no new persistent SGPRs. Reuse `allocTmpSgpr` scopes. Forbid `StreamKXCCMapping=3` with the new param. |
| **`ttmp9` workaround corrupts index under clustering.** | `StreamK.py:2495-2498` reads raw `wg_x`. | §2.2 fix: use the cluster-remapped `WorkGroup0`. Covered by a codegen snapshot assert that `StreamKIdx` derivation contains the `cluster_x*C + wg_x` form. |
| **Fence relaxation incorrect.** | Cluster-scope `global_wb/global_inv` may not be coherent. | The MVP keeps `SCOPE_DEV` (correct superset). Relaxation is a gated future opt pending HW confirmation. |
| **Multicast side effects.** | `ClusterDim!=[1,1]` forces `Multicast=True`, changing load codegen and adding SGPR mask compute (`KernelWriterAssembly.py:2646-2655`). | Decouple (§5.2): barrier-only v1 keeps `Multicast=False`. |

---

## 7. Test plan (behavior → test) — as shipped

Mirrors existing patterns: CPU asm-string unit tests, syrupy snapshot
characterization, and a GPU roundtrip (run under the arch's simulator/hardware).

| Behavior | Test type | Where |
|---|---|---|
| `clusterReduceSignal/Wait` emit exactly one wave-0 election branch + `-3` barrier ids; `HasClusterBarrier` asserted; reduction gate matrix (SK3/linear/non-atomic/non-DP only) | CPU asm-string unit | `Tensile/Tests/unit/test_streamk_cluster_reduction.py` |
| Cluster config emits real gfx1250 assembly (`err==0`); fast-path handshake present (`s_barrier_signal -3` peer arrive, `s_barrier_wait -3` owner wait); global-flag reduction retained as fallback; order-invariant golden | snapshot char | `Tensile/Tests/unit/characterization/_codegen/test_streamk_cluster_reduction_gfx1250_char.py` (designed config `_designed/gfx1250/streamk_cluster_reduction.yaml`, golden `__snapshots__/test_streamk_cluster_reduction_gfx1250_char.ambr`) |
| Store/release/signal/wait/acquire/accumulate sequence runs on gfx1250, reduces correctly, and does not deadlock, for `C` in {2,4} | GPU roundtrip | `Tensile/Tests/unit/test_streamk_cluster_reduction_gpu.py` (`@requires_gpu_gfx1250`; watchdog on hang) |
| Real multi-WG cluster StreamK GEMM end-to-end | C++ client | `Tensile/Tests/common/streamk/gfx1250/core/sk_mxf8gemm_cluster_reduction.yaml` (+ mxf4 sibling) |

The cooperative-load / multicast siblings of these tests
(`test_streamk_cluster_coop_load_gfx1250_char.py`,
`test_streamk_cluster_multicast_gfx1250_char.py`, `test_streamk_multicast.py`,
`test_cluster_load_component.py`) cover the companion feature documented in
`cluster-load-component-and-streamk-multicast.md`.

The gfx1250 GPU marker lives in `Tensile/Tests/unit/gpu_test_helpers.py`
(`HAS_GFX1250` + `requires_gpu_gfx1250`), independent of the existing gfx950
`requires_gpu`; the target is driven via the `TENSILE_GPU_TARGET=gfx1250`
override.

---

## 8. Follow-up work (out of MVP scope)

- **Fence scope:** a cluster/shader-engine-scoped `global_wb`/`global_inv`
  (§3.3) could replace `SCOPE_DEV` on the fast path once HW confirms it is
  semantically sufficient given cluster co-residency, and once a `CacheScope` /
  arch cap exposes it.
- **LDS partials (§3.2):** eliminate the global-WS round-trip by staging
  cluster-local partials in LDS/TDM-multicast; this changes `fixupStep`
  addressing and the epilogue LDS budget.
- **Broader partitions:** relax the fixed even split / one-tile-per-cluster
  constraint (§2.3) toward general SK tiling with the runtime straddle guard, and
  extend beyond SK3 where peer sets can be statically clustered.

---

## 9. Summary of decisions

- **Approach:** co-locate a StreamK tile's `fixup_peers` into one 1-D cluster
  (`ClusterDim=[C,1]`), exploiting the HW remap `WorkGroup0 = cluster_x*C + wg_x`
  which already clusters consecutive `StreamKIdx`. Replace the global-flag
  spin-wait with a single cluster split barrier; keep the global-flag path as a
  runtime-selected fallback.
- **Variant:** SK3 two-tile (MVP), fixed even split; SK4/SK5 out of scope
  (dynamic/atomic peer sets can't be statically clustered). Barrier-only here;
  cooperative-load multicast shipped separately (see
  `cluster-load-component-and-streamk-multicast.md`).
- **Indexing:** `ClusterDim=[C,1]`, fix the `ttmp9` workaround to use the
  cluster-remapped `WorkGroup0`, `cluster_base = StreamKIdx & ~(C-1)`; host
  rounds `skGrid` to a multiple of `C`.
- **Reduction:** owner `s_barrier_wait -3` once + `acquireFence` + existing
  `fixupStep`; peers write WS partial + `releaseFence` + wave-0 `s_barrier_signal -3`;
  partials stay in global WS for v1; fences stay `SCOPE_DEV` for v1.
- **Barrier emission:** inline in `StreamK.py` epilogue via rocisa
  `SBarrier(...,clusterBarrier=True)` + wave-0 election (reuse the
  `Subtile/ClusterBarrier.py` primitive shape); **not** the stinkytofu pass /
  subtile splice (their anchors don't reach the epilogue).
- **Enablement:** new `StreamKClusterReduction` solution param requiring SK3 +
  `[C,1]` + gfx1250 `HasClusterBarrier` + `TDMInst!=0`; decouple Multicast.
- **Top risks:** cluster-barrier deadlock if any member fails to signal (⇒
  one-tile-per-cluster alignment + all-paths-signal invariant + fallback), SGPR
  pressure (⇒ cheap bitwise cluster coords), big/little imbalance (⇒ fixed even
  split + runtime guard).
