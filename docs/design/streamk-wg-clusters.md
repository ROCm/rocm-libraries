<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Design: WG Clusters for StreamK GEMM (gfx1250)

**Target arch:** gfx1250 (ISA `(12,5,0)`, wave32).

The StreamK partial-tile reduction co-locates a tile's fixup peers in a single
gfx1250 workgroup cluster and replaces the cross-CU global-flag spin-wait with an
intra-cluster split barrier. The global-flag reduction is retained as a runtime
fallback for the general case (peers span more than one cluster). This applies to
StreamK variant 3 (two-tile, DP-first); SK4/SK5 dynamic/atomic peer sets cannot
be statically clustered.

---

## 1. Background

### 1.1 Cluster hardware

- **Split barrier objects** are addressed by negative id: `-1` = workgroup,
  `-3` = cluster. The rocisa `SBarrier(separate, wait, clusterBarrier, comment)`
  emitter lowers to id `-3` when `clusterBarrier=True`. One wave per WG
  signals/arrives; any wave in the cluster may wait.
- **Constraints:** `maxWGsInCluster = 16`; `validClusterDimensions` is all
  `[i,j]` with `i*j <= 16` (`Common/ValidParameters.py`). `ClusterDim` components
  must be powers of two.
- **Host launch is cluster-aware:** `HipSolutionAdapter.cpp` uses
  `hipDrvLaunchKernelEx` + `hipLaunchAttributeClusterDimension`, gated on
  `HIP_HAS_CLUSTER_LAUNCH`, with `kernel.clusterDim` from
  `ContractionSolution` / `Contractions.py`. The grid dimension must be a
  multiple of the cluster size.
- **Kernel-side WG-id remap under clustering:** `WorkGroup0 = cluster_x*nwg_x +
  wg_x`, so the WGs of one cluster occupy a *contiguous* `WorkGroup0` range
  `[cluster_x*C, cluster_x*C + C)`. WGM/XCC remap is bypassed under clustering.

### 1.2 StreamK internals

- Variant 3 `StreamKTwoTileDPFirst` is the clustered path. `StreamKIdx` is derived
  from the cluster-remapped `WorkGroup0` (the raw-`ttmp9` workaround is skipped
  under `enableCluster`, so `StreamKIdx` is the global linear index, not the
  within-cluster `wg_x`).
- `StreamKIter = StreamKIdx * SKItersPerWG (+ extras)`, so consecutive
  `StreamKIdx` map to consecutive global-iteration windows.
- `skTileIndex` magic-divides the global iter into a tile index and derives
  `StreamKLocalStart`/`StreamKLocalEnd`; `skIndexToWG` maps tile ->
  `WorkGroup0/1/2`. The **owner** of a tile is the WG with
  `StreamKLocalStart == 0` (it runs the reduction + final store).
- Non-owners write their partial to the global workspace slot
  (`AddressWS + StreamKIdx*(MT0*MT1*bpeCinternal)`); the owner accumulates peer
  partials with `fixupStep` (`VAddF32`).
- **Memory ordering:** gfx1250 uses `StreamKMemoryOrderingDevScopeFences` (cap
  `HasInvWbDevFences`); the flag is read via VMEM with `s_wait_xcnt 0` before
  volatile VMEM. `AddressFlags == 0` is the sentinel for the parallel /
  post-kernel reduction.

---

## 2. Indexing and grid

### 2.1 Reduction shape and geometry

Pure reduction is expressed as `ClusterDim = [1, C]` (Cs = ClusterDim[0] = 1,
Ck = ClusterDim[1] = C the K-split peer count), a genuine 2-D cluster whose launch
grid is `[skGrid/C, C, 1]` (so `gridDimY % C == 0` holds). The Y rank is folded
into the linear StreamK index at preLoop:

```
StreamKIdx = WorkGroup0*Ck + WorkGroup1   (k = WorkGroup1 fastest)
```

This keeps `StreamKIdx` a dense unique index; the 1-D `Ck==1` path emits no fold.
C must be a power of two in `2..16`.

Reduction (Ck) is **derived** from the cluster shape, not a user parameter: there
is no `StreamKClusterReduction` / `StreamKClusterKSplit` opt-in. The internal
derived key falls out of `ClusterDim[1] > 1`.

### 2.2 Cluster ownership property

Because the HW remap already clusters consecutive `StreamKIdx`, cluster `c` owns a
contiguous `StreamKIdx` range, which is also a contiguous block of global
iteration windows. A tile whose peer set is a contiguous run is fully
intra-cluster iff its first and last peer indices fall in the same cluster block.
No reshuffling is needed.

### 2.3 Grid sizing

`getSKGridImpl` rounds `skGrid` to a multiple of `C` when the cluster path is
active, and that rounded value flows into `rv.numWorkGroups` and the
`skGrid`/`SKItersPerWG`/`skTiles` kernel args. The workspace-overflow guards
(tree-fixup `< 2^24 / 2^16`) apply to the rounded grid. The reduction uses a fixed
even split (`skSplit == C`), so a tile's peers are exactly one cluster.

---

## 3. Reduction handshake

### 3.1 What is replaced

When a tile's peers co-reside in a cluster, the cross-CU global-flag spin-wait
(plus the per-peer flag reset) is replaced by **one cluster split barrier**:

- **Peer WG**, after computing its partial: write the partial to its global
  workspace slot; `releaseFence`; wave-0 `s_barrier_signal -3`; **and**
  `s_barrier_wait -3`; then exit (no flag store).
- **Owner WG**, after its own partial: `s_barrier_signal -3` + `s_barrier_wait -3`
  once (not per-peer); `acquireFence`; run the existing `fixupStep` over the peer
  slots; normal alpha/beta store.

### 3.2 Symmetric-barrier invariant

The split barrier must be **SYMMETRIC**: every cluster member (owner and every
peer) both arrives (`s_barrier_signal -3`) and waits (`s_barrier_wait -3`) on the
same barrier. An arrive-and-exit peer is not a genuine synchronisation point --
the owner could sum a peer slot before that cross-WGP peer's partial is globally
visible across gfx1250's partitioned L2, dropping ~(C-1)/C of the result for
C >= 4 (a cluster-aligned column stripe confined to the grid tail).

### 3.3 Memory visibility (SCOPE_SYS)

The cluster split barrier orders **execution** across the C co-resident peers, but
not memory. On gfx1250's partitioned L2 a `SCOPE_DEV` invalidate is not guaranteed
to re-fetch a peer partial written back on a different partition. The
cluster-reduction handshake therefore uses paired **SCOPE_SYS** fences: the peer's
partial writeback release is escalated to `SCOPE_SYS` (globally visible past the
partitioned L2), and the owner's acquire is a paired `SCOPE_SYS` fence, so the
owner observes the just-published partials. Non-cluster StreamK keeps the
`SCOPE_DEV` default.

### 3.4 Fallback and deadlock invariant

The per-tile global-flag handshake still exists, but only for a cluster that
straddles the SK grid boundary (not fully populated). The intra-cluster predicate
(`clusterReduceIntraCheck`) is SCC=1 when the cluster hosts a single fully
populated StreamK tile, SCC=0 otherwise; it is a pure function of the cluster's
top StreamK index and the SK grid size (both cluster-uniform), so every peer
computes the identical verdict. Both the owner wait and the peer arrive/wait gate
on that identical uniform predicate, so a cluster is never split between barrier
participants and flag setters (**deadlock invariant**). DP tiles are in DP-only
clusters that never wait on the reduction barrier.

---

## 4. Where the cluster barrier is emitted

The cluster barrier is emitted inline in `StreamK.py` (`clusterReduceSignal` /
`clusterReduceWait`) using the rocisa `SBarrier(..., clusterBarrier=True)` emitter
with a wave-0 election (`s_cmp_eq_u32 WaveIdx,0` / `s_cbranch_scc0` /
`s_barrier_signal -3`), asserting `asmCaps["HasClusterBarrier"]`.

The stinkytofu `InsertClusterBarrierPass` does not fire here: its anchors
(`tensor_load_to_lds`, `label_GSU_1`, `Tail Loop`) are mainloop/prologue/tail
markers, whereas the StreamK reduction is in the epilogue `storeBranchesCommon`.
Emitting inline keeps the barrier co-located with the owner logic and the
intra-cluster runtime guard, where the needed SGPRs are live.

---

## 5. Enablement

- Reduction is derived from `ClusterDim = [1, C]` (Ck > 1); there is no user
  opt-in parameter. Requirements (SK3; `ClusterDim` power-of-two `2..16`; gfx1250
  `HasClusterBarrier`; `TDMInst != 0`; not `StreamKAtomic`; not
  `StreamKForceDPOnly`; `StreamKXCCMapping != 3`) are rejected at build time by
  `_validateStreamKClusterReduction`.
- The runtime `itersPerTile % C == 0` requirement (unknown at build time) is a
  per-problem selection reject via the `ClusterReductionIterCheck` predicate.
- `Multicast` is decoupled from `ClusterDim` (tri-state); the barrier-only
  reduction path keeps `Multicast` off. Cooperative-load multicast ships as a
  sibling feature (see `cluster-load-component-and-streamk-multicast.md`).

---

## 6. Summary

- Co-locate a StreamK tile's fixup peers into one cluster (`ClusterDim=[1,C]`,
  reduction along Ck), exploiting the HW remap that already clusters consecutive
  `StreamKIdx`. Replace the global-flag spin-wait with a single cluster split
  barrier; keep the global-flag path as a runtime-selected fallback for a
  boundary-straddling cluster.
- The barrier is symmetric (every member arrives and waits); peer -> owner
  visibility rides on paired `SCOPE_SYS` release/acquire fences.
- Reduction/Ck is derived from the cluster shape (no user parameter); the host
  rounds `skGrid` to a multiple of `C` with a fixed even split.
- Deadlock invariant: owner and peers gate on the identical uniform intra-cluster
  predicate, so a cluster never mixes barrier participants with flag setters.
