<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Design: `ClusterLoad` Component + StreamK Cooperative Multicast Loads (gfx1250)

**Target arch:** gfx1250 (ISA `(12,5,0)`, wave32, TDM: `MXLoadInst=TDM` → `TDMInst=3`).

This note describes, as shipped, three coordinated pieces:

1. The **`ClusterLoad` component**, which owns the multicast ("cluster load") mask
   machinery (value compute, `MulticastMask*` SGPR declare/undeclare, combined-vs-split
   topology decision, per-load-site descriptor attach), replacing copies that lived in
   `KernelWriter` / `KernelWriterAssembly` / `SubtileGREmit`.
2. **`Multicast` as an explicit tri-state** solution parameter, decoupling multicast from
   `ClusterDim` so barrier-only clustering and cooperative-load clustering compose
   independently.
3. **StreamK cooperative cluster loads** for the data-parallel (DP) region: a `[C,1]`
   cluster of consecutive workgroups multicasts B across M-adjacent tiles.

The barrier-only cluster split-barrier reduction is described separately in
`docs/design/streamk-wg-clusters.md`.

---

## 1. Background

### 1.1 What a "cluster load" is

A cluster load is a TDM `tensor_load_to_lds` whose descriptor `Group1[word0]` has a
multicast-mask bit field OR'd in (`TensorDataMover.setMulticastMask`, a single `SOrB32`).
The hardware broadcasts the loaded tile to every workgroup in the cluster whose bit is set.

### 1.2 Mask value and the three topologies

The mask value is computed from the workgroup's position within the cluster and `ClusterDim`:

- `maskA = OR over idx in range(ClusterDim[1]) of (1 << (idx*ClusterDim[0]))`, shifted left
  by `wg_x`: selects every workgroup sharing the same `wg_x` column (same M block) across all
  `ClusterDim[1]` rows.
- `maskB = (1 << ClusterDim[0]) - 1`, shifted left by `wg_y * ClusterDim[0]`: selects every
  workgroup in the same `wg_y` row (same N block).

Three name topologies (the selection matrix `ClusterLoad` must preserve):

| Topology | Predicate | SGPR name(s) |
|---|---|---|
| Combined single-parity | `tdmA and tdmB and NumWaves>1 and not UseSubtileImpl` | `MulticastMask` (even wave = maskA, odd wave = maskB, chosen by `WaveIdx` parity) |
| Split A/B | otherwise (subtile, single-tensor, or `NumWaves==1`) | `MulticastMaskA`, `MulticastMaskB` |
| Metadata (sparse) | `enableTDMMetadata` | `MulticastMaskMetadata` (follows A for `Sparse==1`, B for `Sparse==2`) |

Declare and undeclare mirror the same predicate.

### 1.3 Data-reuse fact (drives the StreamK design)

Two DP workgroups share **A** iff same WG0 (M) and overlapping K; share **B** iff same WG1
(N) and overlapping K. DP tiles compute the whole tile over full K, so K always overlaps.

- K-split peers of one tile (the `[C,1]` reduction cluster) have the same WG0/WG1 but
  disjoint K → zero reuse; multicasting a reduction cluster multicasts nothing.
- Real reuse is spatial over a common full-K range: adjacent DP tiles. Consecutive-WG
  clustering (`[C,1]`) gives M-adjacent tiles → shared B. (Tiles `nWG0` apart are N-adjacent
  → shared A; not reachable by consecutive `[C,1]` clustering.)

**Conclusion: in the StreamK DP region a `[C,1]` cluster multicasts B (not A).** This maps
onto the mask math with `wg_y=0, C0=C, C1=1`: `maskB = (1<<C)-1` (all C workgroups share B),
`maskA = 1<<wg_x` (self only → A per-workgroup).

### 1.4 StreamK addressing (where multicast pays off)

- `skIndexToWG` maps a tile → `(WorkGroup0, WorkGroup1, WorkGroup2)` with `tileID =
  WG2*(nWG0*nWG1) + WG1*nWG0 + WG0` — **WG0 (M) fastest**, so consecutive tiles are
  M-adjacent (same WG1/N block, consecutive WG0).
- In the first DP round (`graWorkGroup`, SK3 TwoTileDPFirst) consecutive workgroups map to
  consecutive tiles → M-adjacent → share the same B over full K.
- The gfx1250 kernel-side WG remap `WorkGroup0 = cluster_x*nwg_x + wg_x` makes cluster `c`
  own a contiguous `WorkGroup0` range `[c*C, c*C + C)`.

---

## 2. The `ClusterLoad` component

`ClusterLoad` is a capability-selected `Component` (`asmCaps = {"HasTDM": True}`,
`kernel = {"TDMInst": 3}`), found exactly like `TensorDataMoverLoad`: `ClusterLoad.find`
returns the TDM impl on gfx1250 and `None` (fallback → no multicast) elsewhere. It owns mask
value + declare/undeclare + topology decision + descriptor attach; it does **not** own
descriptor-group SGPRs, LDS offsets, or the `tensor_load_to_lds` itself.

| Method | Responsibility |
|---|---|
| `usesCombinedMask(kernel)` | The single combined-vs-split predicate. |
| `maskSgprName(kernel, tc, *, subtile=False, waveSeparated=False)` | Central name resolver (combined `"MulticastMask"` vs split `f"MulticastMask{strip_MXS(tc)}"` vs metadata). |
| `declareSgprs` / `undeclareSgprs` | Allocate / free the `MulticastMask*` SGPRs. |
| `computeMasks(writer, kernel, *, sgprWgX, sgprWgY, sgprNWgX, sTmp)` | Compute the mask value(s); the caller passes the SGPR operands it already holds. |
| `applyToDescriptor(writer, kernel, group1, tc, *, subtile=False)` | Gate + name choice + the `SOrB32`; empty `Module` when `not (kernel["Multicast"] and clusterEnabled(ClusterDim))`. |

**SGPR-operand contract:** `computeMasks` does not allocate SGPRs; it receives the operands
the surrounding code already holds and emits into them, so callers own SGPR lifetime.

---

## 3. `Multicast` tri-state

`Multicast` is an explicit tri-state solution parameter (`ValidParameters.py`):

- `-1` = auto: `ClusterDim != [1,1]` implies multicast, except StreamK cluster paths, which
  drive it through the derived `StreamKMulticast` collapse instead;
- `0` = force off;
- `1` = force on (independent of `ClusterDim`).

Default `-1` reproduces the historic `ClusterDim`-coupled derivation, so configs that omit
`Multicast` are unchanged. `Multicast=1` requires a matching `ClusterLoadTDM` (`TDMInst=3` on
gfx1250 with `HasTDM`) and is rejected otherwise.

---

## 4. StreamK cooperative cluster loads (DP region)

### 4.1 Derivation and validation

`StreamKMulticast` is a derived-only internal state key (no `ValidParameters.py` entry). It
is auto-enabled when `ClusterDim != [1,1]` on SK3; a StreamK cluster always carries a
cooperative role. `_validateStreamKMulticast` hard-rejects a derived config it cannot
satisfy:

- `StreamK == 3` (SK3 DP schedule + `skIndexToWG` assumptions);
- `Multicast != 0` (the mask SGPRs are gated on `Multicast` while the predicate /
  boundary-clear emitters are gated on `StreamKMulticast`; `Multicast=0` would reference
  undeclared masks);
- `StreamKAtomic == 0` (the atomic path skips the workspace/tile DP structure);
- `StreamKXCCMapping == 0` (WGM/XCC remap is bypassed under clustering; XCC=3 overflows the
  SGPR budget);
- `ClusterDim = [C,1]`, `C` a power of two with `C ∈ [2,16]`;
- gfx1250 with `HasTDM`, `TDMInst == 3`, and `HasClusterBarrier`.

The `nWG0 % C` runtime requirement is enforced by the `ClusterDimCheck` predicate at
selection time (not a silent fallback).

### 4.2 Mask derivation and runtime validity

For StreamK the authoritative WG coordinates are the ones produced by `skIndexToWG`. For the
DP `[C,1]` cluster `wg_x = StreamKIdx & (C-1)`, `wg_y = 0`, so `computeMasks` yields the
B-broadcast mask `(1<<C)-1` with A self-only.

For the C workgroups to genuinely share B on every issued load, the C tiles they process
must be M-adjacent (same WG1/N block, consecutive WG0). An init-time predicate
`clusterMulticastValid` (from `NumWorkGroups0` / total tiles / cluster base; cheap AND/compare
with `C` a power of two) gates the mask: `applyToDescriptor` ORs the mask only when the
predicate holds, otherwise the workgroup loads B normally. Incorrect multicast would write
the wrong B tile into a peer's LDS, so the gate is conservative.

### 4.3 DP→SK boundary clear

The static spatial mask is valid only for the DP (full-tile) round. At the DP→SK boundary
`streamKMulticastBoundaryClear` drops the mask to self-only (a single `SAndB32` on
`Group1[word0]`) so the SK partial-tile loads are not multicast; the SK tail reduces via the
existing workspace / global-flag path. Cooperative DP loads therefore work inside a full SK3
kernel, not only under `StreamKForceDPOnly`.

### 4.4 Host plumbing

- `sizeMapping.streamKMulticast` mirrors `streamKClusterReduction`.
- `getSKGridImpl` rounds `skGrid` up to a multiple of `C`; tail workgroups (if
  `totalTiles % C != 0`) are idle-but-present and the init-time predicate disables their
  cluster's multicast.
- The launch `enableCluster` branch already keeps the grid cluster-friendly and sets
  `rv.clusterDim`; no launch change beyond threading `streamKMulticast`.

---

## 5. Interaction notes

- gfx1250 StreamK already uses TDM (`MXLoadInst=TDM` → `TDMInst=3`), so `HasTDM` is the
  natural cap; no new transport is introduced.
- `StreamKAtomic` is incompatible (the atomic path skips the workspace/tile structure the DP
  schedule relies on) and is rejected.
- Consecutive `[C,1]` clustering shares B, not A; A stays per-workgroup. A-multicast needs an
  N-strided / 2-D cluster.
