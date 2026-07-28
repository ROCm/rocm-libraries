<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Design: `ClusterLoad` Component + StreamK Cooperative Multicast Loads (gfx1250)

**Target arch:** gfx1250 (ISA `(12,5,0)`, wave32, TDM: `MXLoadInst=TDM` -> `TDMInst=3`).

This note describes the shipped design of two coordinated pieces:

1. the reusable `ClusterLoad` component that owns the multicast ("cluster load")
   mask machinery, and
2. StreamK cooperative cluster loads, which multicast the shared B operand across
   a 1-D `[C,1]` StreamK data-parallel (DP) cluster.

The barrier-only StreamK cluster reduction is described in
`docs/design/streamk-wg-clusters.md`.

---

## 1. Background

### 1.1 What a "cluster load" is

A cluster load is a normal TDM `tensor_load_to_lds` whose descriptor
`Group1[word0]` has a multicast-mask bit field OR'd in. The HW then broadcasts
the loaded tile to every workgroup in the cluster whose bit is set. Attachment is
a single `SOrB32` (`Components/TensorDataMover.py`, `setMulticastMask`).

### 1.2 Mask value computation and the three topologies

The mask *value* is a function of the WG's position within the cluster and
`ClusterDim`:

- `maskA = OR over idx in range(ClusterDim[1]) of (1 << (idx*ClusterDim[0]))`,
  shifted left by `wg_x`. Bit `wg_y*ClusterDim[0] + wg_x` is the cluster-linear
  index of the WG, so `maskA` selects every WG sharing the same `wg_x` column
  (same M block) across all `ClusterDim[1]` rows.
- `maskB = (1 << ClusterDim[0]) - 1`, shifted left by `wg_y * ClusterDim[0]`.
  Selects every WG in the same `wg_y` row (same N block).

Three name topologies:

| Topology | Predicate | SGPR name(s) |
|---|---|---|
| Combined single-parity | `tdmA and tdmB and NumWaves>1 and not UseSubtileImpl` | `MulticastMask` (even wave = maskA, odd wave = maskB, chosen by `WaveIdx` parity) |
| Split A/B | otherwise (subtile, single-tensor, or `NumWaves==1`) | `MulticastMaskA`, `MulticastMaskB` |
| Metadata (sparse) | `enableTDMMetadata` | `MulticastMaskMetadata` (follows A for `Sparse==1`, B for `Sparse==2`) |

Declare and undeclare mirror the same predicate.

### 1.3 StreamK addressing (where multicast pays off)

- `skIndexToWG` maps a tile to `(WorkGroup0, WorkGroup1, WorkGroup2)` with
  `tileID = WG2*(nWG0*nWG1) + WG1*nWG0 + WG0` -- **WG0 (M) is fastest**, so
  consecutive tiles are M-adjacent (same WG1/N block, consecutive WG0).
- SK3 `graWorkGroup` (TwoTileDPFirst): in the first DP round consecutive WGs map
  to consecutive tiles -> **M-adjacent -> share the same B (N-block) over full K**.
- gfx1250 kernel-side WG remap `WorkGroup0 = cluster_x*nwg_x + wg_x`, so cluster
  `c` owns a contiguous `WorkGroup0` range `[c*C, c*C + C)`.

### 1.4 Data-reuse fact (drives the StreamK design)

Two DP WGs share **A** iff same WG0 (M) and overlapping K; share **B** iff same
WG1 (N) and overlapping K. DP tiles compute the whole tile over full K, so K
always overlaps.

- K-split peers of one tile (the reduction cluster `[1,C]`): same tile but
  **disjoint K** -> **zero reuse**. Multicasting a reduction cluster multicasts
  nothing.
- Real reuse is **spatial** over a common full-K range: adjacent DP tiles.
  Consecutive-WG clustering (`[C,1]`) gives M-adjacent tiles -> **shared B**.

**Conclusion: in the StreamK DP region a `[C,1]` cluster multicasts B (not A).**
This maps onto the mask math with `wg_y=0, C0=C, C1=1`: `maskB = (1<<C)-1` (all C
WGs share B), `maskA = 1<<wg_x` (self only -> A per-WG).

---

## 2. The `ClusterLoad` component

`ClusterLoad` is a capability-selected tensilelite `Component`
(`asmCaps={"HasTDM":True}`, `kernel={"TDMInst":3}`, like `TensorDataMoverLoad`),
so `ClusterLoad.find(writer)` returns the TDM implementation on gfx1250 and
`None` (fallback -> no multicast) elsewhere.

It **owns** the multicast mask machinery and nothing else (it does not own
descriptor-group SGPRs, LDS offsets, or the `tensor_load_to_lds` itself):

| Method | Responsibility |
|---|---|
| `usesCombinedMask(kernel)` | Single source of truth for the combined-vs-split decision (`tdmA and tdmB and NumWaves>1 and not UseSubtileImpl`; StreamK multicast always uses split A/B). |
| `maskSgprName(kernel, tc, ...)` | Central name resolver: combined `"MulticastMask"` when wave-separated and not subtile; else `f"MulticastMask{strip_MXS(tc)}"`. |
| `declareSgprs` / `undeclareSgprs` | Allocate / free the `MulticastMask*` SGPRs. |
| `computeMasks(writer, kernel, *, sgprWgX, sgprWgY, sgprNWgX, sTmp)` | Compute the mask value(s). The caller passes the SGPR operands it already holds (`sTmp+4` slot is scratch) rather than re-allocating. |
| `applyToDescriptor(...)` | Fold the gate + name choice + the `SOrB32`; returns an empty `Module` when multicast is inactive. |

The SGPR-operand contract on `computeMasks`/`applyToDescriptor` (caller supplies
the operands) is deliberate: the component never re-allocates the mask SGPRs.

---

## 3. Decoupling `Multicast` from `ClusterDim`

`Multicast` is an explicit tri-state solution parameter:

```python
# -1 = auto (ClusterDim!=[1,1] implies Multicast, except StreamK cluster paths)
#  0 = force multicast off
#  1 = force multicast on (independent of ClusterDim coupling)
"Multicast": [-1, 0, 1],
```

Default `-1` reproduces the ClusterDim-coupled derivation, so YAML that omits
`Multicast` is unchanged. `Multicast=1`/`0` are explicit overrides, and the
derived `StreamKMulticast` path (§4) sets `Multicast=True` for itself without
relying on the coupling.

---

## 4. StreamK cooperative cluster loads (DP region)

### 4.1 Shape and derivation

The StreamK cluster is fully described by `ClusterDim = [Cs, Ck]`; there are no
user factoring/reduction knobs. Cs = ClusterDim[0] is the spatial B-multicast
axis, Ck = ClusterDim[1] is the K-split reduction axis. On SK3 a non-`[1,1]`
`ClusterDim` auto-enables the cooperative cluster path; both derived booleans fall
out of the shape:

- `StreamKMulticast` iff Cs > 1 (pure multicast `[C,1]`, 1-D launch),
- the cluster reduction iff Ck > 1 (pure reduction `[1,C]`, 2-D launch).

`StreamKMulticast` is a **derived-only** internal state key (like `ClusterBarrier`);
it has no `ValidParameters.py` entry. `_validateStreamKMulticast` hard-rejects a
derived config it cannot satisfy (SK3, `ClusterDim=[C,1]` with C a power of two in
`2..16`, gfx1250 `HasTDM`/`TDMInst=3`, `StreamKXCCMapping != 3`, not
`StreamKAtomic`).

### 4.2 Mask derivation and validity

For the DP `[C,1]` cluster the authoritative coordinates are the ones produced by
`skIndexToWG`; for this shape they coincide with the raw cluster position
(`wg_x = StreamKIdx & (C-1)`, `wg_y = 0`), yielding the B-broadcast mask
`(1<<C)-1`.

The static mask is baked into the descriptor once and applied to every mainloop
load, so the sharing relationship it encodes must hold for every load the WG
issues. The runtime `ClusterDimCheck` predicate (`Contractions.py`) enforces the
`nWG0 % C == 0` M-adjacency requirement at selection time (not a silent
fallback). Incorrect multicast would write the wrong B tile into a peer's LDS
(silent wrong results, not just a perf loss), which is why validity is checked
rather than assumed.

### 4.3 DP -> SK boundary clear

In a full SK3 kernel the WG transitions from DP tiles (where the spatial mask is
valid) into SK partial tiles (where it is not). The multicast mask bit is cleared
at the DP -> SK boundary so the SK-region loads do not carry a stale spatial mask.
This boundary clear is emitted as part of the shipped path.

### 4.4 Host plumbing

- `sizeMapping.streamKMulticast` mirrors the reduction size-mapping.
- `getSKGridImpl` rounds `skGrid` up to a multiple of `C`; the tail WGs of a
  not-fully-populated cluster are disabled by the validity predicate.
- The `enableCluster` launch branch already keeps the grid cluster-friendly and
  sets `rv.clusterDim`; no launch change is required.

---

## 5. Enablement summary

- **`ClusterLoad`**: capability-selected component owning mask value
  (`computeMasks`), declare/undeclare, topology decision
  (`usesCombinedMask`/`maskSgprName`), and descriptor attach
  (`applyToDescriptor`). It does not own descriptor groups or LDS offsets.
- **`Multicast`**: tri-state (`-1` auto = legacy identity, `0` off, `1` on); the
  `ClusterDim!=[1,1]` coupling no longer unconditionally forces multicast.
- **StreamK cooperative loads**: DP-region B-multicast on a `[C,1]`
  consecutive-WG cluster (M-adjacent tiles share the N-block). `StreamKMulticast`
  is derived from the cluster shape, gated by the `ClusterDimCheck` validity
  predicate, with the DP -> SK boundary mask clear present.
