<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Design: `ClusterLoad` Component + StreamK Cooperative Multicast Loads (gfx1250)

**Target arch:** gfx1250 (ISA `(12,5,0)`, wave32, TDM: `MXLoadInst=TDM` → `TDMInst=3`).
**Scope:** Two coordinated pieces of work.

1. **`ClusterLoad` component + behavior-preserving refactor.** Extract the multicast
   ("cluster load") mask machinery — value computation, SGPR declare/undeclare, and the
   three descriptor apply sites (dense, wave-separated, subtile) — into a single reusable
   `ClusterLoad` tensilelite `Component`, and refactor the existing subtile Multicast path
   onto it. This is a **pure refactor**: zero change to emitted assembly for existing
   configs, gated by codegen snapshots.
2. **Decouple `Multicast` from `ClusterDim`** into an independent opt-in, so barrier-only
   clustering (shipped `StreamKClusterReduction`) and cooperative-load clustering compose
   independently instead of `ClusterDim != [1,1]` unconditionally forcing `Multicast=True`.
3. **StreamK cooperative cluster loads.** Build TDM multicast on top of `ClusterLoad` for
   the StreamK **data-parallel (DP) region**, resolving the tension between the spatial
   `[C0,C1]` cluster cooperative loads want and the `[C,1]` same-tile K-peer cluster the
   shipped reduction wants.

The shipped barrier-only reduction (`docs/design/streamk-wg-clusters.md`,
`StreamKClusterReduction`) stays intact and orthogonal.

Source-line anchors reference the code as it exists now and were verified against the tree.

---

## 1. Background (verified anchors)

### 1.1 What a "cluster load" is

A cluster load is a normal TDM `tensor_load_to_lds` whose descriptor `Group1[word0]` has a
multicast-mask bit field OR'd in. The HW then broadcasts (multicasts) the loaded tile to
every workgroup in the cluster whose bit is set. Attachment is a single `SOrB32`:

```511:514:projects/hipblaslt/tensilelite/Tensile/Components/TensorDataMover.py
    def setMulticastMask(self, group1: int | str, mask: str, writer: "KernelWriterAssembly") -> Module:
        mod = Module()
        mod.add(SOrB32(sgpr(f"{group1}"), sgpr(f"{group1}"), sgpr(f"{mask}")))
        return mod
```

### 1.2 Mask value computation and the three topologies

The mask *value* is computed once in `defineAndResources`
(`KernelWriterAssembly.py:2646-2694`), from the WG's position within the cluster and
`ClusterDim`:

- `maskA = OR over idx in range(ClusterDim[1]) of (1 << (idx*ClusterDim[0]))`, then shifted
  left by `wg_x`. Bit `wg_y*ClusterDim[0] + wg_x` = the cluster-linear index of the WG.
  So `maskA` selects every WG sharing the same `wg_x` column (same M block) across all
  `ClusterDim[1]` rows.
- `maskB = (1 << ClusterDim[0]) - 1`, then shifted left by `wg_y * ClusterDim[0]`. Selects
  every WG in the same `wg_y` row (same N block).

Three name topologies (this exact selection matrix must be preserved by the refactor):

| Topology | Predicate | SGPR name(s) |
|---|---|---|
| Combined single-parity | `tdmA and tdmB and NumWaves>1 and not UseSubtileImpl` | `MulticastMask` (even wave = maskA, odd wave = maskB, chosen by `WaveIdx` parity) |
| Split A/B | otherwise (subtile, single-tensor, or `NumWaves==1`) | `MulticastMaskA`, `MulticastMaskB` |
| Metadata (sparse) | `enableTDMMetadata` | `MulticastMaskMetadata` (follows A for `Sparse==1`, B for `Sparse==2`) |

Declare (`KernelWriter.py:9163-9176`) and undeclare (`KernelWriter.py:2838-2848`) mirror the
same predicate.

### 1.3 The three descriptor apply sites (the duplication to unify)

All three OR the mask into `Group1` under the same gate
`kernel["Multicast"] and enableCluster`, where `enableCluster = prod(ClusterDim) != 1`:

- **Dense** `initTDMDescriptor` — `KernelWriterAssembly.py:18901-18902`; local
  `maskSgprName(tc)` returns `MulticastMask{A|B}` (split).
- **Wave-separated** `initTDMDescriptorWaveSeparatedImpl` — `KernelWriterAssembly.py:19059-19060`;
  local `maskSgprName` returns the combined `"MulticastMask"`.
- **Subtile** `initTDMDescriptorSubtile` — `Components/Subtile/SubtileGREmit.py:1108-1113`;
  uses `MulticastMask{tc}` (split).

`initTDMDescriptorSubtile` (`SubtileGREmit.py:1061-1155`) is a near-clone of the writer
`initTDMDescriptor` (`KernelWriterAssembly.py:18843-18994`); the mask attachment is the
piece we lift into `ClusterLoad`. The rest of the descriptor build (LDS offset, tensor
dims/strides, padding) stays where it is — `ClusterLoad` does **not** own descriptor-group
allocation or LDS offsets.

### 1.4 Cooperative-thread partition (shared math)

`numCooperativeWGs = ClusterDim[1] for A / ClusterDim[0] for B`, duplicated in
`Components/GL2Prefetch.py:26` and `:78`. `ClusterLoad` centralizes this as
`cooperativeThreadPartition`.

### 1.5 Component framework

- Auto-registration metaclass: `Component.py:113-124` (`__init__` sets `implementations`
  and `setattr` on each base). `matches()` partial-matches `asmCaps`/`archCaps`/`kernel`
  (`Component.py:132-144`); `find()` requires exactly one match, raises on >1, returns
  `None` on 0 → fallback (`Component.py:167-177`).
- Categories (abstract, no `__call__`) declared near `Component.py:305-313`
  (`TensorDataMover`, `GL2Prefetch`); `from .Components import *` at `:318`.
- `Components/__init__.py:28-55` `__all__` lists each component module.
- `TensorDataMoverLoad` shows the concrete pattern: `asmCaps = {"HasTDM": True}`,
  `kernel = {"TDMInst": 3}`, method-bag returning rocisa `Module`s
  (`TensorDataMover.py:13-15`).
- New-file SPDX header (`# Copyright Advanced Micro Devices, Inc., or its affiliates.` /
  `# SPDX-License-Identifier: MIT`).

### 1.6 `Multicast` / `ClusterDim` coupling (current)

`Solution.py:1046-1064` forces `Multicast=True` for any `ClusterDim != [1,1]`, except it is
already suppressed for `StreamKClusterReduction`. `Multicast` is **not** a user parameter in
`ValidParameters.py` (it is a derived state var). `StreamKClusterReduction` is a real param
(`ValidParameters.py:839`) and is fully wired host-side (see §1.8).

### 1.7 StreamK addressing (where multicast pays off)

- `skTileIndex` (`StreamK.py:439`, math `469-483`) magic-divides `StreamKIter` into a tile
  index and `StreamKLocalStart`/`StreamKLocalEnd` (iter offsets within the tile).
- `skIndexToWG` (`StreamK.py:487-503`) maps tile → `(WorkGroup0, WorkGroup1, WorkGroup2)`
  with the linearization `tileID = WG2*(nWG0*nWG1) + WG1*nWG0 + WG0` — **WG0 (M) is
  fastest**, so consecutive tiles are M-adjacent (same WG1/N block, consecutive WG0).
- The K-direction StreamK offset (`StreamKLocalStart*DepthU*strideL`) is added in
  `computeLoadSrdCommon` (`StreamK.py:548`, `555-562`) and `graAddressesCommon`
  (`StreamK.py:629`, `636-645`); both are `tc`-generic (shared A/B). **The `WorkGroup0*MT0`
  / `WorkGroup1*MT1` tile-base term lives in the base KernelWriter GRA/SRD code, keyed on
  the post-`skIndexToWG` `WorkGroup0/1`.** A/B thus bind to WG0/M and WG1/N respectively;
  the K offset is identical for A and B.
- `graWorkGroup` (SK3 TwoTileDPFirst, `StreamK.py:2846`; DP shift `2921-2925`): a WG `w`
  processes tiles `w, w+skGrid, w+2*skGrid, …`. In the **first DP round** consecutive WGs
  map to consecutive tiles → **M-adjacent → share the same B (N-block) over full K**.
- `StreamKIdx = WorkGroup0` (`StreamK.py:2658-2663`), preceded by the `ttmp9` raw-wgid
  workaround (`StreamK.py:2645-2656`), which is **already skipped under
  `StreamKClusterReduction`** so the cluster-remapped `WorkGroup0` survives.
- DP/SK region split in `graWorkGroup` at `StreamK.py:2907-2932`; the "started &
  finished the whole tile → regular store, no reduction" branch at `StreamK.py:922-933`.
- gfx1250 kernel-side WG remap: `WorkGroup0 = cluster_x*nwg_x + wg_x`
  (`KernelWriterAssembly.py:2628-2632`), so cluster `c` owns a contiguous `WorkGroup0`
  range `[c*C, c*C + C)`.

### 1.8 Data-reuse fact (drives the whole StreamK design)

Two DP WGs share **A** iff same WG0 (M) and overlapping K; share **B** iff same WG1 (N) and
overlapping K. DP tiles compute the whole tile over full K, so K always overlaps.

- K-split peers of one tile (the shipped `[C,1]` reduction cluster): same WG0/WG1 but
  **disjoint K** → **zero reuse**. Multicasting the reduction cluster multicasts nothing.
- Real reuse is **spatial** over a common full-K range: adjacent DP tiles. Consecutive-WG
  clustering (`[C,1]`) gives M-adjacent tiles → **shared B**. (Tiles `nWG0` apart are
  N-adjacent → shared A; not reachable by consecutive `[C,1]` clustering.)

**Conclusion: in the StreamK DP region a `[C,1]` cluster multicasts B (not A).** This maps
exactly onto the existing mask math with `wg_y=0, C0=C, C1=1`: `maskB = (1<<C)-1` (all C WGs
share B), `maskA = 1<<wg_x` (self only → A not shared). A is loaded per-WG.

### 1.9 Host cluster plumbing (already present)

- `sizeMapping.streamKClusterReduction` (`ContractionSolution.hpp:146`) and `clusterDim`
  (`:120`) exist; Python side `Contractions.py:630,725,763`.
- `getSKGridImpl` at `ContractionSolution.cpp:3925-4093`; it already rounds the grid for
  cluster reduction — `skGrid = clusterDim.x * tiles` at `:4087-4090`, plus a second
  round-up in `solve` at `:3214-3218`. Tree-fixup `< 2^24 / 2^16` guards at `:4061-4073`.
- StreamK grid assignment + `enableCluster` branch (keeps y/z unflattened for cluster
  kernels) at `ContractionSolution.cpp:1776-1794`; `rv.clusterDim = sizeMapping.clusterDim`
  at `:1794`.
- SK3 per-WG kernarg accounting: cluster-reduction fixed even split at
  `ContractionSolution.cpp:933-950`, parallel-reduction `skSplit = grid/tiles` at
  `:951-960`, default two-tile at `:961-992`.

---

## 2. The `ClusterLoad` component

### 2.1 Registration

- New category in `Component.py` next to `TensorDataMover`/`GL2Prefetch` (~`305-313`):

```python
class ClusterLoad(Component):
    """
    Cluster (multicast) TDM load: multicast-mask compute + descriptor attach.
    """
```

- New concrete module `Tensile/Components/ClusterLoad.py`:

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from ..Component import ClusterLoad
...
class ClusterLoadTDM(ClusterLoad):
    asmCaps = {"HasTDM": True}
    kernel  = {"TDMInst": 3}
    def __call__(self, writer, kernel):  # abstract-satisfying no-op, mirrors TensorDataMoverLoad
        pass
```

- Add `"ClusterLoad"` to `Components/__init__.py:__all__` (`:28-55`).

Selection is capability-based (`HasTDM` + `TDMInst=3`), identical to how
`TensorDataMoverLoad` is found, so `ClusterLoad.find(writer)` returns the TDM impl on
gfx1250 and `None` (fallback → no multicast) elsewhere.

### 2.2 API (method bag; all returning rocisa `Module` or plain values)

`ClusterLoad` owns **mask value + declare + attach + topology decision + cooperative
partition**. It does **not** own descriptor-group SGPRs, LDS offsets, or the
`tensor_load_to_lds` itself.

| Method | Signature | Responsibility |
|---|---|---|
| `usesCombinedMask` | `(kernel) -> bool` | The single predicate `tdmA and tdmB and NumWaves>1 and not UseSubtileImpl`. Kills the 3 duplicated copies (`KernelWriter.py:9170`, `:2842`, `KernelWriterAssembly.py:2668`). |
| `maskSgprName` | `(kernel, tc, *, subtile=False) -> str` | Central name resolver: combined `"MulticastMask"` when `usesCombinedMask` and not `subtile`; else `f"MulticastMask{strip_MXS(tc)}"`; `"MulticastMaskMetadata"` for metadata. Subsumes the two local `maskSgprName` closures + the subtile literal. |
| `declareSgprs` | `(writer, kernel) -> None` | Moves `KernelWriter.py:9163-9176` (uses `writer.defineSgpr`). |
| `undeclareSgprs` | `(writer, kernel) -> Module` | Moves `KernelWriter.py:2838-2848` (uses `writer.undefineSgpr`). |
| `computeMasks` | `(writer, kernel, *, sgprWgX, sgprWgY, sgprNWgX, sTmp) -> Module` | Moves `KernelWriterAssembly.py:2646-2694` verbatim, including the wave-parity branch and the metadata cases. Takes the exact SGPR operands the caller already holds so emitted asm is byte-identical. |
| `applyToDescriptor` | `(writer, kernel, group1, tc, *, subtile=False) -> Module` | Folds the gate + name choice + the `SOrB32`. Returns an **empty `Module`** when `not (kernel["Multicast"] and enableCluster)` — identical to today's skipped `if`. Internally calls `TensorDataMover.setMulticastMask`. Replaces the 3 apply sites. |
| `cooperativeThreadPartition` | `(kernel, tc) -> int` | `ClusterDim[1] if tc-ends-A else ClusterDim[0]`. Shared with `GL2Prefetch` (`:26`, `:78`). |

Rationale for passing SGPR indices into `computeMasks`: the current code emits into
`sTmp+1..4` allocated by the surrounding `defineAndResources`. To guarantee **zero asm
diff**, the component must not re-allocate; it receives the same operands and emits the same
instructions in the same order. This is a mechanical lift, not a rewrite.

### 2.3 Behavior-preserving refactor wiring

- `KernelWriter.py:9163-9176` → `ClusterLoad.find(self).declareSgprs(self, kernel)`.
- `KernelWriter.py:2838-2848` → `... .undeclareSgprs(self, kernel)`.
- `KernelWriterAssembly.py:2646-2694` → `... .computeMasks(self, kernel, sgprWgX=..., sgprWgY=..., sgprNWgX=..., sTmp=...)` (same operands as today).
- `KernelWriterAssembly.py:18901-18902` (dense) → `... .applyToDescriptor(self, kernel, descSgprName(1), tc)`.
- `KernelWriterAssembly.py:19059-19060` (wave-separated) → same call (component resolves to combined name via `usesCombinedMask`).
- `SubtileGREmit.py:1108-1113` (subtile) → `... .applyToDescriptor(writer, kernel, descSgprName(1), tc, subtile=True)`.

Guard: `ClusterLoad.find` returns `None` when `HasTDM`/`TDMInst` don't match; callers keep a
`comp = ClusterLoad.find(self)` + `if comp:` fallback exactly like `TensorDataMoverLoad.find`
usage, so non-TDM archs are untouched.

**Regression gate (must stay green, zero asm change):**
`test_r4_xccremap_char.py` (drives `ClusterDim=[2,2]` with both a `MIWaveGroup` prod==1 →
split `MulticastMaskA/B` kernel and a prod==4 → combined `MulticastMask` kernel) and
`test_streamk_cluster_coop_load_gfx1250_char.py`. Add a byte-exact golden snapshot of the
emitted mask/apply region for a `[2,2]` combined config and a subtile split config before
the refactor; diff to zero after.

---

## 3. Decoupling `Multicast` from `ClusterDim`

### 3.1 Mechanism

Add an explicit tri-state solution parameter (`ValidParameters.py`, near `ClusterDim` /
`StreamKClusterReduction`):

```python
# -1 = auto (legacy: ClusterDim!=[1,1] implies Multicast, except StreamK cluster paths)
#  0 = force multicast off
#  1 = force multicast on (independent of ClusterDim coupling)
"Multicast": [-1, 0, 1],
```

Default `-1` preserves **all** existing emitted asm. Rewrite `Solution.py:1046-1064`:

```python
state["ClusterBarrier"] = False
mc = state.get("Multicast", -1)
if mc == 1:
    state["Multicast"] = True
elif mc == 0:
    state["Multicast"] = False
else:  # auto (legacy behavior)
    state["Multicast"] = (state["ClusterDim"] != [1, 1]
                          and not state.get("StreamKClusterReduction", 0)
                          and not state.get("StreamKMulticast", 0))
# ClusterBarrier decision unchanged (only in the legacy/cluster-subtile path)
if state["ClusterDim"] != [1, 1] and not state.get("StreamKClusterReduction", 0) \
   and state["Multicast"] and state["TDMInst"] != 0 \
   and isaInfoMap[state["ISA"]].asmCaps.get("HasClusterBarrier", False):
    state["ClusterBarrier"] = True
```

Net effect for existing configs: `Multicast` default `-1` → identical derivation to today
(subtile/dense clustered configs keep `Multicast=True`; `StreamKClusterReduction` keeps it
`False`). New capability: `Multicast=1`/`0` explicit override, and `StreamKMulticast` (§4)
turns on multicast for its own path without relying on the coupling.

This is the one step in the sequence that *can* legitimately change asm — but only for
configs that opt in. Keep existing subtile/multicast YAMLs on `Multicast=-1` so their
snapshots don't move.

---

## 4. StreamK cooperative cluster loads (DP region)

### 4.1 The spatial vs reduction tension, resolved

- Cooperative loads need a **spatial** cluster over **distinct** tiles (`[C,1]` consecutive
  WGs → M-adjacent → shared B).
- The shipped reduction needs a **`[C,1]` same-tile K-peer** cluster (host ties
  `skGrid = C*tiles`, `skIndexToWG` collapses the C peers onto one tile).
- A WG's HW cluster membership is fixed at launch, so a single kernel cannot have the
  cluster mean "spatial DP" and "K-peers" at once.

**Resolution: mutual exclusion.** For v1, `StreamKMulticast` and `StreamKClusterReduction`
are mutually exclusive. When `StreamKMulticast` is on, `ClusterDim=[C,1]` is free to denote
the spatial cluster; the reduction is off, so no `[C,1]` is claimed for K-peers. **No
separate `SpatialClusterDim` is needed in v1** — the same `ClusterDim=[C,1]` is reused, its
meaning selected by which StreamK cluster mode is enabled. (A future 2-D `[C0,C1]` mode for
A-multicast or combined reduction+coop-load would introduce a distinct spatial dim; deferred.)

### 4.2 MVP target: DP-only, B-multicast

The multicast mask is computed **once** at kernel init and baked into the descriptor
`Group1[word0]`; it is applied to every mainloop TDM load. For correctness, the sharing
relationship it encodes must hold for **every** load the WG issues. Therefore v1 requires
that the WG never enters a partial (K-split) tile with a stale spatial mask:

**v1 MVP = cooperative B-multicast for the DP region with `StreamKForceDPOnly=1`** (no SK
partial tiles → no reduction → the static spatial mask is valid for all issued loads), and
single DP round per cluster (see §4.4).

Recommended smallest correct MVP:
- **`StreamKMulticast=1`** requires `StreamKForceDPOnly=1` and `StreamKClusterReduction=0`.
- Reuse `ClusterDim=[C,1]`, `C` a power of two in `2..16`.
- Multicast B across the cluster; A loaded per-WG.

Deferred (see §7): mask-toggle at the DP→SK boundary to allow cooperative DP loads inside a
full SK3 kernel (SK tail reduces via the existing global-flag path); combined
reduction+coop-load; A-multicast / 2-D spatial cluster; multi-round with per-round validity.

### 4.3 Mask derivation from post-`skIndexToWG` coords

The existing `computeMasks` derives `wg_x`/`wg_y` from the raw HW cluster registers
(`ttmp*`). For StreamK the authoritative WG coordinates are the ones produced by
`skIndexToWG` (`StreamK.py:503`), not the raw HW position, because StreamK re-derives
`(WG0,WG1)` from `StreamKIdx`. For the DP `[C,1]` MVP the two coincide
(`wg_x = StreamKIdx & (C-1)`, since the reduction ttmp9 workaround is skipped and
`WorkGroup0 = cluster_x*C + wg_x`), so the MVP feeds `computeMasks` with
`sgprWgX = StreamKIdx & (C-1)`, `wg_y = 0`, `C0=C`, `C1=1`, yielding the B-broadcast mask
`(1<<C)-1`. The general requirement (mask from post-`skIndexToWG` coords) is documented so
the 2-D/A-multicast follow-ups compute `wg_x`/`wg_y` from `(WG0 mod C0, WG1 mod C1)` rather
than the raw cluster registers.

### 4.4 Correctness invariant + runtime predicate

For the C WGs of a cluster to genuinely share B on every issued load, in every DP round the
C tiles they process must be M-adjacent (same WG1/N block, consecutive WG0). With
`tile(WG=cC+j, round r) = cC + r*skGrid + j`, this holds when:

1. `C | nWG0` (M tiled to a multiple of C), and
2. `skGrid` is a multiple of `C` (already true — host rounds `skGrid`), and
3. the cluster is **fully populated** for the round (all C WGs have a valid tile
   `< totalTiles`).

Because incorrect multicast writes the wrong B tile into a peer's LDS (silent wrong
results, not just a perf loss), the MVP is conservative:

- **v1 single-round** (`skGrid == totalTiles`, rounded up to a multiple of `C`): sharing is
  invariant across the (single) round, so a single **init-time predicate**
  `clusterMulticastValid = (nWG0 % C == 0) && (cluster fully populated) && (M-adjacent)`
  suffices. `ClusterLoad.applyToDescriptor` ORs the mask only when the predicate holds;
  otherwise it ORs nothing and every WG loads B normally → correct for all sizes, multicast
  win on the aligned/full/interior common case. This mirrors the reduction feature's runtime
  `intra_cluster` guard philosophy.
- Multi-round DP (equal rounds per WG) is a near follow-up once the single-round path is
  validated on hardware.

`nWG0`, `totalTiles`, and cluster base are available at kernel init from sizes +
`StreamKIdx`; the predicate is cheap (`AND`/compare, `C` a power of two).

### 4.5 Host plumbing

- Add `sizeMapping.streamKMulticast` (mirror `streamKClusterReduction` in
  `ContractionSolution.hpp:146` and `Contractions.py:630,725`).
- Grid: for `StreamKMulticast` set `skGrid` to `totalTiles` rounded **up to a multiple of
  C** (reuse the rounding shape at `ContractionSolution.cpp:4087-4090` / `:3214-3218`, but
  `= ceil(tiles/C)*C` rather than `C*tiles`). Extra tail WGs (if `totalTiles % C != 0`) are
  idle-but-present; the init-time predicate disables their cluster's multicast.
- Kernarg accounting: `StreamKForceDPOnly` path already provides the DP schedule; no new
  kernarg beyond threading `streamKMulticast` (the mask is derived kernel-side from
  `StreamKIdx`, `NumWorkGroups0`, `ClusterDim`).
- Launch: the `enableCluster` branch (`ContractionSolution.cpp:1783-1792`) already keeps the
  grid cluster-friendly and sets `rv.clusterDim`; no launch change.

---

## 5. Enablement / validation

> **Status (as shipped):** `StreamKMulticast` is **not** a public/benchmark parameter.
> It is a derived-only internal state key (like `ClusterBarrier`), enabled automatically
> when `ClusterDim != [1,1]` on SK3 (and not `StreamKClusterReduction`) — see §3.1 / the
> collapse in `assignProblemIndependentDerivedParameters`. It has no `ValidParameters.py`
> entry; `_validateStreamKMulticast` is the internal guard that hard-rejects a derived
> config it cannot satisfy. The `[0,1]` "param" framing below reflects the original plan.

### 5.1 Parameters

| Param | Values | Meaning |
|---|---|---|
| `Multicast` | `[-1,0,1]` | Decoupled multicast control (§3). Default `-1` = legacy auto. |

`StreamKMulticast` is internal-derived (see status note above), not a user-settable param.
`StreamKClusterReduction` (existing) is unchanged.

### 5.2 Validation rules (`Solution.py`, reject-with-reason pattern)

The derived `StreamKMulticast` on-state is rejected (`_validateStreamKMulticast`) unless all hold:
- `StreamK == 3` (SK3; DP schedule + `skIndexToWG` assumptions).
- `StreamKForceDPOnly == 1` (v1: no partial tiles reach the static mask).
- `StreamKClusterReduction == 0` (mutually exclusive — cannot claim `[C,1]` two ways).
- `ClusterDim == [C,1]`, `C` power of two, `2 <= C <= 16` (`ClusterDim[1] == 1`; StreamK
  grid is effectively 1-D along x).
- gfx1250 `asmCaps["HasTDM"]` and `TDMInst != 0` (MX TDM loads; multicast is a TDM feature).
- `StreamKXCCMapping == 0` (XCC=3 overflows SGPRs; WGM/XCC remap is bypassed under
  clustering anyway).

Also set `Multicast=True` for this path via the decoupled derivation (§3.1) so
`ClusterLoad.applyToDescriptor` fires.

### 5.3 Interaction notes

- gfx1250 StreamK already uses TDM (`MXLoadInst=TDM` → `TDMInst=3`), so `HasTDM` is the
  natural cap and no new transport is introduced.
- `StreamKAtomic` is incompatible (atomic path skips the workspace/tile structure the DP
  schedule relies on); reject.

---

## 6. Ordered implementation task breakdown

Sequencing keeps the tree green: **pure extraction first (zero asm), then the decouple,
then the feature.** `[P]` marks tasks that can proceed in parallel within a group.

### Group 0 — Snapshot baseline (do first, no source change)
- **T0.1** Add byte-exact golden snapshots for the mask compute+apply region: a `[2,2]`
  combined-mask config, a `[2,2]` split-mask config, and a subtile split config. Anchor:
  extend `test_r4_xccremap_char.py` and the subtile char config under
  `Tests/unit/characterization/_codegen/data/test_data/_designed/gfx1250/`.

### Group 1 — `ClusterLoad` extraction (pure refactor, snapshot-gated)
- **T1.1** Declare `class ClusterLoad(Component)` at `Component.py:305-313`; add
  `"ClusterLoad"` to `Components/__init__.py:__all__`.
- **T1.2** Create `Components/ClusterLoad.py` with `ClusterLoadTDM` (SPDX header;
  `asmCaps={"HasTDM":True}`, `kernel={"TDMInst":3}`); implement `usesCombinedMask`,
  `maskSgprName`, `cooperativeThreadPartition`.
- **T1.3** Move mask compute into `computeMasks`; wire `KernelWriterAssembly.py:2646-2694`
  to call it with the existing SGPR operands.
- **T1.4 [P]** Move declare/undeclare into `declareSgprs`/`undeclareSgprs`; wire
  `KernelWriter.py:9163-9176` and `:2838-2848`.
- **T1.5 [P]** Implement `applyToDescriptor`; wire the 3 apply sites
  (`KernelWriterAssembly.py:18901-18902`, `:19059-19060`, `SubtileGREmit.py:1108-1113`).
- **T1.6** Point `GL2Prefetch.py:26,78` at `cooperativeThreadPartition` (optional but DRY).
- **Gate:** T0.1 snapshots + `test_r4_xccremap_char.py` +
  `test_streamk_cluster_coop_load_gfx1250_char.py` all diff to zero.

### Group 2 — Decouple `Multicast` (may change asm only for opt-in configs)
- **T2.1** Add `"Multicast": [-1,0,1]` to `ValidParameters.py`.
- **T2.2** Rewrite `Solution.py:1046-1064` per §3.1; keep existing YAMLs on `-1`.
- **Gate:** existing multicast/subtile/reduction snapshots unchanged; add a snapshot for an
  explicit `Multicast=1`/`0` override.

### Group 3 — StreamK cooperative DP multicast (feature)
- **T3.1** Derive `StreamKMulticast` internally (no `ValidParameters.py` entry — it is a
  derived-only state key, auto-enabled on SK3 + `ClusterDim!=[1,1]` not reduction) +
  `Solution.py` validation (§5.2) + decoupled `Multicast` enablement.
- **T3.2 [P]** Host: `sizeMapping.streamKMulticast` (`ContractionSolution.hpp:146`,
  `Contractions.py:630,725`); `getSKGridImpl` grid = `ceil(tiles/C)*C`
  (`ContractionSolution.cpp:4087-4090`/`:3214-3218` region); thread through launch (no
  launch change — `:1783-1794` already cluster-aware).
- **T3.3** Kernel: at StreamK preLoop compute `wg_x = StreamKIdx & (C-1)` and the
  `clusterMulticastValid` predicate (§4.4) from `NumWorkGroups0`/`totalTiles`; feed
  `ClusterLoad.computeMasks` with `wg_y=0, C0=C, C1=1` and gate the mask OR on the predicate
  (via `applyToDescriptor`). Anchors: `StreamK.py:2645-2663` (StreamKIdx/ttmp9),
  `skIndexToWG` `:487-503`.
- **T3.4** Ensure A stays per-WG (maskA = self) and B multicast (maskB = all-C); verify the
  `[C,1]` path selects split `MulticastMaskA/B` names.

Dependencies: Group 1 → Group 2 → Group 3 (T3.1). T3.2 (host) is parallel to T3.3/T3.4
(kernel) once T3.1 lands the param.

---

## 7. Correctness, edge cases, risks

| Risk | Why | De-risk |
|---|---|---|
| **Wrong-B multicast → silent wrong results** | Static mask applied to a load where the C WGs don't actually share B (SK partial tile, non-adjacent, partial cluster). | v1 requires `StreamKForceDPOnly` (no partial tiles) + single round + init-time `clusterMulticastValid` predicate that ORs the mask only when provably valid; else load normally. |
| **Idle WG in a partial last cluster** | `totalTiles % C != 0` leaves tail WGs with no tile; a multicast expecting them may hang/mis-deliver. | Predicate disables multicast for a not-fully-populated cluster. **OPEN Q**: exact HW behavior when a masked target WG is not at a matching TDM load (see §8). |
| **DP→SK boundary with a live spatial mask** | In a full SK3 kernel the mask would still be applied to SK-tile loads. | v1 forbids the SK region (`StreamKForceDPOnly`). Deferred: clear the mask bit (single `SAndB32` on `Group1[word0]`) at the boundary (`StreamK.py:2907-2932`) since DP is first and SK last. |
| **Refactor changes asm** | Component lift could reorder/realloc SGPRs. | `computeMasks` takes the exact existing SGPR operands; snapshot gate (Group 0) diffs to zero. |
| **`Multicast` decouple regresses existing configs** | Changing the coupling could flip `Multicast` for subtile/dense clustered kernels. | Default `-1` = identity derivation; existing YAMLs untouched; snapshot-gated. |
| **Mask from raw HW coords instead of skIndexToWG** | StreamK re-derives `(WG0,WG1)`. | For DP `[C,1]` the two coincide (`wg_x=StreamKIdx&(C-1)`); documented general rule for 2-D follow-ups. |
| **SGPR pressure** | gfx1250 StreamK SGPRs tight; XCC=3 overflows. | Reuse existing `MulticastMask*` SGPRs; predicate uses `allocTmpSgpr` scopes and bitwise ops (C power of two); forbid `StreamKXCCMapping=3`. |
| **A-multicast expectation** | Consecutive `[C,1]` cluster shares B, not A. | Documented; A stays per-WG in v1; A-multicast needs an N-strided/2-D cluster (deferred). |

---

## 8. Test plan (behavior → test) — as shipped

Mirrors existing patterns: CPU asm-string unit, syrupy snapshot characterization, GPU
roundtrip (run under the arch's simulator/hardware), and C++ client end-to-end.

| Behavior | Test type | Where |
|---|---|---|
| **Refactor is byte-exact**: `computeMasks`/`applyToDescriptor`/declare/undeclare emit identical asm for `[2,2]` combined, `[2,2]` split, and subtile split configs | snapshot char (regression gate) | `test_r4_xccremap_char.py`, `test_streamk_cluster_coop_load_gfx1250_char.py` |
| `ClusterLoad.find` returns TDM impl on gfx1250, `None` (fallback) on non-TDM; `usesCombinedMask`/`maskSgprName` matrix (combined vs split vs metadata) | CPU unit | `Tests/unit/test_cluster_load_component.py` |
| `Multicast` tri-state: `-1` reproduces legacy derivation exactly; `0` forces off; `1` forces on independent of `ClusterDim` (derived value is an int on all paths) | CPU unit | `Tests/unit/test_multicast_tristate.py`, `Tests/unit/test_multicast_legacy_coercion.py` |
| `StreamKMulticast` validation matrix: accepted only for SK3 + `ClusterDim=[C,1]` (pow2 2..16) + gfx1250 `HasTDM`/`TDMInst=3` + XCC=0 + not `StreamKClusterReduction`/`StreamKAtomic`; auto-enabled config with `Multicast=0` rejected; rejected otherwise | CPU unit | `Tests/unit/test_streamk_multicast.py` |
| Cluster config emits real gfx1250 asm (`err==0`); DP loads carry the B-broadcast mask (`MulticastMaskB` OR into `Group1`); A load carries self-only mask; predicate gate + DP→SK boundary clear present | snapshot char | `test_streamk_cluster_multicast_gfx1250_char.py` (designed config `_designed/gfx1250/streamk_cluster_multicast.yaml`, `__snapshots__/*.ambr`) |
| DP GEMM with `StreamKMulticast`, `C∈{2,4}` reduces correctly and does not hang; matches non-multicast reference | GPU roundtrip | `Tests/unit/test_streamk_cluster_reduction_gpu.py` and cluster char roundtrips (`@requires_gpu_gfx1250`; watchdog on hang) |
| Real multi-WG cluster StreamK DP GEMM end-to-end | C++ client | `Tests/common/streamk/gfx1250/core/sk_mxf8gemm_cluster_multicast.yaml` (+ mxf4 sibling) |

The gfx1250 GPU marker reuses `Tests/unit/gpu_test_helpers.py`
(`HAS_GFX1250`/`requires_gpu_gfx1250`, `TENSILE_GPU_TARGET=gfx1250`).

---

## 9. Open questions (need a human decision at the design gate)

1. **HW multicast semantics when a masked target WG is not executing a matching TDM load**
   (idle tail WG, or a WG that finished its tiles): does the loader hang, drop the
   broadcast, or corrupt? This determines whether the init-time predicate is sufficient or
   whether the grid must be constrained to always-full clusters. (Mirrors the reduction
   feature's deadlock concern.)
2. **Is `StreamKForceDPOnly`-scoped v1 valuable enough**, or should we prioritize the
   DP→SK boundary mask-clear so cooperative loads apply inside a normal SK3 kernel (the
   larger real-world win)? The boundary-clear is a modest add (§7) but widens the test
   matrix.
3. **`Multicast` as tri-state `[-1,0,1]` vs a separate boolean** (`MulticastEnable`) — the
   tri-state keeps one knob and a clean legacy default; a separate boolean is more explicit
   but adds a second interacting param. Recommend tri-state; confirm.
4. **Grid rounding policy** for `StreamKMulticast`: `ceil(tiles/C)*C` (idle tail WGs) vs
   requiring `totalTiles % C == 0` at selection time (fewer sizes, no idle WGs). Depends on
   Q1.

---

## 10. Summary of decisions

- **`ClusterLoad` component**: new capability-selected category (`asmCaps HasTDM`,
  `kernel TDMInst=3`) owning mask value (`computeMasks`), declare/undeclare, topology
  decision (`usesCombinedMask`/`maskSgprName`), descriptor attach (`applyToDescriptor`,
  gate folded in), and `cooperativeThreadPartition`. It does **not** own descriptor groups
  or LDS offsets. The three apply sites and the subtile clone unify onto it; refactor is
  byte-exact and snapshot-gated.
- **Decouple**: `Multicast` becomes a tri-state param (`-1` auto = legacy identity, `0`
  off, `1` on); `Solution.py:1046-1064` derivation rewritten so `ClusterDim!=[1,1]` no
  longer unconditionally forces multicast.
- **StreamK target = DP region, B-multicast** on a `[C,1]` consecutive-WG cluster
  (M-adjacent tiles share the N-block). Spatial-vs-reduction tension resolved by **mutual
  exclusion** (`StreamKMulticast` xor `StreamKClusterReduction`), reusing `ClusterDim=[C,1]`
  with mode-selected meaning; no separate spatial dim in v1.
- **MVP** = `StreamKMulticast=1` requiring `StreamKForceDPOnly=1`, single DP round, init-time
  `clusterMulticastValid` predicate gating the mask. Deferred: DP→SK boundary mask-clear,
  multi-round, A-multicast / 2-D cluster, combined reduction+coop-load.
- **Enablement**: `Multicast` (`[-1,0,1]`) is the one public knob; `StreamKMulticast` is
  internal-derived (auto-enabled on SK3 + `ClusterDim!=[1,1]`, not reduction), not a param.
- **Sequencing**: extraction (green) → decouple → feature.
