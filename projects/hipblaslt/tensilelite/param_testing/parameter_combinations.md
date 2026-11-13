# Legal parameter combinations

Within-group combinations that currently survive derivation (`Solution.py` /
`ValidParameters.py`). Sentinels that auto-resolve are listed as the values
they become. Cross-group prerequisites are noted only when they gate a value
inside the group.

Source: `parameter_groups.md`.

---

## 1. TDM loads (gfx1250)

### Core transport

| Combination | Notes |
| --- | --- |
| `TDMInst=0`, `MXLoadInst=BufferLoad` | Non-TDM path. Auto → this when `TDMInst=0`. |
| `TDMInst=3`, `MXLoadInst=TDM` | Only live TDM path. Auto promotes `TDMInst=0`→`3` when `MXLoadInst=TDM`. |

With MX scales present, those pair with:

| `MXLoadInst` | Legal `MXScaleFormat` |
| --- | --- |
| `TDM` | `InMemorySwizzle` only |
| `BufferLoad` | `NoSwizzle`, or `HostPreSwizzle` (gfx950) |

### Knobs on top of `TDMInst=3`

| Param | Legal values | Requires |
| --- | --- | --- |
| `TDMSplit` | `False` | `True` is disabled (read-token / tensorcnt races). |
| `TDMPlusLdsBuf` | `0` | Forced off (cross-wave LDS RAW race). |
| `TDMLoadWaveSync` | `False`; `True` | `True` → `ScheduleIterAlg=4` and `NumWaves>1` (else cleared to `False`). |
| `TDMFuse` | `0` | Always. |
| | `1` | `TDMInst=3`, `NumWaves>1`, `UseSubtileImpl=0`. |
| | `2`, `3` | Same as `1`; with MX on both tensors also `NumWaves ∈ {2,4}` and equal PGRA/PGRB LDS-block counts. |
| `TDMIterateMode` | `-1` (auto), `0` | Always (with TDM). |
| | `1` | `UnrollMajorLDSA`, `TLUA=False`. |
| | `2` | `UnrollMajorLDSB`, `TLUB=False`. |
| | `3` | Both of the above. |

Wave-separated TDM also needs `NumWaves` a power of two (≥2 when multi-wave).

---

## 2. Workgroup cluster (gfx1250)

### `ClusterDim`

All `[Cs, Ck]` with `Cs*Ck ∈ [1, 16]` **except** `[16,1]` and `[1,16]`.

| Shape class | Legal when |
| --- | --- |
| `[1,1]` | Always (clustering off). |
| `[Cs,1]`, `Cs>1` | DataParallel, or StreamK/StaticGrid (SK3). Not SK4/SK5. |
| `[Cs,Ck]`, `Ck>1` | Persistent spatial cluster: DataParallel or StreamK/StaticGrid with spatial clustering. Not SK4/SK5. |

Power-of-two `Cs`/`Ck` with product in `[2,16]` is required for the persistent spatial-cluster multicast path.

### `ClusterAIDMapping`

| Value | Meaning |
| --- | --- |
| `-1` | Off (default). |
| `-2` | Runtime AID via `MSG_RTN_GET_SE_AID_ID` (pairs with WGM-after-CAR). |

Both are legal for any `ClusterDim`; `-2` is only useful when clustering is on.

---

## 3. LDS segment / bank conflict (gfx1250)

| Combination | Legal when |
| --- | --- |
| `LDSSegmentInterleave=-1` | Auto: applies only the no-LDS-growth (“tight”) layout when eligible. |
| `LDSSegmentInterleave=0` | Always (force baseline). |
| `LDSSegmentInterleave=1` | Wave-separated TDM, gfx1250, `NumWaves==4`, `1LDSBuffer=0`, unroll-major A/B, no sparse/subtile/DTV, bf16/fp16/fp8/fp4, and `MIWaveGroup` in `{[2,2],[4,1],[1,4]}` (with matching VW / port-split rules). |
| `LDSSegmentBRotate=0` | Always. |
| `LDSSegmentBRotate=1` | `LDSSegmentInterleave=1` resolved, `[2,2]` / 4 waves / wave32, 2 LDS buffers, B component stride in `[64K,128K)`. |

---

## 4. Memory temporal / volatility hints

| Param | Legal values |
| --- | --- |
| `TemporalHint` / per-tensor fields | `-1` (broadcast / default), or `0..7` on arches with `HasTHModifier` (gfx1250). |
| `NonVolatile` / per-tensor fields | `-1`, or `0`/`1` on arches with `HasNVModifier`. |
| `AdaptiveGemmNTAB` | `0`, `1` (`1` bumps `KernArgsVersion` to ≥3). |

Hints are independent of each other; non-zero values need the matching ISA modifier.

---

## 5. Prefetch (SWI + GL2)

### `SwInstructionPrefetch`

| Value | Legal when |
| --- | --- |
| `-1` Auto | Always (→ Absolute on gfx1250 non-StreamK non-f64; else Relative). |
| `0` Off | Always. |
| `1` Relative | Always. |
| `2` Absolute | gfx1250, non-StreamK, non-f64. |

### `PrefetchGL2` family

| Combination | Legal when |
| --- | --- |
| `PrefetchGL2=0`, `Inc64Bit=False`, `SAddr=False` | Always (`Inc`/`SAddr` forced off). |
| `PrefetchGL2 ∈ {1,2}` × `Inc64Bit ∈ {F,T}` × `SAddr ∈ {F,T}` | `HasGlobalPrefetch`, not fp6, StridedBatched if batched, StaticGrid if persistent, and sparse ⇒ `DirectToVgprSparseMetadata=0`. |

---

## 6. Persistent tile reuse

| Value | Legal when |
| --- | --- |
| `ReuseAcrossPersistent=0` | Always. |
| `ReuseAcrossPersistent=1` | DataParallel/StaticGrid + `PGR≥1` + `TDMInst=3` + `NumWaves>1` + `InnerUnroll=1` + `ExpandPointerSwap=0` + `UnrollMajorLDSA` + no DTV A/MXSA + no subtile + no SpaceFillingAlgo; with MX scales, `TDMFuse=0` (or no MX and fuse 1–3). |

---

## 7. Epilogue / store scheduling

| Param | Legal values |
| --- | --- |
| `LDSTrStoreShuffle` | `0`; or `1`/`2` when `enableLDSTrA`, `VectorWidthA>1`, `SourceSwap`, no `MXBlockA`, `StoreRemapVectorWidth=0`. |
| `InterleaveAlpha` | Matrix-instruction kernels: `0`, `2`. Non-MI: `0`, `1`, `2`. |
| `PostLoopStoreInNll` | `False` always; `True` only for gfx950 FP4 + `UseSubtileImpl` + `PGR≥1` + `BufferStore` + paired-store / SK-atomic-free / spill-budget gates (else auto-cleared to `False`). |

---

## 8. Codegen / RA tooling

| `StinkyTofuRegisterAllocation` | Legal |
| --- | --- |
| `0` / `1` / `2` / `3` | Off / shadow / apply / force. |
| Config string | Any string StinkyTofu’s RA parser accepts (e.g. `"apply,classes=vs"`), mode in `0..3`. |
