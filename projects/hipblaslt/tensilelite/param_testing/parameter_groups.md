# TensileLite parameter groups

Feature-affinity clustering of gfx1250-oriented (and a few related) solution
parameters. Items marked *(recent)* were previously in a chronological
"Newest additions" bucket and have been folded into the families they belong to.

Legal within-group combinations: [`parameter_combinations.md`](parameter_combinations.md).

OOB workload harness: [`bbs_tn_oob_param_groups.yaml`](bbs_tn_oob_param_groups.yaml)
(representatives: [`oob_representatives.md`](oob_representatives.md)).

Per-group run folders + scripts: [`runs/`](runs/) (`./runs/run_all.sh smoke`).

| Group | Parameter | Domain | Explanation |
| --- | --- | --- | --- |
| **TDM loads (gfx1250)** | `TDMInst` | 0, 1, 2, 3 | Hardware Tensor Data Mover for VRAM→LDS: 0 off, 1 A, 2 B, 3 both. (1, 2 currently rejected) |
| | `MXLoadInst` | Auto \| TDM \| BufferLoad \| GlobalLoad | How A/B and MX scales are loaded. Auto → TDM when `TDMInst != 0`, else BufferLoad |
| | `TDMSplit` | False, True | Split one TDM tensor load into two `tensor_load_to_lds` (half macro-tile each); also 2:4 sparse metadata |
| | `TDMFuse` | 0, 1, 2, 3 | Which tensors (A/B/MXSA/MXSB) share one TDM descriptor set per wave |
| | `TDMPlusLdsBuf` | -1, 0, 1 | Force a 3rd LDS buffer on TDM PGR=2 so next-iter loads don’t collide with in-flight reads |
| | `TDMLoadWaveSync` | False, True | Barrier between urgent and deferrable TDM groups (SIA=4 only) |
| | `TDMIterateMode` | -1, 0, 1, 2, 3 | Iterate-mode TDM: 0 off, 1 A, 2 B, 3 both |
| **Workgroup cluster (gfx1250)** | `ClusterDim` | `validClusterDimensions` | Cluster shape (≤16 WGs/cluster, one per WGP) for StreamK / DataParallel |
| | `ClusterAIDMapping` | -2, -1 | Remap clusters onto the 2 AIDs so each GL2 reads one operand once. Domain narrowed from -2..7; only runtime AID (-2) or off (-1), with WGM-after-CAR ordering |
| **LDS segment / bank conflict (gfx1250)** | `LDSSegmentInterleave` | -1, 0, 1 | Split an operand’s two components across the two 64 KiB LDS segments |
| | `LDSSegmentBRotate` | 0, 1 | Rotate B’s LDS segment read pattern (pairs with interleave) |
| **Memory temporal / volatility hints** | `TemporalHint` (+A/B/C/D/E/MXSA/MXSB/WS/Metadata/Gate) | -1..7 (per-field 0..7) | gfx1250 temporal-hint modifier on loads/stores |
| | `NonVolatile` (+same fields) | -1, 0, 1 (per-field 0, 1) | gfx1250 non-volatile access modifier |
| | `AdaptiveGemmNTAB` | 0, 1 | Pick `NonTemporalA/B` at runtime from problem size / stride alignment *(recent)* |
| **Prefetch (SWI + GL2)** | `SwInstructionPrefetch` | -1 Auto, 0 Off, 1 Relative, 2 Absolute | Software instruction prefetch; Absolute is gfx1250 non-StreamK only |
| | `PrefetchGL2` | 0, 1, 2 | `global_prefetch_b8` GL2 prefetch: 0 off, 1/2 = 1 or 2 tiles ahead |
| | `PrefetchGL2Inc64Bit` | False, True | 64-bit (vs 32-bit) PrefetchGL2 address-increment SGPR *(recent)* |
| | `PrefetchGL2SAddr` | False, True | PrefetchGL2 via scalar base + VGPR offset instead of VGPR address pairs *(recent)* |
| **Persistent tile reuse** | `ReuseAcrossPersistent` | 0, 1 | Keep an operand (and MX scales) in VGPRs across persistent tiles *(recent)* |
| **Epilogue / store scheduling** | `LDSTrStoreShuffle` | 0, 1, 2 | Reorder epilogue accums so `VectorWidthA > 1` works with `ds_read_tr` (ex-`AccumShuffle`) *(recent)* |
| | `InterleaveAlpha` | 0, 1, 2 | Where alpha lands in global write: 0 up-front, 1 interleaved, 2 deferred *(recent)* |
| | `PostLoopStoreInNll` | False, True | Fuse D-tile store into NLL so early accums store while later ones compute (gfx950 MXFP4) *(recent)* |
| **Codegen / RA tooling** | `StinkyTofuRegisterAllocation` | mode int or config string | StinkyTofu RA: off / shadow / apply / force *(recent)* |

## Residual recent set

If a chronological "newest" view is still useful for notes or changelogs:

- `PrefetchGL2Inc64Bit`
- `PrefetchGL2SAddr`
- `ReuseAcrossPersistent`
- `LDSTrStoreShuffle`
- `InterleaveAlpha`
- `PostLoopStoreInNll`
- `AdaptiveGemmNTAB`
- `StinkyTofuRegisterAllocation`
- `ClusterAIDMapping` domain narrowing (-2..7 → -2, -1)
