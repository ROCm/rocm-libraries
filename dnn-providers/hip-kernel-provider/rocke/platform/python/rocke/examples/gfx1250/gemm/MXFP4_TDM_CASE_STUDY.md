# MX FP4 TDM/LDS GEMM case study

This case study records the functional implementation of a configurable
double-buffered MX FP4 GEMM for `gfx1250`. It contains correctness and code
generation evidence only; it does not contain performance measurements.

## Contract

The builder implements the rocKE RCR view
`D[m,n] = alpha * sum_k(A[m,k] * B[n,k]) + beta * C[m,n]`.
The matrix inputs are packed FP4 E2M1, A and B use independent E8M0 scales for
each block of 32 K values, accumulation is FP32, and C and D are FP8 E4M3.
The compute atom is `v_wmma_scale_f32_32x16x128_f4`.

`MxFp4TdmGemmSpec` exposes workgroup tiles as `tile_m`, `tile_n`, and `tile_k`;
wave groups as `warp_m`, `warp_n`, and `warp_k`; and fixes the wave atom at
`32x16x128`. Wave repeats are derived from the workgroup tile and wave group.
The default feasible workgroup tile is `128x64x128` with a `2x2x1` wave group.

## Data movement and pipeline

The builder uses the in-tree `IRBuilder.tdm_descriptor_2d` operation. One wave
issues row-major descriptors for packed A and B tiles and
`tensor_load_to_lds` transfers them into LDS. Two A/B LDS buffers alternate in
the dynamic K loop: after waiting for the current transfer and synchronizing
the workgroup, the kernel issues the next transfer before reading the current
buffer and executing WMMA.

Explicit LDS vector reads construct the A `<16xi32>` and B `<8xi32>` fragments.
E8M0 scales remain direct global loads. The FP8 alpha/beta epilogue runs after
the final K tile.

The resource validator rejects unsupported geometry, non-divisible dimensions,
K-split waves, LDS over the arch cap (`ArchTarget.lds_capacity_bytes` =
327680), and accumulator tiles whose FP32 values per lane exceed gfx1250
`MaxVgpr` (`ArchTarget.limits.vgprs` = 1024, from StinkyTofu
`Gfx1250Formats.def` / TensileLite `hardware_caps.hpp`). The YAML MacroTile
`256x256x512` (512 values/lane, 262144 B LDS) fits both budgets.

## Functional evidence

Under the gfx1250 functional model, the independent FP8 reference matched
byte-for-byte for the default tile at K values 128, 256, and 512, for a
`64x32x128` tile, and for `tile_k=512`. Host checks also confirm TDM loads,
tensor waits, LDS reads, the target WMMA intrinsic, and FP8 conversion
instructions in emitted LLVM IR.

The C++ engine mirrors the representative `64x32x128`, K=256 configuration.
Canonical IR and LLVM 23 output are byte-identical to the Python engine. Other
C++ configurations are rejected explicitly until the general mirror is added.

## Geometry probe

Step 5 kept the two-buffer pipeline unchanged and asked two questions: which
tiles the resource validator rejects, and whether every accepted tile still
matches the FP8 reference. After correcting rocKE's gfx1250 `MaxVgpr=1024` and
`DeviceLDS=327680` caps, the probe was re-run with the accumulator gate tied to
`ArchTarget.limits.vgprs`.

Accepted under FFM with `bad=0` (no bug found):

| Workgroup tile | Waves | Accumulators / lane | Notes |
| --- | --- | --- | --- |
| `128x64x128` | `2x2` | 64 | Default; K in `{128,256,512}` |
| `128x64x512` | `2x2` | 64 | Four WMMA K-steps per transferred tile |
| `64x32x128` | `1x1` | 64 | One-wave workgroup |
| `256x64x128` and `256x64x512` | `2x2` | 128 | Grown M |
| `128x128x128` and `128x128x512` | `2x2` | 128 | Grown N |
| `128x64x128` | `4x1` | 64 | Alternate wave group |
| `256x64x128` | `4x2` | 64 | Alternate wave group on grown M |
| YAML `256x256x512` | `2x2` | 512 | SourceSwap MacroTile; 262144 B LDS |

Rejected before launch (kernel limitations, not bugs):

| Workgroup tile | Waves | Accumulators / lane | Reason |
| --- | --- | --- | --- |
| `512x512x128` | `2x2` | 2048 | Accumulators alone exceed `MaxVgpr=1024` |

With `SourceSwap=True`, the YAML MacroTile is `256x256x512` (TensileLite client
confirmed). It now passes the resource gate and matches the FP8 reference under
FFM (`bad=0`). No Step 4 defect was found on the accepted geometries.

HSACO resource check (`compile_kernel` + `analyze_hsaco`, llvm23) before Step 6:

| Workgroup tile | `vgpr_count` | `vgpr_spill_count` | `sgpr_spill_count` | `private_segment_fixed_size` | LDS |
| --- | --- | --- | --- | --- | --- |
| `128x64x128` | 130 | 0 | 0 | 0 | 24576 |
| `256x64x512` | 408 | 0 | 0 | 0 | 163840 |
| YAML `256x256x512` | 735 | 0 | 0 | 0 | 262144 |

The YAML tile uses 735 of 1024 VGPRs with no VGPR/SGPR spill and no scratch.

## Cumulative variants (Step 6)

Named presets over the same builder live in
`rocke.instances.gfx1250.mxfp4_tdm_gemm_variants`. Step 6 brings the kernel to
the hipBLASLt YAML winner (`mxf4_tn_maf.yaml`, SourceSwap, MT 256x256x512,
MIWaveTile 8x4, waves 2x2) one source-level feature at a time. Each
`step6_N_mxfp4_gemm` includes every earlier flag, so a sweep shows what each
feature is worth. Every variant is built at the YAML tile. Tensile A/B/M are
rocKE B/A/N.

`python -m rocke.examples.gfx1250.gemm.mxfp4_tdm_gemm --yaml-tile --m 1024 --n
1024 --k 2048 --variant all` checks the chain byte-exactly (`--saturation`
adds the FP8 overflow check). `mxfp4_tdm_gemm_variants_bench` checks at beta 0
and 0.5, then times beta 0 (host-timed batch and HIP-event median) on the
Tensile client's inputs (A `TrigSin`, B `TrigCos`, `Random` scales, alpha 1;
`--timing-inputs check` times the check inputs instead);
`--host-only` reports resources without a device. Timings stay outside the
repo.

| Step | Variant | Adds | YAML parameter |
| --- | --- | --- | --- |
| 6.1 | `base_kernel_mxfp4_gemm` | (none) | MT, MIWaveTile, waves, DepthU |
| 6.2 | `step6_2_mxfp4_gemm` | `vector_epilogue` | `VectorWidthA=8`, `StoreVectorWidth=8`, saturating FP8 store |
| 6.3 | `step6_3_mxfp4_gemm` | `local_read_prefetch` | `PrefetchLocalRead=1`, `HalfPLRA=1` (rocKE B) |
| 6.4 | `step6_4_mxfp4_gemm` | `wave_parity_tdm` | `TDMInstruction=3`, `TDMInstMode=1` |
| 6.5 | `step6_5_mxfp4_gemm` | `scale_tdm` | `MXLoadInstr=TDM`, `MXSwizzle=InMemorySwizzle`, `VectorWidthB=4` |
| 6.6 | `step6_6_mxfp4_gemm` | `lds_pad` | `LdsPad*`, `LdsBlockSizePerPad*` |
| 6.7 | `step6_7_mxfp4_gemm` | `lds_segment_interleave` | `LDSSegmentInterleave=1` |
| 6.8 | `step6_8_mxfp4_gemm` | `cluster_dim=(4,4,1)`, `tdm_multicast` | `ClusterDim=4x4`, `TDMMulticast` |
| 6.9 | `step6_9_mxfp4_gemm` | `expert_scheduling` | `EnableESM2` |
| 6.10 | `step6_10_mxfp4_gemm` | `tdm_desc_hoist` | SGPR TDM descriptors (`OptLevel: 0` loop) |
| 6.11 | `step6_11_mxfp4_gemm` | `wmma_n_outer` | WMMA order (`OptLevel: 0` loop) |
| 6.12 | `step6_12_mxfp4_gemm` | `cluster_aid_remap` | `ClusterAIDMapping=-2` |
| 6.13 | `step6_13_mxfp4_gemm` | `epilogue_nan_clamp`, `epilogue_pk8_cvt` | FP8 store sequence |
| 6.14 | `step6_14_mxfp4_gemm` | `init_c_wmma`, `tile_refill_mid`, `tile_read_ahead`, `source_order`, `cluster_signal_wmmas=6`, `cluster_signal_wave0`, `prefetch_gl2`, `coarse_ds_wait`, `loop_unroll2` | winner's main loop (`PrefetchGlobalRead=2`, `PrefetchGL2`, `InitCIterWmma`, `OptLevel: 0` loop) |

The chain keeps only steps that are faster than the step before them, or
that a later faster step needs: 6.4 is slower on its own, but `scale_tdm`,
`lds_segment_interleave` and `tdm_multicast` all build on it. Three YAML
features were slower than the step before them on the YAML shape, so they
are trials (`--trials`), not chain steps. They build on 6.8, not 6.9:

| Trial | Variant | Adds to 6.8 | YAML parameter |
| --- | --- | --- | --- |
| T1 | `trial_prefetch_gl2_mxfp4_gemm` | `prefetch_gl2` | `PrefetchGL2` (cluster-split) |
| T2 | `trial_init_c_wmma_mxfp4_gemm` | `init_c_wmma` | `InitCWMMA` |
| T3 | `trial_tile_refill_mid_mxfp4_gemm` | `init_c_wmma`, `tile_refill_mid` | `PrefetchGlobalRead=2`, `SuppressNoLoadLoop=1` |

T4 is not a YAML feature. It replaces 6.6 with CK Tile's `comp_tdm` LDS
padding, so it builds on 6.5 and compares against 6.6:

| Trial | Variant | Adds to 6.5 | Source |
| --- | --- | --- | --- |
| T4 | `trial_ck_lds_pad_mxfp4_gemm` | `lds_pad_ck` | CK Tile `GetLdsPaddingConfig` |

T5 to T8 build on 6.9. They follow the winner compiled at StinkyTofu
`OptLevel: 0`, which runs no scheduler:

| Trial | Variant | Adds to 6.9 | Source |
| --- | --- | --- | --- |
| T5 | `trial_tile_refill_mid_on6_9_mxfp4_gemm` | `init_c_wmma`, `tile_refill_mid` | T3 on 6.9 |
| T6 | `trial_read_ahead_on6_9_mxfp4_gemm` | T5, `tile_read_ahead` | `PrefetchLocalRead` across tiles |
| T7 | `trial_read_ahead_srco_on6_9_mxfp4_gemm` | T6, `source_order` | StinkyTofu `OptLevel: 0` |
| T8 | `trial_srco_on6_9_mxfp4_gemm` | `source_order` | StinkyTofu `OptLevel: 0` |
| T9 | `trial_csig6_on6_9_mxfp4_gemm` | T7, `cluster_signal_wmmas=6` | split cluster barrier, `OptLevel: 0` loop |
| T10 | `trial_csig0_on6_9_mxfp4_gemm` | T7, `cluster_signal_wmmas=0` | split cluster barrier |
| T11 | `trial_pgl2_csig6_on6_9_mxfp4_gemm` | T9, `prefetch_gl2` | `PrefetchGL2=2` |
| T12 | `trial_pgl2e_csig6_on6_9_mxfp4_gemm` | T11, `gl2_prefetch_early` | `PrefetchGL2=2`, `OptLevel: 0` loop |
| T13 | `trial_refill_mid_nopeel_on6_9_mxfp4_gemm` | T5 without `init_c_wmma` | control |
| T14 | `trial_read_ahead_srco_nopeel_on6_9_mxfp4_gemm` | T7 without `init_c_wmma` | control |
| T15 | `trial_dhst_on_t12_mxfp4_gemm` | T12, `tdm_desc_hoist` | SGPR TDM descriptors |
| T16 | `trial_nout_on_t12_mxfp4_gemm` | T12, `wmma_n_outer` | WMMA order |
| T17 | `trial_cdsw_on_t12_mxfp4_gemm` | T12, `coarse_ds_wait` | `s_wait_dscnt` per WMMA block |
| T18 | `trial_csw0_on_t12_mxfp4_gemm` | T12, `cluster_signal_wave0` | wave-0 cluster signal |
| T19 | `trial_loop_safe_on_t12_mxfp4_gemm` | T15 + T16 + T17 | `OptLevel: 0` loop |
| T20 | `trial_loop_all_on_t12_mxfp4_gemm` | T19, `cluster_signal_wave0` | `OptLevel: 0` loop |
| T21 | `trial_noarb_on6_11_mxfp4_gemm` | 6.11, `wmma_arb_stall_off` | `DisableXdlArbStall` |
| T22 | `trial_pka_on6_12_mxfp4_gemm` | 6.12, `kernarg_preload` | `PreloadKernArgs` |
| T23 | `trial_setup_on6_11_mxfp4_gemm` | 6.11, T21, T22 | winner's kernel setup |
| T24 | `trial_caid_on6_10_mxfp4_gemm` | 6.10, `cluster_aid_remap` | `ClusterAIDMapping=-2` |
| T25 | `trial_nclp_on6_12_mxfp4_gemm` | 6.12, `epilogue_nan_clamp` | FP8 store clamp |
| T26 | `trial_pk8_on6_12_mxfp4_gemm` | 6.12, `epilogue_pk8_cvt` | FP8 store convert |
| T27 | `trial_ipf_on6_13_mxfp4_gemm` | 6.13, `epilogue_inst_prefetch=6` | winner's epilogue |
| T28 | `trial_ipf_on6_12_mxfp4_gemm` | 6.12, `epilogue_inst_prefetch=6` | SW instruction prefetch |
| T29 | `trial_unr2_on6_13_mxfp4_gemm` | 6.13, `loop_unroll2` | main loop unrolled (winner: 3 tiles) |
| T30 | `trial_unr2_on_t20_mxfp4_gemm` | T20, `loop_unroll2` | constant LDS buffers on T20's loop |
| T31 | `trial_t20_unr2_on6_13_mxfp4_gemm` | 6.13, T20's loop flags, `loop_unroll2` | winner's main loop on 6.13 |
| T32 to T40 | `trial_t31_no_<flag>_mxfp4_gemm` | T31 with one feature off (and the flags that need it) | which part of T31 matters |
| T41 | `trial_misched_on6_14_mxfp4_gemm` | 6.14 without `source_order` | LLVM machine schedulers on, same IR |

Design of each step:

- **6.2 vector epilogue.** The N-repeats of one WMMA column position become
  adjacent D columns: `n = wave_n*128 + n_local*8 + rn`. Each lane then owns 8
  consecutive FP8 outputs per row and writes them with one 8-byte store.
  `beta == 0` takes a branch that never reads C. The FP8 conversion
  saturates like hipBLASLt: +-inf become NaN, finite values clamp to +-448,
  and NaN is preserved by an explicit `fcmp uno` select (`fmax`/`fmin` lower
  to `maxnum`/`minnum`, which drop NaN). The reference has a matching
  `saturate=` mode.
- **6.3 local-read prefetch.** HalfPLR order: before the WMMAs of K-step `s`,
  the kernel reads K-step `s+1`'s scales, A fragments and the first half of
  B. The second half of B is read after the first half of the WMMAs.
  gfx1250 LDS loads are volatile, so source order is the issue order.
- **6.4 wave-parity TDM.** Each operand is split into two row components.
  Even waves load component 0 and odd waves load component 1, so two waves
  issue the operand's TDM loads. The wave id goes through `readfirstlane`
  and both descriptors are built. Each descriptor group is picked with a
  `select`, so the issue has no branch. Step 6.5 uses this same select and
  adds the scale TDM load after it.
- **6.5 scale TDM.** Scales come in TensileLite's gfx1250 `InMemorySwizzle`
  layout `[K/128][MN][4]` (`swizzle_mx_scales_gfx1250`) and are staged to
  LDS by TDM next to the data. The A mapping changes to
  `m = wave_m*128 + m_local*4 + rm`, the YAML's `VectorWidthB=4`. A lane's
  A scales for a K-step are then one 16-byte read, and with 6.2's N mapping
  its B scales are two. That makes 12 scale reads per K tile.
- **6.6 LDS pad.** These pads match the winner's local-read math:
  - rocKE A data: 16 B every 1024 B. This uses the TDM pad fields
    `pad_interval = log2(period/4) - 1` and `pad_amount = pad/4 - 1`.
  - rocKE B data: one TDM transfer in iterate mode. It moves 8 rows per
    iteration, with a 2064 B LDS stride (2048 B plus 16 B of pad).
  - rocKE B scales: 16 B every 256 B, with the region aligned to 512 B.
- **6.7 segment interleave.** The buffer becomes `[B0][A][B1][B scales][A
  scales]`, so the second B component sits past A. The resulting offsets
  are the winner's: A at 33024, B1 at 99584, B scales at 132608, A scales at
  137216, and the second buffer at 141312.
- **6.8 cluster and multicast.** The grid launches as 4x4 clusters. Every
  workgroup in a TDM multicast mask issues the same full-tile load:
  `0x1111 << x` for rocKE B, `0xF << 4y` for rocKE A. A fenceless cluster
  barrier replaces the workgroup barrier. It is the expansion of
  `llvm.amdgcn.s.cluster.barrier`, emitted as inline asm: LLVM expands the
  intrinsic into a real branch (only the first wave signals the cluster
  barrier). That branch splits the main loop into several blocks, and LLVM
  then keeps about 35 LDS reads in flight instead of 62.
- **6.9 expert scheduling.** The kernel is compiled with
  `-mllvm -amdgpu-expert-scheduling-mode`, LLVM's counterpart of the
  winner's `EnableESM2`. The wave sets `SCHED_MODE` to 2 at entry, the
  hardware stops tracking ALU dependencies, and LLVM inserts the
  `s_wait_alu` waits itself (346 in the YAML-shape kernel, none in 6.8).
  The LLVM IR is 6.8's; only the compile option differs.
- **6.10 descriptor hoist.** Like the winner's SGPR descriptors, both
  wave-parity TDM descriptors (data and scales) are built and selected once
  before the loop. Each tile then patches only the LDS buffer address, the
  64-bit global address and the remaining-size words. The loop has no
  `v_readfirstlane`, and its per-tile instruction count drops to the
  winner's.
- **6.11 WMMA order.** Within each half of a K-step the B fragment is the
  outer index and the A fragments the inner one, so consecutive WMMAs share
  their 8-register operand, as in the winner.
- **T1 GL2 prefetch.** The cluster's 16 workgroups split the GL2 prefetch
  of tile `k + 2`'s A, B and swizzled-scale lines, one 256 B line per thread,
  with `th:TH_LOAD_NT scope:SCOPE_SE` as in the winner. The A and B lines
  fill the 2048 threads exactly. The 128 scale lines take a second round, in
  which every thread prefetches line `thread mod 128`. Past K, the last tile
  is prefetched again. Both choices keep the prefetch branch-free, for the
  same reason as the asm barrier. The prefetch is issued ahead of the
  cluster barrier. Both scale prefetches use `global_prefetch` because
  rocKE IR has no global-to-flat pointer cast; the winner uses
  `flat_prefetch` for one of them.
- **T2 InitC WMMA.** The first K tile is peeled. Its first K-step multiplies
  into C = 0 directly, so the kernel never zeroes 512 accumulators with moves.
  The chain clears them once before the loop, so this saves only those
  moves. The extra peeled tile changes how LLVM schedules the unchanged loop:
  it issues 70 LDS reads before a tile's first WMMA instead of 36, and the
  tile start gets longer.
- **T3 mid-tile refill.** The TDM runs two tiles ahead, as TensileLite's
  `PrefetchGlobalRead=2` with two LDS buffers. With HalfPLR, the tile's last
  LDS read is issued during K-step `steps-2`. Before the last K-step's
  WMMAs, each tile waits for its LDS reads, barriers (the cluster barrier
  with multicast), and issues tile `k + 2` into the buffer it just read. At
  the tile boundary it waits for tile `k + 1` with `s_wait_tensorcnt 2`, so
  tile `k + 2`'s two loads stay in flight, then barriers. TensileLite places
  this second sync at unrolled iteration `LoopIters - PLR`, because its
  `PrefetchLocalRead` reads the next tile there. Without those reads, a
  mid-tile wait only adds an exposed barrier.

  The refill sits between `~{memory}` fences and `sched_barrier`s that let
  only ALU work cross. Without them LLVM reads the whole tile before the
  first K-step's WMMAs and refills there, because the WMMAs are
  register-only and the barriers do not access memory. A second such
  barrier follows the first K-step's reads. Without it LLVM mixes K-step
  1's reads in, and the first WMMA waits for 43 reads instead of 21. As with
  `SuppressNoLoadLoop`, the tail has no peel or guard: past K the last tile
  is loaded again into a buffer nobody reads. After the loop a
  `s_wait_tensorcnt 0` (and the cluster barrier with multicast) keeps any
  workgroup from exiting with a TDM write into its LDS in flight. If it is
  combined with T1, the GL2 prefetch moves to four tiles ahead (PGR +
  PrefetchGL2, TensileLite's `skipPGR`), and the prologue prefetches tiles 2
  and 3.
- **T4 CK LDS pad.** CK Tile pads A and B by 16 B after every
  `max(256 B, row)`: one pad per sweep of the 64 LDS banks. With 256 B rows
  that is one pad per row, set through the TDM pad fields on both operands
  (`pad_interval = 5`, `pad_amount = 3`), with no iterate mode. The scales
  stay unpadded, as in 6.5; CK keeps them in registers. The period follows
  the bank count, not the read mapping. 6.5's lanes read every 4th A row
  and every 8th B row, so 16 lanes of a `ds_load_b128` start at only 4
  (A) or 2 (B) distinct 16 B bank slots. The 6.6 pads give 16 distinct
  slots.
- **T6 tile read-ahead.** The sync of T3 moves to the start of the last
  K-step: the tile waits for its own LDS reads and for tile `k + 1`
  (`s_wait_tensorcnt 0`, the only TDM load in flight), barriers, and issues
  tile `k + 2`. It then reads tile `k + 1`'s first K-step from the other
  buffer during its last K-step, HalfPLR-style, and the loop carries those
  fragments to the next tile. The tile boundary has no wait or barrier, and
  only the prologue waits with `s_wait_tensorcnt 2`. The last tile reads
  ahead from the buffer holding the clamped reload: valid LDS that is never
  used.
- **T7 / T8 source order.** `-mllvm -enable-misched=false -mllvm
  -enable-post-misched=false`: no machine scheduling, as StinkyTofu at
  `OptLevel: 0`. LLVM's schedulers move WMMAs across the refill's barrier
  despite the `sched_barrier`s. In T6 the waits and the barrier land at
  WMMA 40 and the TDM at WMMA 96. In T7 all of them stay at WMMA 96,
  between the 96th and 97th WMMA, with no spills. The LLVM IR is T6's
  (T8: 6.9's).
- **T9 / T10 split cluster barrier.** The cluster barrier is split into its
  signal and its wait, as in the winner's `OptLevel: 0` loop. Each tile
  waits for its LDS reads and signals `cluster_signal_wmmas` WMMAs after its
  last read, which is issued with the second half of K-step `steps-2`. At
  the last K-step, `s_wait_tensorcnt 0` and a workgroup barrier publish
  tile `k + 1`, whose reads start there. Half a K-step later the cluster
  wait is followed by tile `k + 2`'s TDM. In T9 the signal is at WMMA 86,
  the workgroup barrier at 96, and the wait and TDM at 112, the winner's
  positions. No LDS read is issued between the signal and the workgroup
  barrier. This needs `source_order`, which keeps the WMMAs between the
  phases.
- **T11 / T12 GL2 prefetch placement.** With `PrefetchGL2=2` the winner
  prefetches tile `k + 4` (PGR + PrefetchGL2, TensileLite's `skipPGR`) once
  per tile, after its 16th WMMA. T11 issues the same distance at the
  refill (WMMA 96). T12 moves it after the first K-step's first half (WMMA
  16), so every loop event of T12 sits at the winner's WMMA position.
- **T15 to T20 main-loop features on T12.** T15 and T16 add the 6.10 and
  6.11 features. `coarse_ds_wait` issues one `s_wait_dscnt` per 16-WMMA block that leaves a
  K-step of reads in flight. It is inline asm, so LLVM keeps its own finer
  waits, which are already satisfied when they run. `cluster_signal_wave0`
  is the winner's wave-0 cluster signal without an LDS drain. A workgroup
  barrier after the cluster wait then publishes every wave's reads before
  the TDM. This relies on that barrier, as the winner does.
- **6.12 cluster AID remap.** `cluster_aid_remap` reads the XCC with
  `MSG_RTN_GET_SE_AID_ID` and renumbers clusters as the winner does:
  `l = aid*Nc/2 + (c//8)*4 + xcc%4`, then the AID's half of the cluster
  rows (or columns). Like the winner it assumes cluster `c` runs on XCC
  `c % 8`, and grids that are not whole clusters in groups of eight keep
  the hardware order (so the one-cluster smoke IR is 6.11's).
- **T21 to T23 kernel setup.** `wmma_arb_stall_off` sets `SCHED_MODE` bit 2
  at entry, after the ESM write of bits 0 to 1, as the winner's "Disable
  WMMA arb stall". `kernarg_preload` preloads all 11 kernel arguments
  (17 SGPRs) with `-amdgpu-kernarg-preload-count`, so the prologue has no
  `s_load` before the first TDM; a compile option only.
- **6.13 and T25 to T28 epilogue.** The winner's beta = 0 store path
  (`GW_B0_FD0_VW8_GSU1`) stores 8 FP8 per lane with `buffer_store_b64`, the
  width of rocKE's `global_store_b64`; its `buffer_store_b128` stores belong
  to the GSU workspace path. What differs is the FP8 store sequence and the
  instruction prefetch. `epilogue_nan_clamp` clamps with
  `llvm.minimum(llvm.maximum(v, -448), 448)`, one NaN-propagating
  `v_maximumminimum_f32` per value, instead of `v_med3` plus an unordered
  compare and a select. `epilogue_pk8_cvt` converts eight values with one
  `v_cvt_scalef32_pk8_fp8_f32` (scale 1.0) instead of four
  `v_cvt_pk_fp8_f32`. 6.13 takes both; T25 and T26 try each alone on
  6.12. `epilogue_inst_prefetch` issues the winner's
  `s_wait_xcnt 0; s_prefetch_inst_pc_rel 0, null, 31` (the next 4 KiB)
  before every N-th vector store. The dispatcher fetches only the first
  32640 B (`INST_PREF_SIZE` 255), and LLVM lays out the beta = 0 path after
  the beta != 0 one, so on 6.12 it sits entirely past that window. The
  winner places its prefetches every 4 KiB of layout; IR has no byte
  offsets, so the trial spaces them by store count, and six stores stay
  under 4 KiB on the timed beta = 0 path of T27 and T28.
- **T29 two-tile loop.** The winner's loop body holds three tiles, so each
  tile's LDS buffer is fixed. `loop_unroll2` runs tile pairs, buffer 0 then
  buffer 1, so the buffer is a constant and no parity is carried. The pair
  count comes from `spec.K`: an odd count of loop tiles peels one tile in
  front, and the kernel needs the runtime K to equal `spec.K`. A runtime
  remainder loop after the pairs loop instead spilled 146 VGPRs. Per tile
  the loop drops from 479 to 444 instructions with no VALU and no `v_nop`,
  and the closest accumulator reuse rises from 18 to 32 WMMAs, as in the
  winner. The `s_wait_dscnt` count and the zero read-to-use gap at the
  tile boundary do not change: they come from not reading ahead. T29 is
  slower than 6.13.
- **6.14 and T30 to T40 main loop.** A thread trace of one wave shows 6.13
  losing time to waits at the tile boundary. T20's loop waits about as
  little as the winner, but the runtime buffer parity adds LDS address
  math between the WMMAs, and those VALU instructions cost more than the
  waits they replace. `loop_unroll2` on the refill-mid loop (T30) removes
  that math: VALU per tile drops from 33 to 12, with T20's wait placement
  unchanged. T30 is faster than 6.13 where T20 alone is slower, and T31
  adds the 6.13 steps on top. Leave-one-out runs on T31 (T32 to T40) show
  that `source_order`, `prefetch_gl2`, `loop_unroll2` and `tile_read_ahead`
  are each needed, the largest cost coming from dropping `source_order`.
  The cluster signal helps only with `cluster_signal_wave0`. `init_c_wmma`
  and `coarse_ds_wait` are within run noise. Dropping `gl2_prefetch_early`
  (T37) is faster than T31. Both run the same GL2 prefetch instructions;
  T37 issues them at the refill, 96 WMMAs into the tile, instead of 16
  WMMAs in. 6.14 is T37.

Host resources at M=4096, N=4096, K=65536 (no VGPR/SGPR spills and no scratch
for any step except T6, which spills 2 VGPRs, 6.14, T37 and T38, which spill
8, and T40, which spills 16):

| Step | VGPR | LDS |
| --- | --- | --- |
| 6.1 | 735 | 262144 |
| 6.2 | 699 | 262144 |
| 6.3 | 682 | 262144 |
| 6.4 | 813 | 262144 |
| 6.5 | 805 | 278528 |
| 6.6 | 815 | 282624 |
| 6.7 | 815 | 282624 |
| 6.8 | 819 | 282624 |
| 6.9 | 819 | 282624 |
| 6.10 | 838 | 282624 |
| 6.11 | 854 | 282624 |
| 6.12 | 854 | 282624 |
| 6.13 | 854 | 282624 |
| 6.14 | 864 | 282624 |
| T1 | 826 | 282624 |
| T2 | 1008 | 282624 |
| T3 | 942 | 282624 |
| T4 | 819 | 294912 |
| T5 | 942 | 282624 |
| T6 | 1024 | 282624 |
| T7 | 860 | 282624 |
| T8 | 811 | 282624 |
| T9 | 860 | 282624 |
| T10 | 860 | 282624 |
| T11 | 868 | 282624 |
| T12 | 876 | 282624 |
| T13 | 776 | 282624 |
| T14 | 858 | 282624 |
| T15 | 860 | 282624 |
| T16 | 876 | 282624 |
| T17 | 876 | 282624 |
| T18 | 876 | 282624 |
| T19 | 860 | 282624 |
| T20 | 860 | 282624 |
| T21 | 854 | 282624 |
| T22 | 854 | 282624 |
| T23 | 854 | 282624 |
| T24 | 838 | 282624 |
| T25 | 854 | 282624 |
| T26 | 854 | 282624 |
| T27 | 854 | 282624 |
| T28 | 854 | 282624 |
| T29 | 834 | 282624 |
| T30 | 876 | 282624 |
| T31 | 876 | 282624 |
| T32 | 860 | 282624 |
| T33 | 860 | 282624 |
| T34 | 796 | 282624 |
| T35 | 876 | 282624 |
| T36 | 880 | 282624 |
| T37 | 864 | 282624 |
| T38 | 870 | 282624 |
| T39 | 876 | 282624 |
| T40 | 812 | 282624 |
| T41 | 836 | 282624 |

Host tests pin one LLVM-IR fact per step and check the LDS layout model.
`MxFp4LdsLayout` is the single address model used by the builder. The test
checks it against an independent simulation of the TDM writes (pad and
iterate semantics), and also checks that no writes overlap and that all
writes stay inside the buffer.

Known differences from the winner, kept on purpose:

- Both scale prefetches use `global_prefetch`, as noted under T1.

StinkyTofu / SIA4 schedule fidelity (instruction interleave and
assembly-level tuning) stays out of scope for Step 6. The base variant is
byte-identical between the Python and C++ engines; the C++ builder rejects
every Step 6 flag.

A four-way feature comparison against CK Tile `comp_tdm` / `comp_tdm_v2` and
the hipBLASLt YAML lives in `my_experiment/MXFP4_PIPELINE_COMPARISON.md`
(repo root).
