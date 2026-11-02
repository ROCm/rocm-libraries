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

Host resources at M=4096, N=4096, K=65536 (no VGPR/SGPR spills and no scratch
for any step):

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
| T1 | 826 | 282624 |
| T2 | 1008 | 282624 |
| T3 | 942 | 282624 |
| T4 | 819 | 294912 |

Host tests pin one LLVM-IR fact per step and check the LDS layout model.
`MxFp4LdsLayout` is the single address model used by the builder. The test
checks it against an independent simulation of the TDM writes (pad and
iterate semantics), and also checks that no writes overlap and that all
writes stay inside the buffer.

Known differences from the winner, kept on purpose:

- The FP8 pack uses two `cvt_pk_fp8_f32x4` calls instead of one 8-wide pack.
- NaN is kept with a select instead of `v_maximumminimum`.
- Both scale prefetches use `global_prefetch`, as noted under T1.

StinkyTofu / SIA4 schedule fidelity (instruction interleave and
assembly-level tuning) stays out of scope for Step 6. The base variant is
byte-identical between the Python and C++ engines; the C++ builder rejects
every Step 6 flag.

A four-way feature comparison against CK Tile `comp_tdm` / `comp_tdm_v2` and
the hipBLASLt YAML lives in `my_experiment/MXFP4_PIPELINE_COMPARISON.md`
(repo root).
