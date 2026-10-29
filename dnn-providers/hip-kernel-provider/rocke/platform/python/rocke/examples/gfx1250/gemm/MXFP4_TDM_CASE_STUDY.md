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
`rocke.instances.gfx1250.mxfp4_tdm_gemm_variants`. Each landed substep adds a
cumulative `step6_N_mxfp4_gemm` entry. Use
`python -m rocke.examples.gfx1250.gemm.mxfp4_tdm_gemm --variant all` for FFM
checks and
`python -m rocke.examples.gfx1250.gemm.mxfp4_tdm_gemm_variants_bench` for the
silicon bench harness (timings outside the repo).

| Variant | Flags on | FFM |
| --- | --- | --- |
| `base_kernel_mxfp4_gemm` | (none) | `bad=0` (Step 4/5) |
| `step6_1_mxfp4_gemm` | `local_read_prefetch` | `bad=0` |
| `step6_2_mxfp4_gemm` | + `prefetch_gl2` | `bad=0` |
| `step6_3_mxfp4_gemm` | + `scale_double_buffer` | `bad=0` (YAML `256x256x512`: `vgpr=774`, `vgpr_spill=4`, scratch=20; smoke tile_k=512 spill-free) |
| `step6_4_mxfp4_gemm` | + `wave_separated_tdm` | `bad=0` |
| `step6_5_mxfp4_gemm` | + `scale_tdm` (row-major rocKE LDS staging, not Tensile swizzle) | `bad=0` (YAML LDS 278528, `vgpr=768`, spill=0) |
| `step6_6_mxfp4_gemm` | + `lds_pad` (+16 B A/B row stride) | `bad=0` |
| `step6_7_mxfp4_gemm` | + `lds_segment_interleave` (`[A0][B-full][A1][scales]` bcontig layout) | Host/IR coverage: YAML MacroTile is valid; on-silicon numeric check pending |
| `step6_9_mxfp4_gemm` | + `tdm_plus_lds_buf` (auto depth 3 when it fits) | Host/IR coverage; the bcontig YAML buffer stays depth 2 |
| `step6_10_mxfp4_gemm` | + `init_c_wmma` (peeled first K tile reads C=0; the loop has no select) | Host/IR coverage |
| `step6_11_mxfp4_gemm` | + `cluster_dim=(4,4,1)` (needs grid multiple; smoke grows M/N) | Host/IR coverage |
| `step6_12_mxfp4_gemm` | + `b_scale_n_major` (B_scale is `[N][K/32]`, like A_scale) | Host/IR coverage; on-silicon numeric check pending |
| `step6_13_mxfp4_gemm` | + `tdm_multicast` (cluster `workgroup_mask` on A/B/scale TDM, cluster barrier per tile) | Host/IR coverage; on-silicon numeric check pending |
| `step6_14_mxfp4_gemm` | + `cross_tile_prefetch` (barrier and refill before the last K-step's WMMAs) | Host/IR coverage (YAML tile `vgpr=1018`, spill=0); on-silicon numeric check pending |
| `step6_15_mxfp4_gemm` | + `cluster_gl2_prefetch` (cluster threads split the GL2 prefetch of the cluster's A/B tiles) | Host/IR coverage; on-silicon numeric check pending |
| `step6_17_mxfp4_gemm` | + `a_scale_lds_pad` (8 B LDS pad per 256 B of A scales, via the TDM pad fields) | Host/IR coverage (YAML tile `vgpr=1016`, spill=0); on-silicon numeric check pending |

`cluster_gl2_prefetch` follows TensileLite's GL2Prefetch: the cluster
consumes `cy` A tiles and `cx` B tiles, and all `cx·cy·block_size` threads
split their rows in 256 B chunks, A first and then B. On the YAML cluster each
thread issues one prefetch per tile, choosing the A or B address with a select
instead of a branch. Scale tensors are not prefetched.

`a_scale_lds_pad` pads only the A-scale tile. An A-scale read is one
`ds_load_b32` per lane, and lane `l` reads row `atom_row + l`, with rows
16 B apart. Rows `r` and `r + 16` are then 256 B apart and share a bank, so
the 32 lanes collide in pairs. An 8 B pad after every 256 B moves rows 16–31
by two banks onto banks the first 16 rows do not use. The TDM descriptor
inserts the pad (`pad_interval` 256 B, `pad_amount` 8 B), and the read adds
`(offset >> 8) * 8`.

The A/B data keep the per-row pad from `lds_pad`. TensileLite pads A every
2048 B and B every 1024 B, but rocKE's fragment reads put 16 lanes on 16
consecutive rows at the same K offset. Only a pad on every row gives those
16 rows distinct banks; TensileLite's gfx1250 bank model scores its block
sizes on that pattern as 8-way (A) and 4-way (B) conflicts.

`tdm_multicast` sets each TDM descriptor's `workgroup_mask` from
`cluster_workgroup_id`. A is shared along a cluster row (x is the N tile) and
B along a cluster column, so each tile can be fetched once per cluster. Peers
then write into this workgroup's LDS, which is why the per-tile barrier becomes
`cluster_barrier(fences=False)`. LDS is ordered by `s_wait_tensorcnt`, so the
cluster-scope fences (and the `global_inv` they cost) are left out.

`cross_tile_prefetch` runs the tile-boundary wait, barrier and TDM refill after
the last K-step's operands are read, but before its WMMAs. It then reads the
next tile's first-step A/B fragments and scales, which are carried through the
loop. Compiler memory clobbers on both sides of the barrier keep LLVM from
moving LDS reads across it.

`b_scale_n_major` changes the B_scale input contract from `[K/32][N]` to
`[N][K/32]`. Scale TDM then stages B scales as `tile_n × (tile_k/32)` bytes,
so each lane's four E8M0 groups are one aligned 4-byte LDS load instead of
four byte loads plus packing. The LDS footprint is unchanged.
`kernel_inputs()` in `mxfp4_tdm_gemm.py` transposes the host array for these
kernels; `mxfp4_tdm_gemm_variants_bench.py --with-flag b_scale_n_major`
applies the flag to any variant.

Step 6.8 (`lds_segment_b_rotate`, odd-wave B N±8) is dropped from the
cumulative chain and listed in `MXFP4_GEMM_BLOCKED_VARIANTS`: rotate needs a
split-B layout, while bcontig keeps B contiguous. The resolved TensileLite
winner for the YAML problem also uses `LDSSegmentBRotate=0`. Steps 6.9–6.11
build directly on Step 6.7.

Step 6.7 does not require a whole prefetch buffer to fit one 64 KiB segment.
For the supported `2×2` wave group, it splits A into two M components and
places B as one contiguous block between them. A component placement keeps the
two A read groups in separate segments; the full double-buffered LDS allocation
is checked against the 320 KiB workgroup LDS capacity. This is rocKE's
row-padded bcontig layout, not a byte-identical implementation of TensileLite's
block-padded `InMemorySwizzle` layout.

## Deferred controls

Step 6 lands hipBLASLt SIA0 features one default-off flag at a time (HalfPLR,
PrefetchGL2, scale buffering/TDM, wave-separated TDM, LDS layout, third buffer,
InitC WMMA, cluster). StinkyTofu / SIA4 schedule fidelity stays out of scope.

A four-way feature comparison against CK Tile `comp_tdm` / `comp_tdm_v2` and
the hipBLASLt YAML lives in `my_experiment/MXFP4_PIPELINE_COMPARISON.md`
(repo root).
