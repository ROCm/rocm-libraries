# Known issues: gfx1250 MXF4 subtile

## Correctness failures outside the common-test shapes (open, found 2026-09-30)

Reproduce on GPU 2 (found on GPU 3):

```bash
TENSILE_YAML=configs/regress_subtile_mxf4.yaml ./run.sh tensile
```

The config has the two kernels from `Tensile/Tests/common/gemm/gfx12/subtile_mxf4_gfx1250.yaml`
(TN, e8 scales, block 32, DepthU 256, PGR 2, StreamK 3, TDMInst 3), with beta = 1 and random C.
Sizes are (M, N, batch, K).

| Kernel | Size | Result |
| --- | --- | --- |
| MT128x64, MI 32x16, WT 2x2 | (128, 64, 1, 256), (128, 64, 1, 512), (256, 128, 1, 256), (96, 48, 1, 256), (128, 64, 1, 384) | pass |
| | (1000, 500, 2, 768) | wrong values, 7713 of 1000000 |
| | (128, 64, 3, 4096) | wrong values, 6141 of 24576 |
| | (200, 100, 1, 1152) | wrong values, 720 of 20000 |
| MT256x256, MI 32x16, WT 4x8 | (4096, 4096, 1, 8192), (256, 256, 1, 256), (512, 512, 1, 2048), (2048, 1024, 1, 384) | pass |
| | (1000, 700, 1, 1280) | wrong values, 25445 of 700000 |
| | (300, 200, 2, 1152) | wrong values, 13786 of 120000 |
| | (256, 256, 1, 8192) | illegal memory access (GPU hang, aborts the client) |

**Update 2026-10-07 (`logs/*-gen-v6`, GPU 0):** every wrong-value size above passes except
(128, 64, 3, 4096), with `mx-scale-jitter=4`. The crash at (256, 256, 1, 8192) is still open.
The cause was the NonEdge store of the last partial M tile. With MI 32x16 in wave32 and fp32
output, each lane stores 16 consecutive rows of one column (four `b128`) under one exec mask:
lanes 0-15 hold rows 0-15 of the 32-row block and lanes 16-31 rows 16-31.
`_emitAlign8ExecMask` built the mask as `-1 >> (32 - validRows)`, one lane per row, and
`checkIsEdgeSubtile` sent any remainder that is a multiple of 4 to that path. At M = 300 the last
block has 12 valid rows: columns 0-11 stored all 16 rows (4 rows spill into the next column) and
columns 12-15 stored none. The check now requires a multiple of 16 for this layout, so other
remainders take the per-element edge path. A per-store mask (one per `b128`) would keep 4-row
remainders on the fast path. `validate.py LOGDIR [SIZE ...]` reruns a build at given sizes.

**Update 2026-10-07, StreamK (`logs/*-gen-v7`, GPU 0):** all 15 regression sizes pass, including
(128, 64, 3, 4096) and (256, 256, 1, 8192). Both failures were the StreamK partials write
(`partialsWriteProcedure` / `partialsWriteBatch`), taken only when a tile's K range is split across
workgroups (few tiles, long K).
- The write first copies the accumulators into the element registers that `StoreState` checked
  out of the VGPR pool. Those indices are absolute, but the subtile branch wrote the copy to
  `ValuC + idx` and stored from `idx + startVgprValu`, both 64 registers too high
  (`vgprValuC = 64` on gfx1250).
- So the copies overwrote live accumulators, and the last 32 targeted v1024+ (`s_set_vgpr_msb 257`,
  "dst: 4"). That wraps to v0-v31 and clobbers `Serial` and the StreamK state in v1-v7.
- The WS store offset `v15 = Serial * 64` was then accumulator data, so the stores landed 0.7-2 GB
  past the 128 MiB workspace (TCP write page fault, GPU hang).
- Fix: use the absolute index on both sides, the same as the fixup read and the D store. The branch
  was neutral on gfx950, where `startVgprValu` is presumably 0.
- Found with `rocgdb` (`HSA_ENABLE_PRECISE_MEMORY=1`, `info registers` at the faulting store).
  `asm_drop.py` (comment out matching lines and reassemble) did the earlier bisection.
- (256, 512, 1, 16384) shows 1 of 131072 values off by 0.004 (-2.23828 vs -2.24219), a
  reduction-order rounding difference from the split K.

- Not caused by the 32x16 accumulator row fix-up (`emitWmma32x16AccRowFixup`). Tensile built
  from `e91596bfb48`, before that fix, has the same pass/fail result on every size. The count of
  wrong values differs slightly between the two builds.
- Every batch > 1 size fails. Some single-batch sizes with partial M/N tiles and K of 1152 or 1280
  also fail. Edge-only (96, 48, 1, 256) and tail-only (128, 64, 1, 384) sizes pass.
- The crash is one 256x256 tile with a long K, so StreamK 3 splits a single tile's K range
  across workgroups (root cause in the StreamK update above).

## DepthU = 512 (enabled 2026-10-07)

`configs/du512_timing.yaml` (MT256x256x512, `LDSSegmentInterleave: [0]`) passes the 10 default
`validate.py` sizes plus 7 more (K tails, StreamK split, batch 2, partial tiles) with
`mx-scale-jitter=4`. On GPU 0 at (4096, 4096, 1, 65536) it runs in 206.4 us vs 223.0 us for
DepthU 256 (gen-v7), 8.0% faster, 5 of 5 paired rounds. LDS is 304 KiB of 320.

Five gaps, all hidden at DepthU 256 because there is only one K-subtile and the scale tile
happens to be 2 KiB:
- Rejected by "Unable to load MXSA scales using one load per wave" (`Solution.py`). The check
  is for buffer-load scales (16 B per thread, 2 KiB at 128 threads, against 4 KiB at DepthU 512). TDM
  moves the whole tile, so it now applies only when the scales are not TDM-loaded. `MXLoadInst`
  is still "Auto" at that point, so TDM is inferred from `TDMInst != 0`.
- `LDSSegmentInterleave=2` cannot fit: A alone is 64 KiB plus padding, more than one segment.
  It used to assert in `KernelWriter._initKernel`. The segment oracle now rejects it with a
  reason. Two padded buffers (about 136 KiB each) fit only in the tight layout.
- Assembly failed: the tail-loop "tighten Srd{MXSA,MXSB}+2" (`computeTailLoopSrdLimit`, only for
  DepthU > 256) referenced `SrdMXSA`, which does not exist when scales are TDM-loaded. The TDM
  filter now drops MXSA/MXSB as well as A/B.
- Wrong values on every size: `emitSingleDsRead` computed the second K-subtile's LDS offset
  as `sId1 * subtileShapeK * instK * int(bpe)`, and `int(0.5) = 0` for fp4. So K-subtile 0 was
  read twice. The offset is now truncated after the multiply (+128 B).
- Wrong values in M rows 128-255 (only with varying scales): the MXSA/MXSB LDS regions were
  sized `loadWidthGR * 32 * numWaves = 2 KiB`, but the scale tile is (DepthU / 32) * MT = 4 KiB.
  MXSB overwrote the upper half of MXSA. Each region now holds the whole tile.

Profile vs earlier builds (GPU 0, (4096, 4096, 1, 65536); `pmc_compare.sh 2 ...` for the counters,
medians over dispatches, logs `logs/*-pmcc-*`; `alt_timing.py 6 ...` for time):

| Build | LDS bank | LDS addr | Segment stall (share of CU cycles) | Time (us) |
| --- | --- | --- | --- | --- |
| 10-05 final (`20261005-190611-final-gen`) | 0 | 0 | 0.123 M (0.085%) | 270.77 |
| gen-v4 (TDM scale fixes) | 0 | 0 | 0.097 M (0.089%) | 224.20 |
| gen-v5 (swap hoist) | 0 | 0 | 0.091 M (0.084%) | 222.82 |
| gen-v5, `LDSSegmentInterleave: 0` | 0 | 0 | 24.55 M (22.9%) | 223.26 |
| gen-v6 (NonEdge store fix) | 0 | 0 | 0.091 M (0.085%) | 222.65 |
| gen-v7 (StreamK partials fix) | 0 | 0 | 0.094 M (0.087%) | 222.63 |
| DepthU 512, `LDSSegmentInterleave: 0` | 0 | 19 | 15.45 M (15.9%) | 207.54 |

- The DepthU 256 changes added no bank, address or segment conflicts.
- DepthU 512 has no bank conflicts, but it brings segment conflicts back because mode 2 does not
  fit (above). That is still fewer than DepthU 256 in the same layout.
- At DepthU 256 the same 23% stall share costs 0.4 us (si0 vs si2), so these stalls are mostly
  hidden. Whether that holds at DepthU 512 is not measured: no conflict-free DepthU 512 layout
  fits in 320 KiB with two buffers.

## Reference-kernel ideas applied to DepthU 512 MXF4 (2026-10-14)

All on GPU 1, (4096, 4096, 1, 65536), `alt_timing.py` paired against the DepthU 512 baseline
(`configs/du512_timing.yaml`, 202-203 us). Each idea was prototyped as an asm patch on the
generated kernel, then the winners were moved into Tensile.

| Idea (patch script) | Result |
| --- | --- |
| TDM multicast across a cluster (`ClusterDim`, StreamK 3 DP-only or StreamK 0) | 1.23x; the only large win |
| A/B address + descriptor SALU above the FREE barrier (`patch_tdm_hoist.py`) | +0.7% on multicast, +0.3% alone |
| One wave per tensor issues the whole tile (`patch_tdm_owner.py`) | 1% slower |
| Spreading the four TDM issues through the loop (`tdm_spread.py`) | neutral alone, 3% slower on multicast |
| Scale loads issued early | 2.3% slower |
| WMMAs moved above the READY wait (`patch_late_ready.py`, 4-24 WMMAs) | neutral to 1% slower |
| Instruction prefetch at entry (`patch_iprefetch.py`) | neutral |
| `PrefetchGL2` 1 or 2 | 0.9% slower |
| Fewer `s_set_vgpr_msb` (245 to 73 per iteration, `patch_msb_reorder.py`) | 0.5% slower |

Cluster shape matters: [2,4] (B shared by 2 M-peers, A by 4 N-peers) is best at 164.7 us
with StreamK 0, then [2,2] 166.9, [4,4] 167.6, [4,2] 168.8, [1,4] 169.6; [8,1] is slower than
no cluster (207 us). StreamK 3 is within 1 us of StreamK 0 for the same shape.

### Upstreamed

- `StreamK.py preLoop`: the StreamK multicast prologue arrive is skipped for subtile kernels.
  Subtile already emits a balanced signal/wait pair per section, so the extra arrive was never
  consumed and leaked into the next launch's -3 barrier phase. StreamK 3 cluster kernels hung
  after 2-10 back-to-back launches. Test: `test_streamk_multicast.py::TestSubtileClusterBarrierBalance`
  (config `_designed/gfx1250/subtile_mxf4_sk3_cluster.yaml`), fails without the fix.
- `LogicalScheduler.hoist_tdm_setup` (new pass, on when A and B are both TDM): `gr_inc` and
  the A/B descriptor refresh (`TDMRefreshOp`) move ahead of the GR chain's barrier. The scale
  descriptor refresh stays after the A/B loads because those SGPRs alias the A/B descriptors.
  Test: `test_SubtileBasedLogicalScheduler.py::TestHoistTDMSetup`.

Generated by Tensile with both changes, `configs/du512_cl24.yaml` (StreamK 3, DP-only,
`ClusterDim [2,4]`) passes the 10 `validate.py` sizes and runs 164.2 us, 1.237x over the
baseline in 6 of 6 rounds (hand-patched equivalent: 163.6 us). The baseline config with the
hoist alone is 202.8 vs 203.1 us (within noise).

### ATT: where the multicast time went

`att_loop.py` over rocprofv3 ATT stats at (4096, 4096, 1, 8192), cycles per main-loop iteration
per wave (latency + idle):

| | output3 | baseline | [2,4] + hoist |
| --- | --- | --- | --- |
| WMMA | 1500 | 1778 | 1783 |
| `ds_load` | 272 | 420 | 305 |
| `tensor_load_to_lds` issue | 419 | 382 | 305 |
| `s_wait_tensorcnt` | 284 | 253 | 3 |
| `s_wait_dscnt` | 271 | 262 | 228 |
| workgroup barrier wait | 142 | 94 | 33 |
| cluster barrier wait | 0 | 0 | 5 |
| total | 2999 | 3317 | 2807 |

Multicast removes the TDM wait almost completely and most of the READY-barrier wait; the
cluster barriers cost 5 cycles. WMMA is now 63% of the loop, so the next gains are in WMMA
issue (dependency and reuse stalls), not in data supply. (output3 is a longer-K trace of an
earlier build; its totals are not directly comparable.)

### Open: the generated cluster protocol has a latent race

The loop's cluster wait guards LDS buffer i%2, but iteration i+1's multicast writes buffer
(i+1)%2, which a slow peer may still be reading. No failure was observed (10/10 validation,
long timing runs). `patch_cluster_safe.py` moves the cluster signal/wait to right after the
FREE barrier, which closes the window but costs 2.8% (168.5 us, still 1.204x). Not upstreamed;
it is a trade-off to decide.

## DepthU 512 main loop: TDM supply is the limit (open, found 2026-10-08)

All on the DepthU 512 build (`logs/20261007-205701-gen-du512`), (4096, 4096, 1, 65536), GPU 3
unless noted.

### Measure in cycles, not microseconds

The part is power-throttled, and its clock depends on the data. The unmodified kernel runs the
same 2.98 M GPU-active cycles with zero A/B as with random data, but at 2.21 vs 1.84 GHz
(168 vs 203 us). Uniform scales change nothing. Diagnostic builds that drop loads leave stale or
near-zero data in LDS, so their microsecond timings mix the code change with a clock change.
`cycles.sh name=dir ...` reports `GRBM_GUI_ACTIVE` cycles and duration per build. The noise is
about 0.4%: two baseline runs measured 3.022 and 3.009 M.

### Thread trace works with the default target CU

`rocprofv3 --att -d DIR -- tensilelite-client --config-file INI` (defaults: target CU 1,
shader engine 0) records instruction timing. `configs/rocprof_att.yaml` (and so `run.sh att`)
gives empty `.att` files, even with `att_target_cu: 1`, shader engine mask 0x1 and no
`LD_LIBRARY_PATH` override. The 1 GiB buffer, `att_consecutive_kernels: 1`,
`kernel_include_regex` and `--truncate-kernels` all work from the command line. Suspected but
untested: `att_gpu_index: "0"` selects physical GPU 0, because `HIP_VISIBLE_DEVICES` does not
renumber the runtime's agents. Use the command line. Stall-only sums miss part of the
time. A wave that cannot issue at all is counted under `Idle`, so per-iteration time is
`Latency + Idle` of the loop instructions. Shader engines 0-3 and CUs 1/3/5 all show the same
breakdown (2360-2480 cycles of latency per iteration), so CU 1 is representative.

### Where an iteration goes

About 3076 cycles per iteration (latency + idle), 128 WMMAs. The four TDM loads are issued in a
burst right after barrier 1 (after WMMA 95). Per iteration:

| Point | Cycles |
| --- | --- |
| A, B, MXSA `tensor_load_to_lds` issue (idle) | 10 / 10 / 17 |
| MXSB `tensor_load_to_lds` issue (idle: the wave cannot issue it) | 373 |
| `s_wait_tensorcnt 4` | 309 |
| Barrier 2 (after the wait) | 189 |
| The four `s_wait_dscnt 0` (LDS reads of each K-step) | 257 (26 / 75 / 72 / 83) |
| WMMA dependency stalls | 246 |

Roughly 870 cycles (28%) are tied to TDM. Moving the TDM issues does not reduce that total
(`tdm_spread.py`, results correct, 10 default sizes pass, no failures with jitter):

| Build | MXSB issue | Wait | Barrier 2 | Iteration | Kernel cycles |
| --- | --- | --- | --- | --- | --- |
| Burst after barrier 1 (current) | 373 | 309 | 189 | 3076 | 3.022 M |
| spread4: A/B/MXSA/MXSB after WMMA 95/103/111/119, wait 1 | 9 | 755 | 162 | 3066 | -0.3% |
| spreadMX: A, B at 95, MXSA 107, MXSB 119, wait 2 | 11 | 677 | 226 | 3051 | -1.4 to -1.8% |
| Wait relaxed to 12 (results wrong) | 470 | 2 | 15 | 3011 | -2.5% |

The issue stall moves into the wait, and a relaxed wait moves back into the issue stall.
TDM cannot deliver an iteration's data (2 x 64 KiB + 2 x 4 KiB per workgroup) in under about
3000 cycles, about 45 B/cycle per CU, or about 22 TB/s of L2-to-LDS across the GPU. Scheduling
changes only move the waiting around.

Not yet explained: dropping both scale TDM loads (6% of the bytes, 10% of the L2 requests)
saved 26% of cycles, dropping MXSB alone 3.7%, dropping B 29% and all loads 36% (GPU 0, cycles,
results wrong). That is not proportional to bytes, so the limit may be the number of TDM loads or
L2 requests rather than bandwidth. L2 counters (`GL2C_*`, three passes because one pass exceeds
the hardware) show the scale loads read exactly their bytes (4 sectors per 128 B request, no
waste), and L2 hit rates are high everywhere. Removing the scale LDS padding (384 B after each
128 B row) or reordering the scale loads changed nothing.

### Scale LDS reads

Each K-step issues 32 `ds_load_b128` (data) and 12 `ds_load_b32` (4 MXSA, 8 MXSB), then
`s_wait_dscnt 0`. The scale reads are about half of that wait. Removing them all cuts the four
waits from 257 to 138 cycles per iteration. Replacing them with 3 `ds_load_b128` per K-step (same
registers and bytes, 48 to 12 instructions per iteration) cuts the waits to 192. In both cases
most of the saving moves into the TDM wait and barrier. Kernel cycles: -1.3% (wide), 0.0% (all
removed). Worth doing once the TDM limit moves. A real implementation needs a scale LDS layout
with each lane's four K-chunks contiguous ([M block][row][K chunk][4 B] instead of
[M block][K chunk][row][4 B]) and about 4x the live scale VGPRs.

**VGPR headroom re-examined (2026-10-13), "zero headroom" was wrong.** Both DepthU 256 and 512
builds report `.amdhsa_next_free_vgpr 1024`, which looked like a hard "+24 VGPR, no room"
blocker for the wide-read idea above. A `PrintRP` trace (`KernelWriter.py`'s
`self.db["PrintRP"]`, temporarily set, reverted after) showed the 1024 ceiling is not from
main-loop register pressure: `RegisterPool.size()` (`register.py`) is a monotonic high-water
mark for the whole kernel generation (prologue through epilogue), and the two overflow events
that push it to 1024 both come from `setOccupancy`/`refineOccupancy`'s GlobalWrite-epilogue
`growPool(..., "grow-pool for GlobalWrite")` calls (`KernelWriterAssembly.py` ~16777-16984), not
the main loop. gfx1250 also never runs the pass that would tighten `.amdhsa_next_free_vgpr` to
the true liveness peak (`updateOccupancyFromMaxVgpr`'s `ArchAccUnifiedRegs` check only fires on
gfx90a/gfx942/gfx950). Separately, `reserveMxScaleVgprs`/`numScaleTileVgprs` (`KernelWriter.py`
11873-11896) already reserves `localMMATileGrid[0] * localMMATileGrid[1]` VGPRs per scale
tensor as a bank-0 hole, purely to satisfy the gfx1250 VGPR-MSB bank constraint -- 16 for MXSA
+ 32 for MXSB = 48 total, already carved out of today's budget, of which only 24 are live
destinations (the scheduler ping-pongs 2-deep across K-steps). So the wide-read idea's "+24
VGPR" fits inside already-reserved, half-idle headroom; it needed a scheduling change, not a
bigger pool.

**TDM hardware ruled out for the wide read (2026-10-13), by encoding, no GPU test needed.**
Re-read `TensorDataMover.py`'s descriptor fields: Group1 has exactly one stride word
(`setStride0`, GLOBAL-side only), so there is no independent per-dimension LDS stride to
redirect a K-major global stream into a lane-major LDS layout. The LDS padding mechanism
(`setPadding`/`calPadInterval`) only inserts bytes after a run of `ldsBlockSizePerPad` bytes --
monotonic, one-directional, cannot reorder or transpose a stream -- and `calPadInterval`
additionally requires `ldsBlockSizePerPad // 4` to be a power of 2 with a minimum of 2 dwords
(8 bytes), ruling out the 4-byte-granularity interleave a true `ds_load_b128` consolidation
would need. The originally-planned minimal GPU probe (patch MXSA's padding fields on a
known-good `kernel.co`) was dropped as moot once this fell out of static analysis -- never run.

**`ScaleLRFullResidency` implemented and GPU-validated (2026-10-13).** Opt-in kernel param
(`ScaleLRFullResidency`, default 0): gives SA/SB each `(group, K-chunk)` its own persistent
VGPR tile for the whole unroll iteration, using the already-reserved 48-VGPR
`mxScaleVgprReserve` block, instead of the current 2-deep ping-pong reuse. Pure
scheduling/allocation change in `LogicalScheduler.py`'s `_assign_vgpr_tiles_deterministic`
(new `_num_sets_for`, replacing the hardcoded 2-way `_tile_set_idx`) plus plumbing in
`Kernel.py`, `GlobalParameters.py`, `ValidParameters.py` -- zero new instructions, zero LDS
writes, zero TDM descriptor changes. CPU-only codegen (`configs/du512_scalefullres.yaml`)
confirmed 48 distinct, non-aliased `ds_load_b32` destinations (v38-v85, all 12 scale groups x
4 K-steps) with `.amdhsa_next_free_vgpr` unchanged at 1024, exactly as predicted. `validate.py`
on GPU 2, 10/10 default sizes, `mx-scale-jitter=4`: all PASSED (`logs/20261013-151009`). This is
the VGPR-residency *prerequisite* only -- removing the 2-deep aliasing so the now-non-aliased
reads *can* be clustered. The actual clustering (moving the SA/SB narrow reads to issue right
after the TDM wait instead of spread across subIterK 0-3, to realize a wait-reduction) is not
yet implemented.

**Scale reads clustered after `ScaleLRFullResidency` (2026-10-13), small real win.** With the
2-deep ping-pong gone, SA/SB's current-iteration reads no longer need to stay 1-ahead-of-MFMA
staggered across subIterK 0-3 -- `_place_LRs_for_partition` (`LogicalScheduler.py`) now pins
every same-iteration SA/SB `LRPlacement` to subIterK slot 0 (right after the first barrier,
alongside the K=1 reads already there) when `ScaleLRFullResidency` is set and there is a single
partition; the cross-iteration wrap prefetch chunk is untouched. Dependency annotation
(`annotate_deps`/`_tiles_overlap`) matches producer to consumer by K-range, not by this
placement slot, so the existing cross-subIterK wait machinery (the same mechanism that already
staggers A/B's 1-ahead prefetch) picks this up for free -- no new preOp/wait logic needed.
Confirmed in the generated assembly (`logs/20261013-154939` vs `logs/20261013-151009`): all 24
K=2/K=3 scale `ds_load_b32`s move from their old positions (deep in phase 1/phase 2) to right
after the first barrier, next to the K=1 reads; several `s_set_vgpr_msb` bank-select prefixes
collapse since consecutive reads now share a prefix (35771 vs 35795 total lines, -24).
`.amdhsa_next_free_vgpr` unchanged at 1024. `validate.py` on GPU 2, 10/10 default sizes,
`mx-scale-jitter=4`: all PASSED. `alt_timing.py` on GPU 2, 10 alternating rounds,
(4096, 4096, 1, 65536) against both the unmodified baseline and the residency-only build
(`logs/20261013-155226`, `logs/20261013-151009`): 203.02 (baseline) / 202.94 (residency-only,
faster in 6/10) / 202.29 us (clustered, faster in 7/10 vs baseline, mean +0.74 us, sd 1.38) --
about 0.36%, 10831 to 10871 TFLOPS. Small and partly inside the round-to-round noise (sd exceeds
half the mean diff), but consistently positive and free (same instruction count, same VGPRs, no
TDM descriptor changes). Matches the standing finding above that this kernel is TDM-supply
bound and scheduling changes mostly "move the waiting around" rather than remove it -- this one
happened to hide a little more of it. The wrap-prefetch chunk (next iteration's K=0) and the
multi-partition path are untouched; clustering those too is a possible follow-up but likely
same order of magnitude or smaller.

**Why the win is small: clustering added a third barrier (2026-10-13), found statically, no
profiling needed.** Live GPU profiling (ATT and every PMC/`--pmc` mode) is currently blocked
(see "rocprofv3 cannot run against this client" below), so this was found by diffing the
generated main-loop assembly directly instead. In the unmodified baseline and in the
residency-only build (`logs/20261013-155226`, `logs/20261013-151009`), all four TDM-loaded
tensors (A, B, MXSA, MXSB) share **one** combined buffer-swap/reissue barrier, positioned
between `subIterK=2` and `subIterK=3` -- 674 main-loop instructions, 2 `s_barrier_signal`/
`s_barrier_wait` pairs, 5 `s_wait_dscnt`. In the clustered build (`logs/20261013-154939`),
moving the SA/SB reads earlier also pulled their buffer-swap-and-TDM-reissue earlier with them
(the scheduler recognized the old buffer is now fully drained sooner, so it is safe to recycle
sooner) -- but A/B's own reissue stayed at its old, later position. The net effect: the one
shared barrier **split into two separate ones**, for 3 barrier pairs per iteration instead of 2
(666 instructions, 6 `s_wait_dscnt` -- fewer instructions overall from collapsed
`s_set_vgpr_msb` prefixes, but one more full-workgroup rendezvous point). Confirmed by counting
`s_barrier_signal`/`s_barrier_wait`/`s_wait_dscnt` in the `label_LoopBeginL` .. `s_cbranch_scc0
label_LoopBeginL` range of all three builds' `.s` files. Barrier rendezvous has its own fixed
cost (the "Where an iteration goes" breakdown above measured one barrier instance at 189 cycles
on this kernel), so this extra barrier is a plausible explanation for why the measured win
(+0.36%) is small rather than large: the aliasing-removal and front-loaded reads are a real
gain, partly offset by this new synchronization point. Not yet attempted: deferring the scale
buffer swap+reissue back to the same later point A/B already uses (merging back to 2 barriers)
while keeping the reads clustered early, which should be independent -- the reads just need to
happen before use, the swap/reissue can still wait for the shared barrier.

**rocprofv3 cannot run against this client (2026-10-13), unrelated to this work, unresolved.**
Both `rocprofv3 --att` and every `--pmc`/`--input <yaml>` PMC job (via `pmc_compare.sh`,
`pmc_tdm.sh`, `cycles.sh`, direct `rocprofv3` invocation) fail deterministically (9/9+
reproductions) with:
```
amd_comgr_set_data(...) INVALID_ARGUMENT :: isa=(amdgcn-amd-amdhsa--gfx12-5-generic, 0, 0)
terminate: Client must be linked with an embedded library or a library must be specified at runtime.
```
Reproduces identically on an unmodified baseline client, and persists after rebuilding the
client with `-DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK=ON` (`invoke build-client -g gfx1250
--enable-rocprof`) -- so not caused by anything in this round's changes, and not fixed by the
one client-side build flag available. The plain, unwrapped client runs fine with identical
env/args. Looks like a `rocprofiler-sdk`/`comgr` incompatibility with gfx1250's "generic"
code-object target variant on this ROCm build (`/opt/rocm-10.1.0a20260908+bkc.20260917`); no
alternate ROCm/rocprofv3 install exists on this box to try instead. The `umr` LDS-counter
registers for GPU 2 (DRM instance 17, per "Enabling the TX_VMW LDS counters" in `README.md`)
were correctly written and verified via readback this round, but are moot until this is fixed --
remember these do not survive a GPU reset/reboot. Blocks live confirmation of the barrier-count
finding above; the static assembly comparison was used instead.

**Barrier-merge follow-up tried and ruled out (2026-10-13).** Attempted the natural next step
suggested by the finding above: defer MXSA/MXSB's buffer-swap-and-TDM-reissue back to the same
later barrier A/B already uses, while keeping the narrow reads clustered early, to merge 3
barriers back to 2. Root cause of the split, found by instrumenting `place_GRs` directly (not
guessed): a GR's physical slot is decided by `_build_gr_slot_bounds`'s conflict check, which
pushes it later only while a same-iteration LR of that tensor still reads *after* it in program
order -- clustering removed all such later SA/SB reads, so nothing pushed their GR past slot 0
anymore. Fix: added `LRPlacement.data_ready_slot` to record each clustered LR's pre-clustering
("natural") slot, and fed that into the conflict check instead of the physical slot, so SA/SB's
GR still gets pushed to slot 2 (matching A/B) while the reads stay at slot 0. Verified
structurally exactly as designed: barriers 3 -> 2, main-loop instructions 666 -> 663 (fewer than
either prior build), reads still clustered at `subIterK=0`, `validate.py` 10/10 PASSED on GPU 2
(`logs/20261013-163638`). But `alt_timing.py` on GPU 2, 16 alternating rounds against `baseline`
and the already-committed `cluster` build, says otherwise: median 205.52 (baseline) / **204.95
(cluster, fastest)** / 205.86 (merged, slowest of the three; one baseline round was an unrelated
428 us system-noise outlier, excluded, means move the same direction either way). The merged
build is not just no better than `cluster`, it is slightly worse, including versus plain
baseline. Likely explanation, consistent with "TDM supply is the limit" above: issuing MXSA/
MXSB's TDM reissue early (even paying for an extra barrier) gives that fetch more overlap time
with compute before its data is needed, which matters more for a TDM-bound kernel than avoiding
one barrier -- merging it back into the later, bunched reissue point removes that head start.
Reverted (`LRPlacement.data_ready_slot` and the `_build_gr_slot_bounds` conflict-flat change);
`cluster` (3 barriers, committed in `83101adfc8c`) remains the best measured build.

Environment note from this round: the prebuilt `tensilelite-client` (dated Oct 7) started
throwing `"Failed to load solution library"` (SIGABRT) on startup for every build, including a
clean unmodified baseline -- a stale client vs. current Python-side library format, fixed by
`invoke build-client -g gfx1250 --clean`. Separately, two node reboots happened during this
investigation; `sudo journalctl -k -b -1`/`-b -2` traced both to a GPU-driver MES deadlock
(`tensilelite-cli` stuck 245-491+ s in `kfd_ioctl_runtime_enable` -> `amdgpu_mes_reg_write_reg_wait`,
a D-state hang that `SIGKILL`/timeouts cannot clear) rather than anything in the code above --
no TDM-descriptor-touching GPU test was ever run this round. Worth remembering: on this shared
gfx1250 node, a wedged MES blocks every user's KFD process until the node is rebooted.

### Ruled out at DepthU 512

- Main-loop buffer swaps (14 `v_xor_b32` + 4 `s_xor_b32`): removing them saves 0.5% of cycles,
  so unrolling to remove them is not worth it.
- Deeper prefetch: three DepthU 512 buffers need about 456 KiB, and the relaxed-wait bound above
  is 2.5%.

### Next

- Find what TDM is limited by (bytes vs loads vs L2 requests): L2 request rates, and variants
  that change one without the other.
- Cluster / multicast TDM loads, so workgroups that share A or B rows fetch them once (if gfx1250
  TDM supports it; Tensile has cluster barrier support).
- spreadMX in the generator (small, correct win), and the wide scale reads after that.
- A "fake fuse" test (at most 2 TDM loads per wave, same bytes) hung the GPU. The doubled-row
  loads of wave 3 and the padded 32-row scale loads probably write past the LDS allocation.
  Dropped for now.

## Scale addressing bugs hidden by validation (fixed 2026-10-06)

With the client's MX data generation every block gets the same scale, so validation passes no
matter which scale bytes the kernel reads. For `init-a/b=Random` (mapped to `rand_int`) and
`SerialIdx` (`Sequential`), all of these pass on (256..4096, 256..4096, 1, 2048..8192):
the scale address step per iteration set to 0 (every iteration reads K block 0's scales),
all waves reading wave 0's M rows, and the main-loop scale loads removed. The step-0 build's
D is bit-identical to the unmodified build's. `init-scaleA/B` are the scale-AB tensors and
`init-mx-a/b` is ignored unless the data init is random-like and the scale init is constant.

So the passing results in this file say nothing about scale addressing. The client option
`mx-scale-jitter=N` (`client/src/DataInitialization.cpp`) shifts each generated E8 scale by a
hash-based amount in [-N, N] before the upload permute, so every block gets its own scale. With
`mx-scale-jitter=4` the MT128x64 subtile kernel still passes and the MT256x256x256 kernel fails
on every size, including (256, 256, 1, 256). The HEAD build fails the same way. Two bugs:

- **Scale LDS layout.** The scale reads (`scaleMXSA/B[group g, K=k]`) expect
  [MMA group][K group][W bytes] per 128-row (A) or 128-column (B) partition: A at
  partition * 1024 + g * 256 + k * 128, B at partition * 1024 + g * 128 + k * 64, plus lane * 4.
  The TDM load writes each wave's 2 x 256 B as [K group 0: 64 rows x 4 B][K group 1: ...] at
  wave * 512. The two agree only for MT128-like geometry.
- **Scale step.** The TDM path advances the scale address by `ti.depthUBytes` =
  (DepthU / 32) * MT = 2048 bytes per iteration (`emitScaleGRPtrUpdate`) and per StreamK
  iteration (`tdmApplyStreamKOffsetSubtile`). The descriptor reads the {K/128, M, 4} layout
  (row stride `SizeI * 4`), where one iteration is (DepthU / 128) * Size * 4 = Size * 8 bytes.
  The two agree only when Size = MT. The non-subtile path (`KernelWriter.py`, around line
  10012) already uses Size * DepthU / MXBlock.

Both are fixed in the generator (2026-10-06). The scale LDS layout is now
[M/N chunk][K group][128 B] for TDM, written by the TDM load and read by `InstructionEmitter`
(details in the TDM supply section). The scale step and StreamK offset are
Size * DepthU / MXBlock bytes (`emitScaleTdmStepBytes`). With `mx-scale-jitter=4`:

- MT256x256 passes (256, 256, 1, 256), (256, 256, 1, 2048), (512, 256, 1, 2048),
  (256, 512, 1, 2048), (512, 512, 1, 2048), (2048, 1024, 1, 384), (4096, 4096, 1, 8192) and the
  edge size (384, 320, 1, 2048).
- Both kernels give the same pass/fail result on the regression sizes in the first section as
  without jitter. The failing sizes have about the same wrong-value counts with and without
  jitter, so those failures are not scale addressing. The MT256 counts dropped: 4627 (was 25445)
  and 1154 (was 13786). (300, 200, 1, 1024) and (1000, 1500, 1, 2048) fail with and without
  jitter, and with the old kernel; their counts change between runs.

Edge tiles: each wave's TDM row count is clamped to its part of the tensor, so whole 128 B
chunks past Size are zero-filled. A partial last chunk (Size not a multiple of 32) still reads up
to 124 bytes past Size. That is in the next K group's row, or past the end of the buffer for the
last one. Those scales only reach out-of-range rows of D. The K dimension is not bounded
(it was not before either). A K tail reads scale rows past K, past the end of the buffer for the
last batch. Those scales multiply zero-filled data, so they only matter if a byte is 0xFF (NaN).

`patch_scale_split.py` and `patch_scale_pad.py` apply the same fixes to a finished build (full
tiles only). (1024, 768, 1, 4096) faults with or without the patch (the StreamK crash above).

## Main-loop overhead vs the BF16 subtile kernel (open, found 2026-10-05)

Compared against BF16 MT256x256x128, PGR 2, StreamK 3, TDMInst 3 (from
`Tests/common/gemm/gfx12/subtile_bf16_gfx1250.yaml`). Per iteration of the current MXF4 loop:

- **TDM descriptors rebuilt from scratch, about 32 SALU per load, 4 loads.** The incremental
  update (`_emitGRPtrUpdate_TLU0`: address += DepthU bytes; `_tdmSwapLdsBuffer`: XOR the LDS
  address) already leaves the descriptor ready. `refreshTDMDescriptorSubtile`
  (`InstructionEmitter.emit_gr`) then re-runs the full init before every load, rewriting the
  control word, dims, tile sizes, stride and the per-wave row clamp (`v_readfirstlane` of
  Serial). It also writes the address fields a second time. This is needed today because the
  MXSA/MXSB descriptors alias the A/B descriptor SGPRs (`.set sgprtdmMXSAGroup0,
  sgprtdmAGroup0`). BF16 does the same rebuild without any aliasing.
- **The rebuild sits in a WMMA-free window.** Order: phase-1 WMMAs, 24 reads,
  `s_wait_dscnt 0`, barrier, about 128 SALU and 4 TDM loads with 2 WMMAs, barrier, 14 address
  swaps, `s_wait_alu depctr_va_vdst(0)`, 21 reads, then the next WMMA. BF16 spreads its
  descriptor SALU one or two per WMMA and keeps WMMAs running through both barriers.
- **12 A/B read-address VGPRs instead of 2, so 14 swaps per iteration.** A has one base per K
  read (8) and B one per K read (4), each with its own swap-mask VGPR, built in "LR Offset
  Calculation for Subtile Based Tiling" (`SubtileLREmit.py`). All of them get the same padding
  (pad block = row / 2 for K offsets under 128 B, +128 B for the 2048 B group), so they differ
  only by constants: A = base + {0, 32, 64, 96, 2176, 2208, 2240, 2272}, B = base + {0, 32, 64,
  96}. Folding these constants into the ds_load immediates would leave 4 swaps and free 20
  VGPRs. BF16 has the same pattern (4 bases per tensor).
- **The read gate bunches reads at the end of phase 1.** Ungated, phase 1 ends with reads
  paired with WMMAs and 7 trailing reads. With the gate it ends with 11 WMMAs and then 24 reads,
  so `s_wait_dscnt 0` before the barrier is exposed.

None of this is on the critical path today. `inject_delay.py` inserts `s_nop` cycles at one
point of both main-loop copies and reassembles (with code object v4 the unmodified round trip
is byte-identical). Extra cycles per iteration vs the unmodified kernel, 6-8 alternating
rounds on GPU 2 (a point fully on the critical path would cost about 17 us per 128 cycles):

| Delay point | +128 cycles | +512 | +1024 |
| --- | --- | --- | --- |
| After the first barrier (descriptor window) | +1.6 us | | |
| Before `s_wait_tensorcnt` | +0.3 us | -2.4 us | +30.3 us |
| After the second barrier (swaps) | +2.0 us | | |
| Middle of phase 1 | +1.8 us | -0.8 us | +31.6 us |
| Middle of phase 2 | +2.1 us | | |

Each wave has roughly 800 idle cycles per iteration (of about 2100), the same wherever the
delay goes. The loop is paced by something outside the waves' instruction stream, most likely
TDM data supply: 68 KiB per workgroup per iteration (A and B 32 KiB each, scales 2 KiB each)
is about 16.5 TB/s over 256 CUs at 1.08 us per iteration. Removing the overheads above only
pays off once that limit moves.

**Re-measured after the scale fix (2026-10-06, `logs/*-inj-g2_*`, 8 alternating rounds, GPU 3).**
The scale fix moved that limit, and the loop is now partly latency-bound. Base time is 232.3 us.

| Delay point | +128 cycles | +256 | +512 | +1024 |
| --- | --- | --- | --- | --- |
| After the first barrier (descriptor window) | +5.0 us | | | |
| Before `s_wait_tensorcnt` | +0.9 us | | | |
| After the second barrier (swaps) | +5.1 us | | | |
| Middle of phase 1 | +4.6 us | +11.3 us | +28.1 us | +72.5 us |
| Middle of phase 2 | +5.0 us | | | |

Only the delay between the last TDM load and `s_wait_tensorcnt` is absorbed. The loads of
iteration i are waited on in iteration i + 1, so the period is about the TDM latency plus
everything from that wait to the next iteration's loads. That is the second barrier, the swaps,
both phases, the first barrier and the descriptor rebuild. Every cycle there now costs about a
third of a fully exposed cycle at +128, rising to about two thirds at +1024. The descriptor
rebuild sits on that path: about 120 SALU and 4 `v_readfirstlane` between the first barrier and
the last `tensor_load_to_lds`. Two fixes:

- Issue the loads right after the first barrier with incremental descriptors (the swap and
  address update already leave them ready) and un-alias the MXS descriptor SGPRs.
- Deepen the prefetch so the TDM latency is hidden. This is harder. LDS mode 2 pads each buffer
  to two 64 KiB segments, 256 KiB of the 320 KiB, so a third buffer does not fit in that layout.
  The unpadded footprint is about 70 KiB per buffer (A and B 32 KiB each plus padding, scales
  4 KiB), so three buffers fit only with a different segment layout.

The floor with no TDM loads at all was 159.8 us.

**TDMGroup1Reuse attempt (2026-10-12): hoisting the descriptor rebuild found a correctness bug.**
Independently re-confirmed the "TDM descriptors rebuilt from scratch" finding above by reading
the current production DepthU 512, 2-buffer main loop
(`outdir3/.../build_tmp/SOURCE/assembly/...-cI-NVA=.s`, lines 1967-2640, 674 instructions/iter):
323 scalar (`s_*`) vs 142 vector (`v_*`, 128 of them `v_wmma_scale`) vs 176 `ds_load`
instructions. Of the 323 scalar ops, `s_set_vgpr_msb` (the gfx1250 extended-VGPR-bank-select
prefix, one needed before almost every VALU/LDS op once a kernel's VGPR footprint exceeds 256)
accounts for 247 -- 81% of (`v_wmma` + `ds_load`) count, tied to tile size (see the macro-tile
VGPR-wall note elsewhere). The other ~53 "plain" SALU ops are dominated by the Group1 rebuild:
confirmed via `grep -c "TDM Group1 control"` that the full rebuild (dim0/dim1/tile0/tile1-clamp/
stride/reserved, ~9-10 SALU x 4 tensors = A/B/MXSA/MXSB) is textually emitted at exactly 4
points in the generated file -- two one-shot PGR prefetch stages before `label_LoopBeginL`, the
one truly repeating body between `label_LoopBeginL` and its back-branch, and one one-shot
straight-line NLL tail block after `label_SkipToNLL` (confirmed no back-branch there). Only the
repeating body's copy costs anything per-K; the other three are one-time. Branch census: exactly
one `s_cbranch` inside the loop body (the back-branch itself) -- no divergent branches, so the
"stalls" are the already-documented barriers/`s_wait_dscnt`/`s_wait_tensorcnt`, not branches.

Tried to hoist Group1 (SizeI/SizeJ/SizeL/StrideA0I/StrideB1J/`tdmClamp{tc}` are all invariant
for the whole kernel invocation, not just the loop) out of the repeating body, relying on the
last PGR prefetch stage having already written it, and leaving only Group0 (global address +
LDS-swap address) refreshed per iteration -- i.e. generator-equivalent of what the "Descriptor
rebuild trimmed" update above already did for the per-wave clamp SGPRs, taken one step further.
New opt-in kernel param `TDMGroup1Reuse` (default 0, zero risk to any other config), threaded
through `SubtileGREmit.py` (`skipGroup1` arg on `initTDMDescriptorSubtile` /
`_initTDMDescriptorSubtileMX` / `refreshTDMDescriptorSubtile`), `InstructionEmitter.emit_gr`
(reads a new `self._stage` tag), and `LogicalScheduler.populate_instructions` (tags each
`populate()` call `preloop`/`mainloop`/`ngll`/`nll`; only `mainloop` is allowed to skip Group1).
`Tensile/Common/{GlobalParameters,ValidParameters}.py` got the new param registered.

Built clean (`configs/du512_group1reuse.yaml` = `du512_timing.yaml` + `TDMGroup1Reuse: [1]`,
`logs/20261012-214438`): confirmed in the generated assembly that `label_LoopBeginL`'s body now
has zero "TDM Group1 control" lines (down from 4 occurrences file-wide to 3), loop body shrank
674 to 640 lines, SALU count 323 to 289 (-34, matching the ~36 estimate). **`validate.py` on
GPU 2 then failed 7 of 10 default sizes** -- `256,256,1,256` (0 main-loop iters), `2048,1024,1,384`
(0 iters, the StreamK edge case) and `300,200,1,1024` (2 iters) passed; every K >= 2048 size
(4+ iters: `256,256,1,2048`, `512,256,1,2048`, `256,512,1,2048`, `512,512,1,2048`,
`4096,4096,1,8192`, `384,320,1,2048`, `1000,1500,1,2048`) failed. Pattern suggests the bug does
not manifest until several iterations in, which does not fit a simple "Group1 never gets
written at all" bug (that would fail even 2-iteration cases) -- leading hypothesis, not yet
tested: the SGPR allocator previously treated the Group1 SGPR range as dead in between the
(always-immediate) rewrites and may have let some other temp (`writer.allocTmpSgpr` or similar)
alias those same physical SGPRs elsewhere in the loop body, now silently corrupting Group1
after the first iteration or two once it is no longer rewritten every pass. Not yet checked:
whether this is StreamK-related (DP-partial tiles reuse/rewrite SGPR ranges differently) given
the known StreamK partials-write aliasing bug above. **Reverted cleanly** (`git checkout --`
on all 5 touched files, confirmed clean `git status`/`git diff --stat`); no generator code is
currently modified. Next: reproduce with a 4-iteration-only size under `rocgdb`
(`HSA_ENABLE_PRECISE_MEMORY=1`) or dump the Group1 SGPR values iteration-by-iteration to see
which word actually goes stale and trace what else writes those SGPR numbers.

**Root cause found (2026-10-13), no GPU execution needed.** The baseline (unmodified)
assembly's own `.set` directives show MXSA/MXSB do not just alias A/B's Group0 (as the
"Descriptor rebuild trimmed" note above already says) -- they alias Group1 too:
`sgprtdmAGroup0=40, sgprtdmAGroup1=72`, `sgprtdmMXSAGroup0 = sgprtdmAGroup0+0`,
`sgprtdmMXSAGroup1 = sgprtdmAGroup1+0` (B/MXSB the same way at 80/84). So the real per-iteration
sequence is: write A's *full* descriptor (Group0+Group1) into sgpr 40/72, issue A's
`tensor_load_to_lds`, then immediately overwrite those same four SGPRs with MXSA's *full*
descriptor (different control word, different dims) and issue MXSA's load. The "redundant"
Group1 rebuild is not redundant across iterations at all -- it is required within every single
iteration, because the register is time-shared between two unrelated descriptor shapes
(A and MXSA) moments apart. `TDMGroup1Reuse` skipped it on the assumption the value survives
untouched between A's uses, which is false: MXSA's refresh clobbers it before A's next turn.
This matches the pass/fail split exactly -- `K=1024` (2 total iterations, equal to
`PrefetchGlobalRead`'s prefetch depth) never executes the patched main-loop body at all (both
iterations are covered by the two prologue prefetch stages), while `K>=2048` does execute it
and corrupts A/B's descriptor from the first pass. Also explains why past attempts to give
MXSA/MXSB non-aliased SGPRs were rejected (122 SGPRs needed vs. a 106 limit, per the
"Descriptor rebuild trimmed" note) -- the aliasing is load-bearing, not an oversight, and
`TDMGroup1Reuse` as designed cannot work while it stands.

A narrower variant might still be worth it: instead of skipping the Group1 *write* (impossible
while A and MXSA time-share the register), precompute the packed dim0/dim1/tile0 *values* once
per tensor in the prologue into small dedicated (non-aliased) scratch SGPRs, and have the
main-loop refresh become a few plain `s_mov` copies from that scratch into the shared Group1
slot instead of the current 6-instruction shift/or chain. Register must still be rewritten
every use (unavoidable), but the computation feeding it gets cheaper (roughly 9-10 to 6-7 SALU
per tensor, not the full ~9-10 removed as hoped) -- smaller win, extra persistent SGPRs needed,
not yet attempted.

**Descriptor rebuild trimmed (2026-10-06, `logs/*-gen-v4`).** Separate scale descriptors would
need 122 SGPRs (limit 106), so the aliasing stays and the per-load rebuild is cheaper:

- The per-wave bounds (A/B tile1 clamp, scale dim1 rows) are computed once per tile into
  `tdmClamp{A,B,MXSA,MXSB}` (`initTDMClampSubtile`). This brings the kernel to 102 SGPRs and
  removes the 4 `v_readfirstlane` per iteration.
- The address update and LDS swap no longer copy into the descriptor; the refresh does it.
- The data dim words are packed in 6 SALU instead of 11.

The window from the first barrier to the last TDM load went from about 120 SALU and 4 VALU to
62 SALU. Time went from 232.3 us to 230.2 us (9.47 to 9.56 PF, 10 alternating rounds, faster in
10/10), with the same validation results, jitter included.

**LDS read-offset swaps moved off the critical path (2026-10-07, `logs/*-gen-v5`, GPU 0).**
The 14 `v_xor_b32` that switch the read offsets to the other buffer sat after the second
barrier, followed by an `s_wait_alu`, ahead of the first read. They are per-wave and need
neither the TDM data nor the barrier, so `hoistLRSwapAboveWaitTDM` (`WaitAluInsertion.py`)
moves them before the `s_wait_tensorcnt`. It does this only when an `s_wait_dscnt 0` already
drained every read of those offsets, and only in TDM kernels. Both main-loop copies and one exit
path qualify. Time went from 235.1 us to 233.6 us on GPU 0 (faster in 10/10; GPU 0 runs about
5 us slower than GPU 3 for the same kernel), with the same validation results, jitter included.

The A/B tile1 clamp is Size - wId * MT / numWaves, without the workgroup's tile offset, so it
is only right for workgroup 0 along that dimension. Edge workgroups read past Size, but only
into LDS rows whose outputs are never stored, so it did not cause the edge failures (those were
the store mask, see the top section). It is still an out-of-bounds global read.

**Deeper prefetch is worth about 16.5 us (2026-10-07, `logs/*-inj-v5_tc*`, GPU 0).**
`inject_delay.py SRC NAME tcnt 8` changes the main-loop `s_wait_tensorcnt 4` to 8, so each
iteration waits for the loads of two iterations back, which times a 3-deep pipeline (results
are wrong). 233.7 us became 217.1 us (8/8), and `tcnt 12` (4-deep) gives the same 216.7 us. The
same test on a build without segment interleave (`configs/segil0_timing.yaml`) gives 217.4 us
against 217.0 us with it, so a third buffer does not need the mode-2 layout. Three contiguous
buffers of A, B, SA and SB (about 68 KiB each) fit in the 320 KiB. The generator side: the
subtile `PrefetchGlobalRead` 2 is one tile ahead plus a one-partition offset, so this needs a
two-tiles-ahead mode, buffer rotation by an SGPR delta in place of the XOR swaps (which also
frees the 14 swap-mask VGPRs), and the prologue and exit paths for the extra tile.

## TDM supply: 256-byte scale rows cost 17% (fixed in the generator 2026-10-06)

The scale TDM loads read 2 rows of 256 B per wave (tile0 256 x 1 B, tile1 2, row stride
`SizeI * 4`). Rows wider than 128 B reach GL1 as a slow request type. Each costs about 2.1
GL1A busy cycles against about 0.19 for a 128 B A/B row, and GL1A is about 90% busy, so the
scales' 4 KiB of the 68 KiB per workgroup iteration cost 46 us of 275 us. Reading the same
bytes as 128 B rows removes almost all of it.

`patch_tdm.py` and `patch_scale.py` change the main-loop TDM loads of a finished build (timing
only, results are wrong). 8 alternating rounds per group on GPU 3 with `alt_timing.py`, size
(4096, 4096, 1, 65536); the unmodified kernel is 275.1 to 275.9 us in every group:

| Main-loop scale loads (per wave, per tensor) | Time |
| --- | --- |
| Unmodified, 2 rows x 256 B | 275.3 us |
| None, `s_wait_tensorcnt 2` (`dropS_w2`) | 228.3 to 229.2 us |
| 4 rows x 128 B, same bytes (`c128x4`) | 230.8 us |
| 8 rows x 64 B, same bytes (`c64x8`) | 232.9 us |
| 2 rows x 128 B, half the bytes (`c128x2`) | 229.8 us |
| 1 row x 256 B, half the bytes (`half`) | 250.7 us |
| 1 row x 512 B, same bytes (`r1`) | 267.8 us |
| 2 rows x 64 x 4 B elements, same bytes (`w4`) | 276.1 us |
| Issued before A and B (`order`) | 278.1 us |
| All workgroups read the same scales (`shareall`) | 277.8 us |
| Each workgroup reads a private copy (`noshare`) | 273.6 us |
| No TDM loads at all (`dropAll`), for reference | 159.8 us |

- Counters (`pmc_tdm.sh`), unmodified vs no scales: `TX_VMW_GL1_REQ_READ` +1.04M (one per
  256 B row), none of them counted as `TX_VMW_GL1_REQ_READ_128B` or `_64B`; `GL1A_BUSY` 8.43M
  vs 6.27M; `TX_VMW_GL1_PENDING_STALL` 110.8M vs 71.5M; GL1-to-GL2 latency from
  `GL1C_GL2_REQ_READ_LEVEL` about 462 vs 331 cycles. With `c128x4` the scale requests are
  counted as 128 B and `GL1A_BUSY` is 6.33M.
- Ruled out: L2 hot spots (GL2C busy 84% vs 83% of GL2C cycles, busiest channel about 1.1x
  the mean, all scale requests hit), same-address sharing (`shareall`, `noshare`), element size
  (`w4`), issue order (`order`), UTCL0 and set conflicts (no misses, zero stalls).
- `dropS` (scale loads removed, wait left at 4) measured 205 us earlier, but two loads per
  iteration under a wait of 4 lets the previous iteration's A/B loads stay in flight (deeper
  prefetch, and a race on their data). Only the `dropS_w2` number is a fair "no scales" time.
- TDM merges contiguous rows narrower than 128 B into the slow request type. With B in 64 B
  rows (`patch_scale_pad.py` without `bpair`) B's 8 contiguous 64 B rows per wave become 2
  slow requests (0.52M non-128 B requests, `GL1A_BUSY` 7.40M, 253 us). 64 B rows 16 KiB apart
  are not merged.

Correct variants, which also fix the layout and step bugs in the validation section (same
rounds and size):

| Scale loads per wave and tensor | TDM per iteration | Time |
| --- | --- | --- |
| Unmodified (wrong results) | 4 | 276.2 us |
| `split`: A 2 x (2 x 128 B), B 4 x (2 x 64 B), one row per K group | 8 | 249.1 us |
| `pad`: 1 load of 512 contiguous bytes, 128 B rows (A) and 64 B rows (B) | 4 | 253.1 us |
| `pad bpair=1`: as `pad`, both in 128 B rows, B reads re-addressed | 4 | 233.0 us |

The generator now emits `pad bpair=1`. Built from `configs/segil2_timing.yaml`
(`logs/*-gen-v2`), it runs at 232.5 us against 275.9 us unmodified (8 alternating rounds, 9.46 PF).
Segment-conflict stalls are 0.089 M (0.08% of CU cycles; 0.119 M before).
`logs/*-gen-v1` was built from `tensile_mxf4_att.yaml`, which has no `LDSSegmentInterleave`.
It has 24.4 M stall cycles (22%) but the same time, 232.6 us. In general terms (`_MxScaleTdmShape` in `SubtileGREmit.py`,
`ldsChunkBytes` in `Kernel.py`), for MT256x256: wave w loads K group w >> 1, rows or columns
(w & 1) * 128 to +128, as 4 rows of 128 B (row stride 128). TDM LDS padding of 128 B every
128 B (control `(1 << 20) | (4 << 22) | (31 << 25)`) spreads them to a 256 B pitch, starting at
LDS (w & 1) * 1024 + (w >> 1) * 128, from global (w >> 1) * Size * 4 + (w & 1) * 512. The B
scale reads move from g * 128 + k * 64 to (g >> 1) * 256 + k * 128 + (g & 1) * 64, matching A's
[pair of MMA groups][K group][128 B]. `GL1A_BUSY` is 6.39M and all 35.65M GL1 reads are 128 B
requests, the same as `c128x4`.

## LDS segment conflicts: nearly eliminated, no timing change (2026-09-30, finished 2026-10-05)

Kernel: MT256x256x256, MI 32x16x128, TDMInst 3, StreamK 3, size (4096, 4096, 1, 65536).
Counter: `TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES`, summed over the 256
CUs. The share of CU cycles is stalls / 256 / (`GRBM_GUI_ACTIVE` / 8), since `GRBM_GUI_ACTIVE`
is summed over the 8 XCCs. Measured on GPU 3 up to the exit-path split, on GPU 2 after.

| Variant | Stall cycles | Logs (`logs/`) |
| --- | --- | --- |
| Baseline | 23.23 M (16.3% of CU cycles) | `20260930-*-pmc3-base` |
| `LDSSegmentInterleave: 1` | 4.63 M | `20260930-*-pmc3-segil` |
| `LDSSegmentInterleave: 2`, per-port read order only | 1.81 M | `20260930-222002-abs2r` |
| + read gate (4,12) on the even copy | 0.57 M | `20261005-144133-abs2g4_12` |
| + exit paths (NGLL/NLL) split per port | 0.415 M | `20261005-151549-abs2exit` |
| + gate on the odd copy, spread, flip, gates (6,16) | 0.175 M | `20261005-155513-fs6_16` |
| + prologue reads split per port (current) | 0.118 M (0.08%) | `20261005-190611-final` |
| SB in its own segment (reverted) | 1.63 M | `20260930-*-abs3` |
| No per-port read reorder | 25.77 M | `20260930-*-pmc3-nostag` |
| Reorder keyed on wave bit 1 | 5.73 M | `20260930-*-pmc3-bit1` |

The 256-iteration figures come from `pmc_ksweep.sh`, which fits stalls at K = 16k/32k/64k into a
fixed part and a per-iteration part. The current kernel has no fixed part left and about 460
stall cycles per iteration summed over all CUs (under 2 per CU).

Timing does not change. 12 alternating rounds on GPU 2 (`alt_timing.py`): 278.36 us current,
278.08 us before the prologue split, 277.90 us with the per-port read order only. The paired
spread is about 2.5 us. LDS reads are off the critical path.

How mode 2 works (`configs/segil2_timing.yaml`, validated with `configs/segil2_validate.yaml`):

- Per LDS buffer, A and SA sit in segment 0 and B and SB in segment 1.
- Wave bit 0 selects the LDS read port. Even waves read A, SA, B, SB; odd waves run their own
  copy of the prologue reads, the main loop and the NGLL/NLL exit paths, reading B and SB in
  reverse order, then A and SA (`LR_ORDER_*` in `LogicalScheduler.py`).
- Flip (`LR_ORDERS_EVEN/ODD`): each copy reverses its order in the read group right after the
  barriers. There is no barrier between that group and the next iteration's reads, so both
  copies stay in the segment they were in.
- Gate (`LR_GATE_EVEN/ODD`, (phase 1, post-barrier) slack in WMMAs): each copy holds its
  second-segment reads until that many WMMAs after the slot where the other copy finishes that
  segment, then spreads them over the remaining WMMAs. Reads only move later and stay before the
  wait that consumes them.
- `seg_model.py` is a cycle model of both ports over the generated loop. It matched the
  measured loop length (2212 vs about 2180 cycles per iteration) and the ungated stall, and
  guided the gate choice.
- Correctness matches the failures above exactly, all 276 default subtile kernels are
  byte-identical to the code before this work, and the unit tests are unchanged.

What is left and why:

- `seg_model.py` predicts zero for the current kernel, so the remaining ~460 stall cycles per
  iteration (summed over the CUs, about 1.8 per CU) come from timing the model leaves out.
  Diagnostic builds of the current kernel with one class of reads removed (results wrong; logs in
  `logs/20261005-*-res-*`), stall per iteration:

  | Variant | Stall per iteration |
  | --- | --- |
  | Current | 457 |
  | Post-barrier burst reads removed | 62 |
  | Phase-1 reads removed | 252 |
  | Scale reads (SA, SB) removed | 229 |
  | A+SA removed / B+SB removed | 401 / 413 |
  | Gates (10,24) | 302 |
  | Gates (0,0) | 4793 |

  About 85% sits in the read burst right after the barriers, where both ports issue dense reads
  together. It does not depend on which tensor is read, and more slack reduces it, so the two
  ports' bursts overlap more in hardware than in the model. Either the ports leave the barrier
  skewed, or reads (the 4-byte scale reads in particular) hold the port longer than modelled.
  Telling these apart needs per-instruction timing. The ATT trace was empty here because it
  came from `configs/rocprof_att.yaml`. Plain `rocprofv3 --att` works (see "DepthU 512 main
  loop" above). Gates
  (10,24) cut it by a third but push phase-1 reads toward the barrier; not worth it at 0.08%.
- With WaveGroup 2x2 the B half follows wave bit 1, so both ports read the same B and SB data.
  B conflicts can only be avoided by timing, and the waves drift slightly apart.
- Giving SB its own segment removes the overlap on paper, but both buffers' SB copies then share
  the last free segment, the next iteration's TDM writes land in a segment being read, and
  address conflicts appear. Duplicating B and SB per port would remove the shared data but adds
  about 50% L2-to-LDS traffic, the suspected main bottleneck.
