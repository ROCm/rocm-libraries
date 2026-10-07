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

- Not caused by the 32x16 accumulator row fix-up (`emitWmma32x16AccRowFixup`). Tensile built
  from `e91596bfb48`, before that fix, has the same pass/fail result on every size. The count of
  wrong values differs slightly between the two builds.
- Every batch > 1 size fails. Some single-batch sizes with partial M/N tiles and K of 1152 or 1280
  also fail. Edge-only (96, 48, 1, 256) and tail-only (128, 64, 1, 384) sizes pass.
- The crash is one 256x256 tile with a long K, so StreamK 3 splits a single tile's K range
  across workgroups. The partial-tile fixup is the first suspect.
- Not yet checked: the same sizes with StreamK 0, and whether the non-subtile MXF4 kernel
  (`configs/ref_nonsubtile_mxf4.yaml`) passes them.

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
  Telling these apart needs per-instruction timing; the ATT trace is empty on gfx1250. Gates
  (10,24) cut it by a third but push phase-1 reads toward the barrier; not worth it at 0.08%.
- With WaveGroup 2x2 the B half follows wave bit 1, so both ports read the same B and SB data.
  B conflicts can only be avoided by timing, and the waves drift slightly apart.
- Giving SB its own segment removes the overlap on paper, but both buffers' SB copies then share
  the last free segment, the next iteration's TDM writes land in a segment being read, and
  address conflicts appear. Duplicating B and SB per port would remove the shared data but adds
  about 50% L2-to-LDS traffic, the suspected main bottleneck.
