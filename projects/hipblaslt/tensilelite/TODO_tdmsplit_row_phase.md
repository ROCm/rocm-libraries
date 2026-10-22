# TODO: TDMSplit row-phase split (per-wave halves) for LDSSegmentInterleave

Status: idea only, not implemented. Reference kernel: `~/bfss_work/tdms1.s`
(old TDMSplit, `TENSILE_EXP_TDMSPLIT=1`, LDSSegmentBRotate=0, VectorWidthB=-1,
LDSSegmentInterleave=1, MIWaveGroup [2,2], VW == WaveTile = 8, MT 256x256, DU 256 fp8).

## Problem

The barrier that issues the first-half TDM (`tdms1.s` ~3786-3792, and the same
barrier in loop copies 2 and 3) drains everything:

    s_barrier_wait -3
    s_wait_dscnt 0
    s_wait_tensorcnt 0
    s_barrier_signal -1
    s_barrier_wait -1        // auto token transition barrier, sync LDS0, LDS1, LDS6

That drain is the main loss the analyzer found for this kernel
(job 200d81..., about 308 exposed cycles per loop pass vs 175 for the baseline).

With today's split it is required. The TDM "halves" are LDS segments
(= wave components), not per-wave halves:

- Each A/B loading wave `c` (componentId = WaveIdx >> 1) loads 64 contiguous rows:
  LDS base `c * 16512`, start row `c * 64`.
- The second-half TDM adds 66048 to the LDS address (a whole segment) and 128 rows.
- So the first half = segment 0 = everything that M-component-0 waves (A) and
  N-component-0 waves (B) read. Those waves' last reads before the barrier
  (A tiles 4-7) are first-half reads, so dscnt must be 0. Waves 1/3 (A) and
  2/3 (B) read the next buffer's second half after barrier 1, so tensorcnt must be 0.

## Idea

Make the first TDM load tiles 0-3 of every wave (half of segment 0 + half of segment 1).

Which rows each tile holds: each lane's A read address is
`nIdx * 2048 + eIdx * 256`, with 256-byte rows. So tile `e` of component `c`
is the strided rows `c*128 + 8n + e`, for n = 0..15.

- First half (tiles 0-3) = rows 0-3 of every 8-row block in each segment.
- Second half (tiles 4-7) = rows 4-7 of every 8-row block.
- B reads use the same offset pattern.
- The half-tile register reads already split the same way:
  the first A group reads tiles 0-3, the second reads tiles 4-7.

One iterate-mode TDM per half per wave covers this. Descriptor changes:

| Field | Today | Row-phase split |
| --- | --- | --- |
| What A/B wave `c` loads | 64 rows of seg0, then 64 rows of seg1 | its own component `c` (segment `c`) |
| Wave LDS base | `c * 16512` | `c * 66048` (segment jump, as `_segWaveJump`) |
| Wave start row | `c * 64` | `c * 128` |
| Rows per step (tile_dim1) | 8 | 4 |
| Steps (iter_count) | 8 | 16 (HW max is 256) |
| LDS step (lds_inc) | 2064 | 2064 (unchanged) |
| Global step | 8 rows * stride | 8 rows * stride (unchanged) |
| Second-half LDS offset | +66048 | +1024 |
| Second-half global offset | +128 rows | +4 rows |

- LDS layout is unchanged: each segment still holds one component with the same padding.
- Each 4-row step is 1024 contiguous bytes inside a 2064-byte block, so it never crosses a pad.
- Bytes and TDM issues per wave are the same; each step is 1 KB instead of 2 KB.

## Expected waits afterwards

Per wave, TDM issue order is: [first half data, first half scales] at barrier 1,
then [second half data] at barrier 2.

- Barrier 1 (first-half TDM of buffer P):
  - WAR: only first-half reads of P and the scale reads must finish.
    In copy 1 the last one is MXSA at line 3712, then 12 A tile 4-7 reads,
    so `s_wait_dscnt 12`.
  - RAW: if the next buffer's (Q) second-half reads move after barrier 2,
    `s_wait_tensorcnt 1` is enough.
- Barrier 2 (second-half TDM of P):
  - WAR: dscnt covering the remaining P second-half reads.
  - RAW: `s_wait_tensorcnt 2` for Q's second half (issued at the previous barrier 2).

## Generator work

1. Descriptor setup: add a row-phase TDMSplit mode under LDSSegmentInterleave,
   using the table above. Touch points in `KernelWriterAssembly.py`:
   - `tdmWaveLdsAddr`: segment wave jump.
   - `_emitTdmIterateInit` / `_tdmIterTileDim1`: tile_dim1 4, iter_count 16.
   - Wave start row (`woffset = wId * (mt // numComp // dim1Divisor)` in
     `initTDMDescriptorWaveSeparatedImpl`): becomes `wId * mt // numComp`.
   - `tdmSplitLdsBoundary` returns 1024 (half an 8-row block);
     `tdmSplitGlobalInc` returns 4 rows.
   - The in-loop `halfRows` / `H1 = H0 - halfRows` clamp
     (`KernelWriterAssembly.py` ~12065-12100) must follow the new row mapping.
2. Read tagging: tiles 0-3 and scales get `memTokenLdsSplit[buf][0]`, tiles 4-7
   get `memTokenLdsSplit[buf][1]`. Today every A/B read in the loop gets the
   second-half token, so the planner drains to 0. With correct tags:
   - The half-split scheduler (`KernelWriter.py` ~5020, `_isHalf1Read`) moves
     Q's second-half B reads after barrier 2.
     Today they sit between the barriers, e.g. `B_X0` offset 1824 before barrier 2.
   - The barrier planner (`postMainLoopBarrierCheckAndReset`) computes the waits above.
3. Validity: limit the mode to VW == WaveTile (one vIdx per tile, eIdx = row phase),
   WaveTile 8, and MIWaveGroup [2,2]. Reject everything else in `Solution.py`.
4. Partial edge tiles: today's out-of-bounds handling trims the step count and
   sets `dim1` per wave. With 4-row steps spaced 8 rows apart, check whether the
   hardware compares `dim1` against the real row or against step * tile_dim1.
   If it is the latter, an edge tile could read past the end of the matrix.
   Confirm in the ISA documentation or add an edge fallback (for example the
   current split, or no split, when the wave's last block is partial).
   Sizes that are multiples of 256 avoid this, so perf testing is safe either way.

## Test plan

- Build `tdms_val.yaml`-style configs with the new mode plus the baseline and
  old TDMSplit. Validate on aligned and non-aligned M/N; non-aligned sizes
  exercise the edge path.
- Perf twice each (`run_all.sh`), GPUs 2,3 only (`GPUS="2 3"`).
- ATT capture (`att_capture.sh`) and analyzer upload. Compare exposed cycles
  with 200d81... (old TDMSplit) and be2ff... (baseline).
- Unit tests for the descriptor fields and the read-token tagging.
