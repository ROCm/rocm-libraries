# Known issues: gfx1250 MXF4 subtile

## Correctness failures outside the common-test shapes (open, found 2026-09-30)

Reproduce on GPU 3:

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

- Not caused by the 32x16 accumulator row fix-up (`emitWmma32x16AccRowFixup`). Tensile built
  from `e91596bfb48`, before that fix, has the same pass/fail result on every size. The count of
  wrong values differs slightly between the two builds.
- Every batch > 1 size fails. Some single-batch sizes with partial M/N tiles and K of 1152 or 1280
  also fail. Edge-only (96, 48, 1, 256) and tail-only (128, 64, 1, 384) sizes pass.
- The crash is one 256x256 tile with a long K, so StreamK 3 splits a single tile's K range
  across workgroups. The partial-tile fixup is the first suspect.
- Not yet checked: the same sizes with StreamK 0, and whether the non-subtile MXF4 kernel
  (`configs/ref_nonsubtile_mxf4.yaml`) passes them.

## LDS segment conflicts: reduced, not eliminated (parked, 2026-09-30)

Kernel: MT256x256x256, MI 32x16x128, TDMInst 3, StreamK 3, size (4096, 4096, 1, 65536), GPU 3.
Counter: `TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES`, summed over the 256
CUs. The share of CU cycles is stalls / 256 / (`GRBM_GUI_ACTIVE` / 8), since `GRBM_GUI_ACTIVE`
is summed over the 8 XCCs.

| Variant | Stall cycles (median) | % of CU cycles | Logs (`logs/20260930-*`) |
| --- | --- | --- | --- |
| Baseline | 23.23 M | 16.3% | `pmc3-base` |
| `LDSSegmentInterleave: 1` | 4.63 M | 3.3% | `pmc3-segil` |
| `LDSSegmentInterleave: 2` (kept) | 1.81 M | 1.26% | `abs2`, `pmc3-abs2` |
| Mode 2 with SB in its own segment (reverted) | 1.63 M | 1.14% | `abs3`, `pmc3-abs3` |
| Mode 2 without the per-port read reorder | 25.77 M | 17.9% | `pmc3-nostag` |
| Mode 2 with the reorder keyed on wave bit 1 | 5.73 M | 4.0% | `pmc3-bit1` |

Timing does not change: 275.44 us baseline, 275.17 us mode 1 and 275.40 us mode 2, from 12
alternating rounds (`alt_timing.py`). LDS reads are off the critical path, so this is parked.

Mode 2 (`configs/segil2_timing.yaml`, validated with `configs/segil2_validate.yaml`):

- Per LDS buffer, A and SA sit in segment 0 and B and SB in segment 1.
- Wave bit 0 selects the LDS read port (confirmed: keying the reorder on bit 1 is worse).
- Even waves read A, SA, B, SB. Odd waves run a second copy of the main loop that reads B and SB
  in reverse order, then A and SA.
- The NGLL and NLL exit paths are shared and not reordered.
- Correctness matches the baseline (the failures above), and default kernels are byte-identical.

Why the rest remains:

- With WaveGroup 2x2 the A half follows wave bit 0 (the port), so A and SA are private to one
  port. The B half follows wave bit 1, so both ports read the same B and SB data. B conflicts
  can only be avoided by timing, and the waves drift apart.
- Per phase a wave issues 20 read slots on A+SA and 24 on B+SB. With both ports reading B+SB in
  segment 1 for 24 of 44 slots, at least 4 slots overlap even without drift.
- Floor for timing-based separation: about 0.69 M, measured with scale reads removed (diagnostic
  only, `pmc3-noscale2`).
- Giving SB its own segment removes the structural overlap on paper, but both buffers' SB copies
  then share the last free segment. The TDM writes for the next iteration then land in a segment
  being read: 178 address conflicts appear, run-to-run spread widens to 1.40-1.86 M, and LDS
  grows by 4 KiB.

Options for later:

- Rebalance to 22/22 slots per segment by moving part of each port's A rows into the B segment.
  This needs a TDM load layout change and removes only the structural overlap, not the drift.
- Duplicate B and SB per port: no shared data, so zero conflicts, but about 50% more L2-to-LDS
  traffic, which is the suspected main bottleneck.
- TDM with one wave per tensor, so A and B writes also target different segments.
