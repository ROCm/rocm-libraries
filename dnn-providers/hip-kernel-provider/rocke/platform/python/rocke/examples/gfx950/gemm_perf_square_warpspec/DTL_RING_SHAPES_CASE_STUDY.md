# Persistent loops, stream-K, ragged shapes and small problems for the universal GEMM on gfx950 (case study)

Measured scope: the universal-GEMM DTL ring bodies (see [`RING_CASE_STUDY.md`](RING_CASE_STUDY.md) and
[`AGPR_CASE_STUDY.md`](AGPR_CASE_STUDY.md)) on gfx950, f16 and bf16, RCR, at the three square-ladder shapes
(8192^3, 23040x24576x12288, 15360x15360x768), the fourteen canonical contraction rows of the hipTensor cases
(1536x1440x576 up to 46080^2 x 2304), the measured-decision shapes of the dispatcher tests and, for a second pass over
the dispatcher rules, about 220 further (dtype, shape) cells from 1536x1440x64 to 46080^2 x 2304 (K 32-8192),
three rounds of them held out when the rule they test was written. Every timed kernel passed an on-device
full-output check against an f64-accumulated reference on random data, before and after its timing window (the
stream-K kernels also a check that the ready-flag table is clean after the windows). Comparisons use interleaved,
randomised sustained windows (every label timed once per round in a seeded random order, 5-8 rounds, paired ratios
with 95 % intervals), because the board's power cap turns short hot runs into noise. A hipBLASLt solution sweep (hipBLASLt from ROCm 7.2)
ran in the same processes as the reference. Relative results only. Caveat: all windows run back to back with a warm L2
/ MALL; small problems (for example 1536 x 1440 x 576) then run from cache, for every compared kernel alike, which
is not representative of isolated library calls.

## What was built (all default-off `TraitSpec` knobs, both engines, byte-identical)

| knob | what it does | verdict |
|---|---|---|
| `persistent_ctas` on the in-stream bodies | the persistent tile loop now also runs the fragment-prefetch / interleave bodies (the vmcnt proof does not depend on where the barrier sits) | tie at most shapes (within +-1 %); +1-2 % at K = 768; the enabler of the big-grid body below |
| `streamk` | CK Tile's DP + 2-tile stream-K partition in the persistent loop, computed in the kernel from M / N / K and the pool P; linear fixup with one ready flag per (workgroup, wave) that the owner resets after use, so the flag table is zeroed once at allocation; partials as 16-byte buffer stores; params `SkFlags` / `SkPart` (`streamk_workspace_bytes`); launch exactly P co-resident workgroups (validity rejects a P above the static co-residency bound, see below) | correct everywhere (including tails, ragged shapes, flags clean after every launch and after sustained windows). The one partition built (DP + 2-tile SK over all P workgroups with full 256x256 f32 partials) loses everywhere measured; see "What the stream-K result does and does not say" |
| `dtl_tight_rsrc` | every DTL buffer resource is based at the tile origin (`ptr + (tile0 + k) * 2`, soffset 0) with the operand's remaining bytes as range, so rows past M (A) / N (B) read zeros | makes `pad_m` / `pad_n` safe on the ring; cost below 1 % with `dtl_hoist` |
| `pad_fast_tile` | one uniform branch per output tile: tiles fully inside M x N take the unguarded epilogue | removes most of the guarded-epilogue cost (10-25 % at short K); WaveScope: VALU / SALU / branch per K tile down 40 % / 28 % / 93 % in the epilogue share |
| `global_flag_wait_eq(..., ordering="monotonic")` (IR builder, both lowerings) | a relaxed spin; the caller places one acquire fence after it | see below |

## The acquire spin that invalidated the L2

The first stream-K build ran 2-5x slower than the data-parallel kernel. The ISA showed why: an agent-scope acquire
atomic load on gfx942 / gfx950 is `global_load ... sc1` followed by `buffer_inv sc1`, so every iteration of the
owner's spin loop invalidated the L2 of its XCD while the other 31 workgroups there were streaming their tiles. A
relaxed spin with one acquire fence after the wait (and, in the data-parallel path, no fence at all) brought the
stream-K kernel to within the cost of its partial traffic. The same lesson applies to every cross-workgroup flag on
these targets.

## Stream-K co-residency is a validity rule, not a comment

An owner waits on partials of higher-index workgroups, so all P workgroups of the pool must be resident at once; a
launch with more than fit spins forever instead of failing. `is_valid_spec` (both engines) therefore rejects
`streamk` when `persistent_ctas` exceeds `CUs x workgroups per CU`, with the per-CU count bounded by the body's LDS
use and by the register worst case (one 512-register wave per SIMD) -- one workgroup per CU for every 256x256 body,
two for small 1-2-wave bodies. `streamk` also requires `warp_k == 1` (the partial slot is sized `tile_m x tile_n`).
The static bound uses the whole-device CU count; `streamk_launch_error(spec, device_cus, wgs_per_cu)` checks a launch
against the live device (a CPX partition has a fraction of the CUs), and the measurement harness refuses a stream-K
candidate whose pool exceeds `multiProcessorCount x hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` (which equals
the static bound for every body measured here).

## What the stream-K result does and does not say

Only one partition policy was built and measured: DP + 2-tile SK spread over all P workgroups, with one full
256x256 f32 partial tile per contributing workgroup (about 64 MiB of partials written and read per launch at
P = 256 -- as much as the operands at K <= 1536). It lost at every measured shape, including the tail-heavy
4608 x 3840 x 576 (0.36-0.72 of the persistent data-parallel body in the second pass) and 15360^2 x 768
(0.81-0.91). The loss traces to that partial traffic, so the result does not rule out stream-K for the tail-heavy
shapes hipTensor produces: untried variants are SK over the remainder tiles only, SK spread over just the idle
workgroups of the last wave, smaller SK tiles, and bf16 / f16 partials. The knob stays for that work.

## Tile shape before everything else

The 256x256 tiles are the right shape only when they fill the device. The small-tile ring (128x128x64, 8 waves
of 64x32, 2-stage ring, MFMA priority bookends, DTL resource hoist, pad + exact DTL bounds + fast tile) took the
dispatcher's poorly filled shapes up substantially (pick / previous pick): 1792^3 1.7x, 2048^3 1.65x, 2304^3 1.4x,
2048^2 x 512 1.5x, 3072 x 1920 x 576 1.77x, 4608 x 3840 x 576 1.18x, and gave 1536 x 1440 x 576 a kernel at all.

A second measurement pass over the regimes the first version admitted without measurement changed four rules:

* **K floors where the cshuffle default can take the shape.** The small ring now has the aligned ring's K floor
  (`DTL_RING_ADMIT_RULES` k_lo: f16 128, bf16 64). Small ring / cshuffle at K = 64: f16 0.94-1.09 (mixed), bf16
  1.06-1.19; at K = 128: 1.08-1.30.
* **K % 64 == 32.** Every ring body lost at K = 32 (0.75-0.95 of the cshuffle default), so K % 64 == 32 needs
  K >= 96. On up to two waves of 256x128 tiles a 128x128x32 2x2-wave interleave ring replaces the 256x128x32 AGPR
  body: 1.03-1.56 over the default at K = 96-288 (2048^2, 2560^2, 3072^2, 3584 x 2560) where the AGPR body managed
  0.89-1.25. On more tiles the AGPR body stays from K = 160 on (4608^2 x 160 / 288: 1.09-1.36; at K = 96 it was
  0.99-1.06, so that case keeps the default).
* **One wave of tiles: the 3-stage interleave body.** When the 128x128 tiles fit one wave (one workgroup per CU)
  and K >= 128, the small ring takes its 3-stage fragment-prefetch interleave form: 1.03-1.54 over the 2-stage body
  (1792^3 1.36-1.39, 2048^3 1.21-1.22, 1536^2 x 4096 1.46-1.51, held out: 1024 x 2048 x 2048 1.46-1.47,
  2048 x 1536 x 1024 1.18-1.21, 1280 x 2048 x 3072 1.52-1.54, 2304 x 1792 x 1536 1.23-1.27); with more than one wave
  the 2-stage body stays (2304^3: 3-stage 0.93-0.94). WaveScope at 1792^3 (f16, per 64-wide K tile per
  wave): about a third fewer cycles; the 2-stage ring's full-drain `s_waitcnt vmcnt(0) lgkmcnt(0)` disappears
  and the WAIT share of wave states falls by about 40 % (relative) -- with a single workgroup per CU there
  is no co-resident workgroup to hide the DMA latency, so the deeper ring has to.
* **Ragged small K.** At K <= 512 on fewer than 1280 256x256 tiles the small ring beats the pad ring (1.02-1.15 at
  7680 x 6720 / 8192 x 8200 / held-out 6144 x 4104 / 6144 x 6152 / 6656 x 6664), so it takes them even when the
  256x256 wave is well filled; at 1640 / 2352 tiles the pad ring is ahead or even (small ring 0.98-1.03). Ragged shapes have no other
  candidate, so they are not K-gated; ragged K % 64 == 32 (7168 x 7176 x 96) now gets the 256x128x32 body instead
  of no kernel (1.20-1.23 over the AGPR interleave pad body).

The small and pad rings also ask `select_split_k` (with the default candidate's tile) whether the default would
split K, like the aligned ring candidate, instead of relying on their tile floor matching its gate.

## Ragged shapes

Four of the fourteen canonical rows (N = 1440, 6720, 20160, 30240) had no dispatcher candidate at all; a fifth
(15360 x 13440 x 1536) fell to the 128x128 cshuffle default. The padded ring (the 16-wave persistent ring, P = #CUs,
chiplet chunk 32, with pad + exact DTL bounds + fast tile) serves them at 0.82-0.96 of hipBLASLt; the last one went
up 2.1-2.3x.

## Chiplet rasterization of very large grids (rule re-fitted on a second pass)

Narrower chiplet super-tile groups (`chiplet_wgm = 4`) pay on very large grids, but the first rule ("from 9000
256x256 tiles on, the persistent 4-wave AGPR interleave body with wgm 4") was fitted to the canonical shapes and ignored K.
Re-measured on 108 (dtype, shape) cells with >= 9000 tiles (K 64-8192), it lost up to 4.5 % (geometric mean of
chosen / best 0.988, worst 0.955): at 9000-20000 tiles with a moderate footprint the 16-wave wgm-8 ring is faster.
Two intermediate rules failed on held-out shapes (worst 0.84 and 0.89 of the best body: 40960^2 x 512 and
30720 x 30728 x 1280, where the wgm-8 rings lose 8-17 %). The shipped rule:

| condition (>= 9000 tiles) | aligned body | ragged body |
|---|---|---|
| K <= 128 | AGPR interleave, wgm 4 | AGPR interleave pad, wgm 4 |
| >= 30000 tiles, or A + B > 192 MiB, or (>= 20000 tiles, ragged >= 12000, and K >= 512) | 16-wave ring, wgm 4 + `iglp_opt(1)` while A + B <= 256 MiB; AGPR interleave beyond | AGPR interleave pad |
| otherwise | 16-wave ring, wgm 8 (unchanged) | 16-wave pad ring (unchanged) |

Over all 102 cells it reaches a geometric mean of 0.998 of the best measured body (worst 0.962); on the last round
of 20 held-out cells (written before they were measured) 0.993 (worst 0.962: ragged 33280 x 33288 x 768), where the
first rule reaches 0.983 (worst 0.957). The A + B footprint thresholds follow the 256 MiB MALL; the mechanism is not
L2 reuse (rocprofv3 at 46080^2 x 2304: a slightly lower L2 hit rate and more DRAM reads, but 14 % fewer wait cycles
per busy cycle): the narrower groups spread the concurrent DTL streams over the XCDs. WaveScope of the wgm-4 + `iglp_opt(1)`
ring against the wgm-8 ring at 30720^2 x 2048 (f16; sustained 1.10-1.13): DTL issue latency per K tile -11 %, the
full-drain wait -73 %, cycles per K tile -5 % on the traced CU -- the rest of the sustained gain is outside one CU's
view (the stream spread across XCDs and the board clock). The `iglp_opt(1)` ring's ISA does not depend on the
compile order. Cases still off by a few percent: the ragged 12000-20000-tile band at K = 768, and
the band edge around a 200 MiB footprint (33792^2 x 1536 vs 29696^2 x 1792).

The parity configs `gemm[57..64]` pin the exact specs the candidates emit (one per body) in both engines, and
`TestDispatchedSpecsPinned` fails if the dispatcher's spec drifts from them.

## Where the gap to hipBLASLt still is

The headline ratio is the dispatcher pick over the hipBLASLt solution of the same windows: 0.72-0.96 across the
canonical rows with the bodies of this study, 1.06-1.07 at 1536 x 1440 x 576 (the single-wave 3-stage body; that shape informed the
rule, so it is not a held-out result). The "best rocKE label" of a cell is the maximum over the 20-26 labels timed
there while hipBLASLt is one pre-selected solution, so with a 1-3 % run-to-run spread max-of-N inflates best / HB
slightly; treat best-label ratios within a few percent of 1 as ties. At the large shapes the WaveScope comparison
per 64-wide K tile is unchanged in kind from the AGPR case study: DTL issue stalls 1.4-1.9x hipBLASLt's, store issue about 4x,
and the power cap turns per-clock gains into clock loss. The 32x32x16 4-wave bodies win only at 5120 x 16128 x
16384 class shapes.
