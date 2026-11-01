# Four-wave AGPR-accumulator body for the universal GEMM on gfx950 (case study)

Measured scope: the universal-GEMM ring body (`dtl_stages`, see
[`RING_CASE_STUDY.md`](RING_CASE_STUDY.md)) rebuilt as a 4-wave workgroup in which
each wave owns a 128x128 output tile (2x2 waves, 256x256 block), on gfx950, f16
and bf16, RCR, at 8192x8192x8192, 23040x24576x12288 and 15360x15360x768.
Baselines measured in the same jobs on the same board: the shipped gfx950
dispatcher pick (the 16-wave ring), a CK-Tile instance sweep (the hipTensor
batched-contraction instances) and a hipBLASLt solution sweep. Every timed kernel
of every arm passed an on-device full-output check against an f64-accumulated
reference on random data, before and after its timing loop. Relative results only;
absolute numbers are kept in the internal record.

## Why 128x128 per wave

The 16-wave ring body gives each wave a 64x64 tile: four waves share each SIMD,
every K tile ends in a 16-wave barrier, and each MFMA reads twice the LDS
fragment bytes of a 128x128-per-wave tile. Vendor GEMM kernels on this target use
4 waves of 128x128. That tile needs 256 f32 accumulator registers per lane, i.e.
the whole AGPR file, and the earlier attempt to build it (`warp_m = warp_n = 2`,
32x32x16 atoms) produced a loop dominated by `v_accvgpr_read/write/mov` copies and
scratch spills.

## Getting 256 accumulators into AGPRs (`acc_agpr`)

Three separate compiler interactions had to be handled, each found with a single
local compile and the ISA (no GPU needed):

1. **Loop-carried accumulators split into scalars.** The accumulators start from
   the zero constant. LLVM's AMDGPU CodeGenPrepare breaks large PHIs whose
   incoming value is a constant into per-element PHIs; with 16 `<16 x float>`
   accumulators that reassembles every accumulator around every MFMA. Seeding the
   accumulators from an opaque `asm "", "=a,0"(zeroinitializer)` keeps the PHIs
   whole vectors (`-amdgpu-codegenprepare-break-large-phis=0` confirms the cause).
2. **Which MFMA form is selected.** `amdgpu-agpr-alloc = (n, n)` with n the
   accumulator registers selects the AGPR-destination MFMA form with some LLVM
   builds, but a newer compiler selects the VGPR form anyway and then shuffles the
   accumulators between the register files (and puts the operand fragments in
   AGPRs). The kernel attribute `mfma_agpr_form` makes the Python compile path pass
   `-mllvm -amdgpu-mfma-vgpr-form=0` for every LLVM 21+ lowering flavor (a function of
   the kernel and the flavor, not of the compiler that happens to load); an ISA hash check showed
   that the option does not change kernels compiled later in the same process.
   No scheduler-strategy attribute and no other backend option tried had the same
   effect.
3. **The epilogue spilled.** The loop exit copied all 256 AGPRs to VGPRs at once,
   which spilled the store addresses. An ordered `asm sideeffect "", "=a,0"` per
   accumulator atom right before that atom's conversion (split-K: its atomic adds)
   keeps the copies one atom at a time.

`acc_agpr_pin` (each MFMA result routed through `"=a,0"`) is kept as a knob; it did
not fix the newer compiler and costs registers on the 16x16 body.

With these the K loop of the 4-wave body holds only MFMAs, `ds_read`s, DTL issues
and SALU, with no accumulator copies and no scratch, in both engines and with every
compiler tried.

## Scheduling the 1-wave-per-SIMD loop

With one wave per SIMD nothing hides a stall, so the instruction order inside the
K tile matters. The levers, all `TraitSpec` fields, all default off:

| lever | what it does | effect (instruction trace) |
|---|---|---|
| `mfma_interleave` (needs `dtl_frag_prefetch`) | every next-atom fragment `ds_read` and every DTL issue is placed behind one MFMA and the order is pinned with `sched_barrier(0)` after each MFMA | the largest single step: the plain prefetch body front-loads reads and DMAs and the MFMAs wait on them |
| `ilv_merge` | reads and DTL issues on one merged slot schedule, so no MFMA carries two of them | MFMA issue latency per instruction drops to the vendor kernel's level |
| `ilv_barrier_after=k` | the first k MFMAs of a tile's last cluster go in front of its wait + barrier | small; kept |
| `ilv_dtl_every=k` | DTL issues one every k MFMAs instead of evenly spread | packing them loses; kept |
| `lds_swizzle_r=R` | row shift of the LDS XOR swizzle key | the default rule leaves 2-way `ds_read_b128` conflicts for 16x16x32 at tile_k 32 (and 2-way / 4-way for 32x32x16 at tile_k 64 / 32); R = 1 (16x16x32 at tile_k 32, 32x32x16 at tile_k 64) and R = 2 (32x32x16 at tile_k 32) are conflict-free per `rocke.analysis.lds`; the hardware conflict counter confirms it (see the results); the trace and the timing barely moved |
| `dtl_hoist` | one buffer resource and `soffset` per operand and K tile instead of per DTL instruction | about a quarter fewer SALU instructions in the loop |

The best body found is 16x16x32 atoms (the vendor kernels' choice), tile_k 32 with
a 3- or 4-stage ring (three tiles of DMA in flight), fragment prefetch with the
merged interleave, barrier after two MFMAs, R = 1 and the DTL hoist.

## Results (relative, same jobs)

Method: one process per (dtype, shape) holds every candidate (rocKE bodies and
hipBLASLt solutions). After a full-output check of each, 8 to 10 rounds run every
candidate once for a 1.5 to 2 second sustained window, in a fresh random order per
round, and every candidate is checked again afterwards. Each candidate's window is
compared with the dispatcher pick's window of the same round; the table gives the mean
of these paired ratios with a 95 % t interval. A difference counts only when the
interval excludes 1. The dispatcher pick and the identical 16-wave body, timed as two
labels, measure the noise floor: their ratio stayed within 1 +- 0.5 % with an interval
containing 1 in every cell. All arms run at the board power cap.

| dtype, shape | best 4-wave AGPR body vs the 16-wave dispatcher pick | AGPR body throughput per GHz vs pick |
|---|---|---|
| f16 8192^3 | +1.0 to +1.5 % (two runs, different seeds) | about +8 % |
| bf16 8192^3 | +2.3 to +3.0 % (two runs, different seeds) | about +6 % |
| f16 23040x24576x12288 | -1 to -2 % | about -2 to -3 % |
| bf16 23040x24576x12288 | -2 to -3 % | about -4 to -5 % |
| f16 15360x15360x768 | -0.5 to -2.5 % (+1 % with chiplet chunk 32) | about equal |
| bf16 15360x15360x768 | -0.2 to -2.8 % (+0.3 % with chiplet chunk 32) | about -3 % |

`dtl_hoist` and `lds_swizzle_r` do not move the time: the AGPR interleave body with and without the
hoist ties within the noise in every cell. A body compiled by the system ROCm comgr
(with the forced AGPR form) times the same as the torch-comgr build in every cell.
The hipBLASLt solutions remain 1.12x to 1.20x faster than both rocKE bodies on these
shapes; the AGPR body is ahead of the CK-Tile sweep at every shape, as the 16-wave ring
already was.

Each AGPR / interleave lever is admitted only on targets where its bodies passed GPU
full-output numerics: all of them on gfx942 and gfx950 (on gfx942 every AGPR body
passes but loses to the 16-wave ring). Nothing is admitted on gfx90a yet.

## Why the faster loop barely moves the throughput

The board runs at its power cap for every arm. A body that keeps the MFMA pipe busier
per clock is clocked down until it draws the same power, so per-clock efficiency
turns into throughput only when the joules per FLOP also drop. The 4-wave body
raised per-clock efficiency (the trace's cycles per K tile fell by about a third
relative to the plain 4-wave body, and it now issues MFMAs as densely as the vendor
kernel in steady state), but at the cap it lands within a few percent of the
16-wave ring.

What is left, per the instruction trace against a vendor kernel with the same
geometry: each DTL instruction stalls about twice as long at issue (tile_k 32 makes
every DMA instruction touch 16 half cache lines instead of 8 full ones; at tile_k 64
the stall drops by a third but the 2-stage ring then exposes the load latency),
plus small epilogue, SALU and barrier differences.

## Reproduce

The knobs are plain `TraitSpec` fields; build any combination with
`build_universal_gemm(UniversalGemmSpec(...))`. The kernel-name flags are `_agpr`,
`_apin`, `_ilv`, `_de{k}`, `_ba{k}`, `_mrg`, `_swr{R}` and `_dh`. CPU tests:
`tests/instances/test_gemm_acc_agpr.py` (its `TestEngineMirror` also checks the C++
engine's spec-dict path when the pybind engine is built); parity configs `gemm[30..47]`
in `tests/instances/parity/gemm_emit.{py,c}` cover both engines, including the best body
exactly (`gemm[36]`), every `lds_swizzle_r` value, `acc_agpr_pin`, the cshuffle and
split-K epilogues with AGPRs, the gfx942 bodies (`gemm[32]`, `gemm[43..45]`, `gemm[47]`) and
the gfx90a rejection (`gemm[46]` must be rejected by both engines).
