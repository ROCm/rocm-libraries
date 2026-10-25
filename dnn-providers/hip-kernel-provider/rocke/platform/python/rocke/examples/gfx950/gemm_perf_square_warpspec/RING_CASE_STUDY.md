# Alias-scoped DTL LDS ring for the universal GEMM on gfx950 (case study)

Measured scope: the universal-GEMM body of this example's rung F (256x256x64 block,
4x4 waves of 64x64, 16x16x32 MFMA atoms, direct-to-LDS loads, element swizzle,
chiplet grid swizzle) on gfx950, f16 and bf16, RCR, at 8192x8192x8192,
23040x24576x12288 and 15360x15360x768, plus 79 admission-calibration shapes per
dtype (768-16384, K 64-16384). Relative results only; absolute numbers are kept in
the internal record.

## The problem the trace showed

An instruction-level trace (rocprofv3 ATT, WaveScope, source-mapped to the builder
lines) of rung F attributed about a third of every wave's stall time to one
`s_waitcnt vmcnt(0)` in front of the first `ds_read` of each K tile. The builder did
not emit it: `SIInsertWaitcnts` inserts it because the ping and pong halves of the
operand tiles are rows of one LDS allocation selected by a runtime parity, so the
backend must assume the `ds_read` of the current half may alias the
`buffer_load ... lds` just issued into the other half. The next tile's global-load
latency was therefore exposed in every K tile instead of overlapping the MFMAs.

## The levers (all `TraitSpec` fields, default off)

| lever | what it does |
|---|---|
| `dtl_stages=N` (2, 3, 4) | one A and one B LDS allocation per stage, kernel attr `lds_alias_scopes`, K loop unrolled by N so every stage index is static; N >= 3 keeps N-1 tiles in flight with a partial `vmcnt((N-2)*L)` |
| `interwave_setprio` | MFMA-cluster `s_setprio` bookends (already a knob; it loses on rung F and wins on the ring) |
| `dtl_stream_barrier` | the per-tile wait + barrier + next DMA issue moved in front of the last K atom's MFMA cluster |
| `dtl_frag_prefetch` | the above plus one-K-atom-ahead fragment register prefetch (needs the 8-wave 32x32 body: the 16-wave body has no VGPRs left) |
| `dtl_issue_spread` | the next tile's DMAs spread between the MFMAs instead of a burst |
| `mfma_swap_ab` | MFMA fed (B, A): the accumulator holds C^T, so the direct epilogue writes 8-byte vectors instead of 2-byte scalars |
| `iglp_opt=n` | `llvm.amdgcn.iglp.opt(n)` per K tile (combines with the staged body's `sched_barrier(0)` / `s_setprio` fences; it still reorders the first MFMA region's `ds_read`s) |
| `persistent_ctas=P` | persistent tile loop; the next tile's DMA prologue is issued before the current tile's epilogue stores; launched on min(P, tiles) workgroups; with the chiplet swizzle P must be a multiple of the XCD count |

Implementation notes that mattered:

* With separate allocations the loop-invariant `ds_read` addresses multiplied and the
  16-wave body (128 VGPRs per wave) spilled. Pinning the wave-uniform parts (warp
  row offsets, the per-wave LDS DMA slice, the tile origin as the buffer `soffset`)
  in SGPRs and keying the swizzle on the lane's in-atom row brought it back under
  the budget.
* `ds_read` immediates are 16-bit: for an allocation the pool places at 64 KiB or
  more, the backend folded the base with the per-atom row constant into one
  unencodable constant and spent a VGPR per (stage, K atom, atom row). Folding the
  64 KiB multiple of the base into an opaque per-lane row keeps one base VGPR per
  operand and K atom.
* K-tail loads past the K range use a zero-size buffer resource and a zero
  `soffset`, so every tile issues the same number of DMAs and the partial `vmcnt`
  counts stay static.
* The partial `vmcnt` relies on the DMAs completing in order; the 3- and 4-stage
  rings pass the full-output check at every K-tile residue (T mod 3, T mod 4).
* The persistent body with 3 or 4 stages shares `vmcnt` between the epilogue
  stores and the next tile's prologue DMAs (gfx9 has no separate store counter).
  The first wait of the next tile is still exact: it returns only once at least
  (leftover + L + #stores) operations retired, and since the DMAs retire in order
  among themselves, the first tile's L DMAs are among them whatever the stores do.
  Hardware coverage: the 3- and 4-stage persistent variants, with and without the
  chiplet swizzle, with fewer CTAs than CUs and with more CTAs than tiles, pass at
  every residue and after the timing loop.

## Results (gfx950, same job and node for all arms, hot medians)

Method: a standalone HIP timing driver: random U(-1, 1) data, every
candidate checked against an f64-accumulated device reference over the full output
before timing and again on a fresh output after the timing loop; 20 hot calls and
10 cold calls (1 GiB read flush) per final candidate, plus a 4 s sustained window
with clock and power sampled. The full-output check applies three criteria: the
deterministic bound `2u|ref| + 4 * 2^-24 * K * sum|ab|`, a probabilistic bound with
`sqrt(K)` in place of `K` (the deterministic one grows as K^2 and, injected into a
correct output, misses a single skipped 16- or 32-wide K slice of one 16x16 block at
K = 12288; the probabilistic one flags 160-240 of its 256 outputs), and a relative
RMS bound `u + 16 * 2^-24 * sqrt(K)`. Baselines in the same job: hipBLASLt, every
solution `getAllAlgos` returns for the problem (all screened, none skipped), and
CK-Tile, the 120 batched-contraction instances hipTensor's CK-Tile codegen emits for
this layout (compv3 / compv4 / mem pipelines, cshuffle epilogue) run as a plain GEMM
(one batch, one mode per dimension) -- not CK-Tile's tile_engine universal-GEMM set.
Both from ROCm 7.2.

| dtype | shape | ring pick / rung F | ring pick / CK-Tile best | ring pick / hipBLASLt best |
|---|---|---|---|---|
| f16 | 8192^3 | 1.20 | 1.06 | 0.89 |
| f16 | 23040x24576x12288 | 1.21 | 1.04 | 0.87 |
| f16 | 15360^2x768 | 1.12 | 1.18 | 0.86 |
| bf16 | 8192^3 | 1.21 | 1.01 | 0.87 |
| bf16 | 23040x24576x12288 | 1.21 | 1.03 | 0.84 |
| bf16 | 15360^2x768 | 1.15 | 1.14 | 0.84 |

(throughput ratios; > 1 means the ring pick is faster at that configuration.)

Trace after the change (8192^3 f16, cycles per K tile per wave): the compiler's
`vmcnt(0)` is gone; the cycle count per K tile dropped by about a third; the largest
remaining items are the 16-wave per-tile barrier, MFMA issue, the DTL issue burst
and the builder's own wait for the next tile at the barrier.

Lever verdicts: `dtl_stages=2` + `interwave_setprio` is the dispatched body.
`persistent_ctas` helps at K = 768 (stores overlap the next tile's loads) and is
neutral-to-slightly-negative at large K; `mfma_swap_ab` loses (each 8-byte vector
store instruction touches 16 row segments and stalls longer than the four scalar
stores it replaces); the 3- and 4-stage rings need tile_k = 32 and lose to the
2-stage tile_k = 64 ring (twice the barriers); `iglp_opt`, `dtl_stream_barrier`,
`dtl_issue_spread` and the 8-wave fragment-prefetch body are within noise of or
below the pick. All stay selectable.

Caveats: measured on one gfx950 node; all three arms run at the board power cap
in the sustained window, so the ratios include each kernel's clock at that cap.
The dispatcher admits the ring body by a threshold rule fitted on the 79 shapes per
dtype; leave-one-out, its worst regret is 1.13 (f16) / 1.12 (bf16) (in-sample 1.03).
The rule's tile floor is an absolute tile count calibrated on a full 256-CU device;
partitioned modes and other CU counts are not calibrated, and the rule is not applied
where the default candidate would use split-K (not measured). The rung-F body keeps
its own cost-model coefficients. The timed kernels were compiled by a process pool
without torch, where rocKE loads the node's system ROCm comgr; with torch imported it
loads torch's bundled comgr, which emits different (equally correct) machine code for
the same IR. Timed side by side, the two builds of the pick and of the `iglp_opt=1`
variant are within the noise floor (about +-1.5%) of each other at all six
configurations. Within one comgr, a kernel's machine code does not depend on what
was compiled before it in the process (checked for the `iglp_opt=1` variant).
