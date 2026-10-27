# f64 / complex-f64 GETT (strided GEMM and tensor contraction)

`instances/common/gett_f64.py` computes

```text
D[l, m, n] = alpha * sum_k A[l, m, k] * B[l, n, k] + beta * C[l, m, n]
```

in f64 (real, or complex with interleaved `(re, im)` storage) with the
`v_mfma_f64_16x16x4_f64` atom and an f64 accumulator, on every target whose MMA
catalog lists that atom (`F64_GETT_ARCHS`, read from the catalog: gfx90a, gfx942
and gfx950). Every operand is read **in place through its own strides**, so a
row- or column-major GEMM, a transposed operand, a batched GEMM, and a tensor
contraction with arbitrary mode orders (hipTensor / cuTENSOR `P1`..`P5`
patterns) are all one launch, with no permute pass before or after.

## Index maps (`GettF64Problem`)

Each index space of each tensor is a list of *digits* `(extent, stride)`,
fastest first: the element offset of GEMM row `m` of `A` is
`sum_i ((m / prod_{j<i} e_j) % e_i) * s_i`. The problem holds the digits of
A's `M`, `K` and batch `L` indices, B's `N`, `K`, `L`, D's `M`, `N`, `L`, and
optionally C's (C defaults to D's layout but can have its own strides).
`gemm_problem(M, N, K, a_strides=..., b_strides=..., d_strides=..., c_strides=...,
batch=..., batch_strides=...)` builds the one-digit (strided GEMM) case.

## Compile-time and runtime sizes

By default a kernel bakes the problem in: sizes and strides are constants, so
each index map gets its own binary (the kernel name carries `M..N..K..L..` and a
hash of the digits). With `runtime_sizes=True` the kernel takes `M, N, K, L`
and every stride as kernel arguments (`RUNTIME_SIZE_ARGS`, after alpha / beta)
and serves every strided batched GEMM (one digit per index space) of one
*layout class*: which strides are 1, and the parity of an extent that an operand
is contiguous along (a two-element vector load needs an even run).
`runtime_template(problem)` is that class as a canonical problem; the spec of a
runtime kernel carries it, and `runtime_compatible(spec, problem)`,
`gett_f64_runtime_args(spec, problem)` and `gett_f64_grid(spec, problem)` check
and launch a concrete problem. Multi-digit contractions keep compile-time
maps.

## Planning a contraction (`gett_f64_plan.plan_contraction`)

Given the mode lists and element strides of A, B, D (and C), the planner:

1. classifies modes: **M** (in A and D), **N** (in B and D), **K** (in A and B,
   summed), **L** (in all three); a mode in only one tensor is rejected;
2. picks roles: the f64 accumulator holds 16 consecutive GEMM columns per row
   group, so D stores coalesce along N; the operand that owns D's fastest free
   mode becomes GEMM-B (`swapped` when that is hipTensor's A);
3. orders each index space once for every tensor that carries it (M by
   GEMM-A's strides, N by D's, K by the K-contiguous operand, L by D's) and
   merges modes that are contiguous in a tensor into one digit; the K order is
   moved, when needed, to one whose digits split at the K tile (below);
4. splits the contraction into several launches (per-launch base offsets) when
   a tensor's per-launch addressed span would reach 2^31 bytes, cutting the M /
   N mode with the largest strided span first (K is never cut), and when the
   batch of a launch would exceed the grid's y limit (`MAX_GRID_Y`): the
   largest batch mode is cut into chunks of the largest fitting size plus a
   remainder chunk.

## Kernel

* **CTA tiling.** `tile_m x tile_n` outputs per CTA, `warps_m x warps_n` waves,
  each owning a block of 16x16 atoms; `tile_k` K rows per LDS stage.
* **Loads.** Each operand tile is read with 16-byte buffer loads along the
  contiguous direction (M/N or K; 8-byte loads when neither is contiguous) into
  registers one K tile ahead of the MFMAs (`prefetch_depth=2`: two tiles
  ahead, the next tile carried in registers through the loop), then stored to
  the other of two LDS stages: one barrier per K tile. With `lds_stages=1` the
  CTA keeps one LDS stage and adds a barrier before the stores, which halves
  its LDS: a twice-as-deep `tile_k` at the same occupancy (on the 64 KiB-LDS
  targets that makes K-contiguous rows 128 bytes, a full cache line).
  Scheduling barriers keep the prefetch loads above the MFMA block and the LDS
  stores (which wait for them) below it. The last K tile has no next tile, so
  it issues no LDS store and no barrier; a prefetch past the last K tile loads
  zeros without touching memory (also when `k_stagger` would wrap it onto a
  real tile). Out-of-range rows, columns and K rows get an out-of-bounds
  buffer offset, so the hardware returns zero without a branch.
* **Addressing.** When the K digits split at the K tile
  (`koff(kt*T + kl) = ktile(kt) + klocal(kl)`, see `k_split`), the K-tile
  offset and the batch offset move a 64-bit base pointer and every K tile gets
  its own buffer resource; the per-lane offsets (rows plus one K tile) are
  loop-invariant 32-bit values. `arch_spec` / `pick_tile_k` choose a K tile at
  which both operands split. The batch runs on grid y (at most `MAX_GRID_Y`
  per launch; `is_valid_spec` rejects more).
* **Lane layout.** The fragment each lane reads and the output element each
  accumulator slot holds come from the atom's catalog lane maps
  (`a_layout`, `b_layout`, `c_layout` of `mfma_f64_16x16x4_f64`), the same
  maps the f64 MMA probe (`instances/common/f64_mma_probe.py`) checks.
* **LDS layout and swizzle.** `[k][mn]`, the layout of both f64 MFMA operands
  (one `ds_read_b64` per fragment; complex: one `ds_read_b128` of the pair).
  With `lds_swizzle` (real data, tile sides multiples of 32) K row `k` stores
  column `c` at `c ^ (16*(k & 1) ^ 8*((k >> 2) & 7))` (masked to the 64- or
  32-column block): the two K rows a half-wave reads use disjoint banks, and
  the K-contiguous loader's column writes spread over the banks.
  `lds_swizzle_mode` selects the function: 1 is `16*(k & 1) ^ 4*((k >> 1) & 3)`
  and 2 is `4*((k >> 1) & 7)`. The K-contiguous loader stores a K pair per
  lane, so the rows `2c` one store instruction writes land in distinct
  32-byte bank groups (mode 2: all eight; mode 1: four, keeping the odd-row
  flip of mode 0 for conflict-free fragment reads).
* **Complex (fused 4M).** Complex data stay interleaved. Each complex atom is
  four real MFMAs into a real and an imaginary accumulator
  (`Cr += ar*br - ai*bi`, `Ci += ar*bi + ai*br`). Conjugating an operand only
  moves the sign: the negated fragment is one `v_xor` of its sign bit, built
  once per k-step and reused by every atom. `complex_accs=4` keeps the four
  real products (`ar*br`, `ai*bi`, `ar*bi`, `ai*br`) in their own
  accumulators and applies every sign (the subtraction, CONJ) once in the
  epilogue: no negation in the K loop for twice the accumulator registers.
* **Blocked K accumulation.** `k_block=n` restarts the accumulators every `n`
  K tiles and adds each block's sum into a second set of f64 accumulators.
* **Split K.** `split_k=S` runs two kernels. The partial-sum kernel launches
  `S` CTAs per output tile (grid z): split `z` sums the K tiles
  `[floor(z*T/S), floor((z+1)*T/S))` of the `T` K tiles and stores its raw sum
  to a workspace `W[S][L][M][N]` (f64; complex: interleaved pairs), with no
  epilogue. The reduce kernel (`build_gett_f64_reduce`, `REDUCE_THREADS`
  threads per CTA, grid from `gett_f64_reduce_grid`) adds the `S` partials of
  each output in split order and applies the epilogue below. The workspace
  size is `gett_f64_workspace_bytes(spec, problem)`; the caller allocates it
  and passes it as `W` to both kernels (`gett_f64_signature`,
  `gett_f64_reduce_signature`). The order is fixed, so the result is bitwise
  repeatable. `split_k` excludes `unroll2` and `k_stagger`, and every split
  covers at least one K tile.
* **Epilogue.** `d = fma(alpha, acc, beta * c)` in f64, stored straight to D's
  strides; complex: the complex products with fused multiply-adds. alpha and
  beta are f64 kernel arguments (no narrowing). C is read through its own
  strides; `bilinear=False` drops C and beta from the signature. The BLAS rules
  hold: with `beta == 0` (also `-0`) C is not read, so a NaN in C does not
  reach D; with `alpha == 0` the accumulator is not used (`d = beta * c`, and
  `0` when beta is 0 too), so a NaN or Inf in A or B does not reach D. A
  complex alpha (or beta) whose imaginary part is zero scales both components
  by its real part, so an infinite component of the accumulator (or C) does
  not turn the other component into NaN through `0 * Inf`.
* **Pointers.** A and B are `noalias readonly`; C (`readonly`) and D carry no
  `noalias`, so an in-place update with C and D the same tensor is valid. All
  four need only 8-byte alignment (an f64 subtensor view); the buffer loads
  and stores need 4.

## Numerics

The accumulation is a chain of f64 fused multiply-adds per output, in K
order: on gfx90a, gfx942 and gfx950 the device result is bit-identical to a
host model that applies one correctly rounded fma per product in K order
(each `v_mfma_f64_16x16x4` step behaves as four sequential fmas, not as one
rounding of four products). The result does not depend on the tile shape, the
K tile, the LDS layout, the operand layout, the prefetch schedule, the loop
unrolling or `runtime_sizes` (up to the sign of an exact zero, see
`runtime_sizes` with `unroll2`). Three knobs reorder the sum and change the
bits, all within the same bound: `k_stagger` (rotates the K order per CTA),
`complex_accs=4` (sums the four products separately), `k_block` (blocked
summation) and `split_k` (one chain per split, then the ordered sum of the
splits). For real data

`|d - t| <= u*|t| + (K + 2)*u*(|alpha|*sum|a||b| + |beta|*|c|)`

with `u = 2^-53`; complex outputs see `2K + 4` roundings per component. The
`+ 2` is the epilogue: `beta * c` and the fma round once each (the result of
`alpha * acc + beta * c` is not a single rounding). With `k_block=n` an output
goes through about `n*tile_k + K/(n*tile_k)` roundings of the accumulation
instead of `K`, which shortens the error growth at long K (the bound above
still holds); with `split_k=S` it goes through about `K/S + S` (and the two
combine: `k_block` inside each split). Integer-valued inputs whose products and sums are exact in f64
give exact results.

f64 data with f32 compute (hipTensor's `32F` compute descriptor) is served by
the same f64 path. That is at least as accurate as f32 accumulation, and on
gfx90a and gfx942 the f64 MFMA atom has the throughput of the f32 one. (On
gfx950 an f32-MFMA loader would be faster for that combination; it would be a
default-off knob.)

Conjugation is an exact sign flip, so `CONJ(A)` gives the same bits as
conjugating A on the host; scaling an operand by a power of two scales D
exactly, and negating A (and C) negates D exactly.

## Knobs (`GettF64Spec`)

| knob | default | meaning |
|---|---|---|
| `tile_m`, `tile_n` | 128, 128 | CTA tile (complex: 64, 64 via `arch_spec`) |
| `tile_k` | 16 (`arch_spec`: per arch and problem) | K rows per LDS stage |
| `warps_m`, `warps_n` | 2, 2 | waves per CTA |
| `waves_per_eu` | `None` (`arch_spec`: 2) | occupancy hint |
| `lds_swizzle` / `lds_pad` | on / `None` | bank-conflict avoidance |
| `group_m` | 8 | raster: M tiles walked per N tile |
| `xcds` | 0 | CTA-id remap so one chiplet runs consecutive tiles |
| `prefetch` | on | global loads one K tile ahead |
| `prefetch_depth` | 1 | K tiles the register prefetch runs ahead (1 or 2) |
| `lds_stages` | 2 | LDS stages (2: one barrier per K tile; 1: half the LDS, two barriers) |
| `lds_swizzle_mode` | 0 | swizzle function (0, 1, 2; see above) |
| `complex_accs` | 2 | complex accumulators per atom (2, or 4 with the signs in the epilogue) |
| `unroll2` | off | K loop unrolled by two tiles (needs two LDS stages, prefetch depth 1) |
| `k_stagger` | 0 | start each CTA at a staggered K tile and wrap (reorders the K sum) |
| `k_block` | 0 | K tiles per accumulation block (0: one chain; even with `unroll2`) |
| `runtime_sizes` | off | sizes and strides as kernel arguments (one kernel per layout class) |
| `split_k` | 1 | K splits: partial sums into a workspace, then an ordered reduce kernel with the epilogue |
| `sched_fence` | on | scheduling barriers around the MFMA block |
| `bilinear` | on | read C (`beta * C`) |

Python-engine levers (default off; the C++ builder mirror does not have them
yet, so `TUNED_CONFIGS` and the byte-identity family do not use them; the
kernels they produce still lower through either engine):

| knob | default | meaning |
|---|---|---|
| `acc_agpr` | off | accumulators in AGPRs (`amdgpu-agpr-alloc` + the AGPR MFMA form + an opaque AGPR zero seed): up to 256 accumulator registers beside the fragments without spills |
| `kmajor_lds` / `kmajor_pad` | off / 2 | real: an operand read along K gets a K-major LDS tile (`[stage * side + mn][k]`, padded rows): one 16-byte LDS store per 16-byte load and conflict-free fragment reads |
| `ilv` / `ilv_wgap` | 0 / 4 | real: the instruction-level interleaved K loop (below); `ilv` = MFMAs after the barrier |
| `frag_pipe` | off | the fragments of the next k-step read into a second register set before the MFMAs of this one; the LDS store and barrier of the next tile before the last k-step |
| `k_inner` / `mfma_fence` | 1 / off | real: MFMAs of `k_inner` k-steps back to back per accumulator; a scheduling barrier after every MFMA |
| `split_serial` | 0 | with `split_k`: reduce the partials inside the launch (1: agent-scope fences; 2, gfx942 / gfx950: write-through partial stores and loads) instead of the reduce kernel |

**The interleaved K loop (`ilv`).** Each K tile is emitted as one fixed
instruction schedule: the tile's MFMAs in order, and between them the
fragment reads of the next k-step group (spread over the current group), the
LDS stores of the next K tile (it rides in registers: `prefetch_depth` 2),
each followed by the global load of the same pass of the tile after it into
the freed registers (spread up to `ilv_wgap` MFMAs before the barrier), an
LDS-only barrier (`s_waitcnt lgkmcnt(0)` + `s_barrier`: the global loads stay
in flight) `ilv` MFMAs before the end of the tile, and after it the reads of
the next tile's first group. A scheduling barrier after every MFMA keeps the
order. With `unroll2` two K tiles per iteration read compile-time LDS stages
(no per-tile address arithmetic). Every read of a stage completes before the
barrier that lets the next tile overwrite it. None of these levers changes
the K order of an output (`split_serial` sums the partials in the order of
the reduce kernel), so the bits equal the default kernel's.

Why: an ablation of the K loop (global loads, LDS stores, barriers and
fragment reads removed one by one at build time) shows that the MFMA loop
alone runs at the matrix ceiling with any number of accumulators per wave,
global loads are hidden, and the cost is in the LDS stores: issued as a
burst before the barrier they drain the matrix pipe, and the K-contiguous
operand's two-row scatter adds bank conflicts. Spreading every store between
MFMAs, a K-major tile for K-contiguous operands and AGPR accumulators (so a
CTA can hold a 256x128 tile at one or two waves per SIMD) remove that cost.

`arch_spec(problem, arch, **knobs)` gives the reference defaults (two K
stages of 16 rows on gfx950, 8 on the 64 KiB-LDS targets, so two CTAs fit per
CU). `tuned_spec(problem, arch, num_cus=...)` gives the tuned configuration of
the problem's shape class (below), which is what dispatch picks.

## Tuned configurations

`TUNED_CONFIGS` maps (arch, real / complex, shape class) to a knob set;
`gett_f64_shape_class(problem, arch, num_cus)` picks the class. The classes
depend on the CU count of the part (`resolve_num_cus`: the caller's
`num_cus`, else the visible device's when it is that arch; parts that share a
gfx id differ in CU count, so there is no per-arch default):

| class | real | complex |
|---|---|---|
| `small` | the 128x128 grid is under one tile per CU (gfx90a: two) | the 64x64 grid is under four tiles per CU (gfx942: two) |
| `large` | gfx950: K >= 4096 or >= 8 tiles of 128x128 per CU | K >= 2048 (gfx950, gfx90a) |
| `kk` / `kmixed` | gfx90a: both / one operand K-contiguous | - |
| `pow2` | gfx90a, no K-contiguous operand: an operand's K rows are a multiple of 64 KiB apart | - |
| `wide` | gfx90a, no K-contiguous operand: >= 64 tiles of 128x128 per CU and K <= 1024 | gfx90a: >= 64 tiles of 64x64 per CU |
| `medium` | otherwise | otherwise |

The configurations use the single LDS stage with a second barrier (it frees
the LDS for a deeper K tile and keeps the fragment addresses loop-invariant),
the K loop unrolled by two tiles (`unroll2`: compile-time LDS stages, so the
fragment reads need no address arithmetic, at one barrier per K tile; the
large-problem configuration on gfx950), the K-pair swizzles, and prefetch
depth 2 (complex on the 64 KiB-LDS targets). `complex_accs` and `k_stagger`
are default-off knobs.

`tuned_spec` then applies the long-K rule: when the output tiles give fewer
than `SPLIT_CTAS_PER_CU` CTAs per CU, it sets `split_k` to the smallest power
of two that reaches that many CTAs, capped at `SPLIT_K_MAX` and at the
largest split whose K range is still `SPLIT_MIN_K` elements or longer (and
drops `unroll2` and `k_stagger`); when the K range of one CTA is still `K_BLOCK_MIN_K` or longer
it also sets `k_block` to `K_BLOCK_ELEMS / tile_k` tiles. Knobs passed to
`tuned_spec` override both.

## Pipelined configurations

`PIPE_CONFIGS` maps (arch, real, pipe class) to a knob set using the levers
above; `gett_f64_pipe_class(problem, arch, num_cus)` picks the class and
`pipe_spec` the spec (None where no class applies):

| arch | class | rule | configuration |
|---|---|---|---|
| gfx950 | `long` | a `large` problem whose A and B contiguous runs are at least `PIPE_LONG_RUN` elements (or `PIPE_MIN_RUN` with at least `PIPE_SHORT_MAX_TILES_PER_CU` 128x128 tiles per CU), whose D run is at least half of `PIPE_LONG_RUN`, with K >= `PIPE_LONG_MIN_K` and a last wave of 256x128 CTAs at least `PIPE_LONG_MIN_WAVE_EFF` busy (one CTA per CU: a ragged last wave idles CUs) | 256x128x16 tiles, 4x2 waves (two per SIMD from one CTA), AGPR accumulators, K-major LDS, `ilv` |
| gfx950 | `short` | a `medium` problem, or a `large` one under `PIPE_SHORT_MAX_TILES_PER_CU` 128x128 tiles per CU whose shorter A / B run is in [`PIPE_MIN_RUN`, `PIPE_LONG_RUN`) (interleaved contraction layouts) | 128x64x16 tiles, 4x1 waves, three CTAs per CU, `ilv` |
| gfx950 | `wide` | a `large` problem with K < `PIPE_LONG_MIN_K` | 128x128x16 tiles, 2x4 waves, padded LDS rows, AGPR accumulators, K-major LDS, `ilv` |
| gfx950 | `tiny` | a `small` problem with at most one 64x64 tile per CU | 64x64x16 tiles, 2x2 waves, `ilv` |
| gfx950 | `small` | the other `small` problems | 32x32x32 tiles, 2x2 waves, four CTAs per CU, `ilv` |
| gfx950 | complex `long` | at least one 128x128 tile per CU and K >= `PIPE_COMPLEX_MIN_K` | 128x128x8 tiles, 4x2 waves, AGPR accumulators, `ilv` (each fused 4M atom is one slot) |
| gfx90a | `long` | at least `PIPE_GFX90A_LONG_TILES_PER_CU` 128x128 tiles per CU and K >= `PIPE_GFX90A_LONG_MIN_K` | 256x128x8 tiles, 2x2 waves, AGPR accumulators, K-major LDS, `ilv` |
| gfx90a | complex `long` | at least `PIPE_GFX90A_LONG_TILES_PER_CU` 128x128 tiles per CU and K >= `PIPE_COMPLEX_MIN_K` | 128x128x8 tiles, 4x2 waves, AGPR accumulators, `ilv` |
| gfx90a | `short` | fewer tiles | 128x64x16 tiles, 4x2 waves, padded LDS, `ilv` |

Complex data below those sizes or K, gfx950 problems with very short runs
or only a short D run, gfx90a problems with many tiles and a short K, requests with
`runtime_sizes`, problems whose tuned spec sums K in blocks or splits a K of
`K_BLOCK_MIN_K` or more (the shorter rounding chains), and gfx942 (not tuned
yet) keep the tuned configurations. The configurations came from the search
engine (the tuned problems' winners, then cross-evaluated on held-out
problems).

## Dispatch

`rocke.dispatch.families.gett_f64` registers `gett_f64_pipe` (first choice
where `pipe_spec` applies), `gett_f64_tuned` (`tuned_spec`), then the
reference geometries `gett_f64_t128` and `gett_f64_t64`
(the 64x64 geometry when the 128-tile grid would leave CUs idle), on every
target in `F64_GETT_ARCHS`. A request carries the problem, the arch, the CU
count (`num_cus`, or queried from a device of that arch) and optionally
`runtime_sizes`: then the selected spec carries the layout template, so every
problem of one layout and shape class maps to one kernel, and the grid comes
from the request's problem. A selected spec with `split_k > 1` needs the
reduce kernel and a workspace (above) next to the partial-sum kernel. The complex dtype is named `c128` (two f64 per
element). Each candidate's sweep space starts at its spec, then the arch's
other tuned configurations of the dtype, then one-knob moves.

## Relation to UniversalGemm

`UniversalGemmSpec` takes `M`, `N`, `K` as kernel arguments and accepts fp32
accumulation only. A `runtime_sizes` GETT kernel is an f64 strided batched GEMM
with the same runtime-size convention plus per-operand strides, a batch, f64
alpha / beta and C read through its own strides: an f64 `DataSpec` (fp64
operands, fp64 accumulation) can route to it, with the BLAS layout
(`RCR`, `RRR`, ...) fixing which strides are 1 in the template. The C++ API
(`rocke/instance_gett_f64.h`) builds, validates, names and grids a resolved
spec, and gives the runtime template, compatibility check and size arguments,
and the split-K workspace size, reduce kernel name and reduce grid, so a
host without Python can launch precompiled runtime-size kernels; the
contraction planner and the tuned selection stay in the Python dispatch layer.

## Tests

* `tests/instances/test_gett_f64.py`: K splitting, planner digits against
  direct tensor offsets (random contractions, padded strides), role choice,
  launch splitting coverage (spans and the batch limit), spec validation,
  kernel structure at every LLVM flavor, the LDS bank model of every swizzle
  mode, one and two LDS stages, the peeled last K tile, `k_block`, `split_k`
  (workspace, reduce kernel, grid, validity), runtime
  sizes (one kernel per layout class, arguments, grid, compatibility), pointer
  attributes, the catalog arch list, and that the byte-identity family covers
  every tuned configuration.
* `tests/dispatch/dispatch_tests/gett_f64/`: selection (the tuned candidate
  first), the shape classes and their CU count, validity of every tuned
  configuration, the long-K rule (`split_k`, `k_block`), sweep space,
  runtime-size requests, builds.
* `tests/instances/test_gett_f64_levers.py`: validity of the Python-engine
  levers and their combinations, kernel names, the `split_serial` signature,
  workspace and counters and its fences / write-through accesses, the AGPR
  reservation, the K-major LDS tile sizes, the `ilv` barriers, builds on
  every target, and the pipelined classes and dispatch candidate.
* `tests/numeric/test_gett_f64_numeric.py` (every target with the atom):
  every layout, tails, tiles and knobs, scale and `beta == 0` with NaN C, C with
  its own strides and batch, exact integer results, complex with every
  conjugation, a hipTensor `P1` contraction, a multi-launch split, a batch over
  the grid limit, blocked K, split K (bitwise repeatable, with the BLAS
  rules in the reduce kernel), runtime sizes (bitwise against the compile-time
  kernel), against exact (two-product + `fsum`) references; and bitwise on whole
  outputs: NaN / Inf propagation, `beta == 0` / `-0` with NaN C, NaN alpha /
  beta, `alpha == 0` with NaN / Inf in A and B, real complex scalars with an
  infinite component, and CONJ against host-conjugated data; every
  Python-engine lever bitwise against the default kernel on three layouts,
  `split_serial` (both modes, bitwise against the reduce kernel, counters back
  at zero after repeated launches) and every `PIPE_CONFIGS` entry.
* `tests/instances/parity/gett_f64_emit.{py,c}`: the byte-identity family (the
  C++ mirror `cpp/instances/common/gett_f64.cpp`, API
  `rocke/instance_gett_f64.h`): every load direction and knob, blocked K,
  split K (partial-sum and reduce kernels), runtime sizes, and every `TUNED_CONFIGS` entry on a problem of its own shape
  class. The C cases are generated from the Python specs
  (`python gett_f64_emit.py --c-cases`).
