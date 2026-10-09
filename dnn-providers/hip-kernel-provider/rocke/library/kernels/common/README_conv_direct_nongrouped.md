# Direct Non-grouped Convolution (`groups == 1`)

Source: [`conv_direct_nongrouped.py`](conv_direct_nongrouped.py) —
`DirectNongroupedConvSpec`, `is_valid_nongrouped_spec`, `build_direct_conv_nongrouped`,
`nongrouped_specs`, `nongrouped_knobs`, `tile_w_candidates`.

C++ engine twin: `platform/cpp/instances/common/conv_direct_nongrouped.cpp`
(public header `rocke/instance_conv_direct_nongrouped.h`), byte-identical `.ll`.

## Why a separate family

The grouped direct-conv kernels ([`README_conv_direct_grouped.md`](README_conv_direct_grouped.md))
take all their parallelism from the `groups` axis: one wave owns one group. At
`groups == 1` that degenerates to a single wave per output row holding
`ceil(K/16)` accumulator tiles, which leaves the machine almost idle. This family
is the `groups == 1` counterpart. It keeps the property that makes a direct conv
worth having next to implicit GEMM on 3x3, stride ≤ 2 shapes:

> the input tile is staged in LDS **once per channel chunk, with its halo**, and
> all `KH*KW` filter taps read shifted sub-tiles of that single staged copy.

An implicit-GEMM formulation re-reads the activations once per tap, so its
activation-side traffic is roughly `KH*KW` times larger and the L2 has to absorb
the redundancy.

## Operands

| Role | Tensor | Layout | Notes |
|------|--------|--------|-------|
| A | input activations | NHWC `[N, H, W, C]` | `C % 8 == 0` (dwordx4 staging loads) |
| B | weights | KRSC `[K, KH, KW, C]` | |
| D | output | NHWK `[N, Ho, Wo, K]` | `K % atom_tile == 0` |

fp16 and bf16 I/O, f32 accumulation. `stride ∈ {1, 2}`, any `KH`/`KW`/`PAD`.

## AOT kernel arguments

The kernel is AOT: it declares the direct-conv kernarg block of
[`conv_abi.conv_direct_arg_names(direction="fwd")`](conv_abi.py), the same
block every direct-conv kernel takes:

```text
A, B, D, A_bytes, B_bytes, D_bytes,
p_N, p_Hi, p_Wi, p_Ho, p_Wo, p_groups, p_total_c, p_total_k,
p_A_stride_{n,hi,wi}, p_D_stride_{n,ho,wo}
```

| Runtime (kernarg) | Baked (capability / knob) |
|-------------------|---------------------------|
| `N`, `H`, `W`, `Ho`, `Wo`, NHWC / NHWK strides | `KH`, `KW`, `stride`, `PAD`, dilation, dtype |
| `C` (`p_total_c`), `K` (`p_total_k`) | tile, waves, atom, `ck`, `lds_pad`, swizzle, schedule knobs |

Dilation (`DirectConvProblem.dil_h` / `dil_w`) widens the staged halo to
`(tile-1)*stride + (K-1)*dil + 1` and spaces the taps `dil` rows / columns apart
in it; the kernel is the only direct variant that computes a dilated
convolution. The AOT cache bakes a few dilated 3x3 rows next to the undilated
ones (`direct_kernel_sweep._NONGROUPED_DILATED`).

Unlike the grouped kernels, the channel counts are runtime too: they only set
the channel loop's trip count and the staging / store masks. `p_groups` is
ignored (the family is `groups == 1`). One binary therefore serves every shape
whose `C` is a multiple of `ck` (and of 8) and whose `K` is a multiple of the
atom tile; `spec.problem` only has to pass the validator, and the emitted IR
does not depend on it (`library/tests/test_conv_abi.py` asserts this).

Launch with the shared helpers:

```python
from kernels.common.conv_abi import conv_direct_args_signature
from kernels.common.conv_args import ConvArgs

signature = conv_direct_args_signature(dtype)
values = ConvArgs.from_problem(problem).to_launch_values(
    a_ptr, b_ptr, d_ptr, a_bytes, b_bytes, d_bytes)
```

Two host-side contracts move from build time to launch time, so run
`validate()` (or `is_valid_nongrouped_spec`) on the spec rebuilt for the
problem actually launched — the benchmark sweep and the AOT cache's
`plan_for` both do:

* every operand tensor must end at or below the `0x7F000000` masked-access
  offset (see optimization 5);
* the flattened grid must fit `gridDim.x` (2³¹ − 1).

## Computation

Per workgroup, with M = output channels and N = output pixels:

```text
tile_k              output channels   (M; multiple of the atom tile)
tile_h x tile_w     output pixels     (N; tile_w a multiple of the atom tile)
ck                  input channels reduced per LDS stage

for c0 in range(0, C, ck):                 # runtime scf.for; accumulators carried
    stage X[(tile_h-1)*s+KH, (tile_w-1)*s+KW, ck] -> LDS   (halo included)
    stage W[tile_k, KH, KW, ck]                   -> LDS   (MFMA fragment order)
    for k_atom, s:                          # Python-unrolled
        read activation fragments for every input row of the wave window
        for r:
            read weight fragments for tap (r, s)
            for row, col-block, m-tile:  acc += mfma(W_frag, X_frag[row*stride + r])
```

Waves split the block `waves_m x waves_n`: `waves_m` along output channels,
`waves_n` along output rows; every wave keeps all `tile_w` columns.

## The optimizations

1. **LDS halo reuse.** The activation tile is staged with its `KH-1` / `KW-1`
   halo once per channel chunk; all taps read shifted sub-tiles. This is the
   reason the family exists.

2. **Tap-shared activation fragments**.
   Output row `row` at tap row `r` reads input row `row*stride + r`. Indexing
   activation fragments by *input* row lets the `KH` taps of one filter column
   share them: `(ROWS_W-1)*stride + KH` distinct fragments per `(s, k_atom)`
   instead of `ROWS_W*KH`. Filter column `s` is the outer loop so every fragment
   loaded for one `(s, k_atom)` is consumed by all `KH` tap rows and all
   channel tiles of the wave before it dies.

3. **Weights pre-swizzled into MFMA fragment order.** Weight slot
   `((tap*M_TILES + m_tile)*KATOMS + k_atom)*64 + lane` holds exactly the
   `FRAG` halves that `lane` feeds to the MFMA. Both the staging store and the
   consuming read are linear in `lane`, so neither side can bank-conflict and no
   padding is needed. The staging thread's global read is `FRAG` contiguous
   channels of one `k_out` row — contiguous in KRSC.

4. **Conflict-free activation reads via `lds_pad`.** The activation tile is
   stored `[pos][c]` with channel stride `ck + lds_pad`. The default pad of 8
   halves spreads a `ds_read_b128` wave across all banks on gfx950 (64 banks, see
   [`arch/gfx950.md`](../../../platform/dsl_docs/optimization/arch/gfx950.md)).
   Neighbouring pads (4, 16) conflict; treat the value as tuned, not arbitrary.

5. **Hoisted staging predication**.
   The validity masks of the staging loads (image border, `K` tail, pass tail)
   are loop-invariant. They are folded once, before the channel loop, into the
   *base* byte offset: `select(valid, base, 0x7F000000)`. A buffer load whose
   offset lies past `num_records` returns zero — exactly the value a padded pixel
   or an out-of-range filter row needs — so the loop body carries no compare,
   `and` or `select` at all. The epilogue redirects its masked stores to the
   same offset (one sentinel for every masked access), where the hardware drops
   them. `validate()` guarantees the trick holds: every operand tensor must end
   at or below `0x7F000000` bytes, and `0x7F000000` plus the largest per-chunk
   channel offset must still fit in i32. The tensor sizes are kernargs, so this
   holds only for problems the spec was validated against — see
   [AOT kernel arguments](#aot-kernel-arguments).

6. **Software prefetch through loop-carried registers**
  . The global loads
   for chunk `i+1` are issued right after the barrier that publishes chunk `i`
   and travel into the next iteration as `scf.for` iter-args, so chunk `i`'s
   MFMAs cover the DRAM latency of chunk `i+1`. The tail iteration loads one
   chunk past `C`; buffer loads clamp rather than fault, so no masking is needed.

7. **Square 16-wide atoms for widths that are not a multiple of 32.** With a
   32-wide pixel tile the last tile of each row computes columns past `Wo`, and
   that wasted MFMA work is spread over every block of the row.
   `16x16x32` / `16x16x16` atoms plus `tile_w_candidates(Wo, atom_tile)` — which
   prefers widths that divide `Wo` exactly — remove the waste. All four atoms
   are square (`M == N == tile`), which is what lets one code path serve them.

8. **`iglp_opt(0)` on the channel loop**.
   The canned MFMA/memory interleave was the largest single scheduling win;
   level 1 was worse than no hint at all. `iglp=None` leaves the scheduler alone.

9. **Chiplet-aware grid decode.** The 1-D grid of (spatial cell × channel tile)
   is remapped with `chiplet_aware_super_tile_dynamic` (the tile counts follow
   the runtime extents) so the workgroups one XCD receives share operands; `swizzle_wgm` is the number of
   channel tiles walked back-to-back per spatial cell. Ordering the channel tile
   as the fast-varying axis is what matters; `swizzle_wgm=1` (spatial fastest)
   is clearly worse, while the chiplet remap on top contributes little.

10. **A clamped runtime trip count.** The channel loop runs `max(C/ck, 1)`
    times. `validate()` makes the clamp a no-op (`C` is a positive multiple of
    `ck`), but it lets LLVM prove the loop runs: with a bare `C/ck` it keeps a
    zero-trip guard and schedules the prefetch loads late in the body.

11. **Guards that turn hangs and silent corruption into errors.**
    * `validate()` rejects tiles needing more than 256 accumulator registers per
      lane: past that the backend scheduler does not just slow down, it hangs.
    * `validate()` rejects a flattened grid past 2³¹ − 1 workgroups (the kernel
      computes the workgroup count in i32), and `is_valid_nongrouped_spec`
      never raises — the arch-table queries sit inside its error handling too.
    * Staging loops are sized in whole thread-passes, so the last pass can own
      slots past the tile. Both LDS arrays carry a scratch tail those slots are
      redirected to; without it they overwrite the neighbouring allocation (symptom: row 0 correct, row 1
      wrong).

### Levers measured and rejected

Kept as knobs, off by default, because the sweep found them unhelpful on the
target shapes:

| Lever | Knob | Why it did not help |
|-------|------|---------------------|
| LDS ping-pong (1 barrier per chunk instead of 2) | `double_buffer=True` | Doubles LDS, which drops the CU from two resident blocks to one — the lost overlap costs more than the saved barrier. |
| Occupancy hint | `waves_per_eu` | Mattered only without `iglp_opt(0)`; negligible with it. |
| Wider tiles on the shapes that still trail | sweep axes | No change: the residual is barrier / `waitcnt` exposure at two waves per SIMD, not bandwidth. |

## Spec

```python
@dataclass(frozen=True)
class DirectNongroupedConvSpec:
    problem: DirectConvProblem      # groups must be 1
    name: str = "direct_conv_nongrouped"
    tile_h: int = 16                # output rows per workgroup
    tile_w: int = 32                # output cols per workgroup
    tile_k: int = 128               # output channels per workgroup
    ck: int = 16                    # input channels per LDS chunk
    waves_m: int = 2                # waves along tile_k
    waves_n: int = 4                # waves along tile_h
    atom: str = "32x32x16"          # 32x32x16 | 32x32x8 | 16x16x32 | 16x16x16
    wave_size: int = 64
    lds_pad: int = 8
    chiplet_swizzle: bool = True
    swizzle_wgm: int = 8
    chiplet_chunk: int = 64
    num_xcds: int = 8
    double_buffer: bool = False
    iglp: int | None = None         # nongrouped_specs sweeps (0,)
    waves_per_eu: int | None = None # nongrouped_specs sweeps (None, 3)
```

Constraints (`validate()` then `is_valid_nongrouped_spec(spec, arch)`):

* `groups == 1`, dtype fp16/bf16, atom known and present on the target
  (`32x32x16` and `16x16x32` need gfx950; `32x32x8` / `16x16x16` also gfx942);
* tile, wave, `ck`, stride, `C` and `K` values positive (checked before any
  derived geometry, so a bad spec gets a reason, not a `ZeroDivisionError`);
* `tile_w`, `tile_k`, `K` multiples of the atom tile; `tile_h % waves_n == 0`;
  `(tile_k / atom_tile) % waves_m == 0`;
* `ck % atom_k == 0`, `ck % 8 == 0`, `C % ck == 0`, `C % 8 == 0`;
* `lds_pad` a non-negative multiple of 8 (a negative pad overlaps neighbouring
  pixels in LDS; the vec8 LDS access is emitted `align 16`, so the pixel stride
  `ck + lds_pad` must stay a multiple of 8 halves); `swizzle_wgm >= 1`; `iglp` None or ≥ 0; `waves_per_eu` None or ≥ 1
  (the C++ port encodes None as -1 / 0, so those values are reserved in both
  engines); flattened grid ≤ 2³¹ − 1 workgroups; A, B and D each ≤
  `0x7F000000` bytes; accumulators ≤ 256 registers per lane;
* `stride ∈ {1, 2}`; threads, wave size and `lds_bytes` within the arch limits.

Usage:

```python
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_nongrouped import (
    DirectNongroupedConvSpec, build_direct_conv_nongrouped, is_valid_nongrouped_spec,
)

p = DirectConvProblem(N=4, H=64, W=64, groups=1, cpg=640, kpg=640, dtype="bf16")
spec = DirectNongroupedConvSpec(problem=p, tile_h=16, tile_w=64, tile_k=128, ck=16,
                          waves_m=2, waves_n=4, iglp=0)
ok, why = is_valid_nongrouped_spec(spec, arch="gfx950")
kernel = build_direct_conv_nongrouped(spec, arch="gfx950")
```

## Launch grid

```text
grid  = spec.grid() = (n_w_tiles * n_h_tiles * N * n_k_tiles, 1, 1)
block = (spec.threads_per_block, 1, 1)         # waves_m * waves_n * 64
```

The grid is flat; the kernel recomputes the tile counts from `p_Wo`, `p_Ho`,
`p_total_k` and `p_N` and decodes `(k_tile, spatial cell)` from the block id
(through the chiplet swizzle when enabled). The host computes the grid from the
problem it launches, so one binary is launched with whatever grid the shape
needs.

## Candidate generation and benchmarking

`nongrouped_specs(problem, arch=...)` returns every valid, name-deduplicated
spec over `tile_h ∈ {8, 16}`, `tile_k ∈ {32, 64, 128, 256}`,
`ck ∈ {16, 32, 64}`, five wave layouts, and per MFMA tile size the widest-K
atom the target supports (`32x32x16` / `16x16x32` on gfx950, `32x32x8` /
`16x16x16` on gfx942) with the `tile_w_candidates` of each atom.
`benchmarks/common/benchmark_direct_conv.py` routes `groups == 1` shapes to
this sweep:

```bash
python benchmarks/common/benchmark_direct_conv.py --N 4 --Hi 64 --Wi 64 --C 640 --K 640 --groups 1 --verify
```

Every candidate a sweep can pick must be correct, not just the hand-picked
geometries, so the correctness suite runs an evenly spaced sample of
`nongrouped_specs` on silicon.

`nongrouped_specs` filters `nongrouped_knobs(arch, dtype, Wo=...)`; without
`Wo` the same generator yields every width up to six atoms, which is the grid
the AOT kernel cache builds (variant `direct_nongrouped` in
`benchmarks/common/direct_kernel_sweep.py`, 3x3 filters at stride 1 and 2,
`cpg = kpg = 0` in the identity, i.e. runtime channels). At run time the cache
offers a problem the binaries whose width `tile_w_candidates(Wo, atom_tile)`
would pick and whose spec validates on the real shape.

The AOT grid also drops tiles that would spill registers
(`nongrouped_register_reason`): accumulators that fill the 256-entry
accumulator file, or an estimated live set (`nongrouped_live_regs`:
accumulators, carried prefetch, one step's operand fragments) above 80% of the
registers a wave gets once the block's waves share the SIMDs. On the gfx950
grid this cuts 924 kernels to 658 and, with `waves_per_eu` unset, leaves no
spilling binary; the spilling ones were also the slowest to compile. It is a
heuristic, not a validity rule: the per-shape
`nongrouped_specs` sweep still offers those tiles.

## Tests and gates

| Gate | What it proves | Command |
|------|----------------|---------|
| C++ ↔ Python byte-identity | the C++ port emits identical `.ll` for 15 configs covering every atom, dtype, stride, filter size, partial tile, uneven staging pass, DB/SB, swizzle on/off, iglp, waves_per_eu, gfx942/gfx950 | `python platform/tools/check_byte_identity.py --only conv_direct_grouped` (also with `ROCKE_LLVM_FLAVOR=llvm22`) |
| Emitters | the two sides of that gate; shared with the grouped direct-conv family, non-grouped configs are indices 42+ (dilated: 63+) | `tests/parity/conv_direct_grouped_emit.{py,c}` |
| IR golden | Python lowering is byte-stable (7 cases, all llvm flavors) | `conv_direct_nongrouped/*` in `platform/tests/instances/rocke_ir_parity_harness.py` |
| On-silicon numerics | output vs `torch.nn.functional.conv2d` (fp32 reference), rel. tol 5e-2 fp16 / 1e-1 bf16; includes one binary launched on four unrelated shapes | `pytest tests/test_direct_conv_correctness.py -k Nongrouped` |
| AOT ABI / shape invariance | kernarg order matches `conv_direct_arg_names("fwd")`; emitted IR independent of `N`/`H`/`W`/`C`/`K` | `pytest tests/test_conv_abi.py -k nongrouped` |
| AOT cache | runtime channels, width filtering, `groups == 1` only, plans pack against the signature | `pytest tests/test_direct_kernel_cache.py` |

`rocke.core.backend.lower_conv_direct_nongrouped(spec, backend="python"|"cpp"|"both")`
reaches the C++ builder directly through the `rocke_engine.conv_direct_nongrouped_*`
binding.

When changing the builder, change `conv_direct_nongrouped.cpp` in the same change and
keep one IR op per C++ statement in Python order: C++ leaves argument evaluation
order unspecified, so folding two emitting calls into one argument list
reorders `arith.constant`s and breaks byte-identity.

## Open items

* No dispatch wiring: `library/dispatch/` does not select this family yet.
* The remaining gap on the shapes where it trails looks like barrier +
  `waitcnt` exposure at two waves per SIMD. The untried lever is async
  DRAM→LDS (`raw_ptr_buffer_load_lds`), which would drop the prefetch
  registers; it needs lane-contiguous LDS writes, which conflicts with the
  `lds_pad` that keeps the activation reads conflict-free.
