# Coordinate-codeword oracle-free tests

This check launches the shipped gfx942 dense attention kernel and tests where it
writes: that every output element is stored once, at the right batch, query
head, query row and dimension, from the right V head, and that nothing is
written outside the output tensor. It uses no attention reference.

Zero Q makes every legal score equal, so the output row of query `q` is the
plain mean of V over its legal keys `A(q)`. V is a *codeword*: its value at
`(batch, KV head, key, dimension)` encodes that coordinate. Each stored output
then names the coordinate it belongs to, and each expected output is a mean of
codeword values over a legal-key interval, formed in exact rational arithmetic.
The output is allocated between two guards, so a write outside the tensor shows.

It is self-contained: it shares the case type, the legal-key arithmetic, the
exact rounding, the raw-bit encoding and the subject adapter with the
uniform-probability check in [README.md](README.md), and nothing else.

## Fixture

Q is zero. K is the uniform-probability check's finite, nonconstant K; it must
be finite, because zero Q annihilates it only when `0 * k` is zero. For batch
`b`, KV head `h`, key `k` and dimension `d`:

| Lane `d` | `V[b,h,k,d]` | Expected `O[b,q,h_q,d]` | Carries |
|---|---|---|---|
| 0 | `20 + (b+1)/4` | the same | batch |
| 1 | `26 + (h+1)/4` | `26 + (h_kv+1)/4`, `h_kv = h_q // (Hq/Hkv)` | KV head |
| 2 | `-k/2` | `-T_q / (2 N_q)` | query row |
| 3 | `5/32 + (k mod 2)/16` | `5/32 + O_q / (16 N_q)` | query row |
| `d >= 4` | `(d+1)/16` | the same | dimension |

`N_q`, `T_q` and `O_q` are the count, the index sum and the odd-index count of
the legal keys of row `q`. Lanes 0, 1 and `d >= 4` do not depend on the key, so
their mean is the input value. Lanes 2 and 3 identify the row through its legal
interval. The lanes take disjoint value ranges, so a value stored in the wrong
lane is never one the right lane admits.

Every codeword entry is exactly an FP16 and a BF16 value for `B <= 2`,
`Hkv <= 4`, `Sk <= 256` and `D <= 64`; `require_codeword_scope` refuses a case
outside those bounds. BF16's eight significant bits set the key bound: key 257
would store `-128.5`. `encode_bits` refuses any input value that would round.

## Cases

| Dtype | Mask | `(Hq, Hkv)` | Batch | Owns |
|---|---|---|---|---|
| FP16 | sliding window 64 | `(8, 2)` | 1, 2 | KV-head routing, group map, row, dimension; batch at `B = 2` |
| BF16 | top-left causal | `(8, 2)` | 1, 2 | the same |
| FP16 | sliding window 64 | `(4, 4)` | 2 | additionally output-head placement |
| BF16 | top-left causal | `(4, 4)` | 2 | additionally output-head placement |

All are `Sq = Sk = 256`, `D = 64`: six cases, which resolve to two compiled
kernels, because batch, sequence lengths and head counts are kernel arguments on
the runtime-shape path. The `(8, 2)` cases are the uniform-probability check's
two specifications. With zero Q the query heads of one KV group store identical
values, so those cases cannot tell which head of a group an output was written
to; the `(4, 4)` cases, with one query head per KV head, can.

## Verdict

There are two verdicts and no tolerance.

- **Stored bits.** Every stored output equals the correctly rounded
  (round-to-nearest-even) target, bit for bit. The one output whose target is
  zero, lane 2 of row 0, admits both zero signs: it is a sum of signed zeros
  whose sign nothing fixes. Admissibility is decided on bit patterns, so `-0` is
  accepted there and nowhere else.
- **Store safety.** Four exact element counts of the guarded allocation are
  zero (see below).

`analyze_codeword` reports, as diagnostics only, how a failing output fails:

- `near`: within one target-format value of the target. A placement defect moves
  an output by whole codeword steps, so this points at the arithmetic instead;
- `misplaced`: further away and equal to the codeword of another coordinate,
  which the inverse decoding names (a batch, a KV head, a query row or a
  dimension);
- `foreign`: further away and equal to no codeword.

### Why the verdict can be exact

The exact verdict rests on stated premises about the kernel's arithmetic. A
kernel, compiler or device for which one of them fails shows up as `near`
failures; the verdict is not widened to accommodate it.

| Premise | Statement |
|---|---|
| E1 | the base-2 exponential of zero is exactly one, in the fast and the plain lowering |
| E2 | for the masked argument it returns `+0` or the smallest FP32 subnormal |
| R1 | FP32 add, subtract and multiply round to nearest |
| R2 | the reciprocal is the correctly rounded `1/l` |
| R3 | the conversion to the subject dtype is round-to-nearest-even; an FP16 store may round the exact product once instead of twice |
| MFMA-exact | an MFMA step whose accumulator and products are multiples of one power of two `2^e`, with magnitudes summing to at most `2^(24+e)`, returns their exact sum |
| MFMA-faithful | otherwise an MFMA step returns a faithful rounding of its sum |

E1, R1, R2 and the round-to-nearest-even store are the premises of the
uniform-probability check on the same kernel. Under them the denominator is
exactly `N_q` and every numerator is an exact lane sum (each lane lies on a grid
of `1/4`, `1/2`, `1/32` or `1/16` with a total far below the MFMA-exact bound),
so only
`1/N_q` and the final product round. The value converted to the dtype is then
within about `2^-23` of the target, relatively, and every target is further than
that from the nearest rounding boundary of the dtype: by `2^-(p+2)` relatively
for a representable target (`p` fraction bits), and by `5.7e-6` (FP16) or
`4.0e-5` (BF16) for the thinnest of the lane-3 targets that are not
representable. The MFMA-faithful premise is needed only on the window rows that
visit a fully masked tile before their legal keys; those rows have a full
window of 64 keys and representable targets, and the bound it gives is about 32
times inside their margin.

The argument uses the tile sizes only to decide which rows meet such a tile, so
the verdict holds for every tile geometry, grid and persistence mode of a
single-pass kernel. CPU contracts check each step: the lane grids and totals,
the margins, and a path-exact emulation of the kernel's online softmax, which
stores every target for both masked-exponential values, both FP16 store forms
and four tile geometries.

### Guarded allocation and store safety

`run_codeword_case` allocates `guard | logical | guard` and hands the kernel the
address of the logical tensor. Guard elements hold the canary `0xA5A5`, a finite
value no output can equal; logical elements hold the sentinel `0xFFFF`, a NaN.

| Count | Elements | Required |
|---|---|---|
| `prefix`, `suffix` | guard elements, before and after the tensor, that no longer hold the canary | 0 |
| `unwritten` | logical elements that still hold the sentinel | 0 |
| `nonfinite` | other logical elements that are a NaN or an infinity | 0 |

Each guard is one query tile of output, `block_m * Hq * D` elements, with
`block_m` read from the resolved spec: the kernel stores without a row bound and
its unit of work is a query tile, so that is what a missing bound can overrun.
It is no bound on a write to an arbitrary address. A store of exactly the
sentinel bits looks the same as no store, so `unwritten` is reported as a
presumed missing store.

## Mutation self-checks

On every run, each owned mutation is applied to the kernel's own output and the
analyzer must reject it through the channel named here, or the test fails. A
mutation is applied only on the cases that can separate it.

| Mutation | Seam | Channel | Cases |
|---|---|---|---|
| `kv_head_rotate` | V-head routing | KV-head lane | all |
| `group_map_interleaved` | V-head routing | KV-head lane | `(8, 2)` |
| `query_head_broadcast` | output-head base | KV-head lane | all |
| `out_head_swap` | output-head base | KV-head lane | `(4, 4)` |
| `batch_swap` | batch base | batch lane | `B = 2` |
| `dim_rotate_in_vector`, `d_half_swap` | dimension addressing | dimension lanes | all |
| `row_shift`, `adjacent_row_swap` | query-row addressing | row lanes | all |
| `v_token_plus_one` | V addressing | row lanes | all |
| `denominator_plus_one`, `denominator_minus_one` | denominator | batch lane | all |
| `dropped_store` | output store | `unwritten` | all |
| `base_underrun` | output base | `prefix` | all |
| `row_bound_bypass` | row bound | `suffix` | all |

The verdict admits a single bit pattern, so any change of a stored bit is a
rejection and separation needs no margin. The thinnest change is one
target-format value: adjacent BF16 causal rows from row 128 on, where the BF16
spacing equals the row step of `1/4`. A kernel with a placement bug can store
exactly what a placement mutation turns back into the correct output; the
unrejected mutation is then reported together with the output's own violations.

## Negative controls

`test_codeword_dense_controls.py` launches real kernels with one defect each.
There is one control per seam of the table above. The shipped kernel is launched
first on the same case and must pass.

Nine controls are *emitted*: inside `private_build` the dense runner constructs
its IR through a builder that alters one address or one operation, and keeps the
launcher it compiles in a cache of its own. The shipped compile and launch path
is used unchanged; the shipped kernel is neither replaced nor reused. Each
control's lowered IR must differ from the shipped kernel's, which CPU contracts
check for every control and case, together with the fact that the mechanism
without a defect reproduces the shipped IR exactly.

| Control | Emitted defect | Must show |
|---|---|---|
| `kv_head_rotate` | V load offset with the KV-head bit flipped | KV-head lane |
| `out_head_swap` | store index with the query-head bit flipped | KV-head lane on `(4, 4)`; **passes** on `(8, 2)` |
| `batch_swap` | store index with the batch bit flipped | batch lane |
| `adjacent_row_swap` | store index with the row bit flipped | row lanes |
| `d_half_swap` | store index with the lane-half bit flipped | dimension lanes |
| `v_token_plus_one` | V load offset one key later | row lanes |
| `denominator_plus_one` | reciprocal of `l + 1` | batch lane |
| `dropped_store` | first store site of the epilogue skipped | exactly `2 * o_store_width` unwritten elements per row and head (8 at the resolved width of 4) and nothing else |
| `base_underrun` | store index minus four elements | exactly 4 `prefix` and 4 `unwritten` elements |

Flipping one index bit swaps neighbours of one coordinate, so those controls
stay inside their tensors (the extents are powers of two, which the builder
checks). The underrun lands in the guard, and a V load past the last key is
clamped to zero by the buffer bound. The head swap on a `(8, 2)` case swaps two
heads of one KV group and must pass: it demonstrates the stated blind spot on a
real kernel.

The tenth control, `row_bound_bypass`, is host-side. On the runtime-shape path
the query length is a kernel argument, so one query row past a whole tile runs
the shipped binary itself; no IR difference can be required of it. The spec type
and `supports_attention_dense` both refuse that shape, so the control overwrites
`seqlen_q` on a copy of the shipped spec (`block_m + 1` rows, one batch) and
patches the predicate to answer for the shipped shape. CPU contracts check that
the copy differs from the shipped spec in `seqlen_q` only, keeps the shipped
launcher-cache key and lowers to the shipped IR. Q is padded to whole tiles.
The launch must store every logical row correctly and change exactly
`(block_m - 1) * Hq * D` guard elements after the tensor, 65024 on the `(8, 2)`
cases, and nothing else.

## Scope

The gate runs whatever configuration dispatch resolves and asserts only the
scope the derivation needs; a retune inside it keeps running. Before any launch,
as a failure and never a skip, it refuses:

- a case outside the codeword bounds above, or with a query row that has no
  legal key;
- a case in which two query rows store the same rounded codeword (next
  section);
- everything the uniform-probability adapter refuses: another path or
  candidate, a different shape than requested, split-KV, sinks, LSE, ragged,
  bottom-right, paged or varlen;
- `window_wave_skip`, whose handling of fully masked tiles the derivation does
  not cover;
- an architecture other than gfx942: the derivation is for that kernel body.

Nothing about tuning is pinned, and no measured value defines or enlarges what
is admitted.

### Limit of the codeword

Row identity rests on lane 2, whose targets are multiples of `1/4`. BF16 spacing
is `1/4` only below magnitude 64, so a BF16 case whose lane-2 magnitude reaches
64 loses rows: with a sliding window of 64 at `Sq = Sk = 256`, 47 pairs of
adjacent rows store the same codeword, and a swap inside a pair would pass.
`require_codeword_scope` refuses such a case, naming the first colliding rows,
and a CPU contract pins the 47 pairs. The shipped cases do not reach it: BF16
runs causal (magnitude at most 63.75) and the window case is FP16.

### What the check does not cover

- **Query heads inside a KV group.** On the `(8, 2)` cases a swap or a wrong
  base inside a group passes. The `(4, 4)` cases are the complement, at `B = 2`.
- **Anything upstream of the weights.** Q is zero, so the QK contraction, the
  scale, K routing and nonuniform softmax are not exercised.
- **Permutations of legal keys.** The means are permutation-invariant.
- **Other buffers and other writes.** The guards surround the output only. They
  do not show a write beyond the guard, a write that reproduces the canary, or
  an out-of-bounds read.
- **Unexercised features.** Split-KV, sinks, LSE, the bottom-right band and
  supported ragged shapes; the row-bound control exercises only the refused
  shape.

## Skip and failure semantics

A GPU case skips only when the visible device is not its architecture, through
the shared `rocke.tests.gpu.current_device_arch_in` gate, so
`ROCKE_REQUIRE_GFX942=1` turns that skip into a failure. CPU contracts pin both
GPU modules to their case matrices and to that single skip. The suite is
Torch-free and oracle-free under the same source contracts as the
uniform-probability check.

## Running the suite

From the `rocke/library` directory (NumPy and pytest only):

```bash
# CPU contracts: fixture, targets, margins, emulation, analyzer, kill matrix,
# guarded launch on a stand-in device, control identity.
python -m pytest tests/oracle_free/test_codeword_contracts.py -vv

# The gate and its negative controls on a visible gfx942 device.
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/oracle_free/test_codeword_dense.py \
  tests/oracle_free/test_codeword_dense_controls.py -vv -rs
```

In the installed test tree the GPU modules run in `rocke_oracle_free_pytest` and
the CPU contracts in `rocke_library_attention_cpu_pytest`.
