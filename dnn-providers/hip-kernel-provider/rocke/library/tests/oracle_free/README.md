# Oracle-free attention tests

This suite launches the shipped dense attention kernels and checks them without
a complete attention oracle. It has three lanes:

| Lane | Checks | Verdict |
|---|---|---|
| Uniform probabilities ([Exposures](#exposures) to [Scope](#scope)) | masked attention on zero Q | every stored bit equals a closed form |
| [Key-permutation relation](#key-permutation-relation) | unmasked attention on nonuniform scores | two launches agree within a derived bound |
| [Value-shift relation](#value-shift-relation) | unmasked attention on nonuniform scores | two launches differ by the shift of V, within a derived tolerance |

The uniform-probability lane sets Q to zero. Zero Q makes every legal score
equal, so the softmax row for query `q` is uniform over its legal key set
`A(q)`. Each expected output follows from legal-key arithmetic alone, rounded
correctly into the subject dtype.

## Exposures

Each case runs `Sk / D + 1` launches on the same zero-Q, finite, nonconstant K:

| Exposure | V for KV head `h` | Expected `O[q,h_q,d]`, with `s = 2^h_kv(h_q)` |
|---|---|---|
| `basis@o`, `o in {0, D, 2D, ...}` | `V[k,h,d] = 2^h` iff `k = o + d` | `s/\|A(q)\|` if `o + d` is legal, else `0` |
| `folded` | `V[k,h,d] = 2^h` iff `d = k mod D` | `s * \|{k in A(q) : k mod D = d}\| / \|A(q)\|` |

The basis launches expose each probability column exactly once. The folded
launch observes the same row through sums of several probabilities, which
catches errors that cancel across basis launches.

Each KV head scales its V profile by a distinct power of two, `2^h`, and query
head `h_q` expects its KV head's scale, `h_kv = h_q // (Hq / Hkv)`. Without
that, every head would carry identical outputs, so a kernel that reads one V head
for all heads, uses a wrong head stride, maps query heads to the wrong KV group,
or writes one output head everywhere would pass. With it, every legal coordinate
of a head that reads the wrong KV head is off by at least a factor of two. The
quick cases are GQA (`Hq=8`, `Hkv=2`), so the group divisor is exercised: a
kernel that drops it, or interleaves the groups, reads the wrong scale. A power
of two is exact in FP16 and BF16 and commutes with every rounding the kernel
performs (the P cast, FP32 accumulation, normalization, the output cast), so
head `h`'s correct output is head 0's shifted in exponent and the zero-ULP
argument is unchanged. `nhead_k` is capped at 8 to keep the scales distinct and
the outputs in range.

Not covered by design: with Q zero, the logits do not depend on Q or K, so a
wrong Q or K head index is invisible. Query heads that share one KV head also
share a scale, so swapping heads within one group is invisible.

## Device path

The suite does not use Torch. Inputs are built in FP32 and then encoded to raw
FP16/BF16 `uint16` bits by `encode_bits`. That function rejects any value that
would round, so the kernel reads exactly the values the expectations assume.
Signed zero survives the encoding, and NaN or out-of-range inputs are refused.
BF16 goes through rocKE's shared codec (`rocke.numeric.bf16`); FP16 uses numpy's
native `float16`. Only the input and output encoding is shared; the expected
bits come from exact rational rounding in `_analytical.py`.

`run_dense_case` in `_dense_subject.py` launches one case:

1. It allocates one `rocke.runtime.DeviceMem` buffer each for Q, K, V and the
   output. Before upload it checks each buffer's size, 16-byte alignment and
   separation.
2. It prefills the output with quiet-NaN bits, so an unwritten coordinate is
   counted `invalid`.
3. It uploads with `Runtime.memcpy_h2d`.
4. It launches through the architecture's public
   `kernels.<arch>.attention_dense.run_attention_dense_torch`, then synchronizes.
5. It reads the output back with `Runtime.memcpy_d2h` as raw bits, with no float
   conversion.

Without sinks, LSE or a KV split, `run_attention_dense_torch` reads no tensor
attribute, so it accepts `DeviceMem` buffers directly; the pinned SDPA lane
launches it the same way. The suite therefore exercises the shipped compile,
cache and launch path and keeps no copy of the kernel signature, geometry or
runtime arguments.

## Policy

Expected values are correctly rounded (round-to-nearest-even) raw FP16/BF16 bit
patterns of the exact rationals above. `analyze_uniform_policy` compares raw output
bits, and every one of these rules is a hard violation:

- every output is finite (`invalid`), and no output is negative (`negative_max`);
- every legal coordinate matches its target bit-exactly (`target_ulp`, budget 0);
- masked lanes are exactly zero (`masked_max`, `masked_mass`); `-0` is accepted;
- basis lanes past the last key are exactly zero (`trailing_max`);
- per row, the sum of stored outputs equals the sum of stored targets exactly
  (`stored_row_sum`). This is checked for the folded exposure and for the union
  of basis exposures, wherever the exposure covers the full legal set;
- at least one legal coordinate is observed.

No absolute tolerance participates in any verdict. The analyzer reports
per-`N_q = |A(q)|` evidence separately inside and beyond the proven range (see
below). It evaluates whole `[B, Sq, Hq, D]` arrays at once; a CPU contract keeps
the original one-coordinate-at-a-time analyzer as an executable specification
and requires the two to agree on every field of the result.

### Mutation self-checks

On every run, the analyzer is re-applied to the same real output under rules
that are wrong by one key or by one in the denominator. Each mutated rule must
be rejected, or the test fails:

| Case | Mutations |
|---|---|
| causal | `include_future_key`, `exclude_last_key`, `denominator_plus_one` |
| window | `include_old_window_key`, `exclude_oldest_window_key`, `denominator_plus_one` |

The same real output with query head 0 copied into every head
(`broadcast_query_head`) must also be rejected, which shows the run can tell
heads apart. Magnitude diagnostics (masked mass, row-sum residuals) are reported
in probability units, divided by each head's scale, so they do not depend on the
head scaling.

## Scope

The zero-ULP result is a theorem, not a measurement of one kernel
configuration. With Q zero, `exp2(0) = 1` exactly (for the fast and the plain
exp2 lowering), P = 1 survives the cast, and the running-max rescale is 0 or 1.
The row normalizer `l = N_q` and every PV sum are exact in FP32 whatever the
accumulation order. The stored `cast(acc * rcp(l))` (an RNE `fptrunc` in every
variant; no spec field selects another mode) then rounds to the correctly
rounded target for `N_q < 2^(21-p)`, where `p` is the dtype's fraction bits
(two thirds of that for the folded exposure). So the result holds for every
single-pass dense variant, whatever its tile, grid, persistence, runtime-shape
handling, fast exp2, CFVST, buffering or scheduling knobs.

The gate therefore runs whatever configuration dispatch resolves for the
case and asserts only the scope the theorem needs. `require_proven_scope`
refuses a case whose largest `N_q` leaves the proven range, and the subject
adapter refuses, before any launch, a resolved spec that is:

- another path or candidate than the architecture's dense subject;
- a different shape than requested (it sizes the host buffers);
- split-KV (`kv_segments > 1`), whose partial-softmax combine the theorem does
  not cover;
- with sinks or LSE, which need buffers and epilogues the adapter does not
  supply;
- ragged, bottom-right, paged or varlen.

A retune inside that scope keeps running; a defect in any configuration still
fails the exact policy. The quick cases:

| Arch | Dtype | Case | Largest `N_q` |
|---|---|---|---:|
| gfx942 | FP16 | `B1,Sq256,Sk256,H8/2,D64`, sliding window 64 | 64 |
| gfx942 | BF16 | `B1,Sq256,Sk256,H8/2,D64`, top-left causal | 256 |

The pairing complements the pinned SDPA lane, which already runs FP16 causal on
gfx942.

### Architectures

`DENSE_SUBJECTS` in `_dense_subject.py` maps an architecture to its dense
dispatch candidate and runner module, and `QUICK_SUBJECT_CASES` lists the
`(arch, case)` pairs the GPU lane runs. gfx950 has a subject entry
(`gfx950_dense_grid`) whose resolution a CPU contract checks, but no GPU cases
yet: PR CI has no gfx950 provider runner.

### Skip and failure semantics

A GPU case skips only when the visible device is not its architecture. It uses
the shared `rocke.tests.gpu.current_device_arch_in` gate, so the skip reason
names the GPU found, and a lane that sets `ROCKE_REQUIRE_<ARCH>=1` (for example
`ROCKE_REQUIRE_GFX942=1`) fails instead of skipping.

On the architecture, every case in `QUICK_SUBJECT_CASES` runs, and the checks
below are failures, never skips:

- the case leaves the proven range;
- dispatch resolves a spec outside the scope above;
- a mutation or the head-0 broadcast is not rejected;
- no legal coordinate was compared;
- any policy violation.

A CPU contract pins the GPU lane's parametrization to `QUICK_SUBJECT_CASES` and
that matrix to `UNIFORM_QUICK_CASES`, so a case cannot drop out unnoticed. A
kernel with a mask or denominator bug can write exactly what a mutation
predicts. When that happens, the unrejected mutation is reported together with
the control's violations, not as a checker failure.

## Key-permutation relation

Unmasked attention depends on its keys only as a set of (key, value) pairs.
Permuting the key tokens of K and V with one map `pi` therefore leaves the
output unchanged:

`O(Q, pi K, pi V) = O(Q, K, V)`.

This lane launches the subject on both sides of that identity and compares the
two stored outputs with each other. There is no expected output, no closed form
and no reference: the tolerance follows from the kernel's arithmetic path.

The relation needs mask mode none. Under a causal or window mask the legal keys
of a row depend on the token index that the map moves. The gfx942 dense kernel
serves unmasked attention only at `seqlen_q == seqlen_k` with whole tiles, so
the cases are square. A case needs more than 16 key tokens, so that every map
the lane applies exists.

### Inputs

| Tensor | Values |
|---|---|
| Q, K | pseudo-random multiples of 1/8 in `[-3, 3]`, bell-shaped, close to unit variance |
| V | pseudo-random `+1` or `-1` |

The values come from an explicit integer hash of the element index, not from a
library random stream, and are exact in FP16 and BF16. The Q and K lattice makes
every product a multiple of 1/64, and FP32 holds every partial sum of a `D`-term
contraction exactly. The score of a (query, key) pair is therefore the same FP32
value wherever the key sits, which is what lets the two launches be compared.
No query row is zero and no two key or value tokens are equal, so a value paired
with the wrong key changes the output.

### Token maps

A map is stored destination-to-source: destination token `k` receives source
token `map[k]`, for every batch element and KV head. Whole token vectors move
and their bits are copied, so the transformation is exact.

| Map | Definition |
|---|---|
| `shift_one` | `k + 1 mod Sk` |
| `reverse` | `Sk - 1 - k` |
| `cross_tile` | `k + 16 mod Sk`; needs `Sk > 16` |
| `partial_final_tile` | moves the tail span `Sk mod 16` to the front; needs `Sk mod 16 != 0` |
| `adjacent_swap` | exchanges tokens 0 and 1 |

The 16-token span is a boundary of the test, not the kernel's tile size.
`partial_final_tile` needs a ragged key length, which the whole-tile gfx942
kernel does not take; the quick cases run the other four.

`adjacent_swap` closes a gap the other maps leave. Take a kernel that pairs the
score of key slot `j` with the value of slot `f(j)`, for a fixed map `f` that
does not depend on the inputs. It satisfies the relation under a map for all
inputs exactly when `f` commutes with that map. The shifts and the reverse
generate a dihedral group, and for even `Sk` the shift by `Sk / 2` commutes with
all of it: a value read half a sequence away from its key would pass every one
of them. No nontrivial shift commutes with the adjacent swap, so with it the
only fixed pairing that satisfies every map for all inputs is the correct one.
The argument needs at least three tokens: with two, the three maps are one
exchange, and exchanging the two values commutes with it.

That is a statement in exact arithmetic. On a faulty kernel the lane measures
the difference of two faulty runs, `sum_i P_i (V[f(i)] - V[g(i)])` with
`g = map o f o map^-1`, not the distance to a correct output. A defect in slot
`a` contributes at key `a` in the base run and at the key sitting in slot `a` in
the permuted run, and the two contributions can add or cancel. Each stored run
is within the per-run bound of its own exact value, so a defect is certain to
fail only when that exact difference exceeds twice the tolerance at some
coordinate.

### Bound

Per row, both launches compute the same FP32 scores as a multiset, so they share
the row maximum `m` and the ideal weights `p_k = exp2(s_k - m)`. What differs is
which key falls in which tile: the running maximum, the rescale points and every
summation order. Each launch's stored output is within `per_run` of the ideal
`sum(p_k v_k) / sum(p_k)`, and the residual tolerance is `2 * per_run`:

| Term of `per_run` | Source | Size |
|---|---|---|
| Probability cast | the accumulator multiplies V by the weight rounded to the subject dtype; the normalizer sums the unrounded weight | `2^-(p+1) max\|V\|`, `p` the dtype's fraction bits |
| Output store | one conversion of `acc / l` to the subject dtype | half an ULP at the largest admissible output |
| FP32 remainder | one `exp2` and one rounded argument per key tile, accumulation in any order, the reciprocal | about a tenth of the total in FP16, a hundredth in BF16 |

For the quick shape (256 keys in four tiles, `max|V| = 1`) the tolerance is
`2.14e-3` in FP16 and `1.58e-2` in BF16.

The verdict is conditional on the assumptions listed below. A passing launch
is consistent with the bound as a whole. It does not validate any single
assumption, and no primitive-level check of them exists in this lane.

`permutation_bound` evaluates it from the dtype, the key count, the number of
key tiles a query row visits in the kernel under test, `max|V|`, and a bound on
the spread of a row's scores taken from the row norms of Q and K. It reads no
output, so a faulty kernel cannot widen its own tolerance, and it is not fitted
to any measurement. The worst case needs the rounding of every weight to line
up, so a correct kernel sits well inside the bound.

The tolerance grows with the score spread, so the spread is certified from
above, not estimated. Q and K are multiples of 1/8, so their squared row norms
are exact integers; the product of the two largest norms is enclosed from above
by an integer square root, the remaining factors are exact rationals with
`log2(e)` replaced by a rational that is not below it, and the result is the
first float that is not below that rational. `score_spread_bound` refuses inputs
that are not finite or not on the lattice.

FP32 results below the normal range carry absolute errors, not relative ones.
They are kept apart from the relative terms and from the cast of the weight to
the subject dtype: one budget each for the weights, the accumulator and the
normalizer. At the quick shape all three are far below the resolution of the
tolerance.

The bound is derived for `max|V| > 0`, a key count that is a multiple of the
eight products of an MFMA call, `(keys + key_tiles) * 2^-23 < 1`, and a
normalizer error below one, which keeps the computed normalizer positive.
`permutation_bound` refuses anything else.

The bound holds for every single-pass dense variant, whatever its tile, grid,
persistence or scheduling knobs, under these assumptions about the executed
path:

- FP32 addition, multiplication and the reciprocal return one of the two FP32
  neighbours of their exact result;
- each MFMA call returns its accumulator plus its eight products with an error
  of at most eight FP32 units of the summed magnitudes of those nine terms.
  This inequality is the whole assumption: the call is not assumed to add its
  products one by one, nor to round their exact sum;
- `exp2` is within one FP32 ULP where its exact value is a normal FP32 number,
  and within `2^-126` below that range;
- the probability cast and the output store round to nearest even and keep the
  subnormals of the subject dtype;
- no NaN or infinity enters the path: Q, K and V are finite, and so is every
  score, weight and accumulator;
- the normalizer sums uncast weights, the accumulator consumes cast ones, and
  both are rescaled by `exp2(m_old - m_new)` per key tile.

The last item is checked on the CPU. A contract lowers each quick spec to LLVM
IR and classifies every floating-point instruction against that path: no mask on
the scores, a key-tile loop over `seqlen_kv / block_n`, uncast weights in the
normalizer, one cast into the PV operand, one `alpha` for the normalizer and for
every accumulator element, one `fdiv 1.0, l`, and no fast-math flag. A second
contract traces the MFMA operands: the score chains multiply K from its LDS
region by Q, the value chains start from the rescaled accumulator and multiply V
from its LDS region by the cast weights, each weight enters every value chain
once, and the chain lengths are `head_size / 8` and `block_n / 8`. An
instruction or operand outside the path fails with "re-derive the bound"; the
bound is never widened to absorb it.

The other items are assumptions about the hardware and the compiler. Of the
compiled code object this suite reads only the floating-point mode; how each
instruction rounds is not checked here. The MFMA item in particular is a
property of the hardware: an architecture whose FP16 MFMA reads operands below
the smallest normal as zero does not satisfy it, so a new architecture needs
that property established before its cases are added.

### Scope checked before a launch

The bound holds for one path, one tile count and certified input statistics.
The GPU lane establishes them for the kernel under test before it launches
anything, and fails, never skips, when one is missing:

- The resolved spec is the unmasked single-pass dense kernel (see below).
- The kernel is compiled as the launcher compiles it, and the path checks above
  run on the LLVM IR of that artifact, not on an earlier build. The adapter
  repeats the launcher's compile call; a CPU contract fails when the two calls
  differ.
- The number of key tiles a query row visits is taken from that IR: one KV loop
  over `seqlen_kv / block_n` whole tiles, with no mask and no early exit. A
  count inferred from the requested shape alone is not used.
- The kernel descriptor of the code object selects round-to-nearest-even and
  keeps subnormals, for FP32 and for the 16-bit conversions.
- Q, K and V are finite, Q and K are on the lattice, and the score spread is
  certified from above.
- Every map is a permutation of the key tokens, applied to whole token vectors
  of K and V alike, in every batch element and KV head.

The lane prints the SHA-256 of the code object and the compiler that produced
it. Evidence gathered on one code object does not carry over to another digest.

The subject adapter refuses the same specs as for the uniform lane (split-KV,
sinks, LSE, ragged, bottom-right, paged, varlen) and additionally requires the
resolved spec to be unmasked.

### Policy

For every map, each of these is a hard violation:

- a stored value in either output is not finite (`invalid`); the output is
  prefilled with quiet NaN, so an unwritten coordinate counts;
- a stored magnitude exceeds the largest output the inputs admit
  (`magnitude`); a blow-up both launches share would otherwise cancel;
- `|O_A - O_B|` exceeds the bound at any coordinate (`residual`).

For every map and every (batch, query head) the lane also prints four drift
diagnostics. None of them is a verdict:

| Diagnostic | Meaning |
|---|---|
| `r_max` | largest residual over the bound; the residual verdict is `r_max <= 1` everywhere |
| `r_rms` | root-mean-square residual over the bound |
| `r_guard` | largest stored magnitude of either output over the magnitude guard |
| `n_half` | coordinates whose residual exceeds half the bound |

The line also gives the coordinate and the signed residual at `r_max`, and the
mean signed residual. Ratios that grow from one release to the next are a
signal while the verdicts still pass.

### Pairing mutations

Three more launches break the pairing, starting from `reverse`:

| Mutation | K map | V map |
|---|---|---|
| `key_only` | `reverse` | identity |
| `value_only` | identity | `reverse` |
| `different_maps` | `reverse` | `shift_one` |

Each must exceed the bound in every (batch, query head). A head whose output
stays inside the bound ignores which value belongs to which key: a constant or
all-zero output, or attention that does not depend on K, fails here although it
satisfies the identity. When the joint permutations fail as well, the message
reports their violations as the diagnosis.

### What the relation cannot see

- A defect applied identically to both launches cancels. A wrong scale, a wrong
  KV-head routing, a wrong output placement or any fixed transformation of the
  output satisfies the identity. The uniform-probability lane and the pinned
  SDPA lane cover those.
- A fixed pairing defect whose difference stays below the tolerance. On rows
  where one key holds nearly all the probability, a single mispaired slot among
  the others moves the BF16 output by less than the tolerance and passes every
  map; the FP16 tolerance rejects the same defect.
- A wrong pairing that is chosen from the inputs and is carried along by every
  permutation, for example "the value of the highest-scoring key". It satisfies
  every joint permutation and responds to every mutation.

What a pass shows is therefore limited: no fixed, input-independent pairing
defect moved this fixture's outputs by more than twice the tolerance under any
applied map.

### Cases

| Arch | Dtype | Case |
|---|---|---|
| gfx942 | FP16 | `B1,Sq256,Sk256,H8/2,D64`, unmasked |
| gfx942 | BF16 | `B1,Sq256,Sk256,H8/2,D64`, unmasked |

Each case is eight launches: the base inputs, four joint permutations and three
pairing mutations. `PERMUTATION_SUBJECT_CASES` in `_dense_subject.py` lists the
`(arch, case)` pairs, and the skip and failure semantics are those of the
uniform lane: only a device of another architecture skips.

The CPU contracts in `test_permutation_contracts.py` exercise the analyzer on
an FP32 model of the kernel's single-pass online softmax: the modeled joint
permutations stay inside the bound, on the fixture and on rows with a late or
dominant maximum; the modeled mutations leave it in every head; and two modeled
pairing defects are caught by the maps the group argument predicts. Further
contracts check the stored residual of a single mispaired slot against the
two-run formula above, and keep the three limits of the previous section as
executable examples. The model lives only in those contracts.

The same file certifies the score spread of the fixture against an exact
rational bound, and applies the artifact checks of `_permutation_path.py` to the
lowered quick kernels of every compiler lane and to edited copies that each
check must reject.

## Value-shift relation

The attention weights of a row sum to one. Adding one vector `c` to every value
token therefore adds `c` to every output row:

`O(Q, K, V + 1 c^T) = O(Q, K, V) + 1 c^T`.

This lane launches the subject on `V` and on `V + c` with the same Q and K and
judges the residual `E = O(V + c) - O(V) - c` element by element. There is no
expected output, no closed form and no reference: the tolerance follows from
the kernel's arithmetic path and the launched values, and its last term from
the stored bits themselves.

The shift is `c_d = (1/2, -1/4, 1/8, -1/2)[d mod 4]`: alternating signs, three
magnitudes, every component nonzero and exact in both subject dtypes.

The relation holds under any mask whose rows are all non-empty, but the
tolerance is derived for the unmasked path only. The gfx942 dense kernel serves
unmasked attention only at `seqlen_q == seqlen_k` with whole tiles, so the
cases are square.

### Inputs

| Tensor | Values |
|---|---|
| Q, K | pseudo-random multiples of 1/8 in `[-3, 3]`, bell-shaped |
| V | pseudo-random `+1` or `-1` |

The values come from an explicit integer hash of the element index, not from a
library random stream, and are exact in FP16 and BF16. Two attention profiles
use them:

| Profile | Change to Q and K | Effect |
|---|---|---|
| `moderate` | none | every row spreads its probability over many keys |
| `concentrated` | Q is replaced by its magnitude and key token 70 is 2.0 in every dimension | that key holds nearly all of every row's probability; in FP16 most other weights reach the probability cast as subnormals |

The lattice keeps every score finite and far above the kernel's initial running
maximum. The largest score of a row then has weight `exp2(0)`, which bounds the
normalizer away from zero. `require_finite_scores` refuses Q or K off the
lattice.

The tolerance has no term for a rounded input transformation, so the follow-up
tensor must be the exact sum. `shifted_value_bits` forms `V + c` in FP64,
checks that the sum is exact there and in FP32, and encodes it with
`encode_bits`, which refuses a value the dtype cannot hold. With V in
`{+1, -1}` the shifted values are `3/2, -1/2, 3/4, -5/4, 9/8, -7/8, 1/2, -3/2`.
The contract rechecks the launched bytes: the follow-up bits must decode to
`V + c` exactly.

### Tolerance

Both launches read the same Q and K, so they compute the same weights
`p_k = exp2(s_k - m)`, the same rescales and the same normalizer `l`. Key `k`
reaches the output with the coefficient `P_k = cast(p_k) A_k / l`, where `A_k`
is the product of the rescales after its tile. The residual is

`E_d = c_d (sum_k P_k - 1) + accumulation and store errors of both launches`,

and the tolerance bounds each part:

| Term | Source | Size at the quick shape |
|---|---|---|
| Row sum, times `\|c_d\|` | the accumulator multiplies V by the weight rounded to the subject dtype, while the normalizer sums the unrounded weight; plus the FP32 normalizer and its reciprocal | `5.27e-4` (FP16), `3.94e-3` (BF16) |
| Accumulation, once per launch | `Sk / 8` MFMA calls and one rescale per key tile | `3.1e-5` of the column's largest `\|V\|` |
| Store, once per stored value | the conversion of `acc / l` to the subject dtype, with or without an FP32 product in between | half the gap of the stored value to its neighbour away from zero, plus one FP32 unit |

The errors of the exponential and of its argument do not enter. Numerator and
normalizer use the same computed weights, so a weight that is wrong in both
cancels in `sum_k P_k`.

The row-sum and accumulation terms are functions of the dtype, the key count,
the number of key tiles a query row visits in the kernel under test, and the
largest `|V|` of each column in each launch. `value_shift_bound` evaluates them
as exact rationals, and each float the verdict uses is the first one not below
its rational. They read no output and are not fitted to any measurement.

The store term is read from the stored bits (`store_allowance`). A value
rounded to nearest lies within half a gap of what it was rounded from, whatever
that was, so the term is a property of the format's grid at the stored value,
not of a correct answer. It holds for a faulty kernel as well: a wrong store is
still the rounding of something, and the defect shows in the other terms. A
defect that moves an output changes its own store term by at most one binade
step, which is the resolution of the dtype there. An input-only store term
would charge every element the half-gap of the largest admissible output. At
the quick shape that alone is about three quarters of the tolerance of a
column with `|c_d| = 1/2`, whatever the size of the stored value.

The tolerance that does not read the outputs is kept as a diagnostic
(`apriori`). At the quick shape, in the columns with `|c_d| = 1/2`, it is
`1.32e-3` in FP16 and `9.86e-3` in BF16.

The bound is derived for a key count that is a multiple of the eight products
of an MFMA call, `(keys + key_tiles) * 2^-23 < 1`, and a normalizer error below
one, which keeps the computed normalizer positive. `value_shift_bound` refuses
anything else, and `value_shift_contract` refuses a column whose tolerance is
not below the shift it must resolve.

The tolerance holds for every single-pass dense variant, whatever its tile,
grid, persistence or scheduling knobs, under these assumptions about the
executed path:

- FP32 addition, multiplication and the reciprocal return one of the two FP32
  neighbours of their exact result;
- each MFMA call returns its accumulator plus its eight products with an error
  of at most eight FP32 units of the summed magnitudes of those nine terms;
- the probability cast and the output store round to nearest even and keep the
  subnormals of the subject dtype;
- no NaN or infinity enters the path;
- the kernel is deterministic: two launches on identical bytes store identical
  bits;
- the normalizer sums uncast weights, the accumulator consumes cast ones, and
  both are rescaled by the same `alpha` per key tile;
- nothing computed from V reaches a score, a weight, a rescale, the
  normalizer, a branch condition or an address.

The last two are checked on the lowered kernel, on the CPU and again before
every launch. The path is the one the key-permutation bound is derived for, so
the checks of `_permutation_path.py` are reused: they classify every
floating-point instruction of the LLVM IR against the path and trace the MFMA
operands to the stored tensors. `_value_shift_path.py` adds a trace of
everything computed from V: `v_ptr` may be read only by the fills of its LDS
region, that region only as the value operand of the PV calls, and their
results only by the accumulator, its rescale, the reciprocal product and the
store. An instruction, an operand or a V-derived
value outside that fails with "re-derive the tolerance"; the tolerance is never
widened to absorb it. The trace follows LDS regions as a whole. Which row of
its region a fill writes is index arithmetic and is not traced.

Determinism is checked by the replay launch. The other items are assumptions
about the hardware and the compiler. Of the compiled code object this suite
reads only the floating-point mode; how each instruction rounds is not checked
here.

The uncast-normalizer item is a property of this kernel, not a requirement of
attention. With it the weights the kernel applies do not sum to one exactly: a
row sum is off by up to `2^-(p+1)`, which is the largest part of the row-sum
term. The verdict accepts that, and the path check requires it: a kernel that
summed the cast weights would be more accurate and would fail the check until
the tolerance is re-derived for it. So that the accepted behaviour stays a
measured number, every run is also reported against the tolerance of such a
kernel, whose row-sum term is `3.1e-5` in both dtypes (the
`against a normalizer over cast weights` diagnostic below).

### Scope checked before a launch

The lane establishes the following for the kernel under test before it launches
anything, and fails, never skips, when one is missing:

- The resolved spec is the unmasked single-pass dense kernel. The subject
  adapter refuses the same specs as for the uniform lane (split-KV, sinks, LSE,
  ragged, bottom-right, paged, varlen) and additionally requires the resolved
  spec to be unmasked.
- The kernel is compiled as the launcher compiles it, and the path checks above
  run on the LLVM IR of that artifact, not on an earlier build. The adapter
  repeats the launcher's compile call; a CPU contract fails when the two calls
  differ.
- The number of key tiles a query row visits is taken from that IR: one KV loop
  over `seqlen_kv / block_n` whole tiles, with no mask and no early exit.
- The kernel descriptor of the code object selects round-to-nearest-even and
  keeps subnormals.
- Q and K are on the lattice, V is finite, and the follow-up bits are `V + c`
  exactly.

The lane prints the SHA-256 of the code object and the compiler that produced
it. Evidence gathered on one code object does not carry over to another digest.

### Policy

Each of these is a hard violation:

- a stored value in either output is not finite (`invalid`); the output is
  prefilled with quiet NaN, so an unwritten coordinate counts;
- a stored magnitude exceeds the largest output the launched values admit
  (`magnitude`); an error both launches share would otherwise cancel;
- `|E_d|` exceeds its tolerance at any coordinate (`residual`);
- the replay of the base launch differs from it in any stored bit.

The lane also prints diagnostics. None of them is a verdict:

| Diagnostic | Meaning |
|---|---|
| `rho_max` | largest residual over its tolerance, with its coordinate, residual and tolerance; also per (batch, query head) and per `d mod 4`, the period of the shift and of the FP16 store forms |
| `eta_hat` | `E_d / c_d` at that coordinate, the apparent row-sum error, with its allowance `tau_d / \|c_d\|` |
| `eta_row`, `s_row` | largest per-row median of `E_d / c_d`, and largest distance of a column from its row's median; a row-sum defect moves the median, a column defect the spread |
| `rms`, `mean` | root-mean-square and mean signed residual |
| `apriori rho_max`, `tau/apriori` | largest residual over the tolerance that reads no output, and the mean of the applied tolerance over that one |
| `against a normalizer over cast weights` | largest residual over the tolerance of a kernel whose normalizer sums the cast weights, and the number of elements beyond it; a nonzero count is the uncast normalizer of the shipped kernel showing, not a failure |

`E_d` is a difference of stored values and an exact shift, so it is a multiple
of the grid spacing of the stored values. Two store roundings of full size and
equal sign give one spacing, which is the sum of the two store terms. In the
columns with `|c_d| = 1/8`, where the other terms are smallest, such an element
sits near `0.8` of its tolerance. A ratio of that size is explained only for an
element whose residual is exactly one spacing.

### Mutations

Two more launches apply a wrong shift while the analyzer keeps expecting `c`:

| Mutation | Applied shift |
|---|---|
| `omit_second_half` | `c` in the first `D / 2` columns, zero in the rest |
| `rescale_second_half` | `c` in the first `D / 2` columns, `c / 2` in the rest |

The mutated tensor is a legitimate input, so the pair obeys the relation with
the applied shift, and its residual against `c` is the missing shift plus an
error inside the tolerance. A wrong shift counts as rejected
(`shift_mutation_failures`) only when all of these hold:

- every value of the launch is finite and inside its guard: a faulty launch
  does not stand in for a rejection;
- no residual is beyond its tolerance in the first `D / 2` columns, which
  carry `c`;
- every residual is beyond its tolerance in each column whose missing shift is
  more than twice the tolerance that reads no output. At the quick shape that
  is every column of the second half.

The shift has period four, so the correct launch cannot tell an output column
from the one `D / 2` further on. The wrong shifts can: a kernel that exchanged
the halves would put the residuals in the first one.

A normalization mutation is applied on the host to the real outputs: both
stored outputs are scaled by `1 + eta` and converted again, for
`eta = 2^-2, ..., 2^-12`. The mutation acts after the kernel's own conversion.
Above a threshold derived from the tolerance (`output_scale_threshold`:
`7.2e-3` in FP16, `5.6e-2` in BF16 at the quick shape) a passing pair is
certain to be rejected, and the test fails when it is not. Below the threshold
the outcome is printed, together with the smallest scale of the unbroken
rejected run.

When the correct shift fails as well, the message of an unrejected mutation
reports its violations as the diagnosis.

### What the relation cannot see

- Anything both launches share cancels: scores, the scale, the running maximum,
  the exponential, KV-head routing, and a mask that is wrong but renormalized.
  A fixed error added to both outputs is seen only by the magnitude guard. The
  uniform-probability lane and the pinned SDPA lane cover those.
- A joint permutation of the keys and values, and a pairing of scores with the
  wrong values that leaves the row sum at one.
- A row-sum defect below the tolerance. Sensitivity depends on the dtype and on
  the attention profile, and no single claim covers them. One extra equal key
  in the normalizer of a 256-key row changes the row sum by `1/257`. At the
  largest shift that moves the residual by `1.9e-3`: above the FP16 tolerance
  that reads no output but below twice it, so its rejection is not certain,
  and below the BF16 one. On the concentrated profile a defect on a single
  non-dominant key is below both.
- The fixture's V has one significant bit and its shifted values at most four,
  so a precision defect on general value operands does not show.

### Cases

| Arch | Dtype | Profile | Case |
|---|---|---|---|
| gfx942 | FP16 | moderate | `B1,Sq256,Sk256,H8/2,D64`, unmasked |
| gfx942 | FP16 | concentrated | `B1,Sq256,Sk256,H8/2,D64`, unmasked |
| gfx942 | BF16 | moderate | `B1,Sq256,Sk256,H8/2,D64`, unmasked |
| gfx942 | BF16 | concentrated | `B1,Sq256,Sk256,H8/2,D64`, unmasked |

Each case is five launches: the base inputs, their replay, the shifted values
and the two shift mutations. The two profiles of a dtype launch one code
object. `VALUE_SHIFT_SUBJECT_CASES` in `_dense_subject.py` lists the
`(arch, case)` pairs, and the skip and failure semantics are those of the
uniform lane: only a device of another architecture skips.

The CPU contracts in `test_value_shift_contracts.py` exercise the analyzer on
an FP32 model of the kernel's single-pass online softmax. The modeled shift
stays inside the tolerance, on the fixture, on rows with a late maximum, and
with weights that are wrong in the normalizer and the accumulator alike. The
modeled wrong shifts and the output scales above the threshold are rejected.
Normalization defects of the model (a key omitted from or repeated in the
normalizer, a key omitted from the accumulator, the accumulator scaled or
rescaled apart from the normalizer, the weights of one tile changed in the
accumulator only) are rejected wherever the tolerance decides, and reported
otherwise. The model lives only in those contracts.

The same file pins the tolerance's constants against the derived expressions,
requires the vectorized analyzer to agree with a scalar specification on every
field, and applies the artifact checks of `_value_shift_path.py` to the lowered
quick kernels of every compiler lane and to edited copies that each check must
reject.

## Oracle prohibition

Verdicts may use only deterministic profile construction, independent legal-key
arithmetic, exact rational rounding, a tolerance derived from the kernel's
arithmetic path and input statistics (in the value-shift lane also from the
stored bits, for its store term), and diagnostic reductions. They must not call
Torch SDPA, rocKE numeric references, SDPA reference helpers, or a
pinned-reference bundle. A CPU source contract checks the suite for those
dependencies. The kernel models in the key-permutation and value-shift
contracts take part in no GPU verdict.

The whole suite is Torch-free, and CPU contracts enforce it in two ways:

- An AST check over every Python file in this directory rejects any `torch`
  import, at module level or inside a function, and any string-based import of
  it.
- An import-blocker check loads the suite, resolves every quick subject and
  imports its runner module with `torch` made unimportable.

The suite therefore runs, and gates, in environments that provide only NumPy
and the rocKE runtime.

## Running the suite

From the `rocke/library` directory (NumPy and pytest only; Torch is not needed):

```bash
# CPU contracts: rounding, encoding, mutations, policy, vectorized-versus-scalar
# analyzer agreement, proof scope, adapter ABI, Torch-free and oracle-free
# source policy; then the key-permutation maps, inputs, bound and analyzer; then
# the value-shift inputs, tolerance, analyzer and path checks.
python -m pytest tests/oracle_free/test_contracts.py \
  tests/oracle_free/test_permutation_contracts.py \
  tests/oracle_free/test_value_shift_contracts.py -vv

# Uniform-probability, key-permutation and value-shift production launches on a
# visible gfx942 device.
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/oracle_free/test_analytical_dense.py \
  tests/oracle_free/test_permutation_dense.py \
  tests/oracle_free/test_value_shift_dense.py -vv -rs
```

From the installed `bin/hip_kernel_provider` directory, the GPU launches and the
CPU contracts are separate entries:

```bash
ctest -R '^rocke_oracle_free_pytest$' --output-on-failure --no-tests=error
ctest -R '^rocke_library_attention_cpu_pytest$' --output-on-failure --no-tests=error
```
