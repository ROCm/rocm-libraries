# Oracle-free attention tests

This suite launches the shipped dense attention kernels and checks them without
a complete attention oracle. It has two lanes:

| Lane | Checks | Verdict |
|---|---|---|
| Uniform probabilities ([Exposures](#exposures) to [Scope](#scope)) | masked attention on zero Q | every stored bit equals a closed form |
| [Key-permutation relation](#key-permutation-relation) | unmasked attention on nonuniform scores | two launches agree within a derived bound |

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

`permutation_bound` evaluates it from the dtype, the key count, the key-tile
count of the spec dispatch resolved, `max|V|`, and a bound on the spread of a
row's scores taken from the row norms of Q and K. It reads no output, so a
faulty kernel cannot widen its own tolerance, and it is not fitted to any
measurement. The worst case needs the rounding of every weight to line up, so a
correct kernel sits well inside the bound. The largest residual is reported as
a fraction of the bound; a fraction that grows over time is a drift signal even
while the test passes.

The bound holds for every single-pass dense variant, whatever its tile, grid,
persistence or scheduling knobs, under these assumptions about the executed
path:

- FP32 addition, multiplication and the reciprocal return one of the two FP32
  neighbours of their exact result, and each MFMA call accumulates its products
  by such additions in some order;
- `exp2` is within one FP32 ULP;
- the probability cast and the output store round to nearest even and keep
  subnormals;
- the normalizer sums uncast weights, the accumulator consumes cast ones, and
  both are rescaled by `exp2(m_old - m_new)` per key tile.

The last item is checked on the CPU. A contract lowers each quick spec to LLVM
IR and classifies every floating-point instruction against that path: no mask on
the scores, a key-tile loop over `seqlen_kv / block_n`, uncast weights in the
normalizer, one cast into the PV operand, one `alpha` for the normalizer and for
every accumulator element, one `fdiv 1.0, l`, and no fast-math flag. An
instruction outside the path fails with "re-derive the bound"; the bound is
never widened to absorb it. The rounding behaviour of the compiled code object
(the first three items) is not checked here.

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

## Oracle prohibition

Verdicts may use only deterministic profile construction, independent legal-key
arithmetic, exact rational rounding, a tolerance derived from the kernel's
arithmetic path and input statistics, and diagnostic reductions. They must not
call Torch SDPA, rocKE numeric references, SDPA reference helpers, or a
pinned-reference bundle. A CPU source contract checks the suite for those
dependencies. The kernel model in the key-permutation contracts takes part in no
GPU verdict.

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
# source policy; then the key-permutation maps, inputs, bound and analyzer.
python -m pytest tests/oracle_free/test_contracts.py \
  tests/oracle_free/test_permutation_contracts.py -vv

# Uniform-probability and key-permutation production launches on a visible
# gfx942 device.
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/oracle_free/test_analytical_dense.py \
  tests/oracle_free/test_permutation_dense.py -vv -rs
```

From the installed `bin/hip_kernel_provider` directory, the GPU launches and the
CPU contracts are separate entries:

```bash
ctest -R '^rocke_oracle_free_pytest$' --output-on-failure --no-tests=error
ctest -R '^rocke_library_attention_cpu_pytest$' --output-on-failure --no-tests=error
```
