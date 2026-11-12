# Oracle-free attention tests

This suite launches the shipped dense attention kernels and checks masked
uniform probabilities without a complete attention oracle. Zero Q makes
every legal score equal, so the softmax row for query `q` is uniform over its
legal key set `A(q)`. Each expected output follows from legal-key arithmetic
alone, rounded correctly into the subject dtype.

A second lane with structured logits, where Q and K reads matter, is described
in [README_P4.md](README_P4.md).

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

## Oracle prohibition

Verdicts may use only deterministic profile construction, independent legal-key
arithmetic, exact rational rounding, and diagnostic reductions. They must not
call Torch SDPA, rocKE numeric references, SDPA reference helpers, or a
pinned-reference bundle. A CPU source contract checks the suite for those
dependencies.

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
# source policy.
python -m pytest tests/oracle_free/test_contracts.py -vv

# Uniform-probability production launches on a visible gfx942 device.
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/oracle_free/test_analytical_dense.py -vv -rs
```

From the installed `bin/hip_kernel_provider` directory, the GPU launches and the
CPU contracts are separate entries:

```bash
ctest -R '^rocke_oracle_free_pytest$' --output-on-failure --no-tests=error
ctest -R '^rocke_library_attention_cpu_pytest$' --output-on-failure --no-tests=error
```
