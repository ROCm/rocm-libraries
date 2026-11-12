# Oracle-free attention tests

This suite launches the shipped gfx942 dense attention kernel and checks it
without a complete attention oracle, in two lanes:

- **Uniform probabilities** (this section through "Calibration and scope").
  Zero Q makes every legal score equal, so the softmax row for query `q` is
  uniform over its legal key set `A(q)`. Each expected output follows from
  legal-key arithmetic alone, rounded correctly into the subject dtype.
- **Structured logits (P4)**, below. Structured Q and K give every row two
  logit classes, so the softmax, the online rescale and the per-head Q/K reads
  all matter. Each output is predicted as the set of values the kernel's
  pinned floating-point path can store.

## Exposures

Each case runs five launches on the same zero-Q, finite, nonconstant K:

| Exposure | V for KV head `h` | Expected `O[q,h_q,d]`, with `s = 2^h_kv(h_q)` |
|---|---|---|
| `basis@o`, `o in {0,64,128,192}` | `V[k,h,d] = 2^h` iff `k = o + d` | `s/\|A(q)\|` if `o + d` is legal, else `0` |
| `folded` | `V[k,h,d] = 2^h` iff `d = k mod D` | `s * \|{k in A(q) : k mod D = d}\| / \|A(q)\|` |

The four basis launches expose each of the 256 probability columns exactly
once. The folded launch observes the same row through sums of several
probabilities, which catches errors that cancel across basis launches.

Each KV head scales its V profile by a distinct power of two, `2^h`, and query
head `h_q` expects its KV head's scale, `h_kv = h_q // (Hq / Hkv)`. Without
that, every head would carry identical outputs, so a kernel that reads one V head
for all heads, uses a wrong head stride, or writes one output head everywhere
would pass. With it, every legal coordinate of a wrong head is off by at least a
factor of two. A power of two is exact in FP16 and BF16 and commutes with every
rounding the kernel performs (the P cast, FP32 accumulation, normalization, the
output cast), so head `h`'s correct output is head 0's shifted in exponent and
the zero-ULP argument is unchanged. `nhead_k` is capped at 8 to keep the scales
distinct and the outputs in range.

Not covered by design: with Q zero, the logits do not depend on Q or K, so a
wrong Q or K head index is invisible here; the P4 lane gives each head its own
logit and covers it. Query heads that share one KV head (GQA) also share a
scale, so swapping heads within one group is invisible.

## Device path

The suite does not use Torch. Inputs are built in FP32 and then encoded to raw
FP16/BF16 `uint16` bits by `encode_bits`. That function rejects any value that
would round, so the kernel reads exactly the values the expectations assume.
Signed zero survives the encoding, and NaN or out-of-range inputs are refused.
BF16 goes through rocKE's shared codec (`rocke.numeric.bf16`); FP16 uses numpy's
native `float16`. Only the input and output encoding is shared; the expected
bits come from exact rational rounding in `_analytical.py`.

`run_dense_case` in `_gfx942_subject.py` launches one case:

1. It allocates one `rocke.runtime.DeviceMem` buffer each for Q, K, V and the
   output. Before upload it checks each buffer's size, 16-byte alignment and
   separation.
2. It prefills the output with quiet-NaN bits, so an unwritten coordinate is
   counted `invalid`.
3. It uploads with `Runtime.memcpy_h2d`.
4. It launches through `kernels.gfx942.attention_dense.launch_attention_dense`.
5. It reads the output back with `Runtime.memcpy_d2h` as raw bits, with no float
   conversion.

`launch_attention_dense` is the same compile, cache and launch path that
`run_attention_dense_torch` delegates to after validating its tensors. The suite
therefore exercises the shipped launch path and does not keep its own copy of
the kernel signature, geometry or runtime arguments.

## Policy

Expected values are correctly rounded (round-to-nearest-even) raw FP16/BF16 bit
patterns of the exact rationals above. `analyze_uniform_policy` compares raw output
bits, and every one of these rules is a hard violation:

- every output is finite (`invalid`), and no output is negative (`negative_max`);
- every legal coordinate lies within `ulp_budget = 0` ULP of its target
  (`target_ulp`), so the match is bit-exact;
- masked lanes are exactly zero (`masked_max`, `masked_mass`); `-0` is accepted;
- basis lanes past the last key are exactly zero (`trailing_max`);
- per row, the sum of stored outputs equals the sum of stored targets exactly
  (`stored_row_sum`). This is checked for the folded exposure and for the union
  of basis exposures, wherever the exposure covers the full legal set;
- at least one legal coordinate is observed.

No absolute tolerance participates in any verdict. The analyzer reports
per-`N_q = |A(q)|` evidence separately inside and beyond the proven range
`N_q < 2^(21-p)`, where `p` is the dtype's fraction bits (scaled by `2/3` for
the folded exposure).

### Mutation self-checks

On every run, the analyzer is re-applied to the same real output under rules
that are wrong by one key or by one in the denominator. Each mutated rule must
be rejected, or the test fails regardless of calibration:

| Case | Mutations |
|---|---|
| causal | `include_future_key`, `exclude_last_key`, `denominator_plus_one` |
| window | `include_old_window_key`, `exclude_oldest_window_key`, `denominator_plus_one` |

The same real output with query head 0 copied into every head
(`broadcast_query_head`) must also be rejected, which shows the run can tell
heads apart. Magnitude diagnostics (masked mass, row-sum residuals) are reported
in probability units, divided by each head's scale, so the calibration records
do not depend on the head scaling.

## Calibration and scope

`UNIFORM_CALIBRATION` in `_analytical.py` gates a result only through a ratified
entry keyed by `(arch, candidate_name, dtype)`:

| Dtype | Case | Resolved tile | `ulp_budget` |
|---|---|---|---:|
| FP16 | `B1,Sq256,Sk256,H4/4,D64`, top-left causal | `128x64` | 0 |
| BF16 | `B1,Sq256,Sk256,H4/4,D64`, sliding window 64 | `64x64` | 0 |

Each entry closes its scope with the exact case IDs and a `UniformSpecFingerprint`
of the dispatch-resolved spec. The fingerprint records path, persistence,
runtime shape, head size, `block_m`, `block_n`, fast exp2, sinks and LSE.
Installing an entry is the ratification. The develop revision and the gfx942
evidence behind it are recorded in the commit that edits the table, not in the
code, so a rebase does not need a re-pin.

Some fields are deliberately left out of the fingerprint: `waves_per_eu`,
`nonpersist_decode`, `use_cfvst`, K double buffering (`lds_num_buffers`) and V
register prefetch (`v_reg_prefetch`). They change only scheduling, data movement
or the CTA-to-work mapping, not the arithmetic the zero-ULP argument rests on.
None of them is active in the calibrated cases anyway:

- CFVST is enabled only at D128.
- K double buffering and V prefetch require the BF16 D128 persistent body.

Before launching, the subject adapter checks only what the launch itself needs:

- the dense gfx942 candidate;
- the requested shape, which sizes the host buffers;
- no sinks and no LSE, because it allocates only Q, K, V and O;
- the scope the fingerprint does not record: single-pass (`kv_segments == 1`),
  non-ragged, top-left.

If the dispatch outcome moves outside that, resolution fails. A drift in any
other fingerprint field (persistence, runtime shape, tile, fast exp2) still
launches and writes its calibration records, then fails at `require_calibration`
with the remedy.

### Skip and failure semantics

The GPU cases skip only when the visible device is not gfx942. They use the
shared `rocke.tests.gpu.current_device_arch_in` gate, so the skip reason names
the GPU found, and a lane that sets `ROCKE_REQUIRE_GFX942=1` fails instead of
skipping.

On gfx942, the cases in `UNIFORM_QUICK_CASES` always run. The checks below are
failures, never skips. `require_calibration` reports each one with field-level
diagnostics from `CalibrationEntry.mismatch` plus the re-ratification remedy:

- no calibration entry for the resolved `(arch, candidate_name, dtype)`;
- the case is not among the entry's case IDs;
- any fingerprint field differs from the calibrated value;
- no legal coordinate was compared.

A dispatch change that selects a different variant for a calibrated shape
therefore fails the gate until it is re-verified. A CPU contract pins
`UNIFORM_QUICK_CASES` to the calibrated case set, so a calibrated case cannot
drop out of the parametrization unnoticed.

Mutation self-checks and calibration capture both run before the calibration
gate, so a failing run still writes its records. A kernel with a mask or
denominator bug can write exactly what a mutation predicts. When that happens,
the unrejected mutation is reported together with the control's violations,
not as a checker failure. Budgets are never widened from a run under judgment.
If a ratified case stops being bit-exact, remove its entry and investigate.

### Re-ratifying an entry

Set `ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>` to write one JSONL file per case,
with one record per `(mutation, exposure, N_q)`: rows, compared, exact, max ULP
and max stored row-sum residual. These records are the evidence for ratifying
an entry.

When a kernel or dispatch change makes the gate fail with a calibration
mismatch:

1. Run the complete suite on gfx942 with
   `ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>` set.
2. Confirm every control record is bit-exact (`compared == exact`, max ULP `0`,
   stored row-sum residual `0`) and that every mutation was rejected.
3. In one reviewed change, update the affected `UNIFORM_CALIBRATION` entry
   (`spec`, `case_ids`). Name the develop revision under test and the gfx942
   run in that commit's message.

## Structured logits (P4)

The P4 lane (`_p4_analytical.py`, `test_p4_analytical_gfx942.py`) runs the two
calibrated quick shapes above with structured Q and K. Every row then has two
logit classes: the distinguished keys, with logit `c`, and the background keys,
with logit 0. For a row with `n_c` legal distinguished and `n_0` legal
background keys, an exposed coordinate that selects `m_c` and `m_0` of them has
the exact value

`y* = 2^h (m_c e^c + m_0) / (n_c e^c + n_0)`.

The V exposures and the `2^h` KV-head scales are the uniform lane's.

### Fixtures

All inputs are zero or signed powers of two, so `encode_bits` accepts them
exactly. `D = 64` is also the period `P`; key `k` is distinguished for row `q`
when `k + r = q (mod 64)`.

| Fixture | Q[q,h,:] | K[k,h,:] | `r` | Logit `c` |
|---|---|---|---:|---|
| `identity` | `a_h` one-hot at `q mod 64` | `b_h` one-hot at `k mod 64` | 0 | `a_h b_h / 8` |
| `cyclic` | as `identity` | one-hot at `(k + 37) mod 64` | 37 | `a_h b_h / 8` |
| `walsh` | `a_h` times Walsh row `q mod 64` | `b_h (-1)^k` times Walsh row `(k + 1) mod 64` | 1 | `8 a_h b_h`, signed by row parity |
| `late_max` | as `identity` | as `identity`, rows `k < 64` zeroed | 0 | as `identity`, keys `k >= 64` only |

The head scales `(a_h, b_h)` give each head its own logit (periodic fixtures
`8, -8, 2, -1/2`; Walsh magnitudes `8, 2, 1/2, 4`). They were chosen by a
maximin search so that every wrong Q-head or K-head read changes some stored
value; a CPU contract recomputes the committed margin. Causal rows contain
masked distinguished keys (`k = q + 64j`), so any mask leak is visible.
Window-64 rows straddle two KV tiles, and `late_max` moves the row maximum
into a later tile; both exercise the online rescale.

### Prediction: Tier B

For each coordinate the analyzer computes **Tier B**, the set of FP16/BF16
values the kernel's pinned floating-point path can store. Closed intervals of
FP32 values are propagated, with exact rational arithmetic, through the score
scale, `v_exp_f32`, the FP32 denominator sum in any order, the single P cast,
the PV MFMA, the reciprocal `fdiv 1.0, l` and the output cast. Rounding, `+`,
`*` and the casts are monotone on nonnegative arguments, so the endpoint images
enclose every reachable state.

The denominator sum uses the `gamma` bound only on a tile with normal-scale
terms. A fully masked tile after a legal one holds only weights of at most
`2^-149` and is enclosed explicitly by `[0, 2^-143]`. A fully masked leading
tile (rows 127, 191 and 255 of the window kernel) sums 64 values near one; the
first legal tile rescales that state by at most `2^-149`, which leaves the
denominator unchanged and a residue of at most `2^(h-149)` in the numerator. A
row with more than one fully masked leading tile, with no legal tile, or with a
denominator outside `[1/2, 2^9)` has no enclosure: it raises `P4ScopeError`
instead of being widened.

The path itself is pinned twice on the CPU. An IR contract lowers both quick
specs and classifies every floating-point instruction, follows each MFMA
operand back to a stored value or the probability cast, and requires eight
chained calls per accumulator, from zero for a score and from the rescaled
accumulator for PV. It also evaluates the KV loop limits and the mask selects
of the lowered IR for every row: the kernel must visit exactly the modeled
tiles and keep exactly the keys with `q - W < k <= q`. An ISA contract
disassembles the compiled code objects: one division with every FP32 FMA
inside it, the FP16 epilogue as the only fused site, round-to-nearest-even
modes in the kernel descriptor, no write to the MODE register, and the
round-to-nearest-even sequence at every BF16 conversion. Any instruction
outside the model fails with "re-derive the path model": Tier B is re-derived
for the new path, never widened to absorb it.

Tier B rests on six named assumptions. A-E, A-MFMA and A-F are versioned in
the calibration entry today. A-QK, A-DIV and A-MODE name premises the same
enclosure has always used; they add no value to any set and join the entry
when it is re-ratified.

| Assumption | Statement |
|---|---|
| A-E v1 | `v_exp_f32` returns an FP32 value within `ulp32(2^x)` (the spacing above `2^x`) of `2^x` |
| A-MFMA v1 | an MFMA call with at most one nonzero product `p` returns `c + p` rounded down or up (exact when `c = 0`) |
| A-QK v1 | an MFMA call whose accumulator and products are multiples of `2^e` with total magnitude at most `2^(24+e)` returns their exact sum (the Walsh scores) |
| A-DIV v1 | the executed `fdiv 1.0, l` returns `RN(1/l)` for every FP32 `l` in `[1/2, 2^9)` |
| A-F v1 | each FP16 output store is `q(RN32(o R))` or `q(o R)`; every other operation, BF16 stores included, follows the IR |
| A-MODE v1 | FP32 add, subtract and multiply round to nearest even; the FP16 and BF16 conversions round to nearest even and keep subnormals |

A-F covers an AMDGPU backend behaviour: affected compilers select some
unflagged `fptrunc(fmul float)` FP16 stores as `v_fma_mix{lo,hi}_f16`, which
rounds the exact product once. LLVM PR
[#236759](https://github.com/llvm/llvm-project/pull/236759) makes that fold
require `contract`. The ISA contract accepts both affected and fixed compilers
and checks that the denominator combine is never fused.

**Tier A**, the certified correctly rounded target `q(y*)` with a small ULP
radius, is diagnostic: it names the failure class, never the verdict.

### Policy and failure labels

`analyze_p4_policy` compares raw output bits. A coordinate passes if and only
if its bits lie in Tier B. Structural coordinates (masked keys, basis
coordinates past the last key, rows without a legal key) have Tier B `{+0}`
and must store exactly `+0`; unlike the uniform lane, `-0` fails. A failing
value that equals what the IR-exact path stores with one assumption violated
carries that assumption's label; the other assumption labels are heuristics:

| Label | Meaning |
|---|---|
| `invalid` | NaN (including the unwritten prefill) or Inf |
| `negative` | sign bit set, `-0` included |
| `structural` | a required `+0` coordinate stores another value |
| `assumption A-MODE` | the store of a truncating conversion; or an FP16-subnormal target stored near it or flushed to `+0` |
| `assumption A-F` | an FP16 store of the exact product truncated toward zero |
| `assumption A-DIV` | the store of a reciprocal one FP32 step from `RN(1/l)` |
| `assumption A-QK` | the store of a Walsh score one FP32 step from exact |
| `assumption A-MFMA` | a folded addition near the target, with the basis exposures clean |
| `assumption A-E` | explained only by a looser (2-ULP) `v_exp_f32` |
| `path model` | outside Tier B, inside Tier A: re-pin the path, never widen |
| `numerical` | outside Tier A |

A CPU contract replaces one primitive at a time by a value its assumption
excludes and requires every coordinate that leaves Tier B to fail with a named
label. Only the conversions are visible in the stored bits: a truncating cast
and a truncated FP16 store move about 6 to 9 percent of the coordinates out of
Tier B. An exponential two FP32 steps outside its candidate set, a folded
addition two steps outside the faithful pair, a Walsh score one step off, a
reciprocal one step off and round-toward-zero FP32 arithmetic leave Tier B at
no coordinate of the quick matrix (`P4_UNOBSERVED_VIOLATIONS`). That is a
limit of what the stored bits can show, not evidence for those assumptions.

### Mutation self-checks

On every run the analyzer is re-applied to the same real output under wrong
semantics, on block, tile and window edge rows. Each mutation must be
rejected:

| Applies to | Mutations |
|---|---|
| every fixture | `omit_sqrt_d`, `exp_encoding`, `k_row_plus_one`, `k_row_minus_one`, `q_head_rotate`, `k_head_rotate`, `omit_rescale`, `denominator_plus_one` |
| `cyclic` | `inverse_shift` |
| `walsh` | `omit_walsh_block_{0,1,3}`, `walsh_sign` |
| causal / window | the one-key mask-boundary mutations of the uniform lane |

A CPU contract proves, for every applicable pair, that the correct and mutated
Tier B are disjoint at some coordinate and records the gap. The exception is
`omit_rescale` on the causal `identity`, `cyclic` and `walsh` cases: their row
maximum is always in the first visited tile, so the omitted rescale lies inside
the correct Tier B. `late_max` separates it on the same kernel.

Two mutated rows have no derived enclosure and are left out of the comparison:
row 0 under `exclude_last_key` (no legal key remains) and row 0 of `identity`
under `denominator_plus_one` (the phantom key takes the denominator to about
`e^8`, outside the A-DIV range). Both mutations stay separated on other rows.

The kernel is also launched once with scale `1` instead of `1/sqrt(D)` on the
`identity` fixture (`basis@0`); that output must fail Tier B.

### Calibration

`P4_CALIBRATION` has one entry per dtype, keyed like the uniform table. A
`P4CalibrationEntry` names the policy (`tier_b`), the assumption versions, the
literal case IDs and the uniform lane's dispatch fingerprint. It has no width,
ULP or tolerance field, so no measurement can enlarge Tier B. A CPU contract
requires the calibrated case set to equal `P4_QUICK_CASES`, split by dtype.
Changing a fixture, a head scale or a quick shape changes the case IDs, and the
gate fails closed with the re-ratification remedy. Skip and failure semantics
are the uniform lane's.

Each case runs the five exposures twice, and the two runs must be
bit-identical. With `ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>` set, each case
writes `<case_id>.jsonl` with three record types:

- `rocke_p4_group`: per mutation, exposure, legal count and head, the label
  counts, exact hits on the target and two-value Tier B outcomes;
- `rocke_p4_summary`: per mutation, whether it was rejected and the CPU gap;
- `rocke_p4_diagnostics`: determinism, assumptions, the compiler and HSACO
  digest, agreement with the IR-exact emulation and with the named
  affected-compiler artifact, folded MFMA realizations and the Tier B width
  histogram. None of these enters the verdict.

It also writes `<case_id>.npz`, the raw output bits of every exposure. The
replayable audit `_p4_capture_audit.py` explains every saved coordinate by the
exact union of the IR-exact and the single-rounded epilogue, checks the named
artifact's per-site rule and Tier B membership, and compares further captures
bit for bit. Unexplained coordinates are reported, never absorbed:

```bash
# From rocke/library.
PYTHONPATH=.:../platform/python python -m tests.oracle_free._p4_capture_audit \
  --artifact rocm-7.13.0 <capture dir> [<capture dir> ...]
```

To re-ratify, run the complete suite on gfx942 with records enabled, confirm
zero coordinates outside Tier B, every mutation rejected and determinism, run
the audit on the captures, then update `P4_CALIBRATION` (`spec`, `case_ids`)
in one reviewed change that names the develop revision and the gfx942 run.

## Oracle prohibition

Verdicts may use only deterministic profile construction, independent legal-key
arithmetic, exact rational rounding, and diagnostic reductions. They must not
call Torch SDPA, rocKE numeric references, SDPA reference helpers, or a
pinned-reference bundle. A CPU source contract checks the suite for those
dependencies.

The P4 predictor is within these rules. Its inputs are closed-form scalars, the
logit of each key class and the legal counts of each row, and it propagates
FP32 intervals through the pinned per-row arithmetic. It never forms `Q K^T`
from input tensors or evaluates a softmax over data; general attention stays
banned.

The whole suite is Torch-free, and CPU contracts enforce it in two ways:

- An AST check over every Python file in this directory rejects any `torch`
  import, at module level or inside a function, and any string-based import of
  it.
- An import-blocker check loads the suite with `torch` made unimportable.

The suite therefore runs, and gates, in environments that provide only NumPy
and the rocKE runtime.

## Running the suite

From the `rocke` directory (NumPy and pytest only; Torch is not needed):

```bash
# CPU contracts: rounding, encoding, mutations, policy, calibration scope and
# fail-closed gating, adapter ABI, Torch-free and oracle-free source policy.
python -m pytest library/tests/oracle_free/test_contracts.py -vv

# P4 CPU contracts: reachable-set primitives, fixtures, Tier B soundness,
# mutation separation, IR and ISA path contracts, policy and calibration.
python -m pytest library/tests/oracle_free/test_p4_contracts.py -vv

# Uniform-probability and P4 production launches on a visible gfx942 device.
HIP_VISIBLE_DEVICES=0 python -m pytest \
  library/tests/oracle_free/test_analytical_gfx942.py \
  library/tests/oracle_free/test_p4_analytical_gfx942.py -vv -rs

# Complete suite, capturing calibration records.
ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir> HIP_VISIBLE_DEVICES=0 \
  python -m pytest library/tests/oracle_free -vv -rs
```

From the installed `bin/hip_kernel_provider` directory:

```bash
ctest -R '^rocke_oracle_free_pytest$' \
  --output-on-failure --no-tests=error
```

The installed `rocke_oracle_free_pytest` CTest entry already collects this whole
directory, so new support files need no CMake change.
