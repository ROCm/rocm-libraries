# Oracle-free attention tests

This suite launches the shipped gfx942 dense attention kernel and checks masked
uniform probabilities without a complete attention oracle. Zero Q makes
every legal score equal, so the softmax row for query `q` is uniform over its
legal key set `A(q)`. Each expected output follows from legal-key arithmetic
alone, rounded correctly into the subject dtype.

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
wrong Q or K head index is invisible. Query heads that share one KV head (GQA)
also share a scale, so swapping heads within one group is invisible.

## Device path

The suite does not use Torch. Inputs are built in FP32 and then encoded to raw
FP16/BF16 `uint16` bits by `encode_bits`. That function rejects any value that
would round, so the kernel reads exactly the values the expectations assume.
Signed zero survives the encoding, and NaN or out-of-range inputs are refused.

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
`source_revision` names the develop revision whose kernel source was ratified
on gfx942.

Some fields are deliberately left out of the fingerprint: `waves_per_eu`,
`nonpersist_decode`, `use_cfvst`, K double buffering (`lds_num_buffers`) and V
register prefetch (`v_reg_prefetch`). They change only scheduling, data movement
or the CTA-to-work mapping, not the arithmetic the zero-ULP argument rests on.
None of them is active in the calibrated cases anyway:

- CFVST is enabled only at D128.
- K double buffering and V prefetch require the BF16 D128 persistent body.

The subject adapter additionally requires a non-persistent, single-pass
(`kv_segments == 1`), non-ragged, top-left dense kernel without sinks or LSE.
If the dispatch outcome moves outside that scope, resolution fails, which forces
a recalibration instead of reusing the old one.

### Skip and failure semantics

The GPU cases skip only when the visible device is not gfx942. The skip reason
names the architecture HIP reports.

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
gate, so a failing run still writes its records. Budgets are never widened from
a run under judgment. If a ratified case stops being bit-exact, remove its
entry and investigate.

### Re-ratifying an entry

Set `ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>` to write one JSONL file per case,
with one record per `(mutation, exposure, N_q)`: rows, compared, exact, max ULP
and max stored row-sum residual. These records are the evidence for ratifying
an entry.

After a kernel, dispatch or base-revision change:

1. Run the complete suite on gfx942 with
   `ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>` set.
2. Confirm every control record is bit-exact (`compared == exact`, max ULP `0`,
   stored row-sum residual `0`) and that every mutation was rejected.
3. In one reviewed change, update the affected `UNIFORM_CALIBRATION` entry:
   `spec`, `case_ids`, and `source_revision` (the develop revision under test).

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
- An import-blocker check loads the suite with `torch` made unimportable.

The suite therefore runs, and gates, in environments that provide only NumPy
and the rocKE runtime.

## Running the suite

From the `rocke` directory (NumPy and pytest only; Torch is not needed):

```bash
# CPU contracts: rounding, encoding, mutations, policy, calibration scope and
# fail-closed gating, adapter ABI, Torch-free and oracle-free source policy.
python -m pytest library/tests/oracle_free/test_contracts.py -vv

# Uniform-probability production launches on a visible gfx942 device.
HIP_VISIBLE_DEVICES=0 python -m pytest \
  library/tests/oracle_free/test_analytical_gfx942.py -vv -rs

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
