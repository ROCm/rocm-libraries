# Oracle-free attention tests

This suite launches the shipped gfx942 dense attention kernel and checks masked
uniform probabilities without a complete attention oracle. Zero Q makes
every legal score equal, so the softmax row for query `q` is uniform over its
legal key set `A(q)`. Each expected output follows from legal-key arithmetic
alone, rounded correctly into the subject dtype.

## Exposures

Each case runs five launches on the same zero-Q, finite, nonconstant K:

| Exposure | V | Expected `O[q,d]` |
|---|---|---|
| `basis@o`, `o in {0,64,128,192}` | `V[k,d] = 1` iff `k = o + d` | `1/\|A(q)\|` if `o + d` is legal, else `0` |
| `folded` | `V[k,d] = 1` iff `d = k mod D` | `\|{k in A(q) : k mod D = d}\| / \|A(q)\|` |

The four basis launches expose each of the 256 probability columns exactly
once. The folded launch observes the same row through sums of several
probabilities, which catches errors that cancel across basis launches.

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

The subject adapter additionally requires a non-persistent, single-pass
(`kv_segments == 1`), non-ragged, top-left dense kernel without sinks or LSE.
If the dispatch outcome moves outside that scope, resolution fails, which forces
a recalibration instead of reusing the old one.

When there is no entry, or the case or fingerprint falls outside the entry's
scope, the test skips with the full diagnostic line. Mutation self-checks and
calibration capture both run before that skip. Budgets are never widened from
a run under judgment. If a ratified case stops being bit-exact, remove its
entry and investigate.

Set `ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>` to write one JSONL file per case, with one
record per `(mutation, exposure, N_q)`: rows, compared, exact, max ULP and max
stored row-sum residual. These records are the evidence for ratifying an entry.

## Oracle prohibition

Verdicts may use only deterministic profile construction, independent legal-key
arithmetic, exact rational rounding, and diagnostic reductions. They must not
call Torch SDPA, rocKE numeric references, SDPA reference helpers, or a
pinned-reference bundle. `_analytical.py` is Torch-free, and a CPU source
contract checks the suite for those dependencies.

## Running the suite

From the `rocke` directory:

```bash
# CPU contracts: rounding, mutations, policy, calibration scope, adapter ABI.
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
