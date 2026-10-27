# ADR 0028: Strengthen bounded set-cover observables

Status:  Accepted
Defect:  none — behavior is intended

## Context

ADR 0026 replaced basename snapshots with expected generated-kernel counts, but
the harness derives and emits at most eight fork permutations. An expected
count of eight therefore meant “at least eight” for 24 cases. Cases marked with
`all_ok=False` also accepted any combination of emitter return codes, including
failure of every sampled kernel.

The canonical assembly text already removes the only known scheduler-state
difference, the `matrix_a_reuse` and `matrix_b_reuse` hints. Per-call throwaway
emits duplicated work without changing the recorded result.

## Decision

Record the complete pre-limit fork-permutation count for every selected problem
group. For the bounded emitted sample, record the exact multiset of emitter
return codes rather than a success/known-failure boolean. Continue to assert
focused source patterns where the final assembly exposes a stable behavior.

Remove throwaway warm-up emits from the config- and logic-driven harnesses.
Canonicalization, repeated-emission tests, and the saved expected results remain
responsible for detecting order-dependent output.

## Consequences

A change beyond the eight-permutation sample is visible through the full fork
count, and a sampled kernel changing between success and failure is visible
through the status multiset. The suite still does not claim to emit every
solution from large configuration files.

The rejection tests intentionally record exact diagnostic text because the
message identifies which guard fired. Rewording a diagnostic requires reviewing
and updating the corresponding saved expected result.
