# COMGR capability churn and interpretation toil

## Contents

- [Scope](#scope)
- [Observed churn](#observed-churn)
- [Interpretation work](#interpretation-work)
- [Steady-state assessment](#steady-state-assessment)
- [Decision](#decision)

## Scope

This report assesses the nine selected weekly snapshots, three release
baselines, and five processor contexts recorded in
[`historical_observations.json`](historical_observations.json). The selected
weekly rows retain the first broad-survey sample and both sides of every observed
structural transition. This is compiler-compatibility evidence, not numeric or
performance qualification.

## Observed churn

The selected transition pairs contain four structural changes:

- one changes accepted buffer-resource ABI candidates and named barriers; and
- three change named-barrier address-space outcomes, including crash versus
  normal rejection and parser-stage versus codegen-stage failure.

The initial 43-week survey found no structural change at 37 of 42 adjacent
boundaries. Those unchanged intermediate artifacts do not remain in the
maintained replay set.

Generated HSACO digests are also excluded. A digest change can route a compiler
to generated-code or resource qualification, but it is not a configuration
capability.

## Interpretation work

A new structural result requires bounded review:

1. confirm the exact COMGR digest and public artifact;
2. compare candidate outcome and failed stage per affected processor;
3. determine whether the change affects a form rocKE emits or may select;
4. run complete production-kernel validity for a proposed selection change;
5. update Python and C++ policy together when emission changes; and
6. run generated-code, resource, and numeric qualification where a microprobe
   cannot establish the required property.

The report removes the need to correlate unrelated Clang, `opt`, and COMGR
identities. It does not infer policy from acceptance.

## Steady-state assessment

The maintained report has six capability families and 15–16 bounded compiler
jobs per processor. The previous broad matrix repeated global policy choices for
every operation and included options, attributes, and output fingerprints that
did not belong to the flavor-replacement decision.

Schema growth follows four rules:

- add a candidate only for a production COMGR distinction rocKE must select;
- group every operation controlled by one policy into one all-family candidate;
- retain processor and failed-stage context; and
- retire settled probes and unchanged historical samples.

## Decision

Direct COMGR probing remains the evidence layer for replacing automatic
LLVM-major profile inference. Production integration stays incremental:
evidence loading and normalized policy first, then one independently qualified
emission decision per reviewable change. Legacy flavor modes remain temporary
offline and fallback interfaces until each production consumer and artifact
identity has migrated.
