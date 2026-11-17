# COMGR capability churn and interpretation toil

## Contents

- [Scope](#scope)
- [Observed churn](#observed-churn)
- [Interpretation work](#interpretation-work)
- [Steady-state assessment](#steady-state-assessment)
- [Decision](#decision)

## Scope

This report assesses the 43-weekly-snapshot, three-release-baseline,
five-processor COMGR study recorded in
[`historical_observations.json`](historical_observations.json). It distinguishes
structural compatibility outcomes from generated-code fingerprints; neither is
numeric or performance qualification.

## Observed churn

The structural capability vector changes at five of 42 adjacent weekly
boundaries. The five distinct vectors are separated by changes in:

- focused `-O0` code generation;
- accepted buffer-resource ABI candidates;
- named-barrier pointer candidates;
- compiler crash versus normal rejection; and
- parser-stage versus codegen-stage failure.

The optimization-level HSACO fingerprint changes at 31 boundaries. Those
changes are retained as routing evidence for deeper qualification, not promoted
to configuration booleans.

## Interpretation work

A new structural result requires bounded review:

1. confirm the exact COMGR digest and public artifact;
2. compare candidate outcome and failed stage per processor;
3. determine whether the change affects a form rocKE emits or may select;
4. run complete production-kernel validity for any proposed selection change;
5. update Python and C++ policy together when emission changes; and
6. run generated-code/resource and numeric qualification where the microprobe
   cannot prove the required property.

The report removes the previous work of reconciling Clang, `opt`, and COMGR
identity. It does not remove qualification work and does not infer policy from
acceptance.

## Steady-state assessment

The current schema has nine independent capability families and 47–48 bounded
candidates per processor. The candidate count is driven by known current and
historical rocKE forms, not by LLVM's complete feature surface.

Schema growth is controlled by closure rules:

- add a candidate when rocKE emits a new form or archaeology exposes a real
  compatibility distinction;
- keep independently varying families separate;
- retain processor and failed-stage context;
- do not add general compiler-feature booleans unrelated to rocKE; and
- retire settled probes after their migration consumers and supported compiler
  window no longer need them.

## Decision

Direct COMGR probing is practical as the evidence layer for replacing automatic
LLVM-major profile inference. Production integration remains incremental:
evidence loading and normalized policy first, then one independently qualified
emission decision per reviewable slice. Flavor modes remain temporary offline
and fallback interfaces until every production consumer and artifact identity
has migrated.
