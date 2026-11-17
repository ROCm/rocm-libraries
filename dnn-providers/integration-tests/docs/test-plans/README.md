# Per-Op Test Plans

## Contents
- [Plans](#plans)
- [Tiers](#tiers)
- [Writing a plan](#writing-a-plan)

A per-op test plan lists what an op's bundles must exercise and how the sweep is spread across
levels. Plans follow [RFC 0022](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md); the bundles
implement them. A change that adds, moves or deletes bundle cases for an op updates that op's plan in
the same pull request. Ops with a plan are written by their generator; edit the plan and the generator,
not the bundles.

## Plans

| Plan | Bundles |
|---|---|
| [BatchnormInference](batchnorm-inference.md) | `BatchnormInference/Default`, `BatchnormInferencePointwise/Default`, written by `reference-data-scripts/generate_batchnorm_inference_bundles.py` |
| [SDPA backward](sdpa-bwd.md) | `SdpaBackward/<Topology>`, written by `reference-data-scripts/generate_sdpa_bwd_bundles.py` |

## Tiers

Every plan spreads its cases over the bundle tiers `quick`, `standard`, `comprehensive` and `full`.
See [Test tiers](../running-tests.md#test-tiers) and RFC 0022 §8.2.

## Writing a plan

Follow RFC 0022 §6 to §13 (structure, feature inventory, sweep construction, golden reference and
pass criteria). Start from an existing plan, add a row to the table above, and write a generator
that owns the op's bundle files.
