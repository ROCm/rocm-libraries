# Per-Op Test Plans

Each plan in this folder covers one operation or fusion family. It states what
must be tested at each tier, how results are judged, and which products are
deliberately not run and why. Every plan follows
[RFC 0022](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md) and
describes its own method. The bundles under
[`integration-test-bundles/`](../../integration-test-bundles/) implement the
plan:

- every plan item maps to a bundle case, or the plan lists it as not yet
  runnable;
- every bundle case maps back to an item in the plan.

A change that adds, moves or deletes bundle cases for an op updates that op's
plan in the same pull request. Ops with a plan are written by their generator;
edit the plan and the generator, not the bundles.

| Plan | Bundles |
|---|---|
| [Convolution forward](convolution-fwd.md) | `ConvolutionFwd/Default`, `ConvolutionFwdPointwise/Default`, `ConvolutionFwdPointwise/Variant2` … `Variant6` (written by `reference-data-scripts/generate_conv_fwd_bundles.py`) |
| [SDPA backward](sdpa-bwd.md) | `SdpaBackward/<Topology>` (written by `reference-data-scripts/generate_sdpa_bwd_bundles.py`) |

The generators are described in the
[reference data scripts README](../../reference-data-scripts/README.md).

## Tiers

The plans call the four tiers quick, PR (`standard/`), nightly
(`comprehensive/`) and full. Their bundle directories, CTest labels and
cascade are in [Test tiers](../running-tests.md#test-tiers); each tier's
question, cadence and budget are in
[RFC 0022 §8.2](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#82-the-four-levels).
Long-cadence (release-train) testing is out of scope; release readiness covers
it.

## Writing a plan

Follow RFC 0022. Its sections cover:

- the feature inventory
  ([§6](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#6-feature-inventory));
- the shape catalog and size ladder
  ([§7](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#7-shape-catalog-and-size-ladder));
- the sweep rules and levels
  ([§8](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#8-sweep-construction-levels-and-lanes));
- case purpose
  ([§9](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#9-case-purpose));
- the trade-off register
  ([§10](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#10-trade-off-register));
- oracles and reference prerequisites
  ([§11](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#11-oracles-and-reference-prerequisites));
- golden tests
  ([§12](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#12-golden-tests));
- the authoring process, the plan template and the review checklist
  ([§13](../../../../projects/hipdnn/docs/rfcs/0022_PerOpTestPlans.md#13-authoring-process-and-plan-template)).

A plan states the method it used for its own sweep. The convolution plan does
so in its last section.
