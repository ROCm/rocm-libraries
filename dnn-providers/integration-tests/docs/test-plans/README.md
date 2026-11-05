# Per-Op Test Plans

Each plan in this folder covers one operation or fusion family. It states what
must be tested at each tier, how results are judged, and which products are
deliberately not run and why. Every plan follows the
[methodology](methodology.md), which also holds the plan template, a review
checklist and the lessons learned so far. The bundles under
[`integration-test-bundles/`](../../integration-test-bundles/) implement the
plan:

- every plan item maps to at least one bundle case;
- every bundle case maps back to an item in the plan.

A change that adds, moves or deletes bundle cases for an op updates that op's
plan in the same pull request.

| Plan | Bundles |
|---|---|
| [Convolution forward](convolution-fwd.md) | `ConvolutionFwd/Default`, `ConvolutionFwdPointwise/Default`, `ConvolutionFwdPointwise/Variant2` … `Variant6` (written by `reference-data-scripts/generate_conv_fwd_bundles.py`) |

## Tiers

The tiers match the bundle directories and the CTest labels in each provider's
`test_categories_integration.yaml`. They cascade: each tier also runs every
tier below it. The one exception is `full_*`, which is run only by `full`.

| Tier | Bundle dir | CI cadence | Budget (whole suite, cumulative) |
|---|---|---|---|
| quick | `quick/` | pre-commit | ~5 min |
| PR | `standard/` | PR gate | ~30 min |
| nightly | `comprehensive/` | nightly | ≤ 60 min |
| full | `full/` | weekly | none |

Long-cadence (release-train) testing is out of scope; release readiness covers it.

## Writing a plan

Follow [methodology.md](methodology.md). It covers:

- how to derive the feature inventory from the API;
- how to audit the existing bundles;
- the S / L / M / X shape sets and per-tier size ceilings;
- the combination rules;
- the trade-off register;
- reference prerequisites and the golden baseline;
- the plan document template.
