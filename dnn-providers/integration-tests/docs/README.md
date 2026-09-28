# Integration Test Documentation

The cross-provider integration suite runs one set of graph tests — bundles and
sweeps under `integration-test-bundles/` — against every provider's engine
(MIOpen, hipBLASLt, hip-kernel, …). These documents describe it for the
developer who has to run it, read it, or add to it.

| Document | Read it to… |
|---|---|
| [File Formats](file-formats.md) | Understand every file in the bundle tree and the provider configs: graph JSON, template sweeps, metadata, golden data and DVC pointers, support-claim sidecars, per-engine TOML, tier YAML — and how test names are derived. |
| [Running the Tests](running-tests.md) | Run the suite (CTest, check targets, the binary directly), know what the tiers and filters select, and read the output: coverage summary, support-claim summary, hard stops, and runs that look green but are not. |
| [Adding Tests and Updating Claims](adding-tests.md) | Add a bundle, add or update golden data, update support claims when a run reports `unclaimed_support`, record an engine limitation, or add a C++ test when a bundle cannot express it. |
| [Support Claim Enforcement](support-claim-enforcement.md) | Go deeper on claims: the verdict model, the harness lifecycle inside `TestBody()`, every summary key, and who owns what in the harness. |

Tool-specific references live next to the tools:
[`migration-scripts/README.md`](../migration-scripts/README.md) (capture,
import, and the bulk C++-to-bundle pipeline) and
[`reference-data-scripts/README.md`](../reference-data-scripts/README.md)
(golden-data generators and the bundle verifier). Design rationale is in RFC
0006 (plugin-agnostic tests), RFC 0011 (golden reference validation) and RFC
0015 (engine support claims) under `projects/hipdnn/docs/rfcs/`.

## Signals a run prints, and what they ask of you

| When the output shows… | It means | Go to |
|---|---|---|
| `SUPPORT CLAIM SUMMARY` with entries under `unclaimed_support` | The engine accepts graphs that no sidecar claims. Record them. | [Updating support claims](adding-tests.md#updating-support-claims) |
| `claim_failures` with `CLAIM_BROKEN` or `QUERY_ERRORED` | An engine stopped accepting a graph it is claimed to support. The run fails. | [Support-claim summary](running-tests.md#support-claim-summary), then [Retract a claim](adding-tests.md#retract-a-claim) only if the drop is intended |
| `failed_in_use` entries | The engine accepts the graph but the test fails. Do not claim that cell. | [Support-claim summary](running-tests.md#support-claim-summary) |
| `no_applicable_claim` equal to every claim-bearing graph | Nothing was enforced on this arch/platform. | [Updating support claims](adding-tests.md#updating-support-claims) |
| `WARNING ONLY -- NOT ENFORCED` in the summary header | Broken claims were reported, not failed (no `--test-engine`, or enforcement turned off). | [Command-line reference](running-tests.md#command-line-reference--hipdnn_integration_tests) |
| `Skipped:` equal to (or close to) the total | Green, but little or nothing was tested. Read the skip reasons. | [Test coverage summary](running-tests.md#test-coverage-summary) |
| `Error: zero tests ran.` | Discovery or filter misconfiguration. | [Hard stops](running-tests.md#hard-stops) |
| `No tests were found!!!` from CTest | Wrong directory, wrong label, or only disabled suites — exit 0 anyway. | [Runs that look green but are not](running-tests.md#runs-that-look-green-but-are-not) |
| `FATAL: --enforce-support-claims is active and … not one of them was ever queried` | Enforcement verified nothing. | [Hard stops](running-tests.md#hard-stops) |
| `UNVERIFIABLE BUNDLES` / `REFERENCE EXECUTOR ERRORS` | Bundles ran with no working oracle; their output was not checked. | [Unverifiable bundles](running-tests.md#unverifiable-bundles) |
| `SUPPORT CLAIM WRITE SUMMARY` | An authoring run; sidecars in the source tree may have changed. | [Record claims](adding-tests.md#record-claims-with---write-support-claims) |

## Maintaining these documents

These files are the single source of truth for how the suite works. The
`hipdnn-integration-testing` AI skill (`projects/hipdnn/tools/ai/skills/`)
carries no knowledge of its own: it reads this index and the documents it
links, from the checkout or from GitHub. To change what developers — and the
skill — are told, change these files. Keep the file names and section headings
linked above stable, and add any new document to the table at the top.
