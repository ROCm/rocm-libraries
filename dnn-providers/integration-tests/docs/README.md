# Integration Test Documentation

The cross-provider integration suite runs one set of graph tests against the
engine of every provider (MIOpen, hipBLASLt, hip-kernel, …). The graph tests
are the bundles under `integration-test-bundles/`. These documents help the
developer who must run the suite, read its output, or add to it.

| Document | Read it to… |
|---|---|
| [File Formats](file-formats.md) | Understand every file in the bundle tree and the provider configs: graphs, template sweeps, metadata, golden data and DVC pointers, support-claim sidecars, per-engine TOML, tier YAML. Also learn how test names are derived. |
| [Running the Tests](running-tests.md) | Run the suite (CTest, check targets, the binary directly), time the engine with `--benchmark`, learn what the tiers and filters select, and read the output: coverage summary, support-claim summary, hard stops, and runs that look green but are not. |
| [Adding Tests and Updating Claims](adding-tests.md) | Add a bundle (end to end, including a new op), add golden data, update support claims when a run reports `unclaimed_support`, record an engine limitation, add an engine lane, or add a C++ test when a bundle cannot express it. |
| [Support Claim Enforcement](support-claim-enforcement.md) | For harness maintainers: the claim verdict model, the lifecycle inside `TestBody()`, the coverage ladder, and who owns what in the harness. |
| [Support Matrix and Feature Coverage](support-matrix.md) | Render the human-readable matrix of which engine supports which graph on which GPU and OS, validate the sidecars it is built from, see the pre-commit and CI checks around it, and measure per-op feature coverage. |

Tool references are next to the tools:
[`migration-scripts/README.md`](../migration-scripts/README.md) (the bulk
C++-to-bundle pipeline, `find_case.py`) and
[`reference-data-scripts/README.md`](../reference-data-scripts/README.md)
(golden-data generators and the bundle verifier). Design rationale is in RFC
0006 (plugin-agnostic tests), RFC 0011 (bundle test framework) and RFC
0015 (engine support claims) under `projects/hipdnn/docs/rfcs/`.

## Terms

| Term | Meaning |
|---|---|
| **Bundle** | One test graph plus the files that describe it. It is either a **single-graph bundle** (`{Name}.json`) or a **sweep** (a `graph.template.json` expanded once per case in `sweep.json`). "Bundle" always includes both. |
| **Case** | One registered test: a single-graph bundle, or one entry of a sweep. |
| **Engine** | One implementation inside a provider plugin, named by `--test-engine` (`MIOPEN_ENGINE`, `HIP_MLOPS_ENGINE`, …). A plugin can contain more than one engine. |
| **Lane** | One provider engine's run of the suite: `hipdnn_integration_tests` configured for that plugin and engine, and registered in CTest for each tier. |
| **Test article** | The plugin library under test (`--test-article`). |
| **Ranked list** | The engines that hipDNN offers for a graph. An engine *accepts* a graph when the engine is on that list. |
| **Oracle** | What an engine's output is compared against: golden data, the GPU reference executor, or the CPU reference executor. |
| **Claim** | A checked-in promise, in a support-claim sidecar, that an engine accepts a graph on one arch and platform. |
| **Cell** | One (engine, arch, platform[, sweep case]) combination. It is the unit that a claim covers. |
| **Arch token** | The GPU architecture as a claim records it: the part of `gcnArchName` before the first `:` (`gfx942`). |
| **Enforcement level** | The point a run must reach for a claim to count as confirmed: `applicability` (accepted), `buildable` (plans compile) or `full` (outputs verified). Each bundle sets it in its metadata. |

## Signals a run prints, and what they ask of you

The rows are grouped by cost. Each row that fails the run turns CI red. The
other rows do not, so they are easy to miss.

| Output | Fails the run? | Meaning | Go to |
|---|---|---|---|
| `[  FAILED  ]` / `Failed: N` in the coverage summary | **yes** | A test failed. The output of the engine did not match its oracle, or another failure occurred. | [Test coverage summary](running-tests.md#test-coverage-summary) |
| `claim_failures` → `CLAIM_BROKEN` | **yes** (when enforcing) | An engine stopped accepting a graph that a claim says it supports. | [Support-claim summary](running-tests.md#support-claim-summary); retract the claim only if the drop is intended: [Retract a claim](adding-tests.md#retract-a-claim) |
| `claim_failures` → `QUERY_ERRORED` | **yes** (when enforcing) | The support query failed, so acceptance is unknown. Investigate. Do not retract the claim for this signal. | [Support-claim summary](running-tests.md#support-claim-summary) |
| `failed_in_use` | **yes** (the test failed) | A claimed cell. The engine accepts the graph but gives a wrong result. | [Support-claim summary](running-tests.md#support-claim-summary) |
| `UNVERIFIABLE BUNDLES` / `Unverifiable: … tried: …` | **yes** (those tests failed) | The engine ran the bundle, but no oracle could check its output: no golden data and no reference executor for the op. Nothing was tested. | [Unverifiable bundles](running-tests.md#unverifiable-bundles); fix: [When no reference executor supports the op](adding-tests.md#when-no-reference-executor-supports-the-op) |
| `Error: zero tests ran.` | **yes** | A discovery or filter misconfiguration, or, with GTest sharding, an empty shard. | [Hard stops](running-tests.md#hard-stops) |
| `FATAL: --enforce-support-claims is active and …` | **yes** | Enforcement verified nothing. | [Hard stops](running-tests.md#hard-stops) |
| `benchmark failed: …` | **yes** (that test failed) | Under `--benchmark`, the verified graph failed when it ran again, or the timer returned too few measurements. | [Time the engine](running-tests.md#4-time-the-engine---benchmark) |
| `unclaimed_support` in the `SUPPORT CLAIM SUMMARY` | no | The engine accepts graphs that no sidecar claims. Record them. | [Updating support claims](adding-tests.md#updating-support-claims) |
| `Skipped:` equal to (or close to) the total | no | Green, but little or nothing was tested. Read the skip reasons. | [Test coverage summary](running-tests.md#test-coverage-summary) |
| `WARNING: Bundle tests are enabled but …` / `WARNING: No bundles could be loaded from …` | no | No bundles registered. Only compiled-in C++ tests ran. | [Troubleshooting](running-tests.md#troubleshooting) |
| `No tests were found!!!` from CTest | no (exit 0) | Wrong directory, wrong label, or only disabled suites. | [Runs that look green but are not](running-tests.md#runs-that-look-green-but-are-not) |
| `No HIP devices available; skipping …` | no (exit 0) | There is no GPU, so nothing ran. | [Runs that look green but are not](running-tests.md#runs-that-look-green-but-are-not) |
| `REFERENCE EXECUTOR ERRORS` | sometimes | A reference executor threw. The test fails if this was the last oracle to try. Otherwise, the run fell back. | [Unverifiable bundles](running-tests.md#unverifiable-bundles) |
| `verdicts` all zero while `unenforced.no_applicable_claim` > 0 | no | The harness read the sidecars, but they claim nothing for this arch and platform. Nothing was enforced here. | [Updating support claims](adding-tests.md#updating-support-claims) |
| `WARNING ONLY -- NOT ENFORCED` summary header | no | Enforcement was off (`--enforce-support-claims=false`). Broken claims were reported but did not fail the run. | [Command-line reference](running-tests.md#command-line-reference--hipdnn_integration_tests) |
| `SUPPORT CLAIM WRITE SUMMARY` | if it reports errors | An authoring run. The run could have changed sidecars in the source tree. | [Record claims](adding-tests.md#record-claims-with---write-support-claims) |

A green CTest run with `--output-on-failure` does not print the per-test
output above. See [Seeing the output of a passing run](running-tests.md#seeing-the-output-of-a-passing-run).

## Maintaining these documents

These documents are the single source of truth for how the suite works. The
`hipdnn-integration-testing` AI skill (under `projects/hipdnn/tools/ai/skills/`)
reads this index and the linked documents at run time. The skill has no
knowledge of its own. A change here changes what the skill knows.

Do not change file names. Do not change a heading that has a link to it, from
this index, from the other documents, or from elsewhere in the repository. For
example, RFC 0011 links `running-tests.md#test-tiers` and
`#how-tiers-cascade`. Add each new document to the table at the top.
