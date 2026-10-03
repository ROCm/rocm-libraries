# Integration Test Documentation

The cross-provider integration suite runs one set of graph tests — the bundles
under `integration-test-bundles/` — against every provider's engine (MIOpen,
hipBLASLt, hip-kernel, …). These documents are for the developer who has to run
the suite, read its output, or add to it.

| Document | Read it to… |
|---|---|
| [File Formats](file-formats.md) | Understand every file in the bundle tree and the provider configs: graphs, template sweeps, metadata, golden data and DVC pointers, support-claim sidecars, per-engine TOML, tier YAML — and how test names are derived. |
| [Running the Tests](running-tests.md) | Run the suite (CTest, check targets, the binary directly), know what the tiers and filters select, and read the output: coverage summary, support-claim summary, hard stops, and runs that look green but are not. |
| [Adding Tests and Updating Claims](adding-tests.md) | Add a bundle (end to end, including a new op), add golden data, update support claims when a run reports `unclaimed_support`, record an engine limitation, add an engine lane, or add a C++ test when a bundle cannot express it. |
| [Support Claim Enforcement](support-claim-enforcement.md) | For harness maintainers: the claim verdict model, the lifecycle inside `TestBody()`, the coverage ladder, and who owns what in the harness. |

Tool references live next to the tools:
[`migration-scripts/README.md`](../migration-scripts/README.md) (the bulk
C++-to-bundle pipeline, `find_case.py`) and
[`reference-data-scripts/README.md`](../reference-data-scripts/README.md)
(golden-data generators and the bundle verifier). Design rationale is in RFC
0006 (plugin-agnostic tests), RFC 0011 (golden reference validation) and RFC
0015 (engine support claims) under `projects/hipdnn/docs/rfcs/`.

## Terms

| Term | Meaning |
|---|---|
| **Bundle** | One test graph plus the files that describe it. Either a **single-graph bundle** (`{Name}.json`) or a **sweep** (a `graph.template.json` expanded once per case in `sweep.json`). "Bundle" always covers both. |
| **Case** | One registered test: a single-graph bundle, or one entry of a sweep. |
| **Engine** | One implementation inside a provider plugin, named by `--test-engine` (`MIOPEN_ENGINE`, `HIP_MLOPS_ENGINE`, …). A plugin can hold several. |
| **Lane** | One provider engine's run of the suite: `hipdnn_integration_tests` pointed at that plugin and engine, registered in CTest per tier. |
| **Test article** | The plugin library under test (`--test-article`). |
| **Ranked list** | The engines hipDNN offers for a graph. An engine *accepts* a graph when it is on that list. |
| **Oracle** | What an engine's output is compared against: golden data, the GPU reference executor, or the CPU reference executor. |
| **Claim** | A checked-in promise, in a support-claim sidecar, that an engine accepts a graph on one arch and platform. |
| **Cell** | One (engine, arch, platform[, sweep case]) combination — the unit a claim covers. |
| **Arch token** | The GPU architecture as a claim records it: the part of `gcnArchName` before the first `:` (`gfx942`). |
| **Enforcement level** | How far a run must get for a claim to count as confirmed: `applicability` (accepted), `buildable` (plans compile) or `full` (outputs verified). Set per bundle in metadata. |

## Signals a run prints, and what they ask of you

Grouped by what they cost you. Anything under **Fails the run** turns CI red;
the rest do not, and are easy to miss for exactly that reason.

| Output | Fails the run? | Meaning | Go to |
|---|---|---|---|
| `[  FAILED  ]` / `Failed: N` in the coverage summary | **yes** | A test failed: the engine's output did not match its oracle, or something else broke. | [Test coverage summary](running-tests.md#test-coverage-summary) |
| `claim_failures` → `CLAIM_BROKEN` | **yes** (when enforcing) | An engine stopped accepting a graph it is claimed to support. | [Support-claim summary](running-tests.md#support-claim-summary); retract only if the drop is intended: [Retract a claim](adding-tests.md#retract-a-claim) |
| `claim_failures` → `QUERY_ERRORED` | **yes** (when enforcing) | The support query itself failed, so acceptance is unknown. Investigate; never retract over this. | [Support-claim summary](running-tests.md#support-claim-summary) |
| `failed_in_use` | **yes** (the test failed) | A claimed cell: the engine accepts the graph but gets it wrong. | [Support-claim summary](running-tests.md#support-claim-summary) |
| `Error: zero tests ran.` | **yes** | Discovery or filter misconfiguration. | [Hard stops](running-tests.md#hard-stops) |
| `FATAL: --enforce-support-claims is active and …` | **yes** | Enforcement verified nothing. | [Hard stops](running-tests.md#hard-stops) |
| `unclaimed_support` in the `SUPPORT CLAIM SUMMARY` | no | The engine accepts graphs that no sidecar claims. Record them. | [Updating support claims](adding-tests.md#updating-support-claims) |
| `Skipped:` equal to (or close to) the total | no | Green, but little or nothing was tested. Read the skip reasons. | [Test coverage summary](running-tests.md#test-coverage-summary) |
| `WARNING: Bundle tests are enabled but …` / `WARNING: No bundles could be loaded from …` | no | No bundles registered; only compiled-in C++ tests ran. | [Troubleshooting](running-tests.md#troubleshooting) |
| `No tests were found!!!` from CTest | no (exit 0) | Wrong directory, wrong label, or only disabled suites. | [Runs that look green but are not](running-tests.md#runs-that-look-green-but-are-not) |
| `No HIP devices available; skipping …` | no (exit 0) | No GPU: nothing ran. | [Runs that look green but are not](running-tests.md#runs-that-look-green-but-are-not) |
| `UNVERIFIABLE BUNDLES` | no | Bundles the engine ran with no working oracle; their output was not checked. | [Unverifiable bundles](running-tests.md#unverifiable-bundles) |
| `REFERENCE EXECUTOR ERRORS` | sometimes | A reference executor threw. Fails the test if it was the last oracle to try; otherwise the run fell back. | [Unverifiable bundles](running-tests.md#unverifiable-bundles) |
| `verdicts` all zero while `unenforced.no_applicable_claim` > 0 | no | Sidecars were read but claim nothing for this arch and platform: nothing was enforced here. | [Updating support claims](adding-tests.md#updating-support-claims) |
| `WARNING ONLY -- NOT ENFORCED` summary header | no | Enforcement was turned off (`--enforce-support-claims=false`); broken claims were reported, not failed. | [Command-line reference](running-tests.md#command-line-reference--hipdnn_integration_tests) |
| `SUPPORT CLAIM WRITE SUMMARY` | if it reports errors | An authoring run; sidecars in the source tree may have changed. | [Record claims](adding-tests.md#record-claims-with---write-support-claims) |

A green CTest run with `--output-on-failure` prints none of the per-test
output above; see [Seeing the output of a passing run](running-tests.md#seeing-the-output-of-a-passing-run).

## Maintaining these documents

These documents are the single source of truth for how the suite works. The
`hipdnn-integration-testing` AI skill (under `projects/hipdnn/tools/ai/skills/`)
reads this index and the documents it links at run time and carries no
knowledge of its own, so a change here is a change to what the skill knows.

Keep file names stable, and keep stable any heading that is linked — from this
index, from the other documents, or from elsewhere in the repository (RFC 0011
links `running-tests.md#test-tiers` and `#how-tiers-cascade`). Add any new
document to the table at the top.
