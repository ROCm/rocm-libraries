# Integration Testing

Owner: @ROCm/hipdnn-reviewers | Last Updated: 2026-10-21

One shared test project builds a graph, runs it on a provider's engine, and checks the result against a reference. Every provider runs the same project, so a case added for one provider tests all of them. This page is the overview. The [integration test documentation](../../../../dnn-providers/integration-tests/docs/README.md) is the reference for flags, formats and authoring.

## What it is

The project lives in `dnn-providers/integration-tests/`. Its provider-facing binary is `hipdnn_integration_tests`. Each provider (MIOpen, hipBLASLt, hip-kernel) runs that binary against its own plugin:

```bash
hipdnn_integration_tests --test-article <plugin.so> --test-engine <ENGINE>
```

- **Engine-pinned lanes.** `--test-engine` pins a run to one engine. An op that engine does not support skips instead of falling through to another loaded engine.
- **Bundles.** A *bundle* is a graph stored as JSON plus a case matrix of shapes, datatypes and layouts. A bundle may also carry golden tensors. New graph-verification tests are bundles: adding a case is a data edit, not a recompile. Bundles live in `dnn-providers/integration-tests/integration-test-bundles/`.
- **Oracles.** An *oracle* is the reference a result is checked against. `--verification-mode` selects it:

  | Mode | Oracle |
  |---|---|
  | `auto` (default) | golden data, then GPU reference, then CPU reference; fails if none can verify |
  | `golden` | golden tensors, DVC-tracked |
  | `gpu` | GPU reference executor |
  | `cpu` | CPU reference executor |

  See [Verification modes](../../../../dnn-providers/integration-tests/docs/running-tests.md#verification-modes).
- **Support claims.** A bundle may carry a support sidecar (`<graph>.support.json`, or `support.json` for a sweep) that promises a named engine supports that graph on an arch and platform. A broken claim fails the test instead of skipping. See [support-claim-enforcement.md](../../../../dnn-providers/integration-tests/docs/support-claim-enforcement.md).
- **Tiers.** The tier of a bundle is its directory: `quick/`, `standard/`, `comprehensive/`, `full/`. A higher tier includes the lower ones. Tier meaning, budgets and where each tier runs in CI are in [TESTING.md](../TESTING.md#test-tiers).
- **Per-provider category filtering.** Each provider keeps a `test_categories_integration.yaml` (hip-kernel keeps one per engine, named `<ENGINE>_test_categories_integration.yaml`) that selects which cases run per tier for the shared binary. Lower tiers are scoped by bundle and name-prefix patterns. `full` is `*` for MIOpen, hipBLASLt and the HIP_MLOPS engine, and is limited to SDPA cases for the ASM_SDPA and gfx950 dense attention engines. A provider that implements a subset of ops relies on the harness's engine-support check, which skips ops the engine does not support. See [Test tiers](../../../../dnn-providers/integration-tests/docs/running-tests.md#test-tiers).
- **Golden data checks.** `hipdnn_golden_data_tests` recomputes golden tensors with the reference executors. It validates our data, not a provider.

## Adding a provider

A provider joins the shared project with four pieces. [Add an engine lane](../../../../dnn-providers/integration-tests/docs/adding-tests.md#add-an-engine-lane) has the full wiring.

| Piece | What it is |
|---|---|
| Plugin | The provider's engine plugin `.so`, passed as `--test-article`. |
| Test target | `add_external_integration_test_target()` in the provider's CMake takes `TARGET_NAME`, `PLUGIN_TARGET` and `ENGINE_NAME` (required), plus optional `TEST_CONFIG` and `TEST_CATEGORIES_YAML`. The MIOpen, hipBLASLt and hip-kernel providers name the target `<provider>-external-integration-check`. |
| `test_categories_integration.yaml` | Per-tier case selection for the shared binary. |
| `config/<ENGINE>.toml` | Per-engine tolerance overrides, validator overrides and `[[test_skips]]`. Skips are data, optionally scoped by `archs` and `platforms`, and each carries a `reason` shown in the skip message. |

To start, add the target, run `ctest -L '^quick$'`, and add `.support.json` sidecars for the graphs the engine promises to run. Every skip and tolerance override needs a tracked issue (see [Flaky tests and skips](TESTING_STRATEGY.md#flaky-tests-and-skips)).

### One shared set covers every provider

A provider-specific regression or unique graph-run case goes in the shared bundle set, not in a provider-local C++ test. Every other provider then runs it too. An engine that cannot run the case skips it, or fails its support claim if it promised to. Provider-local C++ tests are for behavior that is not "a graph runs and matches a reference", such as error paths or plugin-specific behavior.

## Today and direction

Today, bundles check that a graph runs on an engine and matches an oracle, with thin per-op depth. The direction is a unified gate that gives high confidence in the quality of a new integration or kernel source, built from detailed per-op coverage. See [Roadmap: Testing](../Roadmap.md#testing-1).
