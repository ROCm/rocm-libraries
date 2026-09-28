# Integration Tests

Integration tests validate hipDNN provider plugins (engine libraries such as
`libmiopen_plugin.so` or `libhipblaslt_plugin.so`) by building a graph, running
it through the plugin's engine, and comparing the result against a reference.

This directory builds one binary, `hipdnn_integration_tests`. Each provider
(miopen-provider, hipblaslt-provider, hip-kernel-provider, …) runs it against
its own plugin, so **one set of graph tests runs against every engine**.

## Documentation

| Document | Covers |
|---|---|
| [File Formats](docs/file-formats.md) | Bundles, template sweeps, metadata, golden data and DVC pointers, support-claim sidecars, per-engine TOML, tier YAML, test naming |
| [Running the Tests](docs/running-tests.md) | CTest lanes, check targets, the binary's flags, tiers and filtering, verification modes, reading the output, CI |
| [Adding Tests and Updating Claims](docs/adding-tests.md) | Adding bundles, golden data with DVC, updating support claims, recording engine limitations, C++ and reference-executor tests |
| [Support Claim Enforcement](docs/support-claim-enforcement.md) | The claim verdict model and harness lifecycle in depth |

Start at [`docs/README.md`](docs/README.md), which also lists what each
message a run prints asks of you.

## Quick start

```bash
# Superbuild: run one provider's quick tier from its build directory
cd build/dnn-providers/miopen-provider
ctest -L quick --output-on-failure

# Or run the binary against one engine
./bin/hipdnn_integration_tests \
    --test-article /path/to/libmiopen_plugin.so \
    --test-engine  MIOPEN_ENGINE \
    --gtest_filter='quick_*'
```

Read the `TEST COVERAGE SUMMARY` and `SUPPORT CLAIM SUMMARY` it prints, not just
the exit code — see [Reading the output](docs/running-tests.md#reading-the-output).

## Two ways to test a graph

| | **Bundles + sweeps** (default) | **C++ integration tests** (special cases) |
|---|---|---|
| What it is | Graph JSON + a case matrix of shapes/dtypes/layouts | `buildGraph()` + `INSTANTIATE_TEST_SUITE_P` |
| Add a case | Run a tool — no compile | Write C++, recompile |
| Built by default | yes | only files listed in `HIPDNN_IT_ALWAYS_BUILT_SOURCES`; graph tests need `-DBUILD_CPP_GRAPH_TESTS=ON` |
| Use for | "does this graph run and match a reference on this engine" | anything else: error paths, API contracts, serialization, determinism |

Bundles are the CI driver. New graph-verification coverage must be a bundle;
see [Adding Tests](docs/adding-tests.md).

## Layout

```
integration-test-bundles/   the bundle tree: {tier}/{Op}/... graphs, sweeps, sidecars
docs/                       the documentation above
src/                        the harness and the binaries' entry points
src/integration-tests/      C++ integration tests (mostly opt-in)
tests/                      reference-executor tests and harness unit tests
cmake/                      add_external_integration_test_target() for providers
migration-scripts/          capture/import tooling (import_graph.py, find_case.py, ...)
reference-data-scripts/     golden-data generators and verify_golden_bundles.py
scripts/                    verify_support_claims.py (pre-commit)
```

## See also

- [RFC 0006 — Plugin-Agnostic Integration Tests](../../projects/hipdnn/docs/rfcs/0006_PluginAgnosticIntegrationTests.md)
- [RFC 0011 — Golden Reference Validation](../../projects/hipdnn/docs/rfcs/0011_GoldenReferenceValidation.md)
- [RFC 0015 — Engine Support Claims](../../projects/hipdnn/docs/rfcs/0015_EngineSupportClaims.md)

## Project policies

This suite is part of the hipDNN project. Shared project documentation and
policies are maintained in hipDNN:

- [hipDNN Overview](../../projects/hipdnn/README.md)
- [Contributing Guidelines](../../projects/hipdnn/CONTRIBUTING.md)
- [Security Policy](../../projects/hipdnn/SECURITY.md)
