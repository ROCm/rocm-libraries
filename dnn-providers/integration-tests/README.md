# Integration Tests

Integration tests validate hipDNN provider plugins (engine libraries such as
`libmiopen_plugin.so` or `libhipblaslt_plugin.so`). Each test builds a graph,
runs it through the engine of the plugin, and compares the result against a
reference.

This directory builds one binary, `hipdnn_integration_tests`. Each provider
(miopen-provider, hipblaslt-provider, hip-kernel-provider, …) runs it against
its own plugin, so **one set of graph tests runs against every engine**. The
tests are data (bundles under `integration-test-bundles/`), not code. C++ tests
exist only for what a bundle cannot express.

**Documentation: start at [`docs/README.md`](docs/README.md).** It describes
the file formats, how to run the tests and read their output, and how to add
tests and support claims. It also lists what each message from a run asks you
to do.

## Quick start

Run these commands from the repository root of a superbuild built in `build/`:

```bash
# One provider's quick tier
ctest --test-dir build/dnn-providers/miopen-provider -L quick -V

# Or the whole lane for one engine
cmake --build build --target miopen-provider-external-integration-check
```

Read the `TEST COVERAGE SUMMARY` and `SUPPORT CLAIM SUMMARY` that the run
prints. Do not read only the exit code. See
[Reading the output](docs/running-tests.md#reading-the-output).

## Layout

```
integration-test-bundles/   the bundle tree: {tier}/{Op}/... graphs, sweeps, sidecars
docs/                       the documentation
src/                        the harness and the binaries' entry points
src/integration-tests/      C++ integration tests (mostly opt-in)
tests/                      harness unit tests; tests/gpu-ref/ reference-executor tests
cmake/                      add_external_integration_test_target() for providers
migration-scripts/          capture/import tooling (import_graph.py, find_case.py, ...)
reference-data-scripts/     golden-data generators and verify_golden_bundles.py
scripts/                    verify_support_claims.py, harvest_support_claims.py (+ support_sidecar.py)
```

## See also

- [RFC 0006 — Plugin-Agnostic Integration Tests](../../projects/hipdnn/docs/rfcs/0006_PluginAgnosticIntegrationTests.md)
- [RFC 0011 — Bundle Test Framework](../../projects/hipdnn/docs/rfcs/0011_BundleTestFramework.md)
- [RFC 0015 — Engine Support Claims](../../projects/hipdnn/docs/rfcs/0015_EngineSupportClaims.md)

## Project policies

This suite is part of the hipDNN project. Shared project documentation and
policies are maintained in hipDNN:

- [hipDNN Overview](../../projects/hipdnn/README.md)
- [Contributing Guidelines](../../projects/hipdnn/CONTRIBUTING.md)
- [Security Policy](../../projects/hipdnn/SECURITY.md)
