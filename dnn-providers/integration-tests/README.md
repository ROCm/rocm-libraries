# Integration Tests

Integration tests validate hipDNN provider plugins (engine libraries such as
`libmiopen_plugin.so` or `libhipblaslt_plugin.so`) by building a graph, running
it through the plugin's engine, and comparing the result against a reference.

This directory builds one binary, `hipdnn_integration_tests`. Each provider
(miopen-provider, hipblaslt-provider, hip-kernel-provider, …) runs it against
its own plugin, so **one set of graph tests runs against every engine**. The
tests are data — bundles under `integration-test-bundles/` — rather than code;
C++ tests remain only for what a bundle cannot express.

**Documentation: start at [`docs/README.md`](docs/README.md).** It covers the
file formats, running the tests and reading their output, and adding tests and
support claims, and lists what each message a run prints asks of you.

## Quick start

From the repository root of a superbuild built in `build/`:

```bash
# One provider's quick tier
ctest --test-dir build/dnn-providers/miopen-provider -L quick -V

# Or the whole lane for one engine
cmake --build build --target miopen-provider-external-integration-check
```

Read the `TEST COVERAGE SUMMARY` and `SUPPORT CLAIM SUMMARY` it prints, not just
the exit code — see [Reading the output](docs/running-tests.md#reading-the-output).

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
scripts/                    verify_support_claims.py
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
