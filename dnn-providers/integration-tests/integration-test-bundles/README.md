# Integration Test Bundles

This directory holds the test data for the cross-provider integration suite:
graph JSON, template sweeps, metadata and support-claim sidecars, and DVC
pointers to golden tensors. The harness discovers every bundle here
automatically and runs it against the engine of each provider.

```
{tier}/{Op}/{Layout}/{DataType}/{Name}/{Name}.json   single-graph bundle
{tier}/{Op}/{Topology}/graph.template.json + sweep.json   template-sweep bundle
```

- What each file is, and the rules the harness applies:
  [File Formats](../docs/file-formats.md)
- Adding bundles, golden data (`dvc pull` / `dvc push`), and support claims:
  [Adding Tests and Updating Claims](../docs/adding-tests.md)
- Running them: [Running the Tests](../docs/running-tests.md)

Run DVC commands from the repository root. Never commit `.bin` tensor data.
