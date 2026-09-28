# Integration Test Bundles

The test data for the cross-provider integration suite: graph JSON, template
sweeps, metadata and support-claim sidecars, and DVC pointers to golden tensors.
Every bundle here is discovered automatically and run against each provider's
engine.

```
{tier}/{Op}/{Layout}/{DataType}/{Name}/{Name}.json   single-graph bundle
{tier}/{Op}/{Topology}/graph.template.json + sweep.json   template-sweep bundle
```

- What each file is, and the rules the harness applies:
  [File Formats](../docs/file-formats.md)
- Adding bundles, golden data (`dvc pull` / `dvc push`), and support claims:
  [Adding Tests and Updating Claims](../docs/adding-tests.md)
- Running them: [Running the Tests](../docs/running-tests.md)

DVC commands run from the repository root. `.bin` tensor data is never committed.
