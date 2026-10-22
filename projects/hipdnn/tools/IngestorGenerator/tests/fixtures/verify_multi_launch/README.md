# `verify_multi_launch`

A generated multi-launch bundle for `tests/test_verify_variant_sets_multi_launch.py`:
one engine, two kernels, each carrying a `sources` map of two launches (`decode`, and
`merge` with `role: merge`) of one rocKE builder. `config.yaml` is its source;
`bundle/` is the descriptor output, committed as generated (ids included).

Regenerate from `projects/hipdnn/tools/IngestorGenerator/`:

```bash
python generate.py --config tests/fixtures/verify_multi_launch/config.yaml \
    --output-dir /tmp/verify_multi_launch
cp /tmp/verify_multi_launch/descriptors/rocke/verify_multi_launch/*.json \
    tests/fixtures/verify_multi_launch/bundle/
```
