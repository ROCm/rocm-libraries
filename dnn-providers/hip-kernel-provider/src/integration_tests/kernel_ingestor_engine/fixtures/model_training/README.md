# Inputs for the packaged fixtures' trained heuristics

Kept out of `packaged/` on purpose. A packed tree should contain what ships, and
training inputs are provenance, not payload. The packer no longer over-carries
them -- it stages exactly the file `tree_data.artifact` names -- but a source
tree that mixes the two still invites the mistake.

## `pointwise_model_training_data.csv`

Produces `packaged/pointwise_model/`'s heuristic. Two rows' worth of signal: the
64 tile is fast, the 256 tile is slow. Deliberately the inverse of
`hipkernel.pointwise.score`, which returns `block_size` and so always prefers
256 -- the two engines rank the same catalog in opposite orders, which is what
makes the integration assertion mean something.

```bash
cd projects/hipdnn/tools
python -m uhd_gen train \
    --input <this dir>/pointwise_model_training_data.csv \
    --features kernel.block_size \
    --target tflops \
    --output-dir <repo>/dnn-providers/hip-kernel-provider/src/integration_tests/kernel_ingestor_engine/fixtures/packaged/pointwise_model \
    --name "packed pointwise model selector" \
    --descriptor-name packed_pointwise_model \
    --num-boost-round 30
```

Regenerating mints a new UHD id, so `packed_pointwise_model.ued.json`'s
`heuristic` field has to be updated to match.
