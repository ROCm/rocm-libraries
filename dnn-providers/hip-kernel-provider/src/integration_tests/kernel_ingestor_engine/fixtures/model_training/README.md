# Inputs for the packaged fixtures' trained heuristics

Kept out of `packaged/` on purpose. Every non-descriptor file beside a
`kind: "model"` UHD is carried into the packed tree with it (see
`_model_artifacts` in `hkp_pack/descriptors.py`), so anything left in that folder
ships to a customer. Training inputs are provenance, not payload.

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
