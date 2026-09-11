# Committed `uhd_gen` output

The artifacts a real `uhd_gen train` run produced, checked in so
`TestUhdGeneratedModel.cpp` can load them without a Python toolchain in the
loop.

| File | What it is |
|---|---|
| `tile_selector.uhd.json` | the UHD: `features_signature`, objective, score units, and the artifact it names |
| `model.bin` | the GBDT artifact |
| `train_manifest.json` | provenance, including the `features_hash` both sides must agree on |
| `training_data.csv` | the input, so the model is reproducible rather than magic |

## What it models

Two features, `$kernel.tile_m` and `$q.seqlen`. Sequences of 1024 and longer
prefer the 128 tile; shorter ones prefer 64. The crossover is the point: a
heuristic that only ever picked the same kernel would be indistinguishable from
no heuristic at all, and a static ordering would satisfy any test that did not
change the problem underneath it.

## Regenerating

```bash
cd projects/hipdnn/tools
python -m uhd_gen train \
    --input <this dir>/training_data.csv \
    --features kernel.tile_m q.seqlen \
    --descriptor-tree <this dir>/descriptors \
    --engine test:generated_features \
    --target tflops \
    --output-dir <this dir> \
    --name "Tile Selector UHD" \
    --descriptor-name tile_selector \
    --training-arches gfx942 \
    --model-version 1.0.0
```

The `id` in `tile_selector.uhd.json` is a fresh UUID on every run, and the model
carries a training date, so a regeneration produces different bytes for the same
inputs. That is expected; nothing asserts on either.

The fixture descriptor tree records the engine and metadata revision used for training.
Both the model and its provenance are regenerated together; do not copy provenance into
an older artifact after training.

The categorical fixture is regenerated with the same command, replacing the input/output
directory with `<this dir>/dtype_selector`, the features with
`kernel.dtype kernel.tile_m q.seqlen`, and the name/descriptor name with
`"Dtype Selector UHD"`/`dtype_selector`. Its descriptor tree remains
`<this dir>/descriptors`.

## Why these are committed rather than built

The point of the test that reads them is that the *tool's own output* loads:
that the descriptor the Python tool emits is the one `DescriptorLoader` parses,
and that the `features_hash` Python computed is the one the C++
`FeatureExtractor` recomputes. Rebuilding them from the C++ side under test would assert nothing.
