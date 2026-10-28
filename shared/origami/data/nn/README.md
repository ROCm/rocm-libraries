# Origami NN weight data

TWREC (tilewright), ESREC (embedding similarity) and ORIBOOST (boost, Origami Boost) model
weights live under this tree. Manifest YAML, weight sidecars and `.orr` models are **not tracked
in git**; an external fetch script will be added after the initial NN backend PRs land.

## Layout

```
data/nn/
  tilewright/<arch>/
    origami_nn_index              # tracked: logic_stem → manifest mapping
    *.tilewright.yaml             # fetched: TWREC manifest
    *.tilewright.wts.yaml         # fetched: learned weights sidecar
  embedding_similarity/<arch>/    # PR 2
    origami_nn_index
    *.embedding.yaml
    *.embedding.wts.yaml
  boost/<arch>/
    origami_nn_index              # tracked: logic_stem → model mapping
    origami_boost_<arch>_<dtype>.orr  # released: ORIBOOST model per dtype class
```

A hipBLASLt build writes one `origami_nn_index` per library directory: the indexes of the
backends it colocates, concatenated (boost always, tilewright with `ORIGAMI_ENABLE_NN=ON`).

## Local development

Point `ORIGAMI_NN_WEIGHTS_DIR` at a directory containing fetched weight YAML, or populate
`tilewright/gfx950/` with manifests and sidecars listed in `origami_nn_index`. For boost, copy the
released `.orr` files into `boost/gfx950/`.

Tests tagged `[nn][tilewright]` that require loaded weights skip automatically when manifests
are absent.
