# Origami NN weight data

TWREC (tilewright) and ESREC (embedding similarity) model weights live under this tree.
Manifest YAML and weight sidecars are **not tracked in git**; an external fetch script will
be added after the initial NN backend PRs land.

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
```

## Local development

Point `ORIGAMI_NN_WEIGHTS_DIR` at a directory containing fetched weight YAML, or populate
`tilewright/gfx950/` with manifests and sidecars listed in `origami_nn_index`.

Tests tagged `[nn][tilewright]` that require loaded weights skip automatically when manifests
are absent.
