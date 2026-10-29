# Origami NN model data

Model files of the `origami::nn` backends live under this tree, one directory per backend and
architecture, each with an `origami_nn_index`:

```
data/nn/
  <backend>/<arch>/
    origami_nn_index              # logic_stem -> model file
    ...                           # the backend's model files
```

Index lines are tab-separated (`#` starts a comment):

```
<logic_stem>  <backend>  <model file>
```

`<logic_stem>` is a hipBLASLt logic file's name without its extensions
(`TensileLibrary_<type>_..._<arch>`) and `<backend>` the backend's name in
`origami::nn::backend_id_t` (`boost`; `tilewright` and `embedding_similarity` are reserved).

hipBLASLt copies every `<backend>/<arch>/` directory next to the `<arch>` kernel library
(`projects/hipblaslt/cmake/OrigamiNNWeights.cmake`, target `origami-nn-colocate-<arch>`), merging
the backends' indexes into one `origami_nn_index`, and installs them with the library.
`origami::nn::load_model_by_index(logic_stem, backend, library_dir)` reads that index;
`ORIGAMI_NN_WEIGHTS_DIR` points it at another directory when no directory is given.

## Origami Boost (`boost/`)

```
data/nn/boost/<arch>/
  origami_nn_index                  # TensileLibrary_<BB|SS|F8F8>_..._<arch> -> model of its dtype class
  origami_boost_<arch>_<dtype>.orr  # ORIBOOST model per dtype class (bf16, tf32, fp8)
```

The models are tracked (2.1–2.5 MB each; `*.orr binary` in `shared/origami/.gitattributes`). A
boost model loads only through `origami::nn::load_boost_model`, which checks it against the build
and the kernel library it serves.

Each model is stamped with the Origami sources it was trained against, and a build whose sources
differ refuses it. `origami-tests "[shipped]"` fails when that would happen and says whether to
re-stamp (`shared/origami/tools/origami_boost_restamp.py`) or retrain; see "Shipped Origami Boost
models" in `shared/origami/README.md`. Retraining happens outside this repository (OrigamiML).
