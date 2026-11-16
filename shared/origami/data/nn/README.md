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
`origami::nn::backend_id_t` (`tilewright`, `embedding_similarity`, `param`).

hipBLASLt copies every `<backend>/<arch>/` directory next to the `<arch>` kernel library
(`projects/hipblaslt/cmake/OrigamiNNWeights.cmake`, target `origami-nn-colocate-<arch>`), merging
the backends' indexes into one `origami_nn_index`, and installs them with the library.
`origami::nn::load_model_by_index(logic_stem, backend, library_dir)` reads that index;
`ORIGAMI_NN_WEIGHTS_DIR` points it at another directory when no directory is given.

`param/<arch>/` holds parameter-predictor models (`backend_id_t::param_v1`, msgpack; read only when
Origami is built with msgpack-cxx). Each is a small MLP over M, N, K and batch with one softmax head
per kernel parameter (macro tile, DepthU, GSU, WGM, StaggerU, ...) that scores every candidate by its
own parameter values, distilled offline from a larger selector. A model is bound to the one logic
file its index line names, and hipBLASLt also refuses it unless the library's solutions are of the
model's problem type (`problem_key`); `HIPBLASLT_ORIGAMI_PARAM=0` turns the binding off.
