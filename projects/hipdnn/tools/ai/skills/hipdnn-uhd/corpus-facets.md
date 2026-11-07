# Choosing the corpus operation and facets

Shape sets come from `hipdnn_corpus_gen` (source:
`projects/hipdnn/tools/corpus_gen/CorpusGen.cpp`, `--help` for its usage). Every problem is
offered to the engine on the GPU, so the set is what the engine actually serves. It runs once
per operation the engine serves, with `--engine-name`, `--plugin-dir`, `--count`, `--seed`,
`--exclude-corpus` (every active set of the engine and device) and any `--keep` facets.

## 1. Operation

One `*.opmeta.json` per operation under `projects/hipdnn/tools/corpus_gen/operations/`:

| Family | Operations |
|---|---|
| attention | `sdpa_fwd`, `sdpa_bwd` |
| matmul | `matmul`, `matmul_epilogue`, `mx_matmul` |
| convolution | `conv_fwd`, `conv_bwd_data`, `conv_bwd_weights`, `conv_bias_activation` |
| normalization | `batchnorm_inference`, `batchnorm_fwd_training`, `batchnorm_bwd`, `batchnorm_activation_bwd`, `layernorm_fwd`, `layernorm_fwd_training`, `layernorm_bwd`, `rmsnorm_fwd`, `rmsnorm_fwd_training`, `rmsnorm_bwd` |
| elementwise / other | `pointwise`, `pointwise_unary`, `reduction`, `resample_fwd`, `resample_bwd` |

Pick the operation whose graph the engine accepts: for a descriptor engine, its UED's graph
matcher (`graph_match.native`, plus any `scope: graph` UMD); for an opaque engine, its plan
builders. The engine name is not evidence: the example `hkp_example:tiled_attention` UED
(`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/rocKE/gfx942_tiled_attention/`)
has `graph_match` `hipkernel.pointwise.graph_match` and its UMD matches
`hipkernel.pointwise.add_match`, so it serves `pointwise` graphs (two operands, identical
rank-4 extent, `mode` ADD). `--operation` restricts generation to that one operation.

When the graph-match symbol names no operation (a pack that is not an operation, e.g.
`attention_dense`), run corpus_gen without `--operation` and a small `--count`: its first
pass offers a few problems of every operation, and the operations the engine admits are the
ones it serves.

## 2. Facets (`--keep`)

`--keep q.<p>=<v>` keeps only problems with that facet, where `p` is a key of the operation
file's `parameters` and `v` a value it takes, compared as text with the value as printed: an
enum one of its `values`, an integer in decimal, a bool `true` or `false`. Different parameters conjoin; the same
parameter repeated widens it. Examples:

- `sdpa_fwd` for a bf16, head-dim-128, non-causal engine:
  `--keep q.dtype=bf16 --keep q.head_dim=128 --keep q.is_causal=false`
  (`dtype` values are `fp16`, `bf16`; `is_causal` is a bool).
- `pointwise` for an fp16 ADD kernel: `--keep q.dtype=fp16 --keep q.mode=ADD`
  (`mode` uses the FlatBuffers enumerator names ADD, MUL, SUB, DIV, MAX_OP, MIN_OP).

Keep facets to what the engine **declares** it supports (its UMD/KMD constraints), not to what
is convenient: every facet narrows the regimes the evaluation can see, and a regime missing
from the set is a regime the report cannot warn about.

## 3. Pack coverage (`--kdp-root`)

`projects/hipdnn/tools/corpus_gen/operations/engines.json` lists engines whose coverage is
`pack`: they serve exactly their pack's shapes (every shape field baked into the binary), so
`--kdp-root` is required and no search is run. An engine not listed there is searched. A new
pack-coverage engine adds its entry there on its branch.

## 4. Size, identity and exits

- `--count` is the set size per operation; `--count` and `--seed` are the set's identity, so
  changing either is a new set and a full re-collection.
- Each set's evaluation slice ([data.md](data.md)) is held out of training through
  `uhd_gen generate --eval-benchmarks`; a closed shape set uses `--recall` instead.
- `--regime-quota` / `--regime-quotas` ask at least n problems in a named regime (a sizing's
  quotas use these); `--regime-floor n` asks n in every regime the operation declares and
  reports a regime the engine does not serve as short, without failing.
- `--include-extremes` takes the served point at each end of every numeric parameter first;
  `--ceiling` is the largest extent searched. Every manifest records `served_extent` (how far
  the search reached; a maximum near the ceiling is marked `at_search_ceiling`).
- `--exclude-corpus` keeps another set's graphs out; corpus_gen walks through the excluded
  configurations and returns new ones.
- `--model-shapes` adds recorded real-model shapes as `q.<parameter>` columns.
- Exit status: 0 success; 1 bad arguments or an unregistered engine; 2 empty set (the engine
  serves none of the proposed problems); 3 fewer than `--count` without the engine being shown
  to serve no more.
