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
  (`dtype` values are `fp16`, `bf16`, `fp32`; `is_causal` is a bool).
- SDPA also varies `form` (which optional attributes and tensors the graph binds: `plain`,
  `scale_unset`, `scale_tensor`, `dropout`, `padded`, `bias`, and on `sdpa_fwd` `ragged` and
  `sink`), `layout` (`bshd`, `bhsd`, `kv_bhsd`: the strides), `scale` (`inv_sqrt_d`, `unit`,
  `jitter`) and, on `sdpa_fwd`, `window` (`none`, `local_128`, `local_1024`, `local_4096`). An
  engine that serves only plain BSHD keeps `--keep q.form=plain --keep q.layout=bshd`;
  otherwise leave them free, since a set with one form, layout or scale teaches a model nothing
  about the others. Not every pairing is declared, so a `--keep` on one facet can pin another:
  `scale_unset` and `scale_tensor` take `scale=inv_sqrt_d` only (they state no scale value, or
  pass it as a tensor, so the facet would list one graph under three scales), and `ragged`
  takes `layout=bshd` only (it packs tokens through offset tensors, so it needs token-major
  strides). A window needs seqlen_q <= seqlen_k whatever the form. Every other pairing, dropout,
  sink, bias and padded included with any window and layout, is declared: whether an engine
  serves it is the engine oracle's decision, not the declaration's.
- `pointwise` for an fp16 ADD kernel: `--keep q.dtype=fp16 --keep q.mode=ADD`
  (`mode` uses the FlatBuffers enumerator names ADD, MUL, SUB, DIV, MAX_OP, MIN_OP).

Keep facets to what the engine **declares** it supports (its UMD/KMD constraints), not to what
is convenient: every facet narrows the regimes the evaluation can see, and a regime missing
from the set is a regime the report cannot warn about.

An enum parameter may declare `weights` (value -> relative weight; unlisted values weigh 1): a
combination's share of the set scales with its values' weights, raked over the combinations
that can hold a problem (the excluded pairings above do not take a share). The SDPA files
weight `layout`, `scale` and `window`, keeping the common configuration (BSHD, `1/sqrt(head_dim)`,
no window) at about half of each facet; they do not weight `form`, so each form is a stratum of
its own and counts toward the first pass's budget like `dtype` does. That budget is
`--per-combination` (default 50) for each live combination of the unweighted parameters
(`is_causal`, `alignment`, `dtype`, `form`, and on `sdpa_fwd` `generate_stats`), not of the
weighted ones: without `--count`, `sdpa_fwd` is about 8,900 problems and `sdpa_bwd` about 2,700.
The pipeline passes `--count 2000` per operation. At `--count 2000` the forms come out about
even, not exactly: `sdpa_fwd` holds some 300 each of `plain`, `dropout`, `padded`, `bias` and
`sink` and 130 to 175 each of `scale_unset`, `scale_tensor` and `ragged`, and `sdpa_bwd` 400 to
430 each of `plain`, `dropout`, `padded` and `bias` and 160 each of `scale_unset` and
`scale_tensor`, because those forms are declared with one scale (`scale_unset`,
`scale_tensor`) or one layout (`ragged`) only, so fewer combinations can hold a problem.
`--form-floor` tops each form up to its count regardless. Because the
product of sdpa_fwd's facets (6912 combinations) is past the 1024 bound, the forward explores a
scattered sample of the combinations that can hold a problem, every declared value present.
A `--keep` on an enum or bool facet narrows the combinations before that bound applies, so the
sample is of the combinations the keep admits (a keep that leaves 8 of the 6912 explores all
8). `manifest.json` reports the count under `reports.categorical_combinations.<operation>`:
`total` is the product the keeps admit and `not_explored` how many of them the bound left out
(0 when none); the run prints the same count on stderr.

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
- `--anchor-scaling` keeps the declared archetype/neighbourhood mixture at a large `--count`;
  without it the anchored share falls as exploration fills the set. It changes the set, so it
  is part of the set's identity like `--count` and `--seed`.
- `--form-floor n` asks n in every optional **form** the operation declares (`form_label`):
  for `sdpa_fwd` plain, `scale_unset`, `scale_tensor`, `dropout`, `padded`, `ragged`, `bias`,
  `sink`; for `sdpa_bwd` all but `ragged` and `sink`. Each form varies one optional attribute
  or tensor from the plain graph, so a model trains and is validated on graphs that leave it
  unbound (RFC 0019 §6.7, absent is not zero, and §6.8 rule 9). Forms are a manifest column of
  their own, not part of the regime label, and are reported as `form:<label>` in
  `reports.regime_quota`; the live engine decides which it serves.
- A graph's name lists every parameter, so the new SDPA facets change it:
  `..._is_causaltrue_seqlen_k512_seqlen_q1` is now
  `..._is_causaltrue_layoutbshd_scaleinv_sqrt_d_seqlen_k512_seqlen_q1_windownone` (the leading
  `...` now also holds `formplain`). The `.fb` file, `results/<name>.csv` and `--problem-id`
  follow the name; the `benchmark` key (the content key, `graph_key` in `hipdnn_bench`) of a
  `plain`-form, BSHD, `inv_sqrt_d`, no-window problem is the same as before. Join an old set to a new one by `benchmark`, never by name.
- A `--model-shapes` CSV with no `form`, `layout`, `scale` or `window` column (shape files mined
  before those parameters existed) reads as `plain`, `bshd`, `inv_sqrt_d` and `none`: the
  operation file's `model_shape_defaults`, one list for all four. `manifest.json`'s
  `reports.model_shapes[].defaulted` counts the rows that took one. Any other missing column,
  `alignment` or `generate_stats` included, and a listed column left empty in a row, still
  refuses the row.
- `--include-extremes` takes the served point at each end of every numeric parameter first;
  `--ceiling` is the largest extent searched. Every manifest records `served_extent` (how far
  the search reached; a maximum near the ceiling is marked `at_search_ceiling`).
- `--exclude-corpus` keeps another set's graphs out; corpus_gen walks through the excluded
  configurations and returns new ones.
- `--model-shapes` adds recorded real-model shapes as `q.<parameter>` columns (defaults for a file
  that predates a parameter: above).
- Exit status: 0 success; 1 bad arguments or an unregistered engine; 2 empty set (the engine
  serves none of the proposed problems); 3 fewer than `--count` without the engine being shown
  to serve no more.
