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
combination's share of the sweep scales with the product of its values' weights, raked over the
combinations that can hold a problem (the excluded pairings above do not take a share), and the
cut to `--count` keeps those shares: each combination's part of the cut is proportional to its
raked size, with every unweighted enum and bool parameter's values at equal shares. Where the
declared constraints tie an unweighted parameter's values to others' (in `sdpa_bwd` only a
causal mask has a bottom-right anchor, so equal `alignment` and `is_causal` shares would leave
no causal top-left graph), equal shares would leave some combination with almost no rows; that
parameter is kept as the first pass left it, the run says so on stderr, and every combination
keeps its place in the cut. A declaration that weights
nothing is cut in proportion to what its first pass holds. The SDPA files weight `layout`,
`scale` and, on `sdpa_fwd`, `window`, keeping the common configuration (BSHD, `1/sqrt(head_dim)`,
no window) at about half of each facet of the sweep from about `--count 300` up; a shorter cut
cannot place a share that fine (no window is 0.42 of `sdpa_fwd` at `--count 100`). The weights
shape the sweep and its cut only: recorded model shapes (`--model-shapes`) and kernel-catalog
entries (`--kdp-root`) are kept as recorded. Each is a real configuration whose facets are what
they are (a pack's from its `kernel_catalog.constants`, a shape file's from its columns or the
`model_shape_defaults` below), and moving them to the declared mixture would misstate what was
recorded, so a set with a large model or kernel share holds the common configuration more often
than the declared weights say.

The SDPA files do not weight `form`, so each form is a stratum of its own and counts toward the
first pass's budget like `dtype` does. That budget is `--per-combination` (default 50) for each
live combination of the unweighted parameters (`is_causal`, `alignment`, `dtype`, `form`, and on
`sdpa_fwd` `generate_stats`), not of the weighted ones: without `--count`, `sdpa_fwd` is about
8,900 problems and `sdpa_bwd` about 2,700. That is the default corpus cost: an engine-backed
default `sdpa_fwd` run issues about four times the engine queries of the declaration without the
SDPA facets (1.03 million against 0.26 million, 1,600 s against 470 s, for one RDNA engine at
`--count 300`), and a lower `--per-combination` shortens it in proportion. The pipeline passes
`--count 2000` per operation.

Without `--count` the forms come out in proportion to the combinations that can hold them:
`sdpa_fwd` holds about 1,400 graphs each of `plain`, `dropout`, `padded`, `bias` and `sink`,
about 650 each of `scale_tensor` and `ragged` and about 560 of `scale_unset`, because those
forms are declared with one scale (`scale_unset`, `scale_tensor`: `inv_sqrt_d` only) or one
layout (`ragged`: `bshd` only), so fewer combinations can hold a problem. With `--count` the cut
takes each form evenly, as far as the first pass holds it: `sdpa_fwd` at `--count 2000` comes
out at about an eighth each, while `sdpa_bwd` holds only about 220 rows each of `scale_unset`
and `scale_tensor`, all of which a 2000 cut takes (about 11% each, against about 19.5% for each
of the other four forms). Those two forms state only `1/sqrt(head_dim)`, so with them used up
its share of `sdpa_bwd` at `--count 2000` is about 0.44. `--form-floor` tops each form up to its
count regardless.

Because the product of sdpa_fwd's facets (6912 combinations) is past the 1024 bound, the forward
explores a scattered sample of the combinations that can hold a problem, every declared value
present. A `--keep` on an enum or bool facet narrows the combinations before that bound applies,
so the sample is of the combinations the keep admits (a keep that leaves 8 of the 6912 explores
all 8). `manifest.json` reports the combinations under
`reports.categorical_combinations.<operation>`: `total` is the product the keeps admit,
`constraint_empty` how many of them the declaration's constraints leave unable to hold a problem
(`ragged` over `bhsd`, say), and `not_explored` how many that can hold a problem the bound left
out (0 when the product fits under it). The run prints the same counts on stderr, and says so
when every combination is constraint-empty.

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
  reports a regime the engine does not serve as short, without failing. A regime is reported
  `saturated` only when the search proved it holds no more problems; a focused search that stops
  without that proof reports the regime `not reached`, and a quota left short then exits 3.
- `--anchor-scaling` keeps the declared archetype/neighbourhood mixture at a large `--count`;
  without it the anchored share falls as exploration fills the set. It changes the set, so it
  is part of the set's identity like `--count` and `--seed`. `manifest.json`'s
  `reports.anchored_share.<operation>` (`anchored`, `delivered`, `share`, `anchor_scaling`) gives
  the share of the sweep's graphs delivered after the cut to `--count` that come from the
  archetypes and their neighbourhoods, with or without it.
- `--form-floor n` asks n in every optional **form** the operation declares (`form_label`):
  for `sdpa_fwd` plain, `scale_unset`, `scale_tensor`, `dropout`, `padded`, `ragged`, `bias`,
  `sink`; for `sdpa_bwd` all but `ragged` and `sink`. Each form varies one optional attribute
  or tensor from the plain graph, so a model trains and is validated on graphs that leave it
  unbound (RFC 0019 §6.7, absent is not zero, and §6.8 rule 9). Forms are a manifest column of
  their own, not part of the regime label, and are reported as `form:<label>` in
  `reports.regime_quota`; the live engine decides which it serves.
- A graph's name is the operation, its regime and every parameter in alphabetical order
  (`..._is_causaltrue_layoutbshd_scaleinv_sqrt_d_seqlen_k512_seqlen_q1_windownone`), with one
  limit: a name longer than 200 characters (`MAX_GRAPH_NAME`, `GraphIdentity.hpp`) keeps its
  leading parameters and ends in `_` and a 16-hex hash of the full name. About two in five
  default `sdpa_fwd` names are cut that way and do not show their last parameters (`seqlen_q`,
  `seqlen_k`, `window`, ...), so read a problem's parameters from its `q.<parameter>` manifest
  columns, never from the name. The `.fb` file, `results/<name>.csv` and `--problem-id` follow
  the name; the `benchmark` key (the content key, `graph_key` in `hipdnn_bench`) does not: a
  `plain`-form, BSHD, `inv_sqrt_d`, no-window problem has the key of the same graph built
  without those facets. Join sets by `benchmark`, never by name.
- A `--model-shapes` CSV with no `form`, `layout`, `scale` or `window` column reads as `plain`,
  `bshd`, `inv_sqrt_d` and `none`: the operation file's `model_shape_defaults`, one list for all
  four. `manifest.json`'s `reports.model_shapes[].defaulted` maps each such parameter to the
  number of rows that took its default. Any other missing column, `alignment` or
  `generate_stats` included, and a listed column left empty in a row, still refuses the row.
- `--include-extremes` takes the served point at each end of every numeric parameter first;
  `--ceiling` is the largest extent searched. Every manifest records `served_extent` (how far
  the search reached; a maximum near the ceiling is marked `at_search_ceiling`).
- `--exclude-corpus` keeps another set's graphs out; corpus_gen walks through the excluded
  configurations and returns new ones.
- `--model-shapes` adds recorded real-model shapes as `q.<parameter>` columns (defaults for a file
  with no column for a parameter: above).
- Exit status: 0 success; 1 bad arguments or an unregistered engine; 2 empty set (the engine
  serves none of the proposed problems); 3 fewer than `--count` without the engine being shown
  to serve no more.
