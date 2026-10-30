# tilewright

tilewright is a small CPU library that ranks candidate GEMM kernels for a
problem with a trained model. A caller hands it a GEMM problem, the device's
properties and its pool of candidate kernels; tilewright filters the pool,
scores the survivors with the model trained for the problem's region of the
problem space, and returns the pool best-first.

hipBLASLt's TensileLite uses it, when `TENSILE_USE_TILEWRIGHT` is set, to order
the kernels of a library's Prediction rows, falling back to its analytical
(Origami) ranking when tilewright has no model for the library or scores none
of its kernels. tilewright itself
depends on no GEMM framework: it does not include Tensile, TensileLite or
Origami headers, and callers convert their own types into tilewright's.

The models are produced by the training pipeline in
[`training_pipeline/`](training_pipeline/README.md).

The library files of a problem type are rows that TensileLite searches in
order, and a GridBased row answers every problem size. A TensileLite that
looks ahead to a loaded model (with `TENSILE_USE_TILEWRIGHT` set; earlier
builds search the rows in their order) searches the Prediction row of a
library whose model is loaded before the GridBased rows a problem reaches
ahead of it, so that a GridBased row of a few kernels does not answer in the
model's place. The other rows ahead of that Prediction row still answer
first: the rows before the first GridBased row (Equality rows for the sizes
they were tuned for, Range and FreeSize rows, Prediction rows without a
model) and the rows other than GridBased ones between it and the model's.
The look-ahead stays within the rows of one hardware row's library, and
without a loaded model the rows keep their order. The training pipeline's
stage 7 reports which rows serve its problems as production calls them (the
production route).

TensileLite can also offer tilewright the problem type's tuned Equality kernels
(the bank) for every problem size: with `TENSILE_TILEWRIGHT_BANK=1`, or, when
the variable is unset, when the model was trained on the bank (`bank` in
`describe`, MLREC_v3 serving flag bit 6) or reads kernel attributes
(`attribute_names`), which the variants of one tile in a bank need to be told
apart (`TENSILE_TILEWRIGHT_BANK=0` turns it off). The pool is then the
Prediction table in table order, followed, in ascending solution index, by the
Equality-row solutions that the device's hardware predicates accept from every
library file of the same problem type and architecture (the same file name once
its `_CU<count>` and `_ID<chip ids>` parts are removed), without the solutions
of the Prediction table. `top1_index` in `TILEWRIGHT_PICK_LOG` is a position in
that pool, so a training pipeline that ranks the same files for the same device
builds the same positions. Equality rows still answer their exact sizes first,
and tilewright does not offer a kernel such a row already returned;
`TENSILE_PREDICTION_LIB=1` skips those rows, so that tilewright alone ranks the
bank pool. A library builds its bank pool, and the pool's `CandidateSet`, on
its first call on each device.

A library stem can also ship a blocklist next to its `tilewright_index`,
`<stem>.tilewright.blocklist`: the names (`ContractionSolution::name()`, the
solution names of the logic file) of kernels that compute wrong results, one
per line, after comment lines that start with `#`. TensileLite leaves them out
of every pool it ranks with the model, the Prediction table's and the bank's,
and the analytical ranking that fills the remaining slots of a call skips them
too. A name lists every kernel whose name equals it once both drop their IA
(InterleaveAlpha) and PKA (PreloadKernArgs) tokens, which builds of
TensileLite spell differently for one kernel, so a blocklist written from one
build lists the same kernels in another; a kernel that differs from another in
those tokens alone is listed with it. TensileLite warns when names match no
kernel of a pool it ranks: they leave nothing out. A view key whose first
member is listed is served by its next member, a key whose members are all
listed is no candidate, and `top1_index` counts the kernels that remain, as
the training pipeline's pools do. A blocklist that cannot be read makes the
model unusable for the library.

## Layout

| Path | Contents |
|------|----------|
| `include/tilewright/` | Public API: `types.hpp` (problem, kernel and device types), `model.hpp` (loading, routing, ranking) |
| `src/tilewright/` | The engine: `format.cpp` (MLREC_v2 and MLREC_v3 loader, CRC-32, lazy dequantization), `features.cpp` (feature catalogs and their registry, LDS gate, feasibility rules), `kernels.cpp` (fp32 inner-product kernels with run-time ISA selection), `rank.cpp` (routing, execution contexts, view-key grouping, filtering, scoring, tie-breaks, `CandidateSet`), `registry.cpp` (file loading, path deduplication, `tilewright_index`, environment knobs), `cache.cpp` (the cache limit) |
| `python/` | nanobind bindings (`import tilewright`) and their pytest suite |
| `tests/` | Catch2 suite; `model_writer.*` writes synthetic models so the suite needs no shipped weights |
| `cmake/` | Package configuration template for `find_package(tilewright)` |
| `training_pipeline/` | Training pipeline that produces the models |

## How a problem is ranked

**Execution context.** A ranking describes where the kernels will run: the CUs
they may use and the tile scheduling (`ExecutionContext {cu_budget, schedule}`;
the default is exclusive use of the device with the default schedule).
`supports(model, context, hardware)` tells whether a model declares a context.
MLREC_v2 models declare every CU of the device with the default schedule, the
context they were trained with. An MLREC_v3 model declares the range of usable-CU
fractions (`N_eff / N_CU`, with `N_eff = cu_budget` when `0 < cu_budget <
N_CU`, else `N_CU`) and the schedules it was trained for; its features
describe the context. The fraction is relative to the device it is given, and
a compute partition is a device of its own with fewer CUs, so a model may also
declare a CU floor: the fewest usable CUs it was trained with.

Neither a CU budget nor a device's CU count (one or more) leaves a ranking
unscored, so TensileLite never ranks a call with Origami for its CUs. A context
outside the declared ones only by its CU budget is ranked at the nearest one the
model declares on that device (`scoring_context(model, context, hardware)`): a
budget below the range as the fewest usable CUs it declares (its CU floor or its
smallest fraction of the device, whichever is more), a budget above it as the
most (every CU, for an MLREC_v2 model). A device on which the model declares no
context, one with fewer CUs than the floor, is ranked at the most CUs its
largest fraction allows: the whole device, for a model that declares every CU.
The features and launch plans of the ranking are those of that context, so it
ranks bit for bit as a call made with it; the caller still launches the chosen
kernel with its own budget. On a compute partition that context is one of the
partition, which the model declares but need not have been trained on. A
schedule the model does not list leaves every kernel unscored, and TensileLite
then uses its Origami ranking.

**Routing.** A problem maps to one of 96 base cells
`<M tier>|<N tier>|<K tier>|<batch tier>`: M and N are `Tiny` (≤ 32), `Small`
(≤ 128), `Mid` (≤ 512) or `Large`; K is `TinyK` (≤ 32), `MidK` (≤ 512) or
`LargeK`; batch is `Bnone` (batch == 1) or `Bany`. A model's split tree refines
a base cell along one axis at a time (`M`, `N`, `K` or `B`; values at or below
the threshold go to the `lo` child), giving labels such as
`Large|Large|LargeK|Bnone#N<=10092`. The problem is scored by the leaf's cell,
or by its nearest trained ancestor (the label with trailing `#...` segments
removed) when the leaf has no trained cell. With no trained cell at all, every
kernel of an MLREC_v2 model is returned unscored. An MLREC_v3 model instead
routes the problem through the other base cells of its split tree, nearest
first (the sum of the tier differences on M, N, K and batch, ties to the lower
base index), and scores it with the first trained cell found.

**Filtering.** Each kernel of the pool then passes, in order:

1. an LDS-capacity gate: `mt_m*mt_k*bytes(A) + mt_n*mt_k*bytes(B)` must fit in
   the device's LDS;
2. a feasibility filter. MLREC_v2 models apply its validity rule: an operand
   whose free dimension is contiguous holds at least one global-read vector
   along it (M ≥ `grvw_a` when A is not transposed, N ≥ `grvw_b` when B is
   transposed, except for Int8x4 inputs), as TensileLite's
   `LeadingFree0/1SizesGreaterOrEqual` predicates require. MLREC_v3 models
   read no framework-specific parameter and leave that check to the
   integration, whose own predicates re-check each ranked kernel. Its soft rules are
   selection heuristics: a small batched problem (M, N ≤ 256, K < 1024,
   batch ≠ 1) must fit in one macro tile; Dot2 kernels (MI 1x1x64) only serve
   M < 3; and hinted kernels (`cache_hints != 0`) are only used when K and the
   kernel's `mt_k` are both 128-byte aligned in A's element size, where a skinny
   problem requires the non-temporal variant of its large operand
   (`cache_hints == 4`) when the pool contains such a variant for that operand,
   while every other problem rejects hinted kernels. MLREC_v2 models apply the
   first two soft rules and the non-temporal rule unless their file says they
   are served without it; MLREC_v3 models apply each soft rule only when their
   serving flags switch it on. An MLREC_v3 model may also refine the
   non-temporal rule family-wise: for a batched problem with K ≤ 512 that is
   thin along exactly one of M and N (M ≤ 32 < N, or 32 < M ≤ 128 with
   N ≤ 32), a kernel family (the pool's kernels with equal `mt` and `mi`)
   whose every member the rule rejects keeps one of its signatures, the
   hint-free one if it has one, else the one with the lowest
   `(cache_hints_a, cache_hints_b)`. The rule then passes every kernel of that
   signature, so with spec attributes or launch keys a family can keep
   several candidates. A `CandidateSet` finds each kernel's family once, when
   it is built;
3. the cell's whitelist: kernel signatures
   `(mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, cache_hints_a, cache_hints_b)` and, in
   MLREC_v3 files, view-key hashes (see below). An MLREC_v3 model whose serving
   flags turn the whitelist tier off skips this step.

The feasible whitelisted kernels form tier 1. If the whitelist rejects every
feasible kernel, all feasible kernels form tier 1. When the caller asks for
more results than tier 1 holds (`min_scored`), the other feasible kernels are
scored as tier 2.

An MLREC_v3 model with the trained-key gate lists the view keys (see below)
that had training data. A `CandidateSet` marks the pool's other kernels when
it is built, and the tiers hold only listed kernels unless no listed kernel
is feasible for the problem; then every feasible kernel takes part as above.
A model never trained on a kernel's view key cannot rank it reliably, so a
pool that grows after training ranks what the model knows. The gate does not
change what the non-temporal rule reads from the pool: whether it holds a
non-temporal variant of each operand, and which signature each family keeps,
count every kernel, listed or not, so kernels added after training can still
change which listed kernels that rule passes.

A model with a cascade width K (in its file) scores a tier 1 that is not a
whitelist, and in an MLREC_v3 model also tier 2, in two stages when the tier
has more than K view-key groups (see below): every group is scored by the
embedding term of the score, and only the K best by that term get the full
score, further groups following in embedding-term order while the tiers so far
keep fewer than `min_scored` configs. A kept group's feasible kernels get the
full score in all their launch classes (see launch plans). The groups left out
are unscored. MLREC_v2 models give every group of tier 2 the full score.

**Scoring.** Each cell is a two-tower model. Features are computed in double
precision and rounded once to float, as in the training pipeline, and are
whitened with the cell's means and standard deviations, `(x - mean) / std` (a
standard deviation below 1e-6 counts as 1):

- query features (55 in v2, 59 in v3) describe the problem, the device and,
  in v3, the execution context;
- item features (12 in v2; 8 in v3, plus the attribute features of the
  model's spec) describe the kernel;
- interaction features (37 in v2, 44 in v3) describe the problem and kernel
  together, including the matrix-instruction latency, which uses the MI-latency
  table and the `parallel_mi_cu` constant stored in the model (v2 also uses the
  bandwidth constants stored in it).

An MLREC_v3 model may store per-feature bounds; every raw feature is then
clamped to its cell's `[lo, hi]` before whitening.

An item feature whose stored standard deviation is below 1e-3 is treated as
constant: in training it took a single value, which lies within rounding noise
of the stored mean. A value within two standard deviations of the mean,
`|x - mean| <= 2 * std` in float, whitens as above; any other value whitens to
0, the value at the training mean, rather than to a huge multiple of a tiny
deviation (for example an occupancy the cell never saw). Query and interaction
features always use the plain whitening. The training pipeline whitens its
features with the same rule.

```
score = dot(query_mlp(q), item_mlp(i)) / max(|temperature|, 0.1) + interaction_mlp(x)
```

The first term is the embedding term.

Higher is better. A model file names the feature catalog it was trained on by a
hash, and the loader selects that catalog; this engine computes two,
`e7fe4b524851e895` (MLREC_v2, `feature_catalog_hash()`) and `b8b9f4a518add96f`
(MLREC_v3), listed by `feature_catalog_hashes()`, and rejects other hashes.

**Kernels that score alike.** The v2 catalog reads twelve numeric `Config`
fields, the kernel's view key: `mt` and `mi` (m, n, k each), `occupancy`,
`cache_hints_a`, `cache_hints_b`, `grvw_a`, `grvw_b` and `gwvw_d`. Kernels with
equal view keys get identical features, feasibility, whitelist membership and
scores, so a pool is grouped by view key and each group is filtered and scored
once; a `CandidateSet` caches item embeddings per group. The result still has
one entry per kernel, exactly as if every kernel were scored on its own.
An MLREC_v3 view key is the kernel's signature, the parameters every GEMM
kernel framework defines: `mt` and `mi` (m, n, k each), `cache_hints_a` and
`cache_hints_b`, followed by the attributes of the model's attribute spec, each
taking its default when the kernel lacks it. Kernels that differ only in other
fields (`occupancy`, `grvw_a/b`, `gwvw_d`) or other attributes are one
candidate to a v3 model unless the model serves launch keys and the launch
model tells their launches apart (see launch plans). `index` and the attributes a model does not read
(named integer kernel properties, see the API) never change a ranking.

**Result.** `rank` returns one `Result {config_index, score, scored}` per
input kernel (`config_index` is the kernel's position in the pool): tier 1
best-first, then tier 2 best-first, then the unscored kernels in input order.
Equal scores keep input order (`TieBreak::PoolOrder`, the default). With
`TieBreak::Prior` and a caller-supplied prior (one value per kernel), equal
scores are ordered by ascending prior, non-finite priors last, then input order;
the prior never moves a kernel past one with a different score or into another
tier. A kernel whose score is not finite is unscored, and invalid hardware
(`N_CU`, `lds_capacity` or `L2_capacity` equal to 0) leaves every kernel
unscored.

**Numerics.** Scoring is fp32. Every inner product uses one fixed summation
order: 16 interleaved lanes accumulated with fused multiply-adds, a fixed
pairwise reduction of the lanes, then the remaining elements added in order.
The scalar, AVX2 and AVX-512 kernels (chosen at run time from the CPU's
features) implement that order exactly, including when the interaction MLPs
of up to 16 kernels are evaluated together with one kernel per vector lane,
and the library is built without
fast-math or floating-point contraction, so rankings do not depend on the
instruction set the CPU offers or on the compiler (GCC or Clang). Features are
computed in double with the C library's `log2` and rounded once to float. Weights stay
packed in memory and a cell's weights are converted to fp32 on the first
ranking that reaches the cell.

**Memory.** The fp32 weights of the cells in use and the per-cell caches of
candidate sets (whitelist membership and item embeddings) are kept while their
total, over every model and candidate set of the process, is within the cache
limit (`set_cache_limit`; 1 GiB unless `TILEWRIGHT_CACHE_MB` sets it). Past it,
the least recently used of them are released and built again from the packed
weights when a ranking needs them, so rankings are the same under any limit;
only their first use after a release takes longer. `CandidateSet::prepare()`
keeps a copy of what rankings of a cell's whitelist read for as long as the
candidate set lives, and each thread that ranks keeps its scratch arrays in one
2 MiB block until it exits, both in memory of `allocate_huge_pages`; the cache
cannot release either, so neither counts against the limit or in
`cache_usage()`. The thread that calls `prepare()` writes the copy, except the
kernel part of the interaction features when that part is built later: the
thread that builds it then copies it in. A thread's first ranking takes its
block. Rankings read the copy bit for bit as the data it copies, so a candidate
set ranks alike prepared or not.

## Model files (MLREC_v2)

tilewright loads MLREC_v2 and MLREC_v3 files; MLREC_v3 is described after
MLREC_v2.

A model file is little-endian:

| Field | Type | Notes |
|-------|------|-------|
| magic | `char[8]` | `MLREC_v2` |
| endian marker | `u32` | `0x01020304` |
| header_size | `u32` | bytes from offset 0 to the first payload byte |
| payload_size | `u64` | payload bytes (the trailer excluded) |
| payload_crc32 | `u32` | CRC-32/ISO-HDLC of the payload |
| weight_dtype | `u8` | 0 fp32, 1 bf16, 2 int8, 3 int4 |
| serving_flags | `u8` | bit 0: served without the non-temporal rule; the other bits are 0 |
| cascade_keys | `u16` | the cascade width, 0 for none |
| q_dim, i_dim, x_dim | `u32` each | 55, 12, 37 |
| n_cells, n_splits | `u32` each | |
| feature_catalog_hash | `u16` length + ASCII | must equal `feature_catalog_hash()` |
| arch | `u16` length + ASCII | for example `gfx950`, `gfx1250-strict` |
| parallel_mi_cu | `f64` | > 0 |
| bw_c0, bw_c1, bw_c2 | `f64` each | `bw_occ = min(1, c0*a*a + c1*a + c2)`, `a = min(total tiles, N_CU)` |
| mi_default_cycles | `f64` | cycles for an MI shape missing from the table |
| n_mi | `u32` | at most 1024 |
| MI table | `n_mi` x {`u32` m, n, k; `i32` dtype; `f64` cycles} | dtype is a `tilewright::DataType` value |

The payload follows: `n_splits` split records
(`u16`+parent label, axis char and NUL, `i32` threshold, `u16`+lo label,
`u16`+hi label), then `n_cells` cell records (`u16`+label, `u32` embed, hidden
and interaction widths, `f32` temperature, the six whitening vectors, `u32`
signature count and `i32[8]` signatures, then the weight matrices and biases of
the three towers). Matrices use `weight_dtype` (bf16: `bits << 16`; int8: `f32`
scale then `i8` values; int4: `f32` scale then two values per byte, low nibble
first, value = scale * (nibble - 8)); biases, whitening vectors and
temperatures are always fp32. The file ends with `MLRECEND`, and its size is
exactly `header_size + payload_size + 8`.

The MI table is looked up by `(mi_m, mi_n, mi_k, dtype)` with the fnuz FP8
types mapped to their non-fnuz counterparts; the latency feature is
`cycles / max(parallel_mi_cu, 1)`.

The loader validates everything before using it: sizes and counts against the
bytes that remain, the CRC, the feature dims and hash, finite floating-point
values, split axes, labels (non-empty printable ASCII without spaces), duplicate cell labels
and split parents, an acyclic split tree, the MI table, and the trailer. MLREC_v1
files are rejected; convert them with
`training_pipeline/scripts/convert_mlrec_v1_to_v2.py`. The training pipeline
writes MLREC_v2 and MLREC_v3 with `training_pipeline/lib/mlrec.py`.

## Model files (MLREC_v3)

An MLREC_v3 file keeps the layout of MLREC_v2 with these changes. The fixed
header has magic `MLREC_v3`, and bytes 29-31 (v2's serving flags and cascade
width) must be 0. The header tail is:

| Field | Type | Notes |
|-------|------|-------|
| feature_catalog_hash | `u16` length + ASCII | `b8b9f4a518add96f` |
| feature_spec_hash | `u64` | hash of the attribute spec, below |
| arch | `u16` length + ASCII | |
| parallel_mi_cu, mi_default_cycles | `f64` each | > 0 |
| n_mi, MI table | as in v2 | |
| serving_flags | `u32` | bit 0 small-batched rule, bit 1 Dot2 rule, bit 2 non-temporal rule (each applied when set), bit 3 no whitelist tier, bit 4 launch keys (the pool is grouped by launch key), bit 5 family-wise non-temporal rule (only with bit 2), bit 6 trained on the integration's bank as well as its table (ranking ignores it; an integration with a bank serves both), bit 7 trained-key gate, bit 8 CU floor; other bits 0 |
| cascade_keys | `u32` | the cascade width, 0 for none |
| min_cu_fraction, max_cu_fraction | `f64` each | `0 < min <= max <= 1`: the `N_eff / N_CU` the model supports |
| schedules | `u8` | bit 0 Default, bit 1 Dynamic, bit 2 Auto; not 0 |
| clip | `u8` | 1: cells carry feature bounds |
| n_attributes | `u16` | at most 64, then per attribute: `u16`+name, `i64` default, `u8` transform (0 identity, 1 log2p1, 2 nonzero, 3 onehot) and, for onehot, `u16` value count (1-64) and distinct `i64` values |
| trained keys | with bit 7 only | `u32` count (at least 1), then that many `u64` view-key hashes, strictly ascending: the view keys with training data |
| CU floor | with bit 8 only | `u32`, at least 1: the fewest usable CUs (`N_eff`) the model supports on any device; a file without it has no floor |
| provenance | optional | present when header bytes remain: `u16` entry count (1-5), then per entry a `u16`+key and a `u16`+value (printable ASCII without spaces), the keys `config_id`, `training_data_hash`, `source_revision`, `date` and `launch_model`, each at most once |

A loader refuses a file with a serving flag it does not know (`unknown serving
flags`). Every MLREC_v3 model the training pipeline writes declares a CU floor,
so an engine that predates the floor refuses those models, and hipBLASLt then
ranks with Origami and warns: ship such models only with an engine that has
the floor. `with_cu_floor(data, None)` of `training_pipeline/lib/mlrec.py`
removes the floor from a file without retraining; the file then loads in such
an engine and ranks as a model without a floor.

`i_dim` is 8 plus the attribute features: one per identity, log2p1 and
nonzero attribute (`x`, `log2(1 + max(x, 0))`, `x != 0`) and one per value of a
onehot attribute (`x == v`), in spec order; `x` is the kernel's attribute, or
the default. `feature_spec_hash` is the FNV-1a 64 hash of the text
`<catalog hash>\nversion\t3\n`, then the feature names in vector order as
`query\t<name>,<name>,...\n`, `item\t...\n` (the attribute features named
`attr_<name>`, `attr_log2p1_<name>`, `attr_nz_<name>` or `attr_<name>_eq_<v>`)
and `interaction\t...\n`, then one line per attribute,
`<name>\t<default>\t<transform>[\t<v1>,<v2>,...]\n` (values for onehot only,
decimal), then for each of four fixed golden inputs (`kGoldenCases` in
`features.cpp`, `_GOLDEN_CASES` in `training_pipeline/lib/mlrec.py`)
`golden\t<case>\t<bits>\n`, the bits of its query, item and interaction
features under fixed arch constants as 8-digit lowercase hex, joined by `,`.
The golden inputs have fixed launch plans (the default, and dynamic queues
and partial reductions) and execution contexts (exclusive use, CU budgets and
every schedule), so launch and context features are covered too. Golden case c
sets attribute j to its one-hot value `(j + c) mod count`, or to
`(7j + 3c) mod 17 - 4`. A model whose names, spec or feature computation differ
from the engine's has another hash and is rejected; for the empty spec the hash
is `640b5d2e92a9f47f`.

Each cell record adds, after the whitening vectors, the bounds when `clip` is
1 (`f32` q_lo, q_hi, i_lo, i_hi, x_lo, x_hi; infinities allowed, no NaN,
lo ≤ hi) and, after the signatures, `u32` view-key count and `u64` view-key
hashes: FNV-1a 64 over the view key's values, each as 8 little-endian bytes.

The v3 features are v2's with `N_eff` wherever v2 uses the device's CU count,
without `bw_occ` and without what only some frameworks define. The item
features are v2's eight of the signature (`log2_mt_m/n/k`, `log2_mi_m/n/k`,
`nta_norm`, `ntb_norm`), then the attribute features; the interaction features
drop v2's `log2_grvw_a_bytes` and `log2_grvw_b_bytes`. The additions:

- query: `cu_fraction` = `N_eff / N_CU`, and the schedule one-hot
  `schedule_default`, `schedule_dynamic`, `schedule_auto`;
- interaction, after `active_cus_norm`: the kernel's launch plan (below). With
  `tiles` = `ceil(M / mt_m) * ceil(N / mt_n) * batch`, `wgs` = `grid_wgs`, or
  `tiles` when it is 0, and `waves` = `ceil(wgs / N_eff)`: `is_dynamic` =
  `dynamic_queue`; `log2_grid` = `log2(max(wgs, 1))`; `log2_grid_per_cu` =
  `log2(max(wgs / N_eff, 0.001))`; `log2_wg_waves` = `log2(max(waves, 1))`;
  `wg_wave_eff` = `wgs / (waves * N_eff)` (1 when `waves` is 0);
  `log2_tiles_per_wg` = `log2(max(tiles / max(wgs, 1), 0.001))`;
  `uneven_work` = `wgs > 0` and `tiles % wgs != 0`; and `reduction_tree`,
  `reduction_parallel`, `reduction_atomic`, the one-hot of
  `partial_reduction` (all 0 for `None`).

**Attribute specs.** A spec may name only parameters that every GEMM kernel
framework defines; each one joins the view key and splits kernels the
signature alone would group. Framework-specific knobs (vector widths,
occupancy, workgroup mapping, stagger, prefetch and the like) never enter a
spec. The default spec is empty, so a default v3 kernel is its signature.

**Launch plans.** The engine knows no framework's launch rule. An integration
passes a `LaunchModel` in `PoolOptions::launch`; `plan(config, problem,
context)` returns `LaunchPlan {dynamic_queue, grid_wgs, partial_reduction}`:
whether workgroups take output tiles from a shared queue, the workgroups
launched (0: one per output tile), and how the workgroups that split an output
tile combine their partial results (`None` when none is split, `Tree`,
`Parallel`, `Atomic`). Without a launch model every kernel gets `LaunchPlan{}`,
a static launch of one workgroup per output tile.

`launch_key(config)` names a kernel's launch class, a `u64` the engine only
compares. The contract: kernels with equal signature, equal spec attributes
and equal launch key get the same plan for every problem and context. The
model decides whether keys count: a v3 model whose serving flags ask for
launch keys (`ModelInfo::launch_keys`) groups its pool by view key and launch
key, so each group is one candidate with one plan; whitelist hashes cover view
keys and the cascade's width counts them, so a whitelisted or kept view key
brings all its launch classes. Any other model groups by view key alone and
never asks for a key, as if every key were 0: kernels of one view key are one
candidate, scored with its first member's launch. The default key, 0 for every
kernel, declares that kernels of one view key launch alike. The engine asks
for keys once per kernel when a pool of a model with launch keys is grouped,
and for plans only in rankings of v3 models under a supported
context, once per group that gets the full score (not for the cascade's
embedding-only stage), with the group's first config in pool order. Since
`plan` is on the ranking's path, a launch model should hold what it reads of
each kernel (by `Config::index`, prepared with the pool, or from the
framework's own kernel struct) instead of searching `Config::attributes` by
name per call; `hipblaslt_launch::Launch` takes the pool's configs for this.
`prefetch(config)` lets a launch model ask the memory system for what `plan`
will read of `config` without waiting for it: in an engine built with
`TW_LEVER_PREFETCH` (see Building and testing), a ranking whose first tier is
a cell whitelist calls it during the query tower for the first config of each
key of that tier, before it asks for their plans. It must not change what
`plan` or `launch_key` return and must not throw; the default asks for
nothing.

A model must be served with the plans it was trained with; the training
pipeline computes hipBLASLt's (`training_pipeline/lib/hipblaslt_launch.py`),
and TensileLite's adapter must return the same plans from TensileLite's launch
code. `LaunchModel::id()` names the rule and its version (empty by default); the
pipeline records it as the model's `launch_model` provenance, and a
`CandidateSet` built on a model that records one throws
`std::invalid_argument` unless `PoolOptions::launch` has that id. hipBLASLt's
id is `hipblaslt:<arch>:<RULE_VERSION>`, where `<arch>` is the compiler target
of the library's stem suffix (`_gfx1250` and `_gfx1250-strict` both give
`gfx1250`); an adapter returns the id of the rule version it implements.

## C++ API

```cpp
#include "tilewright/model.hpp"

std::string error;
tilewright::ModelPtr model = tilewright::load_model(path, &error);  // nullptr + error on failure
if (!model) { /* report error */ }

std::vector<tilewright::Config> pool = ...;               // the caller's kernels
tilewright::CandidateSet candidates(model, std::move(pool));  // reuse for every problem

tilewright::Hardware hw{n_cu, lds_bytes, l2_bytes};       // queried from the device
std::vector<tilewright::Result> ranked = candidates.rank(problem, hw, /*min_scored=*/0);

// A call that runs with a CU budget or another tile schedule. A budget
// outside the model's range is ranked at the nearest one in it; only a
// schedule it does not list leaves the call to another ranking.
tilewright::ExecutionContext context{cu_budget, tilewright::Schedule::Dynamic};
if (!tilewright::scoring_context(*model, context, hw)) { /* use another ranking */ }
```

- `load_model(path)`, `load_model_from_memory(data, size)`,
  `load_model_by_index(logic_stem, dir)` and
  `load_model_by_index(logic_stem, dir, pci_chip_id)` never throw; they return
  `nullptr` and set the error message on failure. `load_model` returns the
  already loaded model for a path that resolves to the same file.
  `load_model_by_index` reads `<dir>/tilewright_index` (see Weights) and loads
  the model of the stem's row for the device's PCI chip ID (`pci_chip_id`, 0
  when unknown), else of its row without one; it returns `nullptr` with an
  empty error when the index or such a row is absent, and with an error when
  a row of the stem is malformed. Pass the chip ID as a `std::uint32_t` (a
  variable, `0u`, `std::uint32_t{0}`): a third argument of a plain `0` or
  `NULL` is ambiguous between the overloads, since it also converts to the
  error pointer.
- `Config` holds the kernel's numeric parameters, the caller's `index` (not
  interpreted) and `attributes`: further named `int64` properties
  (`Attribute {name, value}`) in any order, with non-empty, unique names. A
  feature catalog reads the attributes it knows; the shipped catalog reads none.
- `CandidateSet(model, configs, PoolOptions{tie_break, launch})` holds a fixed
  pool and its launch model (`std::shared_ptr<const LaunchModel>`, null for
  none; see launch plans above), and throws `std::invalid_argument` for a null
  model or a config with an empty or duplicate attribute name. Whitelist membership and the item embeddings of
  the whitelisted kernels are computed the first time a problem reaches a
  cell, the embeddings of the other kernels the first time a ranking of the
  cell scores one of them, and all are reused afterwards within the cache
  limit. The interaction
  features that depend only on the kernel, the problem's data types and the
  device are computed on the first ranking that scores a kernel and reused for
  problems with those data types on that device. It is safe to share between
  threads when its launch model's `plan` is; exceptions from `plan` propagate
  out of `rank`.
  `rank(problem, hardware, min_scored)` ranks for exclusive use of the device;
  `rank(problem, hardware, context, min_scored, prior)` ranks for `context`
  (ranked at `scoring_context(model, context, hardware)`, every kernel unscored
  when that has no value) and, with
  `TieBreak::Prior`, orders equal scores by `prior` (a vector with one value per
  kernel, or null; another size throws `std::invalid_argument`).
  `rank_top(problem, hardware, context, k, prior)` returns only the best `k`
  kernels: the scored results among the first `k` of
  `rank(problem, hardware, context, k, prior)`. It orders and writes no other
  result, so its cost does not include a result per pool kernel; an overload
  writes into a caller's `std::vector<Result>` and reuses its capacity.
  `prepare()`, `prepare(cells)` and `prepare(problem, hardware)` build ahead
  of time what the first rankings of those cells would build (the cells' fp32
  weights, whitelist membership and the item embeddings of every kernel; with
  a problem, also the kernel-only interaction features for its data types), so
  that an integration can pay for it at load time instead of on its first
  rankings.
- `rank_configs(model, problem, hardware, configs, min_scored)`,
  `rank_configs(model, problem, hardware, context, configs, min_scored,
  tie_break, prior)` and `rank_configs(model, problem, hardware, context,
  configs, options, min_scored, prior)` return the same results as a
  `CandidateSet` with default `PoolOptions`, with `tie_break` or with `options`
  for an ad-hoc list, without caching, and validate attributes and refuse a
  launch model the model does not record the same way.
- `attribute_names(model)` lists the Config attributes the model's feature
  catalog reads (none for v2 models, the attribute spec for v3), so a caller
  can attach only those.
- `route(model, problem)` and `cell_label(model, cell)` expose routing;
  `describe(model)` returns the arch, hash, weight type and sizes, the bytes of
  the packed weights and of the fp32 weights of every cell, and for v3
  the spec hash, supported contexts (the CU floor as `min_effective_cus`, 0
  for none), serving flags (with `bank`, which an
  integration reads to decide whether it serves its bank, and the number of
  view keys the trained-key gate lists), the number of base
  labels without a trained cell and the provenance the file records;
  `CandidateSet::untrained_keys()` and `untrained_configs()` count the pool's
  kernels the gate leaves out;
  `compute_features(model, problem, config, hardware[, context[, plan]])`
  returns the raw (unwhitened, unclamped) feature vectors for the config
  launched as `plan` (default `LaunchPlan{}`).
- `set_cache_limit(bytes)`, `cache_limit()` and `cache_usage()` set and report
  the process-wide limit on cached fp32 weights and candidate-set caches (see
  Memory above); a part in use is kept even when it alone exceeds the limit.
- `allocate_huge_pages(bytes)` and `free_huge_pages(p, bytes)` hand out and
  take back zeroed memory for data rankings read on every call (TensileLite
  keeps its launch records there): on Linux whole 2 MiB pages at a 2 MiB
  boundary, on transparent huge pages unless the system has them off, and
  heap memory where the system refuses such a mapping or is not Linux.
  `allocate_huge_pages` throws `std::bad_alloc` only when the heap fails too.

`DataType` values are part of the model format (MI-table keys). They follow
Origami's enumeration, which has an `Int4` entry that Tensile's does not, so
convert Tensile data types by name rather than by value.

## Python

```bash
pip install shared/tilewright/python
```

```python
import tilewright as tw

logic = "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
model = tw.load_model(f"{logic}/gfx950/gfx950/Tilewright/<name>.tilewright.bin")
pool = tw.CandidateSet(model, [tw.Config(mt=tw.Dim3(256, 256, 64), mi=tw.Dim3(16, 16, 32), occupancy=1)])
problem = tw.Problem(size=tw.Dim3(4096, 4096, 4096), a_transpose=tw.Transpose.T,
                     a_dtype=tw.DataType.BFloat16, b_dtype=tw.DataType.BFloat16,
                     c_dtype=tw.DataType.BFloat16, d_dtype=tw.DataType.BFloat16,
                     mi_dtype=tw.DataType.BFloat16)
ranked = pool.rank(problem, tw.Hardware(N_CU=64, lds_capacity=65536, L2_capacity=4 << 20))
```

The module mirrors the C++ API. Enumerations are integer enums (`DataType.None`
is spelled `DataType.None_`), structures accept keyword arguments, and the GIL
is released while models load and pools rank. `Config.attributes` is a
`dict[str, int]`; `CandidateSet(model, configs, tie_break=TieBreak.PoolOrder,
launch=None)` takes as `launch` a callable `(config, problem, context) ->
LaunchPlan` (launch key 0), an object with such a `plan` method and
optionally a `launch_key(config) -> int` method (a `u64`; negative values are
taken modulo 2**64), or a list of `LaunchPlan` indexed by `Config.index`
(plans for one problem and context, grouped by identical plan; a missing index
raises `IndexError`), and errors raised by a launch model reach the caller;
`compute_features(..., context=None, plan=None)` takes a `LaunchPlan`;
`CandidateSet.rank(problem, hardware, min_scored=0, context=None, prior=None)`,
`CandidateSet.rank_top(problem, hardware, k, context=None, prior=None)`,
`CandidateSet.prepare(cells=None, problem=None, hardware=None)` (every cell
when given neither cells nor a problem)
and `rank_configs(model, problem, hardware, configs, min_scored=0, context=None,
tie_break=TieBreak.PoolOrder, prior=None, launch=None)` take the context and tie-break
arguments of the C++ API (`context=None` is exclusive use),
`supports(model, context, hardware)` reports whether a model declares a
context, `scoring_context(model, context, hardware)` the context it ranks one at
(`None`: not ranked),
`attribute_names(model)` lists the attributes it reads, and
`set_cache_limit(bytes)`, `cache_limit()` and `cache_usage()` manage the cache
limit.
`load_model` and `load_model_from_memory` raise `ValueError` with the loader's
message when the data is not a valid model, and invalid attribute names or a
prior of the wrong length raise `ValueError`; `load_model_by_index(logic_stem,
dir, pci_chip_id=0)` returns `None` when the index or the stem's row is
absent and raises `ValueError` when the listed model fails to load or a row of
the stem is malformed.

## Environment variables

Read once per process. For `TILEWRIGHT_DIAG` and `TILEWRIGHT_PICK_LOG`, a value
of `0` or an empty value counts as unset.

| Variable | Effect |
|----------|--------|
| `TILEWRIGHT_DIAG` | One stderr line per loaded model: `[TILEWRIGHT_DIAG FILE] path=... arch=... qhash=... qdim=55 idim=12 xdim=37 n_cells=N n_splits=N weights=int4` (the model's dims), and `[TILEWRIGHT_DIAG FAIL] <reason>` for a load that fails, a malformed `tilewright_index` row of the stem included |
| `TILEWRIGHT_PICK_LOG` | One stderr line per ranking that scores a kernel: `[TILEWRIGHT_PICK] m=.. n=.. k=.. b=.. tA=.. tB=.. leaf=<label> top1_sig=(mt_m=..,mt_n=..,mt_k=..,mi_m=..,mi_n=..,mi_k=..,cha=..,chb=..) top1_score=.. top1_index=.. n_configs=..`; `top1_index` is the pool position of the top-ranked kernel |
| `TILEWRIGHT_FORCE_CELL` | Score every problem with this cell index, for debugging; ignored when it is not a valid index of the model |
| `TILEWRIGHT_CACHE_MB` | Initial cache limit in MiB (a whole number; anything else leaves the default of 1 GiB); `set_cache_limit` changes it later |

tilewright writes nothing else to stderr. Whether hipBLASLt uses tilewright at
all is controlled by hipBLASLt (`TENSILE_USE_TILEWRIGHT`).

## Weights

hipBLASLt ships the models with the library logic they rank, in a
`Tilewright` directory of each library subtree's logic:
`projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/<subtree>/Tilewright/`,
where `<arch>` is the logic the subtree is built from: the subtree itself, as
in `asm_full/gfx950/gfx950/Tilewright/`, or `gfx1250-strict` for hipBLASLt's
`gfx1250v0` subtree. The libraries of an architecture's other logic
directories (`gfx950_128cu`, `gfx950_id75a3`, ...) load into the same subtree,
so their models are rows of the subtree's index. A directory holds one
`.tilewright.bin` per trained library, named after the logic file it was
trained on (so a model shared by several trees keeps the name of the tree it
was trained for), the libraries' blocklists (`<stem>.tilewright.blocklist`,
see above) and a `tilewright_index` that maps Tensile library stems to model
files:

```
# comment
<TensileLibrary stem>	<model file>
<TensileLibrary stem>	<model file>	chip=0x<PCI chip ID>
```

Fields are separated by blanks and a `#` starts a comment; a line of fewer
than two fields is skipped. A row may name, as `chip=0x` and one to four hex
digits that are not all zero, the PCI chip ID of the devices it is for, so that
device classes that load the same library file (parts of one architecture,
each in any compute-partition mode) can ship different models:
for a device, `load_model_by_index` loads the model of the stem's first row
naming its chip ID, else of the stem's first row without one, else none. A
row of the stem whose third field has any other form, or that has a fourth
field, is malformed and makes the lookup of that stem fail on every device.
An engine that predates the chip ID reads the first two fields of a stem's
first row whatever follows them: the row without a chip ID comes first, and
such an engine serves a stem that has rows for chip IDs only the model of its
first row on every device, so ship those rows only with an engine that reads
them. A CU-count class of devices that hipBLASLt gives a library file of its
own (a stem with a `_CU<n>_` token) loads that file, so a model serves it only
through a row of that stem.

The model files are ordinary binary files in the repository (marked `binary` in
`projects/hipblaslt/.gitattributes`), so any checkout that includes hipBLASLt
has them. TensileCreateLibrary reads only the `.yaml` files of the logic tree,
and hipBLASLt's configuration fails when a `Tilewright` directory holds one.
hipBLASLt's build target `hipblaslt-tilewright-models` (part of the default
build) copies each subtree's index, models and blocklists next to its Tensile
library files (`library/<subtree>/`, installed under
`lib/hipblaslt/library/<subtree>/`); TensileLite finds them only for libraries
loaded from per-logic files, which is hipBLASLt's default layout
(`HIPBLASLT_ENABLE_LAZY_LOAD=ON`).

## Building and testing

tilewright needs CMake 3.25.2 and a C++17 compiler. Standalone:

```bash
cd shared/tilewright
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

| Option | Default | Effect |
|--------|---------|--------|
| `TILEWRIGHT_BUILD_TESTING` | ON standalone, OFF as a subproject | Build the Catch2 suite (and the pytest suite with the bindings) |
| `TILEWRIGHT_ENABLE_PYTHON` | OFF | Build the Python module |
| `TILEWRIGHT_ENABLE_INSTALL` | ON standalone, OFF as a subproject | Install the library, headers and CMake package |
| `TILEWRIGHT_ENABLE_FETCH` | ON | Fetch Catch2 (and nanobind for CMake builds of the bindings) when they are not installed |
| `TILEWRIGHT_LEVER_PREFETCH` | ON | Define `TW_LEVER_PREFETCH` to 1, or to 0 when OFF, for the engine and what links it (see below) |
| `TILEWRIGHT_LEVER_MEMO` | ON | Define `TW_LEVER_MEMO` to 1, or to 0 when OFF, for the engine and what links it (see below) |
| `TILEWRIGHT_TEST_WEIGHTS_DIR` | `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full` of this checkout | Directory the real-model tests search for shipped models; they are skipped when it holds none |

The tests use `find_package(Catch2 3)` and fetch Catch2 only when it is missing
and fetching is enabled. They write their own synthetic models; the real-model
tests additionally load every shipped model. The environment-variable tests run
as separate ctest entries because the knobs are read once per process.

Two compile-time options, both on by default, shorten a ranking call without
changing any result: every ranking, and so every pick, is bit-identical to the
engine's without them, and with both off the engine compiles exactly as it did
before they existed. Each is off when its macro is defined to 0 and on
otherwise: `model_impl.hpp` defines a macro the build leaves undefined to 1,
and the CMake options define both, to 1 or 0. To build without one, configure
with `-DTILEWRIGHT_LEVER_PREFETCH=OFF` or `-DTILEWRIGHT_LEVER_MEMO=OFF` (a
parent project may set the variable before `add_subdirectory`); a build
without CMake defines `TW_LEVER_PREFETCH=0` or `TW_LEVER_MEMO=0` for the
engine and for whatever includes its private headers. CMake keeps an
option's cached value, so a build directory configured while the options
were off by default, a hipBLASLt one included, stays without them when its
sources update: configure it again with `-DTILEWRIGHT_LEVER_PREFETCH=ON
-DTILEWRIGHT_LEVER_MEMO=ON`, or use a new build directory. The copy that
`CandidateSet::prepare()` keeps and the threads' scratch blocks (see Memory)
are not such options but part of every build: rankings read the copy bit for
bit as the data it copies, and the scratch arrays hold the same values
wherever their memory comes from.

- `TW_LEVER_PREFETCH`: a ranking whose first tier is a cell whitelist asks,
  as it starts, for the scattered per-key data it is about to read and for
  the query tower's tensors, and during the query tower for the whitelisted
  keys' configs and kernel features, the scoring tensors and, through
  `LaunchModel::prefetch`, what the launch model's `plan` reads of those keys,
  so that a call with cold caches waits less. As it changes no result, no
  test can see what it brings into the caches (`tests/test_v3.cpp` checks
  only which configs a launch model is asked to prefetch): only timing
  ranking calls with cold caches shows whether it still helps.
  It asks for data that a call with warm caches already holds, so it costs
  such a call a little; turn it off where warm calls matter more.
- `TW_LEVER_MEMO`: when the thread's last ranking scored a cell whitelist
  alone, a ranking of the same problem, hardware and context over the same
  candidate set and launch model that needs no more configs than that tier
  holds (no prior) orders the remembered scores instead of computing them
  again. The memo serves once and is forgotten by the thread's next ranking
  that scores. This is TensileLite's re-rank, which asks
  `rank(..., min_scored = 1)` right after `rank_top(k = 1)` when its own
  predicates reject the top kernel. The two rankings need not belong to one
  call, though: the memo can also answer the first ranking of TensileLite's
  next call on the thread, when that call ranks an equal problem. The scores
  read the launch model's plans, so the memo requires `LaunchModel::plan` to
  return the same plan for the same config, problem and context on that
  thread between the two rankings, across calls too. `TW_LEVER_MEMO_CHECK`,
  on by default when `NDEBUG` is not defined and off when defined to 0, makes
  the memo ask the launch model again for every plan its scores read and rank
  afresh when one differs; without it, a ranking the memo answers asks the
  launch model for no plan. As a ranking that repeats the thread's last one
  is answered from the memo, a loop that ranks one problem again and again
  alternates fresh and remembered rankings. The selection-time benches
  `bench_shipped_rank` and `bench_selection_time` time such loops, so where
  the pipeline builds them (`training_pipeline/lib/shipped_subset.py`,
  `bench_selection_time.sh`) it builds them without the memo; build them so
  too to time fresh rankings.

The suite's ctest entries test the engine as the options build it and read
the macros it was compiled with: the options define them for what links the
engine as well, and `model_impl.hpp`, which the white-box tests include, sets
their defaults and that of `TW_LEVER_MEMO_CHECK`. Whatever the options say,
three more entries each run the whole suite in one process against another
build of the engine: `tilewright-levers` with both options and the memo
check, `tilewright-levers-nocheck` with both and without the check (as a
release build compiles the memo), and `tilewright-levers-off` with neither.
Under `training_pipeline/scripts/experiments`, `rerank_check` and
`memo_check` hash TensileLite's re-rank and pairs of rankings the memo may or
must not answer, for comparing builds on shipped models, and count the
rankings the memo answered. The entry `tilewright-levers-parity`
(`tests/levers_parity.cmake`) builds both against the engine with neither
option, with each alone, and with both, with and without the check, and runs
them on every shipped model with the sample kernel pool and the benches'
default shapes: every build must print the hashes of the build with neither,
and the memo must answer some re-rank in every build that has it. Without a
materialized shipped model it is skipped.

As a subproject, `add_subdirectory(shared/tilewright)` provides the static
library `roc::tilewright`. An installed tilewright is found with
`find_package(tilewright)`, which provides the same target.
