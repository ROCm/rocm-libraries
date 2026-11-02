# SDPA Forward Test Plan: How the Cases Were Derived

> These test cases were generated from a PoC but they have already shown value in
> revealing defects in SDPA engines and are therefore being treated as an initial
> set while the specific approach is reviewed and finalized.

This document tells how the SDPA forward plan cases were made. It starts from
everything the operation could vary. It ends with the bundles that the
integration suite runs. Each step removes cases for a stated reason, so you can
check every step.

The files that this document describes are:

| File | What it holds |
|---|---|
| [`test-plans/sdpa-fwd/model.yaml`](../../test-plans/sdpa-fwd/model.yaml) | The model: the settings, their values, the labels, the tier recipe. |
| [`test-plans/sdpa-fwd/rules.yaml`](../../test-plans/sdpa-fwd/rules.yaml) | The rules that remove combinations. |
| [`test-plans/sdpa-fwd/injections.yaml`](../../test-plans/sdpa-fwd/injections.yaml) | The cases that are added on purpose, outside the generated sets. |
| [`test-plans/sdpa-fwd/cases.csv`](../../test-plans/sdpa-fwd/cases.csv) | Every generated case (16,729 rows), with its tier, its topology, and whether it became a bundle. |
| `integration-test-bundles/<tier>/SdpaFwdPlan/<Topology>/` | The bundles: one sweep per tier and topology. |

The tool that turns the model into the CSV is a local proof-of-concept (PoC)
tool. It is not part of the repository. Section 12 tells what it does, so that
a reader can check the CSV against the model.

## Contents

1. [Why this plan exists](#1-why-this-plan-exists)
2. [The theoretical base](#2-the-theoretical-base)
3. [Narrowing: meaning, then the frontend](#3-narrowing-meaning-then-the-frontend)
4. [The model](#4-the-model)
5. [The rules file](#5-the-rules-file)
6. [Generating the cases](#6-generating-the-cases)
7. [Tiers](#7-tiers)
8. [The CSV](#8-the-csv)
9. [From rows to bundles](#9-from-rows-to-bundles)
10. [Known engine defects and their skips](#10-known-engine-defects-and-their-skips)
11. [Limits and what comes next](#11-limits-and-what-comes-next)
12. [How to regenerate](#12-how-to-regenerate)

## 1. Why this plan exists

**SDPA** means *scaled dot-product attention*. It is the "attention" step of a
transformer model. For each query position, it compares the query with every
key, turns the scores into weights, and adds up the values with those weights.
The inputs are three tensors: **Q** (queries), **K** (keys) and **V** (values).
The main output is **O**. The forward direction computes O. The backward
direction (training) computes gradients; this plan does not cover it.

An **engine** is one implementation of an operation inside a provider plugin.
For example, `ASM_SDPA_ENGINE` in the hip-kernel provider computes SDPA with
assembly GPU kernels. hipDNN offers a graph to each engine. An engine
**accepts** a graph that it can run, or **declines** it.

An SDPA request has many settings: the number format of each tensor, the
memory layout, masks, dropout, paging, extra outputs, and more. An engine
supports only some combinations. To test an engine well, we need a list of
cases that:

- reaches every setting that hipDNN can express,
- puts settings together in pairs, because many defects need two settings at
  the same time,
- stays small enough to run in the time that each test tier has,
- has an expected answer for each case, so that a wrong output fails.

This plan makes that list from a written **model** and written **rules**,
with a script. A person does not pick cases one by one. To change the list,
change the model or a rule, and run the script again.

## 2. The theoretical base

The first step lists everything the forward operation could vary, with no
thought about what makes sense or what hipDNN can express. This list is
**Layer 0**.

- An **axis** is one thing that a case can vary. Example: `dtype_q`, the number
  format of Q.
- A **value** is one choice on an axis. Example: `bf16`.
- A **child axis** exists only when its parent axis has a given value. Example:
  `causal_alignment` (which diagonal the causal mask uses) exists only when
  `causal_mask` is `present`.
- A **combination** is one value for every active axis.

Layer 0 has 85 axes and 297 values. That gives about 1.5 × 10³⁹ combinations.
Nobody can run that many cases. So we count **pairs**: two values of two
different axes, for example `dtype_q = fp16` together with
`causal_mask = present`. A pair is **valid** when at least one allowed
combination holds it. Layer 0 has 43,301 valid pairs.

**Pairwise testing** puts every valid pair in at least one case. Most defects
need only one or two settings in a bad state, so pairwise testing finds most of
them with a small number of cases. For this reason, the plan measures its size
in pairs, not in combinations.

The layer numbers in sections 2 and 3 come from research files that are not
in the repository. A script counted them by arithmetic over those files: it
enumerates the size axes, splits the other axes into independent groups, and
multiplies the counts of the groups. It does not list the combinations one by
one.

## 3. Narrowing: meaning, then the frontend

**Layer 1 removes combinations that have no meaning.** These are
**definition** rules (ids `D-*`). A definition rule is a fact about attention
or about numbers. Examples:

- `D-DROPOUT-P-RANGE`: a dropout probability below 0 or above 1 is not a
  probability.
- `D-HEADS-K-DIVIDES`: each query head must use exactly one K head, so the K
  head count must divide the query head count.
- `D-OUTPUT-SELF-ALIAS`: output strides that send two output elements to one
  address cannot hold both results.

Layer 1 has 27 definition rules. They leave about 2.0 × 10³⁴ combinations and
38,386 valid pairs.

**Layer 2 removes combinations that hipDNN's frontend cannot express.** These
are **frontend** rules (ids `F-*`). The frontend is the hipDNN C++ API that
builds the graph. Examples:

- `F-EXTENT-NONZERO`: every tensor dimension must be at least 1, so an empty
  batch cannot be stated.
- `F-ONE-DIAGONAL`: the node has one diagonal alignment, so a causal mask and a
  sliding window cannot use different diagonals.
- `F-SCALE-TENSOR-BY-VALUE`: a scale tensor must be passed by value, so a
  scale in device memory cannot be stated.

Layer 2 has 10 frontend rules. They leave about 7.4 × 10³⁰ combinations and
33,314 valid pairs.

A removed combination is not lost. When the frontend accepts a meaningless
request, that request comes back as a **rejection probe** in
`injections.yaml`: a case that expects hipDNN to refuse it. Rejection probes
are not part of this plan's bundles.

## 4. The model

The model (`model.yaml`) is the Layer 2 space in a form that a script can
use. It has 79 axes (28 of them child axes) and 271 values.

**Classes, not numbers.** Most size axes hold a **class**, chosen by meaning.
Example: `batch_size` is `one` or `many`. A **relationship** axis says how two
sizes relate. Example: `seq_q_vs_seq_kv` is `shorter`, `equal` or `longer`.
The **class-to-value table** (`class_to_value` in `model.yaml`) turns each
class into numbers. Example: `k_head_sharing = gqa` means the K head count is
the query head count divided by 4, at least 2.

Rules hold only for the values that were tested. A result at head size 128
says nothing about head size 96.

**Base case.** `base_cases.shared` is the plain case that other cases start
from: bf16 in and out, fp32 compute, batch 2, 8 heads, head size 128,
packed BHSD layout, scale 1/sqrt(128), no mask, no optional input or output.
`ASM_SDPA_ENGINE` accepts it on gfx942 and gfx950.

**Labels.** Each axis has three labels for each value, written as
`<value>: [usage, changes_numbers, answer_key]`:

```yaml
  - id: dtype_q
    labels:
      fp32: [rare, "yes", "yes"]
      bf16: [common, "yes", "yes"]
      fp8_e4m3: [common, "yes", partial]
```

| Label | Values | Meaning |
|---|---|---|
| `usage` | `common`, `rare`, `fringe` | How often real software uses the value. `fringe`: no mainstream software uses it. |
| `changes_numbers` | `"yes"`, `layout`, `unclear`, `"no"` | `"yes"`: the correct output changes. `layout`: the output is the same, but the engine reads or writes memory in a different way. `unclear`: the evidence supports both. `"no"`: no number can change; only acceptance can be tested. |
| `answer_key` | `"yes"`, `partial`, `"no"` | Can a **reference executor** (a simple, slow program that computes the expected output) compute this value? `partial`: only together with some other values. |

YAML reads a bare `yes` or `no` as true or false, so these words are quoted.

Each axis also has:

- `group`: one of eight **interaction groups** (G1 to G8), for example
  `G1-mask-seq` or `G5-batch-heads-kvstorage`. Axes in one group can plausibly
  interact.
- `size_sensitive`: `"yes"` when a larger shape can exercise something that the
  smallest shape cannot (tile edges, many pages, long sums), else `"no"`.

**Answer-key rules.** Some combinations have no expected answer, although each
value alone has one. `level_labels.answer_key_needs` lists them, with ids
`AK-*`. Example: `AK-Q-O-FORMATS` lists the (Q format, O format) pairs that the
reference executors support. A tier that must be fully checkable applies these
rules.

**Bridges.** `level_labels.bridges` (B1 to B13) name pairs of axes in different
groups that still interact. Example: B3, "element size changes byte strides and
alignment".

**Other added cases.**

- **Targeted cases** (`level_labels.targeted`, X2 to X10): the base case plus a
  few changes, each aimed at one known hazard. Example: X6b, "a scale tensor may
  be ignored (scale 1.0 used)".
- **Real-world cases** (`level_labels.real_world`, 11 entries): settings of
  public models, for example Llama 3 8B decode and Mistral 7B sliding window.
- **Seeds** (`level_labels.seeds`, 15 entries): combinations that public
  defect reports or serving workloads showed to matter. Example: S1, Stats with
  ragged queries over a paged GQA decode.

## 5. The rules file

`rules.yaml` holds every rule. The script applies all rules at once; their
order has no effect. Each rule has a `kind`:

| Kind | Ids | Meaning | When to change it |
|---|---|---|---|
| `definition` | `D-*` | A fact about attention or about numbers. | Never, to fit a library. |
| `frontend` | `F-*` | A limit of hipDNN's frontend API. | When the API changes. |
| `policy` | `P-*` | A request that has meaning and that the frontend accepts, but that hipDNN should refuse. | When hipDNN's policy changes. |
| `choice` | `CH-*` | A judgement call: classes, representative values, fixed settings. | Freely, with the reason recorded. |

Each rule also has a `form`:

| Form | What it does |
|---|---|
| `exclusion` | Removes rows. It has `forbid` (a map of axis to values; a row matches when every listed axis has a listed value) or `forbid_any` (a list of such maps). |
| `value_list` | Explains why a value or an axis is not in the model. It removes nothing. |
| `structural` | Names a fact that the shape of the model already enforces, for example one shared batch size. It removes nothing. |
| `setting` | Records a choice about how the model is built. It removes nothing. |
| `covered` | A policy rule whose requests another rule already removes. |

The model applies 27 exclusion rules: 21 definition, 4 frontend and 2 policy.
Together they make 5,434 of the 35,993 structurally possible pairs invalid.
30,559 valid pairs remain.

Example of a `forbid`:

```yaml
  - id: D-REL-WINDOW
    forbid_any:
      - {seq_kv: [one], window_left: [partial]}
      - {seq_kv: [one], window_right: [partial]}
```

With one key, a window cannot cover "some but not all" keys, so these rows
are removed.

The two policy rules remove negative strides (`P-NO-NEGATIVE-STRIDE`) and a
zero stride on the head dimension of Q, K or V (`P-NO-HEAD-DIM-BROADCAST`). They
name ticket ALMIOPEN-3114, which asks hipDNN to refuse these requests. The
requests come back as rejection probes MV-22 and MV-23.

## 6. Generating the cases

The script builds rows. A **row** is one value for every active axis. A
**case** is a row at one **shape tier** (one set of concrete sizes, section 7).
One row can give several cases.

The script uses these methods:

- **Low-change pairwise.** Each row starts from the base case and changes as few
  settings as possible. First, the script covers every pair that a row with at
  most 3 changes can hold. Then it covers the pairs that need 4 changes, then
  5, and so on, until every pair is covered. A child axis that its parent
  switches on does not count as a change. Example: base case plus
  `bias = present` plus `alignment_k = element_aligned` plus
  `alignment_o = element_aligned` is 3 changes; `bias_broadcast` and
  `bias_dtype` follow from `bias = present` and do not count. A failure in a
  low-change case is easy to trace to one setting.
- **Packed pairwise.** Each row holds as many uncovered pairs as possible.
  This gives fewer rows, but each row changes many settings. An engine that
  cannot handle one of them declines the whole row.
- **One-at-a-time.** The base case with exactly one value changed (plus any
  value that a rule forces).
- **Targeted, real-world and seed rows** (section 4). The script completes each
  one with the fewest extra changes that obey the rules.

**Super-setting.** In the quick tier, the data-type axes (`dtype_q`,
`dtype_k_vs_q`, `dtype_v_vs_q`, `dtype_o`, `dtype_stats`, `compute_precision`)
act as one setting. Each complete data-type combination counts as one change
and pairs with every other value that quick allows. Quick holds 12 such
combinations.

**Warm start.** A tier asks only for the pairs that lower tiers did not cover
already.

**Determinism.** The script uses seed 2893. The same model, rules, timing files
and seed give the same CSV, byte for byte.

## 7. Tiers

The integration suite has four **tiers**: `quick`, `standard`, `comprehensive`
and `full`. The tiers are **cumulative**: a run of `standard` also runs every
`quick` case. A case's tier is the first tier that needs it.

The tier of each case comes from the labels and from a short **recipe**
(`levels:` at the end of `model.yaml`). The tiers are a draft; the labels and
the recipe are open to review.

| Tier | Values it takes | Must every case be checkable? | Pairs | Extra rows | Shape tiers | Per-case ceiling | Budget |
|---|---|---|---|---|---|---|---|
| quick | `usage` common; `changes_numbers` not "no"; answer key yes or partial | yes | all, low-change | real-world rows and checkable seeds | S | 20 MFLOP | 5 min |
| standard | as quick, plus `usage` rare | yes | all, low-change | targeted | S, H, M | 200 MFLOP | 30 min |
| comprehensive | as standard | yes | within groups and bridges, packed | one-at-a-time at S and L | S, H, M, L (one per row) | 8,000 MFLOP | 60 min |
| full | everything, including `fringe` and answer key "no" | no | all, packed | one-at-a-time at S and L; the base case at X | S, H, M, L, X | none | none |

The base value of each axis is always allowed. A required child axis whose
values all have `changes_numbers` "no" (for example `max_seq_len_kv` under
paging) does not hold its parent back.

**MFLOP** is a million floating-point operations. The work of one case is
2 × B × H_q × S_q × S_kv × (d_qk + d_v).

Most comprehensive and full cases are rows of a lower tier at a larger shape
tier. Example: a standard row at shape tier L is a comprehensive case, because
standard does not run shape tier L.

**Shape tiers** (`level_labels.shape_tiers`). A tier gives the size of every
`many`; a `one` stays 1.

| Shape tier | Batch | Query heads | Length (equal / short side) | Purpose |
|---|---|---|---|---|
| S small | 2 | 8 | 64 / 32 | the feature works |
| H hazard | 3 | 8 | 127 / 65 | odd batch; lengths one off a 64/128 tile edge |
| M model | 2 | 32 | 256 / 128 | the head layout of Llama 3 8B |
| L medium | 4 | 8 | 512 / 256 | a realistic length |
| X stress | 8 | 32 | 4096 / 2048 | long accumulation (full only) |

Rules that keep a tier small:

- An S case is never removed by a ceiling: S is the smallest form of a case.
- In comprehensive, each row keeps exactly one shape tier. A tier above S goes
  only to a row that changes a size-sensitive setting. Every value keeps at
  least one M case and one L case. The cases that comprehensive drops go to
  full.
- **Fill.** If a tier has spare time, the script pulls cases down from higher
  tiers until the tier uses 50 % of its budget. A pulled case must fit the
  tier's shapes, ceiling, labels and change limit.
- **Cost.** The time of a case comes from a per-case timing file, measured on
  one gfx942 node and one gfx950 node with the PoC runner. A case that was not
  measured gets a fitted estimate: a + b × MFLOP + c × tensor-MB. Each tier
  run adds 10 s of overhead. gfx942 is the budget basis.
- **Over budget.** The script only reports; it does not move cases.

Result for the committed CSV (from `cases.csv`; times are the summed
`est_seconds` columns plus the overhead):

| Tier | Cases in this tier | Cumulative cases | Cumulative s, gfx942 | Cumulative s, gfx950 | Budget use, gfx942 |
|---|---|---|---|---|---|
| quick | 1,330 | 1,330 | 46.1 | 45.3 | 15.4 % |
| standard | 6,782 | 8,112 | 432.6 | 321.8 | 24.0 % |
| comprehensive | 3,080 | 11,192 | 1,624.5 | 1,353.1 | 45.1 % |
| full | 5,537 | 16,729 | 4,823.2 | 3,800.4 | — |

These times are the PoC runner's times for one engine run plus one reference
run. The suite does more work per case, so its times are larger.

**Seeds.** The script completes each of the 15 seeds with the fewest changes
under the rules and the answer-key rules. A seed whose row the references can
check goes to quick, at shape S. Any other seed goes to full. 14 seeds are in
quick; `S3b-vllm-fp8kv-paged` is in full, because a K or V format that differs
from Q has answer key "no". In the CSV a seed row has `origin` `real_world` and
`detail` `seed:<id>`. One seed (`S10-blockmask-ragged`) matched a row that
low-change pairwise had already made; that row's `detail` carries the seed
name.

## 8. The CSV

`cases.csv` has one row per case: 16,729 rows. It has no quoted fields. A `-`
means that a child axis is not active in that row.

**Case columns (13).**

| Column | Meaning |
|---|---|
| `case_id` | `lv-NNNN`. It changes on every regeneration. Do not use it as a key. |
| `level` | The tier: `quick`, `standard`, `comprehensive` or `full`. |
| `shape_tier` | `S`, `H`, `M`, `L` or `X`. |
| `origin` | `low-change`, `pairwise` (packed), `one-at-a-time`, `targeted` or `real_world`. |
| `detail` | For low-change rows: the number of changes. For other rows: the id of the targeted case, real-world case or seed. |
| `filled_from` | Empty, or the tier that the recipe gave the case before a fill pulled it down. |
| `est_seconds` | The time of the case on gfx942. |
| `cost_source` | `measured`, `modelled` or `UNMEASURED` (outside the range of the fitted estimate). |
| `work_mflop` | The work of the case. |
| `tensor_mb` | The memory of the case's tensors. |
| `sizes` | The concrete sizes, for example `B2 Hq8 Hk1 Hv1 Sq64 Skv64 Dqk128 Dv128`. |
| `case_key` | 16 hex digits: a hash of the tier, the sizes and every axis value. It is unique per row and stays the same when `case_id` values shift. Use it as the key. |
| `est_seconds_gfx950` | The time of the case on gfx950. |

**Axis columns (79).** One column per model axis, from `batch_packing` to
`runtime_shape_override`, in the order of `model.yaml`.

**Appended columns (5).** These columns say what became of each row in the
suite. Remove them and you get the generator's output byte for byte.

| Column | Meaning |
|---|---|
| `topology` | The sweep folder name of the row (section 9). Every row has one. |
| `bundle_status` | `bundle` or `excluded`. |
| `exclusion_reasons` | For an excluded row: the reasons, joined with `;`. |
| `bundle_path` | For a bundle row: `<tier>/SdpaFwdPlan/<Topology>`. |
| `gtest_name` | For a bundle row: the GTest name that the suite gives the case. |

## 9. From rows to bundles

A **bundle** is a graph test in `integration-test-bundles/`. A **sweep** is a
`graph.template.json` that the suite expands once for each case in its
`sweep.json` ([File Formats](../file-formats.md)).

### 9.1 Which rows a bundle can express

A bundle holds a graph and, for each case, a few fixed values. It cannot hold
exact tensor contents. So a row that needs exact contents cannot be a bundle
yet. It stays in the CSV, marked `excluded` with its reasons. A row can have
several reasons.

| Reason | Why a bundle cannot express it | Rows |
|---|---|---|
| `seq_len_contents` | Per-sequence lengths are tensor contents (also needed by every paged row). | 3,553 |
| `page_table_contents` | Page tables are tensor contents. | 1,702 |
| `block_mask_bits` | The block-mask bits are tensor contents. | 1,449 |
| `dropout_mask_cycle` | A caller-supplied dropout mask is tensor contents. | 1,213 |
| `ragged_offsets` | Ragged offsets are tensor contents. | 1,082 |
| `descale_lists` | A per-head or finer descale is a list of factors in device memory. | 950 |
| `tensor_overlap` | Two tensors that share memory need control of the pointers, which a bundle does not have. | 540 |
| `arch_dtype` | The number format depends on the GPU (for example fp8 E4M3 becomes E4M3 FNUZ on gfx942), and one graph has one format. | 323 |
| `zero_stride_needs_pr14709` | A zero stride on an input needs a data-SDK change that is not on the base branch yet (PR #14709). Only the selection step removes these rows. | 182 (6 otherwise expressible) |
| `dropout_without_rng_dump` | Random dropout without an `rng_dump` output has no oracle: the references check random dropout only through the dumped mask. | 44 (8 otherwise expressible) |

9,883 rows became bundles. 6,846 rows are excluded. The selection step never
changes the model or the CSV rows. It only filters. To bring a row back, change
the filter and regenerate the bundles.

| Tier | All rows | Bundle rows | Sweeps | Largest sweep | Summed `est_seconds` of the bundle rows, gfx942 / gfx950 |
|---|---|---|---|---|---|
| quick | 1,330 | 749 | 23 | 286 | 20.0 / 19.7 |
| standard | 6,782 | 4,152 | 102 | 1,358 | 232.3 / 165.2 |
| comprehensive | 3,080 | 1,582 | 59 | 639 | 564.6 / 498.7 |
| full | 5,537 | 3,400 | 109 | 971 | 2,304.7 / 1,684.0 |
| total | 16,729 | 9,883 | 293 | | |

### 9.2 How a graph is made from a row

1. The PoC graph builder reads the row and the class-to-value table, and makes
   concrete sizes, strides, formats and attributes.
2. It builds the graph with the hipDNN frontend and calls `validate()`.
3. It serializes the graph to JSON with `Graph::to_json()`. This is the same
   converter that the suite's graph capture uses. No GPU is needed.

The graphs exported for gfx942 and for gfx950 are byte-identical for every
bundle row, because the rows that depend on the GPU are excluded
(`arch_dtype`).

### 9.3 Topologies and sweeps

A sweep cannot add or drop a tensor. So the cases of one sweep must have the
same set of tensors. The **topology** of a row is that set, plus the form of
each scalar tensor (passed by value at build time, or supplied at run time).

The topology name lists the optional tensors in a fixed order, one token
each. A row with only Q, K, V and O is `Base`. Examples:

| Topology | Tensors beyond Q, K, V, O |
|---|---|
| `Stats` | the per-row log-sum-exp output |
| `Bias` | an additive bias |
| `ScRt` | a scale supplied at run time |
| `DqBv_DkBv_DvBv` | fp8 descales of Q, K and V, passed by value |
| `Stats_DqBv_DkBv_DvBv` | Stats plus the three descales |

A name longer than 64 characters keeps its first tokens and ends with 6 hex
digits of a hash. The 16,729 rows have 405 topologies; the bundle rows use 109.
One sweep holds the cases of one tier and one topology.

The suite's import tool `migration-scripts/import_graph.py` groups the graphs.
Its default behaviour does not change. The grouping uses opt-in flags
([Grouped import for generated cases](../../migration-scripts/README.md#grouped-import-for-generated-cases-opt-in)):

- `--op-dir SdpaFwdPlan --topology <Topology>` name the folders.
- `--template-attributes` and `--template-alignment` let node attributes and
  tensor alignment vary from case to case in one sweep.
- `--skeleton-value-form` keeps scale forms (by value or at run time) in
  different sweeps.
- `--only-op-dir` and `--require-marked-sweep` keep these cases out of every
  other sweep, and other cases out of these sweeps.
- `--case-id` sets a stable case id.

Each grouped sweep records the flags in an `import_mode` object in its
`sweep.json`. The default import never appends into such a sweep.

These sweeps use the compact layout. Each child of `values` and each sibling
of `values` is on its own line, and each tensor is on its own line.
`import_mode.format` records the layout, and an append keeps it.

### 9.4 Ids, names and metadata

- **Case id**: `<dtype_q>_<case_key>`, for example `bf16_e50bd5c4cdf64be7`.
- **GTest name**: `<tier>_SdpaFwdPlan_<Topology>.<case id>`, for example
  `quick_SdpaFwdPlan_Base.bf16_e50bd5c4cdf64be7`.
- **Metadata** of each case: `format_version`, `generator`,
  `generator_version`, `reference_source`, `seed`, `case_key`, `csv_case_id`
  and `generated_from_sha256` (the sha256 of the CSV before the appended
  columns). Use `case_key` to find the row in `cases.csv`.
- **Seed**: the input seed of a case is the first 8 hex digits of its
  `case_key`, masked to 31 bits. Example: `case_key` `a69ef8a0…` gives seed
  0x269ef8a0 = 647,952,544.
- **Inputs**: the suite fills the input tensors. A scale that is supplied at
  run time gets a fixed fill in `metadata.inputs`
  (`{"kind": "fixed", "value": …}`).

The bundles have no golden data. The suite checks each output against the
hipDNN GPU and CPU SDPA reference executors.

### 9.5 Lanes

A **lane** is one engine's run of the suite. The `ASM_SDPA_ENGINE` lane selects
every op folder whose name contains `Sdpa`, so it runs these cases. The
HIP_MLOPS, MIOPEN and hipBLASLt lanes have no SDPA support. Their tier YAML
files exclude `*_SdpaFwdPlan_*`, so they do not load these cases only to
decline them.

## 10. Known engine defects and their skips

Some cases are accepted by `ASM_SDPA_ENGINE` but give a wrong output. An
accepted case with a wrong output fails, whether or not a claim covers it. So
each known defect gets a skip in
[`ASM_SDPA_ENGINE.toml`](../../../hip-kernel-provider/config/ASM_SDPA_ENGINE.toml).
The cases stay in their tier. Each skip's `reason` starts with a tag and a
ticket key: `[SKIP-Tn] ALMIOPEN-NNNN: <defect>`.

| Tag | Ticket | Defect | Bundle rows that can reach it |
|---|---|---|---|
| `[SKIP-T2]` | ALMIOPEN-3042 | Zero or overlapping sequence strides on Q, K or V give a wrong O. | `strides_q`, `strides_k` or `strides_v` = `self_overlapping` |
| `[SKIP-T3]` | ALMIOPEN-3042 | Stats is written with only the head stride. | `stats_output` = `requested` with non-packed `strides_stats` |
| `[SKIP-T4]` | ALMIOPEN-3124 | Max and Sum_exp are accepted but never written. | `max_sumexp_output` other than `not_requested` |
| `[SKIP-T5]` | ALMIOPEN-3125 | FP8_E4M3_FNUZ gives a wrong O on gfx942. | `dtype_q` = `fp8_e4m3_fnuz`, on gfx942 |
| `[SKIP-T6]` | ALMIOPEN-3126 | The RNG dump is accepted at p = 0 but never written. | random dropout, `dropout_probability` = `zero`, `rng_dump` requested |
| `[SKIP-T7]` | ALMIOPEN-3043 | A scale of zero or below gives NaN. | `scale_value` = `zero` or `negative` |
| `[SKIP-T9]` | ALMIOPEN-3041 | Any head-dimension stride is accepted and read as 1. | No bundle row varies the head-dimension stride. |

Other known defects that no bundle row reaches:

- ALMIOPEN-3123: a caller-supplied dropout mask is ignored. Rows with a
  caller mask are excluded (`dropout_mask_cycle`).
- ALMIOPEN-3127: a block mask is ignored. Rows with a block mask are excluded
  (`block_mask_bits`).
- ALMIOPEN-3128: a decline gives no reason. This is in the hipDNN backend; a
  test cannot detect it.

A skip runs before the support query, so a skipped case is never observed and
never claimed. When a fix lands (the engine computes the case correctly, or
declines it), the change that lands the fix deletes the matching skip entry.

## 11. Limits and what comes next

- The cases come from a PoC. The labels, the tier recipe and the budgets are a
  draft for review.
- Only `ASM_SDPA_ENGINE` claims these cases, on gfx942 and gfx950 (Linux).
- 6,846 rows are not bundles, mostly because they need exact tensor contents.
  They stay in the CSV with their reasons.
- The rejection probes and the API-path variants in `injections.yaml` are not
  bundles. They belong to a rejection suite.
- A sweep of awkward sizes (sizes that do not fit tile edges) is not part of
  this plan.
- The suite is expected to read the CSV directly in a later step. The bundles
  under `SdpaFwdPlan` will then be removed.

## 12. How to regenerate

The generator and the graph exporter are local PoC tools. They are not in the
repository. These are the steps that made the committed files:

1. **Rows and tiers.** The case generator reads `model.yaml`, `rules.yaml` and
   `injections.yaml` with seed 2893. Its level step adds the tiers, with one
   per-case timing file for gfx942 and one for gfx950. The output is the CSV
   without the five appended columns.
2. **Selection.** A filter applies the reasons of section 9.1. Two parameters
   control which rows become bundles: the tiers that get bundles (default: all
   four) and the smallest sweep size (default: 1). A switch controls the
   zero-stride filter (default: on).
3. **Graphs.** The PoC graph exporter writes one graph JSON and one metadata
   JSON per selected row, with no GPU.
4. **Bundles.** For each graph, sorted by tier, topology and `case_key`, run
   `migration-scripts/import_graph.py` with `--tier <tier>`,
   `--op-dir SdpaFwdPlan`, `--topology <Topology>`, `--case-id <id>`, and the
   grouped-import flags of section 9.3.
5. **CSV.** Append the five columns of section 8.

Checks:

- The generator output must equal `cases.csv` with the last five columns
  removed, byte for byte.
- Two bundle runs into two copies of the tree must give identical trees.
- After a change to the bundle tree, run `scripts/verify_support_claims.py`
  and regenerate `SUPPORT_MATRIX_OVERVIEW.md` with
  `scripts/support_features/feature_matrix.py`
  ([Support Matrix](../support-matrix.md)).
