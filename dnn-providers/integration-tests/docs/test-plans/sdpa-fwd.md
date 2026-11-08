# SDPA Forward Test Plan: How the Cases Were Derived

> A proof of concept (PoC) generated these test cases. A PoC is a first working
> version, made to show that a method works. The cases have already found defects
> in SDPA engines. So we use them as a first set while the method is reviewed and
> finalized.

This document tells how the SDPA forward plan cases were made. It starts from
everything the operation could vary. It ends with the bundles that the
integration suite runs. Each step removes cases for a stated reason, so you can
check every step.

The files of the plan in this repository are:

| File | What it holds |
|---|---|
| [`test-plans/sdpa-fwd/model.yaml`](../../test-plans/sdpa-fwd/model.yaml) | The model: the settings, their values, the labels, the tier recipe. |
| [`test-plans/sdpa-fwd/rules.yaml`](../../test-plans/sdpa-fwd/rules.yaml) | The rules that remove combinations. |
| [`test-plans/sdpa-fwd/injections.yaml`](../../test-plans/sdpa-fwd/injections.yaml) | The cases that are added on purpose, outside the generated sets (section 12). |
| [`test-plans/sdpa-fwd/layers/`](../../test-plans/sdpa-fwd/layers/) | The layer files: Layer 0 (the axes), Layer 1 (the definition rules) and Layer 2 (the frontend rules). |
| [`docs/test-plans/sdpa-fwd/axes.md`](sdpa-fwd/axes.md) | The axis catalogue: every axis, what it is for and its full values. |
| `integration-test-bundles/<tier>/SdpaFwdPlan/<Topology>/` | The bundles: one sweep per tier and topology (section 9). |
| [`ASM_SDPA_ENGINE.toml`](../../../hip-kernel-provider/config/ASM_SDPA_ENGINE.toml) | The skips of the cases that hit known engine defects (section 10). |
| The `*test_categories_integration.yaml` file of each lane | Which lanes run the cases (section 9.5). |

Three programs made the cases:

- The **case generator** reads three plan input files and writes the **case
  list**: a CSV file with one row per case (section 8). The input files are
  `model.yaml` (the settings, their values, the labels and the tier recipe,
  section 4), `rules.yaml` (the rules that remove combinations, section 5) and
  `injections.yaml` (the cases that are added on purpose, outside the generated
  sets).
- The **graph exporter** turns each row of the case list into a hipDNN graph
  (section 9.2).
- The **bundle tools** choose the rows that a bundle can hold and write the
  bundles (sections 9.1, 9.3 and 9.4).

These programs are a PoC. They are still being developed, so they are not in
the repository. Neither is the case list. Until they are, the bundles are the
record of the cases, and this document tells how the plan input files became
the bundles.

### Words used in this document

| Word | Meaning |
|---|---|
| tensor | A multi-dimensional array of numbers, for example a 4-D array of size batch × heads × length × head size. |
| CSV | A text table: one line per row, with commas between the fields. |
| YAML, JSON, TOML | Three text formats for structured data. The plan input files are YAML, the graphs are JSON, the engine configuration files are TOML. |
| SHA-256 | A 64-digit hex number computed from the bytes of a file. If one byte changes, the number changes. Two files with the same SHA-256 are the same file. |
| hex | Base-16 numbers, written with the digits `0`-`9` and `a`-`f`. |
| arch | The GPU architecture name, for example `gfx942` or `gfx950`. |
| fp8, bf16, fp16, fp32 | Number formats of 8, 16, 16 and 32 bits. fp8 has variants, for example E4M3 (4 exponent bits, 3 mantissa bits). |
| descale | A factor that turns stored fp8 numbers back into their real values. |
| BHSD, BSHD | The order of the dimensions of a tensor in memory: B batch, H heads, S sequence length, D head size. |
| ragged | A tensor that packs sequences of different lengths one after the other. Offsets tell where each sequence starts. |
| CTest, GTest | CTest is the test runner of CMake. GTest (GoogleTest) is the C++ test framework of the suite. |
| plugin | A shared library that hipDNN loads at run time. A provider plugin holds one or more engines. |

## Contents

1. [Why this plan exists](#1-why-this-plan-exists)
2. [The theoretical base](#2-the-theoretical-base)
3. [Narrowing: meaning, then the frontend](#3-narrowing-meaning-then-the-frontend)
4. [The model](#4-the-model)
5. [The rules file](#5-the-rules-file)
6. [Generating the cases](#6-generating-the-cases)
7. [Tiers](#7-tiers)
8. [The case list](#8-the-case-list)
9. [From rows to bundles](#9-from-rows-to-bundles)
10. [Known engine defects and their skips](#10-known-engine-defects-and-their-skips)
11. [Limits and what comes next](#11-limits-and-what-comes-next)
12. [The injected cases](#12-the-injected-cases)

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
with the case generator. A person does not pick cases one by one. To change the
list, change the model or a rule, and generate the list again.

## 2. The theoretical base

The first step lists everything the forward operation could vary, with no
thought about what makes sense or what hipDNN can express. This list is
**Layer 0**.

- [`axes.md`](sdpa-fwd/axes.md) gives every axis and why: what it is for, its
  full values and why some values are chosen.
- [`layers/L0-axes.yaml`](../../test-plans/sdpa-fwd/layers/L0-axes.yaml) holds
  the same axes in machine-readable form.
- [`layers/L1-definition.yaml`](../../test-plans/sdpa-fwd/layers/L1-definition.yaml)
  holds the narrowing by meaning (section 3).

Terms:

- An **axis** is one thing that a case can vary. Example: `dtype_q`, the number
  format of Q.
- A **value** is one choice on an axis. Example: `bf16`.
- A **child axis** exists only when its parent axis has a given value. Example:
  `causal_alignment` (which diagonal the causal mask uses) exists only when
  `causal_mask` is `present`.
- A **combination** is one value for every active axis.

Layer 0 has 105 axes and 379 values. That gives about 4.5 × 10⁴⁸ combinations.
Nobody can run that many cases. So we count **pairs**: two values of two
different axes, for example `dtype_q = fp16` together with
`causal_mask = present`. A pair is **valid** when at least one allowed
combination holds it. Layer 0 has 70,613 valid pairs.

**Pairwise testing** puts every valid pair in at least one case. Most defects
need only one or two settings in a bad state, so pairwise testing finds most of
them with a small number of cases. For this reason, the plan measures its size
in pairs, not in combinations.

The layer numbers in sections 2 and 3 come from the layer files in
[`test-plans/sdpa-fwd/layers/`](../../test-plans/sdpa-fwd/layers/). A script
that is not in the repository counted them by arithmetic: it enumerates the size
axes, splits the other axes into independent groups, and multiplies the counts
of the groups. It splits a large group again by fixing one axis at a time. It
does not list the combinations one by one.

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
- `D-RAGGED-NOT-THROUGH-PAGES`: a ragged start position on a paged K or V. No
  specification says whether it counts tokens before or after paging, or
  whether a length counts tokens or pages. A ragged Q over a paged K and V
  keeps its meaning.

Layer 1 has 35 definition rules. They leave about 5.8 × 10⁴² combinations and
63,206 valid pairs.

**Layer 2 removes combinations that hipDNN's frontend cannot express.** These
are **frontend** rules (ids `F-*`). The frontend is the hipDNN C++ API that
builds the graph. Examples:

- `F-EXTENT-NONZERO`: every tensor dimension must be at least 1, so an empty
  batch cannot be stated.
- `F-ONE-DIAGONAL`: the node has one diagonal alignment, so a causal mask and a
  sliding window cannot use different diagonals.
- `F-SCALE-TENSOR-BY-VALUE`: a scale tensor must be passed by value, so a
  scale in device memory cannot be stated.

Layer 2 has 17 frontend rules. They leave about 3.2 × 10³⁷ combinations and
52,348 valid pairs.

Some values and combinations have meaning, but the frontend cannot state them,
refuses them, or is to refuse them. Layer 2 removes them. The model does not
hold the values of `F-INDEX-TENSORS-INT32`. It holds the values of the other
rules below, and the rule removes them from every row.
`F-PAGED-NEEDS-SEQ-LENS` removes a combination, not a value:

- `F-SINK-FLOAT-ONLY`: the frontend takes only an fp32 sink, so a sink in any
  other format cannot be stated. This is a possible frontend defect.
- `F-INDEX-TENSORS-INT32`: the length tensors and the page tables must be
  32-bit integers.
- `F-NO-CU-SEQ-LEN`: no API takes cumulative sequence lengths (running totals
  of the lengths).
- `F-BLOCK-MASK-TILE-128`: the block mask holds one flag per 128 × 128 score
  tile, so a mask with another tile size, or with one flag per score, cannot be
  stated.
- `F-LEFT-BOUND-NOT-ZERO`: a left window bound of 0 is refused.
- `F-PAGED-NEEDS-SEQ-LENS`: a paged K or V needs both length tensors.
- `F-NO-NEGATIVE-STRIDE`: a negative stride. The frontend accepts it today, but
  hipDNN intends to refuse it (ticket ALMIOPEN-3114). The rule is marked "not
  enforced yet".

Some combinations are valid but no reference executor can check them. They are
not Layer 1 or Layer 2 rules. The answer key (section 4) holds them. Example: the
dropout pattern output with a caller-supplied mask.

**Hints for a fuzzer.** Some Layer 0 axes describe the numbers inside tensors:
special numbers such as NaN or infinity (`special_values`), numbers large
enough to overflow (`value_magnitude`), sink logits (`sink_values`), descale
and scale factors (`scale_factor_values`), and a scale that is NaN or infinite
(`scale_value` = `non_finite`). They are in Layer 0 and in the model, but they
are not for test-case generation. They are hints for a fuzzer. The choice rule
`CH-FUZZER-HINTS` keeps every case at the axis's ordinary value (`ordinary` or
`none`) and never uses `scale_value` = `non_finite`.

A removed combination is not lost. When the frontend accepts a meaningless
request, that request comes back as a **rejection probe** in
`injections.yaml`: a case that expects hipDNN to refuse it. Rejection probes
are not part of this plan's bundles. Section 12 explains how to read that
file and where it records why each case is there.

## 4. The model

The model (`model.yaml`) is the Layer 2 space in a form that the case
generator can read. It has 92 axes (39 of them child axes) and 339 values.

**Classes, not numbers.** Most size axes hold a **class**, chosen by meaning.
Example: `batch_size` is `one` or `many`. A **relationship** axis says how two
sizes relate. Example: `seq_q_vs_seq_kv` is `shorter`, `equal` or `longer`.
The **class-to-value table** (`class_to_value` in `model.yaml`) turns each
class into numbers. Example: `k_head_sharing = gqa` means the K head count is
the query head count divided by 4, at least 2.

Rules hold only for the values that were tested. A result at head size 128
says nothing about head size 80.

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

**Inputs that share memory.** With `tensor_overlap = inputs_overlap`, one input
is a view into another input's buffer: V inside K, or K inside Q when V's head
size is larger. The view reads through its owner's strides, so its layout
values (`strides_*`, `dim_order_*`) are `inherited`. Two rules keep this true:
`D-INPUTS-OVERLAP-VIEW-DTYPE` (the view has its owner's format) and
`D-INPUTS-OVERLAP-VIEW-LAYOUT` (only the view is `inherited`). The graph
exporter gives the view no buffer of its own when the batch is padded, K and V
are not paged, and every dimension of the view fits inside its owner's; a
bundle then records the view in its metadata `aliases`. Otherwise the row is
not a bundle (section 9.1).

**Numeric domains.** `numeric_domains` gives, for each size axis, the smallest
and largest size the frontend takes, the range of each class, and boundary
values (powers of two, multiples of 16 to 128, and the values next to them).
Each entry cites its source: the frontend code where the frontend sets a limit,
otherwise the rule or entry it follows.

**Bridges.** `level_labels.bridges` (B1 to B14) name pairs of axes in different
groups that still interact. Example: B3, "element size changes byte strides and
alignment".

**Other added cases.**

- **Targeted cases** (`level_labels.targeted`, X2 to X11c): the base case plus a
  few changes, each aimed at one known hazard. Example: X6b, "a scale tensor may
  be ignored (scale 1.0 used)". A targeted case that the references cannot check
  (an answer-key rule matches its row) goes to full. X3c is such a case.
- **Real-world cases** (`level_labels.real_world`, 40 entries): settings of
  public models, for example Llama 3 8B decode and Mistral 7B sliding window,
  and 11 ragged (varlen) cases in quick, most of them token-major: vLLM-style
  serving (ragged queries over a paged GQA cache), varlen prefill with one
  token-count offset list per side, with GQA and bottom-right causal, a sliding
  window, a bias, longer slots than sequences, INT64 offsets, a packed Stats,
  ragged queries over dense K/V, fp16, and one head-major case for contrast;
  and 16 more public-model shapes at their real head counts and head sizes
  (Llama 3.1 and 3.2, Qwen2.5, Phi-3 and Phi-4, Gemma 3, DeepSeek-V3 including
  MLA decode with head sizes 576 and 512, gpt-oss) and two paged serving
  defaults.
- **Seeds** (`level_labels.seeds`, 15 entries): combinations that public
  defect reports or serving workloads showed to matter. Example: S1, Stats with
  ragged queries over a paged GQA decode.
- **Coverage rows** (origin `coverage`): every value whose answer key is not
  "no" is in a row that the references can check. If no other row holds a value
  that way, the case generator adds the checkable row with the fewest changes
  that holds it, in full at shape tier S.

## 5. The rules file

`rules.yaml` holds every rule. The case generator applies all rules at once;
their order has no effect. Each rule has a `kind`:

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

The model applies 46 exclusion rules: 33 definition, 12 frontend and 1 choice.
Together they make 14,226 of the 58,766 structurally possible pairs invalid.
44,540 valid pairs remain. One more definition exclusion,
`D-RAGGED-NO-OVERLAP`, removes nothing: the value it forbids is not in the
model.

Example of a `forbid`:

```yaml
  - id: D-REL-WINDOW
    forbid_any:
      - {seq_kv: [one], window_left: [partial]}
      - {seq_kv: [one], window_right: [partial]}
```

With one key, a window cannot cover "some but not all" keys, so these rows
are removed.

The frontend rule `F-NO-NEGATIVE-STRIDE` removes negative strides. It names
ticket ALMIOPEN-3114, which asks hipDNN to refuse these requests. The requests
come back as rejection probe MV-22.

## 6. Generating the cases

The case generator builds rows. A **row** is one value for every active axis. A
**case** is a row at one **shape tier** (one set of concrete sizes, section 7).
One row can give several cases.

The case generator uses these methods:

- **Low-change pairwise.** Each row starts from the base case and changes as few
  settings as possible. First, the case generator covers every pair that a row
  with at most 3 changes can hold. Then it covers the pairs that need 4
  changes, then 5, and so on, until every pair is covered. A child axis that
  its parent switches on does not count as a change. Example: base case plus
  `bias = present` plus `alignment_k = element_aligned` plus
  `alignment_o = element_aligned` is 3 changes; `bias_broadcast` and
  `bias_dtype` follow from `bias = present` and do not count. A failure in a
  low-change case is easy to trace to one setting.
- **Packed pairwise.** Each row holds as many uncovered pairs as possible.
  This gives fewer rows, but each row changes many settings. An engine that
  cannot handle one of them declines the whole row. Every tier uses it only
  for the pairs that no bundle can hold (see "Rows a bundle can hold").
- **One-at-a-time.** The base case with exactly one value changed (plus any
  value that a rule forces).
- **Targeted, real-world and seed rows** (section 4). The case generator
  completes each one with the fewest extra changes that obey the rules.

**Token-major ragged rows.** A ragged tensor is usually stored token-major
(`bshd`, the layout varlen attention calls THD). In a row with a ragged Q side,
`dim_order_q = bshd` and `dim_order_o = bshd` count as no change, as the base
value `bhsd` does. The same holds for K and V on a ragged K/V side. A search
for the fewest changes tries `bshd` first in such a row. This applies to the
low-change rows, the size-sweep crossings and the coverage rows. Packed rows
take any order.

**Rows a bundle can hold.** Some rows need a harness capability before a
bundle can hold them, for example an output written over an input, or random
dropout without the dumped mask. Every tier builds bundle rows first: its
low-change steps take only the pairs that a row with no such need can hold,
and build only such rows. So each of those pairs is in a row that a bundle can
hold. Then packed pairwise covers the tier's other pairs, for example a ragged
setting with a mask, in a few rows. When two values cost the same number of
changes, the case generator takes the value with no need first.

**Super-setting.** In the quick tier, the data-type axes (`dtype_q`,
`dtype_k_vs_q`, `dtype_v_vs_q`, `dtype_o`, `dtype_stats`, `compute_precision`)
act as one setting. Each complete data-type combination counts as one change
and pairs with every other value that quick allows. Quick holds 12 such
combinations.

**Warm start.** A tier asks only for the pairs that lower tiers did not cover
already.

**Determinism.** The case generator uses seed 2893. The same model, rules,
timings (section 7) and seed give the same case list, byte for byte.

## 7. Tiers

The integration suite has four **tiers**: `quick`, `standard`, `comprehensive`
and `full`. The tiers are **cumulative**: a run of `standard` also runs every
`quick` case. A case's tier is the first tier that needs it.

The tier of each case comes from the labels and from a short **recipe**
(`levels:` at the end of `model.yaml`). The tiers are a draft; the labels and
the recipe are open to review.

| Tier | Values it takes | Must every case be checkable? | Pairs | Extra rows | Shape tiers | Per-case ceiling | Budget |
|---|---|---|---|---|---|---|---|
| quick | `usage` common; `changes_numbers` not "no"; answer key yes or partial | yes | all, bundle rows first, low-change; packed for the rest | real-world rows and checkable seeds | S | 20 MFLOP | 5 min |
| standard | as quick, plus `usage` rare | yes | all, bundle rows first, low-change; packed for the rest | targeted | S, H, M | 200 MFLOP | 30 min |
| comprehensive | as standard | yes | within groups and bridges, bundle rows first, low-change; packed for the rest | one-at-a-time at S and L | S, H, M, L (one per row) | 8,000 MFLOP | 60 min |
| full | every `usage` (also `fringe`) and every `changes_numbers`; answer key yes or partial | yes | all, bundle rows first, low-change; packed for the rest | one-at-a-time at S and L; uncheckable singles at S; the base case at X | S, H, M, L, X | none | none |

**Uncheckable singles.** A case is uncheckable when it holds a value whose
answer key is "no", or a combination of an answer-key rule
(`level_labels.answer_key_needs`). No reference can compute its expected output.
No tier pairs such a value or combination with other settings. Instead, full
holds one case for each of them, at shape tier S. A single holds only its own
uncheckable reason when a valid row can. A target that no valid row can hold
has no single. The harness reports a single that an engine accepts as
Unverifiable (section 10.5).

**Size sweep.** Each tier names the size-sweep rows it takes (`size_sweep`
in `levels:`). quick takes the typical sizes, which are most likely to pass.
standard takes the corner sizes, which are most likely to find a defect: on the
plain case, and crossed with the settings whose `size_sensitive` label is
"yes". comprehensive takes standard's crossings again at shape tier L. full
takes the stress sizes of shape tier X.

The case generator derives the sweep sizes from `numeric_domains`,
`size_sweep`, the shape tiers and `class_to_value`. Each size is one row
(origin `size-sweep`): the plain row, or for a crossing the row with the fewest
changes that holds the crossed value, with one size changed. The row starts
from the sizes of shape tier S (tier L for comprehensive), and its class
columns follow its sizes. The `sizes` column gives every size. The sizes
override sets the batch, the head counts, the sequence lengths and the head
sizes. Page size and the window bounds get only a stress row: the class value
that reaches the key length, at the key length of shape tier X.

Corner sizes sit next to a representative size or, for the sequence lengths
and the head sizes only, next to a tile edge (64, 128, 256): the edge and the
sizes one below and one above it (`size_sweep.corner.tile_edge_axes`). The
batch and the head count keep small corner sizes. Each crossed setting value
takes 3 entries of the interleaved corner list, and the next value starts 3
entries further on, so the values share out every corner size. This is why the
bundles hold odd head sizes such as 63, 65, 95, 97, 127 and 129, and a Q / K
head size that differs from V's.

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
- **Fill.** If a tier has spare time, the case generator pulls cases down from
  higher tiers until the tier uses 50 % of its budget. A pulled case must fit
  the tier's shapes, ceiling, labels and change limit.
- **Cost.** The time of a case was measured on one gfx942 node and one gfx950
  node by a **case runner**, a PoC program that runs one case on one engine and
  on a reference. A case that was not measured gets a fitted estimate:
  a + b × MFLOP + c × tensor-MB. Each tier run adds 10 s of overhead. gfx942 is
  the budget basis.
- **Over budget.** The case generator only reports; it does not move cases.

Result for the case list (section 8; times are the summed `est_seconds`
columns plus the overhead):

| Tier | Cases in this tier | Cumulative cases | Cumulative s, gfx942 | Cumulative s, gfx950 | Budget use, gfx942 |
|---|---|---|---|---|---|
| quick | 1,506 | 1,506 | 63.3 | 51.1 | 21.1 % |
| standard | 9,555 | 11,061 | 580.7 | 436.6 | 32.3 % |
| comprehensive | 4,643 | 15,704 | 2,333.2 | 1,685.0 | 64.8 % |
| full | 11,545 | 27,249 | 7,043.4 | 5,052.0 | — |

These times are the case runner's times for one engine run plus one reference
run. The suite does more work per case, so its times are larger.

**Seeds.** The case generator completes each of the 15 seeds with the fewest
changes under the rules and the answer-key rules. A seed whose row the
references can check goes to quick, at shape S. Any other seed goes to full.
Whether the references can check a seed's row is decided by the labels of
quick, standard and comprehensive (full's uncheckable singles do not count).
14 seeds are in quick; `S3b-vllm-fp8kv-paged` is in full, because a K or V
format that differs from Q has answer key "no". In the case list a seed row has
`origin` `real_world` and `detail` `seed:<id>`. When a seed's row equals a row
that the tier already holds, the two merge, and that row's `detail` carries
the seed name.

## 8. The case list

The case generator writes the case list as a CSV file. The case list is not in
the repository (see the start of this document). This section describes it,
because the bundles' metadata points into it (section 9.4) and because the
counts in sections 7 and 9 come from it.

The same input files and the same seed always give the same file, byte for
byte. The case list of these bundles has the SHA-256
`4c7e4621b20689d1c353ea41cad59fa65a8158d9f86fe0692881da130848bc81`. Each case
records it in its metadata `generated_from_sha256`.

The case list has one row per case: 27,249 rows. It has no quoted fields. A
`-` means that a child axis is not active in that row. Each row has 105
columns: 13 case columns, then 92 axis columns.

**Case columns (13).**

| Column | Meaning |
|---|---|
| `case_id` | `lv-NNNN`, the row number. A change to the model can shift it. Do not use it as a key. |
| `level` | The tier: `quick`, `standard`, `comprehensive` or `full`. |
| `shape_tier` | `S`, `H`, `M`, `L` or `X`. |
| `origin` | `low-change`, `pairwise` (packed), `one-at-a-time`, `targeted`, `real_world`, `size-sweep` or `coverage`. |
| `detail` | For low-change rows: the number of changes. For size-sweep rows: the label, the size and the crossed value. For coverage rows: the value. For uncheckable singles: `uncheckable <axis>=<value>` or `uncheckable <rule id>#<index>`. For other rows: the id of the targeted case, real-world case or seed. |
| `filled_from` | Empty, or the tier that the recipe gave the case before a fill pulled it down. |
| `est_seconds` | The time of the case on gfx942. |
| `cost_source` | `measured`, `modelled` or `UNMEASURED` (outside the range of the fitted estimate). |
| `work_mflop` | The work of the case. |
| `tensor_mb` | The memory of the case's tensors. |
| `sizes` | The concrete sizes, for example `B2 Hq8 Hk1 Hv1 Sq64 Skv64 Dqk128 Dv128`. |
| `case_key` | 16 hex digits: a hash of the tier, the sizes and every axis value. It is unique per row and stays the same when `case_id` values shift. Use it as the key. |
| `est_seconds_gfx950` | The time of the case on gfx950. |

**Axis columns (92).** One column per model axis, from `batch_packing` to
`value_magnitude`, in the order of `model.yaml`.

## 9. From rows to bundles

A **bundle** is a graph test in `integration-test-bundles/`. A **sweep** is a
`graph.template.json` that the suite expands once for each case in its
`sweep.json` ([File Formats](../file-formats.md)).

### 9.1 Which rows a bundle can express

A bundle holds a graph and, for each case, a few fixed values. Some tensor
contents are also bundle contents: page tables and the offsets of a packed page
table, ragged offsets, `seq_len` values, block masks, descale lists, a caller's
dropout mask, and dropout seeds and offsets held in device memory. A row that
a bundle cannot express or compare stays in the case list, and the bundle tools
skip it for the reasons below.

The first two reasons say that a bundle cannot express or compare the row. The
last one is an extra exclusion: the row could be a bundle, but the bundle tools
leave it out. The Rows column counts every row that has the reason. A row can
have several reasons, so the counts overlap and do not add up to the number of
excluded rows.

| Reason | Why the row is not a bundle | Rows |
|---|---|---|
| `tensor_overlap` | An output written over an input, inputs that share memory on a row with paged K and V, and inputs that share memory on a ragged row need control of the pointers that a bundle does not have. Inputs that share memory on a padded, non-paged row are bundle contents (metadata `aliases`). The case generator holds no view larger than its owner (rule `D-REL-VIEW-FITS-OWNER`) and no inputs that share memory on a ragged row (rule `CH-RAGGED-SHARED-MEMORY-LATER`). | 57 |
| `ragged_query_without_tokens` | A ragged Q gives no query token to compare: every sequence length is 0, or the lengths come from the Q offsets and a zero sequence stride makes every offset 0. Its outputs have no element to compare, and a comparison that compares no element passes without a check. | 56 |
| `dropout_without_rng_dump` | Random dropout without an `rng_dump` output has no oracle: the references check random dropout only through the dumped mask. | 4 (4 otherwise expressible) |

27,132 of the 27,249 rows became bundles and 117 rows are excluded. Of the
excluded rows, 113 have at least one of the first two reasons. The other 4
could be bundles but have only dropout without `rng_dump`; 3 of them are
uncheckable singles.

A data start past an aligned allocation start (`element_aligned`,
`sub_element`) is a bundle content: the case's metadata `start_offsets` gives
the offset of each such tensor. When the case has per-sequence query lengths in
a `seq_len_q` tensor, the metadata `live_rows` marks the dense Stats, logit-max
and sum-of-exponents outputs, so that a comparison skips their rows past each
sequence's length.

The bundle tools never change the model or the rows of the case list. They only
filter. To bring a row back, the filter of the bundle tools must change and the
bundles must be made again.

| Tier | All rows | Bundle rows | Sweeps | Largest sweep | Summed `est_seconds` of the bundle rows, gfx942 / gfx950 |
|---|---|---|---|---|---|
| quick | 1,506 | 1,500 | 87 | 492 | 53.1 / 41.0 |
| standard | 9,555 | 9,519 | 231 | 2,584 | 515.8 / 384.2 |
| comprehensive | 4,643 | 4,622 | 203 | 1,237 | 1,745.8 / 1,243.7 |
| full | 11,545 | 11,491 | 294 | 2,452 | 4,693.0 / 3,354.7 |
| total | 27,249 | 27,132 | 815 | | |

### 9.2 How a graph is made from a row

The graph exporter makes the graphs. The `generator` field in each case's
metadata holds the name of this program (`hipdnn_sdpa_case_flow_runner`), and
`generator_version` its version.

1. The graph exporter reads the row and the class-to-value table, and makes
   concrete sizes, strides, formats and attributes.
2. It builds the graph with the hipDNN frontend and calls `validate()`.
3. It serializes the graph to JSON with `Graph::to_json()`. This is the same
   converter that the suite's graph capture uses. No GPU is needed.

The graphs exported for gfx942 and for gfx950 are the same for every bundle
row once the bundle import renumbers the tensor uids: each number format is the
same on every GPU (`class_to_value` in `model.yaml`). So the FNUZ fp8 formats
are their own values of `dtype_q`, and every GPU loads every case.

### 9.3 Topologies and sweeps

The **topology** of a row is the set of its tensors, plus the form of each
scalar tensor (passed by value at build time, or supplied at run time) and
whether Stats is ragged.

The topology name lists the optional tensors (every tensor except Q, K, V and
O) in a fixed order, one token each:

| Token | Tensor |
|---|---|
| `RgQ`, `RgK`, `RgV`, `RgO`, `RgSt` | the ragged offsets of Q, K, V, O and Stats, one list each |
| `RgQs`, `RgKvs` | one ragged offset list for the whole Q side or the whole K/V side |
| `SlQ`, `SlKv` | the per-sequence lengths of the query side and of the key side |
| `Bm` | a block mask |
| `Bias` | an additive bias |
| `Sink` | sink logits |
| `ScBv`, `ScRt` | the attention scale, passed by value or supplied at run time |
| `DSeed`, `DOff`, `DSeedBv`, `DOffBv` | the dropout seed and offset, in device memory or passed by value |
| `DMask` | a caller-supplied dropout mask |
| `DScBv` | the dropout scale, passed by value |
| `Rng` | the dumped dropout pattern (`rng_dump`) |
| `Stats`, `StatsRg` | the per-row log-sum-exp output, dense or ragged |
| `Max`, `SumExp` | the per-row logit maximum and sum of exponents |
| `AmaxS`, `AmaxO` | the absolute maximum of the scores and of O (fp8) |
| `Dq`, `Dk`, `Dv` | descale lists of Q, K and V in device memory |
| `DqBv`, `DkBv`, `DvBv`, `DsBv` | descales of Q, K, V and the scores, passed by value |
| `SsBv`, `SoBv` | the fp8 scale of the scores and of O, passed by value |
| `PtK`, `PtV`, `PtKv` | the page tables of K, of V, or one table shared by K and V |
| `PtKOff`, `PtVOff` | the offsets of a packed page table of K or of V |

A name longer than 64 characters keeps its first tokens and ends with 6 hex
digits of a hash.

A sweep cannot add a tensor, but a case can leave tensors out: its
`absent_tensors` lists the template tensors that its graph does not have
([File Formats](../file-formats.md)). So one sweep's template holds the union of
the tensors of several topologies. The bundle tools group per tier: they sort
the topologies largest first and put each one into the smallest earlier
superset that can hold it. A superset can hold a topology when the topology's
tensors are a subset of its tensors and the two have the same scale form, the
same dropout input form, the same page tables and the same ragged offsets. A
topology that fits none starts a new superset. Every superset is the topology
of a row, and the sweep folder takes its name. So there is no `Base` folder:
the rows with only Q, K, V and O run in sweeps whose other tensors are absent
for them. The 27,132 bundle rows fill 815 sweeps under 307 folder names; 23,946
cases name absent tensors.

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

After a change to the bundle tree, run `scripts/verify_support_claims.py` and
regenerate `SUPPORT_MATRIX_OVERVIEW.md` with
`scripts/support_features/feature_matrix.py`
([Support Matrix](../support-matrix.md)).

### 9.4 Ids, names and metadata

- **Case id**: `<dtype_q>_<case_key>`, for example `bf16_0b94e9a2e5764569`.
- **GTest name**: `<tier>_SdpaFwdPlan_<Topology>.<case id>`, for example
  `quick_SdpaFwdPlan_Bias_Sink.bf16_0b94e9a2e5764569`.
- **Metadata** of each case: `format_version`, `generator` (the name of the
  graph exporter), `generator_version`, `reference_source`, `seed`,
  `case_key`, `csv_case_id`, `generated_from_sha256` (the SHA-256 of the case
  list) and `inputs`. Some cases also have `start_offsets`, `live_rows`
  (section 9.1) and `aliases` (section 4). Use `case_key` to find the row in
  the case list (section 8).
- **`reference_source`** names `test-plans/sdpa-fwd/cases.csv`. No file has
  that path in the repository: the generated case list is not shipped. The
  path names the copy of the case list that the bundles came from. That copy
  is the case list of section 8 with five columns added at the end of each row
  by the bundle tools: `topology`, `bundle_status`, `exclusion_reasons`,
  `bundle_path` and `gtest_name`. Its rows have the same `case_key` values as the case list, so
  `case_key` links each case to its row.
- **Seed**: the input seed of a case is the first 8 hex digits of its
  `case_key`, masked to 31 bits. Example: `case_key` `a69ef8a0…` gives seed
  0x269ef8a0 = 647,952,544.
- **Inputs**: the suite fills the input tensors. `metadata.inputs` gives a seed
  for each input. An input whose contents matter gives them too: a scale that
  is supplied at run time gets a fixed fill (`{"kind": "fixed", "value": …}`),
  and page tables, offsets, lengths, block masks, descale lists, dropout masks
  and dropout seeds and offsets get their exact values
  (`{"kind": "values", …}`).

The bundles have no golden data. The suite checks each output against the
hipDNN GPU and CPU SDPA reference executors.

### 9.5 Lanes

A **lane** is one engine's run of the suite. The `ASM_SDPA_ENGINE` lane selects
every op folder whose name contains `Sdpa`, so it runs these cases. The
HIP_MLOPS, MIOPEN and hipBLASLt lanes have no SDPA support. Their tier YAML
files exclude `*_SdpaFwdPlan_*`, so they do not load these cases only to
decline them. The other attention lanes of the hip-kernel provider select
`*_SdpaFwd_*` names, which do not match `SdpaFwdPlan`.

## 10. Known engine defects and their skips

An engine can accept a case and then give a wrong output. The test of that
case then fails. When the cause is a known engine **defect** (a bug), the case
gets a **skip**: an entry in the engine's TOML file that tells the suite not to
run the case. The case stays in its tier. This section tells how the skips for
these cases were made, and how to remove them.

The skips are in
[`ASM_SDPA_ENGINE.toml`](../../../hip-kernel-provider/config/ASM_SDPA_ENGINE.toml),
after the comment line `-- SdpaFwdPlan known engine defects --`. Read that file
for the current list. This document does not copy it.

### 10.1 Running the cases

1. Run the `ASM_SDPA_ENGINE` lane of `hipdnn_integration_tests` (section 9.5)
   on the `SdpaFwdPlan` cases of each tier, on a gfx942 GPU and on a gfx950 GPU.
2. Run each tier two times, once with each **input fill**. The input fill is
   how the suite makes the input data. The default fill makes large inputs on
   the GPU. The host fill (`HIPDNN_TEST_HOST_INPUT_FILL=1`) makes every input on
   the CPU ([Running Tests](../running-tests.md)). The two fills give different
   data, and some defects show with one fill only.
3. Record the result of each case on each GPU: `passed`, `declined` (the engine
   does not accept the case), or `failed`.

A case **fails** when an output tensor does not match the reference. A case
also fails when the engine accepts it but no reference executor can check it.
The suite calls this **unverifiable**.

A case that failed with either fill counts as a failure.

### 10.2 Tracing each failure to a ticket

Each failure must have a known cause before it gets a skip.

1. Find the case's row in the case list with the `case_key` in the case's
   metadata (section 9.4). The row gives every setting of the case.
2. Read the failure message. It names each output tensor that does not match,
   for example O, Stats, Max or Sum_exp.
3. Compare the settings and the failing tensors with the known defects of the
   engine. Each known defect has a **ticket**: a bug report in Jira, with a key
   such as `ALMIOPEN-NNNN`. A defect also has a **trigger**: the settings that
   make it occur, for example "Stats is requested with a layout that the kernel
   cannot write".
4. If no known defect explains the failure, find the cause in the engine code.
   Then add the case to the ticket of that cause, or open a new ticket.

Do not skip a failure that has no ticket.

A case can have the triggers of more than one defect. A skip applies to the
whole test, and the first entry that matches a test is the one that the suite
uses. So such a case is skipped under one ticket only. Use the ticket of the
defect that the failure message shows, for example the defect that makes the
failing tensor wrong. The case still has its other defects. Section 10.4 tells
what to do about them when a fix lands.

### 10.3 Writing a skip entry

A skip entry is a `[[test_skips]]` table
([File Formats](../file-formats.md)). Write one entry for each defect and each
set of GPUs. An entry has three fields:

```toml
# SdpaFwdPlan cases that <trigger, in a few words>.
[[test_skips]]
archs = ["gfx942", "gfx950"]
filters = [
    "<tier>_SdpaFwdPlan_<Topology>.*",
    "<tier>_SdpaFwdPlan_<Topology>.<case id>",
]
reason = "ALMIOPEN-NNNN: <what the engine does wrong>"
```

- `archs`: the GPUs on which the failure was seen. Do not add a GPU on which
  the case passed, or on which it was not run. The suite matches each value as
  part of the GPU's architecture name, so `gfx942` matches
  `gfx942:sramecc+:xnack-`.
- `filters`: GTest names (section 9.4). A name can hold `*`, which matches any
  run of characters. A name with `*` is a **glob**; a name without `*` is an
  **exact name**.
- `reason`: the text that the suite prints when it skips the case. Start it with
  the ticket key, then a colon, then the defect in one short sentence. When a
  second ticket blocks the check (for example, a reference executor cannot yet
  check the case), name that ticket in the sentence too.

Use a glob only when it is safe. A glob is safe when, on every GPU in `archs`,
every case that it matches either failed with this defect or was declined.
Otherwise, list exact names. Some defects give a wrong output only with some
input data. For those defects, always list exact names: a glob could hide a case
that passes.

A glob also skips the declined cases that it matches. The suite then records no
result for those cases on that GPU.

Check the entries before you commit them. Apply the entries to the recorded
results of section 10.1. Every failure must be skipped. No passing case may be
skipped. Each skipped failure must be skipped under a ticket whose trigger the
case has.

### 10.4 Removing a skip when a fix lands

1. When a fix for a ticket lands, find every entry whose `reason` starts with
   that ticket key. Delete those entries in the change that lands the fix.
2. Run the cases again on every GPU in the deleted entries' `archs`.
3. A case that passes, or that the engine declines, needs nothing more.
4. A case that still fails has another defect (section 10.2). Add it to the
   entry for that defect's ticket, with the rules of section 10.3.

A known defect that needs a setting that no bundle can express needs no skip:
rows with such a setting are excluded (section 9.1).

### 10.5 Unverifiable cases

An uncheckable single (section 7) that the engine accepts is reported as
Unverifiable, and an unverifiable case fails. Such a case has no engine defect:
no reference can compute its expected output. It gets a skip whose ticket is
the one that blocks the check, for example a reference executor that does not
take the case's setting. Each such entry lists exact names. Delete it when the
blocking ticket is fixed, then follow section 10.4.

## 11. Limits and what comes next

- The cases come from a PoC. The labels, the tier recipe and the budgets are a
  draft for review. The case generator, the graph exporter and the bundle
  tools are still being developed and are not in the repository.
- No **support claims** are recorded for these cases. A support claim is a
  `support.json` file beside a sweep. It says that an engine supports a case on
  a given GPU ([Support Matrix](../support-matrix.md)). Each sweep's
  `support.json` is empty. Claims come in a later change.
- 117 rows are not bundles (section 9.1). 113 of them need settings that a
  bundle cannot express or compare. The other 4 have random dropout without
  `rng_dump`. They are rows of the case list, but no bundle holds them.
- `ASM_SDPA_ENGINE` declines every graph that declares a tensor alignment
  below 16 bytes. So the lane gets no engine result for an `element_aligned` or
  `sub_element` row.
- The rejection probes and the API-path variants in `injections.yaml` are not
  bundles. They belong to a rejection suite.
- The sizes override cannot set the page size or the window bounds, so the
  size sweep gives them no typical or corner sizes.
- The suite is expected to read the case list directly in a later step. The
  bundles under `SdpaFwdPlan` will then be removed.

## 12. The injected cases

[`injections.yaml`](../../test-plans/sdpa-fwd/injections.yaml) holds the cases
that are added on purpose, outside the generated sets. None of them is a bundle
(section 11). The file itself records why each case is there, so this section
explains how to read it rather than repeating it.

### 12.1 How the file is laid out

The comment block at the top of the file explains each group and each field.
The entries sit in four groups:

| Group | Ids | What the group holds |
|---|---|---|
| `rejection_probes` | `MV-01` to `MV-22` (no `MV-20`) | Requests that the frontend accepts but that hipDNN must refuse. A definition rule or a policy rule removes each one from the generated sets (section 3); the probe brings it back as a test that expects a refusal. One entry holds one or more probes (`MV-01a`, `MV-01b`, ...). |
| `path_variants` | `PV-*` | One request stated through different API routes. The variants of one entry must give the same outcome class. Only the C++ routes run; `PV-PYTHON` and `PV-CUDNN-COMPAT` are recorded only. |
| `frontend_rejection_controls` | `FC-*` | Requests that the frontend refuses today. They show that a test driver can tell "frontend rejected" from "no engine accepted". They are not rejection probes. The two controls are atypical: unit tests usually cover a frontend refusal, so a future revision of the test cases can remove them. |
| `conditional_probes` | `CP-*` | Probes that become rejection probes only if their open question is decided as "meaningless". All are inactive today. |

Every case starts from the base case `shared` (bf16, head size 128 for Q/K and
for V, no mask) and changes only the settings in `changes` and
`api_overrides`.

### 12.2 Where the reason is

| Field | What it tells you |
|---|---|
| `what` | The request, in a few words. |
| `basis` | The rule in the [rules file](../../test-plans/sdpa-fwd/rules.yaml) that the request breaks (section 5). `none` marks a debatable entry; its `debatable_reason` says why. `also_basis` names a second rule. |
| `why` | Why the frontend lets the request through. For a path variant, why the routes might differ; for a frontend rejection control, which frontend check refuses the request. |
| `open_question` | For a conditional probe, the question that decides whether it becomes active. |
| `moot`, `note`, `not_probed_on_purpose`, `sha_sensitive` | Extra context where an entry needs it. |
| `expect` | What a correct stack does with the request. |

The `why` text cites the frontend source as `ABBREV::function`: the file
abbreviation, then the function or field. A quoted string after the function is
an error message or a condition in it. FBS fields are written as
`FBS SdpaAttributes::field`. The abbreviations are listed under
`api_record.paths` in [`model.yaml`](../../test-plans/sdpa-fwd/model.yaml),
relative to `projects/hipdnn/`. Every citation is at the commit in
`api_record.sha`. For example, `SA` is
`frontend/include/hipdnn_frontend/attributes/SdpaAttributes.hpp` and `NODE` is
`frontend/include/hipdnn_frontend/node/SdpaFwdNode.hpp`.

An example entry:

```yaml
- entry: MV-01
  what: "Dropout probability outside [0, 1]"
  basis: D-DROPOUT-P-RANGE
  why: "set_dropout_probability stores any float; no forward range check (SA::set_dropout_probability; FBS SdpaAttributes::dropout_probability)."
  probes:
    - {id: MV-01a, ..., changes: {dropout: random, dropout_probability: negative, ...}, expect: refused at validate}
```

Read it as: the request is random dropout with a probability below 0 (`MV-01a`)
or above 1 (`MV-01b`). It breaks the definition rule `D-DROPOUT-P-RANGE`. The
frontend lets it through because the setter stores any float and nothing
checks the range: `SdpaAttributes::set_dropout_probability` has no range
check, and the flatbuffer schema field `dropout_probability` is an optional float with no range.
A correct stack refuses it at validate.

There is no `MV-20`. A comment in the file says why: sequence-length tensors
without the padding-mask flag are the model value
`sequence_lengths = variable_no_flag`, a normal case.
