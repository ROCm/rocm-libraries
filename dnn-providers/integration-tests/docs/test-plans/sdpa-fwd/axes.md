# SDPA Forward: Every Test Axis

This catalogue lists every axis that a **forward** attention operation could be
tested across if there were no limit on the number of tests. It ignores cost,
priority and practicality. It does not decide what to test.

[`L0-axes.yaml`](../../../test-plans/sdpa-fwd/layers/L0-axes.yaml) (Layer 0) is
the machine-readable form of this catalogue. The layer files cite this document
as `axes.md § N 'Row title'`. The method that narrows these axes to the test
cases is in [`sdpa-fwd.md`](../sdpa-fwd.md), sections 2 and 3.

## How to read this

- **Forward attention** is the direction that a model uses when it *runs*: it
  takes Q, K and V and computes the output O.
- Every axis is **black-box**: a thing that a caller can set, or a condition
  that a caller can make, from the outside. No axis depends on how a kernel
  works inside.
- **Full values** are the values to test with unlimited time. Many axes are
  endless, for example "every whole number", so the table describes them
  instead of listing them.
- **Footnotes** (¹ ² ³ …) mark values that exist because of *why someone would
  pick them*: tile edges, real model shapes, past defects. They are at the end.
- **Layer 0 axes** gives the ids of the Layer 0 axes that hold the row. "Set
  aside" means that Layer 0 does not hold the row as an attention axis; the
  `set_aside` list of `L0-axes.yaml` gives the reason and where the row enters
  instead.
- A value marked **(no meaning)** is in Layer 0 only so that Layer 1
  ([`L1-definition.yaml`](../../../test-plans/sdpa-fwd/layers/L1-definition.yaml))
  can remove it with a recorded reason.
- What hipDNN's frontend can express of each axis is in
  [`L2-frontend.yaml`](../../../test-plans/sdpa-fwd/layers/L2-frontend.yaml).

Tensor names: Q (query), K (key), V (value), O (output), S (the score matrix),
Stats (the per-row log-sum-exp, LSE, of the scores).

---

## 1. Batch packing

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Batch packing** | Several sequences can be sent together. They can be packed into **equal-length slots** (padded with blanks), or squashed end to end with a list that says where each one starts (**ragged**). The query side (Q and the output O) and the key/value side (K and V) can be packed differently, for example ragged queries that read a paged key/value cache. | dense · ragged · ragged query side only · ragged key/value side only | `batch_packing` |
| **Output packing** | How O stores its rows when Q is ragged. An output row belongs to its query, so O has the sequences and lengths of Q. Only where O stores them can differ. | ragged with the start positions of Q · ragged with its own start positions · equal-length slots | `o_packing` |
| **Value packing** | How V stores its rows when K is ragged. Each key pairs with one value, so V has the sequences and lengths of K. Only where V stores them can differ. | ragged with the start positions of K · ragged with its own start positions · equal-length slots | `v_packing` |

---

## 2. Number types

Computers store numbers in different formats. Smaller formats are faster but
less precise, so each one is a different accuracy challenge.

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Input number type** (Q, K, V) | The format of the three main inputs. | fp32 · fp16 · bf16 · fp8-e4m3 · fp8-e4m3-fnuz · fp8-e5m2 · fp8-e5m2-fnuz | `dtype_q`, `dtype_k`, `dtype_v` |
| **Per-tensor type mixing** | Q, K and V do not have to share a format. For example, K and V can be fp8 while Q is bf16. | every assignment of the formats above to each input tensor separately | `dtype_q`, `dtype_k`, `dtype_v` (one axis each) |
| **Output number type** | The output can use a different format from the inputs ("mixed precision"). | every format above, for each input format | `dtype_o` |
| **Stats / LSE number type** | The format of the optional extra output that the backward pass needs later (the log-sum-exp, see § 8). | fp32 · fp16 · bf16; storing a number in any of the seven formats has meaning (fp8 loses precision but is not meaningless) | `dtype_stats` |
| **Internal working precision** | How precise the operation is *while* it calculates, before it rounds the result. For example, it adds up in fp32 although the inputs are fp16. | fp32 · fp16/bf16, where offered | `compute_precision` |
| **Matrix-unit input precision** | The precision that the matrix-multiply operands are converted to before each product. It is separate from the accumulation precision. | the stored format · fp32 · fp16 · bf16 · fp8 | `matmul_input_precision` |

---

## 3. Sizes (shapes)

These say *how much* data there is. Most are endless.

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Batch size** | How many independent sequences are processed together. | every whole number from 1 up to what fits in GPU memory ¹ ⁴ | `batch_size` |
| **Query length** (`seq_q`) | How many tokens ask. **1** is the "generate the next word" case. hipDNN's frontend rejects 0 as a tensor size; a 0 reaches it only as a per-sequence length (§ 10). | 0 ⁴, 1, then every whole number up to the memory limit ¹ ² | `seq_q` |
| **Key/value length** (`seq_kv`) | How many tokens are looked at. It can differ from the query length. K and V share it: each key pairs with one value. | 0 ⁴, 1, then every whole number up to the memory limit ¹ ² | `seq_kv` |
| **Query-vs-key/value length relationship** | Several features (for example the causal mask in § 5) behave differently when the query length is shorter than, equal to, or longer than the key/value length. | `seq_q < seq_kv` · `seq_q = seq_kv` · `seq_q > seq_kv` | `seq_q_vs_seq_kv` |
| **Query head count** | How many attention "heads" run side by side. | every whole number from 1 upward ¹ | `q_heads` |
| **Key/value head count, and sharing** | Several query heads can share one key/value head to save memory: **MHA** (no sharing), **GQA** (groups share), **MQA** (all share one). K and V each have their own sharing. | every value of `query_heads ÷ kv_heads` that divides evenly, from 1 (MHA) up to `query_heads` (MQA); a K or V head count that does not divide the query head count (no meaning) | `k_head_sharing`, `v_head_sharing` |
| **K heads vs V heads** | Whether K and V have the same number of heads. Each query head can use its own K group and its own V group. | equal · different | `k_heads_vs_v_heads` |
| **Q/K head size** (`head_dim_qk`) | The length of each query and key vector, the `d` in the formula. | every whole number from 1 upward ¹ ² ³; in practice, hardware often needs multiples of 8 | `head_dim_qk` |
| **V head size** (`head_dim_v`) | The length of each value vector, which is also the length of each output vector. It **can differ** from the Q/K head size (for example 192 vs 128). | every whole number from 1 upward, paired with every Q/K head size ³: smaller than, equal to, or larger than it | `head_dim_v`, `head_dim_v_vs_qk` |

---

## 4. How the data is laid out in memory

The same numbers can sit in memory in different orders. A kernel that assumes
the wrong order reads garbage, often **silently**, with no error.

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Dimension order** | Which dimension comes "outermost" in memory. Two common orders are **BHSD** (batch, heads, sequence, dim) and **BSHD** (batch, sequence, heads, dim). | all 24 orderings of the four dimensions, per tensor: BHSD, BSHD, the other 4 with the head dimension innermost, and the 18 with the head dimension not innermost. Stats has one value per row, so where its head dimension sits does not change memory. K or V can also read through the order of the input that it is a view into ("Overlap between tensors" below) | `dim_order_q`, `dim_order_k`, `dim_order_v`, `dim_order_o`, `dim_order_stats` |
| **Strides** | How far apart (in memory) neighbouring elements are. **Packed** means no gaps. Strides can include **padding gaps**, can be **non-contiguous** (a slice of a bigger tensor), or can be **zero**, which repeats the same data (broadcasting). | packed · padded by every gap size · any valid non-contiguous stride · zero-stride (broadcast) on each dimension where it's legal ⁵ · negative (a dimension walked backwards) · self-overlapping (two indices reach one element) · head dimension strided (stride 2 or more). A view reads through the strides of its owner | `strides_q`, `strides_k`, `strides_v`, `strides_o`, `strides_stats` |
| **Per-tensor layout independence** | Every forward tensor (Q, K, V, output O, and Stats if requested) can have its *own* layout. Mixed layouts catch kernels that assume "all the same". | every combination of the two rows above, chosen independently per tensor | the per-tensor axes above |
| **Starting address alignment** | Where in memory each tensor begins. A kernel can assume that it starts on a neat boundary, for example a multiple of 16 bytes. | aligned to every power-of-two boundary, plus deliberately misaligned offsets: on a wide boundary · on a multiple of the element size only · not on a multiple of the element size | `alignment_q`, `alignment_k`, `alignment_v`, `alignment_o`, `alignment_stats` |
| **Overlap between tensors** | Whether the output is written into the same memory as an input ("in place"), or two inputs share memory. | separate · output overlaps an input · inputs overlap each other (one input is a view into the buffer of another) · two outputs overlap (no meaning) | `tensor_overlap` |

---

## 5. Masking: which tokens can see which

A **mask** blocks some query–key pairs, so that they contribute nothing.

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Mask type** | The rule for blocking. **Causal** means a query cannot see the future. **Sliding window** means a query can see only nearby tokens. They can be combined: a key is visible only if every active mask allows it. A **padding mask** means the same as per-sequence lengths (§ 10): positions past the real length of a sequence take no part. A bias that holds −∞ is one more way to mask. | none · causal top-left · causal bottom-right · sliding window · custom block mask · padding mask | `causal_mask`, `sliding_window`, `block_mask`; padding mask: `sequence_lengths` |
| **Causal alignment** | When the query and key/value lengths differ, which diagonal counts as "now". Top-left and bottom-right give **different results**, but only when the lengths differ. | top-left · bottom-right (each crossed with the § 3 length relationship) | `causal_alignment` |
| **Window alignment** | Which diagonal the sliding window is centred on when the lengths differ. A window needs a diagonal, as the causal mask does. Two active masks can use different diagonals. | top-left · bottom-right | `window_alignment` |
| **Sliding-window size** | How far left and right each token can look. | every left size from 0 to `seq_kv` × every right size from 0 to `seq_kv`. Grouped by meaning: no keys, some keys, or every key on each side; on the left, also "no keys and not the diagonal key" | `window_left`, `window_right` |
| **Custom block mask** | An arbitrary pattern of allowed and blocked blocks that the caller supplies. | every possible pattern (astronomically many). Grouped by meaning: every block allowed · some blocked, every query row keeps a key · some query rows fully blocked · every block blocked | `block_mask`, `block_mask_pattern` |
| **Block-mask granularity** | Which scores one flag of the block mask covers. | one flag per square tile of scores, for example blocks of 128×128 (the tile size relative to the lengths decides which scores share a flag, and a partial tile at the edge covers fewer scores) · one flag per score | `block_mask_granularity` |
| **Block-mask element format** | The element format that holds the block-mask flags. | 8-bit unsigned (eight flags per element) · 32-bit (thirty-two flags per element) | `block_mask_dtype` |
| **Fully-masked rows** | Whether some query can see **no key at all**. Then the softmax divides by zero, and the correct result is an open question. | no such rows · some rows · all rows ⁶ | a result of other axes: `derived_conditions` `fully_masked_rows` |

---

## 6. Extra terms added to the scores

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Additive bias** | An extra tensor added to the scores before softmax, to push attention toward or away from some pairs. | none · present, in every broadcastable shape — repeated over batch only, over heads only, or over both, over positions, or one value for everything — × every number type; a shape that cannot broadcast to the scores (no meaning) | `bias`, `bias_broadcast`, `bias_dtype` |
| **ALiBi** | A built-in bias that penalises distance: far-apart tokens get pushed down. Each head has its own **slope**. The standard slopes are calculated differently when the **number of heads is not a power of two** (for example 6 or 12). | off · on × every slope value × head count a power of two or not. Slopes grouped by meaning: the standard geometric sequence · other positive · zero (no penalty) · negative (distance rewarded) | `alibi`, `alibi_slopes`, `alibi_head_count` |
| **Attention sinks** | An extra "dummy" slot that takes part in the softmax denominator but contributes no value vector, so a row does not have to spread all its attention across real tokens. | none · present × every sink value | `attention_sinks` |
| **Sink format** | The storage format of the sink logits. A low-precision format loses precision but has meaning. | every format of § 2 | `sink_dtype` |
| **Sink values** | The sink logits, grouped by what they do to the softmax. These are tensor contents: a hint for a fuzzer, not for test-case generation. | ordinary finite logits · −∞ (the sink adds nothing) · a logit so large that the sink takes almost all the weight · +∞ or NaN | `sink_values` |
| **Logit soft-capping** | Squashes very large scores as cap · tanh(score / cap), so that they cannot grow without limit. FlashAttention offers this. It is one example of a general custom score function. | off · on × every cap value: positive · negative (gives the same result as the absolute value) · zero (no meaning: division by zero) | `logit_softcap`, `softcap_value` |

---

## 7. Scaling and dropout

The formula divides the scores by √d; callers can change that. **Dropout**
randomly sets some attention weights to zero during training, so that the
model does not depend on any one connection.

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Scale source** | Where the scale comes from: calculated automatically, fixed when the operation is set up, or supplied at run time as a tensor. | default (1/√d) · fixed constant · runtime tensor | `scale_source` |
| **Scale value** | The actual number. Unusual values are worth testing. | every real number: 0 ⁴ · negative ⁴ · tiny · huge ⁷ · the default; ±∞ and NaN are a hint for a fuzzer | `scale_value` |
| **Dropout on/off** | Whether dropout is used at all, and how its pattern is made: drawn at random from a probability and a random-number state, or supplied by the caller. | off · random · caller-supplied pattern | `dropout` |
| **Dropout probability** | What fraction of the weights is set to zero. | every value from 0 to 1, including the edges 0 ⁴ and 1 ⁴; below 0 or above 1 (no meaning) | `dropout_probability` |
| **Random seed and offset** | The starting point for the random generator. The same seed must give the same pattern, and backward must be able to make it again. | every seed × every offset; not supplied | `dropout_rng_state` |
| **Seed and offset storage** | Where the seed and the offset are held. | by value in the request · in device memory, read when the operation runs | `rng_state_storage` |
| **Caller-supplied dropout mask** | The caller supplies the dropout pattern and the rescale factor instead of a seed. | keep all · some dropped · drop all (the output is zero) · an entry other than 0 or 1 (no meaning) | `dropout_mask_pattern` |
| **Caller-supplied dropout mask, repeated** | A caller's dropout pattern can be one per sequence and head, or repeated over the batch, the heads, or both. | none · over batch · over heads · over both | `dropout_mask_broadcast` |
| **Dropout mask format** | The storage format of the caller's dropout pattern. 0 and 1 are exact in every format, so the format changes the bytes read, not the pattern. | every format of § 2 | `dropout_mask_dtype` |
| **Dropout scale storage** | Where the rescale factor of a caller-supplied pattern is held. | by value in the request · in device memory, read when the operation runs | `dropout_scale_storage` |

---

## 8. Optional outputs

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Stats / LSE output** | The log-sum-exp of each row (the logarithm of the softmax denominator). Backward uses it again instead of calculating the softmax again. | not requested · requested | `stats_output` |
| **Stats packing** | How Stats stores its rows when several sequences share one tensor. | equal-length `[B, H, S_q, 1]` slots · packed token by token with the token boundaries of Q, with its own list of start positions | `stats_packing` |
| **Max and sum-exp outputs** | The two parts of the LSE, handed back separately: the row maximum and the row sum of exponentials. Each part is a meaningful output on its own. | not requested · both · max only · sum-exp only | `max_sumexp_output` |
| **Dropout-pattern output** | Hand back *which* entries were dropped, so that a test can check them. It exists only when dropout is used. | not requested · requested | `dropout_pattern_output` |
| **fp8 amax outputs** | The largest absolute value seen, used to choose the fp8 scales next time. The largest absolute value of a tensor has meaning for every format. | not requested · requested, per tensor (Q, K, V, S, O) | `amax_q`, `amax_k`, `amax_v`, `amax_s`, `amax_o` |

---

## 9. fp8 scale factors

fp8 is so small that values must be multiplied by a **scale** to fit, then
divided back. Each tensor can have its own. Scaled storage has meaning for
every format, not only fp8.

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Which scale/descale tensors are supplied** | Q, K, V, the score matrix, the output. | every subset of them | `descale_q`, `descale_k`, `descale_v`, `s_quantization`, `scale_o` |
| **How many factors a descale has** | One factor for the whole tensor, one per sequence and head, or finer (per token, per channel or per block). | per tensor · per (batch, head) · finer | `descale_q`, `descale_k`, `descale_v` |
| **Factor storage** | Where a one-element descale or scale factor is held. | by value in the request · in device memory, read when the operation runs | `descale_storage` |
| **Scale values** | The actual multipliers. These are tensor contents: a hint for a fuzzer, not for test-case generation. | every positive real number, including ones that push values to the limits of fp8 ⁷. Grouped by meaning: ordinary · every factor 1 (no effect) · a factor of 0 · ±∞ or NaN | `scale_factor_values` |

---

## 10. Uneven lengths and caches

The KV cache is how a model remembers earlier tokens while it generates. It is
an inference-only feature, so this section has no backward counterpart.

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Per-sequence lengths (varlen)** | In one batch, each sequence can have its own real length. Positions past it are not part of the sequence. The query side and the key/value side vary independently. | every list of lengths, each from 0 ⁴ to the maximum, including all-equal, all-different, and some-zero. Grouped by meaning: all full · all equal and shorter · mixed · some zero · all zero; a length larger than the storage, or below zero (no meaning) | `sequence_lengths`, `per_seq_len_q`, `per_seq_len_kv` |
| **Length form** | How the per-sequence lengths are given. | one length per sequence · running totals (B + 1 entries: lengths 3, 2, 4 give 0, 3, 5, 9) · ragged packing only: the difference of two neighbouring start positions | `length_form` |
| **Ragged offsets** | Where each sequence starts in a squashed (ragged) tensor. | every valid list of starting points: contiguous · gapped (a slot longer than its live sequence) · a gap before the first sequence · a different gap between each pair · stored out of index order; two sequences that share memory, or a start inside a token (no meaning) | `ragged_offsets` |
| **Ragged offset unit** | What one step of a stored start position counts. The element position of a start is the stored number times a multiplier. The same start positions in another unit read other bytes. | elements (multiplier 1) · tokens (the multiplier is the sequence stride of the tensor) | `ragged_offset_unit` |
| **Ragged offset sharing** | Whether the ragged tensors of one side read one list of start positions or one list each. | one list per tensor · one list for Q, O and a packed Stats, and one list for K and V | `ragged_offset_sharing` |
| **Ragged offset format** | The integer format of the start-position lists. | int32 · int64 | `ragged_offset_dtype` |
| **Paged KV cache** | Stored keys and values split into fixed-size **pages** anywhere in memory, with a lookup table (the page table). This is how chat services store conversation history. | off · on × every page size × every page-table order (in order, shuffled, reused) × every page count; K only, V only, or both paged; one table shared by K and V or one each; K and V page sizes equal or different; the table itself dense or packed. Page size grouped by meaning: one token · more than one token but less than a sequence · a whole sequence. Page count: exact · more than needed · fewer than needed (no meaning) | `paged_kv`, `page_table_sharing`, `page_size_k_vs_v`, `page_table_storage`, `page_size`, `page_table_order`, `page_count` |
| **Max KV length** | The declared maximum key/value length of a paged cache, relative to the actual key/value length. | equal · larger · smaller (no meaning) | `max_seq_len_kv` |
| **KV cache update style** | Adding new tokens to an existing cache (**append**), or splitting a long cache across workers (**split-KV**). CK offers these. | none · append · split · both | `kv_cache_append`; split: set aside (`split_kv`) |
| **Index tensor format** | The integer format of the per-sequence length tensors and the page tables. | int32 · int64 | `index_tensor_dtype` |

---

## 11. The numbers inside the tensors

The same shape can hold different numbers. This is **not** an API setting, but
it changes what the kernel does.

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Value distribution** | Gentle values give a "flat" softmax that never stresses the running-max correction. Extreme values do. | uniform small · normal · large magnitude · sharply peaked scores ⁸ · all identical · Q and K nearly parallel (very sharp softmax). The range part, which decides whether a result exists or is exact: ordinary · large enough to overflow · small or different enough to lose precision | `value_magnitude` (a hint for a fuzzer); the rest: set aside (`data_value_distribution`) |
| **Special values** | Numbers that break naive arithmetic. | +0 · −0 · +∞ · −∞ · NaN · denormals (numbers too small for normal precision) · the largest value that each format can hold; −∞ in the bias masks the score it meets | `special_values` (a hint for a fuzzer) |
| **Signs** | Inputs that are only positive, only negative or mixed reach different code paths. | all positive · all negative · mixed | set aside (`signs`) |
| **Data seed** | Which random numbers fill the tensors. | every seed ⁹ | set aside (`data_seed`) |

---

## 12. Where it runs

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **GPU architecture** | Different chip families run different machine code, so a kernel can be correct on one and wrong on another. | every supported AMD architecture (gfx942, gfx950, gfx90a, gfx1201, gfx1151 …) | set aside (`gpu_architecture`) |
| **GPU model within a family** | Chips in one family can have different core counts. This changes how work is split, and even which "random" numbers a seed gives ⁹. | every product in each family | set aside (`gpu_model_within_family`) |
| **Operating system** | Linux and Windows builds can behave differently. | Linux · Windows | set aside (`operating_system`) |
| **Software stack version** | The driver and the libraries below the operation. | every supported version combination | set aside (`software_stack_version`) |
| **Workspace memory given** | Some kernels need scratch memory. Too little must fail cleanly and corrupt nothing. | none · too little · exactly enough · more than enough | set aside (`workspace_memory`) |
| **Repeat runs** | Run the same case again: is the result the same? Run many at once on one GPU: is each still correct? | 1 run · repeated · concurrent on several streams (independent queues of GPU work) | set aside (`repeat_and_concurrent_runs`) |

---

## 13. How the request is made

| Axis | What it's for | Full values | Layer 0 axes |
|---|---|---|---|
| **Which optional settings are explicitly set** | A setting left at its default is not the same as a setting explicitly set to that default. A kernel can ignore a setting that it does not support instead of refusing it. | for every optional setting: left unset · set to its default · set to each other value | set aside (`explicit_setting_state`) |
| **Engine choice** | Ask for one specific implementation by name, or let the library choose. | each engine by name · automatic | set aside (`engine_choice`) |
| **Execution strategy** | How the caller asks for the operation to be computed. The result has the same meaning for every value. | chosen by the library · as several operations · as one fused operation | `execution_strategy` |

---

## A note on "all combinations"

Not every combination of the above has **meaning**. For example, a ragged
start position that falls inside a token, or a dropout probability above 1.
Layer 1 removes these combinations, each with a reason. Other combinations have
meaning but hipDNN's frontend cannot express them; Layer 2 removes those.

An engine that does not support a case is a different matter. Only a run of
the case shows it.

---

## Footnotes: why particular values get picked

¹ **Tile edges.** Kernels process data in fixed-size parts (tiles). Values just
below, at and just above a tile size (127/128/129), and primes such as 11, 17,
37 and 53 that never fit a tile evenly, reach the "leftover" code where defects
often are. AOTriton, CK and PyTorch choose values this way.

² **The generate-next-word case.** `seq_q = 1` with a long, growing `seq_kv`
is what chat services run for every word that they produce.

³ **Real model sizes.** Specific head sizes (64, 128, 192 with a V size of 128,
256) and lengths come from well-known AI models. cuDNN prefers these on
purpose.

⁴ **Extreme edges.** Zero and one (empty, single), and probability exactly 0 or
1. CK tests `seqlen = 0` and `seqlen = 1` on purpose. (In hipDNN, a 0 length is
reachable only through the per-sequence length list — see § 3.)

⁵ **Non-packed strides.** A kernel that assumes packed memory when the memory
is not packed.

⁶ **Fully-masked rows.** Bottom-right causal with `seq_q > seq_kv` makes these.
AOTriton has a test named after a defect report (#235) for exactly this.

⁷ **Overflow territory.** Scale or input values large enough to push numbers
past what a format can hold. That is the reason for the "subtract the row
maximum first" step in softmax, and the narrow range of fp8 makes it worse.

⁸ **The running-max path.** Peaked scores force the tiled ("flash") algorithm
to rescale its running totals during the calculation.

⁹ **Seeds are not portable.** cuDNN found that the same seed gives
**different** data on GPUs with different core counts, so "random but
reproducible" holds only on the same kind of chip.
