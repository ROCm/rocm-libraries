# SDPA forward with a paged K/V cache

This page defines the graph contract for scaled dot-product attention (SDPA) forward over a paged
K/V cache, alone or combined with ragged queries. The semantics follow the cuDNN frontend's paged
attention ([cudnn-frontend `docs/operations/Attention.md`](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md),
"Paged attention", and `PagedCacheLoadNode::pre_validate_node`). Where hipDNN differs from cuDNN,
[Differences from cuDNN](#differences-from-cudnn) says so.

The frontend checks the graph rules in `SdpaFwdNode::pre_validate_node`. The CPU reference
(`CpuFpReferenceSdpaRagged::forwardPaged`) and the GPU reference
(`GpuFpReferenceSdpaRagged::fpropPaged`) implement the runtime semantics. Engines that accept a
paged graph must match them.

## Graph

| Operand | Setter | Dims | Type | Notes |
|---|---|---|---|---|
| Q | `sdpa(q, ...)` | `[B, H_q, S_q, D]` | FP16 or BF16 | Any strides. Dense, or ragged at the sequence axis (below). |
| K | `sdpa(.., k, ..)` | `[num_blocks, H_k, block_size, D]` | same as Q | Block container, any strides. |
| V | `sdpa(.., v, ..)` | `[num_blocks_v, H_v, block_size, D_v]` | same as Q | Block container, any strides. Same `block_size` as K. |
| Page table K | `set_paged_attention_k_table` | `[B, 1, M, 1]` | INT32 | Any strides. Row `b` lists the physical blocks of sequence `b`. |
| Page table V | `set_paged_attention_v_table` | `[B, 1, M, 1]` | INT32 | May be the same tensor as page table K. |
| Sequence length KV | `set_seq_len_kv` | `[B, 1, 1, 1]` | INT32 | Required. Device tensor. |
| Sequence length Q | `set_seq_len_q` | `[B, 1, 1, 1]` | INT32 | Required for ragged Q, optional for dense Q. |
| Max KV length | `set_paged_attention_max_seq_len_kv` | scalar | int32 | Required. `M == ceil(max_seq_len_kv / block_size)`. |
| O | output | `[B, H_q, S_q, D_v]` | same as Q | Dense or ragged, like Q. |
| Stats | `set_generate_stats(true)` | `[B, H_q, S_q, 1]` | FP32 | Optional log-sum-exp. Dense or ragged, like Q. |
| Sink token | `set_sink_token` | `[1, H_q, 1, 1]` | FP32, or same as Q | Optional attention sink, one logit per query head. See [Attention sinks](#attention-sinks). |

GQA and MQA follow from the head counts (`H_q % H_k == 0`, `H_q % H_v == 0`). Masks use the diagonal
band: bottom-right causal is `set_diagonal_alignment(BOTTOM_RIGHT)` with
`set_diagonal_band_right_bound(0)`, and a sliding window adds `set_diagonal_band_left_bound(w)`.

A ragged Q follows [RFC 0014](rfcs/0014_RaggedTensors.md): its `ragged_offset` is INT32
`[B + 1, 1, 1, 1]`, and `ragged_offset_multiplier` scales stored offsets to elements. With the
multiplier set to the token stride (`H_q * D` for packed BSHD), a token-unit cumulative length
table such as vLLM's `query_start_loc` binds directly.

Because containers and page tables take arbitrary strides, a framework's cache binds with no copies.
For example, a cache that stores K and V of one token side by side has K strides
`[H_k * block_size * 2D, block_size * 2D, 2D, 1]`, and V is the same view `D` elements further on.
A `[num_reqs, M]` block table is the page table with strides `[M, M, 1, 1]`.

### Frontend rules

`pre_validate_node` rejects a paged graph unless all of these hold:

- Both page tables are set, or neither is.
- Each page table is rank 4, `[B, 1, M, 1]` with `M >= 1`, INT32, and not ragged.
- `seq_len_kv` is set. A ragged Q also needs `seq_len_q`. Both are INT32 `[B, 1, 1, 1]`.
- `max_seq_len_kv` is set, positive, and `M == ceil(max_seq_len_kv / block_size)` for both tables.
- K and V are rank 4 with the same `block_size` (dim 2). Q's batch is not compared with K's or V's
  dim 0, because that dim is `num_blocks`.
- An attention mask, if any, spans `max_seq_len_kv` keys.

`max_seq_len_kv` without page tables is an error, as in cuDNN. `seq_len_q` and `seq_len_kv`, when set
on any SDPA forward graph, must be INT32 `[B, 1, 1, 1]`.

## Runtime semantics

For each batch `b` (`seq_len_*` values are read on the device at execute time):

- **Query rows.** `seq_q[b]` is `seq_len_q[b]` if set; otherwise the ragged-offset length for a
  ragged Q; otherwise `S_q`. It is at most the rows that `b` owns in Q. Rows at or past `seq_q[b]`
  are not written.
- **Key rows.** `seq_kv[b] = seq_len_kv[b]`, with `0 <= seq_kv[b] <= max_seq_len_kv`. Key `s`
  (`0 <= s < seq_kv[b]`) is `K[page_table_k[b, 0, s / block_size, 0], h_k, s % block_size, :]`, and
  likewise for V. Only the first `ceil(seq_kv[b] / block_size)` page-table entries are read, and
  each must be a valid block index. Unused entries may hold anything, such as `-1`. A
  `seq_len_kv[b]` outside `[0, max_seq_len_kv]` breaks the contract and its result is undefined:
  an engine may clamp it or read page-table entries past row `b`.
- **Mask alignment.** Bottom-right alignment uses `seq_kv[b] - seq_q[b]` as the diagonal offset of
  each sequence. The left bound of a sliding window is relative to that diagonal.
- **Empty rows.** A live query row with no unmasked key, including every row of a sequence with
  `seq_kv[b] == 0`, writes zeros to O and `-inf` to stats (the head's sink instead, when a sink
  token is set). This is how CUDA-graph padding sequences stay finite.
- **One plan for every step.** B in the graph is an upper bound. Sequences past the live batch carry
  `seq_len_q = seq_len_kv = 0`. Lengths and block ids come only from device tensors, so a plan
  built once serves every step and can be captured into a HIP graph.

## Attention sinks

An optional `sink_token` adds one learned logit per query head to the softmax, as in gpt-oss
and StreamingLLM. It follows cuDNN's `SDPA_attributes::set_sink_token`
([cudnn-frontend sample `samples/cpp/sdpa/fp16_fwd_with_sink_token.cpp`](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/sdpa/fp16_fwd_with_sink_token.cpp)
and the softmax node). It applies the same way to dense, ragged and paged graphs, because it is
per head, not per sequence. SDPA backward rejects it.

- **Dims** exactly `[1, H_q, 1, 1]`, any strides. Engines may require a unit head stride.
- **Data type** FLOAT, as in cuDNN, or the Q data type when that is HALF or BFLOAT16. The second
  form is a hipDNN extension: vLLM keeps sinks in the model dtype. Anything else is rejected by
  `pre_validate_node` with `INVALID_VALUE`.

For batch `b`, query head `h` and query row `i`, let `s_ij` be the logits after `attn_scale`,
bias, ALiBi and masking (masked keys are `-inf`) and `sink_h` the sink of head `h`, read as FP32:

- `m_i = max(sink_h, max_j s_ij)`
- `l_i = exp(sink_h - m_i) + sum_j exp(s_ij - m_i)`
- `O_i = sum_j exp(s_ij - m_i) V_j / l_i`. The sink has no value row.
- `Stats_i = m_i + log(l_i)`. The log-sum-exp includes the sink.

The sink is not multiplied by `attn_scale`. A row with no visible key writes `O_i = 0` and
`Stats_i = sink_h` (instead of `-inf` without a sink). `sink_h = -inf` is the same as no sink
for head `h`. Dropout, if any, applies to the normalised probabilities after the sink joined the
denominator.

vLLM stores sinks as `layer.sinks`, a `[H]` tensor in the model dtype holding raw post-scale
logits. Bind it without a copy as the view `[1, H, 1, 1]` with strides `[H, 1, 1, 1]`.

## Differences from cuDNN

| Topic | cuDNN frontend | hipDNN |
|---|---|---|
| Paging one side only | K or V can be paged alone | Both page tables or neither |
| `max_seq_len_kv` | Optional; inferred from the page tables when absent | Required for paged graphs |
| `seq_len_q` with paged K/V | Always required, along with `use_padding_mask` | Optional for dense Q, required for ragged Q; `padding_mask` is not required |
| Block size | K and V containers may differ | Must match |
| Packed (ragged) page tables | Supported (cuDNN 9.10.2+) | Rejected |
| Sink token data type | FLOAT in every sample; no frontend check | FLOAT, or the Q data type when that is HALF or BFLOAT16; checked by `pre_validate_node` |
| `cu_seq_len_q` / `cu_seq_len_kv` | Supported (cuDNN 9.24+) | Not available; use `ragged_offset` with a multiplier |

## Test coverage

- Frontend: `TestSdpaFwdNode.*Paged*`, `TestSdpaFwdNode.*Sink*`, `TestGraph.SdpaFwdPaged*`.
- CPU reference and CPU graph executor: `projects/hipdnn/test_sdk/tests`.
- GPU reference and GPU graph executor: `dnn-providers/integration-tests/tests`.
- Golden bundles generated by gathering pages and running dense PyTorch SDPA:
  `integration-test-bundles/quick/SdpaFwd/bshd/{bf16,fp16}/*_paged/`.
