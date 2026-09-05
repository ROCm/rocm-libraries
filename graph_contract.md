# graph_contract.md — gfx950 2D tiled attention (`hipkernel:Gfx950AttentionTiled`)

**RUNBOOK step 2a.** What hipDNN can ask this kernel to do, and where the two vocabularies
disagree. Produced before any kernel mining (2b), because the matcher written at step 6 is a
translation between these two descriptions and reading only the kernel half is how
silent-wrong-answer defects get written.

**Base:** `users/bharriso/rocke-gfx950-tiled-attention` @ `37970632e15`.
**Kernel:** `kernels/gfx950/attention_tiled_2d.py`, builder `build_unified_attention_2d_tiled`,
spec class `UnifiedAttention2DTiledSpec` (46 fields, 8 required) — re-verified by RUNBOOK 1a
this session, `signature_error` empty.

---

## 1. The operation match

**One node: `SdpaAttributes`.** Not a composition, and no UID edges to walk.

- Schema table: `projects/hipdnn/flatbuffers_sdk/schemas/sdpa_attributes.fbs:24`.
- Frontend node: `frontend/include/hipdnn_frontend/node/SdpaFwdNode.hpp`.
- Frontend attributes: `frontend/include/hipdnn_frontend/attributes/SdpaAttributes.hpp`.

**There is no separate paged or tiled schema.** Paged and varlen are the *same* table with
different optional UIDs populated — `page_table_k/v_tensor_uid` (`:40-41`) and
`seq_len_q/kv_tensor_uid` (`:36-37`). The `.fbs` directory was enumerated by glob; no
tiled/paged file exists. So the outcome is row 1 of the graph-contract table ("one node
matches") and the six-check disconfirmation list does not apply.

`graph_match` therefore opens on `nodeCount() == 1` and a `SdpaAttributes` node type, exactly
as the dense sibling does.

---

## 2. The field audit — every field of `SdpaAttributes`

Every field is **consumed** (read and acted on) or **explicitly rejected**. There is no third
category: an unchecked field is accepted and then silently not honoured.

### 2.1 Required tensor UIDs

| Field | Disposition |
|---|---|
| `q_tensor_uid` | consumed — dims → `num_query_heads`, `head_size`, `total_q` |
| `k_tensor_uid` | consumed — **when paged, this IS the container**; see §5 G2 |
| `v_tensor_uid` | consumed — must share K's shape exactly |
| `o_tensor_uid` | consumed — Q's shape; the epilogue reuses the query base |

### 2.2 Optional input tensor UIDs

| Field | Disposition | Note |
|---|---|---|
| `page_table_k_tensor_uid` | **consumed — REQUIRED** | the tiled ABI is structurally paged; `block_tables_ptr` is an unconditional kernarg |
| `page_table_v_tensor_uid` | **consumed — REQUIRED** | must be present and shape-equal to K's table |
| `seq_len_q_tensor_uid` | consumed | → `query_start_len_ptr` (cu_q) |
| `seq_len_kv_tensor_uid` | consumed | → `seq_lens_ptr` |
| `sink_token_tensor_uid` | consumed | → `sink_ptr`; gates spec `use_sinks` |
| `attn_mask_tensor_uid` | **rejected** | additive bias; the kernel has no additive-mask path (`qq_bias_ptr` is a *different* concept — see §5) |
| `scale_tensor_uid` | **rejected** | scale must be the scalar `attn_scale_value`; the kernel takes `scale` as an F32 kernarg, not a tensor |
| `seed_tensor_uid` | **rejected** | dropout |
| `offset_tensor_uid` | **rejected** | dropout |
| `dropout_mask_tensor_uid` | **rejected** | dropout |
| `dropout_scale_tensor_uid` | **rejected** | dropout |
| `block_mask_tensor_uid` | **rejected** | block-sparse; no kernel path |
| `descale_q/k/v/s_tensor_uid` | **rejected** | fp8 declined in v1 |
| `scale_s/scale_o_tensor_uid` | **rejected** | fp8 declined in v1 |

### 2.3 Optional output tensor UIDs

| Field | Disposition |
|---|---|
| `stats_tensor_uid` | **rejected** — the 2D kernel writes no softmax stats (the 3D split-KV kernel does, and is a separate engine) |
| `max_tensor_uid` | **rejected** — same |
| `sum_exp_tensor_uid` | **rejected** — same |
| `rng_dump_tensor_uid` | **rejected** — dropout |
| `amax_s/amax_o_tensor_uid` | **rejected** — fp8 |

### 2.4 Boolean flags

| Field | Disposition |
|---|---|
| `generate_stats` | **rejected when true** — pairs with `stats_tensor_uid` |
| `alibi_mask` | consumed → spec `use_alibi`, `alibi_slopes_ptr` |
| `padding_mask` | consumed — varlen padding is intrinsic to this kernel; see §5 |
| `causal_mask` | consumed **as deprecated, precedence-first** — see §3 |
| `causal_mask_bottom_right` | consumed as deprecated — see §3 |

### 2.5 Scalar attributes

| Field | Disposition |
|---|---|
| `dropout_probability` | **rejected when set** |
| `attn_scale_value` | consumed → `scale` kernarg |
| `left_bound` | consumed → causal/sliding-window derivation, §5 |
| `right_bound` | consumed → causal/sliding-window derivation, §5 |
| `max_seq_len_kv` | consumed — paged-attention KV bound; see §5 |

### 2.6 Enum attributes

| Field | Disposition |
|---|---|
| `diagonal_alignment` | consumed — `TOP_LEFT`/`BOTTOM_RIGHT`, §3 precedence |
| `mma_core_mode` | consumed — must be `UNSET` or agree with the shipped MFMA dtype |
| `implementation` | **must be checked** — `AUTO`/`COMPOSITE`/`UNIFIED`. The dense sibling's `field_audit.sh` still prints `UNCHECKED: implementation`; this engine will not repeat that. `COMPOSITE` is declined (this kernel is the unified path). |

**Total: 41 schema fields, all accounted for.** `field_audit.sh` at step 6 is the mechanical
re-check of this table, and it must print nothing unaccounted for.

---

## 3. How the frontend spells it — intent, defaults, deprecations

Read from `SdpaAttributes.hpp` alongside the `.fbs`.

**Shape convention (header lines 38-45).** Q `(B, H, S_q, D)`, K `(B, H_k, S_kv, D)`,
V `(B, H_v, S_kv, D_v)`, O `(B, H, S_q, D_v)` — **BHSD dim order always**, with memory layout
carried in the *strides*, never in the dim order. `TensorLayout::BHSD` vs `BSHD` selects the
stride pattern, not a different dim tuple. This matters enormously: the kernel is token-major
(BSHD memory) and takes **no stride kernargs**, so a BHSD-strided graph would be indexed as
if it were BSHD and read in-bounds garbage. The layout check is load-bearing, exactly as in
`Gfx950AttentionDenseNative.cpp:278-301`.

**Causality is derived, and the deprecated booleans take precedence.** hipDNN has no `causal`
field. It has `left_bound`/`right_bound` (diagonal band), `diagonal_alignment`, and the two
**deprecated** booleans `causal_mask`/`causal_mask_bottom_right` (marked `// Deprecated` at
`.fbs:70-71` and in the header). The canonical precedence rule lives in the incumbent engine's
`SdpaPlanUtils.hpp:95-100`: **the deprecated boolean wins and the enum is ignored when the
boolean is set.**

> **Carried forward from the dense sibling as an open contract question, not re-litigated
> here.** The dense run found 116 of 428 `cudnn_attention_inference` graphs setting
> `causal_mask: true` *and* `diagonal_alignment: BOTTOM_RIGHT`, all with `Sq != Skv`, exactly
> where hipDNN's boolean-first rule and PyTorch's enum-first rule disagree. This engine
> mirrors hipDNN faithfully (same rule, same header) and inherits the same ambiguity. Step-9
> follow-on row, with the dense finding referenced.

**`max_seq_len_kv` is the paged-attention KV bound**, not a generic cap — the setter is
literally `set_paged_attention_max_seq_len_kv(int32_t)` (header `:646`), and the header's
`logicallyEqualsImpl` comment (`:677`) calls it "paged-attention limits". It is the only
scalar in the table that exists *for* paging.

**Framework cross-check.** `torch.nn.functional.scaled_dot_product_attention` is the operator
8e validates against. It has **no** paged-KV argument at all — paging is a serving-runtime
concept (vLLM), which is why the schema carries it as tensor UIDs rather than as a scalar.
That absence is what makes §5's G2 a *derivation* rather than a copy.

---

## 4. What a real graph looks like — and where the sources disagree

### 4.1 In-tree bundles

`integration-test-bundles/quick/SdpaFwd` = 38 cases; `standard/SdpaFwd` = 26. Path shape
`<layout>/<dtype>/<case>/<tier>/`. A representative one, read rather than assumed
(`quick/SdpaFwd/bhsd/bf16/hd128_causal_batch/Small/Small.json`):

```
T Q [2, 4, 256, 128]  strides [131072, 32768, 128, 1]  bfloat16
T K [2, 4, 256, 128]  strides [131072, 32768, 128, 1]  bfloat16
T V [2, 4, 256, 128]  strides [131072, 32768, 128, 1]  bfloat16
T O [2, 4, 256, 128]  strides [131072, 32768, 128, 1]  bfloat16
NODE SdpaAttributes
  causal_mask=false, causal_mask_bottom_right=false,
  left_bound=-1, right_bound=0, diagonal_alignment=BOTTOM_RIGHT,
  attn_scale_value=0.0883883, max_seq_len_kv=null,
  implementation=AUTO, mma_core_mode=float
```

Three things this settles:

1. **The modern spelling is live.** This bundle expresses causality through
   `left_bound=-1, right_bound=0` with both deprecated booleans `false` — the opposite
   convention from the model traces the dense run found. A matcher reading only the booleans
   computes "not causal" here, which is a wrong answer, not a decline.
2. **Strides are BHSD-ordered** (`128` on axis 2 = S, `32768` on axis 1 = H): S varies faster
   than H, so this is head-major memory. The `bhsd` path component is honest.
3. **Every shipped SdpaFwd bundle is DENSE.** None populates `page_table_*` or `seq_len_*`
   (grepped across all 64). A tiled matcher declines all 64, and a suite of 64 SKIPs exits 0.

**So the shipped bundles are not a coverage plan for this engine.** New bundles are owed at
8a, and this is written down here rather than discovered at stage 8.

### 4.2 Benchmarking workloads

Deferred to 8e for the full three-bucket triage against an enumerated denominator. What is
already known and relevant to sizing: the dense corpus mined from these carries
`hdim_q ∈ {128, 64}` and **no 256 at all**, no `block_size`, and no `num_seqs`. The D256
cohort that decision D5's composition question is *about* is therefore unreachable from the
dense corpus — which is why a tiled miner (P1) is prerequisite work, not an afterthought.

### 4.3 Where the two sources disagree — the three axes

| axis | in-tree bundles | model traces (dense run's finding) |
|---|---|---|
| **mask spelling** | modern `left_bound`/`right_bound`, booleans false | deprecated `causal_mask: true` + `BOTTOM_RIGHT` |
| **memory layout** | both `bhsd` and `bshd` trees ship | production traces favour token-major |
| **shape magnitude** | S=256, B=2, H=4 (toy) | orders of magnitude larger |

All three matter, and the matcher must handle **both** mask spellings or it passes the whole
suite and mis-serves every production graph.

---

## 5. The disagreement table — what step 6 consumes

One row per field the kernel pins. `Kind` is: *same name*, *different name*, **derivation**,
**no hipDNN field**, or **no rocKE field**.

| Kernel field / kernarg | hipDNN spelling | Kind | Note |
|---|---|---|---|
| `head_size` | `Q.dim[3]` | derivation | must equal `K.dim[3]`; V's may differ (`D_v`), and the kernel requires `D_v == D` |
| `num_query_heads` | `Q.dim[1]` | different name | |
| `num_kv_heads` | `K.dim[1]` | different name | GQA ratio `num_queries_per_kv = Hq/Hkv` |
| **`block_size`** | **`K.dim[2]` when paged** | **DERIVATION — the highest-risk row** | see G2 below |
| `num_seqs` | `Q.dim[0]` (batch) | derivation | count of sequences; also bounds the block-table guard |
| `total_q` | `Q.dim[0] * Q.dim[2]` | derivation | varlen: sum of per-seq q lengths |
| `dtype` | Q/K/V `data_type` | different name | `bf16`/`fp16` only; spelling normalises (`bf16` ↔ `BFLOAT16`) |
| `sliding_window` | `left_bound` | derivation | `sliding_window = -left_bound` when a finite left bound is set; `0` = unbounded |
| causal | `left_bound`/`right_bound`/`diagonal_alignment`, deprecated booleans first | derivation | §3; both spellings must be handled |
| `use_sinks` | `sink_token_tensor_uid` present | derivation | presence, not a boolean |
| `use_alibi` | `alibi_mask` | different name | |
| `block_tables_ptr` | `page_table_k_tensor_uid` | different name | required, not optional, for this kernel |
| `block_table_stride` | `page_table_k.stride(0)` | derivation | **elements, not bytes** — an i32 row stride |
| `seq_lens_ptr` | `seq_len_kv_tensor_uid` | different name | |
| `query_start_len_ptr` (cu_q) | `seq_len_q_tensor_uid` | derivation | cumulative, not per-seq — an exclusive prefix sum |
| `scale` | `attn_scale_value` | different name | scalar only; the *tensor* spelling is declined |
| `has_softcap` | — | **no hipDNN field** | **gap G1** |
| `use_qq_bias` / `qq_bias_ptr` | — | **no hipDNN field** | gap G3 — ship `False` only |
| `k_scale`/`v_scale`/`out_scale` | fp8 descale UIDs | **no rocKE field** in v1 | fp8 declined; kernargs passed as 1.0 |
| `use_fp8_*`, `kv_storage_dtype` | descale/scale UIDs | derivation | declined in v1 |
| — | `block_mask_tensor_uid` | **no rocKE field** | must be explicitly rejected |
| — | `dropout_*`, `seed`, `offset`, `rng_dump` | **no rocKE field** | must be explicitly rejected |
| — | `stats`/`max`/`sum_exp` | **no rocKE field** (2D) | must be explicitly rejected |
| — | `attn_mask_tensor_uid` (additive bias) | **no rocKE field** | must be explicitly rejected; NOT the same as `qq_bias` |
| — | `implementation == COMPOSITE` | **no rocKE field** | must be explicitly rejected |
| — | `max_seq_len_kv` | consumed as a bound | see G2's corroboration role |

### G1 — `has_softcap` has no schema field

`has_softcap` is a **required** field on `UnifiedAttention2DTiledSpec` with no default;
`SdpaAttributes` carries no softcap attribute anywhere (all 41 fields enumerated above).
A graph therefore *cannot* ask for softcap, so `has_softcap=False` is the only reachable
value. Ship the `False` leg only. This costs nothing — the alternative is unexpressible —
and it is a **schema gap** to report at stage 9, not an integration gap. The kernarg
`softcap` (slot 15) is still passed, as `0.0f`.

Same reasoning applies to `use_qq_bias` (G3): no schema field, so `False` only. Note
`qq_bias` is a *query-query* bias, distinct from `attn_mask_tensor_uid`'s additive
query-key bias — rejecting one does not reject the other, and conflating them would admit
graphs carrying a bias the kernel silently ignores.

### G2 — the page-table → `block_size` derivation. RESOLVED.

`block_size` is required with no default on the tiled spec, and `SdpaAttributes` has no
page-size scalar. The plan marked this `[INFERENCE], unresolved` and called it the highest-risk
row in the integration: get it wrong and the kernel indexes the KV cache with the wrong stride
and returns silently wrong numbers. **It is now resolved, from three independent sources that
agree.**

**The answer: `block_size = K.dim[2]`. The K/V tensor IS the paged container.**

It is **not** derived from the page table's dims or strides. The page table supplies only
*which* block, never *how large* a block is.

**Source 1 — the kernel's own layout, stated and stride-proven.**
`attention_tiled_2d.py:1936-1938` states the cache layout outright as
`[num_blocks, BS, NUM_KV, HD]`, and the byte strides at `:1884-1887` prove the dim order:

```python
kv_stride_blk_b = BS * NUM_KV * HD * KV_BYTES   # bytes per physical block
kv_stride_tok_b = NUM_KV * HD * KV_BYTES        # bytes per token WITHIN a block
kv_stride_h_b   = HD * KV_BYTES                 # bytes per kv-head
#                 KV_BYTES                        head_size is the unit-stride axis
```

fed into `TensorDescriptor.naive(..., coord_names=("physical_block","token","kv_head","dim"))`
at `:2044-2048`. Strides descend `blk > tok > kv_head > dim`, so the container's axis 1 —
the axis hipDNN calls `S_kv` — **is** `block_size`. The block-table lookup is a separate,
orthogonal transform: `physical_block = block_tables[seq_idx*bt_stride + tile_idx]`
(`:1943-1945`), which resolves axis 0 only.

**Source 2 — the cuDNN convention hipDNN deliberately mirrors.** The frontend header is
explicit that hipDNN's SDPA is close to cuDNN's, and cuDNN's paged-attention contract is:
the K container has shape `[num_blocks, page_size, num_heads, head_dim]`, the page table has
shape `[batch, num_blocks_per_seq]`, and the access rule is

$$K_{cache}[b,h,s,d] = K[\,\text{page\_table\_k}[b, s / bs_k]\,,\; h,\; s \bmod bs_k,\; d\,]$$

The divisor $bs_k$ is a property of the **container**, not of the table. hipDNN carries the
same two tensor UIDs, the same `max_seq_len_kv` scalar, and the same setter *names*
(`set_paged_attention_k_table`, `set_paged_attention_max_seq_len_kv`), so it is the same
contract.

**Source 3 — hipDNN's own node validator, by what it does NOT exempt.**
`SdpaFwdNode.hpp` `pre_validate_node()` enforces, unconditionally:

- Rule 1 (`:53-69`): Q, K, V are **exactly rank-4**.
- Rule 3 (`:90-97`): `K.dim[2] == V.dim[2]` — "seq_kv mismatch between K and V".

`grep -c` for `paged|page_table|Page_table` over that entire file returns **0** (positive
control: the same pattern returns 6 in `SdpaAttributes.hpp`, so the pattern works and the zero
is real). There is **no paged exemption** to either rule. A paged graph is therefore still a
rank-4 K with a meaningful axis 2, and rank-4 with `[num_blocks, page_size, H_k, D]` is
precisely the cuDNN container. If `block_size` had instead been a page-table property, the
container would need rank 4 *plus* a separate logical `S_kv`, which rule 3 gives it nowhere to
put.

**The three agree, so the derivation is not an inference.** Recording the residual caveat
honestly: no *shipped hipDNN artifact* exercises a paged graph end to end — no in-tree bundle
populates `page_table_*`, both reference executors decline it, and the incumbent ASM engine
rejects it (`vllm-integration-techniques.md:124`). So this contract is established by the
schema, the kernel and the cuDNN convention, and will be **confirmed executably** by the first
paged bundle at 8a and by the D1 `GpuSdpaFwdPlan` repair, which must implement exactly this
gather to be correct.

**The matcher's obligations, which follow directly and are all load-bearing:**

1. `block_size = K.dim[2]`, and it must be one of `{16, 32, 64}` (the spec's hard set).
   A container whose axis 2 is anything else is a **decline**, never a rounded value.
2. `K.dim[2] == V.dim[2]` — hipDNN's rule 3 already guarantees it, but the kernel shares one
   `BS` between both caches, so re-assert rather than inherit.
3. `K.dim[1] == V.dim[1] == num_kv_heads`, and `K.dim[3] == head_size`.
4. `page_table_k` and `page_table_v` must **both** be present, rank-2, and shape-equal.
   The kernel has ONE `block_tables_ptr` serving both caches, so two differing tables are
   unservable — and this is exactly the kind of row that reads as a kernel bug at stage 8.
5. `block_table_stride = page_table_k.stride(0)`, in **elements**. Confirmed against the
   host-side computation at `attention_unified.py:4181-4184`, which reads a PyTorch
   `.stride(0)` (an element count) and falls back to `shape[1]`. A byte stride here is a
   4× indexing error into the KV cache — silently wrong numbers, not a fault.
6. **Corroborate with `max_seq_len_kv` when the graph sets it.** If present, it must satisfy
   `max_seq_len_kv <= page_table_k.dim[1] * K.dim[2]`. The graph is then stating its own
   page geometry, and a disagreement means one of the two readings is wrong — decline rather
   than pick a side. This is free redundancy on the single most dangerous row in the
   integration and it is why the row is no longer marked `[INFERENCE]`.

### The layout row, restated because it is the other silent one

The kernel takes **no stride kernargs** for Q/O and bakes a token-major layout. hipDNN dims
are always BHSD with layout in the strides (§3). So Q and O must be verified dense-BSHD
stride-wise, unit-extent axes exempted, exactly as `Gfx950AttentionDenseNative.cpp:278-301`
does. K/V are the paged container and follow the four-axis rule above instead.

---

## GATE

- `ls graph_contract.md` — present.
- §1 names the node (`SdpaAttributes`) and states there are no UID edges.
- §2 accounts for all 41 schema fields, consumed or explicitly rejected.
- §4 records both sources and the three axes on which they disagree.
- §5 has a row per pinned kernel field, and **G2 is resolved with three concurring sources
  plus six derived matcher obligations** — the row the plan left open.
