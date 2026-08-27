# graph_contract.md — gfx942 dense attention as a hipDNN operation

Produced at RUNBOOK step 2a, per `graph-contract.md`. Five sections.

Kernel: `rocke/library/kernels/gfx942/attention_dense.py`, `build_attention_dense`.

---

## 1. The operation match

**Outcome: one node matches.** `NodeType::SDPA_FWD` (enum value 12), table
`SdpaAttributes` in `flatbuffers_sdk/schemas/sdpa_attributes.fbs`.

Evidence, by the file's own method (semantics, not vocabulary): the kernel computes
`O = softmax(scale · Q·Kᵀ + mask) · V` over `(B, S, H, D)` operands with one output
tensor — exactly `SDPA_FWD`'s mathematics. `_run_work_item` walks (query-block,
query-head, batch), loads Q/K/V, applies a causal clamp, an online softmax, and a PV
epilogue writing one O. No epilogue beyond the O store; nothing fused on either end.

Composition check run as instructed (nine chains printed on this tree):

```
  4  BatchnormAttributes -> PointwiseAttributes
  2  ConvolutionFwdAttributes -> PointwiseAttributes
  2  ConvolutionFwdAttributes -> PointwiseAttributes -> PointwiseAttributes
  2  BatchnormInferenceAttributes -> PointwiseAttributes
  2  BatchnormInferenceAttributesVarianceExt -> PointwiseAttributes
  2  BatchnormInferenceAttributes -> PointwiseAttributes -> BatchnormBackwardAttributes
  2  MatmulAttributes -> PointwiseAttributes
  1  MatmulAttributes -> PointwiseAttributes -> PointwiseAttributes
  1  BlockScaleDequantizeAttributes -> BlockScaleDequantizeAttributes -> MatmulAttributes
```

No SDPA chain ships. This kernel is a single-node match, so no UID edges to record —
`graph_match` opens with `nodeCount() == 1 && type == SDPA_FWD`.

**UID edges:** none. Inputs q/k/v are graph inputs; o is a graph output.

---

## 2. The field audit — every field of `SdpaAttributes`

Three categories only: **consumed**, **explicitly rejected**, **must be absent/default**.
An unchecked field is accepted and silently not honoured.

### Input tensor UIDs

| Field | Disposition | Why |
|---|---|---|
| `q_tensor_uid` | **consumed** | required; dims/strides drive shape + layout match |
| `k_tensor_uid` | **consumed** | required |
| `v_tensor_uid` | **consumed** | required |
| `o_tensor_uid` | **consumed** | required output |
| `attn_mask_tensor_uid` | **reject if set** | kernel has no additive-bias input; ABI is 5 args (q,k,v,o,scale) |
| `scale_tensor_uid` | **reject if set** | kernel takes scale as a *scalar kernarg*, not a device tensor. A tensor-valued scale is a runtime pointer the ABI has no slot for. |
| `seq_len_q_tensor_uid` | **reject if set** | varlen — `supports_attention_dense` rejects |
| `seq_len_kv_tensor_uid` | **reject if set** | varlen |
| `seed_tensor_uid` | **reject if set** | dropout; no such path |
| `offset_tensor_uid` | **reject if set** | dropout |
| `dropout_mask_tensor_uid` | **reject if set** | dropout |
| `dropout_scale_tensor_uid` | **reject if set** | dropout |
| `page_table_k_tensor_uid` | **reject if set** | paged KV. **NB:** the kernel's `paged` spec field is *not* validated by `supports_attention_dense` (see §5), so the graph side is the only place this is caught. |
| `page_table_v_tensor_uid` | **reject if set** | paged KV, same |
| `block_mask_tensor_uid` | **reject if set** | block-sparse |
| `sink_token_tensor_uid` | **reject if set** | sinks — `supports_attention_dense` rejects `use_sinks` |
| `descale_q/k/v/s_tensor_uid` | **reject if set** | FP8; kernel is bf16/fp16 only |
| `scale_s_tensor_uid`, `scale_o_tensor_uid` | **reject if set** | FP8 |

### Output tensor UIDs

| Field | Disposition | Why |
|---|---|---|
| `stats_tensor_uid` | **reject if set** | kernel writes only O; no LSE output in the ABI |
| `max_tensor_uid` | **reject if set** | no logit-max output |
| `sum_exp_tensor_uid` | **reject if set** | no sum-exp output |
| `rng_dump_tensor_uid` | **reject if set** | dropout debug |
| `amax_s_tensor_uid`, `amax_o_tensor_uid` | **reject if set** | FP8 |

### Boolean flags

| Field | Disposition | Why |
|---|---|---|
| `generate_stats` | **reject if true** | same as `stats_tensor_uid`; two spellings of one feature — rejecting only one admits the other |
| `alibi_mask` | **reject if true** | no ALiBi slope path in the body |
| `padding_mask` | **reject if true** | padding implies per-sequence valid lengths the dense kernel does not read |
| `causal_mask` | **consumed** (deprecated) | feeds the causality derivation — see §5 |
| `causal_mask_bottom_right` | **reject if true** | see §5: the kernel's causal clamp is TOP-LEFT only |

### Scalars / enums

| Field | Disposition | Why |
|---|---|---|
| `dropout_probability` | **reject if set and non-zero** | no dropout |
| `attn_scale_value` | **consumed** | becomes the `scale` f32 kernarg. Absent ⇒ caller must supply via `scale_tensor_uid`, which we reject ⇒ decline. |
| `left_bound` | **consumed** | causality derivation |
| `right_bound` | **consumed** | causality derivation |
| `max_seq_len_kv` | **reject if set** | paged-attention only (frontend setter is literally `set_paged_attention_max_seq_len_kv`) |
| `diagonal_alignment` | **consumed** | causality derivation; only `TOP_LEFT` servable |
| `mma_core_mode` | **consumed-as-constraint** | `UNSET` or matching the kernel's compute type; anything else is a request for a core mode the builder does not parameterise |
| `implementation` | **consumed-as-constraint** | `AUTO` accepted; a specific `AttentionImplementation` naming another backend must be declined. *(Note: the runbook itself records that the shipped `AttentionDenseNative.cpp` still prints `UNCHECKED: implementation`, i.e. the incumbent gets this wrong.)* |

Also outside the attribute table but part of the contract: **tensor dtype** and
**tensor strides** (from `tensor_attributes.fbs`), both load-bearing — see §5.

---

## 3. The frontend reading (`attributes/SdpaAttributes.hpp`)

- Class doc gives shapes: Q `(B,H,S_q,D_qk)`, O `(B,H,S_q,D_v)`, Stats `(B,H,S_q,1)`.
  **"Memory layout is controlled via strides. Use `TensorLayout::BHSD` for row-major or
  `TensorLayout::BSHD` for sequence-major."** The *logical* dim order is always
  `(B,H,S,D)`; layout is expressed purely in strides. This is the single most important
  sentence on the graph side for this kernel (§5).
- `causal_mask` and `causal_mask_bottom_right` are marked `// Deprecated` in both the
  header and the `.fbs`.
- `set_attn_scale` is **overloaded**: `shared_ptr<TensorAttributes>` (tensor) and
  `float` (scalar → `attn_scale_value`). Two different fields behind one setter name.
  A matcher reading only one admits graphs carrying the other.
- cuDNN shim aliases that redirect into fields we care about:
  `set_is_inference(b)` → `set_generate_stats(!b)`;
  `set_sliding_window_length(n)` → `set_diagonal_band_left_bound(n)`.
  So a caller who never mentions `left_bound` can still set it.
- `set_paged_attention_max_seq_len_kv` names `max_seq_len_kv`'s intent explicitly.
- The FMA-unfuse hint is recorded and warned-and-ignored by `Graph::sdpa` — not our
  concern.

Header vs schema: no disagreement found. Both carry the same field set with the same
defaults.

---

## 4. A real graph

Read: `integration-test-bundles/quick/SdpaFwd/bhsd/bf16/hd128_causal_batch/Small/Small.json`.

One node, `SdpaAttributes`, `compute_data_type: float`. Four tensors:

| uid | name | dims | strides | dtype |
|---|---|---|---|---|
| 0 | Q | `[2, 4, 256, 128]` | `[131072, 32768, 128, 1]` | bfloat16 |
| 1 | K | `[2, 4, 256, 128]` | `[131072, 32768, 128, 1]` | bfloat16 |
| 2 | V | `[2, 4, 256, 128]` | `[131072, 32768, 128, 1]` | bfloat16 |
| 3 | O | `[2, 4, 256, 128]` | `[131072, 32768, 128, 1]` | bfloat16 |

Attributes as shipped for the *causal* case:

```
causal_mask: false, causal_mask_bottom_right: false      <- BOTH deprecated flags OFF
left_bound: -1, right_bound: 0
diagonal_alignment: "BOTTOM_RIGHT"
attn_scale_value: 0.08838834764831843                    <- 1/sqrt(128)
generate_stats: null, padding_mask: false, alibi_mask: false
implementation: "AUTO", mma_core_mode: "float"
```

Two things this graph proves that reading the schema alone would not:

1. **A "causal" graph on this tree sets neither causal boolean.** The name
   `hd128_causal_batch` is causal entirely via `left_bound=-1, right_bound=0`. A matcher
   keyed on `causal_mask()` computes "not causal" for this graph and would then serve a
   non-causal kernel for it — the exact silent-wrong-answer defect `graph-contract.md`
   §5 warns about, present in the shipped test data.
2. **The shipped bundles' `diagonal_alignment` is `BOTTOM_RIGHT`, not `TOP_LEFT`.**
   With `Sq == Skv` the two coincide mathematically, so this is invisible here and
   divergent the moment `Sq != Skv`.

Strides decode: `stride_h = 32768 = 256·128 = S·D`, `stride_s = 128 = D`. So the
shipped bundles are **BHSD**.

---

## 5. The disagreement table

The deliverable step 6 consumes. Rows for every field the kernel pins, plus the
graph-side fields that have no kernel counterpart.

| Kernel field | hipDNN spelling | Kind | Note |
|---|---|---|---|
| `batch` | Q dims[0] | different name | direct |
| `seqlen_q` | Q dims[2] | different name | logical dim order is `(B,H,S,D)` regardless of layout |
| `seqlen_kv` | K dims[2] | different name | |
| `num_query_heads` | Q dims[1] | different name | |
| `num_kv_heads` | K dims[1] | different name | GQA group = `Hq/Hkv`; kernel emits `sdiv` by it |
| `head_size` | Q dims[3] | different name | kernel requires `D_qk == D_v`; hipDNN allows them to differ (O dims[3]) — **must check both** |
| `dtype` (`"bf16"`/`"fp16"`) | tensor `data_type` (`bfloat16`/`half`) | different name | all four tensors must agree; kernel has one dtype for q/k/v/o |
| `causal` (bool) | **no hipDNN field** → **derivation** | **derivation** | See below. hipDNN has no `causal` boolean; causality is derived from `causal_mask` / `causal_mask_bottom_right` / (`left_bound`,`right_bound`,`diagonal_alignment`). |
| `scale` (f32 kernarg, not a spec field) | `attn_scale_value` | different name | kernel pre-multiplies by `LOG2E` internally; the graph value passes through unscaled |
| — (implicit) | tensor **strides** | **derivation** | **The layout disagreement. See below.** |
| `varlen` | `seq_len_q/kv_tensor_uid` | no rocKE support | kernel rejects; must decline |
| `ragged` | tensor `ragged_offset_tensor_uid` | no rocKE support | kernel rejects; must decline |
| `sliding_window` | `left_bound`/`right_bound` other than the causal/no-mask pairs | no rocKE support | kernel rejects; must decline |
| `use_sinks` | `sink_token_tensor_uid` | no rocKE support | kernel rejects; must decline |
| `paged` | `page_table_k/v_tensor_uid` | **no rocKE support, AND unvalidated in the kernel** | `supports_attention_dense` never inspects `spec.paged`. Ship every variant with `paged: false` and decline the graph fields. |
| `block_size`, `num_kv_blocks` | paged-KV geometry | no hipDNN field for us | paged-only; pin to 0 |
| `block_n` | **no hipDNN field** | tuning knob | KV tile width; exposed as a UED knob candidate |
| `block_m` | **no hipDNN field** | tuning knob (gfx942-private) | query tile; 256 default |
| `waves_per_eu`, `lds_k_group_pad`, `lds_row_pad`, `v_row_pad`, `iglp`, `use_cfvst`, `use_v_swizzle`, `use_exp2_fast`, `lazy_rescale`, `interleave` | **no hipDNN field** | tuning knobs | codegen-internal; never graph-visible |
| `persistent`, `num_persistent`, `persist_decode` | **no hipDNN field** | tuning knobs | grid strategy |
| — | `attn_mask_tensor_uid` | **no rocKE field** | decline |
| — | `generate_stats` / `stats_tensor_uid` / `max` / `sum_exp` | **no rocKE field** | decline |
| — | `dropout_*`, `seed`, `offset`, `rng_dump` | **no rocKE field** | decline |
| — | `alibi_mask` | **no rocKE field** | decline |
| — | `padding_mask` | **no rocKE field** | decline |
| — | `descale_*`, `scale_s`, `scale_o`, `amax_*` | **no rocKE field** | decline (FP8) |
| — | `max_seq_len_kv` | **no rocKE field** | decline |
| — | `mma_core_mode` | **no rocKE field** | accept only `UNSET`/`float` |
| — | `implementation` | **no rocKE field** | accept only `AUTO` |
| — | `scale_tensor_uid` | **no rocKE field** | decline: ABI has a scalar scale, not a pointer |

### Derivation 1 — causality

hipDNN has **no** `causal` boolean. The authority is the incumbent's
`asm_sdpa_engine/plans/SdpaPlanUtils.hpp::getMaskType`, read as `graph-contract.md`
§5 instructs (an incumbent engine has almost certainly solved the same derivation):

```
causal_mask && causal_mask_bottom_right  -> throw INVALID_VALUE (mutually exclusive)
causal_mask                              -> TOP_LEFT_CAUSAL      (deprecated wins)
causal_mask_bottom_right                 -> BOTTOM_RIGHT_CAUSAL  (deprecated wins)
left  = left_bound.value_or(-1)
right = right_bound.value_or(-1)
left == -1 && right == -1                -> NO_MASK
left == -1 && right ==  0                -> diagonal_alignment == BOTTOM_RIGHT
                                              ? BOTTOM_RIGHT_CAUSAL : TOP_LEFT_CAUSAL
otherwise                                -> SLIDING_WINDOW
```

Cross-checked against the reference executor
(`gpu-ref/kernels/sdpa/GpuRefSdpaFwd.cpp`), which is what stage 8 verifies against:

```
windowOffset = topLeftAlignment ? 0 : (seqKv - seqQ)
if (rightBound >= 0) mask skv >= sq + 1 + windowOffset + rightBound
if (leftBound  >= 0) mask skv <  sq + windowOffset - leftBound
```

So `rightBound == 0` masks everything strictly right of the diagonal — causal — and
`leftBound == -1` leaves the left side unbounded. The two definitions agree.

**Mapping for this engine:**

| Derived `MaskType` | kernel `spec.causal` | Servable? |
|---|---|---|
| `NO_MASK` | `false` | yes |
| `TOP_LEFT_CAUSAL` | `true` | yes |
| `BOTTOM_RIGHT_CAUSAL` | — | **decline**, except when `Sq == Skv` where it is identical to top-left |
| `SLIDING_WINDOW` | — | **decline** |

The kernel's clamp is top-left: the causal KV-loop bound is derived from
`n_per = block_m // block_n` against the query-block index with no `Skv - Sq` offset
term, so it is `windowOffset == 0`. Serving a `BOTTOM_RIGHT` graph at `Sq != Skv` would
be a silent wrong answer.

**`Sq == Skv` carve-out is deliberate and load-bearing**: every shipped SdpaFwd causal
bundle on this tree sets `diagonal_alignment: BOTTOM_RIGHT` with `Sq == Skv`. Declining
`BOTTOM_RIGHT` outright would decline every shipped causal test. Accept it *only* when
`Sq == Skv`.

### Derivation 2 — layout. The one that would have shipped wrong numbers.

The frontend fixes the logical dim order at `(B, H, S, D)` and expresses layout
**entirely in strides**.

**The kernel is BSHD.** From `_build_attention_dense_single_buffer`:

```python
stride_q_tok = Hq * D          # Q/O: advancing one TOKEN steps over all heads
stride_k_tok = Hkv * D         # K/V
q_base = bt*Sq*stride_q_tok + hq*D
k_base = bt*Skv*stride_k_tok + hkv*D
addr   = q_base + q_tok*stride_q_tok + col
```

Advancing the token index steps by `Hq·D`, and the head index steps by `D`. That is
`[B, S, H, D]` contiguous — **BSHD**.

**The shipped bundles are BHSD**: strides `[131072, 32768, 128, 1]` on dims
`[2, 4, 256, 128]`, i.e. `stride_h = S·D` and `stride_s = D`.

There is no stride argument in the ABI (`attention_dense_signature` is
`q_ptr, k_ptr, v_ptr, o_ptr, scale`) — the layout is **baked into the emitted code**.
Consequences, both mandatory:

1. `graph_match` **must verify the strides**, per tensor, against the BSHD pattern:
   `stride_d == 1`, `stride_s == H·D`, `stride_h == D`, `stride_b == S·H·D`.
   A matcher that checks only dims accepts every shipped BHSD bundle and computes
   garbage with no error. **This is the silent-wrong-answer defect for this kernel.**
2. Step 8a's warning applies literally: **the existing `SdpaFwd/bhsd/**` bundles cannot
   exercise this engine.** New `SdpaFwd/bshd/**` bundles are required, or the stage-8
   run tests a graph the engine correctly declines and proves nothing.

Note the kernel also requires `D_qk == D_v` and one dtype across all four tensors;
hipDNN permits both to vary.

---

## GATE

`graph_contract.md` exists. §1 names the node (`SDPA_FWD` / `SdpaAttributes`) and
records that there are no UID edges. §2 accounts for every field of `SdpaAttributes`.
§5 has a row per pinned kernel field plus the graph-only fields. Disagreement table has
rows; two derivations (causality, layout) written out.
