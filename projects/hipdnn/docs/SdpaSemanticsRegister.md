# SDPA semantics register

hipDNN's SDPA contract is **cuDNN's SDPA semantics**. This register records every semantic choice where the
use cases disagree: cuDNN, PyTorch (SDPA, AOTriton, FlexAttention), vLLM, and FlashAttention-style kernels.
For each entry it says what hipDNN does today, what the contract is, and where the gaps are.

The CPU and GPU reference implementations are the executable form of this contract. A change to an entry's
**Contract** column must land together with the reference changes and the tests that pin it.

Status values:
- **Aligned**: hipDNN matches cuDNN today.
- **Diverges**: hipDNN differs from cuDNN today. The entry tracks the fix.
- **Extension**: cuDNN does not define the behaviour, and hipDNN defines it.
- **Adapter**: the framework integration converts; no hipDNN change is needed.

Notation: query row `i`, key column `j`, `off = 0` for TOP_LEFT and `off = Skv - Sq` for BOTTOM_RIGHT (per
batch when `seq_len_q/kv` are set).

## Masking and alignment

### SEM-001: Meaning of `left_bound`

- **Status:** Diverges.
- **cuDNN:** `left_bound = L` keeps `j > i + off - L`, so each row sees **L** keys including the diagonal.
  `set_sliding_window_length(L)` is the same field.
  - Sources: cudnn-frontend `include/cudnn_frontend/utils/attn_score_modifiers.h:207-247` (retain
    `col + L > row`), `test/python/sdpa/fp16_ref.py:132-147`.
- **hipDNN today:** keeps `j >= i + off - L`, which is **L + 1** keys.
  - The CPU reference (`CpuFpReferenceSdpa.hpp` `isMasked`), the GPU reference (`GpuRefSdpaFwd.cpp`) and the
    golden generator (`generate_sdpa_fwd_golden.py` `build_mask`) all use it.
  - So do the hip-kernel-provider window mapping (`asm_sdpa_engine/plans/SdpaPlanUtils.hpp`) and rocKE
    (`rocke/helpers/attention_band.py`).
  - The convention is consistent inside hipDNN, so no test catches it. A cuDNN-shim user gets one extra key.
- **Other use cases:**
  - FlashAttention, AITER, AOTriton, TransformerEngine and FlashInfer: `window_left = L - 1`, the current
    hipDNN meaning.
  - vLLM `sliding_window = W` keeps W keys including self, so it maps to `L = W`.
- **Contract:** cuDNN's meaning.
- **Fix needs a coordinated change** across the references, golden data and every provider window mapping.
- **Deferred** pending a team decision. Until then, references and tests keep the current hipDNN meaning, and no
  window tests are added that assume either convention.

### SEM-002: Deprecated `causal_mask` / `causal_mask_bottom_right` combined with explicit bounds

- **Status:** Aligned in the hipDNN references; diverges in the rocKE oracle.
- **cuDNN:** the setters compose. The causal setter sets `right_bound = 0` (and BOTTOM_RIGHT for
  `_bottom_right`). Later bound and alignment setters refine it.
- **hipDNN today:**
  - The CPU executor and the dense and ragged GPU references all compose this way, through `PlanUtils.hpp`
    `extractDiagonalBandParams`, as the frontend documents (`SdpaAttributes.hpp:128-135`).
  - The ragged GPU reference forced `(-1, 0)` until this was fixed. Pinned by
    `TestGpuSdpaRaggedFwdPlanDeprecatedCausal`.
  - The rocKE `sdpa_reference.py` (hip-kernel-provider) still forces `(-1, 0)`.
- **Contract:** compose, as cuDNN does. All references use `extractDiagonalBandParams` semantics.

### SEM-003: Causal alignment

- **Status:** Aligned (adapter).
- **cuDNN:** explicit `diagonal_alignment`.
- **PyTorch:** dense `is_causal` is TOP_LEFT. The flash and AOTriton ops and `CausalBias(LOWER_RIGHT)` use
  BOTTOM_RIGHT.
- **vLLM:** always BOTTOM_RIGHT (chunked prefill, prefix caching, decode).
- **Contract:** the framework adapter sets the alignment explicitly.

### SEM-004: Window anchoring

- **Status:** Adapter.
- **cuDNN:** the window is anchored to the same diagonal as `diagonal_alignment`.
- **AOTriton:** numeric windows are top-left anchored even when `Sq != Skv`. That differs from PyTorch
  varlen and FlashAttention, which are bottom-right.
- **Contract:** cuDNN. A PyTorch-on-ROCm result that depended on AOTriton's anchoring changes. Record any
  such reports here.

### SEM-005: Band parameter domain

- **Status:** Diverges (hipDNN is more permissive).
- **cuDNN:** rejects `left_bound <= 0` and `right_bound < 0`. A left bound with `s_q > s_kv` needs padding.
  - Source: survey `~/attn-ref-gaps/04-external-references.md` §3 D3. Re-verify against cudnn-frontend
    validation before the fix lands.
- **hipDNN today:** accepts any bound `>= -1`, with `-1` meaning unbounded.
- **Contract:** to be decided together with SEM-001. Under cuDNN's meaning `left_bound = 0` masks every key,
  so it must be rejected.

### SEM-008: Padding (`seq_len_q` / `seq_len_kv`) and the `padding_mask` flag

- **Status:** Extension (hipDNN is more permissive about the flag); semantics aligned.
- **cuDNN:** `padding_mask = true` requires both seq_len tensors, and seq_len tensors without the flag are an error
  (cudnn-frontend `node/sdpa_support_surface.h:83-88`). Keys past `seq_len_kv[b]` are masked, query rows past
  `seq_len_q[b]` give O = 0 and Stats = -inf (9.14+; 0 before), and bottom-right alignment uses the per-batch
  offset `seq_len_kv[b] - seq_len_q[b]` (`test/python/sdpa/fp16_ref.py`).
- **hipDNN today:** existing graphs (the golden generator and the checked-in `*_group` bundles) send both seq_len
  tensors with `padding_mask = false`. The CPU references and the dense GPU forward reference treat both tensors
  present as padding whatever the flag says, and decline `padding_mask = true` without them or only one tensor (`PlanUtils.hpp`
  `isSupportedSdpaPadding`). Seq_len tensors are INT32 with one value per batch.
- **PyTorch / vLLM:** varlen and padded batches are expressed by lengths (`cu_seqlens`, `seq_lens`); the adapter
  produces seq_len tensors, so the flag question does not reach them.
- **Contract:** cuDNN semantics; the flag is not required. Padding contents are never read.
- **Open:** whether the frontend should set `padding_mask` automatically when seq_len tensors are given, which
  would make hipDNN graphs cuDNN-valid as well.

### SEM-009: Paged KV cache

- **Status:** Aligned (CPU and dense GPU forward references).
- **cuDNN:** K/V are containers `[num_blocks, H, block_size, D]` with arbitrary strides; page tables are INT32
  `[B, 1, table_size, 1]`; logical key `j` of batch `b` is `container[page_table[b, 0, j / block_size, 0], h,
  j % block_size, :]`. Paging requires padding (both seq_len tensors). `max_seq_len_kv` gives the logical key
  extent and is rejected without paging; a bias's last dim must equal it (cudnn-frontend
  `node/sdpa_support_surface.h:275-315`, `samples/cpp/sdpa/fp16_fwd_with_paged_caches.cpp`).
- **vLLM:** `key_cache` / `value_cache` per layer with one shared `block_table [B, max_blocks]` int32; NHD layout
  `[num_blocks, block_size, H, D]` (or HND); unused table entries point at the null block 0; block sizes are not
  limited to powers of two (Qwen3-Next uses 544/784/1056).
- **Conversion:** vLLM's caches map to containers by strides alone (NHD strides `{bs*H*D, D, H*D, 1}`); its block
  table becomes both page tables, reshaped `[B, 1, max_blocks, 1]`; `seq_lens` becomes `seq_len_kv`.
- **hipDNN today:** the CPU forward reference gathers each paged container into a dense tensor, reading only keys
  below `seq_len_kv`, then runs the padded dense path. Pinned by `TestCpuFpReferenceSdpaPropertyPaged` (block sizes
  1/3/4, NHD strides, shuffled blocks, NaN in the null block and unused slots).
- **Frontend:** `SdpaFwdNode` validation no longer compares a paged container's leading dim with Q's batch or its
  dim 2 with the other side's key length; it requires each page table's first dim to be Q's batch and, when
  `max_seq_len_kv` is set, a bias's last dim to match it. Paging support (and its padding requirement) is left to
  the engines. Pinned by `TestSdpaFwdNode.*Paged*` and an end-to-end frontend-built paged graph run on the CPU
  reference.
- **Not covered:** no external paged oracle exists (all external references de-page the same way), so paged
  correctness rests on the gather being an exact rearrangement of the validated dense padded path.

### SEM-007: Boolean attention masks

- **Status:** Adapter.
- **cuDNN:** a boolean bias is rejected ("Bias mask data type cannot be boolean",
  cudnn-frontend `node/sdpa_support_surface.h:78-80`).
- **PyTorch:** `attn_mask` may be bool (True = attend). The math backend converts it to an additive mask of
  0 / -inf before softmax.
- **Contract:** cuDNN. The PyTorch adapter converts a bool mask to a 0 / -inf additive bias (or, for pure
  causal/band patterns, to the diagonal-band attributes). The references need no bool-mask path.

### SEM-006: Bias combined with a band or bottom-right causal

- **Status:** Extension (hipDNN is more permissive).
- **cuDNN:** the support surface rejects these combinations (cudnn-frontend `node/sdpa_support_surface.h`):
  - Bottom-right causal is only supported without bias, ALiBi or dropout (lines 112-114).
  - Left and right bounds are only supported without dropout or bias (lines 247-249).
  - A bias whose last dim is not the KV sequence length is rejected (line 298).
- **hipDNN today:** the CPU and dense GPU references apply an additive bias and then the band, in any
  combination.
- **Contract:** hipDNN keeps the combination as an extension, defined as the bias added to the scaled scores
  followed by the band mask. vLLM and PyTorch need it: chunked prefill with a bias, and windowed models
  with masks.
- **Validation gap:** real cuDNN cannot arbitrate this case, so it rests on the property tests (a band equals a
  -inf bias, so band + bias equals the sum of the two biases).

## Softmax and statistics

### SEM-010: Fully masked or empty rows

- **Status:** Aligned in the hipDNN references.
- **cuDNN:** O = 0 and Stats = -inf (padded query rows since 9.14).
- **PyTorch math, Flex, vLLM and FlashInfer:** O = 0, LSE = -inf.
- **FlashAttention and AOTriton kernels:** LSE = +inf.
- **JAX-XLA:** mean(V).
- **hipDNN today:**
  - All forward references: O = 0, LSE = -inf.
  - CPU backward: such a row contributes zero to every gradient. It previously produced NaN gradients when an
    additive -inf bias masked the whole row. Pinned by `TestCpuFpReferenceSdpaClosedFormDeadRow` (LSE and
    recomputed-softmax paths).
  - No GPU backward reference exists yet.
- **Contract:** O = 0, LSE = -inf, and that row contributes 0 to dQ, dK and dV. Engines whose kernels store
  +inf normalize before returning.

### SEM-011: LSE base, dtype and layout

- **Status:** Aligned.
- **cuDNN:** natural log, fp32, `[B, H, Sq, 1]`.
- **Other use cases:** FlashInfer uses log2. vLLM uses packed `[Hq, T]` for varlen.
- **hipDNN today:** natural log, fp32, `[B, H, Sq, 1]` dense. The ragged GPU reference also accepts packed
  LSE.
- **Contract:** cuDNN. The packed layout follows the ragged-offset model.

### SEM-012: Softmax probability precision in the reference

- **Status:** Diverges between the CPU and GPU references.
- **cuDNN Python reference:** fp32 P.
- **hipDNN today:** for bf16 Q/K/V, the GPU references round P to bf16 (RTNE) before P@V, to mirror the AITER
  kernel (`GpuSdpaFwdPlan.hpp` `sdpaProbabilityMode`). The CPU reference keeps fp32 P.
- **Contract:** to be decided. Recommendation: fp32 P by default so the CPU and GPU references agree, with
  provider rounding available only as an opt-in emulation mode.

### SEM-013: Default scale

- **Status:** Diverges in rocKE.
- **cuDNN:** no scaling unless `attn_scale` is set.
- **PyTorch and vLLM:** default to 1/sqrt(D), but the adapter always passes the scale explicitly.
  vLLM uses per-model scales (Gemma `query_pre_attn_scalar`, DeepSeek `mscale`).
- **hipDNN today:** C++ references and golden data use 1.0. rocKE `sdpa_reference.py` defaults to
  1/sqrt(D).
- **Contract:** cuDNN.

### SEM-014: Bias gradient (dBias)

- **Status:** Aligned (CPU reference); no GPU backward reference yet.
- **cuDNN:** dBias is a reduction (ADD) of dS into the dBias tensor, which has the bias tensor's shape
  (cudnn-frontend `node/scaled_dot_product_flash_attention.h`, `red_dP_dBias`). dS is taken before the attention
  scale because the bias is added after scaling. cuDNN's own tests compare it with autograd for a
  `(1, H, Sq, Skv)` bias (`test/python/sdpa/fp16.py`).
- **PyTorch (AOTriton mem-efficient):** returns the gradient of the expanded `(B, Hq, Sq, Skv)` bias, and autograd
  sums it back to the user's broadcast shape. Same values after that reduction.
- **hipDNN today:** the CPU backward computes dBias in the bias tensor's shape and dtype, summing over broadcast
  dims. Band-masked positions and dead rows give 0. Pinned by finite differences
  (`TestCpuFpReferenceSdpaGradCheckDBias`, mutation-checked against using the scaled dS).
- **Contract:** cuDNN. A dBias whose shape or dtype differs from the bias is declined.

## Score modifiers (extensions and conventions)

### SEM-020: Softcap

- **Status:** Extension (pending API).
- **cuDNN:** no attribute; only possible through score_mod.
- **vLLM (Gemma2) and Flex:** `c * tanh(S / c)` after scaling, before the mask and bias.
- **Contract:** to be defined. Proposed order: scale, then softcap, then bias, then ALiBi, then mask.

### SEM-021: ALiBi

- **Status:** Diverges by convention.
- **cuDNN:** adds `(j - i) * slope`, and requires causal.
- **FlashAttention and TransformerEngine:** `-slope * |i + Skv - Sq - j|`. Under BOTTOM_RIGHT with
  Skv > Sq the output O is the same but the LSE differs.
- **hipDNN today:** no reference. CPU backward silently ignores it.
- **Contract:** cuDNN.

### SEM-022: Attention sinks

- **Status:** Aligned (CPU and dense GPU forward references).
- **Agreed semantics:** one unscaled logit per head, counted in the softmax denominator only, with no V row,
  and included in the LSE. cuDNN shape `[1, H_q, 1, 1]`, fp32 (cudnn-frontend `test/python/sdpa/fp16_ref.py`):
  `M = max(max_j s_j, sink)`, denominator `Σ exp(s_j − M) + exp(sink − M)`.
- **Consequence:** a row with no live key but a finite sink has O = 0 and **LSE = sink**, not -inf. Padded query
  rows stay O = 0, LSE = -inf.
- **hipDNN today:** CPU forward reference implements it for a FLOAT `[1, H, 1, 1]` sink (other dtypes declined;
  vLLM's model-dtype sinks are converted by the adapter). Pinned by `SinkWithUniformScores`,
  `SinkOnFullyMaskedRow` and `SinkEqualsExtraZeroValueKey`.
- **vLLM:** sinks are in the model dtype and kernels upcast them.
- **Open:** the dSink definition. AITER's test uses an LSE without the sink; do not follow it.

## Process

- Add an entry when a reference change, provider or adapter exposes a disagreement between use cases.
- An entry stays open until the references and their tests encode the contract.
