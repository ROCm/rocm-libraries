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

- **Status:** Matches cuDNN (it used to keep one key too many).
- **cuDNN:** `left_bound = L` keeps `j > i + off - L`, so each row sees **L** keys including the diagonal.
  `set_sliding_window_length(L)` is the same field. `left_bound = 0` would keep no keys and is rejected.
  - Sources: cudnn-frontend `include/cudnn_frontend/utils/attn_score_modifiers.h:207-247` (retain
    `col + L > row`), `test/python/sdpa/fp16_ref.py:132-147`.
- **hipDNN:** the same. `left_bound` must be positive or -1 (unbounded); the frontend nodes, the CPU/GPU graph
  plans and the hip-kernel-provider band resolution reject 0 and values below -1.
  - References: `CpuFpReferenceSdpa.hpp` / `CpuFpReferenceSdpaRagged.hpp` `isMasked`, `GpuRefSdpa{Fwd,RaggedFwd,Bwd}.cpp`,
    golden generator `build_mask` (its `--window-left` is the graph's `left_bound`).
  - Kernels count the keys left of the diagonal (FlashAttention `window_size_left`), so providers convert once at the
    boundary: `asm_sdpa_engine/plans/SdpaPlanUtils.hpp` `resolveDiagonalBand` stores `left_bound - 1` (asm forward
    and backward, hip_flash2), and `Gfx1151WmmaAttentionNative.cpp` `maskFor` passes `left_bound - 1` as
    `window_left`. The gfx950 dense pack declines windows.
  - Before the fix every hipDNN component used `j >= i + off - L` (L + 1 keys). It was internally consistent, so no
    test caught it. `TestCpuFpReferenceSdpaFp64.LeftBoundCountsTheDiagonalKey` now pins the count.
- **Other use cases:**
  - FlashAttention, AITER, AOTriton, TransformerEngine and FlashInfer: `window_left = L - 1`.
  - vLLM `sliding_window = W` keeps W keys including self, so it maps to `L = W`.
  - rocKE's own oracle and kernels use the FlashAttention convention; a rocKE kernel served through hipDNN gets
    `L - 1` from the provider.
- **Contract:** cuDNN's meaning.

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
  - Source: cudnn-frontend v1.24.0 `include/cudnn_frontend/node/sdpa_support_surface.h:229-232`. The forward
    check applies only to backends before 9.10; the backward check always applies.
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
- **Extent and table width (cuDNN):** an unset `max_seq_len_kv` is inferred in cuDNN's order: a non-paged K's (then
  V's) sequence length, else the bias's last dim, else `rng_dump`'s last dim, else the smaller `table_size *
  block_size` of the two sides. Each table has exactly `ceil(max_seq_len_kv / block_size)` entries
  (`paged_cache_load.h`); a wider table is rejected.
- **hipDNN today:** the CPU reference gathers each paged container into a dense tensor, reading only keys below
  `seq_len_kv`, then runs the padded dense path; the GPU kernel reads pages in place. Both plans infer the extent and
  check the table width as cuDNN does (`sdpaPagedMaxSeqKv`). Pinned by `TestCpuFpReferenceSdpaPropertyPaged` (block
  sizes 1/3/4, NHD strides, shuffled blocks, NaN in the null block and unused slots), the paged applicability cases,
  and ragged Q over paged K/V through the dense rewrite (`*RaggedQOverPagedKv*`).
- **Frontend:** `SdpaFwdNode` applies cuDNN's rules: INT32 `[B, 1, M, 1]` unpacked tables, `M` = ceil(max /
  block_size) when `max_seq_len_kv` is set, both seq_len tensors, and no `max_seq_len_kv` without paging. K or V may
  be paged alone, each with its own block size. Pinned by `TestSdpaFwdNode.*Paged*`.
- **External oracle:** `generate_sdpa_fwd_paged_golden.py` (gather-then-dense PyTorch, checked against a naive
  per-row loop in its tests); its bundles cover decode, speculative decode, mixed ragged prefill/decode, windows and
  sinks.

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
  - The GPU backward reference (`GpuRefSdpaBwd.cpp`) follows the same rule; pinned by the dead-row cases of
    `TestGpuSdpaBwdFp32`.
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

- **Status:** Aligned.
- **cuDNN Python reference:** fp32 P.
- **hipDNN today:** the CPU reference and the GPU graph plans (dense and ragged) compute P in fp32, so the GPU
  reference is an independent oracle for bf16 as well. Rounding P to bf16 before P@V (what the AITER kernel does) is
  available only as an explicit opt-in emulation mode (`GpuSdpaFwdOptions::probabilityMode` for `GpuFpReferenceSdpa::fprop`,
  `providerSdpaProbabilityMode`).
- **Contract:** fp32 P in every reference by default.
- **Watch:** provider engines checked against the GPU reference in bf16 lose the matching P rounding; the difference
  is expected to stay inside bf16 tolerances.

### SEM-013: Default scale

- **Status:** Diverges in rocKE.
- **cuDNN:** no scaling unless `attn_scale` is set.
- **PyTorch and vLLM:** default to 1/sqrt(D), but the adapter always passes the scale explicitly.
  vLLM uses per-model scales (Gemma `query_pre_attn_scalar`, DeepSeek `mscale`).
- **hipDNN today:** C++ references and golden data use 1.0. rocKE `sdpa_reference.py` defaults to
  1/sqrt(D).
- **Contract:** cuDNN.

### SEM-014: Bias gradient (dBias)

- **Status:** Aligned (CPU and GPU backward references).
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

### SEM-015: Dropout

- **Status:** Aligned for cuDNN's explicit-mask ("custom") dropout (CPU and GPU references); Philox dropout declined.
- **cuDNN:** forward `O = (P ∘ M · s) V` with mask M `[B|1, H|1, Sq, Skv]` and `dropout_scale` s; Stats are the
  pre-dropout softmax's. Backward (composite lowering, `scaled_dot_product_flash_attention.h`):
  `D = rowsum(dO ∘ O) · dropout_scale_inv` (1 when absent), `dS = s · P ∘ (M ∘ dP − D)`, `dV = (P ∘ M · s)ᵀ dO`,
  dBias reduces that dS. With `dropout_scale_inv = 1/s` this is the exact gradient; without it, only for s = 1.
- **PyTorch / AOTriton:** Philox dropout with seed/offset; the keep mask is not exposed except through test
  helpers (`_fill_mem_eff_dropout_mask_`).
- **hipDNN today:** the CPU forward and backward references implement the explicit-mask form (FLOAT mask,
  one-element FLOAT scale, optional scale_inv). `dropout_probability` / seed / offset and RNG_DUMP are declined: no
  reference can reproduce a kernel's random stream. Pinned by `TestCpuFpReferenceSdpaGradCheck.ExplicitMaskDropout`
  (mutation: dropping the scale_inv factor fails dQ, dK and dBias).
- **Philox dropout:** validated through RNG_DUMP. A forward graph with `dropout_probability`, seed,
  offset and an `rng_dump` output FLOAT `[B, H_q, Sq, Skv]` is accepted by the CPU and GPU forward references, which
  read the engine's dumped keep mask as the explicit mask with scale `1/(1−p)`. Engines that claim Philox dropout must
  support RNG_DUMP. Without RNG_DUMP the references decline (no exact oracle). The integration harness checks the
  dumped mask (values in {0, 1}, keep rate within a 6-sigma binomial bound of `1−p`), copies it into the reference
  run and does not compare `rng_dump` itself (`harness/bundle/RngDumpHandoff.hpp`).

## Score modifiers (extensions and conventions)

### SEM-023: Max and Sum_exp row statistics

- **Status:** Matches cuDNN.
- **cuDNN:** `Max` = row max of the masked scores, raised to the sink logit when there is a sink;
  `Sum_exp` = sum of exp(s - Max) over the live keys plus exp(sink - Max). Stats (LSE) = Max + log(Sum_exp).
  - Source: cudnn-frontend `test/python/sdpa/fp16_ref.py` (score_max_ref, score_sum_exp_ref).
- **hipDNN:** the same, FLOAT [B, H, Sq, 1] (the GPU reference takes rank 3 like the LSE). A row with no live key
  and no sink has Max = -inf and Sum_exp = 0, so LSE = -inf (SEM-010); with a sink, Max = sink and Sum_exp = 1.
  Rows past seq_len_q have Max = -inf and Sum_exp = 0.
- **Other use cases:** cuDNN added these for Muon-clip style score monitoring; FlashAttention and vLLM expose only
  the LSE.

### SEM-024: fp8 softmax and output scaling (cuDNN sdpa_fp8)

- **Status:** Matches cuDNN up to 128 keys per row; above that it differs by tile (below).
- **cuDNN:** with fp8 Q/K/V: S = (Q K^T) * descale_q * descale_k * scale (+ bias, masks); P is the unnormalized
  exp(S - m), stored in the input fp8 type as fp8(P * scale_s), and P @ V is multiplied by descale_s * descale_v;
  the result is divided by the sum of the *unquantized* exp (plus the sink term). O = fp8(O_f * scale_o);
  amax_o = max |O_f|. cuDNN's reference computes this online per 128-key tile, so m is the running row max of
  the tiles seen so far, not the final one.
  - Source: cudnn-frontend `test/python/sdpa/fp8_ref.py` (compute_ref), scales from `helpers.py`.
- **hipDNN:** the references quantize exp(S - Max) with the final row max (Max includes the sink). With at most
  128 keys this is exactly cuDNN's computation: the fp8 golden bundles (`generate_sdpa_fwd_golden.py
  --fp8-scaling`) reproduce cuDNN's `compute_ref` bit for bit. With more keys, earlier tiles see a smaller m and
  their P lands on different fp8 codes; the difference is a quantization artifact, not a different softmax.
  amax_s = max exp(S - Max) over live keys before scale_s (1 for any row whose max is a key; below 1 when the sink
  dominates). Encoding is round-to-nearest-even and saturating (hipdnn_data_sdk); PyTorch's cast does not
  saturate, which only matters past the fp8 range. scale_s / descale_s need fp8 inputs; scale_o and the amax
  outputs are accepted with any fp8 graph. Dropout with fp8 scaling is declined.
- **Contract:** cuDNN for <= 128 keys. A tile-faithful mode would need the tile size as an input.

### SEM-025: Block-sparse mask

- **Status:** Matches cuDNN (forward only; cuDNN's backward node takes no block mask).
- **cuDNN:** `block_mask` is UINT8 `[B, H_q, ceil(Sq / 128), ceil(ceil(Skv / 128) / 8)]`. Bit i of byte j
  enables the 128 x 128 score tile in tile row `sq / 128` and tile column `8 j + i`; a clear bit adds -inf to the
  tile's scores.
  - Source: cudnn-frontend `test/python/sdpa/fp16_ref.py` (block_mask expansion) and `fp16.py` (shape).
- **hipDNN:** the references mask the cleared tiles' keys to -inf with the band, in absolute positions; padding
  and paging do not move tiles (Skv is the paged key extent). A row whose every key is cleared behaves as
  SEM-010. The plans check the dtype and shape; the golden generator's `--block-mask` expansion is identical to
  cuDNN's code.
- **Contract:** cuDNN.

### SEM-020: Softcap

- **Status:** Extension, implemented in the references only; not expressible in the graph.
- **cuDNN:** no attribute; only possible through score_mod.
- **vLLM (Gemma2) and Flex:** `c * tanh(S / c)` after scaling, before the mask and bias.
- **hipDNN today:** the CPU forward/backward references and the GPU forward/backward references take an optional
  `softcap` argument (finite, > 0) with that order: scale, softcap, bias, masks. Backward: dBias takes the
  post-softcap gradient, dQ and dK also take `1 - tanh^2(s / c)`. Pinned by
  `TestCpuFpReferenceSdpaGradCheck.Softcap`, `SoftcapClosedFormAndLargeCapLimit` and
  `TestGpuSdpaSoftcapFp32` (mutations: missing derivative, missing forward cap).
- **Plan:** a first-class attribute will be added; the schema, frontend, backend and engine plumbing is not yet
  done, so nothing in a graph reaches it. Engines that decline from hand-written attribute lists must add it
  to those lists when it lands.

### SEM-021: ALiBi

- **Status:** Matches cuDNN (FA/TE differ by convention).
- **cuDNN:** adds `(j - i) * slope`, and requires causal.
- **FlashAttention and TransformerEngine:** `-slope * |i + Skv - Sq - j|`. Under BOTTOM_RIGHT with
  Skv > Sq the output O is the same but the LSE differs.
- **hipDNN:** the references implement cuDNN's form: `(skv - sq) * slope(h)` added after the bias, with cuDNN's
  per-query-head slopes (`get_alibi_slope`: 2^(-8 (h+1)/n) for the closest power of two n <= H, then
  2^(-4 (2j+1)/n)). The plans accept `alibi_mask` only with top-left causal masking, as cuDNN does; bottom-right or
  non-causal ALiBi is declined. The backward has no extra term.
- **Contract:** cuDNN. Adapters for FA/TE-style ALiBi must convert to cuDNN's form or use an explicit bias.

### SEM-022: Attention sinks

- **Status:** Aligned (CPU and dense GPU forward references).
- **Agreed semantics:** one unscaled logit per head, counted in the softmax denominator only, with no V row,
  and included in the LSE. cuDNN shape `[1, H_q, 1, 1]`, fp32 (cudnn-frontend `test/python/sdpa/fp16_ref.py`):
  `M = max(max_j s_j, sink)`, denominator `Σ exp(s_j − M) + exp(sink − M)`.
- **Consequence:** a row with no live key but a finite sink has O = 0 and **LSE = sink**, not -inf. Padded query
  rows stay O = 0, LSE = -inf.
- **hipDNN today:** both references implement it for a FLOAT `[1, H, 1, 1]` sink, and the frontend rejects other
  shapes and dtypes, as cuDNN does (vLLM's model-dtype sinks are converted by the adapter). Pinned by
  `SinkWithUniformScores`, `SinkOnFullyMaskedRow`, `SinkEqualsExtraZeroValueKey`, CPU/GPU parity and the paged sink
  golden bundles.
- **vLLM:** sinks are in the model dtype and kernels upcast them.
- **Open:** the dSink definition. AITER's test uses an LSE without the sink; do not follow it.

## Process

- Add an entry when a reference change, provider or adapter exposes a disagreement between use cases.
- An entry stays open until the references and their tests encode the contract.
