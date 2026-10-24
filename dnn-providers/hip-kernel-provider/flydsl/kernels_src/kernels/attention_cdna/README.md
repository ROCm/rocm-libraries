<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# FlyDSL attention for CDNA (gfx90a, gfx942, gfx950)

Vendored kernel sources for the `hipkernel:flydsl_sdpa_cdna` engine. The RDNA
attention kernels are separate, under `../attention_rdna/`, and share nothing
with these but the `kernels/common` helpers.

| File | Upstream | Serves |
|---|---|---|
| `flash_attn_gfx950.py` | FlyDSL `kernels/attention/flash_attn_gfx950.py` | gfx950: dual-wave, software-pipelined MFMA forward, head_dim 64 and 128 |
| `flash_attn_generic.py` | FlyDSL `kernels/attention/flash_attn_generic.py` | gfx90a and gfx942 (and gfx950 off the dual-wave path): generic MFMA forward |
| `flash_attn_utils.py` | FlyDSL `kernels/attention/flash_attn_utils.py` | helpers shared by both |

Upstream: https://github.com/ROCm/FlyDSL at
`89ad52fbbb9e252396829a8a44b12e18b7acd970` (v0.3.4.1+19), the same kernel tree
the generators pin (`generators/_flydsl_env.py`). Kept whole, so a re-sync is a
diff against upstream plus the modifications below.

## Modifications

0. **Import path.** `kernels.attention.flash_attn_utils` becomes
   `kernels.attention_cdna.flash_attn_utils` (one line in each kernel file).
1. **Runtime row guard in the cooperative K/V loads** (`flash_attn_utils.py`).
   `coop_load_k`, `coop_store_k_lds`, `coop_load_v` and `coop_store_v_lds` tested
   a runtime row index with a plain Python `if`. These are helper methods, which
   FlyDSL's AST rewriter does not lower (FlyDSL issue #452), so tracing failed
   with "cannot evaluate dynamic 'Boolean' as Python bool". The guard exists only
   when one cooperative-load pass covers more rows than `BLOCK_N` — the generic
   kernel's 256-row tile at head_dim 64, which its launcher picks on gfx942 once
   `batch * seq_len >= 2048`. The guard now goes through
   `ReplaceIfWithDispatch.scf_if_dispatch` (helper `_if_then`), the lowering the
   rewriter applies in kernel bodies. Not yet upstream.
2. **Runtime softmax scale** (`flash_attn_gfx950.py`, `flash_attn_generic.py`,
   `flash_attn_utils.py`). Both kernels take a trailing `sm_scale: fx.Float32`
   argument in place of the scale they derived: the dual-wave kernel from
   `rsqrt(head_dim_runtime)`, the generic one from a constant baked at build time.
   The launchers default it to `1/sqrt(head_dim)`, so a caller that passes nothing
   gets upstream's behaviour. Upstream callers of the contexts that pass no scale
   are unchanged.
3. **Runtime head counts** (both kernels) **and token strides** (the generic
   kernel). The dual-wave kernel takes trailing `num_heads_q, num_heads_kv`; the
   generic one those and `stride_q_n, stride_kv_n` (its token pitches had been
   `num_heads * head_dim`, baked). The head arithmetic is done in 32 bits, since an
   `Index` divide is a 64-bit one. Only the dual-wave kernel's default grid mapping
   reads the runtime counts; its swizzled and batch-interleaved mappings refuse
   them. The launchers pass the build-time values, so upstream callers are
   unchanged.
4. **Runtime LSE switch** (both kernels). Built with `return_lse`, each kernel
   takes a trailing `lse_on`; at 0 the LSE buffer descriptor gets no records, so
   every LSE store falls outside it and is dropped -- no branch, and no second
   object. The launchers pass 1.
5. **Runtime causal diagonal** (both kernels). A trailing `diag_offset`: query
   row i sees key j <= i + diag_offset, where the kernels had computed
   `seq_len_kv - seq_len` (bottom-right). Top-left causal and right-side bands are
   then the same objects. The launchers pass `seq_len_kv - seq_len`.
6. **Runtime operand strides** (both kernels). Trailing `stride_q_b, stride_q_h,
   stride_o_b, stride_o_n, stride_o_h, stride_kv_b, stride_kv_h` beside the token
   strides already there. Q, O, K and V are read through per-(batch, head) buffer
   views: each starts at its head's first row and ends at its last element,
   `(seq_len - 1) * stride_n + head_dim`, so rows past the sequence read as zero
   and stores to them are dropped whatever the stride order, as the per-batch
   BSHD views guaranteed. The head's offset moves from each lane's index into the
   view's base, and O gets its own token stride. The launchers pass contiguous
   BSHD strides.
7. **Extra LLVM options** (`flash_attn_generic.py`). An `extra_llvm_options`
   builder argument merged into the kernel's own compile options, and forwarded
   through the builder's internal re-builds; none by default.
8. **Per-(batch, head) bias view** (`flash_attn_gfx950.py`). Trailing
   `stride_bias_b, stride_bias_h`; the bias is read through a view rebased at
   `batch * stride_bias_b + head * stride_bias_h` and ending at
   `(seq_len - 1) * bias_stride0 + seq_len_kv`, so any of batch, head and query can
   be broadcast (stride 0) and rows and columns past the scores read as zero.
   `_prep_bias` passes the bias as given rather than flattened, keeping the
   argument's layout rank-4 like every other tensor's. The launchers pass 0, 0: a
   dense `[seq_len, seq_len_kv]` bias, upstream's contract.

9. **LDS row padding as builder arguments** (`flash_attn_generic.py`,
   `flash_attn_utils.py`). `k_pad` and `vt_pad` override the K tile's row padding
   (upstream: 4 elements on gfx942 without DMA, else 0, or `FLYDSL_FLASH_ATTN_FUNC_K_PAD`)
   and the transposed V tile's (upstream: 2), forwarded through the builder's re-builds.
   At `head_dim` 256 the padded tiles need 67,072 bytes of LDS, past gfx942's 64 KiB;
   unpadded they need exactly 64 KiB. Unset, upstream's padding.

10. **Rescale every O chunk on the non-fused PV path** (`flash_attn_utils.py`,
    `GenericGemmHelper.gemm2_pv`). Without hardware transpose (gfx942) the O chunks
    after the first are rescaled lazily, one per PV step of the first chunk -- which
    reaches only `PV_K_STEPS` of them. At `head_dim` 256 (eight chunks, four steps) the
    last three were never rescaled when the running max moved, and f16 (which takes
    this path; bf16 takes the fused one) gave wrong results. The rest are now rescaled
    before their first accumulate. An upstream bug; unreachable upstream, whose gfx942
    launcher never builds `head_dim` 256.

11. **Sliding window** (`flash_attn_generic.py`, `flash_attn_utils.py`). The generic
    kernel takes a trailing `window_lo`: query row i sees no key j < i + window_lo. In
    causal builds the KV loop starts at the KV-tile boundary at or below
    `q_start + window_lo`, and a tile reaching below the block's last row's edge is
    masked on that side too; fully masked rows were already floored to a finite max, so
    they contribute nothing. Non-causal builds ignore it. The launchers pass
    `_NO_WINDOW` (-2^30), no window.

12. **Additive bias on the generic kernel** (`flash_attn_generic.py`,
    `flash_attn_utils.py`). A `has_bias` builder argument (part of the JIT cache key)
    and trailing `Bias, bias_stride0, stride_bias_b, stride_bias_h` on every generic
    object. A bias build reads one (batch, query head) slice of the bias through a
    buffer view rebased at `batch * stride_bias_b + head * stride_bias_h` and ending at
    `(seq_len - 1) * bias_stride0 + seq_len_kv`, so any of batch, head and query may be
    broadcast (stride 0) and rows and columns past the scores read as zero, as the
    dual-wave kernel's does (modification 8). Each lane loads its 32 scores' bias as
    eight 4-element runs and adds it to the scaled scores (modification 28). Other
    builds never read the slot.

13. **Separate V strides, a narrower V, and a runtime Q/K head dim on the generic
    kernel** (`flash_attn_generic.py`, `flash_attn_utils.py`). Every generic object
    takes trailing `stride_v_n, stride_v_b, stride_v_h, head_dim_qk`. V is addressed
    through its own strides rather than K's, so K and V need not share a layout.
    - **`head_dim_v`** (builder argument, cache key): V and O are that wide, at most
      `head_dim`, for multi-head latent attention. The PV product, the O
      accumulators, the O store and O's buffer extent cover only those columns. V is
      still staged `head_dim` wide, and the columns past it are read from column 0's
      address rather than past the row, and are never consumed.
    - **`qk_mask`** (builder argument, cache key): `head_dim` is then a bucket, and
      `head_dim_qk` the actual Q/K width. Q columns past it are loaded through an
      out-of-range offset, so they read zero. K columns past it are selected to zero,
      read from column 0's address. Both must be zero, because 0 * NaN from a
      neighbouring row is still NaN. Q's buffer extent ends at it.
    - **Scope:** both are on the dense, non-DMA load paths, and a build with either
      refuses DMA, paging and the permuted V^T store. Builds without them are
      unchanged apart from V's own index.

14. **KV split on the generic kernel** (`flash_attn_generic.py`, `flash_attn_utils.py`).
    Every generic object takes trailing `WsO, WsLse, num_splits`. A `split_kv` build
    (builder argument, cache key) runs `num_splits` grid rows: row `y` takes its
    contiguous share of the block's KV tiles, in whole tiles, so the masks see the
    tile boundaries an unsplit launch would.
    - **One split:** the build writes O and LSE as before.
    - **More than one split:** each split writes its normalized f32 partial O and its
      LSE to the workspace (`[B, Hq, Sq, splits, Dv]` and `[B, Hq, Sq, splits]`),
      through descriptors scoped to its (batch, head) slice, so the padding rows'
      masked stores are dropped. A split past the last tile writes O = 0 and
      LSE = -inf.
    - **Merge:** a second kernel in the same object, `flash_attn_generic_merge_kernel`,
      one 64-thread workgroup per (batch, head, query) row, combines the partials by
      their LSEs into O and, under `lse_on`, the LSE.
    - **Unchanged:** builds without `split_kv` ignore the new arguments and compile to
      the same instructions.

15. **GQA row packing on the generic kernel** (`flash_attn_utils.py`,
    `flash_attn_generic.py`). A `pack_gqa` build (builder argument, cache key; no bias)
    launches one workgroup per (batch, row tile, KV head). Its rows are the GQA group's
    query heads by position, head-major: row `r` is query head
    `kv_head * group + r // Sq` at position `r % Sq`. So one K/V tile serves the whole
    group, rather than one query head's single row of a decode step.
    - **Addressing:** Q and O through each row's head and position strides, with the
      descriptors spanning the group's heads. A row past the group returns an offset
      beyond them, so it reads zero and its store is dropped.
    - **Masks:** compare each row's position, not its packed index. The KV range
      covers the positions `0 .. Sq - 1`.
    - **LSE and split partials:** indexed by the packed row, since the head-major
      order makes a group's rows contiguous.

16. **O column split, V staged by its slice, and Q streamed past 256**
    (`flash_attn_utils.py`, `flash_attn_generic.py`).
    - **`dv_split` (builder argument, cache key):** O's columns split across
      `dv_split` workgroups along the grid's z, each holding `head_dim_v / dv_split`
      of them. Its V loads, V tile and O stores are offset to its slice. The first
      slice writes the LSE, and the split partials go to the slice's columns.
    - **V staging:** V rows are staged `V_WIDTH` wide with a cooperative load mapping
      of their own. That is the whole head as before, or, for a narrower or split V,
      only the slice, so an MLA object's V tile no longer spans the Q/K bucket.
    - **Q streaming:** past `head_dim` 256, Q's packs are read from global at each
      K-step that uses them, rather than held in registers through the KV loop.
    - **K swizzle:** the K tile's XOR swizzle mask is capped at 31. The reader keys
      both 32-row halves of the tile by `lane % 32`, so past `head_dim` 512 the
      uncapped mask (63 at 1024) read the upper half through a different swizzle than
      it was written with.
    - **Unchanged:** builds without any of these compile to the same instructions.

17. **V follows the runtime Q/K head dim** (`flash_attn_utils.py`). A `qk_mask` build
    whose `head_dim_v` equals its bucket serves any head dim up to the bucket -- any
    multiple of 8, its K and V loads bounded in 8-column halves -- for V too:
    - V's columns past the runtime width are read from column 0's address;
    - O's columns past it map past every slice, so their stores are dropped;
    - O's buffer extent ends at it.

    Such builds take neither the KV split nor the column split (the merge kernel knows
    only baked widths), and, like every masked build, the N32 path rather than the
    DMA path.

18. **K staged in column chunks** (`flash_attn_utils.py`, `flash_attn_generic.py`). A
    `k_chunk` build (builder argument, cache key) sizes K's LDS tile, load mapping,
    swizzle and QK steps to a `k_chunk`-wide column chunk of the head, and runs one QK
    pass per chunk for each KV tile. Each pass:
    - waits for every wave to have read the last chunk;
    - overwrites the tile with its own columns;
    - continues the scores from the previous pass.

    Q is read with its absolute column, so a head wider than the LDS holds runs in a
    tile that fits: gfx942's absorbed-MLA objects stage a 768 bucket in three
    256-column chunks. Chunked builds take the plain load path, with no KV global
    prefetch, DMA or three-buffer prefetch.

19. **Attention sinks** (`flash_attn_utils.py`, `flash_attn_generic.py`). Every generic
    object takes trailing `Sink, sink_stride, sink_on`. A `has_sink` build (builder
    argument, cache key) reads one logit per query head, in the query's dtype, and
    folds it into the softmax as GPT-OSS defines it: a key with no value.
    - **A row that reaches keys:** adds the sink's term to its denominator, and its
      LSE includes it, against the max over the scores and the sink (modification 29).
    - **A row that reaches none:** O = 0, LSE = the sink.
    - **Split launches:** the partials carry no sink; the merge kernel adds it once.
    - **With a bias:** a `has_sink` build may be a `has_bias` one too; a bias build's
      max and LSE are already in natural units (modification 28), as the fold takes
      them.
    - **Unchanged:** builds without `has_sink` never read the arguments and compile
      to the same instructions.

20. **LSE strides** (`flash_attn_utils.py`, `flash_attn_generic.py`). The generic
    kernel and its merge kernel take trailing `stride_lse_b, stride_lse_h,
    stride_lse_n` and write the LSE through them, rather than as one packed
    `[B, H, Sq]`. A packed row's head and position come from its packed index. The
    store's buffer view spans one batch's rows through those strides, so an LSE store
    past them is still dropped. Packed strides give the addresses as before.

21. **Softmax scale on the f32 scores** (`flash_attn_gfx950.py`, `flash_attn_utils.py`).
    Upstream's dual-wave kernel multiplied Q by `sm_scale * log2e` and rounded the
    product back to bf16 or f16 before the QK MFMA (`DualwaveQLoader.scale_all`).
    Each Q element then carries a fresh rounding error of up to half an ulp, and the
    resulting score error grows with the score itself. Measured against an f32
    reference, the bf16 LSE was off by up to 1.2e-3 and O by a few bf16 ulps.
    Q now stays as loaded and the MFMA's scores stay raw. `DualwaveSoftmaxHelper`
    keeps the running max over them and applies the scale where the max is subtracted,
    in one FMA (`_scale_sub_score_pair`, as upstream's fp8 path and the generic kernel
    do), and where two maxima are compared (rescale factors, the lazy-rescale
    threshold). The ALiBi slope joins the raw scores divided by the scale (a bias build
    scales its scores first and adds the bias to them, modification 28), and
    the max is scaled once after the loop for the sink, the stores and the split-K
    combine. A row with no key yet is floored at -1e30 rather than -3e38, so the
    scaled floor stays finite. The scale must be positive, which the engine already
    requires. Only the dual-wave objects change; the generic kernel's are
    byte-identical. Its cost is COVERAGE.md section 7.5 item 12. Not yet upstream.

22. **A finite floor for a row no key reaches** (`flash_attn_utils.py`). The generic
    kernel's causal builds floor a row's running max at -3e38 before the softmax
    subtracts `sm_scale * log2e * max`. Past a scale of about 0.78 that product
    overflows, the subtraction becomes `fma(-inf, scale, +inf)`, and a row no key
    reaches gets NaN rather than O = 0 and LSE = -inf (or the sink). A graph with no
    scale runs at 1.0, cuDNN's default, so it hit this. The floor is now -1e30, the
    dual-wave kernel's (modification 21). The generic causal objects change only in
    that constant. Not yet upstream.

23. **The LSE of a row no key reaches, selected** (`flash_attn_utils.py`). Both kernels
    wrote the LSE as `m * scale + log(l)` under full fast math, and for such a row that
    is `-inf + log(0)` by design. With no-infinities fast math LLVM treats it as
    poison: on the generic kernel's path for a q-tile no key reaches, it never computed
    `m * scale` at all, and the store read whatever the register last held. The LSE
    came out right while that was finite and NaN otherwise, so it failed only when
    another process had used the GPU (on gfx942, 22-36% of launches under load, never
    alone). The LSE is now `l == +0 ? -inf : m * scale + log(l)` (modification 29;
    first `l > 0 ? ... : -inf`, which a NaN sum also took), its log without the
    fast flags, in the generic kernel's store, the dual-wave kernel's and the dual-wave
    split-K combine's (not built here). Timing is unchanged. Not yet upstream.
24. **gfx90a (CDNA2)** (`flash_attn_utils.py`, `flash_attn_generic.py`). Upstream knew
    two CDNA targets: its gfx942 branches tested `startswith("gfx942")`, and everything
    else took gfx950's. On gfx90a that meant the 16-byte global-to-LDS DMA (gfx950 only)
    for causal head_dim 128, and FlyDSL's arch-dispatched `s_waitcnt` raised outright.
    - **Branches:** `_is_gfx942_class()` puts gfx90a with gfx942 -- the four-byte DMA,
      the register-staged K/V prefetch, the K row padding, the launcher's 256-row
      threshold -- since both have 64 KiB of LDS, DMA of at most four bytes and the
      32x32x8 MFMA (bf16 `_1k` included). The gfx950-only paths (transpose read, 32x32x16
      MFMA, permlane O store) were already gated on gfx950.
    - **`s_waitcnt vmcnt`:** emitted through `cdna3.s_waitcnt`, the GFX9 encoding all
      three targets share, instead of the dispatch that knows only gfx942 and gfx950.
    - **Unchanged:** gfx942 and gfx950 take every branch they took before and build the
      same code objects. Not yet upstream.
25. **V loaded in the layout its store expects** (`flash_attn_generic.py`). Without the
    register-staged K/V prefetch, the generic kernel loaded V with the plain loader and
    then, where bf16's `v_perm` V^T store applies (one KV batch per tile: head_dim 64 at
    128 rows, 128 at 256 rows), stored it with the permuting store, which reads two rows
    per thread from `coop_load_v_global_perm`. Tracing failed (`IndexError`) on gfx942
    and gfx90a alike. That path now loads with the permuting loader, as the prefetch path
    already did. Every build that traced before is unchanged. Not yet upstream.
26. **Head dims that are not powers of two** (`flash_attn_utils.py`, `flash_attn_generic.py`).
    The generic kernel built only 64, 128 and 256 wide (and their buckets), so 160 ran
    on the 256 bucket with 37.5% of its matrix work on zero columns. Three things kept
    96, 160, 192 and 224 out, all now handled, each behind a compile-time test that is
    false at a power of two:
    - **Load mapping:** a K or V row was loaded by `width / 16` threads, which must
      divide the workgroup. It is loaded as the next power of two wide; a lane whose
      columns start past the row's end reads column 0 instead and stores nothing to the
      LDS (`K_LOAD_GUARD`, `V_LOAD_GUARD`).
    - **K swizzle:** the XOR left a non-power-of-two row (at 192, column 64 ^ 128 = 192).
      Its row mask is capped inside the largest power of two dividing the width.
    - **Split-KV merge:** it covered `HEAD_DIM_V // 64 * 64` columns, silently dropping
      the rest at 96, 160 and 224; it now covers every column.
    - **Unchanged:** every 64/128/256 build compiles to the same object. The DMA paths,
      the permuting V^T store, K chunks and V column slices stay off at these widths.
      Not yet upstream.
27. **A split that reached no key of a row contributes nothing** (`flash_attn_utils.py`,
    `flash_attn_generic.py`). For a fully masked q-block (bottom-right with more queries
    than keys, or a window) a split's accumulators are not valid; the direct store zeroes
    such blocks, but the split store wrote them times 0, which is NaN when they hold
    NaN, and the merge weighted that partial by 0 again. The row came out NaN instead of
    0. The split store now selects 0 for a row with no keys, and the merge selects away
    any partial whose LSE is -inf -- testing -inf by its bits, since under the kernels'
    no-infs fast-math `x == -inf` folds to false: the merge's own m_safe guard had been
    folded away, so even a correct workspace (LSE -inf, O 0 in every split) merged to
    exp(-inf - -inf) = NaN. Seen on gfx942 with two splits; gfx950 ran the case with one
    split, which never merges. Not yet upstream.
28. **Bias builds softmax over the scaled, biased scores** (`flash_attn_utils.py`,
    `flash_attn_generic.py`, `flash_attn_gfx950.py`). Both kernels added `bias / scale`
    to the raw scores and scaled inside the exponent, as `fma(s, c, -c * m)`, with the
    running max floored at -1e30 in causal builds only. A row the bias masks on every
    key with the dtype's lowest finite value (Hugging Face's padding mask) then went
    wrong: divided by a scale below 1 the bias overflowed to -inf, near -FLT_MAX the
    fused exponent overflowed to `-inf + inf`, and -1e30 lies above the bf16 and f16
    lowest values; a non-causal row whose first tile the bias masked kept a max of -inf
    and turned later tiles NaN. PyTorch's SDPA, hipDNN's CPU reference and the RDNA
    kernels (their modifications 14, 19, 22, 23) softmax over `q.k * scale + bias`, so
    such a row's scores are finite and equal: O is the mean of V over the keys the
    causal mask leaves, and the LSE is that score plus log(count). Bias builds, causal
    included, now do the same; every other object builds as before.
    - **The floor** is -3.4e38 (`_BIAS_NEG_FLOOR`): below every finite score a bf16 or
      f16 bias gives (bf16's lowest is -3.39e38), so it lifts only a row of -inf. It is
      not -FLT_MAX because under the kernels' no-infs fast math `max(x, -FLT_MAX)` is a
      no-op LLVM deletes, and so is a -inf bit test (modification 27's) on the max,
      which a fast-math op produced; both were tried and vanished from the ISA.
    - **Generic kernel:** `add_bias` computes `fma(s, scale, bias)`; `init_constants`
      then leaves the max, the exponent and the LSE a scale of 1 (`c_score_scale` holds
      the real one) and sets the floor, which `online_softmax_stats` now applies in
      non-causal bias builds too. `_exp2_arg` exponentiates `(s - m) * log2(e)` (in
      `online_softmax` and
      `gemm2_gpfetch_fused`, which now takes the max). The LSE is `m + ln(l)` in
      natural units, as the non-bias builds write it for the same scores; the sink, the
      split store and the merge take it as before.
    - **Generic kernel's masked scores:** in a bias build `apply_kv_mask` writes -FLT_MAX
      (`_BIAS_MASKED_SCORE`), not -inf, over the keys its key-padding, causal and window
      masks remove. Under the no-infs fast math LLVM folds `select(c, -inf, s) - m` to
      `s - m`, since the -inf arm is poison: with -inf, the first build of this
      modification lost the non-causal key-padding mask from the exponent (the max kept
      it), so a key length that ends in a part tile let the padded keys into the softmax
      (finite, wrong O). -FLT_MAX lies below the floor, so its exponent is 0 against any
      max, and a row of nothing but masked keys still meets the floor with l = 0.
    - **Dual-wave kernel:** `init_types_and_constants(scaled_scores=True)`; the scores
      are scaled in `_score_bias_half` before the bias joins them, `sub_m` takes
      `(s - m) * log2(e)`, `floor_masked_max` applies the floor (in the prologue,
      non-causal bias builds too), and `scaled_max` keeps the max in natural units,
      which the LSE store takes with a factor of 1 (times log2(e) a row at the lowest
      value overflows). The sink fold and the split-K combine take log2 units, so a
      bias build with either now raises; the provider builds neither.
    - **Both:** a row the bias masks on every key with -inf normalizes to O = 0 (1/l
      selected as 0 when l is 0) and LSE -inf. Not yet upstream.
29. **NaN rows stay NaN, and the sink joins the max** (`flash_attn_utils.py`,
    `flash_attn_generic.py`). Parity with PyTorch's SDPA and the RDNA kernels:
    - **"No keys" is l == +0, tested on its bits** (`_no_keys`). Every O and LSE select
      that tells a keyless row from one that reached keys tested `l > 0`, which is false
      for NaN too. A row whose sum is NaN (a NaN or +inf score: NaN in Q, or in K at a
      visible key) therefore got LSE -inf where PyTorch gives NaN, and a split-KV
      decode split with such a row stored O = 0 and LSE = -inf, which the merge skips
      (modification 27): the row came out finite, from the other splits only, a silent
      wrong answer. The split store, the merge, the generic kernel's LSE store and
      sink fold, and the dual-wave kernel's LSE stores (its split-K combine's too, not
      built here) now select the keyless result only when l is +0, so a NaN row keeps
      NaN in its O and LSE and the merge carries it into every weight and the merged O
      and LSE. The test is an integer compare on the bits because under the kernels'
      no-NaNs fast math LLVM rewrote the select on `l == 0` as one on `l != 0`
      (`v_cmp_lg_f32`, false for NaN); the bit test lowers to `v_cmp_class_f32 l, +0`,
      true only for +0 (checked in the ISA of every changed object). The sum starts at
      +0 and adds only non-negative terms, so +0 is its only zero. The merge's max over
      the splits' LSEs is a non-propagating `v_max` under the same flags, so a NaN split
      reaches the merged row through its weight `exp(NaN - max)`, not through the max.
      The O normalization of a plain build keeps its `l > 0` select: a NaN sum comes
      from a NaN probability, which is in the accumulator too, so O is NaN either way.
    - **The sink joins the max** (generic kernel's sink fold and the merge). The sink was
      added as `exp(sink - m)` against the scores' max alone, so the denominator and the
      LSE overflowed to +inf once the sink led the max by more than ~88.7. Both now take
      `m_top = max(m, sink)`, `keep = exp(m - m_top)`, denominator
      `l * keep + exp(sink - m_top)`, O scaled by `keep / denominator` and
      `LSE = m_top + ln(denominator)`, as the RDNA decode merge and the dual-wave
      `fold_sink` do. A sink far above the scores gives O ~ 0 and LSE = the sink. A row
      no key reaches still gets O = 0 and LSE = the sink.
    - **Objects:** every object that writes an LSE changes by one compare
      (`v_cmp_lt_f32` to `v_cmp_class_f32`, or one added where the plain O select still
      needs `l > 0`); the sink objects also gain the max and two exponentials per fold;
      the decode objects' split store selects on the class test. No VGPR count or spill
      changed. Not yet upstream.
30. **A diagonal that passes the keys** (`flash_attn_utils.py`). The generic kernel's
    causal mask hid key `j` from row `i` when `j > i + offset`, and masked the last KV
    tile's padding with the same compare, so a row whose diagonal passed the last key
    (top-left with more queries than keys, a band wider than `Skv - Sq`) let the padded
    keys into its softmax. `apply_kv_mask` now clamps the block's first row's diagonal
    (which decides whether a tile is masked at all) to `Skv - 1`, and, in a tile that
    holds padding, each row's limit to `Skv - 1` too: a scalar select gives the tile's
    key cap (`Skv - 1` there, INT32_MAX elsewhere) and a vector min applies it, so the
    per-score compare is unchanged and nothing new stays live across the KV loop. Such
    a row sees every key and none of the padding, as PyTorch's and the reference's rule
    gives. Clamping each row's limit once per workgroup instead, or OR-ing a compare
    against the last key into every score's mask, cost 2-4 VGPRs and pushed gfx942's
    `head_dim` 128 causal 256-row tile from 2 spilled VGPRs to 4; this form leaves it
    at 2.
    - **Objects:** every generic causal prefill object gains the clamps (a few scalar
      instructions and one vector min per masked KV tile). Most are within 4 VGPRs of
      before; gfx950's `head_dim` 128 causal 64-row tile spills 4 VGPRs where it spilled 2
      (its allowance is 6), and some gfx90a tiles move by 15-50 VGPRs either way. Measured
      against the objects before it (COVERAGE.md section 7.5): none slower by more than
      1.5% on gfx950 and gfx942 (gfx950's `head_dim` 128 causal 64-row tile 7-8% faster
      at one query row); on gfx90a a few tiles by 2-5% either way, the `head_dim` 64
      causal bias tile by -11% to +5% across shapes. The non-causal objects are unchanged.
    - **Split decode builds:** keep the plain compare and compile to the same
      instructions: the provider runs them only on diagonals inside the keys, and serves
      the rest on the prefill objects at any query length.
    - **Dual-wave kernel:** unchanged; its software pipeline masks the padding with the
      diagonal too, and the provider runs it only on diagonals inside the keys.
    - Not yet upstream.
31. **A bias read one key at a time, at a runtime key stride and element type**
    (`flash_attn_utils.py`, `flash_attn_generic.py`). Every generic object takes
    trailing `stride_bias_k, bias_f32`; a `bias_kstrided` build (builder argument,
    cache key; a bias build) reads them, the others never do and compile to the same
    instructions.
    - **Loads:** each of a lane's 32 scores loads its own key's bias, at
      `row * bias_stride0 + key * stride_bias_k` in the slice: the lane's first key's
      offset in a register, each other key's distance from it a scalar the load takes as
      its scalar offset. So a key stride of 0 (a bias broadcast along the keys), any key
      length and any row pitch are read exactly. The slice's view ends at
      `(seq_len - 1) * bias_stride0 + (seq_len_kv - 1) * stride_bias_k + 1` elements.
    - **Element type:** `bias_f32` picks f32 (4-byte loads) or the query's dtype (2-byte
      loads, widened) by a uniform branch around the tile's loads, so one object serves
      both and runs only the loads its launch needs.
    - **Halved scores:** the build keeps its scores halved, `fma(q.k, scale / 2,
      bias / 2)`, with the floor at `-FLT_MAX / 2` and the exponent's and the LSE's
      factor 2 (`c_sm_scale` 2, `c_sm_scale_log2e` 2 log2(e)). An f32 bias reaches
      -FLT_MAX, which is modification 28's masked-score value: unhalved, a row an f32
      padding mask sets to its lowest finite value fell under the floor and came out
      O = 0, LSE -inf, where PyTorch averages V. Halving is exact in binary, so every
      exponent, sum and LSE is bit-for-bit the unhalved one's; a row of -FLT_MAX now
      gets O = mean of V and LSE -FLT_MAX.
    - Not yet upstream.
32. **Per-batch sequence lengths (padding), and one argument change for what follows**
    (`flash_attn_utils.py`, `flash_attn_generic.py`; generic kernel and its merge only).
    - **Slots:** a dense build reads upstream's `CuSeqQ`/`CuSeqKv` slots (renamed
      `SeqLenQ`/`SeqLenKv`; only a VARLEN build reads them as cu_seqlens, and VARLEN
      refuses runtime strides) as a padded launch's per-batch lengths, INT32 `[B]`, and
      the page-table stride slot (only a PAGED build's; renamed `varlen_flags`) as
      flags: bit 0 per-batch lengths, bit 1 a bottom-right diagonal.
    - **Reads:** under a real branch on bit 0, once per workgroup after the batch
      index (`_load_seq_lens`), so a dense launch loads nothing and waits on nothing;
      each length clamped to `[0, the launch's length]`, as hipDNN's GPU reference
      clamps them. They replace the launch's lengths for everything past the grid: the Q,
      K and V views (so keys past a batch's length are never read and no neighbouring
      batch leaks in), the KV loop's bounds, the pad and causal compares (modification
      30's key cap is the general pad compare), the bias's view. The batch and head
      bases keep the launch's strides. A view of 0 rows has 0 records (`_rows_bytes`).
    - **Diagonal:** under bit 1 the diagonal and the window's edge move by each batch's
      `(Skv_b - Skv) - (Sq_b - Sq)`, so a bottom-right mask sits at its own
      `Skv_b - Sq_b`; a top-left one is the same literal for every batch.
    - **Padded query rows** (rows at or past `Sq_b`) are stored as O = 0, LSE = -inf, on
      sink builds too (padding is not a row no key reached): O's view and the LSE
      writer's guard span the launch's rows. Never by a select on f32 values: fast math
      merged such a select with the no-keys test (`_no_keys`) into a class test without
      NaN and lost a NaN row's NaN (modification 29's rule; measured on gfx942). Instead
      a padded row's O words are zeroed after they are packed to the output dtype, and
      its -inf LSE is written by its bits through a second store whose offset is
      redirected for real rows (the real rows' store redirected for padded ones). A
      workgroup-uniform branch around the O stores (dense tiles storing as before) was
      tried and cost registers: one more spill reloaded inside the KV loop of gfx942's
      `head_dim` 128 256-row tile. A query tile wholly past its batch's length runs no
      KV iteration (not on the packed decode builds, whose tiles mix heads). The merge
      kernel, one row per workgroup, branches on its row.
    - **No-mask builds:** a padded batch may have no keys, so a row of a build with no
      mask and no bias can sum to 0; its `1 / l` is now selected by `_no_keys` (O = 0,
      not 0 * inf = NaN; a NaN row stays NaN).
    - **Appended for the changes that follow** (so the argument list moves once): the
      main kernel takes `RowOffQ`, `RowOffKv` (packed sequences' row offsets),
      `DropoutSeed`, `DropoutOffset` (tensors), `dropout_offset2_lo`,
      `dropout_offset2_hi`, `idropout_p` (i32) and `dropout_scale` (f32); the merge
      kernel `SeqLenQ`, `RowOffQ` and `varlen_flags`. Modification 33 reads the dropout
      arguments and modification 34 the main kernel's row offsets; the merge kernel's
      `RowOffQ` stays unread.
    - **Objects:** every generic prefill and decode object rebuilt: no VGPR or spill
      change on the probed objects; a few scalar instructions in the prologue and one
      select per O element in the epilogue. Measured against the objects before it in
      COVERAGE.md section 7.5. The dual-wave kernel is unchanged.
    - Not yet upstream.
33. **Dropout** (`flash_attn_utils.py`, `flash_attn_generic.py`; generic prefill builds).
    A `has_dropout` build (builder argument, cache key) drops attention probabilities
    with AOTriton's mask (>= 0.15), bit for bit, through modification 32's appended
    arguments; every other build takes them unread and compiles as before.
    - **Philox4x32-10** (`_philox4x32`): the aiw kernel's `attention_rdna4/philox.py`
      round (Triton's), restated here so these objects digest only this directory; the
      provider's tests pin the mask to a host oracle and to AOTriton's own masks.
    - **The stream:** key (seed low, seed high), the 8-byte seed read at `DropoutSeed`'s
      base; each run of 4 keys of a row draws one counter, `offset1 + offset2 + ((b * Hq +
      h) * Sq + i) * cdiv(Skv, 4) + j / 4` (`offset1` the 8 bytes at `DropoutOffset`'s base,
      `offset2` the two words, `Sq`/`Skv` the launch's lengths, so a padded batch's mask
      is the dense one's); key `j` takes word `j % 4`, kept iff that word as an int32
      exceeds `idropout_p`, the f32 threshold the host computes.
    - **Lanes:** a lane's 32 scores are 8 runs of 4 consecutive keys starting at a
      multiple of 4 (`_kv_mask_lane_off`), so it draws 8 counters per 64-key tile, one per
      run, from its row's first counter (`ph_row_base`, set once per workgroup).
    - **Where:** after the row's sum is formed (`online_softmax`, or inside the gfx942
      bf16 fused PV loop), so `l` and the LSE stay undropped; a dropped probability is 0
      before the P packs. `dropout_scale` (`1 / (1 - p)`) folds into O's `1 / l`.
    - **Objects:** 12 per arch (bf16/f16, causal or not, `head_dim` 64, 128, 256; no bias,
      no sink), each its class's first generic tier but where the Philox draws spill
      it: the kernel's default occupancy at 64 on gfx942/gfx90a, 128-row tiles at 128 on
      gfx90a, one wave per SIMD at 128 causal on gfx950. No spill on any. Objects without
      dropout are byte-identical to modification 32's.
    - Not yet upstream.
34. **Packed (THD) sequences** (`flash_attn_utils.py`, `flash_attn_generic.py`; generic
    prefill builds).
    - **Flag:** `varlen_flags` bit 2 (`VARLEN_PACKED`): Q/O and K/V are packed through
      modification 32's `RowOffQ`/`RowOffKv`, INT32 `[B + 1]` cumulative token counts.
      Sequence `b` starts at token `off[b]` and has `off[b + 1] - off[b]` tokens, clamped
      to `[0, the launch's length]`.
    - **Reads:** `_load_varlen` replaces `_load_seq_lens` on the prefill builds: one real
      branch on bit 2 loads the two tables' entries, then one on bit 0 loads the padded
      lengths (clamped to the sequence's tokens, so per-batch lengths inside a packed
      sequence need no new kernel code). A dense launch loads nothing.
    - **Bases and views:** each sequence's Q, O, K, V and LSE views start at its first
      token: `tok * stride_n` is added to the batch base, and the host passes a packed
      operand's batch stride as 0. Q, K and V span the sequence's tokens. O spans them
      too, and so does the LSE writer's guard (`rows_b`). The rows past a sequence belong
      to the next one, so no store reaches them: padded-row fills land only on rows in
      `[len, view)`, which is empty when packed. Diagonals move per sequence under bit 1,
      as under padding.
    - **Keyless views (a fix to modification 32 too):** a batch or sequence with no keys
      runs no KV iteration, but the builds that load the first K tile before the loop
      (gfx942's global prefetch) still read its clamped row, -1. For a view at the
      tensor's start (batch 0, head 0 under padding; the first sequence when packed),
      that row lies before K's base. A keyless view at token 0 now starts one token on,
      so the read stays in the tensor; the values are never used. This is on every build
      with per-batch lengths, decode included.
    - **Builds:** the split and packed-GQA (decode) builds read no row offsets. Their
      packing (and the merge kernel's) is later work; the matcher declines packed graphs
      on them and on the dual-wave kernel.
    - **Objects:** every generic prefill and decode object rebuilt. No VGPR or spill
      change. A few scalar instructions in the prologue. The KV loop is
      instruction-identical to modification 33's.
    - Not yet upstream.
35. **Exact f32 operands** (`flash_attn_utils.py`, `flash_attn_generic.py`; generic
    builds, `dtype_str="f32"`).
    - **MMA:** each 16-bit K-step (a 4-element pack per lane along K) becomes four
      `v_mfma_f32_32x32x2f32`, the j-th taking element j of the same pack from A and B.
      Both operands name the same K index on every lane, so the contraction is exact
      fp32 (AOTriton's `TRITON_F32_DEFAULT=ieee` class), and the accumulator layout, the
      softmax, masks, LSE, sink and dropout are the 16-bit path's unchanged. P stays f32.
    - **Path:** f32 takes the gfx942-class path on every arch: no hardware transpose
      read, no K16 MFMA, no permlane O store, no DMA and no KV global prefetch (all
      16-bit only). gfx950 runs it as gfx942 does. The K tile keeps its +4 row pad on
      every arch.
    - **Widths and bytes:** global K/V loads are 8 f32 per lane (32 bytes, the 16-bit
      path's 16 elements); `ELEM_BYTES` (4) sizes the Q/O/K/V descriptor spans and bases
      and the O and zero-O stores. O is stored as one dwordx4 per lane group, by its
      bits, so modification 32's padded-row zeros stay an integer select.
    - **Q:** re-read per K-step from head dim 256 (16-bit: past 256), its packs being
      twice as wide.
    - **Denormals:** f32 builds keep f32 denormal inputs (no DAZ attribute;
      `.amdhsa_float_denorm_mode_32` 3), as PyTorch and hipDNN's reference do.
    - **Fix (K chunks with Q in registers):** a K-chunked build that holds Q indexed
      Q's packs past the first chunk; Q now loads every chunk's packs. Upstream chunks
      K only with Q re-read, so no existing object takes this path.
    - **Objects:** every bf16/f16 object is byte-identical to modification 34's.
    - Not yet upstream.

The repository's `black` hook reformats these files on commit; that changes
formatting only (the syntax trees are unchanged).
