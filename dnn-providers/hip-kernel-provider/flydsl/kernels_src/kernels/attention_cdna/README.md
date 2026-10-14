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
    eight 4-element runs and adds `bias / scale` to the unscaled scores, which the
    kernel scales inside the exponent. Other builds never read the slot.

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
    - **A row that reaches keys:** adds `exp(sink - max)` to its denominator, and its
      LSE includes it.
    - **A row that reaches none:** O = 0, LSE = the sink.
    - **Split launches:** the partials carry no sink; the merge kernel adds it once.
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
    threshold). The bias and ALiBi slope join the raw scores divided by the scale, and
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
    alone). The LSE is now `l > 0 ? m * scale + log(l) : -inf`, its log without the
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

The repository's `black` hook reformats these files on commit; that changes
formatting only (the syntax trees are unchanged).
