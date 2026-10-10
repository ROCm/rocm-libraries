<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# FlyDSL attention for CDNA (gfx942, gfx950)

Vendored kernel sources for the `hipkernel:flydsl_sdpa_cdna` engine. The RDNA
attention kernels are separate, under `../attention_rdna/`, and share nothing
with these but the `kernels/common` helpers.

| File | Upstream | Serves |
|---|---|---|
| `flash_attn_gfx950.py` | FlyDSL `kernels/attention/flash_attn_gfx950.py` | gfx950: dual-wave, software-pipelined MFMA forward, head_dim 64 and 128 |
| `flash_attn_generic.py` | FlyDSL `kernels/attention/flash_attn_generic.py` | gfx942 (and gfx950 off the dual-wave path): generic MFMA forward |
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

The repository's `black` hook reformats these files on commit; that changes
formatting only (the syntax trees are unchanged).
