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

The repository's `black` hook reformats these files on commit; that changes
formatting only (the syntax trees are unchanged).
