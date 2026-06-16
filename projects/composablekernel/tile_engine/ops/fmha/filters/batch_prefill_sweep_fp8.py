# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Filter: comprehensive FP8 PER_TOKEN_HEAD batch-prefill sweep (hdim 128).

Scopes the generated config space to the production-relevant AITER path
(attention_with_kvcache_prefill_fp8, QPERTOKEN_PERHEAD / KPERTOKEN_PERHEAD /
VPERHEAD) while keeping the knob space "comprehensive" so we can re-confirm the
best config now that the v_descale epilogue fold + s_setprio bracket
optimization is shipped in the kernel.

Pair this filter with configs/batch_prefill_sweep_fp8.json, which sets:
    data_type = fp8bf16, mask = top_left (causal), qscale = per_token_head,
    block_per_cu = [-1, 1, 2, 3]   <-- the bpc sweep (config-driven, not filter)

The bpc sweep CANNOT be enabled from this filter -- a filter can only *reject*
configs, it cannot add block_per_cu variants to the generated space. The four
bpc values are produced by the JSON's trait_config.block_per_cu.values and this
filter merely keeps them.

Dimensions covered (the "exhaustive" part):
  * tile shapes:      ALL compilable hdim128 fp8 tiles (tile_m0/n0/k0/n1/k1/k0max)
  * block_per_cu:     {-1, 1, 2, 3}   (from the JSON config)
  * page_size:        {16, 64, 1024}  (excludes the degenerate ps=1 path;
                                       64 requires the additive instance_gen edit)
  * kv_memory_layout: {vectorized, linear}
  * kv_lookup_table:  {vllm, sglang}  (the vLLM lookup variant + sglang)
  * logits:           {on, off}

Excluded (out of scope): bf16 / non-fp8 dtypes, pertensor / kv_blockscale quant,
non-causal & generic masks, bias/alibi, hdims != 128.

Usage:
    python fmha_benchmark.py configs/batch_prefill_sweep_fp8.json \
        --filter-file filters/batch_prefill_sweep_fp8.py \
        --problems "1,8,16384,128"
"""

# Knob allow-lists (kept explicit so the swept space is self-documenting).
_PAGE_SIZES = (16, 64, 1024)
_KV_LAYOUTS = ("vectorized", "linear")
_KV_LOOKUPS = ("vllm", "sglang")
_BLOCK_PER_CU = (-1, 1, 2, 3)


def filter_config(c) -> bool:
    return (
        c.hdim_q == 128
        and c.qscale == "per_token_head"
        and c.page_size in _PAGE_SIZES
        and c.kv_memory_layout in _KV_LAYOUTS
        and c.kv_lookup_table in _KV_LOOKUPS
        and c.block_per_cu in _BLOCK_PER_CU
        # logits on/off both kept (no constraint) -- listed for clarity:
        and c.logits in (True, False)
    )
