# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Filter: FP8 batch-prefill kernels using the PER_TOKEN_HEAD quant path.

Keeps only h128 kernels with Q/K per-token-per-head, V per-head FP8 quant --
the quantization scheme used by attention_with_kvcache_prefill_fp8
(QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD).

Usage:
    python fmha_benchmark.py configs/batch_prefill_fp8.json \
        --filter-file filters/batch_prefill_per_token_head.py \
        --problems "1,8,1,3904,3904,128"
"""


def filter_config(c) -> bool:
    return c.hdim_q == 128 and c.qscale == "per_token_head"
