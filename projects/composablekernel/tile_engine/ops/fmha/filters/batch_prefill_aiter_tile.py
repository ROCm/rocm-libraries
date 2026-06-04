# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Filter: pin the exact kernels AITER runs for FP8 PER_TOKEN_HEAD batch-prefill.

Matches AITER's production kernels for apples-to-apples timing (hdim 128,
page_size 16, linear/vectorized KV layout, per_token_head quant):
  tile 128x128x32x128x32x128  (bk0=32)
  tile 128x128x128x128x32x128 (bk0=128)
Both share bm0=128, bn0=128, bn1=128, bk1=32, bk0max=128; they differ only in
bk0 (the QK gemm unroll).
"""


def filter_config(c) -> bool:
    return (
        c.hdim_q == 128
        and c.tile_k0 in (32, 128)
        and c.tile_m0 == 128
        and c.tile_n0 == 128
        and c.qscale == "per_token_head"
        and c.page_size == 16
        and c.kv_memory_layout in ("linear", "vectorized")
    )
