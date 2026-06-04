# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Filter: pin the exact kernel AITER runs for FP8 PER_TOKEN_HEAD batch-prefill.

Matches AITER's production kernel for apples-to-apples timing:
  tile 128x128x32x128x32x128 (bk0=32), hdim 128, page_size 16, linear KV layout,
  per_token_head quant.
"""


def filter_config(c) -> bool:
    return (
        c.hdim_q == 128
        and c.tile_k0 == 32
        and c.tile_m0 == 128
        and c.tile_n0 == 128
        and c.qscale == "per_token_head"
        and c.page_size == 16
        and c.kv_memory_layout in ("linear", "vectorized")
    )
