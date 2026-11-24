# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 tiled-2D QK paths that can hold fp8 K in LDS, shared by the IR
test and the GPU numeric test so both cover the same set of paths."""

from __future__ import annotations

# The knobs that keep K in LDS as raw fp8 bytes, dequantized on read.
FP8_K_IN_LDS = dict(kv_storage_dtype="fp8e4m3", use_fp8_mfma_qk=True)

# The kernel-spec knobs that select each QK path.
QK_PATHS = {
    "16x16": dict(),
    "32x32 non-transposed": dict(use_mfma_32x32=True, block_m_per_warp=32),
    "32x32 transposed": dict(
        use_mfma_32x32=True, use_transposed_qk_32x32=True, block_m_per_warp=32
    ),
}
