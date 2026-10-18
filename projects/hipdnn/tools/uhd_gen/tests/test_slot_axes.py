# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RFC 0019 §6.3 check 2 against one slot's KMD and knobs (RFC 0021 §8)."""
import pytest

from uhd_gen.features import require_admissible_kernel_axes


def test_a_slot_ranker_is_checked_against_its_own_kmd_and_knobs():
    signature = ["$kernel.block_m", "$kernel.num_warps"]
    # dq's KMD declares both fields, but it exposes only block_m.
    require_admissible_kernel_axes(
        signature, ["block_m", "num_warps"], ["block_m", "num_warps"], "model m", "dkdv"
    )
    with pytest.raises(
        ValueError,
        match=r"ranks on \[num_warps\], which slot 'dq' of the UED does not expose as "
        r"knobs \[block_m\]",
    ):
        require_admissible_kernel_axes(
            signature, ["block_m"], ["block_m", "num_warps"], "model m", "dq"
        )
    with pytest.raises(
        ValueError,
        match=r"reads \[num_warps\], which the KMD of slot 'prep' does not declare",
    ):
        require_admissible_kernel_axes(signature, ["block_m"], ["block_m"], "m", "prep")
