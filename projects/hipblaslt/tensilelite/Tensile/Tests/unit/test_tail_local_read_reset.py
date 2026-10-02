# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.KernelWriter import _tailResetsLocalReadOffsets

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "one_lds_buffer, tdm_a, tdm_b, wider_local_read, deep_ring, expected",
    [
        (False, False, False, False, False, True),  # Non-TDM tail always resets.
        (False, True, False, False, False, True),  # TDM on one side only is not a TDM tail.
        (False, True, True, False, False, False),  # TDM tail stays in the swap-parity buffer.
        (False, True, True, True, False, True),  # Wider local reads recompute the offsets.
        (False, True, True, False, True, True),  # The ring's tail writes buffer 0.
        (True, False, False, False, False, False),  # One LDS buffer has nothing to reset.
    ],
)
def test_tail_local_read_reset_predicate(
    one_lds_buffer, tdm_a, tdm_b, wider_local_read, deep_ring, expected
):
    kernel = {"1LDSBuffer": one_lds_buffer, "enableTDMA": tdm_a, "enableTDMB": tdm_b}

    assert _tailResetsLocalReadOffsets(kernel, wider_local_read, deep_ring) is expected
