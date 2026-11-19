# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmarks/common/conv_det_check.py --output-init poison: an element the
# launch left at the all-0xFF poison counts as unwritten, in every output
# dtype. Host-only.

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from benchmarks.common.conv_det_check import count_unwritten


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_count_unwritten(dtype):
    t = torch.empty((2, 3, 5), dtype=dtype)
    t.view(torch.uint8).fill_(0xFF)
    assert count_unwritten(t) == t.numel()
    t.fill_(0.5)
    assert count_unwritten(t) == 0
    flat = t.view(-1)
    flat.view(torch.uint8).view(-1, t.element_size())[3].fill_(0xFF)
    assert count_unwritten(t) == 1
    # A partly written element (one byte changed) is written.
    flat.view(torch.uint8).view(-1, t.element_size())[3, 0] = 0
    assert count_unwritten(t) == 0
