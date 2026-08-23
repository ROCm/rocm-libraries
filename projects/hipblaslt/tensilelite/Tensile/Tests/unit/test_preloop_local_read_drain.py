# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.KernelWriter import _needsPreLoopLocalReadDrain


@pytest.mark.parametrize(
    "use_custom_schedule, force_unroll_sub_iter, prior_drain, expected",
    [
        (True, True, False, True),
        (True, True, True, False),
        (True, False, False, False),
        (False, True, False, False),
    ],
)
def test_preloop_local_read_drain_predicate(
    use_custom_schedule, force_unroll_sub_iter, prior_drain, expected
):
    kernel = {
        "UseCustomMainLoopSchedule": use_custom_schedule,
        "ForceUnrollSubIter": force_unroll_sub_iter,
    }

    assert _needsPreLoopLocalReadDrain(kernel, prior_drain) is expected
