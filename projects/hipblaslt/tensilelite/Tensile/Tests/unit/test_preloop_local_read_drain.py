# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.KernelWriter import _needsPreLoopLocalReadDrain


@pytest.mark.parametrize(
    "use_custom_schedule, force_unroll_sub_iter, num_iters_plr, prior_drain, expected",
    [
        (True, True, 1, False, True),
        (True, True, 0, False, False),
        (True, True, 1, True, False),
        (True, False, 1, False, False),
        (False, True, 1, False, False),
    ],
)
def test_preloop_local_read_drain_predicate(
    use_custom_schedule, force_unroll_sub_iter, num_iters_plr, prior_drain, expected
):
    kernel = {
        "UseCustomMainLoopSchedule": use_custom_schedule,
        "ForceUnrollSubIter": force_unroll_sub_iter,
    }

    assert _needsPreLoopLocalReadDrain(kernel, num_iters_plr, prior_drain) is expected
