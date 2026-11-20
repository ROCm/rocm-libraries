# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every shipped fixed-length tuned general key gets a dispatch numeric case.

CPU only: :func:`_attention_bwd_key_cases.general_key_case` is host logic, so
the coverage of the device suite's ``test_tuned_general_keys`` is checked
here on every arch of ``routing.ARCH_TUNING``, on any host -- a tuned key
that loses its numeric case fails without a GPU.
"""

from __future__ import annotations

import pytest

from dispatch.attention_bwd import routing

from ._attention_bwd_key_cases import general_key_case


@pytest.mark.parametrize("arch", sorted(routing.ARCH_TUNING))
def test_every_fixed_tuned_key_has_a_numeric_case(arch):
    """Every fixed-length tuned general key of ``arch`` (window keys
    included) gets a dispatch numeric case: none is silently left out."""
    keys = [
        key
        for key in routing.ARCH_TUNING[arch].GENERAL_TUNED_TABLE
        if key[7] == "fixed"
    ]
    assert keys, f"{arch}: no fixed-length tuned general key"
    missing = [key for key in keys if general_key_case(arch, key) is None]
    assert not missing, missing
