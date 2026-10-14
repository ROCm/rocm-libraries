# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Starting point for oracle-free kernel tests on gfx942."""

import pytest

from rocke.runtime.hip_module import get_device_arch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(get_device_arch() != "gfx942", reason="requires a gfx942 GPU"),
]


def test_placeholder() -> None:
    """Keep the suite collectable until oracle-free checks are added."""
    pass
