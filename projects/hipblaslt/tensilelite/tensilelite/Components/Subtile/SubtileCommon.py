# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared utility functions for Subtile emitters."""


def _isPackPair(a, b):
    """True when a,b are a consecutive even-aligned VGPR pair for packed VALU."""
    return (a % 2 == 0) and (b == a + 1)
