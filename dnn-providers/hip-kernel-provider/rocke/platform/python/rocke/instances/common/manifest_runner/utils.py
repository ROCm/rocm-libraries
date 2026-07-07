# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small host helpers shared by manifest-runner problem builders."""

from __future__ import annotations

import ctypes


def require_numpy():
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("rocke.run_manifest requires numpy") from e
    return np


def nbytes(a) -> int:
    return int(a.nbytes)


def as_u8_buffer(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)
