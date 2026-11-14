# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Explicit structural and compiler gates excluded from normal discovery."""

from __future__ import annotations

_preflight_loaded = False


def mark_preflight_loaded() -> None:
    global _preflight_loaded
    _preflight_loaded = True


def preflight_loaded() -> bool:
    return _preflight_loaded
