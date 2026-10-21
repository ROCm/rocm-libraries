# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit convolution enrollments shared by offline qualification and reference verification."""

from __future__ import annotations

from pathlib import Path

from reference_common.registry import AdapterRegistry

_REGISTRY = AdapterRegistry(
    __name__, Path(__file__).resolve().parent, label="convolution"
)
ARCHITECTURES = _REGISTRY.architectures
get_architecture = _REGISTRY.get_architecture
baseline_lock = _REGISTRY.baseline_lock
