# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit SDPA enrollments shared by offline qualification and reference verification."""

from __future__ import annotations

from pathlib import Path

from reference_common.registry import AdapterRegistry

_REGISTRY = AdapterRegistry(__name__, Path(__file__).resolve().parent, label="SDPA")
ARCHITECTURES = _REGISTRY.architectures
get_architecture = _REGISTRY.get_architecture
variants = _REGISTRY.variants
resolve_variant = _REGISTRY.resolve_variant
get_adapter = _REGISTRY.get_adapter
baseline_lock = _REGISTRY.baseline_lock
