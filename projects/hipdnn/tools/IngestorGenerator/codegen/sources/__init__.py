# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Adapters producing a pack/kernel skeleton from something other than a
hand-authored YAML config. Each returns the shape the config loader builds
(``codegen.models``), so the rest of the pipeline is adapter-agnostic."""

from .base import SourceAdapter, SourceAdapterResult
from .hiprtc import HiprtcAdapter
from .interactive import InteractiveAdapter
from .rocke import RockeAdapter, RockeIntrospectionError, introspect

__all__ = [
    "SourceAdapter",
    "SourceAdapterResult",
    "HiprtcAdapter",
    "InteractiveAdapter",
    "RockeAdapter",
    "RockeIntrospectionError",
    "introspect",
]
