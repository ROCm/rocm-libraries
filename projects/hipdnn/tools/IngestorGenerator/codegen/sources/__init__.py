# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Adapters producing a normalized pack/kernel skeleton from something other
than a hand-authored YAML config.

Every adapter returns the same shape the config loader itself builds
(``codegen.models``), so the rest of the pipeline -- pre-mint checks, UUID
minting, template rendering -- never needs to know which one produced its input.
"""

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
