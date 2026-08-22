# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Adapters producing a normalized ``EngineSpec``/pack/kernel skeleton from
something other than a hand-authored YAML config.

Every adapter here returns the same shape the config loader itself builds
(``codegen.models``), so the rest of the pipeline -- pre-mint checks,
UUID minting, template rendering -- never needs to know which adapter
produced its input. v1 ships ``interactive`` (a human/skill fills every
field) and ``hiprtc`` (scan a ``.cpp`` for ``__global__`` entry points).

``rocke`` is a later adapter, added once the packer and kpack launcher
land (per the design's ruling) -- deliberately absent, not stubbed. Adding
it later means adding a new module here implementing the same
``SourceAdapter`` protocol; nothing else in this package changes.
"""

from .base import SourceAdapter, SourceAdapterResult
from .hiprtc import HiprtcAdapter
from .interactive import InteractiveAdapter

__all__ = [
    "SourceAdapter",
    "SourceAdapterResult",
    "HiprtcAdapter",
    "InteractiveAdapter",
]
