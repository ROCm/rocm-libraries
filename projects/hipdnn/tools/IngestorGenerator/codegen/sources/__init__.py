# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Adapters producing a normalized ``EngineSpec``/pack/kernel skeleton from
something other than a hand-authored YAML config.

Every adapter here returns the same shape the config loader itself builds
(``codegen.models``), so the rest of the pipeline -- pre-mint checks,
UUID minting, template rendering -- never needs to know which adapter
produced its input. v1 ships ``interactive`` (a human/skill fills every
field), ``hiprtc`` (scan a ``.cpp`` for ``__global__`` entry points), and
``rocke`` (introspect a rocKE builder's spec dataclass).

``rocke`` differs from the other two in kind: it reads type annotations
rather than text, so its extraction is exact rather than best-effort. It
is also the only adapter whose output cannot be authored in the
``direct_load`` dialect -- a rocKE kernel reaches the runtime already
lowered to ``kpack`` by ``hkp_pack``, never as ``rocke_builder``.

``rocke`` imports the rocKE library lazily, inside ``sources/rocke.py``,
so IngestorGenerator keeps working with rocKE absent from PYTHONPATH.
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
