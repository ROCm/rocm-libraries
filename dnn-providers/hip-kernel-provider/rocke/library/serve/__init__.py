# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke-serve``: a JSON subprocess entry point for external kernel tooling.

An orchestrator that has profiled a serving workload knows which attention
kernel is hot and what shapes reached it, but it cannot build a kernel. rocKE
can, and this package is the seam: JSON in, a planned -- and where a GPU is
present, verified and measured -- kernel out.

The three stages are separate modules because they have different requirements,
and collapsing them would drag the strictest one over everything:

- :mod:`.protocol` -- the wire format. Imports nothing from the library, so a
  request can be validated anywhere.
- :mod:`.planner` -- runs the production dispatch registry. Needs the library,
  needs no device, and is reproducible for an arch that is not attached.
- :mod:`.runner` -- verifies and times. Needs torch and a GPU, and is therefore
  the only optional stage.

The split is what makes the useful degraded mode possible: on a machine with no
GPU, planning still answers whether rocKE serves a shape at all, which is the
question the caller most needs answered before it spends a node on the rest.
"""

from .protocol import (
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    ProtocolError,
    ServeRequest,
    ShapeEntry,
    make_result,
)

__all__ = [
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "ProtocolError",
    "ServeRequest",
    "ShapeEntry",
    "make_result",
]
