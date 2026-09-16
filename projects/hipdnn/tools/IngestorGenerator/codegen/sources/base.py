# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The one interface every ``sources/`` adapter implements.

An adapter's job stops at *candidates*: entry points, signature-derived KMD
field guesses, and a pack-count hint. It never decides the engine name, the
arch list, which fields become knobs, or the UMD-vs-graph_match split -- those
stay engine-level judgment calls a human confirms.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class CandidateKernel:
    """One inferred kernel entry point."""

    entry_point: str
    source_file: str
    #: Template parameters / #define names this entry point varies along, as raw
    #: strings -- the adapter's best guess at KMD field names.
    template_params: list[str] = field(default_factory=list)


@dataclass
class SourceAdapterResult:
    """What a ``sources/`` adapter hands back: candidates only, never a finished
    ``IngestorConfig``."""

    kernels: list[CandidateKernel] = field(default_factory=list)
    #: Best-effort pack-count guess: one pack per distinct source file
    #: implementing a genuinely different operation, or one pack for several
    #: instantiations of the same operation.
    suggested_pack_count: int = 1


class SourceAdapter(Protocol):
    """Produces ``SourceAdapterResult`` candidates from some external input.

    Implementations: ``InteractiveAdapter`` (a human/skill fills every
    field directly, no inference), ``HiprtcAdapter`` (scans one or more
    ``.cpp``/``.hip`` files for ``__global__`` entry points) and
    ``RockeAdapter`` (introspects a rocKE builder's spec surface). Each sits
    behind this protocol alone, so adding another requires no change here
    or to any existing adapter.
    """

    def infer(self, *sources: Path) -> SourceAdapterResult: ...
