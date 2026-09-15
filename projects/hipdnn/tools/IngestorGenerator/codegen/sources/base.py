# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The one interface every ``sources/`` adapter implements.

An adapter's job stops at *candidates*: entry points, signature-derived KMD
field guesses, and a pack-count hint. It never decides the engine name,
the arch list, which fields become knobs, or the UMD-vs-graph_match split
-- those stay engine-level judgment calls per the design (the same
judgment ``hipdnn-ingestor-engine``'s create flow asks a human to confirm
in one batch). An adapter that decided those on its own would be guessing
at exactly the things ``DESIGN-PROPOSAL.md`` §4 reserves for a human.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class CandidateKernel:
    """One inferred kernel entry point."""

    entry_point: str
    source_file: str
    #: Template parameters / #define names this entry point varies along,
    #: as raw strings -- the adapter's best guess at KMD field names. A
    #: human still assigns each one a KMD type before it becomes real.
    template_params: list[str] = field(default_factory=list)


@dataclass
class SourceAdapterResult:
    """What a ``sources/`` adapter hands back: candidates only, never a
    finished ``IngestorConfig``. Every field here still needs a human or
    the skill's create flow to confirm or discard it."""

    kernels: list[CandidateKernel] = field(default_factory=list)
    #: Best-effort pack-count guess: one pack per distinct source file
    #: implementing a genuinely different operation, or one pack for
    #: several instantiations of the same operation. See
    #: 07-descriptor-generation.md §2's "pack_count" config-surface axis.
    suggested_pack_count: int = 1


class SourceAdapter(Protocol):
    """Produces ``SourceAdapterResult`` candidates from some external input.

    Implementations: ``InteractiveAdapter`` (a human/skill fills every
    field directly, no inference), ``HiprtcAdapter`` (scans one or more
    ``.cpp``/``.hip`` files for ``__global__`` entry points). ``rocke`` is
    a later adapter behind this same protocol, added once the packer and
    kpack launcher land -- there is deliberately no ``RockeAdapter`` here
    yet; adding one later requires no change to this protocol or to any
    other adapter.
    """

    def infer(self, *sources: Path) -> SourceAdapterResult: ...
