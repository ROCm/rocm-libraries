# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Interface every ``sources/`` adapter implements. Adapters produce
candidates only -- entry points, KMD field guesses, a pack-count hint. Engine
name, arch list, knob selection and the UMD-vs-graph_match split stay human
decisions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class CandidateKernel:
    """One inferred kernel entry point."""

    entry_point: str
    source_file: str
    #: Template parameter / #define names this entry point varies along; the
    #: adapter's guess at KMD field names.
    template_params: list[str] = field(default_factory=list)


@dataclass
class SourceAdapterResult:
    """Adapter output: candidates only, never a finished ``IngestorConfig``."""

    kernels: list[CandidateKernel] = field(default_factory=list)
    #: Pack-count guess: one pack per source file implementing a distinct
    #: operation; instantiations of one operation share a pack.
    suggested_pack_count: int = 1


class SourceAdapter(Protocol):
    """Produces ``SourceAdapterResult`` candidates from some external input.
    Implementations: ``InteractiveAdapter`` (no inference), ``HiprtcAdapter``
    (scans ``.cpp``/``.hip`` for ``__global__`` entry points), ``RockeAdapter``
    (introspects a rocKE builder spec)."""

    def infer(self, *sources: Path) -> SourceAdapterResult: ...
