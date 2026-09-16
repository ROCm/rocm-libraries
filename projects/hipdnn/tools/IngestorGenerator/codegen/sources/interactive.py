# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The trivial adapter: a human (or the driving skill) supplies every field
directly, so there is nothing to infer from a source file at all."""

from pathlib import Path

from .base import SourceAdapterResult


class InteractiveAdapter:
    """No inference. ``infer()`` always returns an empty result -- callers build
    the ``IngestorConfig`` fields themselves rather than read ``sources``."""

    def infer(self, *sources: Path) -> SourceAdapterResult:
        return SourceAdapterResult(kernels=[], suggested_pack_count=1)
