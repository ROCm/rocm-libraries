# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Adapter for fields supplied directly by a human or the driving skill."""

from pathlib import Path

from .base import SourceAdapterResult


class InteractiveAdapter:
    """No inference: ``infer()`` returns an empty result and callers build the
    ``IngestorConfig`` fields themselves."""

    def infer(self, *sources: Path) -> SourceAdapterResult:
        return SourceAdapterResult(kernels=[], suggested_pack_count=1)
