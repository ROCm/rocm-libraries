# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The trivial adapter: a human (or the driving skill) supplies every field
directly, so there is nothing to infer from a source file at all."""

from pathlib import Path

from .base import SourceAdapterResult


class InteractiveAdapter:
    """No inference. ``infer()`` always returns an empty result -- callers
    using this adapter are expected to build the ``IngestorConfig`` fields
    themselves (e.g. the skill's create flow after its own Step 2 inference
    conversation with a human), not to read anything from ``sources``.
    """

    def infer(self, *sources: Path) -> SourceAdapterResult:
        return SourceAdapterResult(kernels=[], suggested_pack_count=1)
