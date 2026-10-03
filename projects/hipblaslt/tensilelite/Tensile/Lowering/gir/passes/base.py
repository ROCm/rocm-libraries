# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Pass base class.
"""

from __future__ import annotations


class Pass:
    """A pass that may mutate a GIR program."""

    def run(self, prog, am):
        """Mutate `prog` in place."""
        raise NotImplementedError
