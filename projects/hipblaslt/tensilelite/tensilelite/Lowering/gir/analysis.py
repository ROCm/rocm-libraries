# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GIR analysis INFRASTRUCTURE -- the base class + the lazy cached manager.

This module is infra ONLY. The concrete analyses each live in their own module under
`gir/analyses/` as an `Analysis` subclass.

"""

from __future__ import annotations


class Analysis:
    """Base class for a GIR analysis.

    Subclass overrides `run(self, prog, am) -> result` (pure -- never mutates `prog`).  A
    parameterized analysis (e.g. GuardSite(kind)) overrides `cache_key` to fold its params in
    so distinct parameterizations cache separately.
    """

    @property
    def cache_key(self):
        return type(self).__name__

    def run(self, prog, am):
        raise NotImplementedError


class AnalysisManager:
    """StinkyTofu-shaped: lazy + cached + invalidated.  Cache is keyed on (analysis.cache_key,
    prog.version), so a Pass that mutates + bumps `prog.version` transparently invalidates every
    stale result; `invalidate()` also clears eagerly."""

    def __init__(self):
        self._cache = {}

    def get(self, analysis, prog):
        """`analysis` is an Analysis INSTANCE; returns its cached result for this prog version."""
        key = (analysis.cache_key, prog.version)
        if key not in self._cache:
            self._cache[key] = analysis.run(prog, self)
        return self._cache[key]

    def invalidate(self):
        """Drop cached results."""
        self._cache.clear()
