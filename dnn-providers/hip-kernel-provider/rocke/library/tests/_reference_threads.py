# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run the independent pieces of a numpy reference on a few threads.

A host reference over long sequences or many heads (dense SDPA, a chunked paged
decode) splits into pieces computed on their own: one head, or one head and
row chunk. Each piece spends its time in numpy's elementwise passes, which run
on one core and release the GIL, so pieces run in parallel on threads. Each
worker holds one piece's float64 score temporaries (up to a few hundred MiB on
the largest references), so the pool is small: four workers give most of the
gain, more workers cost memory in proportion for little more, and the pool
never exceeds the CPUs this process may use.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

from rocke.tests.host_cpus import available_cpus

__all__ = ["REFERENCE_WORKERS", "run_reference_pieces"]

#: Threads :func:`run_reference_pieces` runs pieces on.
REFERENCE_WORKERS = min(4, available_cpus())


def run_reference_pieces(pieces: Iterable[Callable[[], None]]) -> None:
    """Run each of ``pieces`` (each writes only its own part of the result) on
    :data:`REFERENCE_WORKERS` threads and wait for all of them; the first
    piece, in order, that raised re-raises its error."""
    with ThreadPoolExecutor(REFERENCE_WORKERS) as pool:
        jobs = [pool.submit(piece) for piece in pieces]
        for job in jobs:
            job.result()
