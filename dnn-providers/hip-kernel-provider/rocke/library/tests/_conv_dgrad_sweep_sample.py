# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The sampled dgrad sweep that ``test_conv_dgrad_correctness`` verifies.

:data:`SWEEP_SAMPLE_TILINGS` pins the tile configurations the dgrad sweep of
``benchmarks.common.benchmark_implicit_gemm_conv`` draws with ``--sample 0.05``
(seed 0): its grid (``dgrad_sweep_grid``) sampled by ``sample_combos``, in
draw order. The sweep's split-K value is one constant last axis, so the draw
is the same for every split-K mode, and :func:`sweep_sample` pairs these
tilings with a test's own. The list is pinned rather than recomputed so that a
change to the sweep's axes fails a test instead of silently changing what the
tests cover (``test_the_pinned_sample_is_the_sweeps_draw``).

:func:`first_candidates_per_class` picks, from those combos, the few that check
each pipeline and epilogue class: the pull-request tier checks them, and the
scheduled tier checks :func:`every_candidate` of the sample.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, Future, wait
from typing import Any, NamedTuple

from benchmarks.common.benchmark_implicit_gemm_conv import DgradCombo

__all__ = [
    "CLASS_SAMPLE_SIZE",
    "SWEEP_SAMPLE_FRACTION",
    "SWEEP_SAMPLE_SEED",
    "SWEEP_SAMPLE_TILINGS",
    "ClassSample",
    "every_candidate",
    "first_candidates_per_class",
    "sweep_sample",
]

#: How many sweep candidates of each (pipeline, epilogue) class the class sample
#: checks.
CLASS_SAMPLE_SIZE = 2

#: The ``--sample`` fraction and ``--seed`` the pinned list was drawn with.
SWEEP_SAMPLE_FRACTION = 0.05
SWEEP_SAMPLE_SEED = 0

#: ``(tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline, epilogue)``
#: of each sampled configuration: a ``DgradCombo`` without its ``split_k``.
SWEEP_SAMPLE_TILINGS: tuple[tuple[int, int, int, int, int, int, str, str], ...] = (
    (256, 32, 32, 4, 1, 32, "mem", "cshuffle"),
    (256, 64, 64, 1, 1, 32, "mem", "default"),
    (32, 64, 16, 1, 2, 32, "compv3", "cshuffle"),
    (128, 64, 64, 1, 1, 32, "compv3", "default"),
    (256, 256, 32, 2, 4, 32, "wavelet", "default"),
    (256, 256, 16, 1, 4, 16, "mem", "cshuffle"),
    (256, 64, 16, 8, 1, 16, "compv4", "default"),
    (128, 128, 64, 1, 4, 32, "basic", "cshuffle"),
    (256, 128, 64, 4, 2, 16, "basic", "default"),
    (128, 256, 64, 2, 8, 16, "wavelet", "default"),
    (128, 32, 16, 2, 1, 16, "basic", "default"),
    (256, 256, 32, 1, 2, 16, "basic", "default"),
    (64, 128, 16, 1, 1, 16, "mem", "cshuffle"),
    (128, 128, 16, 4, 8, 16, "wavelet", "cshuffle"),
    (64, 128, 16, 1, 1, 16, "basic", "cshuffle"),
    (32, 256, 64, 2, 8, 16, "compv3", "cshuffle"),
    (128, 64, 32, 2, 1, 16, "compv4", "default"),
    (256, 256, 64, 4, 1, 16, "compv4", "cshuffle"),
    (64, 128, 16, 2, 4, 32, "wavelet", "cshuffle"),
    (128, 128, 64, 4, 2, 32, "mem", "cshuffle"),
    (64, 16, 32, 4, 1, 16, "basic", "default"),
    (32, 256, 16, 1, 1, 16, "basic", "default"),
    (128, 256, 16, 8, 1, 16, "basic", "cshuffle"),
    (256, 128, 64, 2, 2, 16, "compv4", "cshuffle"),
    (64, 16, 64, 4, 1, 16, "basic", "cshuffle"),
    (128, 256, 64, 1, 8, 32, "wavelet", "default"),
    (256, 128, 16, 1, 2, 32, "compv3", "cshuffle"),
    (128, 256, 16, 1, 2, 16, "mem", "default"),
    (64, 256, 64, 2, 4, 32, "mem", "default"),
    (256, 128, 64, 4, 2, 32, "compv4", "cshuffle"),
    (256, 128, 16, 4, 2, 32, "compv3", "cshuffle"),
    (256, 256, 32, 8, 8, 16, "mem", "cshuffle"),
    (128, 64, 64, 1, 2, 32, "wavelet", "cshuffle"),
    (32, 128, 32, 1, 4, 16, "mem", "default"),
    (16, 128, 32, 1, 2, 16, "mem", "default"),
    (32, 256, 64, 2, 1, 16, "basic", "default"),
    (256, 64, 16, 2, 1, 16, "compv4", "default"),
    (16, 16, 32, 1, 1, 16, "basic", "default"),
    (256, 256, 16, 4, 1, 16, "wavelet", "default"),
    (128, 256, 32, 1, 1, 32, "basic", "default"),
    (128, 64, 16, 4, 2, 16, "wavelet", "default"),
    (128, 256, 16, 4, 1, 16, "basic", "default"),
    (32, 128, 32, 1, 4, 32, "mem", "cshuffle"),
    (64, 256, 32, 2, 2, 32, "mem", "default"),
    (128, 32, 16, 8, 1, 16, "compv3", "default"),
    (128, 64, 16, 1, 2, 32, "basic", "cshuffle"),
    (64, 128, 16, 1, 4, 32, "compv4", "default"),
    (256, 128, 16, 8, 4, 16, "basic", "cshuffle"),
    (32, 256, 64, 1, 4, 32, "compv4", "default"),
    (32, 256, 16, 2, 8, 16, "basic", "default"),
    (128, 256, 16, 2, 1, 16, "compv3", "cshuffle"),
    (256, 256, 32, 2, 1, 16, "compv3", "default"),
    (256, 256, 16, 2, 1, 32, "wavelet", "default"),
    (64, 32, 32, 2, 2, 16, "wavelet", "default"),
    (128, 128, 64, 1, 2, 16, "basic", "default"),
    (128, 128, 32, 2, 2, 32, "basic", "cshuffle"),
    (64, 64, 32, 1, 2, 16, "compv4", "default"),
    (128, 256, 32, 1, 1, 32, "mem", "cshuffle"),
    (256, 256, 64, 8, 4, 32, "compv3", "default"),
    (64, 256, 64, 2, 2, 16, "basic", "cshuffle"),
    (128, 128, 32, 1, 4, 32, "compv3", "default"),
    (256, 128, 16, 8, 1, 16, "mem", "default"),
    (32, 256, 64, 1, 8, 16, "mem", "cshuffle"),
    (256, 32, 32, 4, 1, 16, "wavelet", "default"),
    (128, 256, 16, 1, 2, 32, "compv4", "default"),
    (128, 64, 16, 2, 4, 16, "wavelet", "default"),
    (128, 128, 32, 2, 2, 16, "wavelet", "default"),
    (64, 256, 16, 4, 8, 16, "compv3", "default"),
    (64, 256, 32, 2, 1, 16, "compv3", "default"),
    (64, 256, 32, 1, 2, 32, "basic", "cshuffle"),
    (32, 32, 32, 1, 1, 32, "mem", "default"),
    (128, 64, 64, 1, 2, 32, "mem", "default"),
    (256, 128, 64, 4, 2, 16, "wavelet", "cshuffle"),
    (32, 128, 64, 1, 4, 16, "mem", "cshuffle"),
    (32, 256, 64, 1, 2, 32, "mem", "cshuffle"),
    (64, 64, 32, 4, 2, 16, "compv3", "cshuffle"),
    (64, 128, 16, 4, 8, 16, "mem", "default"),
    (32, 64, 16, 1, 1, 16, "compv3", "cshuffle"),
    (32, 256, 16, 2, 8, 16, "compv4", "default"),
    (256, 256, 64, 8, 4, 32, "wavelet", "cshuffle"),
    (256, 32, 64, 2, 2, 16, "mem", "default"),
    (256, 256, 64, 1, 2, 32, "compv4", "default"),
    (128, 128, 16, 2, 2, 32, "compv4", "cshuffle"),
    (256, 256, 32, 8, 8, 16, "basic", "cshuffle"),
    (128, 32, 64, 8, 1, 16, "basic", "default"),
    (128, 16, 64, 4, 1, 16, "compv4", "cshuffle"),
    (256, 64, 32, 8, 4, 16, "mem", "cshuffle"),
    (128, 128, 16, 2, 2, 16, "basic", "cshuffle"),
    (256, 128, 32, 1, 2, 16, "compv3", "default"),
    (256, 256, 16, 2, 8, 32, "mem", "cshuffle"),
    (128, 256, 64, 2, 4, 32, "compv4", "cshuffle"),
    (32, 256, 32, 1, 2, 16, "basic", "cshuffle"),
    (128, 256, 16, 2, 8, 32, "compv3", "cshuffle"),
    (64, 32, 64, 4, 2, 16, "basic", "cshuffle"),
    (256, 256, 16, 1, 4, 16, "basic", "cshuffle"),
    (128, 256, 32, 1, 4, 32, "compv3", "cshuffle"),
    (64, 256, 32, 2, 1, 32, "basic", "cshuffle"),
    (128, 64, 16, 4, 1, 32, "mem", "cshuffle"),
    (16, 128, 64, 1, 1, 16, "compv4", "cshuffle"),
    (128, 128, 16, 1, 2, 32, "mem", "default"),
    (64, 64, 16, 1, 2, 16, "basic", "cshuffle"),
    (128, 32, 16, 4, 1, 32, "compv3", "default"),
    (256, 16, 64, 2, 1, 16, "compv4", "cshuffle"),
    (64, 128, 64, 4, 2, 16, "compv3", "cshuffle"),
    (128, 256, 32, 1, 1, 16, "basic", "default"),
    (256, 64, 64, 4, 1, 16, "mem", "cshuffle"),
    (32, 128, 32, 1, 2, 32, "basic", "default"),
    (64, 16, 64, 4, 1, 16, "basic", "default"),
    (64, 128, 16, 2, 4, 16, "wavelet", "cshuffle"),
    (128, 32, 16, 2, 1, 32, "compv4", "default"),
    (32, 64, 32, 1, 2, 32, "mem", "cshuffle"),
    (256, 256, 64, 4, 2, 32, "compv3", "default"),
    (32, 256, 16, 1, 1, 32, "compv3", "default"),
    (32, 16, 16, 2, 1, 16, "wavelet", "cshuffle"),
    (64, 64, 32, 1, 1, 32, "basic", "default"),
    (64, 256, 32, 1, 8, 16, "basic", "default"),
    (64, 64, 16, 2, 2, 16, "mem", "cshuffle"),
    (256, 32, 64, 2, 1, 32, "basic", "cshuffle"),
    (32, 256, 64, 1, 4, 32, "basic", "cshuffle"),
    (256, 16, 32, 4, 1, 16, "compv4", "default"),
    (64, 64, 16, 1, 1, 32, "mem", "cshuffle"),
    (32, 32, 64, 1, 1, 32, "wavelet", "default"),
    (16, 256, 32, 1, 2, 16, "compv4", "default"),
    (64, 256, 32, 4, 1, 16, "basic", "default"),
    (64, 256, 32, 1, 1, 32, "mem", "default"),
    (64, 64, 32, 1, 1, 16, "basic", "cshuffle"),
    (256, 128, 64, 8, 1, 16, "mem", "cshuffle"),
    (128, 16, 16, 8, 1, 16, "mem", "default"),
    (32, 128, 32, 1, 2, 16, "mem", "default"),
    (16, 256, 32, 1, 8, 16, "compv3", "cshuffle"),
    (256, 64, 64, 2, 4, 16, "compv3", "cshuffle"),
    (64, 32, 16, 1, 1, 32, "compv3", "default"),
    (128, 64, 64, 1, 2, 16, "basic", "default"),
    (32, 128, 64, 1, 4, 32, "wavelet", "default"),
    (128, 32, 16, 4, 1, 32, "basic", "default"),
    (32, 128, 64, 2, 2, 16, "basic", "cshuffle"),
    (128, 128, 64, 1, 2, 16, "compv3", "default"),
    (128, 256, 64, 1, 1, 32, "basic", "cshuffle"),
    (256, 128, 16, 1, 8, 16, "compv4", "default"),
    (64, 256, 16, 2, 4, 32, "compv4", "default"),
    (256, 256, 32, 1, 1, 32, "mem", "cshuffle"),
    (256, 128, 64, 1, 1, 32, "compv3", "cshuffle"),
    (32, 64, 16, 1, 1, 32, "compv4", "cshuffle"),
    (64, 32, 16, 1, 1, 16, "compv3", "cshuffle"),
    (64, 256, 64, 1, 4, 16, "wavelet", "default"),
    (128, 64, 64, 1, 2, 32, "compv3", "default"),
    (128, 256, 64, 2, 8, 32, "compv4", "default"),
    (256, 128, 64, 1, 8, 16, "compv4", "default"),
    (64, 128, 64, 2, 8, 16, "compv4", "cshuffle"),
    (64, 256, 64, 2, 2, 32, "compv3", "default"),
    (32, 128, 16, 2, 2, 16, "mem", "cshuffle"),
    (64, 128, 32, 2, 8, 16, "compv3", "default"),
    (64, 128, 64, 1, 1, 32, "compv3", "cshuffle"),
    (128, 256, 32, 4, 1, 16, "basic", "default"),
    (256, 256, 64, 2, 2, 32, "compv4", "default"),
    (128, 64, 32, 2, 1, 16, "wavelet", "cshuffle"),
    (64, 64, 16, 1, 2, 32, "mem", "default"),
    (256, 128, 16, 4, 2, 16, "wavelet", "default"),
    (64, 256, 16, 1, 4, 16, "compv4", "default"),
    (16, 128, 16, 1, 8, 16, "wavelet", "default"),
    (256, 128, 64, 2, 1, 32, "wavelet", "cshuffle"),
    (256, 64, 32, 1, 2, 32, "compv4", "cshuffle"),
    (256, 256, 32, 2, 1, 32, "compv3", "default"),
    (128, 128, 64, 4, 4, 32, "compv3", "default"),
    (128, 256, 64, 2, 4, 32, "mem", "cshuffle"),
    (256, 32, 64, 1, 1, 16, "wavelet", "cshuffle"),
    (128, 64, 32, 2, 1, 32, "mem", "cshuffle"),
    (64, 128, 32, 1, 4, 32, "compv3", "cshuffle"),
    (16, 128, 16, 1, 4, 16, "compv3", "cshuffle"),
    (256, 128, 32, 4, 1, 16, "compv3", "cshuffle"),
    (32, 256, 16, 2, 2, 16, "compv4", "cshuffle"),
    (128, 256, 32, 1, 8, 16, "compv3", "cshuffle"),
    (32, 64, 32, 1, 2, 32, "basic", "default"),
    (128, 128, 16, 4, 4, 32, "mem", "cshuffle"),
    (64, 64, 64, 2, 1, 16, "basic", "cshuffle"),
    (128, 64, 16, 2, 1, 32, "compv4", "default"),
    (256, 128, 64, 8, 4, 16, "compv4", "default"),
    (128, 256, 64, 1, 4, 32, "mem", "default"),
    (128, 128, 32, 1, 4, 32, "wavelet", "default"),
    (128, 256, 64, 4, 1, 16, "compv4", "cshuffle"),
    (64, 64, 64, 1, 1, 32, "basic", "default"),
    (128, 128, 64, 4, 2, 32, "compv3", "cshuffle"),
    (256, 32, 32, 8, 2, 16, "wavelet", "cshuffle"),
    (256, 64, 32, 4, 1, 16, "basic", "cshuffle"),
    (32, 256, 32, 1, 1, 16, "compv3", "default"),
    (16, 16, 64, 1, 1, 16, "compv4", "default"),
    (64, 256, 32, 2, 4, 32, "mem", "default"),
    (128, 256, 32, 1, 2, 32, "basic", "default"),
    (64, 128, 32, 4, 4, 16, "compv3", "default"),
    (128, 64, 16, 2, 1, 16, "compv3", "default"),
    (128, 32, 32, 1, 1, 16, "compv4", "cshuffle"),
    (256, 128, 16, 8, 4, 32, "compv3", "default"),
    (256, 32, 16, 4, 2, 16, "compv3", "cshuffle"),
    (32, 32, 16, 2, 1, 16, "wavelet", "default"),
    (256, 64, 16, 4, 1, 32, "mem", "default"),
    (256, 64, 32, 8, 2, 16, "compv3", "default"),
    (32, 64, 32, 2, 1, 16, "wavelet", "default"),
    (64, 128, 64, 2, 1, 16, "compv4", "default"),
    (256, 128, 16, 8, 1, 16, "wavelet", "default"),
    (32, 128, 32, 1, 8, 16, "wavelet", "default"),
    (128, 64, 64, 1, 1, 32, "wavelet", "cshuffle"),
    (64, 128, 32, 2, 4, 32, "compv3", "cshuffle"),
    (256, 128, 16, 8, 2, 16, "compv3", "cshuffle"),
    (256, 256, 64, 2, 1, 16, "compv3", "cshuffle"),
    (256, 256, 16, 1, 8, 16, "compv3", "cshuffle"),
    (16, 16, 16, 1, 1, 16, "mem", "cshuffle"),
    (32, 64, 16, 1, 1, 16, "wavelet", "cshuffle"),
    (256, 256, 16, 4, 2, 16, "compv3", "cshuffle"),
    (128, 256, 16, 4, 2, 16, "mem", "default"),
    (128, 128, 64, 4, 8, 16, "compv3", "default"),
    (256, 128, 64, 1, 1, 16, "basic", "cshuffle"),
    (32, 64, 64, 1, 2, 16, "wavelet", "default"),
    (256, 64, 32, 4, 2, 16, "mem", "cshuffle"),
    (64, 256, 32, 1, 8, 16, "mem", "default"),
    (32, 256, 32, 1, 4, 16, "wavelet", "cshuffle"),
    (64, 64, 32, 4, 2, 16, "basic", "default"),
    (16, 128, 32, 1, 4, 16, "mem", "cshuffle"),
    (256, 64, 16, 4, 1, 16, "compv3", "cshuffle"),
    (256, 64, 32, 8, 1, 32, "mem", "default"),
    (16, 32, 32, 1, 1, 16, "compv4", "cshuffle"),
    (128, 16, 32, 8, 1, 16, "basic", "default"),
    (16, 128, 32, 1, 2, 16, "compv4", "default"),
    (16, 32, 16, 1, 1, 16, "basic", "default"),
    (256, 256, 64, 2, 1, 32, "wavelet", "default"),
    (64, 16, 32, 2, 1, 16, "compv3", "default"),
    (64, 256, 32, 2, 2, 16, "mem", "default"),
    (64, 64, 16, 2, 1, 16, "basic", "default"),
    (64, 256, 64, 1, 2, 32, "compv3", "default"),
    (128, 128, 64, 1, 4, 16, "compv4", "default"),
    (128, 128, 16, 4, 2, 32, "wavelet", "cshuffle"),
    (64, 256, 16, 4, 1, 16, "compv4", "cshuffle"),
    (64, 16, 64, 4, 1, 16, "mem", "cshuffle"),
    (256, 128, 64, 4, 1, 32, "compv3", "default"),
    (256, 64, 16, 1, 1, 32, "basic", "cshuffle"),
    (32, 256, 32, 1, 1, 32, "compv3", "default"),
    (16, 256, 32, 1, 2, 16, "wavelet", "cshuffle"),
    (128, 128, 16, 2, 2, 16, "mem", "default"),
    (256, 128, 32, 1, 8, 16, "mem", "cshuffle"),
    (64, 64, 16, 1, 1, 16, "wavelet", "default"),
    (128, 64, 32, 8, 1, 16, "compv3", "default"),
    (64, 64, 64, 1, 2, 32, "compv4", "cshuffle"),
    (256, 256, 32, 8, 4, 32, "compv3", "cshuffle"),
    (128, 256, 32, 8, 1, 16, "wavelet", "default"),
    (64, 32, 64, 4, 2, 16, "compv4", "cshuffle"),
    (64, 128, 32, 2, 1, 16, "mem", "default"),
    (128, 128, 16, 4, 1, 32, "mem", "cshuffle"),
    (16, 256, 16, 1, 1, 16, "compv4", "default"),
    (32, 64, 16, 2, 2, 16, "compv3", "cshuffle"),
    (32, 64, 16, 1, 2, 32, "wavelet", "default"),
    (64, 256, 64, 2, 8, 32, "mem", "default"),
    (128, 64, 64, 1, 2, 16, "compv4", "default"),
    (128, 128, 64, 8, 8, 16, "wavelet", "default"),
    (256, 16, 16, 2, 1, 16, "mem", "cshuffle"),
    (32, 64, 16, 2, 1, 16, "basic", "default"),
    (256, 256, 16, 4, 2, 16, "mem", "cshuffle"),
    (256, 128, 32, 4, 1, 32, "compv4", "default"),
    (256, 128, 16, 1, 4, 32, "compv4", "default"),
    (256, 16, 64, 4, 1, 16, "compv3", "default"),
    (256, 256, 64, 8, 1, 32, "compv3", "cshuffle"),
    (64, 256, 16, 2, 2, 16, "mem", "cshuffle"),
    (64, 256, 64, 4, 4, 16, "compv4", "cshuffle"),
    (256, 32, 16, 2, 1, 16, "compv3", "cshuffle"),
    (128, 128, 32, 2, 2, 32, "basic", "default"),
    (16, 64, 32, 1, 4, 16, "compv4", "cshuffle"),
    (64, 64, 64, 4, 2, 16, "basic", "default"),
    (64, 128, 32, 1, 1, 32, "compv4", "default"),
    (128, 128, 16, 1, 2, 32, "wavelet", "default"),
    (128, 256, 32, 1, 2, 16, "compv3", "default"),
    (128, 256, 32, 2, 1, 16, "basic", "cshuffle"),
    (256, 16, 16, 2, 1, 16, "wavelet", "default"),
    (32, 256, 64, 2, 2, 16, "compv4", "cshuffle"),
    (128, 256, 32, 2, 2, 16, "mem", "cshuffle"),
    (32, 32, 64, 1, 1, 16, "compv4", "default"),
    (32, 64, 16, 1, 4, 16, "compv4", "cshuffle"),
    (128, 128, 16, 1, 1, 16, "wavelet", "cshuffle"),
    (64, 128, 64, 1, 4, 16, "compv4", "default"),
    (64, 128, 16, 4, 4, 16, "basic", "default"),
    (128, 128, 32, 2, 1, 32, "compv3", "cshuffle"),
    (128, 256, 64, 4, 2, 32, "compv3", "cshuffle"),
    (256, 32, 64, 8, 1, 32, "basic", "default"),
    (64, 64, 32, 4, 1, 16, "compv4", "default"),
    (128, 128, 32, 4, 1, 16, "wavelet", "cshuffle"),
    (64, 32, 64, 4, 2, 16, "compv3", "default"),
    (256, 128, 64, 4, 4, 32, "compv3", "default"),
    (128, 64, 16, 2, 1, 16, "wavelet", "cshuffle"),
    (32, 64, 64, 1, 1, 16, "mem", "default"),
    (128, 128, 64, 2, 8, 16, "compv4", "default"),
    (64, 256, 16, 2, 4, 16, "compv3", "default"),
    (256, 256, 64, 1, 1, 16, "basic", "cshuffle"),
    (32, 128, 64, 2, 1, 16, "mem", "cshuffle"),
    (128, 128, 64, 1, 4, 16, "basic", "default"),
    (256, 64, 16, 4, 2, 16, "compv4", "cshuffle"),
    (128, 256, 16, 4, 8, 16, "compv3", "default"),
    (128, 128, 32, 8, 8, 16, "compv3", "default"),
    (256, 64, 32, 4, 1, 32, "compv4", "default"),
    (64, 32, 32, 2, 2, 16, "mem", "default"),
    (64, 16, 64, 1, 1, 16, "basic", "default"),
    (256, 128, 64, 8, 2, 32, "compv3", "cshuffle"),
    (256, 128, 64, 2, 4, 32, "wavelet", "default"),
    (128, 256, 32, 2, 1, 16, "compv3", "default"),
    (128, 256, 32, 4, 2, 32, "mem", "default"),
    (64, 64, 32, 1, 1, 32, "wavelet", "default"),
    (256, 128, 64, 4, 8, 16, "basic", "cshuffle"),
    (64, 64, 16, 1, 1, 32, "mem", "default"),
    (256, 256, 16, 4, 8, 32, "compv4", "default"),
    (256, 64, 64, 4, 1, 16, "basic", "default"),
    (32, 32, 64, 2, 1, 16, "basic", "cshuffle"),
    (128, 128, 64, 1, 2, 32, "wavelet", "default"),
    (128, 256, 32, 1, 4, 16, "basic", "cshuffle"),
    (64, 128, 32, 2, 2, 16, "mem", "default"),
    (64, 128, 64, 2, 1, 32, "basic", "default"),
    (32, 256, 32, 2, 4, 16, "compv4", "default"),
    (32, 128, 32, 2, 4, 16, "basic", "default"),
    (32, 256, 32, 1, 8, 16, "wavelet", "cshuffle"),
    (64, 256, 64, 1, 2, 16, "compv4", "default"),
    (128, 32, 16, 4, 2, 16, "mem", "cshuffle"),
    (32, 128, 32, 1, 2, 16, "mem", "cshuffle"),
    (256, 32, 32, 4, 1, 16, "compv4", "default"),
    (16, 64, 32, 1, 1, 16, "basic", "default"),
    (64, 16, 32, 2, 1, 16, "wavelet", "default"),
    (256, 32, 64, 8, 1, 16, "compv3", "default"),
    (256, 256, 32, 8, 2, 16, "compv3", "cshuffle"),
    (128, 128, 32, 2, 1, 32, "basic", "default"),
    (256, 128, 16, 8, 4, 32, "basic", "default"),
    (256, 256, 16, 2, 1, 16, "compv4", "cshuffle"),
    (128, 32, 16, 1, 1, 32, "basic", "cshuffle"),
    (256, 64, 64, 2, 1, 16, "mem", "cshuffle"),
    (32, 256, 32, 1, 4, 32, "mem", "default"),
    (256, 16, 16, 8, 1, 16, "compv4", "default"),
    (128, 32, 16, 4, 1, 16, "wavelet", "default"),
    (128, 64, 64, 1, 4, 16, "compv4", "default"),
    (64, 128, 64, 2, 1, 32, "basic", "cshuffle"),
    (256, 64, 64, 8, 2, 32, "compv4", "cshuffle"),
    (64, 256, 32, 2, 4, 16, "compv4", "default"),
    (128, 256, 64, 2, 8, 32, "compv3", "cshuffle"),
    (64, 32, 64, 4, 2, 16, "compv4", "default"),
    (32, 16, 32, 2, 1, 16, "compv3", "default"),
    (256, 256, 64, 1, 4, 32, "compv4", "default"),
    (256, 128, 32, 1, 2, 32, "basic", "cshuffle"),
    (64, 256, 64, 2, 1, 16, "compv4", "default"),
    (256, 256, 16, 4, 8, 16, "compv4", "default"),
    (256, 64, 16, 1, 4, 16, "mem", "default"),
    (64, 256, 64, 4, 2, 16, "compv4", "cshuffle"),
    (32, 64, 16, 2, 1, 16, "basic", "cshuffle"),
    (128, 16, 64, 8, 1, 16, "wavelet", "cshuffle"),
    (64, 32, 16, 4, 1, 16, "compv4", "default"),
    (64, 256, 64, 1, 2, 16, "compv3", "cshuffle"),
    (256, 128, 32, 4, 1, 32, "mem", "default"),
    (256, 32, 16, 4, 1, 32, "compv3", "cshuffle"),
    (128, 256, 64, 4, 4, 16, "compv4", "cshuffle"),
    (64, 128, 32, 1, 1, 32, "basic", "cshuffle"),
    (64, 32, 16, 4, 1, 16, "wavelet", "default"),
    (256, 256, 16, 1, 8, 32, "compv3", "cshuffle"),
    (64, 128, 16, 4, 2, 16, "mem", "cshuffle"),
    (256, 64, 16, 8, 2, 16, "mem", "default"),
    (256, 64, 64, 2, 1, 16, "compv4", "default"),
    (256, 256, 32, 8, 4, 32, "basic", "cshuffle"),
    (256, 256, 16, 4, 2, 32, "wavelet", "default"),
    (128, 256, 16, 2, 4, 16, "compv3", "cshuffle"),
    (256, 256, 16, 8, 1, 16, "wavelet", "default"),
    (256, 256, 16, 8, 1, 16, "basic", "cshuffle"),
    (64, 256, 64, 2, 1, 16, "basic", "cshuffle"),
    (16, 64, 64, 1, 1, 16, "basic", "cshuffle"),
    (128, 256, 32, 2, 4, 32, "compv4", "default"),
    (128, 256, 16, 1, 4, 32, "compv4", "default"),
    (128, 256, 16, 2, 2, 32, "compv3", "default"),
    (32, 32, 64, 1, 1, 16, "mem", "cshuffle"),
    (256, 256, 64, 1, 4, 16, "compv3", "cshuffle"),
    (64, 128, 16, 4, 2, 16, "mem", "default"),
    (128, 64, 32, 8, 1, 16, "basic", "default"),
    (64, 128, 32, 2, 2, 16, "compv4", "default"),
    (256, 32, 16, 4, 2, 16, "basic", "cshuffle"),
    (128, 128, 32, 4, 2, 16, "compv3", "cshuffle"),
    (256, 128, 64, 1, 8, 16, "wavelet", "cshuffle"),
    (32, 128, 32, 2, 8, 16, "wavelet", "cshuffle"),
    (32, 256, 32, 1, 8, 16, "wavelet", "default"),
    (256, 256, 32, 4, 8, 16, "compv3", "default"),
    (128, 32, 32, 2, 1, 16, "wavelet", "cshuffle"),
    (32, 64, 16, 1, 2, 32, "compv4", "cshuffle"),
    (128, 128, 64, 1, 1, 32, "compv3", "default"),
    (16, 128, 32, 1, 8, 16, "mem", "default"),
    (256, 128, 16, 8, 4, 32, "basic", "cshuffle"),
    (128, 256, 16, 8, 2, 16, "wavelet", "default"),
    (64, 128, 32, 4, 8, 16, "compv3", "default"),
    (64, 128, 16, 4, 2, 16, "basic", "default"),
    (256, 128, 32, 4, 4, 16, "basic", "default"),
    (256, 16, 64, 1, 1, 16, "compv3", "cshuffle"),
    (256, 256, 32, 1, 2, 32, "compv3", "cshuffle"),
    (256, 32, 32, 1, 2, 16, "compv3", "default"),
    (256, 256, 64, 2, 4, 16, "mem", "default"),
    (256, 256, 16, 8, 8, 32, "mem", "cshuffle"),
    (32, 32, 32, 1, 2, 16, "mem", "default"),
    (32, 256, 64, 1, 4, 16, "compv4", "cshuffle"),
    (256, 256, 32, 8, 1, 16, "wavelet", "default"),
    (32, 256, 16, 1, 4, 32, "mem", "cshuffle"),
    (256, 64, 64, 4, 1, 16, "compv4", "cshuffle"),
    (64, 256, 64, 2, 8, 32, "wavelet", "default"),
    (128, 128, 32, 2, 1, 32, "compv4", "cshuffle"),
    (256, 256, 64, 4, 4, 32, "compv3", "default"),
    (256, 64, 32, 8, 1, 32, "compv4", "cshuffle"),
    (256, 128, 64, 8, 4, 32, "mem", "default"),
    (256, 32, 64, 1, 1, 16, "wavelet", "default"),
    (128, 32, 64, 4, 1, 16, "wavelet", "default"),
    (16, 256, 16, 1, 8, 16, "compv4", "cshuffle"),
    (16, 16, 16, 1, 1, 16, "compv3", "cshuffle"),
    (64, 256, 16, 4, 1, 16, "mem", "cshuffle"),
    (128, 128, 64, 1, 4, 16, "compv4", "cshuffle"),
    (256, 256, 32, 1, 8, 16, "compv4", "default"),
    (128, 64, 32, 4, 1, 32, "basic", "cshuffle"),
    (128, 256, 32, 1, 1, 32, "mem", "default"),
    (32, 128, 32, 2, 4, 16, "compv4", "cshuffle"),
    (256, 256, 16, 4, 1, 16, "compv4", "cshuffle"),
    (128, 64, 64, 2, 1, 32, "compv3", "default"),
    (128, 128, 64, 1, 4, 32, "mem", "cshuffle"),
    (256, 64, 32, 1, 1, 16, "compv4", "cshuffle"),
    (256, 32, 32, 2, 2, 16, "compv4", "default"),
    (256, 32, 32, 2, 1, 32, "wavelet", "default"),
    (64, 128, 64, 1, 4, 16, "compv3", "cshuffle"),
    (64, 64, 32, 2, 1, 32, "wavelet", "default"),
    (128, 64, 16, 1, 4, 16, "compv4", "default"),
    (128, 128, 32, 1, 4, 16, "compv3", "cshuffle"),
    (128, 256, 32, 1, 2, 32, "compv3", "default"),
    (32, 128, 16, 1, 2, 32, "basic", "cshuffle"),
    (32, 32, 64, 1, 1, 16, "basic", "default"),
    (256, 128, 64, 8, 2, 32, "wavelet", "default"),
    (256, 64, 32, 8, 1, 32, "wavelet", "default"),
    (64, 128, 16, 1, 2, 16, "basic", "default"),
    (256, 256, 16, 2, 4, 32, "basic", "default"),
    (32, 256, 32, 1, 1, 32, "wavelet", "cshuffle"),
    (64, 128, 32, 1, 2, 16, "mem", "default"),
    (128, 256, 64, 1, 4, 32, "basic", "cshuffle"),
    (256, 64, 32, 2, 1, 16, "wavelet", "default"),
    (32, 32, 32, 2, 2, 16, "wavelet", "default"),
    (256, 128, 32, 8, 8, 16, "basic", "cshuffle"),
    (256, 32, 32, 8, 1, 16, "compv4", "cshuffle"),
    (256, 128, 32, 4, 1, 32, "basic", "default"),
    (32, 64, 32, 2, 2, 16, "mem", "default"),
    (256, 128, 64, 2, 1, 16, "compv4", "cshuffle"),
    (16, 256, 16, 1, 8, 16, "mem", "cshuffle"),
    (32, 32, 32, 1, 1, 16, "mem", "cshuffle"),
    (64, 64, 64, 1, 2, 16, "compv4", "default"),
    (128, 256, 16, 2, 8, 16, "wavelet", "default"),
    (64, 32, 16, 4, 2, 16, "compv4", "cshuffle"),
    (128, 256, 32, 4, 8, 32, "basic", "cshuffle"),
)


def sweep_sample(split_k: int) -> list[DgradCombo]:
    """The pinned tilings as sweep combos with split-K mode ``split_k`` (1, a
    degree, or -1 for the split-K resolver's pick)."""
    return [DgradCombo(*tiling, split_k) for tiling in SWEEP_SAMPLE_TILINGS]


class ClassSample(NamedTuple):
    """What :func:`first_candidates_per_class` or :func:`every_candidate` found: ``checked``, each chosen
    combo with its candidate, class by class in the sample's draw order;
    ``not_candidates``, how many combos it tried that were no candidate; and
    ``failures``, one line per combo whose candidate raised."""

    checked: list[tuple[DgradCombo, Any]]
    not_candidates: int
    failures: list[str]


def first_candidates_per_class(
    combos: Sequence[DgradCombo],
    per_class: int,
    submit: Callable[[DgradCombo], Future],
) -> ClassSample:
    """The first ``per_class`` candidates of each (pipeline, epilogue) class of
    ``combos``, in their order.

    ``submit(combo)`` starts building ``combo`` and returns a future of its
    candidate, or of None when the combo is no sweep candidate. Each class keeps
    ``per_class`` combos in flight and takes its next combo only when one of
    them is no candidate, so it builds the fewest combos that find its first
    ``per_class`` candidates, and a class with fewer candidates than that gives
    all it has. A combo whose future raises is a failure and takes its class's
    place of a candidate, so the result does not depend on when it raised.
    ``BrokenExecutor`` propagates, after the futures still pending are
    cancelled."""
    if per_class < 1:
        raise ValueError(f"per_class={per_class}, expected at least 1")
    draw_index = {combo: index for index, combo in enumerate(combos)}
    queues: dict[tuple[str, str], deque[DgradCombo]] = {}
    for combo in combos:
        queues.setdefault((combo.pipeline, combo.epilogue), deque()).append(combo)
    found: list[tuple[DgradCombo, Any]] = []
    # The combos each class has settled: a candidate, or a build that raised.
    settled_per_class = dict.fromkeys(queues, 0)
    pending: dict[Future, DgradCombo] = {}

    def top_up(group: tuple[str, str]) -> None:
        in_flight = sum(
            (combo.pipeline, combo.epilogue) == group for combo in pending.values()
        )
        queue = queues[group]
        while queue and settled_per_class[group] + in_flight < per_class:
            combo = queue.popleft()
            pending[submit(combo)] = combo
            in_flight += 1

    for group in queues:
        top_up(group)
    not_candidates, failures = 0, []
    try:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                combo = pending.pop(future)
                group = (combo.pipeline, combo.epilogue)
                try:
                    candidate = future.result()
                except BrokenExecutor:
                    raise
                except Exception as error:  # noqa: BLE001 - reported per combo
                    failures.append(f"{combo}: build/compile raised {error!r}")
                    settled_per_class[group] += 1
                    continue
                if candidate is None:
                    not_candidates += 1
                    top_up(group)
                else:
                    found.append((combo, candidate))
                    settled_per_class[group] += 1
    finally:
        for future in pending:
            future.cancel()
    group_order = {group: index for index, group in enumerate(queues)}
    found.sort(
        key=lambda item: (
            group_order[(item[0].pipeline, item[0].epilogue)],
            draw_index[item[0]],
        )
    )
    return ClassSample(found, not_candidates, failures)


def every_candidate(
    combos: Sequence[DgradCombo], submit: Callable[[DgradCombo], Future]
) -> ClassSample:
    """Every candidate of ``combos``, in their order: ``submit`` as for
    :func:`first_candidates_per_class`, called for every combo at once.
    ``BrokenExecutor`` propagates, after the futures still pending are
    cancelled."""
    futures = [(combo, submit(combo)) for combo in combos]
    found: list[tuple[DgradCombo, Any]] = []
    not_candidates, failures = 0, []
    try:
        for combo, future in futures:
            try:
                candidate = future.result()
            except BrokenExecutor:
                raise
            except Exception as error:  # noqa: BLE001 - reported per combo
                failures.append(f"{combo}: build/compile raised {error!r}")
                continue
            if candidate is None:
                not_candidates += 1
            else:
                found.append((combo, candidate))
    finally:
        for _, future in futures:
            future.cancel()
    return ClassSample(found, not_candidates, failures)
