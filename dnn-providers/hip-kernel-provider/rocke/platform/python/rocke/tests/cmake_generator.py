# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: the generator of a test's CMake configure, or a skip where the
host has no build program for it.

CI's test container has neither ninja nor make, so a configure there finds no
build program. A test that configures a CMake project takes its generator from
:func:`ninja_generator_or_skip`, which skips it there, waiting for the
build-variant lane. Kept apart from ``rocke.tests.platform_build``, which the
runners and hook scripts import without pytest.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .lanes import BUILD_VARIANT, skip_until
from .platform_build import generator_arguments

__all__ = ["ninja_generator_or_skip"]


def ninja_generator_or_skip(build_root: Path) -> list[str]:
    """The generator argument of a test's configure into the fresh tree
    ``build_root``: ``-GNinja``, the generator a fresh platform tree gets
    (``rocke.tests.platform_build.generator_arguments``). Skips the test,
    waiting for the build-variant lane, where cmake or ninja is not on PATH."""
    generator = generator_arguments(build_root)
    # The tree is fresh, so no generator argument means ninja is not on PATH.
    if shutil.which("cmake") is None or not generator:
        skip_until(
            BUILD_VARIANT,
            reason="needs cmake and ninja on PATH: it configures with Ninja, the "
            "generator a fresh platform tree gets",
        )
    return generator
