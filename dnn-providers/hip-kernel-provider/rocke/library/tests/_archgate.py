# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free device-arch gate for the library test tree.

Every existing numeric test opens with its own ``_gpu_ready()`` that imports
torch and reads ``torch.cuda.get_device_properties(0).gcnArchName``. That makes
the gate itself torch-dependent: on a CI node without torch the test does not
merely skip for lack of a GPU, it skips for lack of the *oracle*, and the skip
reason misreports why. This module gates on the arch alone, via
:func:`rocke.runtime.hip_module.get_device_arch`, which talks to HIP through
ctypes and never touches torch -- so a numpy-only replay test can run on a node
that has a GPU and no torch at all.

Two rules the per-file gates got wrong often enough to be worth stating:

* Compare with ``==``, never a substring. ``"gfx94" in arch`` matches gfx941 and
  gfx942 alike, and ``"gfx1250" in arch`` is fine only until a gfx12500 exists.
  ``get_device_arch`` already strips target-id features (``gfx942:sramecc+:xnack-``
  -> ``gfx942``) via ``base_arch_from_target_id``, so the value is directly
  comparable.
* Never gate on the marketing device name. The whole MI300 family is gfx942 but
  the name varies (``MI300X`` / ``MI300A`` / ``MI308X``); a substring check for
  ``"mi300"`` silently misses ``MI308X``, which is exactly the skip that hid the
  dense gfx942 lane on its first run.

The probe is cached: a skipif decorator is evaluated at import time for every
parametrized row, and re-opening the HIP runtime once per row is pure overhead.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

__all__ = [
    "device_arch",
    "requires_arch",
    "has_arch",
    "requires_gfx942_gpu",
    "requires_gfx950_gpu",
]


@lru_cache(maxsize=1)
def device_arch() -> str | None:
    """Base gfx arch of device 0, or ``None`` when no GPU is visible.

    Any failure -- no HIP runtime, no device, ``HIP_VISIBLE_DEVICES=``, an import
    error in the runtime package -- collapses to ``None`` rather than raising.
    A gate that raises during collection turns "no GPU here" into a tree-wide
    error, which is the opposite of what a gate is for.
    """
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0)
    except Exception:  # noqa: BLE001 - see docstring: never raise from a gate
        return None


def has_arch(arch: str) -> bool:
    """True when device 0 is exactly ``arch``."""
    return device_arch() == arch


def requires_arch(arch: str):
    """``pytest.mark.skipif`` that runs the test only on exactly ``arch``.

    Usage (``tests`` is a package -- a bare ``import _archgate`` does not
    resolve; either import relatively or via the ``tests`` package)::

        from ._archgate import requires_arch  # or: from tests._archgate import requires_arch

        @requires_arch("gfx942")
        @pytest.mark.gpu
        def test_something(): ...
    """
    return pytest.mark.skipif(
        not has_arch(arch),
        reason=f"needs a {arch} GPU (device 0 is {device_arch() or 'absent'})",
    )


# Convenience aliases for the two archs every hand-rolled gate in this tree
# actually asked for. Defined here (not per-file) so a rename or a widened
# gfx942 target-id set only needs one edit.
requires_gfx942_gpu = requires_arch("gfx942")
requires_gfx950_gpu = requires_arch("gfx950")
