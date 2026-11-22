# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Point the ``rocke.assets`` locators at the trees an installed test payload stages.

An installed test tree has no platform root, so the defaults of
``rocke.assets.platform_tools_dir()`` and ``rocke.assets.platform_parity_dir()``
name directories that do not exist there. The install stages what its tests
need under the test root instead::

    <tests root>/arch_domain/        the platform tools the tests run
    <tests root>/instances/parity/   the platform parity-emitter tree

:func:`point_locators_at_staged_payload` sets each locator's override to its
staged directory when that directory exists, and leaves a value the caller
already chose. The platform and library ``conftest.py`` files and the
differential runner call it, so the rule has one home. In a checkout the tests
root has no ``arch_domain`` and its ``instances/parity`` is the directory the
default already names, so the call changes nothing there.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

__all__ = ["point_locators_at_staged_payload"]

# Override variable of each locator -> its staged directory under the tests root.
_STAGED_LOCATORS = {
    "ROCKE_PLATFORM_TOOLS_DIR": Path("arch_domain"),
    "ROCKE_PLATFORM_PARITY_DIR": Path("instances") / "parity",
}


def point_locators_at_staged_payload(
    tests_root: Path, environ: MutableMapping[str, str] | None = None
) -> None:
    """Set each locator override in ``environ`` (default ``os.environ``) to its
    staged directory under ``tests_root``, where that directory exists and the
    override is not already set."""
    env = os.environ if environ is None else environ
    for variable, relative in _STAGED_LOCATORS.items():
        staged = tests_root / relative
        if staged.is_dir():
            env.setdefault(variable, str(staged))
