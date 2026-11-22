# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Filesystem locators for non-package rocke assets (data / utility trees).

This module holds the *only* path derivations left in the rocke Python tree.
Everything else resolves through ordinary imports: the ``rocke`` package and the
library packages (``kernels`` / ``builders`` / ``dispatch``) are editable-installed
into the build/dev environment, so no script touches ``sys.path`` to find them
(see rocke/BUILDING.md).

Some assets -- notably ``dsl_docs/``, the ``tools/`` scripts and the loose
``_ua_shape_utils`` benchmark helper -- live under the platform *source* root,
outside the importable ``rocke`` package, so they cannot be reached with
``importlib.resources`` and need an explicit locator.  Each locator honours an environment override so the trees can
be relocated (e.g. consumed out of the source checkout).
"""

import os
from pathlib import Path

# This file: rocke/platform/python/rocke/assets.py
#   parents[0] = rocke   parents[1] = python   parents[2] = platform
_PLATFORM_ROOT = Path(__file__).resolve().parents[2]


def platform_root() -> Path:
    """The ``rocke/platform`` source root (parent of ``python/``, ``dsl_docs/``, ``cpp/``).

    Override with ``ROCKE_PLATFORM_ROOT``.
    """
    override = os.environ.get("ROCKE_PLATFORM_ROOT")
    return Path(override) if override else _PLATFORM_ROOT


def library_root() -> Path:
    """The ``rocke/library`` source root (sibling of ``platform/``).

    For tests/tools that need to hand a subprocess an explicit ``PYTHONPATH``
    (rather than relying on the dev env's editable installs). Override with
    ``ROCKE_LIBRARY_ROOT``.
    """
    override = os.environ.get("ROCKE_LIBRARY_ROOT")
    return Path(override) if override else platform_root().parent / "library"


def dsl_docs_dir() -> Path:
    """The ``dsl_docs`` documentation + utilities tree.

    Override with ``ROCKE_DSL_DOCS``.
    """
    override = os.environ.get("ROCKE_DSL_DOCS")
    return Path(override) if override else platform_root() / "dsl_docs"


def platform_tools_dir() -> Path:
    """The platform ``tools`` tree (the IR-validity and byte-identity scripts).

    Override with ``ROCKE_PLATFORM_TOOLS_DIR``. The installed test payload has
    no ``tools/`` tree; it stages a partial copy, only the scripts its tests
    run, under ``tests/arch_domain``, and
    :func:`rocke.tests.staged_payload.point_locators_at_staged_payload` points
    the override there.
    """
    override = os.environ.get("ROCKE_PLATFORM_TOOLS_DIR")
    return Path(override) if override else platform_root() / "tools"


def platform_instances_dir() -> Path:
    """The platform instance-test tree, home of the representative-IR parity
    harness ``rocke_ir_parity_harness.py`` and of the parity-emitter tree.

    Override with ``ROCKE_PLATFORM_INSTANCES_DIR``. The installed test payload
    has no platform root; it stages the tree at ``tests/instances``, and
    :func:`rocke.tests.staged_payload.point_locators_at_staged_payload` points
    the override there.
    """
    override = os.environ.get("ROCKE_PLATFORM_INSTANCES_DIR")
    return Path(override) if override else platform_root() / "tests" / "instances"


def platform_parity_dir() -> Path:
    """The platform parity-emitter tree, home of the shared emitter driver
    ``_emit_common.py`` that the library emitters re-export.

    Override with ``ROCKE_PLATFORM_PARITY_DIR``; without it, the ``parity``
    directory of :func:`platform_instances_dir`. The installed test payload
    stages the tree at ``tests/instances/parity``, and
    :func:`rocke.tests.staged_payload.point_locators_at_staged_payload` points
    the override there.
    """
    override = os.environ.get("ROCKE_PLATFORM_PARITY_DIR")
    return Path(override) if override else platform_instances_dir() / "parity"


def shape_utils_dir() -> Path:
    """Directory containing the loose ``_ua_shape_utils`` benchmark helper module."""
    return dsl_docs_dir() / "optimization" / "utilities" / "tools" / "stage1_benchmark"
