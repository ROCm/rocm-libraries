# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Keep the mutation-harness fixture trees out of pytest collection.

The files under this directory are synthetic *fixture sources* for the
mutation tooling's own tests (e.g. covering-set/src/... is a fake project
tree the harness copies elsewhere and runs mutmut against). They are named
``test_*.py`` and import fixture-only modules such as ``Tensile.Widget`` that
do not exist in the real package, so collecting them as part of the normal
suite raises ImportError. ``testpaths`` in pyproject.toml points pytest at
``Tensile/Tests``, so without this guard every run under that tree would sweep
them up. Ignoring everything beneath this directory keeps them as data.
"""

collect_ignore_glob = ["*"]
