# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import functools
import os


@functools.lru_cache(None)
def library_path():
    """Return the path to the bundled CK C++ library tree.

    Two layouts are supported:

    1. Pip-installed package layout (``pyproject.toml`` maps
       ``ck4inductor.library`` to ``library/``): the library lives at
       ``ck4inductor/library/``, i.e. *inside* the installed package
       directory, so ``<pkg>/library/src/...`` works directly.

    2. In-tree development layout (this repository checkout):
       ``ck4inductor/`` lives at ``projects/composablekernel/python/
       ck4inductor`` while the library lives at
       ``projects/composablekernel/library``. Going two levels up
       from this file and into ``library/`` resolves it.

    Pick the first one that exists. This keeps the
    pip-installed user path zero-config while also letting
    ``python -m pytest test/test_gen_instances.py`` pass against an
    uninstalled source checkout.
    """
    pkg_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(pkg_dir, "library"),
        os.path.normpath(os.path.join(pkg_dir, "..", "..", "library")),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    # Fall back to the pip-install path; callers will log a useful
    # error explaining which subdirectory they wanted.
    return candidates[0]
