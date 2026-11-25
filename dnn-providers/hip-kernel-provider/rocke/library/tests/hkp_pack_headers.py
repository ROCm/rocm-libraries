# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Locate hip-kernel-provider's ingestor-engine pack headers for the host-side
tests that read them (geometry constants, LDS formulas).

An installed test tree carries a copy of the two headers under ``hkp_packs/``
beside this module (the rocKE umbrella ``CMakeLists.txt`` installs it); a
source checkout reads the provider's own copy. The tests that read them skip
when neither exists, unless ``--rocke-require-hkp-pack-headers`` is given
(the installed CTest entry passes it whenever it installed the copy, and so
does ``run_all.py``, which forwards it on its own detection of the same
copy, the same way it already does for ``sdpa.hipdnn_assets``).
"""

from __future__ import annotations

from pathlib import Path

# Both headers must be present to count as a usable copy, installed or not.
GEOMETRY_HEADERS = (
    "Gfx950AttentionDenseGeometry.hpp",
    "Gfx950AttentionUnified3DGeometry.hpp",
)
# Named like hipdnn_assets.py's ``_INSTALLED``: the installed copy's root.
INSTALLED = Path(__file__).resolve().parent / "hkp_packs"
_CHECKOUT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "packs"
)
DIRS = (INSTALLED, _CHECKOUT)


def installed_copy_present() -> bool:
    """Whether this is an installed tree carrying the header copy (``hkp_packs/``)."""
    return INSTALLED.is_dir()


def find() -> Path | None:
    """The directory holding both geometry headers (installed copy first), or
    ``None`` if neither has them."""
    for d in DIRS:
        if all((d / h).is_file() for h in GEOMETRY_HEADERS):
            return d
    return None
