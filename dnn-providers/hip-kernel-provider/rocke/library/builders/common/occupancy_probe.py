# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared static occupancy probe for the attention parity/bench harnesses.

Wraps the DSL static occupancy probe (``dsl_probes/probe_occupancy.py``,
``llvm-readelf`` on the HSACO notes -- no kernel launch, safe during the build
step) so the per-arch harnesses consume one implementation instead of drifting
into near-identical copies.
"""
from __future__ import annotations


def print_occupancy(hsaco: bytes, num_warps: int, *, arch: str = "gfx950") -> None:
    """Print VGPR/AGPR/SGPR/LDS + coarse waves-per-CU for a compiled kernel.

    Static (reads the HSACO notes); no launch, so it is safe to call during the
    build step. ``arch`` selects the occupancy model (``gfx950`` / ``gfx942`` /
    ``gfx90a``); unknown values fall back to gfx950.
    """
    try:
        import sys as _sys

        from rocke.assets import dsl_docs_dir

        probe_dir = (
            dsl_docs_dir() / "optimization" / "utilities" / "tools" / "dsl_probes"
        )
        if str(probe_dir) not in _sys.path:
            _sys.path.insert(0, str(probe_dir))
        import probe_occupancy as _po  # type: ignore

        arch_caps = getattr(_po, f"ARCH_{arch.upper()}", _po.ARCH_GFX950)
        notes = _po.parse_hsaco_notes(hsaco)
        occ = _po.estimate_occupancy(
            notes=notes, waves_per_wg=num_warps, arch=arch_caps
        )
        print(
            f"    [occupancy] vgpr={notes.get('vgpr_count')} "
            f"agpr={notes.get('agpr_count')} sgpr={notes.get('sgpr_count')} "
            f"spill={notes.get('vgpr_spill_count')} "
            f"lds={notes.get('lds_size')}B  "
            f"waves/CU={occ['waves_per_cu']} wg/CU={occ['wgs_per_cu']} "
            f"limit={occ['limited_by']}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"    [occupancy] unavailable: {type(e).__name__}: {e}")
