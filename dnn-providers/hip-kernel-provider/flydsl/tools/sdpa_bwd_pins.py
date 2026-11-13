# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA backward pins whose object the shipped content does not hold, and their function digests.

``tests/data/sdpa_bwd_reference.csv`` (and the gfx11 one) names every pin by its Spec hash and whole-object sha256. A pin's object
ships as it is, except two kinds the content never holds:

* the main kernel of a combined object (a dB-storing ``bias`` row of the merged or dQ op): the shipped object also carries the dBias
  reduce kernel, so it is not the pin;
* a standalone reduce kernel (``dbias_reduce``, ``dsink_reduce``) of gfx950, whose objects ride inside the dQ and merged ones.

``tests/data/sdpa_bwd_pin_functions.csv`` holds, per such pin, the sha256 of the pin's kernel function bytes. The per-kernel gate
(``sdpa_bwd_gate.py``) and ``sdpa_bwd_compare.py`` read it when the reference directory has no object for the pin.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

PIN_FUNCTIONS_CSV = FLYDSL / "tests" / "data" / "sdpa_bwd_pin_functions.csv"
PIN_FUNCTION_COLUMNS = ("hash", "spec", "function_sha256")
REDUCE_KERNELS = ("dbias_reduce", "dsink_reduce")


def holds_no_shipped_object(spec) -> bool:
    """Whether the pin of @p spec (a ``Spec``) is an object the shipped content does not hold."""
    from generators._sdpa_bwd_cdna_build import reduce_symbol_of

    if spec.arch != "gfx950":
        return False
    return bool(reduce_symbol_of(spec)) or spec.kernel in REDUCE_KERNELS


def function_sha256(blob: bytes) -> str:
    """The sha256 of the one kernel's function bytes of the pin object @p blob."""
    (symbol,) = analyze.kernels(analyze.amdgpu_metadata(blob))
    return hashlib.sha256(analyze.function_bytes(blob, symbol)).hexdigest()


def read_pin_functions(path: Path | None = None) -> dict[str, dict]:
    """The rows of @p path (default ``PIN_FUNCTIONS_CSV``) by Spec hash; none when the file is absent."""
    path = path or PIN_FUNCTIONS_CSV
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["hash"]: row for row in csv.DictReader(handle)}
