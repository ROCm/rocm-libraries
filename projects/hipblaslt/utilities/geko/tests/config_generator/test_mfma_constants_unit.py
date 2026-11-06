# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from geko.config_generator import mi_designer as mid
from geko.config_generator.constants import (
    HARDWARE_MAP,
    ONLY_INCLUDE_MIs_GFX942,
    ONLY_INCLUDE_MIs_GFX950,
    validMFMA,
)
from geko.schemas import GemmConfig, GemmType

# TensileLite's makeValidMFMA()["I8I8"] (tensilelite/Common/ValidParameters.py).
_TENSILE_I8I8 = {
    (32, 32, 4, 2),
    (32, 32, 8, 1),
    (16, 16, 4, 4),
    (16, 16, 16, 1),
    (4, 4, 4, 16),
    (32, 32, 16, 1),
    (16, 16, 32, 1),
    (16, 16, 64, 1),
    (32, 32, 32, 1),
}


def _as_set(mis) -> set[tuple[int, ...]]:
    return {tuple(mi) for mi in mis}


@pytest.mark.parametrize("arch", sorted(HARDWARE_MAP))
def test_only_include_mis_are_valid_mfmas(arch: str) -> None:
    """Every allowlisted MI must exist in validMFMA, or generate_all_mfmas drops it silently."""
    for dtype, mis in HARDWARE_MAP[arch]["ONLY_INCLUDE_MIs"].items():
        assert dtype in validMFMA, f"{arch}: no validMFMA entry for {dtype}"
        missing = _as_set(mis) - _as_set(validMFMA[dtype])
        assert not missing, f"{arch}/{dtype}: {sorted(missing)} not in validMFMA[{dtype!r}]"


def test_valid_mfma_i8_matches_tensile() -> None:
    assert _as_set(validMFMA["I8"]) == _TENSILE_I8I8
    assert len(validMFMA["I8"]) == len(_TENSILE_I8I8), "duplicate entries in validMFMA['I8']"


def test_valid_mfma_i8_per_generation_keys() -> None:
    assert _as_set(validMFMA["I8_908"]) <= _as_set(validMFMA["I8"])
    assert _as_set(validMFMA["I8_940"]) <= _as_set(validMFMA["I8"])
    # gfx94x deprecated the K=8 / K=16 legacy int8 MFMAs.
    assert (32, 32, 8, 1) not in _as_set(validMFMA["I8_940"])
    assert (16, 16, 16, 1) not in _as_set(validMFMA["I8_940"])
    # The fp8-only instructions are not int8 MFMAs.
    assert (32, 32, 64, 1) not in _as_set(validMFMA["I8"])
    assert (16, 16, 128, 1) not in _as_set(validMFMA["I8"])


def test_gfx942_i8_allowlist_is_valid_on_gfx94x() -> None:
    assert _as_set(ONLY_INCLUDE_MIs_GFX942["I8"]) <= _as_set(validMFMA["I8_940"])


def test_gfx950_i8_allowlist_has_native_mfmas() -> None:
    allow = _as_set(ONLY_INCLUDE_MIs_GFX950["I8"])
    assert {(16, 16, 64, 1), (32, 32, 32, 1)} <= allow


@pytest.mark.parametrize(
    "arch, expected_native",
    [("gfx950", True), ("gfx942", False)],
)
def test_generate_all_mfmas_i8_native_mfmas_by_arch(tmp_path: Path, arch: str, expected_native: bool) -> None:
    gt = GemmType.from_tensilelite("N", "N", "I8", "I8", "I")
    cfg = {
        "ARCH": arch,
        "CUs": HARDWARE_MAP[arch]["CUs"],
        "MI_FILTER": 1,
        "StreamK": False,
        "GemmProblem": GemmConfig(gt, [[16, 16, 1, 16]]),
    }
    valid, _, _ = mid.MIDesign(tmp_path, cfg).generate_all_mfmas()
    used = {(m.M, m.N, m.K, m.B) for m in valid}
    assert used, "no MFMA generated"
    assert used <= _as_set(HARDWARE_MAP[arch]["ONLY_INCLUDE_MIs"]["I8"])
    assert ((16, 16, 64, 1) in used and (32, 32, 32, 1) in used) == expected_native
