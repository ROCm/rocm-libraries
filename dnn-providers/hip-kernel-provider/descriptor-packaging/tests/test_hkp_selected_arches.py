"""hkp_selected_arches: which GPU_TARGETS entries name a packable concrete target.

Runs the real function from HkpPackaging.cmake under ``cmake -P``, so the regex, the
operator precedence and the warning are exercised as CMake evaluates them. The only
spelling this must keep apart is a concrete target (gfx1250-strict) versus the family and
generic names that merely look like one (gfx950-dcgpu, gfx11-generic): the first becomes a
shard and reaches ``hipcc --offload-arch``, the second must be dropped with a warning.
"""

import subprocess
from pathlib import Path

import pytest

_CMAKE_DIR = Path(__file__).resolve().parents[1] / "cmake"
_PROVIDER_CMAKE_DIR = _CMAKE_DIR.parents[1] / "cmake"

pytestmark = pytest.mark.quick


def _select(cmake, tmp_path, targets, variable="GPU_TARGETS"):
    """Return (selected arches, ignored entries, source variable) for a target list."""
    script = tmp_path / "select.cmake"
    script.write_text(
        f'list(APPEND CMAKE_MODULE_PATH "{_CMAKE_DIR.as_posix()}" '
        f'"{_PROVIDER_CMAKE_DIR.as_posix()}")\n'
        "include(HkpPackaging)\n"
        f'set({variable} "{targets}")\n'
        "hkp_selected_arches(_arches _source)\n"
        'message(STATUS "SELECTED=[${_arches}] SOURCE=[${_source}]")\n'
    )
    result = subprocess.run(
        [cmake, "-P", str(script)], capture_output=True, text=True, check=True
    )
    out = result.stdout + result.stderr
    selected = next(line for line in out.splitlines() if "SELECTED=[" in line).split(
        "SELECTED=["
    )[1]
    arches, _, rest = selected.partition("] SOURCE=[")
    ignored = [
        line.split("'")[1] for line in out.splitlines() if "hkp: ignoring" in line
    ]
    return (arches.split(";") if arches else []), ignored, rest.rstrip("]")


CONCRETE = [
    "gfx900",
    "gfx906",
    "gfx908",
    "gfx90a",
    "gfx90c",
    "gfx942",
    "gfx950",
    "gfx1030",
    "gfx1100",
    "gfx1151",
    "gfx1200",
    "gfx1250",
    "gfx1250-strict",
    # A word that merely contains a family word is not a family word.
    "gfx1250-strictall",
    "gfx942-allx",
]

# TheRock family names (including variant spellings whose family word is not last) and
# generic targets: none names one processor.
NOT_CONCRETE = [
    "gfx94X-dcgpu",
    "gfx125X-dcgpu",
    "gfx950-dcgpu",
    "gfx90a-dcgpu",
    "gfx906-dgpu",
    "gfx900-dgpu",
    "gfx90c-igpu",
    "gfx908-dcgpu",
    "gfx950-all",
    "gfx950-dcgpu-asan",
    "gfx950-dcgpu-tests",
    "gfx90a-dcgpu-asan",
    "gfx11-generic",
    "gfx9-4-generic",
    "gfx12-5-generic",
    "gfxalpha-all",
    # Shapes that are not target names at all.
    "native",
    "GFX942",
    "gfx",
    "gfx1250-Strict",
    "gfx1250--strict",
    "gfx1250-",
    "gfx942-4",
    "gfx1250-all-strict",
    "gfxhkpcensuscontrol",
]


@pytest.mark.parametrize("target", CONCRETE)
def test_concrete_target_is_selected_whole(cmake, tmp_path, target):
    assert _select(cmake, tmp_path, target)[:2] == ([target], [])


@pytest.mark.parametrize("target", NOT_CONCRETE)
def test_non_concrete_name_is_dropped_with_a_warning(cmake, tmp_path, target):
    assert _select(cmake, tmp_path, target)[:2] == ([], [target])


def test_strict_and_base_arch_stay_separate_entries(cmake, tmp_path):
    assert _select(cmake, tmp_path, "gfx1250;gfx1250-strict")[0] == [
        "gfx1250",
        "gfx1250-strict",
    ]


def test_feature_suffix_is_stripped_before_the_check(cmake, tmp_path):
    arches, ignored, _ = _select(
        cmake, tmp_path, "gfx942:xnack-;gfx1250-strict:sramecc+;gfx950-dcgpu:xnack-"
    )
    assert arches == ["gfx942", "gfx1250-strict"]
    assert ignored == ["gfx950-dcgpu:xnack-"]


def test_mixed_list_keeps_only_concrete_targets(cmake, tmp_path):
    arches, ignored, _ = _select(
        cmake, tmp_path, "gfx94X-dcgpu;gfx11-generic;gfx950;gfx1250-strict"
    )
    assert arches == ["gfx950", "gfx1250-strict"]
    assert ignored == ["gfx94X-dcgpu", "gfx11-generic"]


def test_duplicates_collapse(cmake, tmp_path):
    assert _select(cmake, tmp_path, "gfx942;gfx942:xnack-;gfx942")[0] == ["gfx942"]


def test_amdgpu_targets_is_the_fallback_source(cmake, tmp_path):
    arches, _, source = _select(
        cmake, tmp_path, "gfx1250-strict", variable="AMDGPU_TARGETS"
    )
    assert (arches, source) == (["gfx1250-strict"], "AMDGPU_TARGETS")
