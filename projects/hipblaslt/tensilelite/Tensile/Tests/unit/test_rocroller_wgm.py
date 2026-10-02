# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocRoller workgroup-mapping kernargs match libdivide branchfree magic.

The frozen table was captured from libdivide 5.4.0 (the header hipBLASLt
fetches for rocRoller) plus the divisor 0 and 1 special cases in
rocRoller Expression_evaluate_unary.cpp. Tensile magicNumber() is a
different algorithm and is not what these kernels consume.
"""

import subprocess
from pathlib import Path

import pytest

from Tensile.CustomKernels import _metadataArgToCustomArg
from Tensile.RocRollerWgm import (
    WGM_ROLE_ORDER,
    classify_wgm_expression,
    evaluate_wgm,
    magic_div_s32,
    magic_multiple_s32,
    magic_multiple_s64,
    magic_multiple_u32,
    magic_shift_and_sign_s32,
    magic_shift_and_sign_s64,
    magic_shifts_u32,
    s32,
)

pytestmark = pytest.mark.unit

# (kind, divisor, magic, second). For u32, second is MagicShifts printed as a
# signed int, which is how rocRoller returns `1 << 31` for divisor 1.
# For s32/s64, second is MagicShiftAndSign (the zero-extended `more` byte).
_LIBDIVIDE = [
    ("s32", 0, 1073741823, 0),
    ("s32", 1, 0, 0),
    ("s32", -1, 0, 128),
    ("s32", 2, 0, 1),
    ("s32", 3, -1431655765, 65),
    ("s32", 5, -858993459, 66),
    ("s32", 6, -1431655765, 66),
    ("s32", 7, -1840700269, 66),
    ("s32", -5, -858993459, 194),
    ("s32", 15, -2004318071, 67),
    ("s32", 16, 0, 4),
    ("s32", 17, -252645135, 68),
    ("s32", 31, -2078209981, 68),
    ("s32", 32, 0, 5),
    ("s32", 100, -1546188226, 70),
    ("s32", 127, -2130574327, 70),
    ("s32", 128, 0, 7),
    ("s32", 255, -2139062143, 71),
    ("s32", 256, 0, 8),
    ("s32", 1000, -2095944040, 73),
    ("s32", 1024, 0, 10),
    ("s32", 2147483647, -2147483646, 94),
    ("s32", -2147483648, 0, 159),
    ("s32", -2147483647, -2147483646, 222),
    ("s32", 9, -477218588, 67),
    ("s32", 10, -858993459, 67),
    ("u32", 0, 2147483647, 0),
    ("u32", 1, 0, -2147483648),
    ("u32", 2, 0, 0),
    ("u32", 3, 1431655766, 1),
    ("u32", 4, 0, 1),
    ("u32", 5, 2576980378, 2),
    ("u32", 6, 1431655766, 2),
    ("u32", 7, 613566757, 2),
    ("u32", 16, 0, 3),
    ("u32", 17, 3789677026, 4),
    ("u32", 100, 1202590843, 6),
    ("u32", 2147483648, 0, 30),
    ("u32", 4294967295, 2, 31),
    ("s64", 0, 4611686018427387903, 0),
    ("s64", 1, 0, 0),
    ("s64", -1, 0, 128),
    ("s64", 2, 0, 1),
    ("s64", 3, -6148914691236517205, 65),
    ("s64", 7, -7905747460161236406, 66),
    ("s64", 16, 0, 4),
    ("s64", 100, -6640827866535438581, 70),
    ("s64", 4294967296, 0, 32),
    ("s64", 4294967297, -4294967295, 96),
    ("s64", -5, -3689348814741910323, 194),
    ("s64", -9223372036854775808, 0, 191),
    ("s64", 9223372036854775807, -9223372036854775806, 126),
]

# signed_magic_div results, including divisor 0 (not a truncating division).
_DIV = [
    (0, 0, 0),
    (1, 0, 1),
    (1, 1, 1),
    (5, 3, 1),
    (32, 2, 16),
    (-20, 6, -3),
    (7, 0, 8),
]

# evaluate_wgm(m, n, tile_m, tile_n, wgm) in WGM_ROLE_ORDER.
_WGM = [
    ((4096, 4096, 128, 256, 2), [16, 0, 0, 2, 1073741823, 1, 0, 4, 0, 9, 16]),
    ((0, 0, 128, 32, 2), [0, 1073741823, 0, 2, 1073741823, 1, 1073741823, 0, 0, 0, 0]),
    ((1, 1, 16, 256, 1), [1, 0, 0, 1, 1073741823, 0, 0, 0, 0, 0, 1]),
    ((2147483647, 1048576, 32, 64, 2), [0, 1073741823, 0, 2, 1073741823, 1, 0, 14, 0, 0, 33554432]),
    ((17, 19, 256, 256, 6), [0, 1073741823, -1431655765, 6, 0, 66, 0, 0, 0, 0, 0]),
    ((5, 0, 64, 64, 2), [0, 1073741823, 0, 2, 0, 1, 1073741823, 0, 0, 0, 0]),
]

_HEADER = Path(__file__).resolve().parents[3] / "include" / "Tensile" / "RocRollerWgmKernargs.hpp"


def _python_magic(kind, divisor):
    if kind == "s32":
        return magic_multiple_s32(divisor), magic_shift_and_sign_s32(divisor)
    if kind == "u32":
        return magic_multiple_u32(divisor), s32(magic_shifts_u32(divisor))
    if kind == "s64":
        return magic_multiple_s64(divisor), magic_shift_and_sign_s64(divisor)
    raise AssertionError(kind)


def test_magic_matches_rocroller_libdivide_edges():
    """Divisor 0 and 1 take rocRoller's special cases, not a raw libdivide call."""
    assert magic_multiple_s32(0) == 1073741823
    assert magic_shift_and_sign_s32(0) == 0
    assert magic_multiple_s32(1) == 0
    assert magic_shift_and_sign_s32(1) == 0
    assert magic_multiple_u32(0) == 2147483647
    assert magic_multiple_u32(1) == 0
    assert magic_shifts_u32(0) == 0
    assert magic_shifts_u32(1) == 1 << 31
    assert magic_multiple_s64(0) == 4611686018427387903
    assert magic_shift_and_sign_s64(0) == 0
    assert magic_multiple_s64(1) == 0
    assert magic_shift_and_sign_s64(1) == 0


@pytest.mark.parametrize("kind,divisor,magic,second", _LIBDIVIDE)
def test_magic_matches_libdivide_branchfree(kind, divisor, magic, second):
    assert _python_magic(kind, divisor) == (magic, second)


@pytest.mark.parametrize("numer,denom,expected", _DIV)
def test_signed_magic_div_matches_rocroller_tree(numer, denom, expected):
    assert magic_div_s32(numer, denom) == expected


@pytest.mark.parametrize("sizes,expected", _WGM)
def test_wgm_kernargs_follow_the_hoisted_formula(sizes, expected):
    got = evaluate_wgm(*sizes)
    assert [got[name] for name in WGM_ROLE_ORDER] == expected


def test_cpp_evaluator_matches_python(tmp_path):
    """The launch header is the same integer arithmetic as the Python oracle."""
    if not _HEADER.is_file():
        pytest.fail(f"missing {_HEADER}")
    source = tmp_path / "wgm_match.cpp"
    source.write_text(
        """
#include <Tensile/RocRollerWgmKernargs.hpp>
#include <cstdint>
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    using namespace TensileLite;
    std::string cmd = argv[1];
    if (cmd == "s32") {
        int32_t d = static_cast<int32_t>(std::stol(argv[2], nullptr, 0));
        std::cout << rocRollerMagicMultipleS32(d) << " " << rocRollerMagicShiftAndSignS32(d) << "\\n";
    } else if (cmd == "u32") {
        uint32_t d = static_cast<uint32_t>(std::stoul(argv[2], nullptr, 0));
        std::cout << rocRollerMagicMultipleU32(d) << " " << static_cast<int32_t>(rocRollerMagicShiftsU32(d)) << "\\n";
    } else if (cmd == "s64") {
        int64_t d = static_cast<int64_t>(std::stoll(argv[2], nullptr, 0));
        std::cout << rocRollerMagicMultipleS64(d) << " " << rocRollerMagicShiftAndSignS64(d) << "\\n";
    } else if (cmd == "wgm") {
        auto k = evaluateRocRollerWgmKernargs(std::stoll(argv[2]), std::stoll(argv[3]),
            static_cast<uint32_t>(std::stoul(argv[4])), static_cast<uint32_t>(std::stoul(argv[5])),
            static_cast<int32_t>(std::stol(argv[6])));
        std::cout << k.quotientTilesByBlock << " " << k.magicMultipleWgmMainBlock << " "
                  << k.magicMultipleWgm << " " << k.workgroupMapping << " "
                  << k.magicMultipleWgmTail << " " << k.magicShiftAndSignWgm << " "
                  << k.magicMultipleNumTilesN << " " << k.magicShiftAndSignNumTilesN << " "
                  << k.magicShiftAndSignWgmTail << " " << k.magicShiftAndSignWgmMainBlock << " "
                  << k.quotientTilesMByWgm << "\\n";
    }
    return 0;
}
"""
    )
    binary = tmp_path / "wgm_match"
    compiled = subprocess.run(
        ["g++", "-O2", "-std=c++17", f"-I{_HEADER.parents[1]}", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr

    def run(args):
        out = subprocess.check_output([str(binary), *args], text=True).split()
        return [int(item) for item in out]

    for kind, divisor, magic, second in _LIBDIVIDE:
        assert run([kind, str(divisor)]) == [magic, second]
    for sizes, expected in _WGM:
        assert run(["wgm", *(str(v) for v in sizes)]) == expected


def _args_section(text):
    import yaml

    lines = []
    capture = False
    for line in text.splitlines(keepends=True):
        if line.startswith("    .args:"):
            capture = True
            lines.append(line)
            continue
        if capture:
            if line.strip() == "..." or (
                line.startswith("    .") and not line.startswith("      ")
            ):
                break
            lines.append(line)
    return yaml.safe_load("".join(lines))[".args"]


def test_shipped_wgm_kernels_classify_to_the_eleven_roles():
    root = Path(__file__).resolve().parents[2] / "CustomKernels" / "rocroller"
    kernels = sorted(root.glob("*_WGM_.s"))
    assert len(kernels) == 20
    for path in kernels:
        text = path.read_text(errors="replace")
        # Flattened workgroup X. These kernels do not read workgroup Y.
        assert "grid: [TilesXY, One, Batch]" in text, path.name
        roles = []
        for arg in _args_section(text):
            mapped = _metadataArgToCustomArg(arg, path.stem)
            if mapped["semantic"] in WGM_ROLE_ORDER:
                roles.append(mapped["semantic"])
        assert tuple(roles) == WGM_ROLE_ORDER, path.name


def test_wgm_expression_classifier_reads_the_tree_not_the_suffix():
    wgm = {
        "type": "CommandArgument",
        "name": "WGM",
        ".size": 4,
        ".value_kind": "by_value",
    }
    assert classify_wgm_expression({"type": "CommandArgument", "name": "WGM"}) == "WorkgroupMapping"
    meta = {
        ".name": "MagicMultiple_22",
        ".size": 4,
        ".value_kind": "by_value",
        ".variableType": {"dataType": "Int32"},
        ".expression": {"type": "MagicMultiple", "arg": wgm},
    }
    assert _metadataArgToCustomArg(meta) == {"type": "int32", "semantic": "MagicMultipleWgm"}
