#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx1250 compile-time tile guard in the RowColQuant/TensorQuant ctypes libs.

That block of static_asserts is the only one of the three gfx1250 guards that
survives Tile-Engine deprecation -- the Python validator goes away with
tile_engine, and the runtime tile selector only covers the Python build path --
and until now nothing referenced it. A guard with no test is a guard that gets
deleted during an unrelated refactor and is missed until a device returns wrong
numbers.

No GPU and no ROCm are needed. The guard is pure C++: a constexpr prefix test on
the GFX_ARCH macro and static_asserts over SelectedKernel's compile-time
constants. So this test extracts the block verbatim from the shipped .cpp
(between the BEGIN/END markers), pairs it with a mock SelectedKernel, and asks
the host C++ compiler whether it compiles.

Extraction rather than a transcribed copy is deliberate: a copy passes forever
after the original is deleted. test_guard_block_is_present_in_both_sources fails
if the markers go, and test_guard_is_what_rejects_the_bad_configs fails if the
asserts inside them stop doing anything.

Run: python3 -m pytest tests/test_gfx1250_compile_guard.py -v
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
CTYPES_DIR = SCRIPT_DIR.parent / "bindings" / "ctypes"

SOURCES = {
    "rowcolquant": CTYPES_DIR / "grouped_gemm_rowcolquant_ctypes_lib.cpp",
    "tensorquant": CTYPES_DIR / "grouped_gemm_tensorquant_ctypes_lib.cpp",
}

_BEGIN = "// --- BEGIN gfx1250 compile-time tile guard ---"
_END = "// --- END gfx1250 compile-time tile guard ---"

# Any C++17 host compiler will do; the block contains no HIP.
_CXX = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")


def _extract_guard(path: Path) -> str:
    text = path.read_text()
    start = text.index(_BEGIN) + len(_BEGIN)
    end = text.index(_END)
    return text[start:end]


def _mock_tu(guard: str, gfx_arch: str, kernel: dict) -> str:
    """A translation unit: mock kernel + the real guard, nothing else."""
    return (
        f'#define GFX_ARCH "{gfx_arch}"\n'
        "struct SelectedKernel\n"
        "{\n"
        f"    static constexpr int WarpTileM = {kernel['warp_tile_m']};\n"
        f"    static constexpr int WarpTileN = {kernel['warp_tile_n']};\n"
        f"    static constexpr int WarpTileK = {kernel['warp_tile_k']};\n"
        f"    static constexpr int WarpPerBlock_M = {kernel['warp_m']};\n"
        f"    static constexpr int WarpPerBlock_N = {kernel['warp_n']};\n"
        f"    static constexpr int WarpPerBlock_K = {kernel['warp_k']};\n"
        "};\n"
        f"{guard}\n"
        "int main() { return 0; }\n"
    )


def _cfg(warp_tile, warp_map):
    tm, tn, tk = warp_tile
    wm, wn, wk = warp_map
    return {
        "warp_tile_m": tm,
        "warp_tile_n": tn,
        "warp_tile_k": tk,
        "warp_m": wm,
        "warp_n": wn,
        "warp_k": wk,
    }


# Configurations validated on device for these two bridges on gfx1250.
LEGAL = {
    "default tile 16x16x128, 1x4x1": _cfg((16, 16, 128), (1, 4, 1)),
    "16x16x64, 1x4x1": _cfg((16, 16, 64), (1, 4, 1)),
    "16x16x128, 2x2x1": _cfg((16, 16, 128), (2, 2, 1)),
    "16x16x64, 1x1x1": _cfg((16, 16, 64), (1, 1, 1)),
}

# Each of these compiles cleanly without the guard and is wrong or undependable
# on the device, which is why the guard exists.
ILLEGAL = {
    # No 32x32 WMMA fragment exists on gfx1250. This is the tile behind every
    # wrong-result row of the off-tree sweep.
    "32x32x32 (no 32x32 WMMA)": _cfg((32, 32, 32), (1, 4, 1)),
    # 16x16xK is the right shape but 32 is not one of the two legal K depths.
    "16x16x32 (K not 64 or 128)": _cfg((16, 16, 32), (1, 4, 1)),
    # 8-warp maps: no loadable kernel entry is emitted for them by this codegen.
    "8 warps 2x4x1": _cfg((16, 16, 128), (2, 4, 1)),
    "8 warps 1x8x1": _cfg((16, 16, 128), (1, 8, 1)),
    "8 warps 4x2x1": _cfg((16, 16, 128), (4, 2, 1)),
    "8 warps 8x1x1": _cfg((16, 16, 128), (8, 1, 1)),
    # warp_k > 1: product is 4 so it clears the warp cap, but it is measured to
    # compile and return wrong results. Rejected by its own assert.
    "[1,2,2] (warp_k > 1)": _cfg((16, 16, 128), (1, 2, 2)),
}

# The asserts in the block overlap: 32x32x32 trips the M, N and K rules at once,
# so deleting any one of them still leaves it rejected and a suite built only on
# realistic tiles would not notice. These probes are each caught by exactly one
# assert, which makes every rule in the block individually load-bearing.
SINGLE_RULE_PROBES = {
    "32x16x64 (only the warp_tile_m rule)": _cfg((32, 16, 64), (1, 4, 1)),
    "16x32x64 (only the warp_tile_n rule)": _cfg((16, 32, 64), (1, 4, 1)),
    "16x16x256 (only the warp_tile_k rule)": _cfg((16, 16, 256), (1, 4, 1)),
    "16x16x64 8x1x1 (only the warp-count rule)": _cfg((16, 16, 64), (8, 1, 1)),
    "16x16x64 1x1x2 (only the warp_k rule)": _cfg((16, 16, 64), (1, 1, 2)),
}

ILLEGAL.update(SINGLE_RULE_PROBES)


@unittest.skipIf(_CXX is None, "no host C++ compiler on PATH")
class TestGfx1250CompileGuard(unittest.TestCase):
    def _compiles(self, guard, gfx_arch, kernel):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "guard_probe.cpp"
            src.write_text(_mock_tu(guard, gfx_arch, kernel))
            proc = subprocess.run(
                [_CXX, "-std=c++17", "-fsyntax-only", str(src)],
                capture_output=True,
                text=True,
            )
            return proc.returncode == 0, proc.stderr

    def test_guard_block_is_present_in_both_sources(self):
        for name, path in SOURCES.items():
            with self.subTest(op=name):
                text = path.read_text()
                self.assertIn(_BEGIN, text, f"{path.name} lost its guard markers")
                self.assertIn(_END, text)
                self.assertIn("static_assert", _extract_guard(path))

    def test_legal_gfx1250_configurations_compile(self):
        for op, path in SOURCES.items():
            guard = _extract_guard(path)
            for label, kernel in LEGAL.items():
                with self.subTest(op=op, config=label):
                    ok, err = self._compiles(guard, "gfx1250", kernel)
                    self.assertTrue(ok, f"{label} should compile on gfx1250:\n{err}")

    def test_illegal_gfx1250_configurations_do_not_compile(self):
        for op, path in SOURCES.items():
            guard = _extract_guard(path)
            for label, kernel in ILLEGAL.items():
                with self.subTest(op=op, config=label):
                    ok, _ = self._compiles(guard, "gfx1250", kernel)
                    self.assertFalse(
                        ok,
                        f"{label} compiled on gfx1250 -- the guard is not stopping a "
                        f"configuration that returns wrong or undependable results",
                    )

    def test_suffixed_gfx1250_is_still_guarded(self):
        # GFX_ARCH can carry feature flags; the guard prefix-matches so that a
        # suffixed target is not accidentally exempted.
        for op, path in SOURCES.items():
            guard = _extract_guard(path)
            with self.subTest(op=op):
                ok, _ = self._compiles(
                    guard, "gfx1250:xnack-", ILLEGAL["32x32x32 (no 32x32 WMMA)"]
                )
                self.assertFalse(ok)

    def test_gfx9_targets_are_not_affected(self):
        # The guard is gfx1250-scoped. 32x32x32 is the correct gfx942 fp8 tile and
        # must keep compiling; a regression here would break the shipped archs.
        for op, path in SOURCES.items():
            guard = _extract_guard(path)
            for arch in ("gfx942", "gfx950", "gfx942:sramecc+:xnack-"):
                for label, kernel in ILLEGAL.items():
                    with self.subTest(op=op, arch=arch, config=label):
                        ok, err = self._compiles(guard, arch, kernel)
                        self.assertTrue(ok, f"{label} must compile on {arch}:\n{err}")

    def test_guard_is_what_rejects_the_bad_configs(self):
        """Negative control: with the static_asserts removed, all of ILLEGAL compiles.

        Without this, the suite above would still pass if the block failed to
        compile for some unrelated reason -- a syntax error in the extraction, a
        missing symbol -- and would be measuring nothing.
        """
        strip = re.compile(r"static_assert\(.*?\);", re.DOTALL)
        for op, path in SOURCES.items():
            disarmed = strip.sub("", _extract_guard(path))
            self.assertNotIn("static_assert", disarmed)
            for label, kernel in ILLEGAL.items():
                with self.subTest(op=op, config=label):
                    ok, err = self._compiles(disarmed, "gfx1250", kernel)
                    self.assertTrue(
                        ok,
                        f"{label} still failed with the asserts removed, so the "
                        f"rejection above was not the guard:\n{err}",
                    )


if __name__ == "__main__":
    unittest.main()
