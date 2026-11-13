# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The universal GEMM's TDM K-loop module and its arch-neutral validators.

``TdmKLoop.emit`` picks the TDM K-loop shape by ``tdm_depth`` and
``tdm_early_refill``; the C++ engine's ``rocke_gemm_emit_kloop_tdm`` makes the
same choice, which byte identity and the differential golden cover per shape.
These tests pin the selection itself, and that the universal GEMM's code in
both engines decides nothing by comparing the arch name with ``"gfx1250"``:
what it admits per arch comes from ``ArchTarget.gemm`` (``arch_specs.json``
"gemm"). The C++ half reads the source tree and skips in an installed test
layout, which carries no ``cpp/``.
"""

from __future__ import annotations

import ast
import re
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

from rocke.assets import platform_root
from rocke.instances.common import _gemm_kloop_tdm, gemm_universal
from rocke.instances.common._gemm_kloop_tdm import TdmKLoop

_SHAPES = ("unpipelined", "ping_pong", "early_refill", "ring")


def _picked(depth: int, early_refill: bool = False) -> str:
    """The loop shape ``emit`` calls for a trait, with every shape stubbed."""
    loop = replace(
        TdmKLoop(**{f.name: None for f in fields(TdmKLoop)}),
        trait=SimpleNamespace(tdm_depth=depth, tdm_early_refill=early_refill),
    )
    calls = []
    for shape in _SHAPES:
        object.__setattr__(loop, shape, lambda shape=shape: calls.append(shape))
    loop.emit()
    assert len(calls) == 1, calls
    return calls[0]


class TestTdmKLoopSelection(unittest.TestCase):
    def test_each_depth_picks_its_loop_shape(self):
        for depth, early, shape in (
            (1, False, "unpipelined"),
            (2, False, "ping_pong"),
            (2, True, "early_refill"),
            (3, False, "ring"),
            (4, False, "ring"),
        ):
            with self.subTest(depth=depth, early_refill=early):
                self.assertEqual(_picked(depth, early), shape)


def _python_code_strings(path: Path) -> list[tuple[int, str]]:
    """String constants of a module's code, docstrings excluded."""
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _cpp_code_strings(path: Path) -> list[str]:
    """String literals of a C/C++ source, comments stripped."""
    src = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return re.findall(r'"(?:[^"\\\n]|\\.)*"', src)


class TestNoArchNameLiterals(unittest.TestCase):
    def test_python_universal_gemm_has_no_gfx1250_literal(self):
        for module in (gemm_universal, _gemm_kloop_tdm):
            path = Path(module.__file__)
            with self.subTest(module=path.name):
                hits = [
                    (line, s)
                    for line, s in _python_code_strings(path)
                    if "gfx1250" in s
                ]
                self.assertEqual(hits, [])

    def test_cpp_universal_gemm_has_no_gfx1250_literal(self):
        common = platform_root() / "cpp" / "instances" / "common"
        if not common.is_dir():
            raise unittest.SkipTest("reads the C++ sources, which this layout omits")
        for name in (
            "gemm_spec.cpp",
            "gemm_universal.cpp",
            "gemm_build_compute.cpp",
            "gemm_build_kloop_tdm.cpp",
        ):
            with self.subTest(source=name):
                hits = [s for s in _cpp_code_strings(common / name) if "gfx1250" in s]
                self.assertEqual(hits, [])

    def test_the_scan_finds_a_planted_literal(self):
        # Guards the scanners: a literal in code is found, one in a comment or
        # docstring is not.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp) / "m.py"
            py.write_text('"""gfx1250 docstring"""\nX = "gfx1250"\n# "gfx1250"\n')
            self.assertEqual(
                [s for _, s in _python_code_strings(py) if "gfx1250" in s],
                ["gfx1250"],
            )
            cpp = Path(tmp) / "m.cpp"
            cpp.write_text(
                '/* "gfx1250" */\n// "gfx1250"\nconst char* a = "gfx1250";\n'
            )
            self.assertEqual(_cpp_code_strings(cpp), ['"gfx1250"'])


if __name__ == "__main__":
    unittest.main()
