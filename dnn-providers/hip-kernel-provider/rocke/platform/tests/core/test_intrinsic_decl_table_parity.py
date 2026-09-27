# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dual-engine parity for the LLVM intrinsic-declaration tables.

The Python engine keeps its declares in ``lower_llvm._INTRINSIC_DECLS`` (plus a
per-flavor override dict); the C++ engine keeps the same declares in flat
``rocke_ll_decl_t`` arrays in ``cpp/core/lower_llvm/data.cpp``. The two are
maintained **by hand**, in two files, in two languages. Nothing else in the tree
compares them.

They must agree on three things, and the third is the one that is easy to miss:

* **key set** — a key one engine has and the other does not is a ``_need()`` that
  resolves on one backend and raises on the other;
* **declaration text** — the bytes that land in the emitted module;
* **insertion order** — ``finalize`` emits declares in table order, so reordering
  two entries changes the emitted bytes without changing a single character of
  any declare. Byte-identity would catch that only if some kernel in the parity
  corpus happens to need both keys.

This is the cheapest check in the validation stack and the only one that guards
byte-identity *upstream* of emission: it needs no LLVM, no comgr, no GPU, and no
built C++ engine — it reads ``data.cpp`` as text. It therefore runs everywhere,
including hosts where the toolchain gate
(``tools/check_ir_validity.py``) can only report UNVALIDATED.

Scope: this says nothing about whether either table is *correct* — two engines
can agree on a declare that LLVM rejects. That is the toolchain gate's job.

Skipped when the C++ source tree is absent (an installed staging prefix ships
the package, not ``cpp/``).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from rocke.core.lower_llvm import (
    _INTRINSIC_DECLS,
    _INTRINSIC_DECLS_LLVM22_OVERRIDES,
    _INTRINSIC_DECLS_LLVM23_OVERRIDES,
)

_HERE = Path(__file__).resolve().parent
_DATA_CPP = _HERE.parents[1] / "cpp" / "core" / "lower_llvm" / "data.cpp"

# Python dict <-> C array. The Python side is an ordered dict (insertion order is
# the contract, see the module docstring); the C side is an ordered array.
_TABLES = (
    ("ROCKE_LL_INTRINSIC_DECLS", _INTRINSIC_DECLS),
    ("ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES", _INTRINSIC_DECLS_LLVM22_OVERRIDES),
    ("ROCKE_LL_INTRINSIC_DECLS_LLVM23_OVERRIDES", _INTRINSIC_DECLS_LLVM23_OVERRIDES),
)


def _c_string_literals(text: str) -> list[str]:
    """Every C string literal in ``text``, in order, unescaped.

    The C tables wrap long declares across lines as adjacent literals, which the
    compiler concatenates; this returns them individually so the caller can
    rejoin them the same way.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != '"':
            i += 1
            continue
        i += 1
        buf: list[str] = []
        while i < n and text[i] != '"':
            if text[i] == "\\" and i + 1 < n:
                nxt = text[i + 1]
                buf.append({"n": "\n", "t": "\t", "0": "\0"}.get(nxt, nxt))
                i += 2
                continue
            buf.append(text[i])
            i += 1
        i += 1  # closing quote
        out.append("".join(buf))
    return out


def _parse_c_table(source: str, name: str) -> list[tuple[str, str]]:
    """Extract ``{key, decl}`` pairs from a ``rocke_ll_decl_t`` array, in order."""
    start = re.search(
        r"\brocke_ll_decl_t\s+" + re.escape(name) + r"\s*\[\s*\]\s*=\s*\{", source
    )
    if start is None:
        raise AssertionError(f"array {name} not found in {_DATA_CPP.name}")

    # Walk to the matching close brace so a nested initializer cannot end the
    # scan early.
    i = start.end()
    depth = 1
    while i < len(source) and depth:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
        i += 1
    body = source[start.end() : i - 1]

    entries: list[tuple[str, str]] = []
    j = 0
    while j < len(body):
        if body[j] != "{":
            j += 1
            continue
        k, d = j + 1, 1
        while k < len(body) and d:
            if body[k] == "{":
                d += 1
            elif body[k] == "}":
                d -= 1
            k += 1
        literals = _c_string_literals(body[j + 1 : k - 1])
        if len(literals) < 2:
            raise AssertionError(
                f"{name}: entry at offset {j} has {len(literals)} string "
                "literal(s); expected a key and at least one decl fragment"
            )
        entries.append((literals[0], "".join(literals[1:])))
        j = k
    return entries


@unittest.skipUnless(_DATA_CPP.is_file(), f"C++ source not present ({_DATA_CPP})")
class TestIntrinsicDeclTableParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _DATA_CPP.read_text(encoding="utf-8")

    def test_key_sets_match(self):
        for name, py_table in _TABLES:
            with self.subTest(table=name):
                cpp_keys = {k for k, _ in _parse_c_table(self.source, name)}
                py_keys = set(py_table)
                self.assertEqual(
                    cpp_keys - py_keys,
                    set(),
                    msg=f"{name}: keys in data.cpp but not in the Python table",
                )
                self.assertEqual(
                    py_keys - cpp_keys,
                    set(),
                    msg=f"{name}: keys in the Python table but not in data.cpp",
                )

    def test_insertion_order_matches(self):
        # Not cosmetic: finalize emits declares in table order, so a reorder
        # changes emitted bytes while every declare stays character-identical.
        for name, py_table in _TABLES:
            with self.subTest(table=name):
                self.assertEqual(
                    [k for k, _ in _parse_c_table(self.source, name)],
                    list(py_table),
                    msg=f"{name}: declare emission order differs between engines",
                )

    def test_declaration_text_matches(self):
        for name, py_table in _TABLES:
            cpp_table = dict(_parse_c_table(self.source, name))
            for key, py_decl in py_table.items():
                if key not in cpp_table:
                    continue  # reported by test_key_sets_match
                with self.subTest(table=name, key=key):
                    self.assertEqual(
                        cpp_table[key],
                        py_decl,
                        msg=f"{name}[{key!r}]: declare text differs between engines",
                    )

    def test_tables_are_non_empty(self):
        # A parse that silently yielded nothing would make every comparison
        # above vacuously true.
        for name, py_table in _TABLES:
            with self.subTest(table=name):
                self.assertTrue(py_table, f"{name}: Python table is empty")
                self.assertTrue(
                    _parse_c_table(self.source, name), f"{name}: C++ array is empty"
                )

    def test_override_keys_exist_in_base_table(self):
        # The contract stated at the override dicts: only the declaration TEXT
        # differs per flavor, so call-site _need() lookups stay flavor-agnostic.
        # An override key with no base entry would resolve only on the flavors
        # that carry the override.
        for name, py_table in _TABLES[1:]:
            with self.subTest(table=name):
                self.assertEqual(
                    set(py_table) - set(_INTRINSIC_DECLS),
                    set(),
                    msg=f"{name}: override key absent from the base table",
                )


if __name__ == "__main__":
    unittest.main()
