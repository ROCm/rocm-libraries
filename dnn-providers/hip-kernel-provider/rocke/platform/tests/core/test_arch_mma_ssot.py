# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SSOT guards for the bare-op_id MMA metadata shared by the two engines.

The Python authoring frontend and the C++ backend each expose a bare-``op_id``
lookup used by ``IRBuilder.mma`` to size a ``tile.mma`` result vector without an
``ArchTarget`` in hand:

  * accumulator dtype -- ``target._op_id_c_dtype()`` (Python) /
    ``rocke_arch_mma_op_id_c_dtype`` (C, ``query.cpp``);
  * accumulator fragment length -- ``_MMA_FRAGMENT_INFO[op_id].c_frag_len``
    (Python) / the ``rocke_ati_mma_frag[]`` table (C, ``data.cpp``).

The C frag table is a hand-maintained mirror of the Python SSOT rather than
codegen'd from it (see the NOTE above ``rocke_ati_mma_frag`` in ``data.cpp``).
These tests are the drift guard that lets the two hand-written copies coexist:
if either side gains/loses an atom or edits a fragment length, the mirror test
fails until both agree. They also pin the first-wins / raise-on-drift contract
of ``_op_id_c_dtype`` so it stays deterministic and matches the C engine.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest import mock

from rocke.core.arch.target import (
    _MMA_FRAGMENT_INFO,
    _load_specs,
    _op_id_c_dtype,
    normalize_dtype,
)

# op_ids the Python SSOT registers but the C99 arch registry intentionally omits
# because it has no gfx1250 target (they take the (0,0,0,64) fallback in C). Kept
# as a prefix match so new gfx1250 atoms don't need to be enumerated here.
_C_ABSENT_PREFIXES = ("wmma_gfx1250_",)

# cpp/core/arch/data.cpp, resolved relative to this file (tests -> platform).
_DATA_CPP = Path(__file__).resolve().parents[2] / "cpp" / "core" / "arch" / "data.cpp"


def _c_absent(op_id: str) -> bool:
    return op_id.startswith(_C_ABSENT_PREFIXES)


def _parse_c_frag_table() -> dict:
    """Extract ``{op_id: c_frag_len}`` from the ``rocke_ati_mma_frag[]`` array in
    data.cpp by parsing the source directly (no built engine required)."""
    text = _DATA_CPP.read_text(encoding="utf-8")
    m = re.search(
        r"rocke_ati_mma_frag\[\]\s*=\s*\{(.*?)\};",
        text,
        re.DOTALL,
    )
    if not m:
        raise AssertionError(
            f"could not locate rocke_ati_mma_frag[] initializer in {_DATA_CPP}"
        )
    body = m.group(1)
    rows = re.findall(r'\{\s*"([^"]+)"\s*,\s*(\d+)\s*\}', body)
    if not rows:
        raise AssertionError("parsed rocke_ati_mma_frag[] but found no rows")
    table: dict = {}
    for op_id, frag in rows:
        if op_id in table:
            raise AssertionError(f"duplicate op_id {op_id!r} in rocke_ati_mma_frag[]")
        table[op_id] = int(frag)
    return table


class TestOpIdCDtype(unittest.TestCase):
    def test_matches_catalog_first_hit(self):
        # Every op_id in the catalog resolves to its normalized accumulator dtype,
        # taking the first arch that lists it (dict preserves catalog order).
        expected: dict = {}
        for row in _load_specs().values():
            for o in row["mma"]:
                expected.setdefault(o["op_id"], normalize_dtype(o["c"]))
        self.assertEqual(_op_id_c_dtype(), expected)

    def test_c_dtype_invariant_across_arches(self):
        # The whole premise of the bare-op_id lookup: an op_id's accumulator dtype
        # is invariant across the arches that list it, so building the map must not
        # raise on the real catalog. (The raise path is exercised below.)
        try:
            _op_id_c_dtype()
        except ValueError as exc:  # pragma: no cover - only hit on real drift
            self.fail(f"_op_id_c_dtype() raised on the shipped catalog: {exc}")

    def test_raises_on_cross_arch_disagreement(self):
        specs = _load_specs()
        # Find an op_id and clone its row into a fake arch with a different c dtype.
        sample = next(o for row in specs.values() for o in row["mma"])
        original_c = normalize_dtype(sample["c"])
        other_c = "i32" if original_c != "i32" else "f32"
        clash = dict(sample)
        clash["c"] = other_c
        drifted = dict(specs)
        drifted["_synthetic_drift"] = {"mma": [clash]}

        _op_id_c_dtype.cache_clear()
        try:
            with mock.patch("rocke.core.arch.target._load_specs", return_value=drifted):
                with self.assertRaises(ValueError):
                    _op_id_c_dtype()
        finally:
            _op_id_c_dtype.cache_clear()


class TestMmaFragTableMirror(unittest.TestCase):
    """The C ``rocke_ati_mma_frag[]`` table must mirror the Python
    ``_MMA_FRAGMENT_INFO`` c_frag_len column exactly (modulo the gfx1250 atoms the
    C arch registry cannot emit)."""

    def test_c_rows_exist_in_python_with_same_frag_len(self):
        c_table = _parse_c_frag_table()
        for op_id, c_frag in c_table.items():
            self.assertIn(
                op_id,
                _MMA_FRAGMENT_INFO,
                msg=f"{op_id!r} is in data.cpp but not _MMA_FRAGMENT_INFO",
            )
            self.assertEqual(
                _MMA_FRAGMENT_INFO[op_id].c_frag_len,
                c_frag,
                msg=f"c_frag_len drift for {op_id!r}: python="
                f"{_MMA_FRAGMENT_INFO[op_id].c_frag_len} vs C={c_frag}",
            )

    def test_python_atoms_present_in_c_table(self):
        c_table = _parse_c_frag_table()
        for op_id, info in _MMA_FRAGMENT_INFO.items():
            if _c_absent(op_id):
                self.assertNotIn(
                    op_id,
                    c_table,
                    msg=f"{op_id!r} is expected absent from data.cpp (no C "
                    f"gfx1250 target) but was found",
                )
                continue
            self.assertIn(
                op_id,
                c_table,
                msg=f"{op_id!r} is in _MMA_FRAGMENT_INFO but missing from the "
                f"rocke_ati_mma_frag[] table in data.cpp",
            )
            self.assertEqual(
                info.c_frag_len,
                c_table[op_id],
                msg=f"c_frag_len drift for {op_id!r}: python={info.c_frag_len} "
                f"vs C={c_table[op_id]}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
