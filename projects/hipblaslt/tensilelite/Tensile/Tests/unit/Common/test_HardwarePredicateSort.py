################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Tests for HardwarePredicate sort ordering in Hardware.py.

The sort order directly determines the row order in serialized master solution
libraries, and therefore controls runtime fallback behaviour.  The key invariant
is: more-specific rows must appear *before* less-specific rows so that
ExactLogicLibrary::findBestSolution (first-match-wins) picks the right library.

Sort priority (highest first):
  1. Predicates with a PCI chip ID beat those without.
  2. Among predicates with chip IDs, *higher* chip IDs come first (descending)
     so that fallback-source devices (e.g. mi355=0x75a3) get their exact row
     before the fallback-target row (e.g. mi350=0x75a0).
  3. Higher CU counts come first (SPX=256 > CPX=64 > no-CU).
  4. TruePred always sorts last.
"""

import pytest
import tempfile
import os

from Tensile.Hardware import HardwarePredicate
from Tensile.Common.Architectures import gfxToIsa
from Tensile.Common.Utilities import state
from Tensile.SolutionLibrary import PredicateLibrary, SingleSolutionLibrary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GFX950_ISA = gfxToIsa("gfx950")  # e.g. (9, 5, 0)
_GFX942_ISA = gfxToIsa("gfx942")


def _hw(isa=_GFX950_ISA, cuCount=None, deviceNames=None):
    """Shorthand for HardwarePredicate.FromHardware."""
    return HardwarePredicate.FromHardware(isa, cuCount, deviceNames)


def _true():
    return HardwarePredicate("TruePred")


def _sorted_preds(preds):
    """Return a fresh sorted copy."""
    return sorted(preds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChipIdDescendingOrder:
    """Chip IDs must sort descending so fallback-source rows come first."""

    def test_mi355_before_mi350(self):
        mi350 = _hw(deviceNames=["Device 75a0"])
        mi355 = _hw(deviceNames=["Device 75a3"])
        assert _sorted_preds([mi350, mi355]) == [mi355, mi350]

    def test_mi355_before_mi350_reversed_input(self):
        mi350 = _hw(deviceNames=["Device 75a0"])
        mi355 = _hw(deviceNames=["Device 75a3"])
        assert _sorted_preds([mi355, mi350]) == [mi355, mi350]

    def test_three_chip_ids_descending(self):
        mi350  = _hw(deviceNames=["Device 75a0"])
        mi355  = _hw(deviceNames=["Device 75a3"])
        mi350p = _hw(deviceNames=["Device 75a8"])
        result = _sorted_preds([mi350, mi350p, mi355])
        assert result == [mi350p, mi355, mi350]


class TestChipIdBeatsNoChipId:
    """A predicate with a chip ID is more specific and must sort first."""

    def test_chip_id_before_processor_only(self):
        with_id = _hw(deviceNames=["Device 75a0"])
        no_id   = _hw()
        assert _sorted_preds([no_id, with_id]) == [with_id, no_id]


class TestCuCountDescendingOrder:
    """Within the same chip ID, higher CU counts sort first."""

    def test_spx_before_cpx(self):
        spx = _hw(cuCount=256, deviceNames=["Device 75a0"])
        cpx = _hw(cuCount=64,  deviceNames=["Device 75a0"])
        assert _sorted_preds([cpx, spx]) == [spx, cpx]

    def test_cu_before_no_cu(self):
        with_cu = _hw(cuCount=256, deviceNames=["Device 75a0"])
        no_cu   = _hw(deviceNames=["Device 75a0"])
        assert _sorted_preds([no_cu, with_cu]) == [with_cu, no_cu]

    def test_spx_cpx_nocu(self):
        spx   = _hw(cuCount=256, deviceNames=["Device 75a0"])
        cpx   = _hw(cuCount=64,  deviceNames=["Device 75a0"])
        no_cu = _hw(deviceNames=["Device 75a0"])
        assert _sorted_preds([no_cu, cpx, spx]) == [spx, cpx, no_cu]


class TestTruePredLast:
    """TruePred is the catch-all and must always sort last."""

    def test_truepred_after_chip_id(self):
        specific = _hw(deviceNames=["Device 75a0"])
        fallback = _true()
        assert _sorted_preds([fallback, specific]) == [specific, fallback]

    def test_truepred_after_processor_only(self):
        proc     = _hw()
        fallback = _true()
        assert _sorted_preds([fallback, proc]) == [proc, fallback]


class TestFullScenarioOrder:
    """End-to-end: verify the full row order expected by the fallback scenarios."""

    def test_mi350_mi355_spx_cpx_catchall(self):
        """Rows for gfx950 with mi350 + mi355, SPX + CPX + catch-all."""
        mi355_spx = _hw(cuCount=256, deviceNames=["Device 75a3"])
        mi350_spx = _hw(cuCount=256, deviceNames=["Device 75a0"])
        mi355_cpx = _hw(cuCount=64,  deviceNames=["Device 75a3"])
        mi350_cpx = _hw(cuCount=64,  deviceNames=["Device 75a0"])
        mi355_any = _hw(deviceNames=["Device 75a3"])
        mi350_any = _hw(deviceNames=["Device 75a0"])
        catchall  = _hw()  # processor only

        shuffled = [mi350_any, catchall, mi355_cpx, mi350_spx, mi355_any, mi350_cpx, mi355_spx]
        expected = [mi355_spx, mi355_cpx, mi355_any, mi350_spx, mi350_cpx, mi350_any, catchall]

        assert _sorted_preds(shuffled) == expected


# ---------------------------------------------------------------------------
# Stub solution for PredicateLibrary row construction
# ---------------------------------------------------------------------------

class _StubSolution:
    """Minimal object that looks like a Contraction.Solution for serialization."""
    def __init__(self, idx):
        self.index = idx


class _StubLib:
    """Minimal single-solution library with a state() and tag for merge()."""
    Tag = "Single"

    def __init__(self, idx):
        self.solution = _StubSolution(idx)

    @property
    def tag(self):
        return self.Tag

    def state(self):
        return {"type": self.tag, "index": self.solution.index}

    def merge(self, other):
        pass

    def remapSolutionIndices(self, indexMap):
        pass


def _make_hw_row(pred, idx):
    """Build a row dict the same way MasterSolutionLibrary.FromOriginalState does."""
    return {"predicate": pred, "library": _StubLib(idx)}


# ===========================================================================
# Round-trip serialization tests
#
# These tests verify that PredicateLibrary.merge() sorts hardware rows
# correctly, and that the sorted order survives serialization to both
# msgpack and YAML and back.  This guards against:
#   - Regressions in HardwarePredicate.__lt__
#   - Changes to state() or the serialization path that reorder rows
#   - Format-level issues that might not preserve array order
# ===========================================================================

class TestPredicateLibraryMergeOrder:
    """Verify that PredicateLibrary.merge() produces the correct row order."""

    def test_merge_sorts_hardware_rows_by_predicate(self):
        """Two separate Hardware PredicateLibraries merged together should
        have rows sorted descending by chip ID."""
        lib_mi350 = PredicateLibrary(tag="Hardware", rows=[
            _make_hw_row(_hw(deviceNames=["Device 75a0"]), idx=1),
        ])
        lib_mi355 = PredicateLibrary(tag="Hardware", rows=[
            _make_hw_row(_hw(deviceNames=["Device 75a3"]), idx=2),
        ])

        lib_mi350.merge(lib_mi355)

        # After merge, mi355 (0x75a3) should be first (descending chip ID)
        chip_ids = [_extract_chip_id(r["predicate"]) for r in lib_mi350.rows]
        assert chip_ids == [0x75a3, 0x75a0], \
            f"Expected descending chip IDs [0x75a3, 0x75a0], got {[hex(c) for c in chip_ids]}"

    def test_merge_full_scenario(self):
        """Merge multiple Hardware rows and verify full ordering."""
        lib = PredicateLibrary(tag="Hardware", rows=[
            _make_hw_row(_hw(deviceNames=["Device 75a0"]), idx=1),
        ])
        for idx, pred in enumerate([
            _hw(cuCount=256, deviceNames=["Device 75a3"]),
            _hw(cuCount=64,  deviceNames=["Device 75a3"]),
            _hw(cuCount=256, deviceNames=["Device 75a0"]),
            _hw(deviceNames=["Device 75a3"]),
            _hw(),
        ], start=2):
            other = PredicateLibrary(tag="Hardware", rows=[_make_hw_row(pred, idx)])
            lib.merge(other)

        chip_ids = [_extract_chip_id(r["predicate"]) for r in lib.rows]
        cu_counts = [_extract_cu_count(r["predicate"]) for r in lib.rows]

        # Expected order: mi355 CU=256, mi355 CU=64, mi355 no-CU, mi350 CU=256, mi350 no-CU, catch-all
        assert chip_ids == [0x75a3, 0x75a3, 0x75a3, 0x75a0, 0x75a0, None]
        assert cu_counts == [256, 64, None, 256, None, None]


class TestSerializationRoundTrip:
    """Verify that serialization preserves the row order produced by merge()."""

    def _build_merged_lib(self):
        """Build a Hardware PredicateLibrary with multiple chip IDs, merge, return it."""
        lib = PredicateLibrary(tag="Hardware")
        rows = [
            (_hw(cuCount=256, deviceNames=["Device 75a0"]), 1),
            (_hw(cuCount=256, deviceNames=["Device 75a3"]), 2),
            (_hw(deviceNames=["Device 75a0"]),              3),
            (_hw(deviceNames=["Device 75a3"]),              4),
            (_hw(),                                         5),
        ]
        for pred, idx in rows:
            other = PredicateLibrary(tag="Hardware", rows=[_make_hw_row(pred, idx)])
            if not lib.rows:
                lib = other
            else:
                lib.merge(other)
        return lib

    def _extract_solution_indices(self, serialized_rows):
        """Extract solution indices from serialized row dicts."""
        indices = []
        for row in serialized_rows:
            lib_data = row["library"]
            if isinstance(lib_data, dict) and "index" in lib_data:
                indices.append(lib_data["index"])
            else:
                indices.append(lib_data)
        return indices

    def test_state_preserves_row_order(self):
        """state() should emit rows in the same order as the in-memory list."""
        lib = self._build_merged_lib()

        # Capture in-memory order
        mem_indices = [r["library"].solution.index for r in lib.rows]

        # Serialize to dict via state()
        serialized = state(lib)
        ser_indices = self._extract_solution_indices(serialized["rows"])

        assert mem_indices == ser_indices, \
            f"state() reordered rows: memory={mem_indices}, serialized={ser_indices}"

    def test_msgpack_round_trip_preserves_order(self):
        """Write to msgpack, read back, verify row order is identical."""
        msgpack = pytest.importorskip("msgpack")

        lib = self._build_merged_lib()
        mem_indices = [r["library"].solution.index for r in lib.rows]
        serialized = state(lib)

        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            tmpfile = f.name
            msgpack.pack(serialized, f)

        try:
            with open(tmpfile, "rb") as f:
                loaded = msgpack.unpack(f, raw=False)
            loaded_indices = self._extract_solution_indices(loaded["rows"])
            assert mem_indices == loaded_indices, \
                f"msgpack round-trip reordered rows: original={mem_indices}, loaded={loaded_indices}"
        finally:
            os.unlink(tmpfile)

    def test_yaml_round_trip_preserves_order(self):
        """Write to YAML, read back, verify row order is identical."""
        yaml = pytest.importorskip("yaml")

        lib = self._build_merged_lib()
        mem_indices = [r["library"].solution.index for r in lib.rows]
        serialized = state(lib)

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            tmpfile = f.name
            yaml.dump(serialized, f)

        try:
            with open(tmpfile, "r") as f:
                loaded = yaml.safe_load(f)
            loaded_indices = self._extract_solution_indices(loaded["rows"])
            assert mem_indices == loaded_indices, \
                f"YAML round-trip reordered rows: original={mem_indices}, loaded={loaded_indices}"
        finally:
            os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# Extraction helpers for inspecting predicates inside PredicateLibrary rows
# ---------------------------------------------------------------------------

def _extract_chip_id(pred):
    """Extract the PCI chip ID from a HardwarePredicate, or None."""
    from Tensile.Hardware import _extractPciChipIDs
    if pred.tag == "TruePred":
        return None
    inner = pred.value  # the AMDGPU-level predicate
    if inner.tag == "And":
        pci_pred = next((p for p in inner.value if p.tag in ("PciChipID", "Or")), None)
        return _extractPciChipIDs(pci_pred)
    return None


def _extract_cu_count(pred):
    """Extract the CU count from a HardwarePredicate, or None."""
    if pred.tag == "TruePred":
        return None
    inner = pred.value
    if inner.tag == "And":
        cu_pred = next((p for p in inner.value if p.tag == "CUCount"), None)
        return cu_pred.value if cu_pred else None
    return None
