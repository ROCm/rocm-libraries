# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the StreamK=5 hybrid mode.

These tests are intentionally codegen-light: they validate that the
parameter enumeration, the solution-hash discriminator, and the component
dispatcher all recognize StreamK=5 as a first-class mode that should emit
both the static (SK3) and dynamic (SK4) paths back-to-back.
"""

import re
from pathlib import Path


# Resolve absolute paths to the modules under test so we can read them as
# plain text. This avoids importing Tensile.Components.StreamK at test
# collection time (which requires rocisa, optional in the unit env).
_TENSILELITE_ROOT = Path(__file__).resolve().parents[3]
_VALID_PARAMS_PY = _TENSILELITE_ROOT / "Tensile" / "Common" / "ValidParameters.py"
_STREAMK_PY = _TENSILELITE_ROOT / "Tensile" / "Components" / "StreamK.py"
_SIGNATURE_PY = _TENSILELITE_ROOT / "Tensile" / "Components" / "Signature.py"
_SOLUTION_PY = _TENSILELITE_ROOT / "Tensile" / "SolutionStructs" / "Solution.py"
_PERSISTENT_LOOP_PY = _TENSILELITE_ROOT / "Tensile" / "Components" / "PersistentLoop.py"


def _read(p: Path) -> str:
    assert p.exists(), f"missing source file: {p}"
    return p.read_text(encoding="utf-8")


def _streamk_hybrid_body(src: str) -> str:
    """Return only the lines inside the StreamKHybrid class body so
    substring assertions don't false-positive against pre-existing SK3
    or SK4 code elsewhere in StreamK.py.
    """
    m = re.search(r"^class\s+StreamKHybrid\b.*$", src, re.MULTILINE)
    assert m, "class StreamKHybrid not found in StreamK.py"
    start = m.start()
    # Stop at the next top-level `class ` (column 0) or EOF.
    end_match = re.search(r"^class\s", src[m.end():], re.MULTILINE)
    end = m.end() + (end_match.start() if end_match else len(src) - m.end())
    return src[start:end]


class TestStreamK5ValidParameters:
    def test_streamk_enum_includes_5(self):
        src = _read(_VALID_PARAMS_PY)
        # Match the canonical line like  "StreamK": [0, 1, 2, 3, 4, 5]
        assert '"StreamK": [0, 1, 2, 3, 4, 5]' in src, (
            "ValidParameters.py must enumerate StreamK 0..5"
        )

    def test_streamk_docstring_mentions_hybrid(self):
        src = _read(_VALID_PARAMS_PY)
        assert "Hybrid" in src or "hybrid" in src


class TestStreamK5Component:
    """The Tensile component registry must dispatch StreamK==5 to the
    StreamKHybrid class so that both SK3 and SK4 paths get emitted."""

    def test_streamk_hybrid_class_defined(self):
        src = _read(_STREAMK_PY)
        assert "class StreamKHybrid" in src

    def test_streamk_hybrid_registered_for_mode_5(self):
        src = _read(_STREAMK_PY)
        # The dispatcher key on the class.
        assert 'kernel = {"StreamK": 5}' in src


class TestStreamK5SolutionHashDiscrimination:
    """StreamK=5 solutions must hash to a distinct name from SK3/SK4 so
    that the runtime dispatcher can tell them apart."""

    def test_solution_validation_handles_mode_5(self):
        src = _read(_SOLUTION_PY)
        # The intersection-of-SK3+SK4 validation must recognize SK5.
        assert 'StreamK"] == 5' in src or 'StreamK") == 5' in src or \
               '(3, 4, 5)' in src or '[3, 4, 5]' in src or \
               '.supportsSubtileImpl' in src


class TestStreamK5DualPathLabels:
    """Check that the StreamKHybrid class body itself emits the
    mode-extraction sequence and uses both static and dynamic path
    label name-roots (renamed via getNameInc to avoid assembler
    collisions). Assertions are anchored to the class body so they do
    not false-positive against pre-existing SK3 / SK4 code elsewhere
    in StreamK.py."""

    def test_streamk_hybrid_source_uses_both_paths(self):
        body = _streamk_hybrid_body(_read(_STREAMK_PY))
        # Static-path label roots (renamed via getNameInc inside SK5).
        assert "SK_FullTile" in body
        assert "SK_PartialTile" in body
        # Dynamic-path label roots (renamed via getNameInc inside SK5).
        assert "SK_UpdateDone" in body
        assert "SK_SplitUpdate" in body

    def test_streamk_hybrid_mode_sgpr_extracted(self):
        body = _streamk_hybrid_body(_read(_STREAMK_PY))
        assert "StreamKHybridMode" in body
        # Mode extraction shifts the MSB of MagicShiftItersPerTile into
        # the StreamKHybridMode SGPR exactly once. The call site may
        # wrap across several lines, so use [\s\S] to span newlines.
        assert re.search(
            r"SLShiftRightB32\([\s\S]*?StreamKHybridMode[\s\S]*?MagicShiftItersPerTile[\s\S]*?\b31\b",
            body), \
            "expected SLShiftRightB32(... StreamKHybridMode ... MagicShiftItersPerTile ... 31) in StreamKHybrid"

    def test_streamk_hybrid_masks_magic_shift(self):
        """The static path inside the SK5 kernel must mask off bit 31
        of MagicShiftItersPerTile before using it as a shift count."""
        src = _read(_STREAMK_PY)
        # Match the SK5-gated mask, not the unrelated 0x1FFFFFF mask in SK3.
        pattern = (r'kernel\["StreamK"\]\s*==\s*5[\s\S]{0,400}?'
                   r'SAndB32\([\s\S]{0,200}?0x1[Ff]\b')
        assert re.search(pattern, src), \
            "expected SK5-gated SAndB32(..., 0x1F) on MagicShiftItersPerTile"

    def test_streamk_hybrid_unique_labels(self):
        """When SK5 inlines both static and dynamic paths, conflicting
        labels must be renamed with getNameInc to avoid assembler
        errors."""
        body = _streamk_hybrid_body(_read(_STREAMK_PY))
        assert "getNameInc(\"SK_FullTile\")" in body
        assert "getNameInc(\"SK_PartialTile\")" in body
        assert "getNameInc(\"SK_UpdateDone\")" in body
        assert "getNameInc(\"SK_SplitUpdate\")" in body


class TestStreamK5SignatureUnion:
    """StreamK=5 kernels need the union of SK3+SK4 args in the signature."""

    def test_signature_handles_mode_5(self):
        src = _read(_SIGNATURE_PY)
        assert 'StreamK"] == 5' in src or '(3, 4, 5)' in src or '[3, 4, 5]' in src


class TestStreamK5PersistentLoop:
    """The persistent-loop close emitter must have an SK5 branch."""

    def test_persistent_loop_handles_mode_5(self):
        src = _read(_PERSISTENT_LOOP_PY)
        assert 'StreamK"] == 5' in src or '(3, 4, 5)' in src or '[3, 4, 5]' in src


class TestStreamK5SixArgCollapse:
    """SK5 must push only the 6 args matching the active runtime mode
    (not the union of SK3+SK4 = 11 args), with SK4 reader names resolved
    via RegSet aliases onto the SK3 primary SGPRs.
    """

    # ---- Signature.py: exactly 6 SK args + 24-byte size ----
    def test_signature_emits_six_sk_args(self):
        src = _read(_SIGNATURE_PY)
        m = re.search(
            r'elif kernel\["StreamK"\] == 5:([\s\S]*?)(?=^\s*elif |^\s*if )',
            src, re.MULTILINE)
        assert m, "SK5 elif block not found in Signature.py"
        block = m.group(1)
        sk_names = re.findall(r'signature\.addArg\("([^"]+)"', block)
        assert sk_names == [
            "ItersPerTile",
            "MagicNumberItersPerTile",
            "MagicShiftItersPerTile",
            "SKItersPerWG",
            "skGrid",
            "skTiles",
        ], f"SK5 signature must emit exactly 6 SK3-named args; got {sk_names}"
        assert "gemmArgumentSize += 24" in block, (
            "SK5 signature must declare gemmArgumentSize += 24 (6 args x 4 B)"
        )
        assert "gemmArgumentSize += 44" not in block, (
            "SK5 must no longer use the legacy 44-byte (11-arg) size"
        )

    # ---- KernelWriter.py: exactly 6 defineSgpr + numSgprStreamK += 6 ----
    def test_define_sgpr_six_streamk_slots(self):
        kw_py = _TENSILELITE_ROOT / "Tensile" / "KernelWriter.py"
        src = _read(kw_py)
        m = re.search(
            r'elif kernel\["StreamK"\] == 5:([\s\S]*?)numSgprStreamK \+= \d+',
            src)
        assert m, "SK5 defineSgpr block not found in KernelWriter.py"
        block = m.group(0)
        define_names = re.findall(r'self\.defineSgpr\("([^"]+)", 1\)', block)
        assert define_names == [
            "ItersPerTile",
            "MagicNumberItersPerTile",
            "MagicShiftItersPerTile",
            "SKItersPerWG",
            "skGrid",
            "skTiles",
        ], f"SK5 must defineSgpr only the 6 SK3-named slots; got {define_names}"
        assert "numSgprStreamK += 6" in block, (
            "SK5 must increment numSgprStreamK by exactly 6"
        )
        assert "numSgprStreamK += 11" not in block

    # ---- KernelWriterAssembly.py: 5 SK4->SK3 RegSet aliases ----
    def test_assembly_emits_five_sk4_aliases(self):
        kwa_py = _TENSILELITE_ROOT / "Tensile" / "KernelWriterAssembly.py"
        src = _read(kwa_py)
        expected_pairs = [
            ("sgprTotalItems",   "sgprMagicNumberItersPerTile"),
            ("sgprSKTiles",      "sgprMagicShiftItersPerTile"),
            ("sgprSKSplit",      "sgprSKItersPerWG"),
            ("sgprSKItersPerWI", "sgprskGrid"),
            ("sgprSKGrid",       "sgprskTiles"),
        ]
        m = re.search(
            r'if kernel\["StreamK"\] == 5:([\s\S]{0,2000}?)(?=^\s{4}elif |^\s{4}module\.addSpaceLine)',
            src, re.MULTILINE)
        assert m, "SK5-gated RegSet alias block not found in KernelWriterAssembly.py"
        block = m.group(1)
        for alias, primary in expected_pairs:
            pat = (rf'RegSet\(\s*"s"\s*,\s*"{re.escape(alias)}"\s*,'
                   rf'\s*"{re.escape(primary)}"\s*,\s*0\s*\)')
            assert re.search(pat, block), (
                f"missing SK5 RegSet alias {alias} -> {primary}+0")

    # ---- StreamK.py: mode-bit mask after extraction ----
    def test_mode_extraction_masks_high_bit(self):
        body = _streamk_hybrid_body(_read(_STREAMK_PY))
        pattern = (r'SLShiftRightB32\([\s\S]*?StreamKHybridMode[\s\S]*?\b31\b'
                   r'[\s\S]{0,400}?'
                   r'SAndB32\([\s\S]*?MagicShiftItersPerTile'
                   r'[\s\S]*?0x7[Ff]{7}\b')
        assert re.search(pattern, body), (
            "expected SAndB32(MagicShiftItersPerTile, 0x7FFFFFFF) right "
            "after the mode-bit extraction in _emitModeExtraction"
        )


_CONTRACTION_PROBLEM_HPP = (_TENSILELITE_ROOT / "include" / "Tensile"
                            / "ContractionProblem.hpp")
_CONTRACTION_SOLUTION_HPP = (_TENSILELITE_ROOT / "include" / "Tensile"
                             / "ContractionSolution.hpp")
_CONTRACTION_SOLUTION_CPP = (_TENSILELITE_ROOT / "src"
                             / "ContractionSolution.cpp")


class TestStreamK5HybridAutoMode:
    def test_contraction_problem_param_setter_renamed_to_mode(self):
        src = _read(_CONTRACTION_PROBLEM_HPP)
        assert "setDynPersistentTileMode" in src, (
            "ContractionProblem.hpp must expose setDynPersistentTileMode(int)"
        )
        assert "dynPersistentTileMode()" in src, (
            "ContractionProblem.hpp must expose dynPersistentTileMode() getter"
        )

    def test_contraction_problem_legacy_bool_api_removed(self):
        src = _read(_CONTRACTION_PROBLEM_HPP)
        assert not re.search(
            r'\bvoid\s+setDynPersistentTile\s*\(\s*bool\b', src), \
            "legacy setDynPersistentTile(bool) must be removed"
        assert not re.search(
            r'\bbool\s+dynPersistentTile\s*\(\s*\)', src), \
            "legacy bool dynPersistentTile() getter must be removed"

    def test_contraction_problem_has_sm_count_target_accessor(self):
        src = _read(_CONTRACTION_PROBLEM_HPP)
        assert "setSmCountTarget" in src
        assert "smCountTarget()" in src

    def test_streamk_settings_uses_int_mode(self):
        src = _read(_CONTRACTION_SOLUTION_HPP)
        assert re.search(
            r'dynPersistentTileMode\s*=\s*2', src), \
            "StreamKSettings::dynPersistentTileMode default 2 (AUTO) not found"
        assert "smCountTarget" in src

    def test_contraction_solve_dispatches_auto_via_origami(self):
        src = _read(_CONTRACTION_SOLUTION_CPP)
        assert "origami::streamk::select_hybrid_mode(" in src, \
            "SK5 AUTO branch must call origami::streamk::select_hybrid_mode"
        assert re.search(
            r'sk\.dynPersistentTileMode\s*=\s*problem\.getParams\(\)\.dynPersistentTileMode\(\)',
            src), \
            "sk.dynPersistentTileMode must be sourced from getParams().dynPersistentTileMode()"
        assert re.search(
            r'sk\.smCountTarget\s*=\s*problem\.getParams\(\)\.smCountTarget\(\)',
            src), \
            "sk.smCountTarget must be sourced from getParams().smCountTarget()"


class TestStreamK5SixArgCollapseNoRegression:
    def test_signature_still_emits_six_sk_args(self):
        src = _read(_SIGNATURE_PY)
        m = re.search(
            r'elif kernel\["StreamK"\] == 5:([\s\S]*?)(?=^\s*elif |^\s*if )',
            src, re.MULTILINE)
        assert m, "SK5 elif block not found in Signature.py"
        block = m.group(1)
        sk_names = re.findall(r'signature\.addArg\("([^"]+)"', block)
        assert len(sk_names) == 6, (
            f"SK5 must still emit exactly 6 SK args; got {len(sk_names)}: {sk_names}"
        )
        assert "gemmArgumentSize += 24" in block
        assert "gemmArgumentSize += 44" not in block
