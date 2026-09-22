# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Gate the arch-domain *warning lane* — the first consumer of the artifact.

``test_arch_domain_artifact.py`` gates the committed data. This file gates what
the lowerer does with it: every intrinsic demand goes through one chokepoint,
that chokepoint consults the table, and it speaks only for a measured, definite
negative.

The three things that can quietly break the lane, and the test that holds each:

* **A demand that skips the chokepoint.** Writing ``self._needs_intrin[k] =
  True`` directly still emits the declare, so nothing visibly breaks — the check
  just never runs for that key. Caught by AST walk, in the style of
  ``library/tests/test_library_layering.py``.
* **A no-data status read as a negative.** ``target_unsupported``,
  ``toolchain_crash``, ``toolchain_timeout`` and a missing column all mean the
  question was never answered. Treating any of them as "no" turns a developer on
  an older ROCm into a wall of warnings about intrinsics that are fine — 14% of
  the llvm20 column is no-data.
* **The implicit target default coming back.** A silent ``gfx950`` makes the
  lookup unanswerable, because the target it is consulted for would be a guess.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from rocke.core.arch import domain as arch_domain
from rocke.core.lower_llvm import ArchDomainWarning, lower_kernel_to_llvm

PLATFORM = Path(__file__).resolve().parents[2]
LOWER_LLVM = PLATFORM / "python" / "rocke" / "core" / "lower_llvm.py"
CPP_CORE = PLATFORM / "cpp" / "core" / "lower_llvm" / "core.cpp"


# --------------------------------------------------------------------------
# the loader's contract
# --------------------------------------------------------------------------


def _committed_flavors() -> list[str]:
    from rocke.core.lower_llvm import LLVM_FLAVORS

    return [f for f in LLVM_FLAVORS if arch_domain.load(f) is not None]


def test_at_least_one_column_is_committed() -> None:
    """Without this, every test below would pass vacuously."""
    assert _committed_flavors(), "no committed arch-domain column to test against"


def test_only_arch_absent_is_negative() -> None:
    for status in (
        arch_domain.STATUS_OK,
        arch_domain.STATUS_NAME_ABSENT,
        *sorted(arch_domain.NO_DATA_STATUSES),
    ):
        assert not arch_domain.is_negative(status), status
    assert arch_domain.is_negative(arch_domain.STATUS_ARCH_ABSENT)


def test_no_data_statuses_partition_the_vocabulary() -> None:
    """Answers and non-answers partition the vocabulary with nothing left over,
    so a status added later cannot fall silently into neither bucket."""
    every = {
        v
        for name, v in vars(arch_domain).items()
        if name.startswith("STATUS_") and isinstance(v, str)
    }
    answers = {
        arch_domain.STATUS_OK,
        arch_domain.STATUS_NAME_ABSENT,
        arch_domain.STATUS_ARCH_ABSENT,
    }
    assert answers & arch_domain.NO_DATA_STATUSES == set()
    assert answers | arch_domain.NO_DATA_STATUSES == every


def test_missing_column_is_none_not_an_error() -> None:
    assert arch_domain.load("llvm-that-nobody-ships") is None


def test_unknown_key_and_unknown_arch_are_no_data() -> None:
    table = arch_domain.load(_committed_flavors()[0])
    assert table is not None
    assert table.lookup("not.a.real.intrinsic", "gfx950") is None
    known_key = table.keys()[0]
    assert table.lookup(known_key, "gfx-not-a-target") is None


def test_load_is_cached() -> None:
    """The lookup sits on the hot path of every intrinsic demand; re-reading and
    re-parsing a ~1000-cell column per demand would be felt."""
    flavor = _committed_flavors()[0]
    assert arch_domain.load(flavor) is arch_domain.load(flavor)


# --------------------------------------------------------------------------
# the chokepoint
# --------------------------------------------------------------------------


def _needs_intrin_writes() -> list[tuple[str, int]]:
    """Every function that assigns into ``self._needs_intrin``, with its line."""
    tree = ast.parse(LOWER_LLVM.read_text(encoding="utf-8"))
    enclosing: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                enclosing.setdefault(id(child), node.name)

    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "_needs_intrin"
            ):
                out.append((enclosing.get(id(node), "<module>"), node.lineno))
    return out


def test_needs_intrin_is_only_written_by_need() -> None:
    """A path that writes the map directly gets its declare emitted and skips
    the arch-domain lookup — the exact failure this lane exists to prevent, and
    one that leaves no trace in the emitted bytes."""
    offenders = [(fn, line) for fn, line in _needs_intrin_writes() if fn != "_need"]
    assert not offenders, (
        "these write _Lowerer._needs_intrin outside _need() and so bypass the "
        f"arch-domain check: {offenders}"
    )


# --------------------------------------------------------------------------
# no implicit target default, either engine
# --------------------------------------------------------------------------


def _a_kernel():
    """An ordinary gfx950 GEMM -- one the table has nothing negative to say
    about, so these tests isolate the arch argument rather than the warning."""
    from rocke.instances.common.gemm_universal import (
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    return build_universal_gemm(
        UniversalGemmSpec(
            name="rocke_arch_domain_lane",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4",
                scheduler="intrawave",
                epilogue="cshuffle",
            ),
        )
    )


def test_python_lowering_requires_an_explicit_arch() -> None:
    with pytest.raises(ValueError, match="explicit gfx target"):
        lower_kernel_to_llvm(_a_kernel())


def test_cpp_backend_resolution_has_no_default_arch() -> None:
    """The C++ side has no diagnostic channel to assert against from Python
    without a built engine, so assert on the source: a NULL arch must not fall
    through to a gfx target."""
    src = CPP_CORE.read_text(encoding="utf-8")
    assert 'arch == NULL || strcmp(arch, "gfx950")' not in src
    assert "if(arch == NULL)" in src


def test_explicit_arch_still_lowers() -> None:
    ir = lower_kernel_to_llvm(_a_kernel(), arch="gfx950")
    assert "target triple" in ir


# --------------------------------------------------------------------------
# the warning itself
# --------------------------------------------------------------------------


def _a_negative_cell() -> tuple[str, str, str]:
    """A measured (key, arch, flavor) the table calls arch_absent."""
    for flavor in _committed_flavors():
        table = arch_domain.load(flavor)
        assert table is not None
        # ArchDomain is not a mapping; .keys() is its accessor.
        for key in table.keys():  # noqa: SIM118
            for arch in ("gfx942", "gfx90a", "gfx1151", "gfx11-generic", "gfx950"):
                cell = table.lookup(key, arch)
                if cell is not None and arch_domain.is_negative(cell.status):
                    return key, arch, flavor
    pytest.skip("no arch_absent cell in any committed column")


def test_warning_carries_structured_fields() -> None:
    """The gate keys on (key, arch, flavor) rather than parsing prose, so these
    attributes are load-bearing, not decoration."""
    key, arch, flavor = _a_negative_cell()
    w = ArchDomainWarning(key, arch, flavor, "arch_absent", "some diagnostic")
    assert (w.key, w.arch, w.flavor, w.status) == (key, arch, flavor, "arch_absent")
    assert key in str(w) and arch in str(w) and "some diagnostic" in str(w)


def test_lane_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from rocke.core.lower_llvm import _ARCH_DOMAIN_ENV

    monkeypatch.setenv(_ARCH_DOMAIN_ENV, "off")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower_kernel_to_llvm(_a_kernel(), arch="gfx950")
    assert not [w for w in caught if isinstance(w.message, ArchDomainWarning)]


# --------------------------------------------------------------------------
# the lane matches its allowlist
# --------------------------------------------------------------------------


def test_corpus_matches_the_expected_warning_set() -> None:
    """Delegates to the gate rather than re-deriving the set, so the tool and
    the test cannot disagree about what the lane is allowed to say."""
    tool = PLATFORM / "tools" / "check_arch_domain.py"
    proc = subprocess.run([sys.executable, str(tool)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
