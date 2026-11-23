# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.engine_modules holds an installed tree to the engine modules its
build made: it skips with the build's reason when there are none, and fails
when one does not import or the ``cpp`` backend does not lower through it.

The import failures block the module the way a lost install does, by making
``import rocke_engine`` raise ImportError in this process. The cases that need
the module to import run where the binding is built and skip elsewhere.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from rocke.tests.engine_modules import (
    SKIP_EXIT_CODE,
    EngineModulesError,
    EngineModulesMarker,
    import_engine_modules,
    lowering_engine,
    main,
    read_engine_modules_marker,
)
from rocke.tests.lanes import BINDING_LANE, skip_until_reason

_ENGINE = "rocke_engine"
_NO_PYBIND11 = "the build's Python had no pybind11"
needs_engine = pytest.mark.skipif(
    importlib.util.find_spec(_ENGINE) is None,
    reason=skip_until_reason(
        BINDING_LANE, reason="the rocke_engine module is not built in this environment"
    ),
)


@pytest.fixture
def default_backend(monkeypatch):
    """The backend CI runs with: ROCKE_BACKEND unset, no strict mode."""
    for variable in ("ROCKE_BACKEND", "ROCKE_CPP_STRICT"):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def engine_lost(monkeypatch):
    """``import rocke_engine`` raises ImportError, as in a tree that lost it."""
    monkeypatch.setitem(sys.modules, _ENGINE, None)


def _write_marker(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "rocke_engine_modules.json"
    path.write_text(json.dumps(document))
    return path


def test_the_marker_reads_back_the_modules_the_build_made(tmp_path):
    path = _write_marker(tmp_path, {"modules": [_ENGINE, "rocke_library_engine"]})
    assert read_engine_modules_marker(path) == EngineModulesMarker(
        (_ENGINE, "rocke_library_engine"), ""
    )


def test_the_marker_reads_back_why_the_build_made_none(tmp_path):
    path = _write_marker(tmp_path, {"modules": [], "absent_reason": _NO_PYBIND11})
    assert read_engine_modules_marker(path) == EngineModulesMarker((), _NO_PYBIND11)


@pytest.mark.parametrize(
    "modules, absent_reason",
    [((), ""), ((_ENGINE,), _NO_PYBIND11)],
    ids=["neither", "both"],
)
def test_a_marker_holds_modules_or_a_reason(modules, absent_reason):
    with pytest.raises(
        ValueError,
        match=r"EngineModulesMarker failed - .*expected either modules or the "
        r"reason the build made none",
    ):
        EngineModulesMarker(modules, absent_reason)


def test_a_marker_names_python_modules():
    with pytest.raises(
        ValueError,
        match=r"EngineModulesMarker failed - module='rocke-engine', expected a "
        r"Python module name",
    ):
        EngineModulesMarker(("rocke-engine",), "")


def test_a_missing_marker_names_its_path(tmp_path):
    path = tmp_path / "rocke_engine_modules.json"
    with pytest.raises(
        ValueError,
        match=rf"read_engine_modules_marker failed - path={path}, expected the "
        r"marker the rocKE configure installs",
    ):
        read_engine_modules_marker(path)


def test_a_build_without_modules_skips_with_its_reason(tmp_path, capsys):
    path = _write_marker(tmp_path, {"modules": [], "absent_reason": _NO_PYBIND11})
    assert main([str(path)]) == SKIP_EXIT_CODE
    assert capsys.readouterr().out == (
        f"SKIP: no C++ engine Python modules to check: {_NO_PYBIND11}\n"
    )


def test_a_module_the_build_made_must_import(tmp_path, capsys, engine_lost):
    path = _write_marker(tmp_path, {"modules": [_ENGINE]})
    assert main([str(path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith(f"FAIL: the build made {_ENGINE}, but this tree's Python")
    assert "cannot import it" in out


def test_import_reports_the_module_it_could_not_import(engine_lost):
    with pytest.raises(EngineModulesError, match=rf"the build made {_ENGINE}, but"):
        import_engine_modules(EngineModulesMarker((_ENGINE,), ""))


def test_a_cpp_backend_that_falls_back_fails(default_backend, engine_lost):
    with pytest.raises(
        EngineModulesError, match=r"backend 'cpp' fell back to the Python engine: "
    ):
        lowering_engine()


def test_the_python_backend_is_reported_without_lowering(monkeypatch, engine_lost):
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    assert lowering_engine() == "python"


@needs_engine
def test_an_installed_engine_passes_and_records_its_backend(
    tmp_path, capsys, default_backend
):
    path = _write_marker(tmp_path, {"modules": [_ENGINE]})
    assert main([str(path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"{_ENGINE}: ")
    assert "(build id " in out[0]
    assert out[-1] == "lower_kernel_to_llvm lowers with the cpp engine"
