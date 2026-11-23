# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.engine_modules holds an installed tree to the engine pieces its
build made: it skips with the build's reasons when there are none, and fails
when a module does not import, the shared engine is not installed beside the
package or does not load, or the ``cpp`` backend does not lower through it.

The import failures block the module the way a lost install does, by making
``import rocke_engine`` raise ImportError in this process, and the lowering
failure hides the shared engine (``rocke.tests.engines.hide_shared_engine``).
The cases that need a piece to load run where it is built and skip elsewhere;
the shared engine counts only when it is installed beside the package, as in
the provider's test tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from rocke.core import native_engine
from rocke.tests.engine_modules import (
    SKIP_EXIT_CODE,
    EngineModulesError,
    EngineModulesMarker,
    import_engine_modules,
    load_shared_engine,
    lowering_engine,
    main,
    read_engine_modules_marker,
)
from rocke.tests.engines import hide_shared_engine
from rocke.tests.lanes import BINDING_LANE, SHARED_LIBRARY_LANE, skip_until_reason

_ENGINE = "rocke_engine"
_NO_PYBIND11 = "the build's Python had no pybind11"
_NO_SHARED = "the build had ROCKE_BUILD_SHARED_ENGINE off"
_NO_MODULES = {"modules": [], "absent_reason": _NO_PYBIND11}
_NO_SHARED_ENGINE = {"shared_engine": "", "shared_engine_absent_reason": _NO_SHARED}
needs_engine = pytest.mark.skipif(
    importlib.util.find_spec(_ENGINE) is None,
    reason=skip_until_reason(
        BINDING_LANE, reason="the rocke_engine module is not built in this environment"
    ),
)
_INSTALLED = native_engine.installed_shared_engine_path()
needs_installed_shared_engine = pytest.mark.skipif(
    _INSTALLED is None,
    reason=skip_until_reason(
        SHARED_LIBRARY_LANE,
        reason="no shared engine is installed beside the rocke package here",
    ),
)


@pytest.fixture
def default_backend(monkeypatch):
    """The backend CI runs with: ROCKE_BACKEND unset, no strict mode, and no
    shared engine named in the environment."""
    for variable in (
        "ROCKE_BACKEND",
        "ROCKE_CPP_STRICT",
        native_engine.SHARED_ENGINE_ENV,
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def engine_lost(monkeypatch):
    """``import rocke_engine`` raises ImportError, as in a tree that lost it."""
    monkeypatch.setitem(sys.modules, _ENGINE, None)


def _write_marker(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "rocke_engine_modules.json"
    path.write_text(json.dumps(document))
    return path


def test_the_marker_reads_back_what_the_build_made(tmp_path):
    path = _write_marker(
        tmp_path,
        {"modules": [_ENGINE, "rocke_library_engine"], "shared_engine": "librocke.so"},
    )
    assert read_engine_modules_marker(path) == EngineModulesMarker(
        (_ENGINE, "rocke_library_engine"), "", "librocke.so", ""
    )


def test_the_marker_reads_back_why_the_build_made_none(tmp_path):
    path = _write_marker(tmp_path, {**_NO_MODULES, **_NO_SHARED_ENGINE})
    assert read_engine_modules_marker(path) == EngineModulesMarker(
        (), _NO_PYBIND11, "", _NO_SHARED
    )


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
        EngineModulesMarker(modules, absent_reason, "", _NO_SHARED)


@pytest.mark.parametrize(
    "shared_engine, absent_reason",
    [("", ""), ("librocke.so", _NO_SHARED)],
    ids=["neither", "both"],
)
def test_a_marker_holds_a_shared_engine_or_a_reason(shared_engine, absent_reason):
    with pytest.raises(
        ValueError,
        match=r"EngineModulesMarker failed - .*expected either the shared engine "
        r"or the reason the build installs none beside the package",
    ):
        EngineModulesMarker((), _NO_PYBIND11, shared_engine, absent_reason)


def test_a_marker_names_the_shared_engine_by_file_name():
    with pytest.raises(
        ValueError,
        match=r"EngineModulesMarker failed - shared_engine='lib/librocke.so', "
        r"expected a file name",
    ):
        EngineModulesMarker((), _NO_PYBIND11, "lib/librocke.so", "")


def test_a_marker_names_python_modules():
    with pytest.raises(
        ValueError,
        match=r"EngineModulesMarker failed - module='rocke-engine', expected a "
        r"Python module name",
    ):
        EngineModulesMarker(("rocke-engine",), "", "", _NO_SHARED)


def test_a_missing_marker_names_its_path(tmp_path):
    path = tmp_path / "rocke_engine_modules.json"
    with pytest.raises(
        ValueError,
        match=rf"read_engine_modules_marker failed - path={path}, expected the "
        r"marker the rocKE configure installs",
    ):
        read_engine_modules_marker(path)


def test_a_build_without_engine_pieces_skips_with_its_reasons(tmp_path, capsys):
    path = _write_marker(tmp_path, {**_NO_MODULES, **_NO_SHARED_ENGINE})
    assert main([str(path)]) == SKIP_EXIT_CODE
    assert capsys.readouterr().out == (
        f"SKIP: no C++ engine to check: no Python modules ({_NO_PYBIND11}); "
        f"no shared engine ({_NO_SHARED})\n"
    )


def test_a_module_the_build_made_must_import(tmp_path, capsys, engine_lost):
    path = _write_marker(tmp_path, {"modules": [_ENGINE], **_NO_SHARED_ENGINE})
    assert main([str(path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith(f"FAIL: the build made {_ENGINE}, but this tree's Python")
    assert "cannot import it" in out


def test_import_reports_the_module_it_could_not_import(engine_lost):
    with pytest.raises(EngineModulesError, match=rf"the build made {_ENGINE}, but"):
        import_engine_modules(EngineModulesMarker((_ENGINE,), "", "", _NO_SHARED))


def test_a_shared_engine_the_build_made_must_be_installed_beside_the_package(
    tmp_path, capsys
):
    # A name no build makes: the library beside the package, if any, is not it.
    path = _write_marker(tmp_path, {**_NO_MODULES, "shared_engine": "librocke-lost.so"})
    assert main([str(path)]) == 1
    assert capsys.readouterr().out == (
        "FAIL: the build made librocke-lost.so, but it is not installed beside "
        "the rocke package\n"
    )


def test_a_cpp_backend_that_falls_back_fails(monkeypatch, default_backend):
    hide_shared_engine(monkeypatch)
    with pytest.raises(
        EngineModulesError, match=r"backend 'cpp' fell back to the Python engine: "
    ):
        lowering_engine()


def test_the_python_backend_is_reported_without_lowering(monkeypatch):
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    hide_shared_engine(monkeypatch)
    assert lowering_engine() == "python"


@needs_engine
def test_installed_modules_pass_without_a_shared_engine(
    tmp_path, capsys, default_backend
):
    path = _write_marker(tmp_path, {"modules": [_ENGINE], **_NO_SHARED_ENGINE})
    assert main([str(path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"{_ENGINE}: ")
    assert "(build id " in out[0]
    assert out[-1] == f"shared engine: none ({_NO_SHARED})"


@needs_installed_shared_engine
def test_an_installed_shared_engine_passes_and_records_its_backend(
    tmp_path, capsys, default_backend
):
    path = _write_marker(tmp_path, {**_NO_MODULES, "shared_engine": _INSTALLED.name})
    assert main([str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"Python modules: none ({_NO_PYBIND11})",
        f"shared engine: {_INSTALLED}",
        "lower_kernel_to_llvm lowers with the cpp engine",
    ]
    marker = read_engine_modules_marker(path)
    assert load_shared_engine(marker) == _INSTALLED
