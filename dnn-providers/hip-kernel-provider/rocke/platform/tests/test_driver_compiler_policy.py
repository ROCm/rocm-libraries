# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Automatic selection requires evidence while explicit emission stays offline."""

import sys
from unittest.mock import Mock

import pytest

from rocke.core import lower_llvm
from rocke.portable_ir.drivers import gpu_replay, parity_matrix, roll_hsaco_parity
from rocke.runtime import comgr


@pytest.fixture
def hsaco_driver(monkeypatch):
    pytest.importorskip("resource")
    from rocke.portable_ir.drivers import hsaco_parity
    from rocke.portable_ir.src import online

    # main writes this variable directly; register its original value for teardown.
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm22")
    monkeypatch.setattr(online, "load", lambda: None)
    run = Mock(
        return_value=(
            dict.fromkeys(("cmp", "det", "eng", "rec", "uncompilable", "refused"), 0),
            [],
            {},
            {},
        )
    )
    monkeypatch.setattr(hsaco_parity, "run", run)

    def select(flavor="auto"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["hsaco_parity", "--flavor", flavor, "--cap-gb", "1", "--no-baseline"],
        )
        assert hsaco_parity.main() == 0
        return run.call_args.args[1]

    return select


@pytest.mark.parametrize("major", [None, 0, 20, 23])
@pytest.mark.parametrize("override", [None, "llvm22"])
def test_auto_selection_requires_evidence_and_preserves_override_precedence(
    monkeypatch, hsaco_driver, major, override
):
    # None: no library; 0: loaded library with unknown version.
    info = (
        None
        if major is None
        else comgr.CompilerInfo(
            (major, 0, 0) if major else None, "fixture", None, None, None
        )
    )
    monkeypatch.setattr(comgr, "loaded_compiler_info", lambda: info)
    if override is None:
        monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    else:
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", override)
    detected = f"llvm{major}" if major else None

    def check(select, expected):
        if expected is None:
            with pytest.raises(comgr.ComgrError, match="Cannot determine LLVM flavor"):
                select()
        else:
            assert select() == expected

    check(lower_llvm._resolve_llvm_flavor, override or detected)
    check(lambda: gpu_replay._resolve_flavor("auto"), detected or override)
    check(lambda: parity_matrix._auto_flavor()[0], detected)
    check(roll_hsaco_parity._flavor, detected)
    check(hsaco_driver, detected)


def test_explicit_driver_flavor_does_not_query(monkeypatch, hsaco_driver):
    query = Mock(side_effect=AssertionError("explicit flavor queried compiler"))
    monkeypatch.setattr(comgr, "loaded_compiler_info", query)
    assert gpu_replay._resolve_flavor("llvm20") == "llvm20"
    assert hsaco_driver("llvm20") == "llvm20"
    query.assert_not_called()


def test_query_errors_propagate_instead_of_selecting_a_default(monkeypatch):
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    monkeypatch.setattr(
        comgr, "loaded_compiler_info", Mock(side_effect=comgr.ComgrError("unavailable"))
    )
    for select in (
        lower_llvm._resolve_llvm_flavor,
        lambda: gpu_replay._resolve_flavor("auto"),
        parity_matrix._auto_flavor,
        roll_hsaco_parity._flavor,
        lambda: comgr._assert_ir_flavor_matches_lib("define void @k() {}"),
    ):
        with pytest.raises(comgr.ComgrError, match="unavailable"):
            select()


def test_legacy_package_metadata_retains_system_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(comgr, "resolved_lib_path", lambda: str(tmp_path / "comgr.so"))
    monkeypatch.setattr(
        comgr,
        "_read_rocm_version_file",
        lambda path: (7, 1) if path == "/opt/rocm/.info/version" else None,
    )
    assert comgr.resolved_lib_rocm_version() == (7, 1)
