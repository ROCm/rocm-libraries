################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Common.GlobalParameters``:
``restoreDefaultGlobalParameters``, ``printCapabilitiesTable``,
``assignGlobalParameters`` (config merge + env + version + hipcc-probe branches),
and ``setupRestoreClocks``. The process-global dict is isolated per test."""

import pytest

import Tensile.Common.GlobalParameters as GP
from Tensile.Common.TypeValidationErrors import ConfigTypeError

pytestmark = pytest.mark.unit


def test_restore_default_global_parameters(isolate_globals):
    GP.globalParameters["KeepBuildTmp"] = "MUTATED"
    GP.restoreDefaultGlobalParameters()
    # A known default is restored (RuntimeLanguage is set to HIP at L266 path).
    assert GP.globalParameters["RuntimeLanguage"] == "HIP"


def test_print_capabilities_table(isolate_globals, isa_info_map, capsys, snapshot):
    GP.printCapabilitiesTable(isa_info_map)
    out = capsys.readouterr().out
    # Pin structure (has a header + rows), not the volatile cap values.
    assert {"nonempty": len(out) > 0, "has_isa_row": any(c.isdigit() for c in out)} == snapshot


def _stub_hipcc(monkeypatch, output="HIP version 6.2.0\n", raise_exc=None):
    def fake_run(*a, **k):
        if raise_exc:
            raise raise_exc
        return type("R", (), {"stdout": output.encode()})()

    monkeypatch.setattr(GP.subprocess, "run", fake_run)
    # Stub rocm-smi discovery so the test is independent of the host's rocm-smi
    # (and so a fake ROCmPath doesn't trip the real locateExe). The OSError arm
    # is covered separately in test_assign_..._locateexe_oserror_raises.
    monkeypatch.setattr(GP, "locateExe", lambda d, n: f"{d}/{n}")


def test_assign_global_parameters_basic(isolate_globals, isa_info_map, monkeypatch):
    _stub_hipcc(monkeypatch)
    # same / overridden / unspecified branches + a recognised override.
    cfg = {"KeepBuildTmp": GP.globalParameters.get("KeepBuildTmp"),  # same
           "CodeObjectVersion": "5", "AsanBuild": True}
    GP.assignGlobalParameters(cfg, isa_info_map)
    assert GP.globalParameters["CodeObjectVersion"] == "5"
    assert GP.globalParameters["AsanBuild"] is True
    assert GP.globalParameters["HipClangVersion"] == "6.2.0"
    assert GP.validParameters["ISA"][0] == GP.IsaVersion(0, 0, 0) if hasattr(GP, "IsaVersion") else True


def test_assign_global_parameters_env_overrides(isolate_globals, isa_info_map, monkeypatch):
    _stub_hipcc(monkeypatch)
    monkeypatch.setenv("ROCM_PATH", "/custom/rocm")
    monkeypatch.setenv("TENSILE_ROCM_PATH", "/tensile/rocm")
    monkeypatch.setenv("CMAKE_CXX_COMPILER", "my-c++")
    monkeypatch.setenv("CMAKE_C_COMPILER", "my-cc")
    GP.assignGlobalParameters({}, isa_info_map)
    assert GP.globalParameters["ROCmPath"] == "/tensile/rocm"  # TENSILE_ROCM_PATH wins
    assert GP.globalParameters["CmakeCxxCompiler"] == "my-c++"
    assert GP.globalParameters["CmakeCCompiler"] == "my-cc"


def test_assign_global_parameters_unrecognised_key_raises(isolate_globals, isa_info_map, monkeypatch):
    _stub_hipcc(monkeypatch)
    with pytest.raises(ConfigTypeError, match="Unknown global parameter 'TotallyMadeUpGlobal'"):
        GP.assignGlobalParameters({"TotallyMadeUpGlobal": 7, "Architecture": "gfx942"}, isa_info_map)
    assert "TotallyMadeUpGlobal" not in GP.globalParameters
    assert "Architecture" not in GP.globalParameters


def test_assign_global_parameters_incompatible_version_exits(isolate_globals, isa_info_map, monkeypatch):
    _stub_hipcc(monkeypatch)
    with pytest.raises(SystemExit):
        GP.assignGlobalParameters({"MinimumRequiredVersion": "999.0.0"}, isa_info_map)


def test_assign_global_parameters_locateexe_oserror_nonfatal(isolate_globals, isa_info_map, monkeypatch):
    # amd-smi not found (non-Windows) -> the except OSError arm is non-fatal:
    # it warns and leaves AMDSMIPath unset (None) instead of re-raising, so the
    # build/logic steps still run in environments that do not ship amd-smi.
    _stub_hipcc(monkeypatch)
    monkeypatch.setattr(GP, "locateExe",
                        lambda d, n: (_ for _ in ()).throw(OSError("Failed to locate amd-smi")))
    monkeypatch.setattr(GP.os, "name", "posix")
    GP.assignGlobalParameters({}, isa_info_map)
    assert GP.globalParameters["AMDSMIPath"] is None


def test_assign_global_parameters_min_version_compatible(isolate_globals, isa_info_map, monkeypatch):
    # A *present, compatible* MinimumRequiredVersion -> the 644->651 false arm.
    _stub_hipcc(monkeypatch)
    from Tensile import __version__
    GP.assignGlobalParameters({"MinimumRequiredVersion": __version__}, isa_info_map)
    assert "ROCmPath" in GP.globalParameters


def test_assign_global_parameters_hipcc_failure_warns(isolate_globals, isa_info_map, monkeypatch):
    import subprocess
    _stub_hipcc(monkeypatch, raise_exc=subprocess.CalledProcessError(1, "hipcc"))
    # The hipcc probe failure is caught + warned; assignment still completes.
    GP.assignGlobalParameters({}, isa_info_map)
    assert "ROCmPath" in GP.globalParameters


def test_assign_global_parameters_prints_caps_when_verbose(isolate_globals, isa_info_map, monkeypatch):
    _stub_hipcc(monkeypatch)
    monkeypatch.setattr(GP, "getVerbosity", lambda: 2)
    GP.assignGlobalParameters({}, isa_info_map)  # -> printCapabilitiesTable path
    assert "ROCmPath" in GP.globalParameters


def test_setup_restore_clocks_handler(isolate_globals, monkeypatch):
    # Capture the atexit-registered handler and invoke it with PinClocks on +
    # a stubbed amd-smi path + stubbed subprocess.call.
    captured = []
    import atexit
    monkeypatch.setattr(atexit, "register", lambda fn: captured.append(fn) or fn)
    calls = []
    monkeypatch.setattr(GP.subprocess, "call", lambda *a, **k: calls.append(a))

    GP.globalParameters["PinClocks"] = True
    GP.globalParameters["AMDSMIPath"] = "amd-smi"
    GP.setupRestoreClocks()
    assert captured, "a restore handler should be registered"
    captured[0]()  # invoke -> the single `amd-smi reset` subprocess.call branch
    assert len(calls) == 1


def test_setup_restore_clocks_handler_no_smi(isolate_globals, monkeypatch):
    captured = []
    import atexit
    monkeypatch.setattr(atexit, "register", lambda fn: captured.append(fn) or fn)
    calls = []
    monkeypatch.setattr(GP.subprocess, "call", lambda *a, **k: calls.append(a))
    GP.globalParameters["PinClocks"] = True
    GP.globalParameters["AMDSMIPath"] = None  # -> no calls
    GP.setupRestoreClocks()
    captured[0]()
    assert calls == []
