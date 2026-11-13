# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark callers retain their settings after an in-process library rebuild."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tensilelite import ClientWriter, GenerateSummations
from tensilelite.Common import GlobalParameters, Utilities
from tensilelite.Common.Types import IsaVersion
from tensilelite.Common.ValidParameters import validParameters

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("caller", ["client", "summations"])
@pytest.mark.parametrize("failure", [None, RuntimeError, OSError, SystemExit])
def test_rebuild_restores_caller_state(monkeypatch, tmp_path, caller, failure):
    parameters = GlobalParameters.globalParameters
    monkeypatch.setitem(parameters, "StinkyTofuModuleOptions", {"nested": [1]})
    monkeypatch.setitem(parameters, "StinkyTofuArchName", "gfx1250-strict")
    monkeypatch.setitem(parameters, "NumBenchmarks", 7)
    monkeypatch.setattr(Utilities, "_verbosity", 0)
    original_isas = [IsaVersion(12, 5, 0)]
    monkeypatch.setitem(validParameters, "ISA", original_isas)
    expected = deepcopy(parameters)

    def rebuild(argv):
        parameters["StinkyTofuModuleOptions"]["nested"].append(2)
        parameters["StinkyTofuArchName"] = ""
        parameters["CpuThreads"] = 64
        parameters["rebuild_only_setting"] = True
        del parameters["NumBenchmarks"]
        original_isas.append(IsaVersion(9, 4, 2))
        validParameters["ISA"] = [IsaVersion(0, 0, 0)]
        Utilities.setVerbosity(2)
        if failure is not None:
            raise failure("rebuild failed")

    if caller == "client":
        monkeypatch.setattr(ClientWriter, "createLibrary", rebuild)
        monkeypatch.setattr(ClientWriter, "copyStaticFiles", lambda path: None)
        monkeypatch.setattr(ClientWriter, "getBuildClientLibraryArguments", lambda *args: [])
        reached_client = []

        def client_files(*args):
            # The client resumes under its own settings, before later setup.
            assert parameters == expected
            reached_client.append(True)
            return [], []

        monkeypatch.setattr(ClientWriter, "clientLibraryFiles", client_files)
        monkeypatch.setattr(ClientWriter, "runClient", lambda *args, **kwargs: 0)

        def invoke():
            return ClientWriter.main(
                [], SimpleNamespace(path="/fake/compiler"), "/fake/compiler", {},
                tmp_path, 0, "gfx1250-strict",
            )
    else:
        monkeypatch.setattr(GenerateSummations, "createLibrary", rebuild)

        def invoke():
            return GenerateSummations.createLibraryForBenchmark(
                tmp_path / "logic", tmp_path / "library", tmp_path, "gfx1250-strict"
            )

    if failure is None:
        invoke()
        if caller == "client":
            assert reached_client == [True]
    else:
        expected_error = SystemExit if caller == "summations" else failure
        with pytest.raises(expected_error):
            invoke()

    assert GlobalParameters.globalParameters is parameters
    assert parameters == expected
    assert Utilities.getVerbosity() == 0
    assert validParameters["ISA"] is original_isas
    assert original_isas == [IsaVersion(12, 5, 0)]
