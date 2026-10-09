################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Per-base layout coverage for CreateBenchmarkClientParametersForSizes.

The helper resolves the per-arch library subdirectory used by the
benchmark client and synthesizes the `libraryFile` path that
writeClientConfigIni now requires. The arch is resolved by prescription
from the value the caller already holds:

  1. `archs` provided → `libraryDir(libraryRootPath, archs[0])` strips
     target features at the first colon, so cooked variants like
     `gfx942:sramecc+:xnack+` collapse to `library/gfx942/`.

  2. `archs=None` but `gfxName` set → `libraryDir(libraryRootPath, gfxName)`.

  3. neither → RuntimeError (no filesystem guessing).

Branches 1-2 compose the libraryFile as
`<resolved_per_base_dir>/TensileLibrary.{dat,yaml}` and pass it as the
`libraryFile` keyword arg to writeClientConfigIni.
"""

import json
import os
from unittest.mock import patch

import pytest

from Tensile.ClientWriter import CreateBenchmarkClientParametersForSizes

pytestmark = pytest.mark.unit


def _stub_problem_type_dict():
    # Minimal ProblemType state. CreateBenchmarkClientParametersForSizes
    # passes it through ContractionsProblemType.FromOriginalState; the
    # exact shape is not asserted on by this test, only that the helper
    # got far enough to compose libraryFile and dispatch to
    # writeClientConfigIni. The fixture below patches the dispatch out.
    return {"OperationType": "GEMM", "DataType": "S"}


def _make_library_tree(root, base_arch, library_format):
    """Build a per-base library tree under `root` so the helper's
    os.listdir / file-glob calls find something realistic."""
    arch_dir = root / "library" / base_arch
    arch_dir.mkdir(parents=True)
    ext = ".yaml" if library_format == "yaml" else ".dat"
    (arch_dir / f"TensileLibrary{ext}").write_text("")
    # A couple of .co files so codeObjectFiles filtering has substrate.
    (arch_dir / f"TensileLibrary_Cijk_gfx942.co").write_text("")
    (arch_dir / f"TensileLibrary_lazy_gfx942.co").write_text("")
    return arch_dir


@pytest.fixture
def captured_call(monkeypatch):
    """Patch writeClientConfigIni so we can inspect the libraryFile arg
    without going through the rest of the client-config emission path."""
    captured = {}

    def _capture(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("Tensile.ClientWriter.writeClientConfigIni", _capture)

    # CreateBenchmarkClientParametersForSizes now runs the passed problemTypeDict
    # through ContractionsProblemType.FromOriginalState, which requires a full
    # problem-type state (TotalIndices, IndicesSummation, ...). These tests cover
    # only per-base libraryFile routing (see _stub_problem_type_dict's note), and
    # the resulting problemType is merely forwarded to the patched
    # writeClientConfigIni, so stub the parse out with a sentinel.
    monkeypatch.setattr(
        "Tensile.ClientWriter.ContractionsProblemType.FromOriginalState",
        lambda d: object(),
    )
    return captured


def _global_params(monkeypatch, library_format="msgpack"):
    """Stub globalParameters['LibraryFormat'] which gates the file extension."""
    from Tensile.ClientWriter import globalParameters
    monkeypatch.setitem(globalParameters, "LibraryFormat", library_format)


# ---------------------------------------------------------------------------
# Branch A: archs explicitly provided
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cooked", ["gfx942", "gfx942:xnack+", "gfx942:sramecc+:xnack+"])
def test_archs_provided_uses_libraryDir_per_base(tmp_path, captured_call, monkeypatch, cooked):
    """archs=[cooked] must collapse to library/<base>/ via libraryDir,
    regardless of which cooked variant the caller passed."""
    _global_params(monkeypatch)
    _make_library_tree(tmp_path, "gfx942", "msgpack")

    CreateBenchmarkClientParametersForSizes(
        libraryRootPath=str(tmp_path),
        problemSizes=None,
        dataFilePath=str(tmp_path / "bench.csv"),
        configFile=str(tmp_path / "ClientParameters.ini"),
        deviceId=0,
        gfxName="gfx942",
        problemTypeDict=_stub_problem_type_dict(),
        archs=[cooked],
    )

    # writeClientConfigIni receives libraryFile as a keyword arg; verify it
    # points at the per-base path with the .dat extension (msgpack format).
    libraryFile = captured_call["kwargs"]["libraryFile"]
    assert libraryFile == os.path.join(str(tmp_path), "library", "gfx942", "TensileLibrary.dat")


def test_archs_provided_yaml_extension(tmp_path, captured_call, monkeypatch):
    """LibraryFormat=yaml must flip the extension to .yaml; the per-base
    directory routing stays the same."""
    _global_params(monkeypatch, library_format="yaml")
    _make_library_tree(tmp_path, "gfx942", "yaml")

    CreateBenchmarkClientParametersForSizes(
        libraryRootPath=str(tmp_path),
        problemSizes=None,
        dataFilePath="ignored",
        configFile="ignored",
        deviceId=0,
        gfxName="gfx942",
        problemTypeDict=_stub_problem_type_dict(),
        archs=["gfx942"],
    )

    libraryFile = captured_call["kwargs"]["libraryFile"]
    assert libraryFile.endswith(os.path.join("library", "gfx942", "TensileLibrary.yaml"))


# ---------------------------------------------------------------------------
# Branch B: archs=None resolves from gfxName (else raises)
# ---------------------------------------------------------------------------
def test_archs_none_uses_gfxName(tmp_path, captured_call, monkeypatch):
    """When the caller omits archs but provides gfxName, resolve the per-base
    dir from gfxName via libraryDir (the prescription) -- no filesystem scan."""
    _global_params(monkeypatch)
    _make_library_tree(tmp_path, "gfx950", "msgpack")

    CreateBenchmarkClientParametersForSizes(
        libraryRootPath=str(tmp_path),
        problemSizes=None,
        dataFilePath="ignored",
        configFile="ignored",
        deviceId=0,
        gfxName="gfx950",
        problemTypeDict=_stub_problem_type_dict(),
        archs=None,
    )

    libraryFile = captured_call["kwargs"]["libraryFile"]
    assert libraryFile == os.path.join(str(tmp_path), "library", "gfx950", "TensileLibrary.dat")


def test_no_arch_raises(tmp_path, monkeypatch):
    """With neither archs nor a non-empty gfxName, the helper fails loudly
    rather than guessing the arch from the filesystem."""
    _global_params(monkeypatch)
    (tmp_path / "library" / "gfx942").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="arch is required"):
        CreateBenchmarkClientParametersForSizes(
            libraryRootPath=str(tmp_path),
            problemSizes=None,
            dataFilePath="ignored",
            configFile="ignored",
            deviceId=0,
            gfxName="",
            problemTypeDict=_stub_problem_type_dict(),
            archs=None,
        )


# ---------------------------------------------------------------------------
# Smoke: codeObjectFiles is filtered to .co under the per-base dir
# ---------------------------------------------------------------------------
def test_codeObjectFiles_filtered_from_per_base(tmp_path, captured_call, monkeypatch):
    """The .co filter must pick up code-object files in the per-base
    subdir (not the library root) so the generated config points at
    real binaries."""
    _global_params(monkeypatch)
    arch_dir = _make_library_tree(tmp_path, "gfx942", "msgpack")
    # Add a non-.co file that should be excluded.
    (arch_dir / "metadata.yaml").write_text("ProblemType: {}\n")

    CreateBenchmarkClientParametersForSizes(
        libraryRootPath=str(tmp_path),
        problemSizes=None,
        dataFilePath="ignored",
        configFile="ignored",
        deviceId=0,
        gfxName="gfx942",
        problemTypeDict=_stub_problem_type_dict(),
        archs=["gfx942"],
    )

    # codeObjectFiles is positional arg 9 (0-indexed) in writeClientConfigIni.
    coFiles = captured_call["args"][8]
    assert all(f.endswith(".co") for f in coFiles)
    assert all("gfx942" in f for f in coFiles)
    assert len(coFiles) == 2
