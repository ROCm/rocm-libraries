# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for Tensile.ClientWriter.clientLibraryFiles.

The LibraryClient step used to collect only `*.yaml` libraries, so with
LibraryFormat=msgpack it found nothing and ClientWriter.main crashed on
`libraryList[0]` with IndexError.
"""

import pytest

import Tensile.ClientWriter as CW
from Tensile.Common.GlobalParameters import globalParameters

pytestmark = pytest.mark.unit


def _touch(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"")
    return str(path)


@pytest.mark.parametrize("library_format, ext", [("msgpack", ".dat"), ("yaml", ".yaml")])
def test_selects_master_library_for_configured_format(tmp_path, monkeypatch, library_format, ext):
    monkeypatch.setitem(globalParameters, "LibraryFormat", library_format)
    arch_dir = CW.libraryDir(tmp_path, "gfx942")
    master = _touch(arch_dir, "TensileLibrary_gfx942" + ext)
    # Always written as msgpack, whatever LibraryFormat is; never the master.
    _touch(arch_dir, "TensileLiteLibrary_lazy_gfx942_Mapping.dat")
    code_object = _touch(arch_dir, "Kernels.so-000-gfx942.co")

    co_list, library_list = CW.clientLibraryFiles(tmp_path, ["gfx942"])

    assert library_list == [master]
    assert co_list == [code_object]


def test_unions_files_across_archs(tmp_path, monkeypatch):
    monkeypatch.setitem(globalParameters, "LibraryFormat", "msgpack")
    expected = [
        _touch(CW.libraryDir(tmp_path, arch), f"TensileLibrary_{arch}.dat")
        for arch in ("gfx942", "gfx950")
    ]

    _, library_list = CW.clientLibraryFiles(tmp_path, ["gfx942", "gfx950"])

    assert sorted(library_list) == sorted(expected)
