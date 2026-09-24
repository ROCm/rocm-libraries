# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for Tensile.ClientWriter.clientLibraryFiles.

The LibraryClient step used to collect only `*.yaml` libraries, so with
LibraryFormat=msgpack it found nothing and ClientWriter.main crashed on
`libraryList[0]` with IndexError.

Library files are written with LibraryIO.write so the on-disk names (e.g.
msgpack's `.dat.zlib`) match what TensileCreateLibrary produces.
"""

import os

import pytest

import Tensile.ClientWriter as CW
from Tensile import LibraryIO
from Tensile.Common.GlobalParameters import globalParameters

pytestmark = pytest.mark.unit

# A lazy-loading shard name; shards share the master's `TensileLibrary_` prefix.
SHARD = "TensileLibrary_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_{arch}"


def _write_library(directory, stem, library_format):
    directory.mkdir(parents=True, exist_ok=True)
    LibraryIO.write(str(directory / stem), {}, library_format)


def _build_arch(root, arch, library_format, lazy):
    arch_dir = CW.libraryDir(root, arch)
    master = f"TensileLibrary_lazy_{arch}" if lazy else f"TensileLibrary_{arch}"
    _write_library(arch_dir, master, library_format)
    if lazy:
        _write_library(arch_dir, SHARD.format(arch=arch), library_format)
        # Always written as msgpack, whatever LibraryFormat is; never the master.
        _write_library(arch_dir, f"TensileLiteLibrary_lazy_{arch}_Mapping", "msgpack")
    code_object = arch_dir / f"Kernels.so-000-{arch}.co"
    code_object.write_bytes(b"")
    ext = ".yaml" if library_format == "yaml" else ".dat"
    return str(arch_dir / (master + ext)), str(code_object)


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "not-lazy"])
@pytest.mark.parametrize("library_format", ["msgpack", "yaml"])
def test_selects_master_library(tmp_path, monkeypatch, library_format, lazy):
    monkeypatch.setitem(globalParameters, "LibraryFormat", library_format)
    monkeypatch.setitem(globalParameters, "LazyLibraryLoading", lazy)
    master, code_object = _build_arch(tmp_path, "gfx942", library_format, lazy)

    co_list, library_list = CW.clientLibraryFiles(tmp_path, ["gfx942"])

    assert library_list == [master]
    assert co_list == [code_object]


def test_msgpack_master_is_logical_dat_name(tmp_path, monkeypatch):
    """The client takes the `.dat` name and probes for `.dat.zlib` itself."""
    monkeypatch.setitem(globalParameters, "LibraryFormat", "msgpack")
    monkeypatch.setitem(globalParameters, "LazyLibraryLoading", True)
    master, _ = _build_arch(tmp_path, "gfx942", "msgpack", lazy=True)

    assert not os.path.exists(master)
    assert os.path.exists(master + ".zlib")
    assert CW.clientLibraryFiles(tmp_path, ["gfx942"])[1] == [master]


def test_unions_files_across_archs(tmp_path, monkeypatch):
    monkeypatch.setitem(globalParameters, "LibraryFormat", "msgpack")
    monkeypatch.setitem(globalParameters, "LazyLibraryLoading", True)
    expected = [_build_arch(tmp_path, arch, "msgpack", lazy=True)[0] for arch in ("gfx942", "gfx950")]

    _, library_list = CW.clientLibraryFiles(tmp_path, ["gfx942", "gfx950"])

    assert library_list == expected
