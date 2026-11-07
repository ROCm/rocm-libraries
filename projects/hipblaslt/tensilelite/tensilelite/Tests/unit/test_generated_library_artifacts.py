# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Preserve runnable generated libraries in build-only/use-cache archives."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def helpers():
    path = Path(__file__).parents[1] / "common" / "artifact_helpers.py"
    spec = importlib.util.spec_from_file_location("library_artifact_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path",
    [
        "library/kernel.co",
        "library/gfx942/kernel.hsaco",
        "library/gfx950/TensileLibrary_lazy_gfx950.dat.zlib",
        "library/gfx950/solutions.dat",
        "library/logic.yaml",
        r"library\gfx942\kernel.co",
    ],
)
def test_generated_library_outputs_are_retained(helpers, path):
    assert helpers._is_use_cache_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "library/kernel.o",
        "library/kernel.s",
        "library/metadata.dat.zlib.tmp",
        "build/library/kernel.co",
        "library-other/kernel.co",
        "/library/kernel.co",
        "library/../kernel.co",
        "",
        "library",
    ],
)
def test_intermediates_and_unexpected_paths_are_excluded(helpers, path):
    assert not helpers._is_use_cache_file(path)


def test_generated_library_archive_roundtrip(helpers, tmp_path):
    output = tmp_path / "generated"
    library = output / "library" / "gfx950"
    library.mkdir(parents=True)
    (library / "kernel.co").write_bytes(b"code object")
    (library / "solutions.dat.zlib").write_bytes(b"solution metadata")
    (library / "kernel.o").write_bytes(b"intermediate")
    artifact = helpers.compress_output(str(output), dest_dir=str(tmp_path / "artifacts"))
    extracted = tmp_path / "extracted" / "generated"
    helpers.extract_artifact(artifact, str(extracted))
    assert (extracted / "library/gfx950/kernel.co").read_bytes() == b"code object"
    assert (extracted / "library/gfx950/solutions.dat.zlib").read_bytes() == b"solution metadata"
    assert not (extracted / "library/gfx950/kernel.o").exists()
