# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from Tensile.Toolchain import Source


pytestmark = pytest.mark.unit


def test_uncached_helper_build_compiles_fresh_artifacts(tmp_path, monkeypatch):
    """A single-solution build must bypass cache setup and rebuild its helpers."""
    def forbidden_cache():
        pytest.fail("An uncached helper build constructed the cache")

    monkeypatch.setattr(Source, "HelperKernelCache", forbidden_cache)
    source = tmp_path / "Kernels.cpp"
    include = tmp_path / "include"
    include.mkdir()
    compiler_calls = []

    def compiler(include_dir, architectures, kernel_path, object_path):
        compiler_calls.append((include_dir, architectures, kernel_path))
        Path(object_path).write_bytes(Path(kernel_path).read_bytes())

    class Bundler:
        def targets(self, object_path):
            return ["hipv4-amdgcn-amd-amdhsa--gfx942"]

        def __call__(self, target, object_path, output_path):
            Path(output_path).write_bytes(b"helper:" + Path(object_path).read_bytes())

    for contents in (b"first source", b"changed source"):
        source.write_bytes(contents)
        artifacts = Source.buildSourceCodeObjectFiles(
            compiler, Bundler(), tmp_path / "library", tmp_path / "objects",
            include, source, ["gfx942"], useCache=False,
        )
        assert artifacts == [str(tmp_path / "library/gfx942/Kernels.so-000-gfx942.hsaco")]
        assert Path(artifacts[0]).read_bytes() == b"helper:" + contents

    assert compiler_calls == [(str(include), ["gfx942"], str(source))] * 2
