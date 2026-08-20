################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.CustomKernels``: the custom-kernel
.s-file embedded-YAML parsing and validation, driven over crafted .s files in
a tmp directory."""

import contextlib

import pytest

from Tensile.CustomKernels import (
    isCustomKernelConfig,
    getCustomKernelFilepath,
    getAllCustomKernelNames,
    getCustomKernelContents,
    _readEmbeddedYaml,
    readCustomKernelConfig,
    getCustomKernelConfig,
)

pytestmark = pytest.mark.unit

_VALID_S = """\
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
...
s_nop 0
s_endpgm
"""

# getCustomKernelConfig's no-explicit-CustomKernel path auto-infers one from
# amdhsa.kernels[0].args (see _buildCustomKernelFromMetadata), so -- unlike the
# other fixtures in this file -- it needs a real amdhsa.kernels entry, mirroring
# the write_kernel() helper in Tests/unit/test_CustomKernelMetadata.py.
_VALID_S_WITH_KERNEL_META = """\
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.kernels:
  - .name: k
    .max_flat_workgroup_size: 256
    .args:
      - .name: D
        .size: 8
        .offset: 0
        .value_kind: global_buffer
...
s_nop 0
s_endpgm
"""


def _write(d, name, contents):
    p = d / (name + ".s")
    p.write_text(contents)
    return p


@contextlib.contextmanager
def _isolate_valid_parameters():
    # getCustomKernelConfig does `validParameters.update(newMIValidParameters)`,
    # a permanent global mutation. Snapshot + restore so the roster other suites
    # (ValidParameters/Naming/SolutionClass) snapshot is left untouched.
    from Tensile.Common.ValidParameters import validParameters
    saved = dict(validParameters)
    try:
        yield
    finally:
        validParameters.clear()
        validParameters.update(saved)


def test_is_custom_kernel_config():
    assert isCustomKernelConfig({"CustomKernelName": "k"})
    assert not isCustomKernelConfig({})
    assert not isCustomKernelConfig({"CustomKernelName": ""})


def test_get_custom_kernel_filepath(snapshot):
    assert getCustomKernelFilepath("mykern", directory="/some/dir").endswith("/some/dir/mykern.s")


def test_get_all_custom_kernel_names(tmp_path, snapshot):
    _write(tmp_path, "alpha", _VALID_S)
    _write(tmp_path, "beta", _VALID_S)
    (tmp_path / "notakernel.txt").write_text("x")
    assert sorted(getAllCustomKernelNames(directory=str(tmp_path))) == snapshot


def test_get_custom_kernel_contents_ok(tmp_path):
    _write(tmp_path, "k", _VALID_S)
    assert "s_endpgm" in getCustomKernelContents("k", directory=str(tmp_path))


def test_get_custom_kernel_contents_missing_raises(tmp_path):
    with pytest.raises(RuntimeError):
        getCustomKernelContents("nope", directory=str(tmp_path))


def test_read_embedded_yaml_parses_custom_config(tmp_path, snapshot):
    # getCustomKernelConfigAndAssembly (raw '---'/'...' line-splitter, returning
    # a (config_text, assembly_text) tuple) was replaced by _readEmbeddedYaml
    # (parses the .amdgpu_metadata YAML block and returns the parsed dict) as
    # part of Gemm-From-Anywhere; see adr/0002-custom-kernels-embedded-yaml-parsing.md.
    _write(tmp_path, "k", _VALID_S)
    assert _readEmbeddedYaml("k", directory=str(tmp_path)) == snapshot


def test_read_custom_kernel_config_ok(tmp_path, snapshot):
    _write(tmp_path, "k", _VALID_S)
    assert readCustomKernelConfig("k", directory=str(tmp_path)) == snapshot


def test_read_custom_kernel_config_bad_yaml_raises(tmp_path):
    _write(tmp_path, "bad", "---\ncustom.config:\n  a: : : :\n...\ns_nop 0\n")
    with pytest.raises(RuntimeError):
        readCustomKernelConfig("bad", directory=str(tmp_path))


def test_get_custom_kernel_config_ok(tmp_path, snapshot):
    _write(tmp_path, "k", _VALID_S_WITH_KERNEL_META)
    with _isolate_valid_parameters():
        cfg = getCustomKernelConfig("k", {"Extra": 9}, directory=str(tmp_path))
    assert {
        "KernelLanguage": cfg["KernelLanguage"],
        "CustomKernelName": cfg["CustomKernelName"],
        "isp": cfg["InternalSupportParams"],
    } == snapshot


def test_get_custom_kernel_config_missing_isp_raises(tmp_path):
    _write(tmp_path, "noisp", "---\ncustom.config:\n  Foo: 1\n...\ns_nop 0\n")
    with pytest.raises(RuntimeError):
        getCustomKernelConfig("noisp", {}, directory=str(tmp_path))


def test_get_custom_kernel_config_missing_kernargs_raises(tmp_path):
    _write(tmp_path, "nokav",
           "---\ncustom.config:\n  InternalSupportParams:\n    Other: 1\n...\ns_nop 0\n")
    with pytest.raises(RuntimeError):
        getCustomKernelConfig("nokav", {}, directory=str(tmp_path))
