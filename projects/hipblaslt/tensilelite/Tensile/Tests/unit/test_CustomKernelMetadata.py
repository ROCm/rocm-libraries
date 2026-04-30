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

from textwrap import dedent, indent

import pytest

from Tensile.AddCustomConfig import (
    _parse_tensile_yaml,
    _read_asm_file,
    build_custom_config_yaml,
    inject_custom_config,
)
from Tensile.CustomKernels import getCustomKernelConfig, validateCustomKernelMetadata
from Tensile.Toolchain.Assembly import validateCustomKernelMetadataAtBuild
from Tensile.ValidateMetadata import validate_all


def write_kernel(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    config_block = indent(dedent(config).strip(), "  ")
    path.write_text(
        ".amdgpu_metadata\n"
        "---\n"
        "custom.config:\n"
        f"{config_block}\n"
        "amdhsa.kernels:\n"
        f"  - .name: {path.stem}\n"
        "    .max_flat_workgroup_size: 64\n"
        "    .args:\n"
        "      - .name: D\n"
        "        .size: 8\n"
        "        .offset: 0\n"
        "        .value_kind: global_buffer\n"
        "...\n"
    )


def test_validate_external_requires_full_custom_kernel_fields(tmp_path):
    write_kernel(tmp_path / "external.s", """\
          Source:
            Origin: test
          Version: 1.0.0
          Features: {}
          InternalSupportParams:
            KernArgsVersion: 0
          ProblemType: {}
          MatrixInstruction: [16, 16, 16, 1]
          CustomKernel:
            args: []
            macrotile: [16, 16, 16]
            threads: [64, 1, 1]
        """)

    valid, msg = validateCustomKernelMetadata("external", str(tmp_path))

    assert not valid
    assert "CustomKernel.grid" in msg


def test_validate_tensile_kernel_only_needs_kern_args_version(tmp_path):
    """Tensile-generated kernels carry only InternalSupportParams.KernArgsVersion;
    ProblemType and tuning state come from the consuming logic file or test YAML."""
    write_kernel(tmp_path / "tensile.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
        """)

    valid, msg = validateCustomKernelMetadata("tensile", str(tmp_path))

    assert valid, msg


def test_validate_tensile_kernel_missing_kern_args_version_fails(tmp_path):
    write_kernel(tmp_path / "tensile.s", """\
          InternalSupportParams: {}
        """)

    valid, msg = validateCustomKernelMetadata("tensile", str(tmp_path))

    assert not valid
    assert "InternalSupportParams.KernArgsVersion" in msg


def test_build_validation_passes_for_minimal_tensile_kernel(tmp_path):
    write_kernel(tmp_path / "tensile.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
        """)
    kernels = [{"CustomKernel": {"name": "tensile"}}]

    assert validateCustomKernelMetadataAtBuild(kernels, str(tmp_path)) == 0


def test_validate_all_uses_recursive_loader_discovery(tmp_path):
    write_kernel(tmp_path / "nested" / "kernel.s", """\
          Source:
            Origin: test
          InternalSupportParams:
            KernArgsVersion: 0
        """)

    errors, warnings = validate_all(str(tmp_path), strict=True)

    assert len(errors) == 1
    assert warnings == []
    assert "nested" in errors[0]


def test_get_custom_kernel_config_infers_metadata_only_kernel(tmp_path):
    write_kernel(tmp_path / "metadata_only.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
          ProblemType: {}
          MatrixInstruction: [16, 16, 16, 1]
        """)

    config = getCustomKernelConfig("metadata_only", {}, str(tmp_path))

    assert config["CustomKernel"]["name"] == "metadata_only"
    assert config["CustomKernel"]["args"][0]["semantic"] == "AddressD"
    assert config["CustomKernel"]["threads"] == [64, 1, 1]


def test_parse_tensile_yaml_fails_when_kernel_not_found(tmp_path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(dedent("""\
        BenchmarkProblems:
          -
            - OperationType: GEMM
            - ForkParameters:
              - CustomKernel:
                - name: present_kernel
                  args: []
                  macrotile: [16, 16, 16]
                  threads: [64, 1, 1]
                  grid: [TilesX, TilesY, Batch]
        """))

    with pytest.raises(RuntimeError, match="Kernel 'missing_kernel' not found"):
        _parse_tensile_yaml(str(yaml_path), "missing_kernel")


def test_read_asm_file_reports_invalid_metadata_yaml(tmp_path):
    asm_path = tmp_path / "bad.s"
    asm_path.write_text(dedent("""\
        .amdgpu_metadata
        ---
        custom.config: [
        ...
        """))

    with pytest.raises(RuntimeError, match="Failed to parse .amdgpu_metadata YAML"):
        _read_asm_file(str(asm_path))


def test_inject_custom_config_dry_run_does_not_modify_file(tmp_path, capsys):
    asm_path = tmp_path / "dry_run.s"
    asm_path.write_text(dedent("""\
        .amdgpu_metadata
        ---
        amdhsa.kernels: []
        ...
        """))
    before = asm_path.read_text()
    file_info = _read_asm_file(str(asm_path))
    config_yaml = build_custom_config_yaml("test", None)

    assert inject_custom_config(file_info, str(asm_path), config_yaml, dry_run=True)
    assert asm_path.read_text() == before
    assert "custom.config block that would be inserted" in capsys.readouterr().out
