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

import os
from textwrap import dedent, indent

import pytest

import Tensile
from Tensile.AddCustomConfig import (
    _parse_tensile_yaml,
    _read_asm_file,
    build_custom_config_yaml,
    inject_custom_config,
)
from Tensile.Contractions import ProblemPredicate
from Tensile.Common.ValidParameters import checkParametersAreValid, validParameters
from Tensile.CustomKernels import (
    getCustomKernelConfig,
    iterCustomKernelFiles,
    validateCustomKernelMetadata,
)
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


# --------------------------------------------------------------------------- #
# Custom-kernel predicates
# --------------------------------------------------------------------------- #


def test_valid_parameters_accept_size_multiple_256():
    checkParametersAreValid(("AssertFree0ElementMultiple", [256]), validParameters)
    checkParametersAreValid(("AssertFree1ElementMultiple", [256]), validParameters)
    checkParametersAreValid(("AssertSummationElementMultiple", [256]), validParameters)


def test_get_custom_kernel_config_preserves_size_multiple_predicate(tmp_path):
    """AssertFree0/1ElementMultiple: 256 must survive custom.config loading
    so AITER kernels can emit runtime size-multiple predicates."""
    write_kernel(tmp_path / "with_predicate.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
          ProblemType: {}
          MatrixInstruction: [16, 16, 16, 1]
          AssertFree0ElementMultiple: 256
          AssertFree1ElementMultiple: 256
        """)

    config = getCustomKernelConfig("with_predicate", {}, str(tmp_path))

    assert config["AssertFree0ElementMultiple"] == 256
    assert config["AssertFree1ElementMultiple"] == 256


def test_get_custom_kernel_config_rejects_bad_predicate_value(tmp_path):
    write_kernel(tmp_path / "bad_predicate.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
          ProblemType: {}
          MatrixInstruction: [16, 16, 16, 1]
          AssertFree0ElementMultiple: -1
        """)

    with pytest.raises(Exception, match="AssertFree0ElementMultiple"):
        getCustomKernelConfig("bad_predicate", {}, str(tmp_path))


def test_problem_predicate_emits_size_multiple_for_assert_free0():
    pred = ProblemPredicate.FromOriginalKeyPair(("AssertFree0ElementMultiple", 256))

    assert pred is not None
    assert pred.tag == "Free0SizeMultiple"
    assert pred.value == 256


def test_problem_predicate_emits_size_multiple_for_assert_free1():
    pred = ProblemPredicate.FromOriginalKeyPair(("AssertFree1ElementMultiple", 256))

    assert pred is not None
    assert pred.tag == "Free1SizeMultiple"
    assert pred.value == 256


def test_problem_predicate_drops_value_one():
    """value==1 means "no constraint" and must not produce a runtime predicate."""
    assert ProblemPredicate.FromOriginalKeyPair(("AssertFree0ElementMultiple", 1)) is None


def _write_minimal_yaml_with_predicate(yaml_path, predicate_value):
    yaml_path.write_text(dedent(f"""\
        BenchmarkProblems:
          -
            - OperationType: GEMM
            - ForkParameters:
              - CustomKernel:
                - name: predicated_kernel
                  args: []
                  macrotile: [256, 256, 64]
                  threads: [256, 1, 1]
                  grid: [TilesX, TilesY, One]
              - AssertFree0ElementMultiple: {predicate_value}
        """))


def test_parse_tensile_yaml_copies_single_valued_predicate(tmp_path):
    yaml_path = tmp_path / "predicate.yaml"
    _write_minimal_yaml_with_predicate(yaml_path, "[256]")

    config = _parse_tensile_yaml(str(yaml_path), "predicated_kernel")

    assert config["AssertFree0ElementMultiple"] == 256


def test_parse_tensile_yaml_rejects_multi_valued_predicate(tmp_path):
    yaml_path = tmp_path / "multi_predicate.yaml"
    _write_minimal_yaml_with_predicate(yaml_path, "[128, 256]")

    with pytest.raises(RuntimeError, match="single-valued ForkParameter"):
        _parse_tensile_yaml(str(yaml_path), "predicated_kernel")


def test_parse_tensile_yaml_rejects_scalar_predicate(tmp_path):
    yaml_path = tmp_path / "scalar_predicate.yaml"
    _write_minimal_yaml_with_predicate(yaml_path, "256")

    with pytest.raises(RuntimeError, match="single-valued ForkParameter"):
        _parse_tensile_yaml(str(yaml_path), "predicated_kernel")


def test_parse_tensile_yaml_rejects_bad_predicate_value(tmp_path):
    yaml_path = tmp_path / "bad_predicate.yaml"
    _write_minimal_yaml_with_predicate(yaml_path, "[0]")

    with pytest.raises(Exception, match="Invalid parameter value: AssertFree0ElementMultiple"):
        _parse_tensile_yaml(str(yaml_path), "predicated_kernel")


def test_build_custom_config_yaml_emits_predicate_after_mi():
    config = {
        "ProblemType": {"OperationType": "GEMM"},
        "MatrixInstruction": [16, 16, 16, 1],
        "CustomKernel": {
            "args": [],
            "macrotile": [256, 256, 64],
            "threads": [256, 1, 1],
            "grid": ["TilesX", "TilesY", "One"],
        },
        "AssertFree0ElementMultiple": 256,
        "AssertFree1ElementMultiple": 256,
        "WavefrontSize": 64,
    }

    rendered = build_custom_config_yaml(origin="test", config=config)

    mi_idx = rendered.index("MatrixInstruction:")
    f0_idx = rendered.index("AssertFree0ElementMultiple:")
    f1_idx = rendered.index("AssertFree1ElementMultiple:")
    wf_idx = rendered.index("WavefrontSize:")

    assert mi_idx < f0_idx < wf_idx
    assert mi_idx < f1_idx < wf_idx
    assert "AssertFree0ElementMultiple: 256" in rendered
    assert "AssertFree1ElementMultiple: 256" in rendered
