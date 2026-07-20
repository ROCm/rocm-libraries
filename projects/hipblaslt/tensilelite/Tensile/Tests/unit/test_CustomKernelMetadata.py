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
import Tensile.TensileLogic.HandleCustomKernel as hck_mod
from Tensile.AddCustomConfig import (
    _fmt_yaml_args,
    _fmt_yaml_inline,
    _fmt_yaml_scalar,
    _parse_tensile_yaml,
    _read_asm_file,
    build_custom_config_yaml,
    inject_custom_config,
)
from Tensile.Contractions import ProblemPredicate
from Tensile.Common.ValidParameters import checkParametersAreValid, validParameters
from Tensile.CustomKernels import (
    _buildCustomKernelFromMetadata,
    _metadataArgToCustomArg,
    getCustomKernelConfig,
    getCustomKernelFilepath,
    isCustomKernelConfig,
    iterCustomKernelFiles,
    readCustomKernelConfig,
    validateCustomKernelMetadata,
)
from Tensile.Toolchain.Assembly import validateCustomKernelMetadataAtBuild
from Tensile.ValidateMetadata import validate_all

pytestmark = pytest.mark.unit


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


# --------------------------------------------------------------------------- #
# AddCustomConfig YAML formatting helpers
# --------------------------------------------------------------------------- #


def test_fmt_yaml_scalar_bool_none_int():
    assert _fmt_yaml_scalar(True) == "true"
    assert _fmt_yaml_scalar(False) == "false"
    assert _fmt_yaml_scalar(None) == "null"
    assert _fmt_yaml_scalar(7) == "7"


def test_fmt_yaml_inline_list_dict_scalar_nested():
    assert _fmt_yaml_inline([1, 2, 3]) == "[1, 2, 3]"
    assert _fmt_yaml_inline({"a": 1, "b": True}) == "{ a: 1, b: true }"
    assert _fmt_yaml_inline("TilesX") == "TilesX"
    assert _fmt_yaml_inline([{"k": [1, 2]}]) == "[{ k: [1, 2] }]"


def test_fmt_yaml_args_empty_single_multiple():
    assert _fmt_yaml_args([]) == "args: []"

    single = _fmt_yaml_args([{"type": "address", "semantic": "AddressD"}])
    assert single == "    args: [ { type: address, semantic: AddressD } ]"

    three = _fmt_yaml_args([
        {"type": "address", "semantic": "AddressD"},
        {"type": "uint32", "semantic": "SizeFree0"},
        {"type": "uint32", "semantic": "SizeFree1"},
    ])
    lines = three.split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("    args: [ { type: address, semantic: AddressD },")
    assert lines[1].strip() == "{ type: uint32, semantic: SizeFree0 },"  # middle arm
    assert lines[2].strip() == "{ type: uint32, semantic: SizeFree1 } ]"


# --------------------------------------------------------------------------- #
# build_custom_config_yaml variants
# --------------------------------------------------------------------------- #


def test_build_config_provenance_only():
    out = build_custom_config_yaml("aiter", None, repository="http://x", version="2.0")
    assert "Origin: aiter" in out
    assert "Repository: http://x" in out
    assert "Version: 2.0" in out
    assert "SupportsBias: false" in out
    assert "KernArgsVersion: 0" in out
    assert "WavefrontSize: 64" in out
    assert "ProblemType:" not in out
    assert "CustomKernel:" not in out
    assert "MatrixInstruction:" not in out


def test_build_config_features_and_isp_overrides():
    config = {
        "Features": {"SupportsBias": True},
        "InternalSupportParams": {"KernArgsVersion": 2, "UseUniversalArgs": True},
        "WavefrontSize": 32,
    }
    out = build_custom_config_yaml("wave", config)
    assert "SupportsBias: true" in out
    assert "SupportsUserArgs: false" in out  # defaulted flag
    assert "KernArgsVersion: 2" in out
    assert "UseUniversalArgs: True" in out
    assert "WavefrontSize: 32" in out


def test_build_config_mi_without_macrotile_no_enable():
    out = build_custom_config_yaml("aiter", {"MatrixInstruction": [16, 16, 16, 1]})
    assert "MatrixInstruction: [16, 16, 16, 1]" in out
    assert "EnableMatrixInstruction" not in out
    assert "MIWaveTile" not in out


def test_build_config_full_mi_emits_wavetile():
    config = {
        "MatrixInstruction": [16, 16, 16, 1],
        "CustomKernel": {"macrotile": [256, 256, 64], "threads": [256, 1, 1]},
        "WavefrontSize": 64,
    }
    out = build_custom_config_yaml("aiter", config)
    assert "EnableMatrixInstruction: True" in out
    assert "MIWaveTile:" in out


# --------------------------------------------------------------------------- #
# _parse_tensile_yaml error / edge branches
# --------------------------------------------------------------------------- #


def test_parse_tensile_yaml_malformed_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("BenchmarkProblems: [\n")
    with pytest.raises(RuntimeError, match="Failed to parse Tensile YAML"):
        _parse_tensile_yaml(str(p))


def test_parse_tensile_yaml_missing_benchmark_problems(tmp_path):
    p = tmp_path / "nobp.yaml"
    p.write_text("GlobalParameters: {}\n")
    with pytest.raises(RuntimeError, match="does not contain BenchmarkProblems"):
        _parse_tensile_yaml(str(p))


def test_parse_tensile_yaml_no_custom_kernel(tmp_path):
    p = tmp_path / "nock.yaml"
    p.write_text(dedent("""\
        BenchmarkProblems:
          -
            - OperationType: GEMM
            - ForkParameters:
              - MatrixInstruction:
                - [16, 16, 16, 1]
        """))
    with pytest.raises(RuntimeError, match="No CustomKernel entry found"):
        _parse_tensile_yaml(str(p))


def test_parse_tensile_yaml_first_kernel_with_mi_and_wavefront(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text(dedent("""\
        BenchmarkProblems:
          -
            - OperationType: GEMM
            - ForkParameters:
              - CustomKernel:
                - name: first_kernel
                  args: []
                  macrotile: [16, 16, 16]
                  threads: [64, 1, 1]
                  grid: [TilesX, TilesY, Batch]
              - MatrixInstruction:
                - [16, 16, 16, 1]
              - WavefrontSize:
                - 32
        """))

    config = _parse_tensile_yaml(str(p))  # kernel_name=None -> first kernel

    assert "CustomKernel" in config
    assert "name" not in config["CustomKernel"]
    assert config["MatrixInstruction"] == [16, 16, 16, 1]
    assert config["WavefrontSize"] == 32


# --------------------------------------------------------------------------- #
# _read_asm_file detection + edges
# --------------------------------------------------------------------------- #


def test_read_asm_file_detects_threads_and_wavefront(tmp_path):
    p = tmp_path / "k.s"
    p.write_text(dedent("""\
        .amdgpu_metadata
        ---
        amdhsa.kernels:
          - .name: k
            .reqd_workgroup_size: [256, 1, 1]
            .wavefront_size: 64
        ...
        """))

    info = _read_asm_file(str(p))

    assert info["detected"]["threads"] == [256, 1, 1]
    assert info["detected"]["wavefront_size"] == 64
    assert info["has_custom_config"] is False
    assert info["insert_idx"] is not None
    assert info["origin"] == tmp_path.name


def test_read_asm_file_flags_existing_custom_config(tmp_path):
    p = tmp_path / "k.s"
    p.write_text(dedent("""\
        .amdgpu_metadata
        ---
        custom.config:
          InternalSupportParams:
            KernArgsVersion: 0
        amdhsa.kernels: []
        ...
        """))

    assert _read_asm_file(str(p))["has_custom_config"] is True


def test_read_asm_file_no_metadata_section(tmp_path):
    p = tmp_path / "k.s"
    p.write_text("s_nop 0\ns_endpgm\n")

    info = _read_asm_file(str(p))

    assert info["insert_idx"] is None
    assert info["detected"] == {}


# --------------------------------------------------------------------------- #
# inject_custom_config real write + no-section
# --------------------------------------------------------------------------- #


def test_inject_custom_config_writes_block(tmp_path):
    p = tmp_path / "k.s"
    p.write_text(dedent("""\
        .amdgpu_metadata
        ---
        amdhsa.kernels: []
        ...
        """))
    info = _read_asm_file(str(p))

    assert inject_custom_config(info, str(p), build_custom_config_yaml("test", None))
    assert "custom.config:" in p.read_text()


def test_inject_custom_config_no_section_returns_false(tmp_path, capsys):
    p = tmp_path / "k.s"
    p.write_text("s_nop 0\n")
    info = _read_asm_file(str(p))

    assert inject_custom_config(info, str(p), "custom.config:\n  x: 1") is False
    assert "No .amdgpu_metadata" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# CustomKernels._metadataArgToCustomArg semantic mapping
# --------------------------------------------------------------------------- #


def _meta_arg(name, size=4, value_kind="by_value"):
    return {".name": name, ".size": size, ".value_kind": value_kind}


def test_metadata_arg_global_buffer_is_address():
    assert _metadataArgToCustomArg(_meta_arg("D", 8, "global_buffer")) == {
        "type": "address", "semantic": "AddressD"
    }


def test_metadata_arg_size8_is_float64():
    assert _metadataArgToCustomArg(_meta_arg("beta", 8)) == {
        "type": "float64", "semantic": "Beta"
    }


def test_metadata_arg_default_uint32():
    assert _metadataArgToCustomArg(_meta_arg("alpha", 4)) == {
        "type": "uint32", "semantic": "Alpha"
    }


def test_metadata_arg_activation_index():
    assert _metadataArgToCustomArg(_meta_arg("activationAlpha", 4)) == {
        "type": "float32", "semantic": "ActivationArg"
    }
    assert _metadataArgToCustomArg(_meta_arg("activationBeta", 8)) == {
        "type": "float64", "semantic": "ActivationArg", "index": 1
    }


def test_metadata_arg_regex_semantics():
    assert _metadataArgToCustomArg(_meta_arg("SizesFree0"))["semantic"] == "SizeFree0"
    assert _metadataArgToCustomArg(_meta_arg("SizesSum0"))["semantic"] == "SizeSum"
    assert _metadataArgToCustomArg(_meta_arg("SizesSum1"))["semantic"] == "SizeSum1"
    assert _metadataArgToCustomArg(_meta_arg("strideA0"))["semantic"] == "StrideA0"
    assert _metadataArgToCustomArg(_meta_arg("strideMetadata0"))["semantic"] == "StrideMetadata0"
    assert _metadataArgToCustomArg(_meta_arg("StrideE0"))["semantic"] == "StrideE0"


def test_metadata_arg_magic_size_index():
    a = _metadataArgToCustomArg(_meta_arg("MagicNumberSizeI"))
    assert a["semantic"] == "MagicNumberSize" and a["index"] == 0
    b = _metadataArgToCustomArg(_meta_arg("MagicShiftSizeJ"))
    assert b["semantic"] == "MagicShiftSize" and b["index"] == 1


def test_metadata_arg_missing_field_raises():
    with pytest.raises(RuntimeError, match="missing required field"):
        _metadataArgToCustomArg({".name": "D"})


def test_metadata_arg_unknown_name_raises():
    with pytest.raises(RuntimeError, match="Unknown amdgpu_metadata arg name"):
        _metadataArgToCustomArg(_meta_arg("totallyBogusArg"))


# --------------------------------------------------------------------------- #
# validate_all (non-strict) + validateCustomKernelMetadataAtBuild branches
# --------------------------------------------------------------------------- #


def test_validate_all_non_strict_reports_warnings(tmp_path):
    write_kernel(tmp_path / "bad.s", """\
          Source:
            Origin: test
          InternalSupportParams:
            KernArgsVersion: 0
        """)

    errors, warnings = validate_all(str(tmp_path), strict=False)

    assert errors == []
    assert len(warnings) == 1


def test_validate_at_build_counts_issues_and_dedups(tmp_path):
    write_kernel(tmp_path / "good.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
        """)
    write_kernel(tmp_path / "bad.s", """\
          InternalSupportParams: {}
        """)
    kernels = [
        {"CustomKernel": {"name": "good"}},
        {"CustomKernel": {"name": "bad"}},
        {"CustomKernel": {"name": "good"}},  # duplicate -> validated once
        {"SomethingElse": 1},                # no CustomKernel -> skipped
        {"CustomKernel": {"name": ""}},      # empty name -> skipped
    ]

    assert validateCustomKernelMetadataAtBuild(kernels, str(tmp_path)) == 1


# --------------------------------------------------------------------------- #
# HandleCustomKernel branch coverage
# --------------------------------------------------------------------------- #


def test_handle_custom_kernel_uses_mapping_name(monkeypatch):
    monkeypatch.setattr(
        hck_mod, "getCustomKernelConfig",
        lambda name, isp, directory: {"MatrixInstruction": [16, 16, 16, 1]},
    )
    sol = {"CustomKernel": {"name": "k0"}}

    out, is_custom = hck_mod.handleCustomKernel(sol, {})

    assert is_custom is True
    assert out["MatrixInstruction"] == [16, 16, 16, 1]


def test_handle_custom_kernel_skips_on_config_error(monkeypatch, capsys):
    def _raise(name, isp, directory):
        raise RuntimeError("missing custom.config")

    monkeypatch.setattr(hck_mod, "getCustomKernelConfig", _raise)
    sol = {"CustomKernelName": "k0"}

    out, is_custom = hck_mod.handleCustomKernel(sol, {})

    assert is_custom is False
    assert out is sol


# --------------------------------------------------------------------------- #
# CustomKernels._buildCustomKernelFromMetadata: grid selection + error branches
# --------------------------------------------------------------------------- #

_D_ARG = {".name": "D", ".size": 8, ".value_kind": "global_buffer"}
_NUMWG_ARG = {".name": "numWG", ".size": 4, ".value_kind": "by_value"}


def _kernel_yaml(args, name="k"):
    return {"amdhsa.kernels": [{".name": name, ".max_flat_workgroup_size": 256, ".args": args}]}


def test_build_from_metadata_no_kernels_raises():
    with pytest.raises(RuntimeError, match="no amdhsa.kernels entries"):
        _buildCustomKernelFromMetadata("k", {}, {"MatrixInstruction": [16, 16, 16, 1]})


def test_build_from_metadata_no_args_raises():
    full = {"amdhsa.kernels": [{".name": "k"}]}
    with pytest.raises(RuntimeError, match="no .args"):
        _buildCustomKernelFromMetadata("k", full, {"MatrixInstruction": [16, 16, 16, 1]})


def test_build_from_metadata_streamk_batched_grid():
    ck = _buildCustomKernelFromMetadata(
        "k", _kernel_yaml([_D_ARG]),
        {"MatrixInstruction": [16, 16, 16, 1], "StreamK": 2, "ProblemType": {"Batched": True}},
    )
    assert ck["grid"][0] == "StreamKWithBatch"


def test_build_from_metadata_numworkgroups_grid():
    ck = _buildCustomKernelFromMetadata(
        "k", _kernel_yaml([_NUMWG_ARG, _D_ARG]),
        {"MatrixInstruction": [16, 16, 16, 1]},
    )
    assert ck["grid"][0] == "TilesXYBatchGSU"


def test_build_from_metadata_default_multidim_grid():
    ck = _buildCustomKernelFromMetadata(
        "k", _kernel_yaml([_D_ARG]), {"MatrixInstruction": [16, 16, 16, 1]}
    )
    assert ck["grid"] == ["TilesX", "TilesY", "Batch"]


def test_build_from_metadata_macrotile_name_fallback():
    # No MatrixInstruction -> computed macrotile is 0, so it falls back to the
    # MTxxx token parsed out of the kernel name.
    ck = _buildCustomKernelFromMetadata("foo_MT128x256x64_bar", _kernel_yaml([_D_ARG]), {})
    assert ck["macrotile"] == [128, 256, 64]


# --------------------------------------------------------------------------- #
# validateCustomKernelMetadata edge branches
# --------------------------------------------------------------------------- #


def test_validate_metadata_no_custom_config_reports_unreadable(tmp_path):
    p = tmp_path / "raw.s"
    p.write_text(".amdgpu_metadata\n---\namdhsa.kernels: []\n...\n")  # no custom.config

    valid, msg = validateCustomKernelMetadata("raw", str(tmp_path))

    assert not valid
    assert "Cannot read custom.config" in msg


def test_validate_metadata_internal_support_params_not_mapping(tmp_path):
    write_kernel(tmp_path / "k.s", """\
          InternalSupportParams: not_a_dict
        """)

    valid, msg = validateCustomKernelMetadata("k", str(tmp_path))

    assert not valid
    assert "InternalSupportParams (mapping)" in msg


def test_validate_metadata_custom_kernel_not_mapping(tmp_path):
    write_kernel(tmp_path / "k.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
          CustomKernel: not_a_dict
        """)

    valid, msg = validateCustomKernelMetadata("k", str(tmp_path))

    assert not valid
    assert "CustomKernel (mapping)" in msg


# --------------------------------------------------------------------------- #
# Remaining CustomKernels branch coverage
# --------------------------------------------------------------------------- #


def test_is_custom_kernel_config_generated_is_false():
    # A generated (Tensile-emitted) CustomKernel mapping is NOT treated as a
    # handwritten custom kernel.
    assert isCustomKernelConfig({"CustomKernel": {"name": "k", "generated": True}}) is False
    assert isCustomKernelConfig({"CustomKernel": {"name": "k"}}) is True


def test_get_custom_kernel_filepath_recursive_fallback(tmp_path):
    # When no flat <dir>/<name>.s exists, the loader searches subdirectories.
    nested = tmp_path / "aiter"
    nested.mkdir()
    write_kernel(nested / "k.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
        """)
    assert getCustomKernelFilepath("k", str(tmp_path)) == str(nested / "k.s")


def test_read_custom_kernel_config_non_mapping_raises(tmp_path):
    p = tmp_path / "k.s"
    p.write_text(".amdgpu_metadata\n---\ncustom.config:\n  - 1\n  - 2\n...\ns_nop 0\n")
    with pytest.raises(RuntimeError, match="must be a YAML mapping"):
        readCustomKernelConfig("k", str(tmp_path))


def test_build_from_metadata_non_dict_yaml_raises():
    with pytest.raises(RuntimeError, match="no parseable .amdgpu_metadata"):
        _buildCustomKernelFromMetadata("k", None, {"MatrixInstruction": [16, 16, 16, 1]})


def test_build_from_metadata_mi_length9_derives_wave_from_mi():
    # MI of length >= 9 carries MIWaveTile (mi[5:7]) / MIWaveGroup (mi[7:9])
    # inline, so macrotile is computed from them.
    full = {"amdhsa.kernels": [{".name": "k", ".max_flat_workgroup_size": 256,
                                ".args": [{".name": "D", ".size": 8, ".value_kind": "global_buffer"}]}]}
    ck = _buildCustomKernelFromMetadata("k", full, {"MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2]})
    # macrotile0 = mi0 * wt0 * wg0 = 16 * 4 * 2 = 128
    assert ck["macrotile"][0] == 16 * 4 * 2
    assert ck["macrotile"][1] == 16 * 4 * 2


def test_get_custom_kernel_config_strips_unknown_metadata_key(tmp_path):
    # A non-tunable, non-passthrough key (e.g. provenance) is dropped from the
    # solution dict rather than fed to the solution.
    write_kernel(tmp_path / "k.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
          ProblemType: {}
          MatrixInstruction: [16, 16, 16, 1]
          Version: 9.9.9
        """)
    config = getCustomKernelConfig("k", {}, str(tmp_path))
    assert "Version" not in config


def test_get_custom_kernel_config_respects_explicit_custom_kernel(tmp_path):
    # An explicit CustomKernel block is used verbatim (no auto-inference).
    write_kernel(tmp_path / "k.s", """\
          InternalSupportParams:
            KernArgsVersion: 0
          ProblemType: {}
          MatrixInstruction: [16, 16, 16, 1]
          CustomKernel:
            args: []
            macrotile: [128, 128, 64]
            threads: [256, 1, 1]
            grid: [TilesX, TilesY, One]
        """)
    config = getCustomKernelConfig("k", {}, str(tmp_path))
    assert config["CustomKernel"]["macrotile"] == [128, 128, 64]
    assert config["CustomKernel"]["grid"] == ["TilesX", "TilesY", "One"]


def test_validate_metadata_tensile_missing_internal_support_params(tmp_path):
    write_kernel(tmp_path / "k.s", """\
          ProblemType: {}
        """)
    valid, msg = validateCustomKernelMetadata("k", str(tmp_path))
    assert not valid
    assert "InternalSupportParams" in msg


def test_validate_metadata_external_source_missing_origin(tmp_path):
    write_kernel(tmp_path / "k.s", """\
          Source:
            Repository: http://x
          Features: {}
          Version: 1.0.0
          InternalSupportParams:
            KernArgsVersion: 0
          ProblemType: {}
          MatrixInstruction: [16, 16, 16, 1]
          CustomKernel:
            args: []
            macrotile: [16, 16, 16]
            threads: [64, 1, 1]
            grid: [TilesX, TilesY, Batch]
        """)
    valid, msg = validateCustomKernelMetadata("k", str(tmp_path))
    assert not valid
    assert "Source.Origin" in msg


# --------------------------------------------------------------------------- #
# _parse_tensile_yaml: skip malformed ForkParameter entries
# --------------------------------------------------------------------------- #


def test_build_from_metadata_no_universal_args_skips_header_reorder():
    full = {"amdhsa.kernels": [{".name": "k", ".max_flat_workgroup_size": 256,
                                ".args": [{".name": "D", ".size": 8, ".value_kind": "global_buffer"}]}]}
    ck = _buildCustomKernelFromMetadata("k", full, {
        "MatrixInstruction": [16, 16, 16, 1],
        "InternalSupportParams": {"UseUniversalArgs": False},
    })
    assert ck["args"][0]["semantic"] == "AddressD"


def test_get_custom_kernel_filepath_iterates_past_nonmatching(tmp_path):
    nested = tmp_path / "aiter"
    nested.mkdir()
    # 'aaa.s' sorts before 'k.s' and does not match, so the search loop must
    # skip it before matching 'k.s'.
    write_kernel(nested / "aaa.s", "InternalSupportParams:\n  KernArgsVersion: 0")
    write_kernel(nested / "k.s", "InternalSupportParams:\n  KernArgsVersion: 0")
    assert getCustomKernelFilepath("k", str(tmp_path)).endswith("/k.s")


def test_read_asm_file_comment_only_metadata_returns_no_detection(tmp_path):
    # yaml_lines non-empty but parses to None (comment only) -> the "no metadata"
    # early return.
    p = tmp_path / "k.s"
    p.write_text(".amdgpu_metadata\n---\n# only a comment\n...\ns_nop 0\n")
    info = _read_asm_file(str(p))
    assert info["detected"] == {}
    assert info["insert_idx"] is not None


def test_parse_tensile_yaml_skips_non_dict_and_nameless_entries(tmp_path):
    # A non-dict fork entry and a CustomKernel entry without a name are skipped;
    # the first named kernel still resolves.
    p = tmp_path / "t.yaml"
    p.write_text(dedent("""\
        BenchmarkProblems:
          -
            - OperationType: GEMM
            - ForkParameters:
              - "a bare string entry"
              - CustomKernel:
                - args: []
                - name: real_kernel
                  args: []
                  macrotile: [16, 16, 16]
                  threads: [64, 1, 1]
                  grid: [TilesX, TilesY, Batch]
        """))
    config = _parse_tensile_yaml(str(p), "real_kernel")
    assert "CustomKernel" in config
