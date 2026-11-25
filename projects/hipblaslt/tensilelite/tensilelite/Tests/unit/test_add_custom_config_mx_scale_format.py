# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""AddCustomConfig declares the MX scale layout of the kernels it registers.

getCustomKernelConfig refuses a handwritten MX kernel without MXScaleFormat, so
a capture that AddCustomConfig registers must carry one: NoSwizzle for a
rocRoller kernel that reads packed scales, HostPreSwizzle for one whose name
carries a pre-swizzle tag, and whatever another origin declares.
"""

import sys
from pathlib import Path
from textwrap import dedent

import pytest

import tensilelite
import tensilelite.AddCustomConfig as acc
from tensilelite.CustomKernels import getCustomKernelConfig

pytestmark = pytest.mark.unit

RR_NAME = "RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2"

META = dedent("""\
    .amdgpu_metadata
    ---
    amdhsa.kernels:
      - .name: {name}
        .reqd_workgroup_size: [256, 1, 1]
        .wavefront_size: 64
        .max_flat_workgroup_size: 256
        .args:
          - .name: D
            .size: 8
            .offset: 0
            .value_kind: global_buffer
    ...
    """)


def _yaml(path, name, mx=True, extra_fork=""):
    blocks = "      MXBlockA: 32\n      MXBlockB: 32\n" if mx else ""
    path.write_text(dedent("""\
        BenchmarkProblems:
          -
            - OperationType: GEMM
              DataType: F8
              DestDataType: h
              ComputeDataType: s
              HighPrecisionAccumulate: True
              TransposeA: True
              TransposeB: False
              UseBeta: True
              Batched: True
        {blocks}    - ForkParameters:
              - CustomKernel:
                - name: {name}
                  args: [ {{ type: address, semantic: AddressD }} ]
                  macrotile: [64, 64, 128]
                  threads: [256, 1, 1]
                  grid: [TilesX, TilesY, Batch]
              - MatrixInstruction:
                - [16, 16, 128, 1]
        {extra}""").format(name=name, blocks=blocks, extra=extra_fork))
    return path


def _register(tmp_path, monkeypatch, name, origin="rocroller", mx=True, flags=(),
              extra_fork=""):
    kernel_dir = tmp_path / origin
    kernel_dir.mkdir(exist_ok=True)
    kernel = kernel_dir / f"{name}.s"
    kernel.write_text(META.format(name=name))
    yaml_path = _yaml(tmp_path / f"{name}.yaml", name, mx=mx, extra_fork=extra_fork)
    monkeypatch.setattr(sys, "argv", ["prog", str(kernel), "--yaml", str(yaml_path), *flags])
    acc.main()
    return kernel


def _config_lines(kernel):
    lines = kernel.read_text().splitlines()
    start = lines.index("custom.config:")
    stop = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith(" "))
    return lines[start:stop]


def _key_after_internal_support_params(lines):
    """The top-level custom.config key that follows the InternalSupportParams block."""
    at = lines.index("  InternalSupportParams:")
    return next(line for line in lines[at + 1:] if not line.startswith("    "))


def _shipped_rocroller_kernel():
    root = Path(tensilelite.__file__).parent / "CustomKernels" / "rocroller"
    for path in sorted(root.glob("RR_GEMM_*.s")):
        text = path.read_text()
        if "  MXBlockA: 32" in text:
            return path
    pytest.fail("no shipped rocRoller MX kernel")


def test_a_rocroller_kernel_declares_no_swizzle_where_the_shipped_kernels_do(
        tmp_path, monkeypatch):
    kernel = _register(tmp_path, monkeypatch, RR_NAME)
    lines = _config_lines(kernel)

    assert _key_after_internal_support_params(lines) == "  MXScaleFormat: NoSwizzle"
    assert lines[lines.index("  MXScaleFormat: NoSwizzle") + 1] == "  ProblemType:"
    shipped = _config_lines(_shipped_rocroller_kernel())
    assert _key_after_internal_support_params(shipped) == "  MXScaleFormat: NoSwizzle"
    assert shipped[shipped.index("  MXScaleFormat: NoSwizzle") + 1] == "  ProblemType:"

    config = getCustomKernelConfig(RR_NAME, {}, str(kernel.parent))
    assert config["MXScaleFormat"] == "NoSwizzle"


@pytest.mark.parametrize("tag", acc.ROCROLLER_PRESWIZZLE_TAGS)
def test_a_rocroller_pre_swizzle_name_declares_host_pre_swizzle(tmp_path, monkeypatch, tag):
    name = RR_NAME.replace("_SA_", f"{tag}SA_")
    kernel = _register(tmp_path, monkeypatch, name)
    assert getCustomKernelConfig(name, {}, str(kernel.parent))["MXScaleFormat"] == "HostPreSwizzle"


def test_a_rocroller_kernel_cannot_declare_a_layout_its_name_contradicts(
        tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _register(tmp_path, monkeypatch, RR_NAME, flags=["--mx-scale-format", "HostPreSwizzle"])
    assert e.value.code == 1
    assert "its name says NoSwizzle" in capsys.readouterr().err
    assert "custom.config" not in (tmp_path / "rocroller" / f"{RR_NAME}.s").read_text()


def test_another_origin_must_declare_its_layout(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _register(tmp_path, monkeypatch, "aiter_mx_kernel", origin="aiter")
    assert e.value.code == 1
    assert "must declare the scale layout" in capsys.readouterr().err
    assert "custom.config" not in (tmp_path / "aiter" / "aiter_mx_kernel.s").read_text()


def test_another_origin_takes_the_flag(tmp_path, monkeypatch):
    kernel = _register(tmp_path, monkeypatch, "aiter_mx_kernel", origin="aiter",
                       flags=["--mx-scale-format", "HostPreSwizzle"])
    assert getCustomKernelConfig("aiter_mx_kernel", {}, str(kernel.parent))[
        "MXScaleFormat"] == "HostPreSwizzle"


def test_another_origin_takes_the_yaml_fork_parameter(tmp_path, monkeypatch):
    kernel = _register(tmp_path, monkeypatch, "aiter_mx_kernel", origin="aiter",
                       extra_fork="      - MXScaleFormat: [InMemorySwizzle]\n")
    assert "  MXScaleFormat: InMemorySwizzle" in _config_lines(kernel)


def test_an_unresolved_yaml_layout_is_refused(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _register(tmp_path, monkeypatch, "aiter_mx_kernel", origin="aiter",
                  extra_fork="      - MXScaleFormat: [Auto]\n")
    assert "'Auto'" in capsys.readouterr().err


def test_a_kernel_without_mx_scales_declares_no_layout(tmp_path, monkeypatch):
    kernel = _register(tmp_path, monkeypatch, "plain_kernel", origin="aiter", mx=False)
    assert not any("MXScaleFormat" in line for line in _config_lines(kernel))
