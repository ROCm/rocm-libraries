# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys
import textwrap

import pytest
import yaml

from tensilelite.KernelWriter import KernelWriter
from tensilelite.KernelWriterConversion import KernelWriterConversion
from tensilelite.Common.ScaleAlphaVec import (
    VALID_SCALE_ALPHA_VEC_MODES, scaleAlphaSupportsDeviceScalar, scaleAlphaVectorMode,
)
from tensilelite.SolutionStructs.Problem import ProblemType

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", VALID_SCALE_ALPHA_VEC_MODES)
def test_scale_alpha_modes_preserve_vector_direction_and_name(mode):
    problem = ProblemType({"DataType": "s", "UseScaleAlphaVec": mode}, False)
    direction = mode & 3
    vector = ProblemType({"DataType": "s", "UseScaleAlphaVec": direction}, False)
    assert scaleAlphaVectorMode(mode) == direction
    assert scaleAlphaSupportsDeviceScalar(mode) is (mode in (5, 6, 7))
    assert str(problem) == str(vector)


@pytest.mark.parametrize("mode", [-1, 4, 8, 9])
def test_scale_alpha_rejects_reserved_or_unknown_modes(mode, capsys):
    with pytest.raises(SystemExit):
        ProblemType({"DataType": "s", "UseScaleAlphaVec": mode}, False)
    assert "UseScaleAlphaVec must be 0, 1, 2, 3, 5, 6, or 7" in capsys.readouterr().out


@pytest.mark.parametrize("isa", [
    (9, 0, 8), (9, 0, 10), (9, 4, 2), (9, 5, 0),
    (12, 0, 0), (12, 0, 1), (12, 5, 1),
])
@pytest.mark.parametrize("mode", [5, 6, 7])
@pytest.mark.parametrize("custom", [False, True])
def test_scalar_capability_rejects_non_gfx1250_before_derivation(
    isa, mode, custom, capsys, monkeypatch
):
    from types import SimpleNamespace

    from tensilelite.Common import IsaVersion
    from tensilelite.SolutionStructs.Solution import Solution

    def unexpected_derivation(*args, **kwargs):
        pytest.fail("Unsupported scalar capability must be rejected before derivation")

    monkeypatch.setattr(Solution, "assignDerivedParameters", unexpected_derivation)
    monkeypatch.setattr(Solution, "_assignCustomKernelParameters", unexpected_derivation)
    config = {"ISA": list(isa), "ProblemType": {"DataType": "s", "UseScaleAlphaVec": mode}}
    if custom:
        config["CustomKernel"] = {"name": "unsupported_scalar"}
    assembler = SimpleNamespace(code_object_version="default")
    isa_info = {IsaVersion(*isa): SimpleNamespace(asmCaps={"SupportedISA": True})}
    solution = Solution(config, False, True, False, assembler, isa_info)

    assert not solution["Valid"]
    assert "device scalar alpha (5/6/7) is only supported on gfx1250" in capsys.readouterr().out


@pytest.mark.parametrize("arch", ["gfx1250", "gfx1250-strict"])
@pytest.mark.parametrize("problem_name, cu_suffix, library_type", [
    ("HSS_BH_Bias_HA_S_SAV_UserArgs", "", "GridBased"),
    ("HSS_BH_Bias_GG_HA_S_SAV_UserArgs", "", "GridBased"),
    ("F4SS_MXAF8B16_MXBF8B16_BH_Bias_HA_S_SAV_UserArgs", "", "GridBased"),
    ("F4BS_MXAF8B16_MXBF8B16_BH_Bias_HA_S_SAV_UserArgs", "", "GridBased"),
    ("F4BS_MXAF8B16_MXBF8B16_BH_SAV_UserArgs", "", "Equality"),
    ("F4BS_MXAF8B16_MXBF8B16_BH_SAV_UserArgs", "_96cu", "Equality"),
    ("F4BS_MXAF8B16_MXBF8B16_BH_SAV_UserArgs", "_192cu", "Equality"),
])
def test_shipped_scalar_logic_explicitly_advertises_capability_gfx1250(
    problem_name, arch, cu_suffix, library_type
):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.LibraryIO import parseLibraryLogicFile
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

    product = Path(__file__).resolve().parents[4] / "library/src/amd_detail/rocblaslt"
    if not product.is_dir():
        pytest.skip("Shipped hipBLASLt logic is not included in standalone TensileLite test artifacts")
    logic = (product / f"src/Tensile/Logic/asm_full/{arch}/{arch}{cu_suffix}/{library_type}"
             / f"{arch}_Cijk_Alik_Bljk_{problem_name}.yaml")
    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    isa_info = makeIsaInfoMap([gfxToIsa("gfx1250")], cxx)
    solutions = parseLibraryLogicFile(str(logic), assembler, False, False, False, isa_info, False).solutions
    assert solutions
    assert all(solution["ProblemType"]["UseScaleAlphaVec"] == 5 for solution in solutions)
    assert all("SupportDeviceScalarAlpha" not in solution["InternalSupportParams"]
               for solution in solutions)


@pytest.mark.parametrize(
    "version, mode, expected",
    [
        (2, 1, "0x3fff"),
        (3, 1, "0xfff"),
        (2, 5, "0x7ff"),
        (3, 5, "0x7ff"),
        (3, 6, "0x7ff"),
        (3, 7, "0x7ff"),
    ],
)
def test_device_scalar_alpha_capability_owns_gsu_bit_11(version, mode, expected):
    kernel = {"InternalSupportParams": {"KernArgsVersion": version},
              "ProblemType": {"UseScaleAlphaVec": mode}}
    assert KernelWriter.gsuMaskHex(None, kernel) == expected


@pytest.mark.parametrize(
    "mode, expected",
    [
        (1, [True, False]),
        (5, [True, False]),
    ],
)
def test_device_scalar_conversion_emits_both_runtime_variants(
    mode, expected
):
    writer = KernelWriterConversion.__new__(KernelWriterConversion)
    writer.state = {"ProblemType": {"UseScaleAlphaVec": mode}}
    assert writer.deviceScalarAlphaEnabled() is scaleAlphaSupportsDeviceScalar(mode)
    assert writer.groupedGemmModes() == expected


def _alpha_benchmark(final_args):
    from tensilelite.BenchmarkStructs import BenchmarkProcess

    return BenchmarkProcess(
        {"OperationType": "GEMM", "DataType": "s", "UseScaleAlphaVec": 1},
        {"BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [64, 64, 64]}]}]
                                      + final_args},
        False,
        gfxName="gfx1250",
    )


@pytest.mark.parametrize("modes", [None, [True], [False], [True, False], [False, True]])
def test_alpha_modes_are_runtime_arguments_not_kernel_forks(modes):
    default = _alpha_benchmark([])
    process = _alpha_benchmark([] if modes is None else [{"DeviceScalarAlpha": modes}])
    assert process[0].deviceScalarAlphaModes == modes
    assert process[0].forkParams == default[0].forkParams
    assert process[0].constantParams == default[0].constantParams


@pytest.mark.parametrize("modes", [None, [], True, "True", [1], ["False"], [True, True]])
def test_invalid_alpha_runtime_arguments_are_rejected(modes, capsys):
    with pytest.raises(SystemExit):
        _alpha_benchmark([{"DeviceScalarAlpha": modes}])
    assert "DeviceScalarAlpha must be a non-empty list of distinct booleans" in capsys.readouterr().out


def test_duplicate_alpha_runtime_arguments_are_rejected(capsys):
    with pytest.raises(SystemExit):
        _alpha_benchmark([{"DeviceScalarAlpha": [True]}, {"DeviceScalarAlpha": [False]}])
    assert "Duplicated DeviceScalarAlpha" in capsys.readouterr().out


def test_mxf4_yaml_runs_both_modes():
    from tensilelite.BenchmarkStructs import BenchmarkProcess

    path = Path(__file__).resolve().parents[1] / "common/gemm/gfx12/mxf4_device_scalar_alpha_gfx1250.yaml"
    config = yaml.safe_load(path.read_text())
    assert "DeviceScalarAlpha" not in config["GlobalParameters"]
    modes = [BenchmarkProcess(problem, benchmark, False, gfxName="gfx1250")[0].deviceScalarAlphaModes
             for problem, benchmark in config["BenchmarkProblems"]]
    assert modes == [[True, False], [True, False], [True, False]]


@pytest.mark.parametrize("failure", ["none", "True", "False"])
@pytest.mark.parametrize("modes", [[True, False], [False, True]])
def test_alpha_runtime_sweep_reuses_kernels_and_preserves_failures(tmp_path, monkeypatch, failure, modes):
    import tensilelite.ClientWriter as cw

    # Execute the generated launch scripts; only the GPU client is replaced.
    client = tmp_path / "client.py"
    client.write_text(textwrap.dedent(f"""\
        from pathlib import Path
        import sys
        config = Path(sys.argv[sys.argv.index('--config-file') + 1])
        values = dict(line.split('=', 1) for line in config.read_text().splitlines())
        mode = values['device-scalar-alpha']
        Path(values['results-file']).write_text(mode)
        sys.exit(7 if mode == {failure!r} else 0)
        """))
    monkeypatch.setattr(cw, "getClientExecutablePath", lambda: f"{sys.executable} {client}")
    for key, value in {"CpuOnly": False, "ParallelGpuExecution": 1, "PinClocks": False,
                       "TimingInstrumentation": False, "MXScaleFormat": "", "ClientExecutionLockPath": None}.items():
        monkeypatch.setitem(cw.globalParameters, key, value)
    ini = tmp_path / "ClientParameters.ini"
    original = ("library-file=library/TensileLibrary.yaml\n"
                "code-object=library/unchanged.co\n"
                "problem-size=64,64,1,64\n"
                f"results-file={tmp_path / 'results.csv'}\n"
                "device-scalar-alpha=False\n")
    ini.write_text(original)

    rc = cw.runClient(None, True, False, "c++", "cc", tmp_path,
                      configPaths=[str(ini)], deviceScalarAlphaModes=modes)

    assert rc == (0 if failure == "none" else 7)
    assert ini.read_text() == original
    assert "DeviceScalarAlpha" not in cw.globalParameters
    assert (tmp_path / "results.csv").read_text() == str(modes[0])
    assert (tmp_path / f"results_DeviceScalarAlpha{modes[1]}.csv").read_text() == str(modes[1])
    for mode in modes:
        content = (tmp_path / f"ClientParameters_DeviceScalarAlpha{mode}.ini").read_text()
        assert f"device-scalar-alpha={mode}\n" in content
        assert "library-file=library/TensileLibrary.yaml\n" in content
        assert "code-object=library/unchanged.co\n" in content
        assert "problem-size=64,64,1,64\n" in content
