# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys
import textwrap

import pytest
import yaml

from tensilelite.KernelWriter import KernelWriter
from tensilelite.KernelWriterConversion import KernelWriterConversion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("problem_name", [
    "HSS_BH_Bias_HA_S_SAV_UserArgs",
    "HSS_BH_Bias_GG_HA_S_SAV_UserArgs",
    "F4SS_MXAF8B16_MXBF8B16_BH_Bias_HA_S_UserArgs",
])
def test_shipped_scalar_logic_explicitly_advertises_capability_gfx1250(problem_name):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Contractions import InternalArgsSupport
    from Tensile.LibraryIO import parseLibraryLogicFile
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

    product = Path(__file__).resolve().parents[4] / "library/src/amd_detail/rocblaslt"
    if not product.is_dir():
        pytest.skip("Shipped hipBLASLt logic is not included in standalone TensileLite test artifacts")
    logic = (product / "src/Tensile/Logic/asm_full/gfx1250/gfx1250/GridBased"
             / f"gfx1250_Cijk_Alik_Bljk_{problem_name}.yaml")
    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    isa_info = makeIsaInfoMap([gfxToIsa("gfx1250")], cxx)
    solutions = parseLibraryLogicFile(str(logic), assembler, False, False, False, isa_info, False).solutions
    assert solutions
    assert all(InternalArgsSupport.FromOriginalState(solution._state).deviceScalarAlpha
               for solution in solutions)


@pytest.mark.parametrize(
    "internal_support, expected",
    [
        ({"KernArgsVersion": 2}, "0x3fff"),
        ({"KernArgsVersion": 3}, "0xfff"),
        (
            {"KernArgsVersion": 2, "SupportDeviceScalarAlpha": True},
            "0x7ff",
        ),
        (
            {"KernArgsVersion": 3, "SupportDeviceScalarAlpha": True},
            "0x7ff",
        ),
    ],
)
def test_device_scalar_alpha_capability_owns_gsu_bit_11(internal_support, expected):
    kernel = {"InternalSupportParams": internal_support}
    assert KernelWriter.gsuMaskHex(None, kernel) == expected


@pytest.mark.parametrize(
    "support_device_scalar_alpha, expected",
    [
        (False, [True, False]),
        (True, [True, False]),
    ],
)
def test_device_scalar_conversion_emits_both_runtime_variants(
    support_device_scalar_alpha, expected
):
    writer = KernelWriterConversion.__new__(KernelWriterConversion)
    writer.supportDeviceScalarAlpha = support_device_scalar_alpha
    assert writer.groupedGemmModes() == expected


def _alpha_benchmark(final_args):
    from Tensile.BenchmarkStructs import BenchmarkProcess

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


def test_mxf4_yaml_runs_both_modes_and_keeps_singlebuffer_scalar():
    from Tensile.BenchmarkStructs import BenchmarkProcess

    path = Path(__file__).resolve().parents[1] / "common/gemm/gfx12/mxf4_device_scalar_alpha_gfx1250.yaml"
    config = yaml.safe_load(path.read_text())
    assert "DeviceScalarAlpha" not in config["GlobalParameters"]
    modes = [BenchmarkProcess(problem, benchmark, False, gfxName="gfx1250")[0].deviceScalarAlphaModes
             for problem, benchmark in config["BenchmarkProblems"]]
    assert modes == [[True, False], [True, False], [True, False], [True]]


@pytest.mark.parametrize("failure", ["none", "True", "False"])
@pytest.mark.parametrize("modes", [[True, False], [False, True]])
def test_alpha_runtime_sweep_reuses_kernels_and_preserves_failures(tmp_path, monkeypatch, failure, modes):
    import Tensile.ClientWriter as cw

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
