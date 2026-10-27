# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Wave32 support is independent of the kernel descriptor's wave-size selector."""

import re
from types import SimpleNamespace

import pytest
import rocisa
from rocisa.code import Module, SignatureBase
from rocisa.instruction import SEndpgm

from gpu_test_helpers import init_rocisa
from tensilelite.Common.Architectures import gfxToIsa

import AMaxGenerator
import LayerNormGenerator
import SoftmaxGenerator
import config_harness


pytestmark = pytest.mark.unit


def _wavefront_directives(source):
    return re.findall(r"^\s*\.amdhsa_wavefront_size32\s+([01])\b", source, re.MULTILINE)


@pytest.mark.parametrize(
    "target,wave_size,expected",
    [
        ("gfx942", 64, []),
        ("gfx1030", 32, ["1"]),
        ("gfx1030", 64, ["0"]),
        ("gfx1100", 32, ["1"]),
        ("gfx1100", 64, ["0"]),
        ("gfx1200", 32, ["1"]),
        ("gfx1200", 64, ["0"]),
        ("gfx1201", 32, ["1"]),
        ("gfx1250", 32, []),
        ("gfx1251", 32, []),
    ],
)
def test_gemm_descriptor_preserves_wave_size_without_reserved_selector(target, wave_size, expected):
    init_rocisa(target=target, wavesize=wave_size)
    signature = SignatureBase(
        kernelName="wavefront_probe",
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=0,
        sgprWorkGroup=(1, 0, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=wave_size,
    )
    signature.setGprs(8, 0, 8)
    source = str(signature)
    assert _wavefront_directives(source) == expected
    assert re.search(rf"\.wavefront_size:\s+{wave_size}\b", source)


def test_converted_gfx1250_gemm_preserves_wave32_without_reserved_selector():
    init_rocisa(target="gfx1250", wavesize=32)
    signature = SignatureBase("converted_probe", 1, "4", 0, (1, 0, 0), 0, 32)
    signature.setGprs(8, 0, 8)
    body = Module("converted_probe")
    body.add(SEndpgm())
    converted = rocisa.toStinkyTofuModule(
        body,
        (12, 5, 0),
        "converted_probe",
        signature=signature,
        options={"wavefrontSize": 32, "OptLevel": 0},
    )
    source = converted.emitAssembly()
    assert _wavefront_directives(source) == []
    assert re.search(r"\.wavefront_size:\s+32\b", source)
    assert "s_endpgm" in source


@pytest.mark.parametrize("generator", [AMaxGenerator, LayerNormGenerator, SoftmaxGenerator])
@pytest.mark.parametrize(
    "target,wave_size,expected",
    [
        ("gfx942", 64, []),
        ("gfx1030", 32, ["1"]),
        ("gfx1100", 32, ["1"]),
        ("gfx1200", 32, ["1"]),
        ("gfx1201", 32, ["1"]),
        ("gfx1250", 32, []),
        ("gfx1250-strict", 32, []),
    ],
)
def test_extop_descriptor_omits_reserved_selector(generator, target, wave_size, expected):
    init_rocisa(target=target, wavesize=wave_size)
    if generator is SoftmaxGenerator:
        source = generator.kernel_rodata("wavefront_probe", gfxToIsa(target))
    else:
        source = generator.kernel_header("wavefront_probe", target, 8, 8, 0)
    assert _wavefront_directives(source) == expected


def test_gfx1250_default_wave_size_remains_32(monkeypatch):
    import tensilelite.BenchmarkProblems as benchmark_problems

    init_rocisa(target="gfx1250", wavesize=32)
    isa = gfxToIsa("gfx1250")
    caps = rocisa.rocIsa.getInstance().getArchCaps()
    info = {isa: SimpleNamespace(archCaps=caps)}
    solution = {
        "ISA": isa,
        "WavefrontSize": -1,
        "MatrixInstruction": [],
        "WorkGroup": [32, 4, 1],
        "ProblemType": {},
    }
    observed = []

    def stop_after_wave_size_selection(state, isa_info):
        observed.append(state["WavefrontSize"])
        return False

    monkeypatch.setattr(benchmark_problems, "validateMIParameters", stop_after_wave_size_selection)
    debug_config = SimpleNamespace(printSolutionRejectionReason=False)
    benchmark_problems._build_and_validate_solution(solution, None, debug_config, info)
    assert observed == [32]


@pytest.mark.parametrize(
    "target,directive,expected",
    [
        ("gfx942", "", 64),
        ("gfx1200", "", 64),
        ("gfx1200", ".amdhsa_wavefront_size32 1", 32),
        ("gfx1200", ".amdhsa_wavefront_size32 0", 64),
        ("gfx1250", "", 32),
        ("gfx1250-strict", "", 32),
        ("gfx1251", "", 32),
        ("gfx12-5-generic", "", 32),
    ],
)
def test_assembly_harness_uses_fixed_wave_size(target, directive, expected, monkeypatch):
    observed = []

    def assembler(arch, wave_size, source_path, object_path):
        observed.append((arch, wave_size))

    monkeypatch.setattr(config_harness, "_assembler_or_reason", lambda: (assembler, None))
    source = f'.amdgcn_target "amdgcn-amd-amdhsa--{target}"\n{directive}\n'
    config_harness.assert_assembles(source, "wavefront_probe")
    assert observed == [(target, expected)]
