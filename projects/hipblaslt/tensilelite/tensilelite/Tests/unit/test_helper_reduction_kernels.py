# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The bias-gradient reduction helper kernels are defined in Kernels.cpp and only declared in
Kernels.h (timing_runs/DECISIONS.md topic 6, C): Kernels.h is included by every translation
unit of the helper compile, so a kernel defined in it would be compiled into each of them."""

import os
import re
import subprocess
from pathlib import Path

import pytest

from Tensile.Common.DataType import DataType
from Tensile.KernelWriterReduction import KernelWriterReduction
from Tensile.TensileCreateLibrary.Run import copyStaticFiles, writeHelpers

pytestmark = pytest.mark.unit

KERNELS_PER_BIAS_TYPE = 7  # one per macro tile / vector width


def _reduction(biasType):
    """A KernelWriterReduction as initReductionKernelObjects builds it."""
    problemType = {
        "NumIndicesC": 3,
        "DataType": DataType("h"),
        "ComputeDataType": DataType("s"),
        "HighPrecisionAccumulate": True,
        "BiasDataType": DataType(biasType),
    }
    return KernelWriterReduction({"ProblemType": problemType})


class _DummySource:
    """Like the activation helpers: a source that is a comment with no trailing newline."""

    def getKernelName(self):
        return "Dummy"

    def getSourceFileString(self):
        return (0, "// This is a dummy file.")

    def getHeaderFileString(self):
        return ""


SIGNATURE = re.compile(r'^extern "C" __global__ void (D\w+_Reduction)\(.*\)$')


def _signatures(text, end=""):
    """Kernel names of the reduction signature lines (ending in `end`) of a source text."""
    names = []
    for line in text.splitlines():
        if line.endswith(end):
            m = SIGNATURE.match(line[: len(line) - len(end)])
            if m:
                names.append(m.group(1))
    return names


def test_header_declares_and_source_defines_every_reduction_kernel():
    kw = _reduction("s")
    header = kw.getHeaderFileString()
    err, source = kw.getSourceFileString()
    assert err == 0

    declared = _signatures(header, end=";")
    assert len(declared) == KERNELS_PER_BIAS_TYPE
    assert all(name.startswith("Dijk_SS_MT") for name in declared)
    assert "{" not in header and "reductionKernel_" not in header

    defined = _signatures(source)
    assert defined == declared
    assert source.count("reductionKernel_ijk<float, float, ") == KERNELS_PER_BIAS_TYPE


def test_written_helpers_define_reduction_kernels_only_in_the_source(tmp_path):
    """writeHelpers concatenates the helper sources: every definition still starts a line, also
    after a source with no trailing newline."""
    writeHelpers(
        tmp_path, [_DummySource(), _reduction("s"), _reduction("h")], "Kernels.cpp", "Kernels.h"
    )
    header = (tmp_path / "Kernels.h").read_text()
    source = (tmp_path / "Kernels.cpp").read_text()

    assert _signatures(header) == []
    assert len(_signatures(header, end=";")) == 2 * KERNELS_PER_BIAS_TYPE
    assert _signatures(source) == _signatures(header, end=";")


REAL_COMPILER = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin" / "amdclang++"


def _compiledKernels(compileDir, source):
    """Names of the kernels in the device code object compiled from one source file."""
    out = compileDir / (Path(source).stem + ".co")
    flags = "-x hip --offload-device-only --no-gpu-bundle-output --offload-arch=gfx950 -O0 -c"
    args = [str(REAL_COMPILER), *flags.split(), "-I", str(compileDir), str(compileDir / source)]
    subprocess.run([*args, "-o", str(out)], check=True, capture_output=True)
    readelf = REAL_COMPILER.resolve().parent / "llvm-readelf"
    syms = subprocess.run(
        [str(readelf), "--syms", "-W", str(out)], check=True, capture_output=True, text=True
    ).stdout
    # Each .kd is listed in both .dynsym and .symtab.
    return sorted(
        {line.split()[-1][: -len(".kd")] for line in syms.splitlines() if line.endswith(".kd")}
    )


@pytest.mark.skipif(not REAL_COMPILER.exists(), reason="needs a ROCm amdclang++")
def test_real_compiler_header_only_translation_unit_has_no_kernels(tmp_path):
    """Compiled for real: Kernels.cpp holds every reduction kernel, and another file including
    only Kernels.h (as each chunk of a split helper compile would) holds none."""
    copyStaticFiles(tmp_path)
    writeHelpers(
        tmp_path, [_DummySource(), _reduction("s"), _reduction("h")], "Kernels.cpp", "Kernels.h"
    )
    (tmp_path / "Other.cpp").write_text('#include "Kernels.h"\n')

    declared = sorted(_signatures((tmp_path / "Kernels.h").read_text(), end=";"))
    assert _compiledKernels(tmp_path, "Kernels.cpp") == declared
    assert _compiledKernels(tmp_path, "Other.cpp") == []
